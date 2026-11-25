#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/expQ/rebuild/ft_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/expQ/rebuild/ft_%j.out
#
# Campaign rebuild, finetunes (campaign_rebuild/CAMPAIGN.md): the recipe of R = ft_mm_4b_184778 (seed 42) /
# ft_mm_4b_184780 (seed 43), argv field by field (28k rows, bs 1 x accum 64, lr 2e-5, bf16, 1,003,520 px).
#   gate 2 of a stage-2 arm: expmon launch --exp rebuild --stage ft_<ARM> --parent <stage-2 jid> --time … scripts/rebuild/ft.sbatch <K|S|W|P> 42
#   arm L (cleaned labels, start = the recorded stage 2): … scripts/rebuild/ft.sbatch L 42   (and L 43)
#   smoke: … scripts/rebuild/ft.sbatch <ARM> <SEED> 20
#   config only (no training, CPU, -p gpu-be --gres=gpu:0): … scripts/rebuild/ft.sbatch <ARM> <SEED> cfg
# Output expQ/rebuild/ft_<ARM>_s<SEED>/ (smoke: …_smoke<N>/). Config checked against R's at start (config_check.py):
# a stage-2 arm may differ only in model_name_or_path, L only in data.training_dataset (+ max_steps in a smoke).
source ~/.bashrc
R=/beegfs/scratch/user/hdejean/pisco
cd $R
ARM=${1:?arm K|S|W|P|L}; SEED=${2:?seed 42|43}; STEPS=${3:-}
case $SEED in
  42) REC=$R/expQ/ft_mm_4b_184778/training_config.yaml; S=() ;;
  43) REC=$R/expQ/ft_mm_4b_184780/training_config.yaml; S=(+hf_training.seed=43) ;;
  *) echo "seed must be 42 or 43 (R's two seeds)"; exit 2 ;;
esac
DATA=$R/data/mm_ft_docpages_20k_1m_teacher:20000,$R/data/mm_ft_natural_8k_teacher:8000
case $ARM in
  K|S|W|P) START=$R/expQ/rebuild/$ARM/model; ALLOW=model_name_or_path ;;
  L) START=$R/expQ/archq3/pt_mm_5M_spread/model; ALLOW=data.training_dataset
     DATA=$R/data/rebuild_ft_clean_docpages,$R/data/rebuild_ft_clean_natural ;;
  *) echo "unknown arm $ARM (K|S|W|P|L)"; exit 2 ;;
esac
BASE=(hf_training.learning_rate=2e-5 +model_config_overrides.image_max_pixels=1003520 "${S[@]}")   # R's argv
if [[ $ARM == P ]]; then   # P's checkpoint saved torch_dtype=float32: its finetune must train in bf16 like R's
  BASE+=(++model_config_overrides.torch_dtype=bfloat16 ++model_config_overrides.frozen_base_dtype=null)
  ALLOW+=,model_config_overrides.torch_dtype,model_config_overrides.frozen_base_dtype
fi
if [[ "$STEPS" == cfg ]]; then      # resolve the config only (the start model and data need not exist yet)
  OUT=$R/expQ/rebuild/ft_${ARM}_s${SEED}_cfg; mkdir -p "$OUT"
  FT_DATA=$DATA FT_OUT=$OUT bash $R/ft_mm_4b.sh "$START" 28000 1 64 "${BASE[@]}" --cfg job > "$OUT/cfg_stdout.txt" 2>&1
  awk '/^output_dir:/{p=1} /^=== EXIT CODE/{p=0} p' "$OUT/cfg_stdout.txt" > "$OUT/training_config.yaml"
  grep -a -iE 'error|exception' "$OUT/cfg_stdout.txt" | grep -v nvidia | head -5
  pixi run python -I scripts/rebuild/config_check.py "$REC" "$OUT/training_config.yaml" "$ALLOW"; RC=$?
  echo "=== EXIT CODE $RC ==="; exit $RC
fi
[[ -f "$START/config.json" ]] || { echo "missing start model $START"; exit 2; }
for d in ${DATA//,/ }; do [[ -d "${d%%:*}" ]] || { echo "missing data ${d%%:*}"; exit 2; }; done
NAME=ft_${ARM}_s$SEED; X=()
if [[ -n "$STEPS" ]]; then NAME=${NAME}_smoke$STEPS; X=(++hf_training.max_steps=$STEPS); ALLOW+=,hf_training.max_steps; fi
OUT=$R/expQ/rebuild/$NAME
if [[ -f "$OUT/model/config.json" ]]; then
  if [[ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]]; then echo "requeue: $OUT/model complete, nothing to redo"; exit 0; fi
  echo "refusing to overwrite $OUT/model"; exit 3
fi
mkdir -p "$OUT"
echo "FACTORS kind=ft arm=$ARM seed=$SEED max_steps=${STEPS:-full} start=$(basename $(dirname $START)) rows=28000 bs=1 accum=64 lr=2e-5 px=1003520"
echo "node $(hostname) restart ${SLURM_RESTART_COUNT:-0}"; git rev-parse --short HEAD; git status --short pisco/ | head
md5sum ft_mm_4b.sh pisco/train.py pisco/model.py pisco/mm_collator.py pisco/mm_utils.py \
  scripts/rebuild/ft.sbatch scripts/rebuild/config_check.py | cut -c1-8,33-
MARK=$OUT/.started_${SLURM_JOB_ID}_${SLURM_RESTART_COUNT:-0}; touch "$MARK"
# R's argv (jobs 184778 / 184780, expmon runs) with this arm's start or data.
FT_DATA=$DATA FT_OUT=$OUT setsid bash $R/ft_mm_4b.sh "$START" 28000 1 64 "${BASE[@]}" "${X[@]}" &
PID=$!
for _ in $(seq 1 180); do
  [[ "$OUT/training_config.yaml" -nt "$MARK" ]] && break
  kill -0 $PID 2>/dev/null || break
  sleep 5
done
echo "=== CONFIG CHECK vs $REC ==="
if ! pixi run python -I scripts/rebuild/config_check.py "$REC" "$OUT/training_config.yaml" "$ALLOW"; then
  echo "config check FAILED or no config written: killing the run"; kill -TERM -- -$PID 2>/dev/null; wait $PID
  echo "=== EXIT CODE 3 ==="; exit 3
fi
wait $PID; RC=$?
echo "=== LOSS (last 20) ==="
tr '\r' '\n' < "$R/expQ/rebuild/ft_${SLURM_JOB_ID}.out" | grep -aoE "\{'loss': '?[0-9.]+'?[^}]*\}" | tail -20
echo "=== EXIT CODE $RC ==="; exit $RC
