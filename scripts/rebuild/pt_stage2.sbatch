#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/expQ/rebuild/pt_stage2_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/expQ/rebuild/pt_stage2_%j.out
#
# Campaign rebuild, stage-2 arms (campaign_rebuild/CAMPAIGN.md, "Arms"): the recorded recipe of
# expQ/archq3/pt_mm_5M_spread (its argv, job 184777, field by field) at seed 43, plus ONE change per arm:
#   K control (none) · S transcription share on document rows · W window AE targets · P fp32 masters
#   smoke: expmon launch --exp rebuild --stage <ARM>_smoke --time 1:30:00 scripts/rebuild/pt_stage2.sbatch <ARM> 20
#   full:  expmon launch --exp rebuild --stage <ARM> --time <smoke s/step x 2499 x 1.25> scripts/rebuild/pt_stage2.sbatch <ARM>
#   config only (no training, CPU): expmon launch --exp rebuild --stage cfg -p gpu-be --gres=gpu:0 --mem 16G --time 0:20:00 \
#          scripts/rebuild/pt_stage2.sbatch <ARM> cfg     (Hydra --cfg job through pt_mm_4b.sh, then config_check.py)
# Output expQ/rebuild/<ARM>/ (smoke: expQ/rebuild/<ARM>_smoke<N>/). As soon as train.py writes the resolved config,
# scripts/rebuild/config_check.py compares it with the recorded one; a difference the law does not allow for this
# arm kills the run (exit 3). Resumable: a requeue restarts in the same dir and the trainer reloads checkpoint-N.
source ~/.bashrc
R=/beegfs/scratch/user/hdejean/pisco
cd $R
ARM=${1:?arm K|S|W|P}; STEPS=${2:-}
REC=$R/expQ/archq3/pt_mm_5M_spread/training_config.yaml
START=$R/expQ/archq3/pt_text_5M_spread/model
DATA=$R/data/mm_pretrain_pdfa_1m_desc:80000,$R/data/mm_pretrain_wiki_1m_desc:30000,$R/data/mm_pretrain_natural_desc:50000
ALLOW=hf_training.seed
case $ARM in
  K) X=() ;;
  S) X=('+collator_kwargs.mix_sources="pdfa,wiki-ss"'
        '+collator_kwargs.mix_override={p_image_ae:0.45,p_text:0.25,p_image_cont:0.05,p_desc:0.15,p_find:0.1}')
     ALLOW+=,collator_kwargs.mix_sources,collator_kwargs.mix_override ;;
  W) X=(+collator_kwargs.ae_window_p=0.5 +collator_kwargs.ae_window_cue=16
        '+collator_kwargs.ae_window_sources="pdfa,wiki-ss"')
     ALLOW+=,collator_kwargs.ae_window_p,collator_kwargs.ae_window_cue,collator_kwargs.ae_window_sources ;;
  P) X=(++model_config_overrides.torch_dtype=float32 ++model_config_overrides.frozen_base_dtype=bfloat16)
     ALLOW+=,model_config_overrides.torch_dtype,model_config_overrides.frozen_base_dtype ;;
  *) echo "unknown arm $ARM (K|S|W|P)"; exit 2 ;;
esac
# The recorded argv of job 184777 (expmon runs; expQ/archq3/pt_mm_5M_spread/.hydra/overrides.yaml), then the seed.
BASE=(+model_config_overrides.image_max_pixels=1003520 hf_training.learning_rate=3e-5
  collator_kwargs.p_text=0.5 collator_kwargs.p_image_ae=0.2 collator_kwargs.p_image_cont=0.05
  +collator_kwargs.p_desc=0.15 +collator_kwargs.p_find=0.1 +model_config_overrides.kd_weight=0.5
  +collator_kwargs.teacher_inputs=true +collator_kwargs.teacher_max_length=6144 ++hf_training.seed=43)
if [[ "$STEPS" == cfg ]]; then      # resolve the config only: proves Hydra takes this arm's overrides as written
  OUT=$R/expQ/rebuild/${ARM}_cfg; mkdir -p "$OUT"
  PT_DATA=$DATA PT_OUT=$OUT bash $R/pt_mm_4b.sh 160000 8 8 merger "$START" "${BASE[@]}" "${X[@]}" --cfg job \
    > "$OUT/cfg_stdout.txt" 2>&1
  awk '/^output_dir:/{p=1} /^=== EXIT CODE/{p=0} p' "$OUT/cfg_stdout.txt" > "$OUT/training_config.yaml"
  grep -a -iE 'error|exception' "$OUT/cfg_stdout.txt" | grep -v nvidia | head -5
  pixi run python -I scripts/rebuild/config_check.py "$REC" "$OUT/training_config.yaml" "$ALLOW"; RC=$?
  echo "=== EXIT CODE $RC ==="; exit $RC
fi
NAME=$ARM
if [[ -n "$STEPS" ]]; then
  NAME=${ARM}_smoke$STEPS; X+=(++hf_training.max_steps=$STEPS); ALLOW+=,hf_training.max_steps
fi
OUT=$R/expQ/rebuild/$NAME
if [[ -f "$OUT/model/config.json" ]]; then
  if [[ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]]; then echo "requeue: $OUT/model complete, nothing to redo"; exit 0; fi
  echo "refusing to overwrite $OUT/model"; exit 3
fi
mkdir -p "$OUT"
echo "FACTORS kind=pt_stage2 arm=$ARM seed=43 max_steps=${STEPS:-full} start=pt_text_5M_spread rows=160000 bs=8 accum=8 lr=3e-5 px=1003520"
echo "node $(hostname) restart ${SLURM_RESTART_COUNT:-0}"; git rev-parse --short HEAD; git status --short pisco/ | head
md5sum pt_mm_4b.sh pisco/train.py pisco/model.py pisco/mm_collator.py pisco/mm_utils.py \
  scripts/rebuild/pt_stage2.sbatch scripts/rebuild/config_check.py | cut -c1-8,33-
MARK=$OUT/.started_${SLURM_JOB_ID}_${SLURM_RESTART_COUNT:-0}; touch "$MARK"
PT_DATA=$DATA PT_OUT=$OUT setsid bash $R/pt_mm_4b.sh 160000 8 8 merger "$START" "${BASE[@]}" "${X[@]}" &
PID=$!
for _ in $(seq 1 180); do      # train.py writes the config before loading data; 15 min at most
  [[ "$OUT/training_config.yaml" -nt "$MARK" ]] && break
  kill -0 $PID 2>/dev/null || break
  sleep 5
done
echo "=== CONFIG CHECK vs $REC (law: only seed and this arm's change; smoke: max_steps) ==="
if ! pixi run python -I scripts/rebuild/config_check.py "$REC" "$OUT/training_config.yaml" "$ALLOW"; then
  echo "config check FAILED or no config written: killing the run"; kill -TERM -- -$PID 2>/dev/null; wait $PID
  echo "=== EXIT CODE 3 ==="; exit 3
fi
wait $PID; RC=$?
echo "=== LOSS (logged every 10 steps, last 30) ==="
tr '\r' '\n' < "$R/expQ/rebuild/pt_stage2_${SLURM_JOB_ID}.out" | grep -aoE "\{'loss': '?[0-9.]+'?[^}]*\}" | tail -30
echo "=== TASK COUNTS (last per worker) ==="
grep -a '\[mm-pretrain\] task counts' "$R/expQ/rebuild/pt_stage2_${SLURM_JOB_ID}.out" | tail -4
echo "=== EXIT CODE $RC ==="; exit $RC
