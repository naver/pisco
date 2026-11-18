#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=1:00:00
#SBATCH --requeue
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/expQ/spread/e_ft_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/expQ/spread/e_ft_%j.out
# Campaign spread, arm E, finetune (TRAINING: launch only on the owner's REQUEST, hard rule 8). Recipe = job 184778
# (seed 42) / 184780 (seed 43): recorded argv (.expmon/runs.jsonl) + data (training_config.yaml line 25, set by FT_DATA
# at launch), through the shared ft_mm_4b.sh (run, never edited). ONE change: the init is arm E's stage-2 model
# (default expQ/spread/e_pt_tail5m/model) instead of expQ/archq3/pt_mm_5M_spread/model.
# Measured: 184778 11:02:12 (homer-010), 184780 13:49:25 (homer-011) for 28k rows = 437 steps of 64 rows
# (90.8 / 113.8 s/step); MaxRSS 23.2 GB.
#   smoke: expmon launch --exp spread --stage E_smoke --time 1:00:00 scripts/spread/ft_tail5m.sbatch e_ft_smoke 42 <INIT> 5
#   full:  expmon launch --exp spread --stage E --time 17:17:00 scripts/spread/ft_tail5m.sbatch e_ft_tail5m_s42 42
# Law invariant (CAMPAIGN.md "Patchable"): training_config.yaml differs from 184778's (s42) / 184780's (s43) only in
# model_name_or_path, max_steps and the output paths. Checked at the end; a run that breaks it exits 5.
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
R=/beegfs/scratch/user/hdejean/pisco
NAME=${1:?name}; SEED=${2:?seed 42 or 43}; INIT=${3:-$R/expQ/spread/e_pt_tail5m/model}; STEPS=${4:-}
case $SEED in
  42) REF=$R/expQ/ft_mm_4b_184778/training_config.yaml; SA=() ;;
  43) REF=$R/expQ/ft_mm_4b_184780/training_config.yaml; SA=(++hf_training.seed=43) ;;
  *) echo "seed must be 42 or 43"; exit 2 ;;
esac
OUT=$R/expQ/spread/$NAME
[[ -e "$OUT/model" ]] && { echo "refusing to overwrite $OUT/model"; exit 3; }
[[ -d "$INIT" ]] || { echo "missing $INIT"; exit 2; }
MS=(); [[ -n "$STEPS" ]] && MS=(++hf_training.max_steps=$STEPS)
echo "FACTORS kind=train arm=E stage=ft name=$NAME seed=$SEED init=$(basename $(dirname $INIT)) layout=tail samples=28000 bs=1 accum=64 lr=2e-5 px=1003520 max_steps=${STEPS:-full}"
echo "node $(hostname)"
md5sum ft_mm_4b.sh pisco/train.py pisco/model.py pisco/mm_collator.py pisco/mm_utils.py scripts/spread/ft_tail5m.sbatch | cut -c1-8,33-
FT_DATA=$R/data/mm_ft_docpages_20k_1m_teacher:20000,$R/data/mm_ft_natural_8k_teacher:8000 \
FT_OUT=$OUT bash $R/ft_mm_4b.sh "$INIT" 28000 1 64 \
  ++hf_training.learning_rate=2e-5 ++model_config_overrides.image_max_pixels=1003520 "${SA[@]}" "${MS[@]}"
RC=$?
echo "=== CONFIG DIFF vs the recipe $REF ==="
diff "$REF" "$OUT/training_config.yaml"
KEEP='^[[:space:]]*(model_name_or_path|max_steps|output_dir|logging_dir|run_name):'
BAD=$(diff <(grep -vE "$KEEP" "$REF") <(grep -vE "$KEEP" "$OUT/training_config.yaml" 2>/dev/null) | grep -c '^[<>]')
echo "INVARIANT other_diff_lines=$BAD (must be 0)"
[[ $RC -eq 0 && "$BAD" -ne 0 ]] && RC=5
echo "=== EXIT CODE $RC ==="; exit $RC
