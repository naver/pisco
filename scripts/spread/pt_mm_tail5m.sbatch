#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:55:00
#SBATCH --requeue
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/expQ/spread/e_pt_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/expQ/spread/e_pt_%j.out
# Campaign spread, arm E, stage 2 (TRAINING: launch only on the owner's REQUEST, hard rule 8). The missing control:
# the TAIL layout at full budget with ONE stage-2 pass. Recipe = job 184777 (-> expQ/archq3/pt_mm_5M_spread): its
# recorded argv (.expmon/runs.jsonl) and its data (training_config.yaml line 25, set by PT_DATA at launch), through
# the shared pt_mm_4b.sh (run, never edited). ONE change: the init is the tail 5M text stage 1
# expQ/qwen35_4b_pt_5M_143228/model, whose training_config.yaml equals the spread stage 1's
# (expQ/archq3/pt_text_5M_spread) except `compressor_mem_layout: spread` + `mem_position_stride: 8` (lines 46-47).
# `+key` of the recorded argv is written `++key` (same value; robust if the base config gained the key since 24b7a2c):
# the config diff below proves the resolved recipe is the same.
# Measured (184777, homer-010): 30:12:20 for 160k rows = 2,500 steps of 64 rows = 43.5 s/step; MaxRSS 42.8 GB.
#   smoke: expmon launch --exp spread --stage E_smoke --time 0:55:00 scripts/spread/pt_mm_tail5m.sbatch e_pt_smoke 20
#   full:  expmon launch --exp spread --stage E --time 37:45:00 scripts/spread/pt_mm_tail5m.sbatch e_pt_tail5m
# Law invariant (CAMPAIGN.md "Patchable"): the run's training_config.yaml differs from the recipe's only in
# model_name_or_path, max_steps and the output paths. Checked at the end; a run that breaks it exits 5.
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
R=/beegfs/scratch/user/hdejean/pisco
NAME=${1:?name}; STEPS=${2:-}
INIT=$R/expQ/qwen35_4b_pt_5M_143228/model
REF=$R/expQ/archq3/pt_mm_5M_spread/training_config.yaml
OUT=$R/expQ/spread/$NAME
[[ -e "$OUT/model" ]] && { echo "refusing to overwrite $OUT/model"; exit 3; }
[[ -d "$INIT" ]] || { echo "missing $INIT"; exit 2; }
MS=(); [[ -n "$STEPS" ]] && MS=(++hf_training.max_steps=$STEPS)
echo "FACTORS kind=train arm=E stage=pt name=$NAME init=qwen35_4b_pt_5M_143228 layout=tail samples=160000 bs=8 accum=8 lr=3e-5 px=1003520 kd=0.5 max_steps=${STEPS:-full}"
echo "node $(hostname)"
md5sum pt_mm_4b.sh pisco/train.py pisco/model.py pisco/mm_collator.py pisco/mm_utils.py scripts/spread/pt_mm_tail5m.sbatch | cut -c1-8,33-
PT_DATA=$R/data/mm_pretrain_pdfa_1m_desc:80000,$R/data/mm_pretrain_wiki_1m_desc:30000,$R/data/mm_pretrain_natural_desc:50000 \
PT_OUT=$OUT bash $R/pt_mm_4b.sh 160000 8 8 merger "$INIT" \
  ++model_config_overrides.image_max_pixels=1003520 ++hf_training.learning_rate=3e-5 \
  ++collator_kwargs.p_text=0.5 ++collator_kwargs.p_image_ae=0.2 ++collator_kwargs.p_image_cont=0.05 \
  ++collator_kwargs.p_desc=0.15 ++collator_kwargs.p_find=0.1 ++model_config_overrides.kd_weight=0.5 \
  ++collator_kwargs.teacher_inputs=true ++collator_kwargs.teacher_max_length=6144 "${MS[@]}"
RC=$?
echo "=== CONFIG DIFF vs the recipe $REF ==="
diff "$REF" "$OUT/training_config.yaml"
KEEP='^[[:space:]]*(model_name_or_path|max_steps|output_dir|logging_dir|run_name):'
BAD=$(diff <(grep -vE "$KEEP" "$REF") <(grep -vE "$KEEP" "$OUT/training_config.yaml" 2>/dev/null) | grep -c '^[<>]')
echo "INVARIANT other_diff_lines=$BAD (must be 0)"
[[ $RC -eq 0 && "$BAD" -ne 0 ]] && RC=5
echo "=== LOSS (last logged lines) ==="
tr '\r' '\n' < "$R/expQ/spread/e_pt_${SLURM_JOB_ID}.out" | grep -aoE "\{'loss': '?[^,}]+'?[^}]*\}" | tail -60
echo "=== EXIT CODE $RC ==="; exit $RC
