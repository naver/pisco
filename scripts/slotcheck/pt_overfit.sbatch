#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:55:00
#SBATCH --requeue
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/expQ/slotcheck/pt_overfit_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/expQ/slotcheck/pt_overfit_%j.out
# Campaign slotcheck, arm O (check 4, the campaign's ONLY training): overfit the stage-2 start model on the 100 dense
# pages (image autoencoding only), then read word recall on the same pages (scripts/slotcheck/recall.sbatch).
# Recipe = expQ/archq3/pt_mm_5M_spread/training_config.yaml (job 184777, 43.5 s/step, MaxRSS 42.8 GB) through the
# shared pt_mm_4b.sh (run, never edited). Law invariant: the run's training_config.yaml differs from the start's only
# in the data path, the task mix, max_steps, the precision field and the output paths -> printed by the diff below.
#   expmon launch --exp slotcheck --stage O_smoke --time … scripts/slotcheck/pt_overfit.sbatch o_smoke 20
#   expmon launch --exp slotcheck --stage O       --time … scripts/slotcheck/pt_overfit.sbatch o_bf16 <STEPS> [bf16|fp32]
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
R=/beegfs/scratch/user/hdejean/pisco
NAME=${1:?name}; STEPS=${2:?max_steps}; PREC=${3:-bf16}; DATA=${4:-$R/data/slotcheck_dense100_train}
START=$R/expQ/archq3/pt_mm_5M_spread/model
OUT=$R/expQ/slotcheck/$NAME
case $PREC in
  bf16) P=() ;;
  fp32) P=(++model_config_overrides.torch_dtype=float32) ;;   # fp32 master weights, bf16 autocast (one variable)
  *) echo "unknown precision $PREC"; exit 2 ;;
esac
[[ -e "$OUT/model" ]] && { echo "refusing to overwrite $OUT/model"; exit 3; }
[[ -d "$DATA" ]] || { echo "missing $DATA"; exit 2; }
echo "FACTORS kind=train arm=O name=$NAME max_steps=$STEPS precision=$PREC data=$(basename $DATA) start=pt_mm_5M_spread task=image_ae_only bs=8 accum=8 lr=3e-5 px=1003520"
echo "node $(hostname)"
md5sum pt_mm_4b.sh pisco/train.py pisco/model.py pisco/mm_collator.py pisco/mm_utils.py scripts/slotcheck/pt_overfit.sbatch | cut -c1-8,33-
# The start's recipe, field by field (training_config.yaml of the start); only the task mix and max_steps differ.
PT_DATA=$DATA PT_OUT=$OUT bash $R/pt_mm_4b.sh 160000 8 8 merger "$START" \
  ++hf_training.learning_rate=3.0e-05 ++hf_training.max_steps=$STEPS \
  ++collator_kwargs.p_image_ae=1.0 ++collator_kwargs.p_text=0.0 ++collator_kwargs.p_image_cont=0.0 \
  ++collator_kwargs.p_desc=0.0 ++collator_kwargs.p_find=0.0 \
  ++collator_kwargs.teacher_inputs=true ++collator_kwargs.teacher_max_length=6144 \
  ++model_config_overrides.image_max_pixels=1003520 ++model_config_overrides.kd_weight=0.5 "${P[@]}"
RC=$?
echo "=== CONFIG DIFF vs the start (law invariant: data path, task mix, max_steps, precision, output paths only) ==="
diff "$R/expQ/archq3/pt_mm_5M_spread/training_config.yaml" "$OUT/training_config.yaml"
echo "=== LOSS (logged every 10 steps) ==="
tr '\r' '\n' < "$R/expQ/slotcheck/pt_overfit_${SLURM_JOB_ID}.out" | grep -aoE "\{'loss': '[^']+'[^}]*\}" | tail -200
echo "=== EXIT CODE $RC ==="; exit $RC
