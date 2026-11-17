#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH --requeue
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/ft_qafreeze_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/ft_qafreeze_%j.out
#
# Campaign qafreeze: the reference QA finetune (184778 / 184780, as copied by expQ/freeze/ft_freeze.sh: argv AND env)
# started from the hybrid init expQ/qafreeze/init_s<seed> (R's QA-trained compressor + connector, stage-2 decoder;
# built by scripts/qafreeze/build_init.sbatch) instead of the stage-2 model.
#   ARM=frozen -> +model_config_overrides.freeze_compressor=true (compressor + connector frozen, eval mode;
#                 only the decoder LoRA trains)                                                     = arm Q
# Usage: expmon launch --exp qafreeze --stage ft --parent <build job> scripts/qafreeze/ft_qafreeze.sbatch <arm> <seed> [SMOKE_STEPS]
#   seed 42 = no seed override (as 184778); any other seed adds +hf_training.seed=<seed> (as 184780 for 43).
#   SMOKE_STEPS (optional) = hf_training.max_steps for a plumbing run; its output goes to ..._smoke.
ARM=${1:?frozen}; SEED=${2:?seed}; SMOKE=${3:-}
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
export HF_HOME=/beegfs/scratch/user/hdejean/Reranking/cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -L

ROOT=/beegfs/scratch/user/hdejean/pisco
CKPT=$ROOT/expQ/qafreeze/init_s$SEED                # hybrid init (QAFREEZE_PROVENANCE.txt inside)
[ -f "$CKPT/QAFREEZE_PROVENANCE.txt" ] || { echo "no hybrid init at $CKPT (run build_init.sbatch $SEED)"; exit 2; }
DATA="$ROOT/data/mm_ft_docpages_20k_1m_teacher:20000,$ROOT/data/mm_ft_natural_8k_teacher:8000"
case "$ARM" in
  frozen)   ARMOV="+model_config_overrides.freeze_compressor=true" ;;   # arm Q
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
SEEDOV=""; [ "$SEED" != 42 ] && SEEDOV="+hf_training.seed=$SEED"
SMOKEOV=""; TAG=""
if [ -n "$SMOKE" ]; then SMOKEOV="+hf_training.max_steps=$SMOKE"; TAG="_smoke"; fi
OUT=$ROOT/expQ/qafreeze/ft_${ARM}_s${SEED}${TAG}
mkdir -p "$OUT"
echo "FACTORS kind=ft_mm arm=qa_$ARM seed=$SEED smoke=${SMOKE:-0} init=qafreeze_init_s$SEED samples=28000 bs=1 accum=64 lr=2e-5 max_pixels=1003520"
cat "$CKPT/QAFREEZE_PROVENANCE.txt"
git rev-parse --short HEAD; git status --short | head -20
md5sum pisco/model.py pisco/train.py pisco/mm_collator.py pisco/configs/finetune_mm.yaml | cut -c1-8,33-

pixi run python -u pisco/train.py --config-name=finetune_mm \
  hydra.run.dir="$OUT" \
  model_name_or_path="$CKPT" \
  "data.training_dataset='$DATA'" \
  data.samples=28000 \
  hf_training.per_device_train_batch_size=1 \
  hf_training.gradient_accumulation_steps=64 \
  hf_training.learning_rate=2e-5 +model_config_overrides.image_max_pixels=1003520 $ARMOV $SEEDOV $SMOKEOV
RC=$?; echo "=== EXIT CODE $RC ==="; exit $RC
