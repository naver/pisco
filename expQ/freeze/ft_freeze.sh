#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH --requeue
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/ft_freeze_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/ft_freeze_%j.out
#
# Campaign `freeze`: the reference QA finetune (jobs 184778 / 184780: ft_mm_4b.sh, argv AND env) with ONE switch.
#   ARM=frozen    -> +model_config_overrides.freeze_compressor=true  (compressor + connector frozen, eval mode;
#                    only the decoder LoRA trains)
#   ARM=unfrozen  -> the reference command, unchanged (used for a 20-step plumbing check or a third seed)
#   ARM=compfrozen / connfrozen -> +model_config_overrides.freeze_part=compressor / connector (split test A1 / A2:
#                    one part frozen and in eval mode, the other part and the decoder LoRA train)
# Usage: expmon launch --exp freeze --stage ft --parent 184777 expQ/freeze/ft_freeze.sh <arm> <seed> [SMOKE_STEPS]
#   seed 42 = no seed override (as 184778); any other seed adds +hf_training.seed=<seed> (as 184780 for 43).
#   SMOKE_STEPS (optional) = hf_training.max_steps for a plumbing run; its output goes to ..._smoke.
ARM=${1:?frozen|unfrozen}; SEED=${2:?seed}; SMOKE=${3:-}
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
export HF_HOME=/beegfs/scratch/user/hdejean/Reranking/cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -L

ROOT=/beegfs/scratch/user/hdejean/pisco
CKPT=$ROOT/expQ/archq3/pt_mm_5M_spread/model      # stage-2 model of the reference chain (job 184777)
DATA="$ROOT/data/mm_ft_docpages_20k_1m_teacher:20000,$ROOT/data/mm_ft_natural_8k_teacher:8000"
case "$ARM" in
  frozen)   ARMOV="+model_config_overrides.freeze_compressor=true" ;;
  unfrozen) ARMOV="" ;;
  compfrozen) ARMOV="+model_config_overrides.freeze_part=compressor" ;;   # split test A1 (gate in CAMPAIGN.md)
  connfrozen) ARMOV="+model_config_overrides.freeze_part=connector" ;;    # split test A2 (gate in CAMPAIGN.md)
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
SEEDOV=""; [ "$SEED" != 42 ] && SEEDOV="+hf_training.seed=$SEED"
SMOKEOV=""; TAG=""
if [ -n "$SMOKE" ]; then SMOKEOV="+hf_training.max_steps=$SMOKE"; TAG="_smoke"; fi
OUT=$ROOT/expQ/freeze/ft_${ARM}_s${SEED}${TAG}
mkdir -p "$OUT"
echo "FACTORS kind=ft_mm arm=$ARM seed=$SEED smoke=${SMOKE:-0} init=pt_mm_5M_spread samples=28000 bs=1 accum=64 lr=2e-5 max_pixels=1003520"
git rev-parse --short HEAD; git status --short | head -20

pixi run python -u pisco/train.py --config-name=finetune_mm \
  hydra.run.dir="$OUT" \
  model_name_or_path="$CKPT" \
  "data.training_dataset='$DATA'" \
  data.samples=28000 \
  hf_training.per_device_train_batch_size=1 \
  hf_training.gradient_accumulation_steps=64 \
  hf_training.learning_rate=2e-5 +model_config_overrides.image_max_pixels=1003520 $ARMOV $SEEDOV $SMOKEOV
RC=$?; echo "=== EXIT CODE $RC ==="; exit $RC
