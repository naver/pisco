#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/expQ/longdoc/audit_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/expQ/longdoc/audit_%j.out
# Campaign longdoc, arm A (feasibility gate): CPU audit of what the compressor / decoder receive per eval variant
# (pisco + base mode) and per training stage. No GPU, no model weights.
# Usage: audit.sbatch [LIMIT=0 (all rows; N = smoke, N rows per variant)] [ONLY= (variant prefix; E_ = arm E sets)]
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
LIMIT=${1:-0}; ONLY=${2:-}; SROWS=2000; [[ -n "$ONLY" ]] && SROWS=0   # ONLY=E_: arm E sets only, no training stages
echo "FACTORS kind=audit arm=A only=${ONLY:-all} limit=$LIMIT model=ft_mm_4b_184778 px=1003520 stride=8"
OUT=/beegfs/scratch/user/hdejean/pisco/expQ/longdoc/audit_${SLURM_JOB_ID}
env CONDA_OVERRIDE_CUDA=12.7 pixi run python -u scripts/longdoc/audit_inputs.py --out "$OUT" --limit "$LIMIT" --only "$ONLY" --stage_rows $SROWS \
  --workers ${SLURM_CPUS_PER_TASK:-16}
