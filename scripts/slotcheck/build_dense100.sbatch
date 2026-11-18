#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/expQ/slotcheck/dense100_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/expQ/slotcheck/dense100_%j.out
# Campaign slotcheck, arm O (check 4) build: 100 dense pages to overfit + 100 held-out (CPU).
# Usage: build_dense100.sbatch [SCAN=60000] [TAG=]   (smoke: build_dense100.sbatch 3000 _smoke)
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
export HF_HOME=/beegfs/scratch/user/hdejean/Reranking/cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SCAN=${1:-60000}; TAG=${2:-}
echo "FACTORS kind=build arm=O_build scan=$SCAN tag=${TAG:-none} n=100 max_tokens=760"
echo "node $(hostname)"
md5sum scripts/slotcheck/build_dense100.py | cut -c1-8,33-
OUT=/beegfs/scratch/user/hdejean/pisco/expQ/slotcheck/dense100_${SLURM_JOB_ID}
env CONDA_OVERRIDE_CUDA=12.7 pixi run python -u scripts/slotcheck/build_dense100.py --out "$OUT" --scan "$SCAN" --tag "$TAG"
