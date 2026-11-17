#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/expQ/longdoc/build_long_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/expQ/longdoc/build_long_%j.out
# Campaign longdoc, arm E: build the long-document eval sets from LongDocURL (CPU; downloads the source inside the job).
# Usage: build_long.sbatch [KINDS=all|pages,text,mixed,needle] [MAX_Q=1500] [SUFFIX=]   (SUFFIX=_smoke for a test build)
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
export HF_HOME=/beegfs/scratch/user/hdejean/Reranking/cache TOKENIZERS_PARALLELISM=false
KINDS=${1:-all}; MAXQ=${2:-1500}; SFX=${3:-}
echo "FACTORS kind=build_long arm=E sets=$KINDS max_q=$MAXQ suffix=${SFX:-none} source=LongDocURL px=1003520 piece=1000"
md5sum scripts/longdoc/longdoc_build.py
env CONDA_OVERRIDE_CUDA=12.7 pixi run python -u scripts/longdoc/longdoc_build.py --kinds "$KINDS" --max_q "$MAXQ" \
  --suffix "$SFX" --workers ${SLURM_CPUS_PER_TASK:-3}
