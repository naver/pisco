#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/probe_attn_route_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/probe_attn_route_%j.out
# Attention-mass routing probe (no training): expmon launch --exp probes --stage probe probe_attn_route.sh <DATA> [CKPT] [TOPK] [N_ROWS]
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
export HF_HOME=/beegfs/scratch/user/hdejean/Reranking/cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA=${1:?eval set dir} CKPT=${2:-expQ/ft_mm_4b_183814/model} TOPK=${3:-10} N_ROWS=${4:-0} PX=${PX:-1003520}
export OUT=/beegfs/scratch/user/hdejean/pisco/expQ/probes/attn_route_$SLURM_JOB_ID
echo "FACTORS kind=probe_attn_route ckpt=$(basename $(dirname $CKPT)) data=$(basename $DATA) topk=$TOPK n_rows=${N_ROWS}"; nvidia-smi -L
env CONDA_OVERRIDE_CUDA=12.7 pixi run python -u scripts/probes/probe_attn_route.py
echo "=== EXIT CODE $? ==="
