#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/expQ/longdoc/split_buckets_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/expQ/longdoc/split_buckets_%j.out
# Campaign longdoc, arm E: one eval set per length bucket of the pages and mixed sets
# (eval_data/longdoc_{pages,mixed}_1500_{p,m}{10,20,40}). CPU only. Exit 1 if a check fails.
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
echo "FACTORS kind=split_buckets arm=E sets=pages,mixed"
md5sum scripts/longdoc/split_buckets.py
env CONDA_OVERRIDE_CUDA=12.7 pixi run python -u scripts/longdoc/split_buckets.py
