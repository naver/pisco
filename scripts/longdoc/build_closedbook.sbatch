#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/expQ/longdoc/build_closedbook_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/expQ/longdoc/build_closedbook_%j.out
# Campaign longdoc, arm E soundness check 1: write eval_data/longdoc_closedbook (every LongDocURL question once, no
# document) from the three arm E sets. CPU only. Exit 1 if a question's text or answers differ between the sets.
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
echo "FACTORS kind=build_closedbook arm=E sets=pages,mixed,text"
md5sum scripts/longdoc/build_closedbook.py
env CONDA_OVERRIDE_CUDA=12.7 pixi run python -u scripts/longdoc/build_closedbook.py
