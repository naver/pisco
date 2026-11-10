#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/build_groups_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/build_groups_%j.out
# Arm B of campaign probes (CPU only): grouped HotpotQA sets (gold joined / distractors joined / g10 shuffled),
# after checking the builder reproduces the existing g10 set row for row.
#   expmon launch --exp probes --stage build scripts/probes/build_groups.sbatch [--kinds ...]
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
echo "FACTORS kind=build_groups args=$(echo "$@" | tr ' ' ',')"
pixi run python -u scripts/probes/build_groups.py "$@"
RC=$?
echo "=== EXIT CODE $RC ==="
exit $RC
