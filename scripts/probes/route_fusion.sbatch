#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/route_fusion_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/route_fusion_%j.out
# Arm D of campaign probes (CPU only): fuse the digit route with attention mass over probe_attn_route.py outputs.
#   expmon launch --exp probes --stage fusion scripts/probes/route_fusion.sbatch <attn_route dir> [...]
#   expmon launch --exp probes --stage fusion scripts/probes/route_fusion.sbatch --selftest <old attn_route dir> <out dir>
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
echo "FACTORS kind=route_fusion args=$(echo "$@" | tr ' ' ',')"
pixi run python -u scripts/probes/route_fusion.py "$@"
echo "=== EXIT CODE $? ==="
