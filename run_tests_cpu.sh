#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/run_tests_cpu_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/run_tests_cpu_%j.out
#
# CPU unit tests (no model, no data): collators + position layout. Never on the login node.
#   expmon launch --exp archq2 --stage test run_tests_cpu.sh
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
echo "FACTORS kind=unit_tests"
git rev-parse --short HEAD
RC=0
for t in test_positions.py test_mm_collators.py test_freeze.py; do
  echo "=== $t ==="; pixi run python -u $t || RC=1
done
echo "=== EXIT CODE $RC ==="; exit $RC
