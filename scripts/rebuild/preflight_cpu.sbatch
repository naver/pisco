#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:40:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/expQ/rebuild/preflight_cpu_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/expQ/rebuild/preflight_cpu_%j.out
#
# Campaign rebuild, before any GPU-hour: the unit tests (TestRebuildOptions included) and the resolved config of every
# arm through the committed launchers (Hydra --cfg job, no training), each checked by config_check.py.
#   expmon launch --exp rebuild --stage preflight scripts/rebuild/preflight_cpu.sbatch
source ~/.bashrc
R=/beegfs/scratch/user/hdejean/pisco
cd $R
echo "FACTORS kind=preflight_cpu"; git rev-parse --short HEAD; git status --short pisco/ test_mm_collators.py scripts/rebuild | head
RC=0
echo "=== unit tests ==="; pixi run python -u test_mm_collators.py 2>&1 | grep -aE '^(Ran|OK|FAILED|FAIL:|ERROR:)' || true
pixi run python -u test_mm_collators.py > /dev/null 2>&1 || { echo "test_mm_collators FAILED"; RC=1; }
for a in K S W P; do
  echo "=== stage 2 $a: config only ==="; bash scripts/rebuild/pt_stage2.sbatch $a cfg | tail -12 || RC=1
  [[ "${PIPESTATUS[0]}" == 0 ]] || RC=1
done
for as in "K 42" "K 43" "L 42" "L 43" "P 42"; do
  echo "=== finetune $as: config only ==="; bash scripts/rebuild/ft.sbatch $as cfg | tail -8
  [[ "${PIPESTATUS[0]}" == 0 ]] || RC=1
done
echo "RESULT preflight_cpu_ok=$((1 - RC))"
echo "=== EXIT CODE $RC ==="; exit $RC
