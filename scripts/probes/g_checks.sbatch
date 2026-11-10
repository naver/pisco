#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/g_checks_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/g_checks_%j.out
# Arm G of campaign probes: the law's checks of a --dense_rate patch in ONE job (the cell budget is nearly spent):
# invariant tests, the 300-question plain regression of 184778, the 300-question RAW=route1 DENSE=8 control, then
# g_checks_compare.py. Exits 1 unless every check passes, so a wake reads one verdict.
# Also judges the pre-patch control's saved answers (job 188525, preempted while judging) for the record.
#   expmon launch --exp probes --stage G --parent 184781 --time 1:30:00 '--exclude=homer-[005-009]' scripts/probes/g_checks.sbatch
# The rate-4 cells are submitted only after this job PASSES (never chained on it).
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
echo "FACTORS kind=g_checks"
unset RAW DENSE PX KEEPF KEEPM MQ STYLE GOLD XD RANK CML TOPK LIMIT
RC=0
for t in test_positions.py test_mm_collators.py test_freeze.py; do
  echo "=== $t ==="; pixi run python -u $t || RC=1
done
if [[ $RC -ne 0 ]]; then echo "tests FAILED"; echo "=== EXIT CODE 1 ==="; exit 1; fi
D=eval_data/mm_docqa_1500_1m
echo "=== plain regression, 300 Q ==="
PX=1003520 bash eval_mm_4b.sh expQ/probes/reg_g_plain2/model pisco 300 $D || RC=1
echo "=== RAW=route1 DENSE=8 control, 300 Q ==="
RAW=route1 DENSE=8 PX=1003520 bash eval_mm_4b.sh expQ/probes/g_dense8_ctrl2/model pisco 300 $D || RC=1
echo "=== judge of the pre-patch control's answers ==="
pixi run python -u scripts/judge_mm_docqa.py \
  expQ/probes/g_dense8_ctrl/eval/attempt_h016_0644_mm_docqa_pisco_px1003520_rawroute1_dense8_mm_docqa_1500_1m_lim300.json \
  --prompt rubric --judge_model Qwen/Qwen3-8B || echo "!! judge of the pre-patch file failed"
python3 scripts/probes/g_checks_compare.py || RC=1
echo "=== EXIT CODE $RC ==="; exit $RC
