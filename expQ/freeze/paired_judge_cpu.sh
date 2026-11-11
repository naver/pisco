#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:15:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/paired_judge_freeze_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/paired_judge_freeze_%j.out
# Campaign `freeze` gate test: scripts/paired_judge.py (unchanged; paired bootstrap 95% CI + exact McNemar on the judge
# score) for every eval set present on BOTH sides of each pair. Never on the login node.
#   args: one or more  A_DIR:B_DIR  (dirs under expQ/freeze, each with eval/), e.g. ft_frozen_s42:ref_s42
#   expmon launch --exp freeze --stage judge expQ/freeze/paired_judge_cpu.sh ft_frozen_s42:ref_s42 ft_frozen_s43:ref_s43
source ~/.bashrc; cd /beegfs/scratch/user/hdejean/pisco
echo "FACTORS kind=paired_judge pairs=$(echo "$@" | tr ' ' ',')"
git rev-parse --short HEAD; md5sum scripts/paired_judge.py
SETS="mm_docqa_pisco_px1003520_mm_docqa_1500_1m mm_docqa_pisco_px1003520_mm_naturalqa_1500 \
mm_docqa_pisco_px1003520_cml4096_text_mrqa_ood_1500 mm_docqa_pisco_k10_px1003520_cml4096_text_hotpotqa_1500"
RC=0
for P in "$@"; do
  A=${P%%:*}; B=${P##*:}
  for S in $SETS; do
    FA=expQ/freeze/$A/eval/$S.json; FB=expQ/freeze/$B/eval/$S.json
    [ -f "$FA" ] && [ -f "$FB" ] || { echo "=== SKIP $A vs $B $S (missing file)"; continue; }
    echo "=== PAIR $A vs $B  set=$S"
    python3 -u scripts/paired_judge.py "$FA" "$FB" --names "$A" "$B" || RC=1
  done
done
echo "=== EXIT CODE $RC ==="; exit $RC
