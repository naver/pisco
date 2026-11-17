#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:10:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/spread_audit_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/spread_audit_%j.out
# Campaign spread, arm A (feasibility gate): case audit + slice power over the RECORDED judged eval files.
# Stdlib python only (no pixi, no model, no GPU). Pairs: scripts/spread/audit_pairs.tsv.
# Usage: case_audit.sbatch [SPEC=scripts/spread/audit_pairs.tsv]
cd /beegfs/scratch/user/hdejean/pisco
SPEC=${1:-scripts/spread/audit_pairs.tsv}
echo "FACTORS kind=audit arm=A spec=$(basename $SPEC) boot=2000 slice_boot=1000"
md5sum scripts/spread/slice_pairs.py "$SPEC" scripts/paired_judge.py
OUT=expQ/spread/audit_${SLURM_JOB_ID}
python3 -u scripts/spread/slice_pairs.py --spec "$SPEC" --out "$OUT" --boot 2000 --slice_boot 1000
rc=$?
echo "=== EXIT CODE $rc ==="
exit $rc
