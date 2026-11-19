#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/expQ/slotcheck/textsets_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/expQ/slotcheck/textsets_%j.out
# Campaign slotcheck, arm X (check 3) build (CPU): eval_data/slotcheck_docqa_{img,text,imgtext}_500[TAG] from the 4
# transcription shards + the audit (mems, prompt max -> dml.txt, cut = 0). Exit 4 = a health check failed.
#   build_text_sets.sbatch [LIMIT] [TAG]      (plumbing: build_text_sets.sbatch 30 _smoke)
# env ROUND=r > 1: the repeat's sets slotcheck_docqa_*_500_r<r> from expQ/slotcheck/transcribe_r<r>/shard0of1.json.
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
export HF_HOME=/beegfs/scratch/user/hdejean/Reranking/cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
LIMIT=${1:-0}; TAG=${2:-}; RD=${ROUND:-1}; RA=()
OUT=/beegfs/scratch/user/hdejean/pisco/expQ/slotcheck/textsets_${SLURM_JOB_ID:-local}
[[ $RD -gt 1 ]] && { RA=(--round "$RD" --tx_dir /beegfs/scratch/user/hdejean/pisco/expQ/slotcheck/transcribe_r$RD --n_shards 1); TAG=${TAG:-_r$RD}; }
echo "FACTORS kind=build arm=X_build limit=${LIMIT} tag=${TAG:-none} round=$RD questions=500 sets=img,text,imgtext cml=256 chunk=1"
echo "node $(hostname)"
md5sum scripts/slotcheck/build_text_sets.py scripts/slotcheck/transcribe_pages.py scripts/longdoc/audit_inputs.py \
  pisco/mm_collator.py scripts/eval_mm_docqa.py | cut -c1-8,33-
if [[ $RD -gt 1 ]]; then   # round 1 re-audited with this cut rule (its sets are kept, never rebuilt); does not gate
  env CONDA_OVERRIDE_CUDA=12.7 pixi run python -u scripts/slotcheck/build_text_sets.py --out "${OUT}_r1"
  echo "=== round-1 re-audit EXIT CODE $? ==="
fi
env CONDA_OVERRIDE_CUDA=12.7 pixi run python -u scripts/slotcheck/build_text_sets.py --out "$OUT" --limit "$LIMIT" --tag "$TAG" "${RA[@]}"
RC=$?
echo "=== EXIT CODE $RC ==="; exit $RC
