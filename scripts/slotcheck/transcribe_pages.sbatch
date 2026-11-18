#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=14G
#SBATCH --time=5:30:00
#SBATCH --requeue
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/expQ/slotcheck/transcribe_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/expQ/slotcheck/transcribe_%j.out
# Campaign slotcheck, arm X (check 3): uncompressed Qwen3.5-4B transcription of the pages of 500 document questions
# (scripts/slotcheck/transcribe_pages.py; prompt + generation imported from scripts/probe_mem_recall.py, never edited).
#   expmon launch --exp slotcheck --stage X_transcribe --time … scripts/slotcheck/transcribe_pages.sbatch SHARD [N_SHARDS=4] [LIMIT]
# Output expQ/slotcheck/transcribe/shard<SHARD>of<N>[_lim<LIMIT>].json (resumes on requeue). Exit 4 = health failed.
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
export HF_HOME=/beegfs/scratch/user/hdejean/Reranking/cache TOKENIZERS_PARALLELISM=false
SHARD=${1:?shard}; NS=${2:-4}; LIMIT=${3:-}
nvidia-smi -L
echo "FACTORS kind=transcribe arm=X_transcribe shard=$SHARD n_shards=$NS limit=${LIMIT:-all} set=mm_docqa_1500_1m questions=500 topk=4 px=1003520 max_new_tokens=1024"
echo "node $(hostname)"
md5sum scripts/slotcheck/transcribe_pages.py scripts/slotcheck/transcribe_pages.sbatch scripts/probe_mem_recall.py \
  scripts/eval_mm_docqa.py pisco/mm_utils.py | cut -c1-8,33-
LIM=(); [[ -n "$LIMIT" ]] && LIM=(--limit "$LIMIT")
pixi run python -u scripts/slotcheck/transcribe_pages.py --shard "$SHARD" --n_shards "$NS" "${LIM[@]}"
RC=$?
echo "=== EXIT CODE $RC ==="; exit $RC
