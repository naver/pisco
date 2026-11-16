#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:30:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/expQ/longdoc/recall_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/expQ/longdoc/recall_%j.out
# Campaign longdoc, arm T: transcription probe (scripts/longdoc/mem_recall.py, stride-aware port of
# scripts/probe_mem_recall.py). Is a page's text still in its <MEM>s, per stage-2 model?
#   expmon launch --exp longdoc --stage T --time … scripts/longdoc/mem_recall.sbatch <ARM> [N] [LIMIT]
#   ARM = ref (uncompressed backbone from the pixels) | v4kd (legacy validation: expQ/pt_mm_v4kd_4b, stride 1)
#       | s2_c256 | s2_l1024 (longchunk stage-2 models, stride 8, via expQ/longdoc/<ARM>/model symlinks)
#   N = pages per source (120 = the recorded legacy probe; 167 = 501 pages), LIMIT = plumbing (pages in total)
# Output expQ/longdoc/recall/<ARM>_n<N>[_lim<LIMIT>].json; pair with scripts/longdoc/recall_pair.py.
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
export HF_HOME=/beegfs/scratch/user/hdejean/Reranking/cache
nvidia-smi -L
R=/beegfs/scratch/user/hdejean/pisco; ARM=${1:?ref|v4kd|s2_c256|s2_l1024}; N=${2:-120}; LIMIT=${3:-0}
DATA=$R/eval_data/mm_docqa_1500_1m
OUT=$R/expQ/longdoc/recall/${ARM}_n${N}.json; [[ "$LIMIT" != "0" ]] && OUT=${OUT%.json}_lim${LIMIT}.json
echo "FACTORS kind=probe_recall arm=$ARM per_source=$N limit=$LIMIT data=$(basename $DATA) px=1003520"
case $ARM in
  ref)      A=(--arm ref) ;;
  v4kd)     A=(--arm ae --checkpoint_path $R/expQ/pt_mm_v4kd_4b/model) ;;
  s2_c256|s2_l1024) A=(--arm ae --checkpoint_path $R/expQ/longdoc/$ARM/model) ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
[[ -e "$OUT" ]] && { echo "refusing to overwrite $OUT"; exit 3; }
pixi run python -u scripts/longdoc/mem_recall.py "${A[@]}" --data_path $DATA --per_source $N --limit $LIMIT \
  --image_max_pixels 1003520 --out $OUT
RC=$?; echo "=== EXIT CODE $RC ==="; exit $RC
