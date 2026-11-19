#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/expQ/slotcheck/eval_x_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/expQ/slotcheck/eval_x_%j.out
# Campaign slotcheck, arm X (check 3): R on the 500 questions with image slots (img), transcription slots (text) and
# both (imgtext), ONE job = one node for every set of a seed (pairs need the same node). Each set is the shared
# eval_mm_4b.sh, unchanged (it resumes per file on requeue), PX=1003520, brief answers, 32 tokens, rubric judge:
#   img     TOPK=4
#   text    TOPK=4 CHUNK=1 CML=256     (one "Document j:" per page; its transcription in 256-token chunks, no cut)
#   imgtext TOPK=8 CHUNK=1 CML=256     (per page: image, then its transcription)
# DML = <set>/dml.txt (the build's max prompt + 256; DML only decides run vs error, never the output).
#   eval_x.sbatch <cell under expQ/slotcheck, model -> R dir> [SETS=img,text,imgtext] [LIMIT] [TAG]
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
export HF_HOME=/beegfs/scratch/user/hdejean/Reranking/cache
unset RAW DML RANK DENSE DROP KEEPK XD CSLICE MQ KEEPF KEEPM HS CHUNK CML GOLD STYLE MERGE CFG NSAMP MINPX TOPK RATE ADAPTER JPROMPT JMODEL
nvidia-smi -L
CELL=${1:?cell under expQ/slotcheck}; SETS=${2:-img,text,imgtext}; LIMIT=${3:-}; TAG=${4:-}
CKPT=/beegfs/scratch/user/hdejean/pisco/expQ/slotcheck/$CELL/model
[[ -e "$CKPT/config.json" ]] || { echo "missing $CKPT"; exit 2; }
echo "FACTORS kind=eval_x arm=X cell=$CELL sets=$SETS limit=${LIMIT:-all} tag=${TAG:-none} judge=${JUDGE:-1} model=$(readlink -f $CKPT) px=1003520 mnt=32"
echo "node $(hostname)"
md5sum eval_mm_4b.sh scripts/eval_mm_docqa.py scripts/judge_mm_docqa.py scripts/judge_prompts.py scripts/paired_judge.py \
  pisco/model.py pisco/mm_collator.py pisco/mm_utils.py scripts/slotcheck/build_text_sets.py | cut -c1-8,33-
RC=0
for S in ${SETS//,/ }; do
  DATA=/beegfs/scratch/user/hdejean/pisco/eval_data/slotcheck_docqa_${S}_500${TAG}
  [[ -f "$DATA/stats.json" && -f "$DATA/dml.txt" ]] || { echo "!! $DATA not built (stats.json / dml.txt)"; RC=2; continue; }
  case $S in
    img)     E=(TOPK=4) ;;
    text)    E=(TOPK=4 CHUNK=1 CML=256) ;;
    imgtext) E=(TOPK=8 CHUNK=1 CML=256) ;;
    *) echo "!! unknown set $S"; RC=2; continue ;;
  esac
  E+=(PX=1003520 DML=$(cat "$DATA/dml.txt"))
  echo "=== set $S: ${E[*]} -> $DATA ==="
  env "${E[@]}" bash eval_mm_4b.sh "$CKPT" pisco "$LIMIT" "$DATA" || { RC=$?; echo "!! set $S rc=$RC"; }
done
echo "=== EXIT CODE $RC ==="; exit $RC
