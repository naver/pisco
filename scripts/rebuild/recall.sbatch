#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=2:15:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/expQ/rebuild/recall_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/expQ/rebuild/recall_%j.out
# Campaign rebuild, gate 1 and its feasibility gate: page word recall through scripts/longdoc/mem_recall.py
# (run as a tool, never edited). Port of scripts/slotcheck/recall.sbatch (b61a7b4c) with outputs under expQ/rebuild/recall/.
#   expmon launch --exp rebuild --stage <stage> --time … scripts/rebuild/recall.sbatch NAME:ARM:SET[:PER_SOURCE] [...]
#   ARM = ref (uncompressed Qwen3.5-4B transcribes the pixels) | any PISCO model dir (absolute path)
#   SET = an eval_data dir name (slotcheck_dense100, slotcheck_dense100_heldout, mm_docqa_1500_1m, rebuild_*)
#   PER_SOURCE = pages per source (default 100; the eval-page read uses 120)
# Output expQ/rebuild/recall/NAME.json (a pass whose file exists is skipped, so a requeue resumes).
# The first ae pass is preceded by a 5-page plumbing pass (NAME_lim5.json) that must print stride 8 and non-empty text.
# Pair with: python3 -I scripts/longdoc/recall_pair.py --ref <ref json> --a <a json> --b <b json>
source ~/.bashrc
R=/beegfs/scratch/user/hdejean/pisco
cd $R
export HF_HOME=/beegfs/scratch/user/hdejean/Reranking/cache
nvidia-smi -L
echo "FACTORS kind=recall passes=$* px=1003520"
echo "node $(hostname) git $(git rev-parse --short HEAD)"
md5sum eval_mm_4b.sh scripts/eval_mm_docqa.py scripts/judge_mm_docqa.py scripts/paired_judge.py pisco/model.py \
  pisco/mm_collator.py scripts/longdoc/mem_recall.py scripts/probe_mem_recall.py scripts/rebuild/recall.sbatch | cut -c1-8,33-
mkdir -p $R/expQ/rebuild/recall
SMOKED=0
for SPEC in "$@"; do
  IFS=: read -r NAME ARM SET PS <<< "$SPEC"
  [[ -n "$NAME" && -n "$ARM" && -n "$SET" ]] || { echo "bad spec $SPEC (NAME:ARM:SET[:PER_SOURCE])"; exit 2; }
  PS=${PS:-100}
  DATA=$R/eval_data/$SET; [[ -d "$DATA" ]] || { echo "missing $DATA"; exit 2; }
  case $ARM in
    ref) A=(--arm ref) ;;
    *)   [[ -d "$ARM" ]] || { echo "missing model dir $ARM"; exit 2; }; A=(--arm ae --checkpoint_path "$ARM") ;;
  esac
  OUT=$R/expQ/rebuild/recall/$NAME.json
  if [[ "$ARM" != ref && $SMOKED == 0 ]]; then
    echo "=== plumbing: $NAME 5 pages ==="
    S=${OUT%.json}_lim5.json
    [[ -e "$S" ]] || pixi run python -u scripts/longdoc/mem_recall.py "${A[@]}" --data_path $DATA --per_source $PS \
      --limit 5 --image_max_pixels 1003520 --out $S || { echo "plumbing pass failed"; exit 4; }
    python3 -I -c "import json,sys; r=json.load(open(sys.argv[1]))['records']; w=[len(x['text'].split()) for x in r]; \
s={x.get('stride') for x in r}; print('[plumbing] pages', len(r), 'words', w, 'stride', s); \
sys.exit(0 if len(r)==5 and min(w)>0 and s=={8} else 4)" "$S" || { echo "PLUMBING FAIL"; exit 4; }
    SMOKED=1
  fi
  [[ -e "$OUT" ]] && { echo "exists, skipped: $OUT"; continue; }
  echo "=== pass $NAME: arm=$ARM set=$SET per_source=$PS ==="
  pixi run python -u scripts/longdoc/mem_recall.py "${A[@]}" --data_path $DATA --per_source $PS \
    --image_max_pixels 1003520 --out $OUT || { echo "pass $NAME failed"; exit 1; }
done
echo "=== EXIT CODE 0 ==="
