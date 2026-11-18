#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=4:30:00
#SBATCH --requeue
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/expQ/slotcheck/recall_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/expQ/slotcheck/recall_%j.out
# Campaign slotcheck, arm O (check 4): word-recall transcription probe on the dense100 sets, through
# scripts/longdoc/mem_recall.py (stride-aware port, validated 360 / 360; run as a tool, never edited).
#   expmon launch --exp slotcheck --stage O_recall --time … scripts/slotcheck/recall.sbatch NAME:ARM:SET [...]
#   ARM = ref (uncompressed Qwen3.5-4B transcribes the pixels) | start (expQ/archq3/pt_mm_5M_spread/model)
#       | a model dir under expQ/slotcheck/ (an overfit run's model)
#   SET = an eval_data dir (slotcheck_dense100, slotcheck_dense100_heldout)
# Output expQ/slotcheck/recall/NAME.json (a pass whose file exists is skipped, so a requeue resumes).
# The first ae pass is preceded by a 5-page plumbing pass (NAME_lim5.json) that must print stride 8 and non-empty text.
# Pair with: python3 -I scripts/longdoc/recall_pair.py --ref <ref json> --a <a json> --b <b json>
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
export HF_HOME=/beegfs/scratch/user/hdejean/Reranking/cache
R=/beegfs/scratch/user/hdejean/pisco
nvidia-smi -L
echo "FACTORS kind=probe_recall arm=O_recall passes=$* px=1003520"
echo "node $(hostname)"
md5sum scripts/longdoc/mem_recall.py scripts/probe_mem_recall.py scripts/eval_mm_docqa.py pisco/model.py \
  scripts/slotcheck/recall.sbatch | cut -c1-8,33-
mkdir -p $R/expQ/slotcheck/recall
SMOKED=0
for SPEC in "$@"; do
  IFS=: read -r NAME ARM SET <<< "$SPEC"
  [[ -n "$NAME" && -n "$ARM" && -n "$SET" ]] || { echo "bad spec $SPEC (NAME:ARM:SET)"; exit 2; }
  DATA=$R/eval_data/$SET; [[ -d "$DATA" ]] || { echo "missing $DATA"; exit 2; }
  case $ARM in
    ref)   A=(--arm ref) ;;
    start) A=(--arm ae --checkpoint_path $R/expQ/archq3/pt_mm_5M_spread/model) ;;
    *)     [[ -d "$ARM" ]] || { echo "missing model dir $ARM"; exit 2; }; A=(--arm ae --checkpoint_path "$ARM") ;;
  esac
  OUT=$R/expQ/slotcheck/recall/$NAME.json
  if [[ "$ARM" != ref && $SMOKED == 0 ]]; then
    echo "=== plumbing: $NAME 5 pages ==="
    S=${OUT%.json}_lim5.json
    [[ -e "$S" ]] || pixi run python -u scripts/longdoc/mem_recall.py "${A[@]}" --data_path $DATA --per_source 100 \
      --limit 5 --image_max_pixels 1003520 --out $S || { echo "plumbing pass failed"; exit 4; }
    python3 -I -c "import json,sys; r=json.load(open(sys.argv[1]))['records']; w=[len(x['text'].split()) for x in r]; \
s={x.get('stride') for x in r}; print('[plumbing] pages', len(r), 'words', w, 'stride', s); \
sys.exit(0 if len(r)==5 and min(w)>0 and s=={8} else 4)" "$S" || { echo "PLUMBING FAIL"; exit 4; }
    SMOKED=1
  fi
  [[ -e "$OUT" ]] && { echo "exists, skipped: $OUT"; continue; }
  echo "=== pass $NAME: arm=$ARM set=$SET ==="
  pixi run python -u scripts/longdoc/mem_recall.py "${A[@]}" --data_path $DATA --per_source 100 \
    --image_max_pixels 1003520 --out $OUT || { echo "pass $NAME failed"; exit 1; }
done
echo "=== EXIT CODE 0 ==="
