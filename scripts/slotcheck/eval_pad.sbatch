#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:45:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/expQ/slotcheck/eval_pad_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/expQ/slotcheck/eval_pad_%j.out
# Campaign slotcheck, arm P (check 2b): R on the 1,500 document questions, compressor batch padded to a multiple of 64
# (training) vs not padded (the eval's own), ONE job = one node for both arms of a pair.
#   eval_pad.sbatch <cell under expQ/slotcheck, holding model -> R dir> <steps: plain|none|64, comma list> [LIMIT]
#   plain = the shared eval_mm_4b.sh itself (PX=1003520, TOPK 4, brief, 32 tokens); none / 64 = scripts/slotcheck/eval_pad.py
#   with the same eval_mm_docqa.py flags. Env JUDGE=0 skips the judge. After the steps: plain vs none must be
#   identical (wrapper invariant, exit 4 if not); none vs 64 is compared and, when judged, paired (64 minus none).
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
export HF_HOME=/beegfs/scratch/user/hdejean/Reranking/cache
unset RAW DML RANK DENSE DROP KEEPK XD CSLICE MQ KEEPF KEEPM HS CHUNK CML GOLD STYLE MERGE CFG NSAMP MINPX TOPK RATE ADAPTER JPROMPT JMODEL
nvidia-smi -L
CELL=${1:?cell under expQ/slotcheck}; STEPS=${2:?steps}; LIMIT=${3:-}
DATA=/beegfs/scratch/user/hdejean/pisco/eval_data/mm_docqa_1500_1m
CKPT=/beegfs/scratch/user/hdejean/pisco/expQ/slotcheck/$CELL/model
[[ -e "$CKPT/config.json" ]] || { echo "missing $CKPT"; exit 2; }
RUN_DIR=$(dirname "$CKPT"); mkdir -p "$RUN_DIR/eval/.done"
PXV=1003520; J=${JUDGE:-1}
echo "FACTORS kind=eval_pad arm=P2b cell=$CELL steps=$STEPS limit=${LIMIT:-all} judge=$J model=$(readlink -f $CKPT) set=$(basename $DATA) px=$PXV topk=4 mnt=32"
echo "node $(hostname)"
md5sum eval_mm_4b.sh scripts/eval_mm_docqa.py scripts/judge_mm_docqa.py scripts/judge_prompts.py scripts/paired_judge.py pisco/model.py pisco/mm_collator.py pisco/mm_utils.py scripts/slotcheck/eval_pad.py scripts/slotcheck/compare_preds.py | cut -c1-8,33-
BASE="$RUN_DIR/eval/mm_docqa_pisco_px${PXV}_$(basename $DATA)"
LIMARG=""; LS=""; [[ -n "$LIMIT" ]] && { LIMARG="--limit $LIMIT"; LS="_lim$LIMIT"; }
json_ok() { python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$1" 2>/dev/null; }
RC=0
for S in ${STEPS//,/ }; do
  case $S in
    plain)
      echo "=== step plain: eval_mm_4b.sh (shared launcher, unchanged) ==="
      PX=$PXV JUDGE=$J bash eval_mm_4b.sh "$CKPT" pisco "$LIMIT" "$DATA" || RC=$? ;;
    none|64)
      OUT="${BASE}_pad${S}${LS}.json"; MARK="$RUN_DIR/eval/.done/$(basename "$OUT").${SLURM_JOB_ID:-nojob}"
      echo "=== step pad=$S -> $OUT ==="
      if [[ -f "$MARK.gen" ]] && json_ok "$OUT"; then echo "[eval_pad] requeue: generation already done, reusing $OUT"
      else
        SLOTCHECK_PAD=$S pixi run python -u scripts/slotcheck/eval_pad.py --mode pisco --checkpoint_path "$CKPT" \
          --data_path "$DATA" --output_path "$OUT" --topk_docs 4 --image_max_pixels $PXV --max_new_tokens 32 $LIMARG \
          || { RC=$?; echo "!! generation pad=$S failed rc=$RC"; continue; }
        touch "$MARK.gen"
      fi
      if [[ "$J" == "1" ]]; then
        if [[ -f "$MARK.judge" ]]; then echo "[eval_pad] requeue: judge already done"
        elif pixi run python -u scripts/judge_mm_docqa.py "$OUT" --prompt rubric --judge_model Qwen/Qwen3-8B; then touch "$MARK.judge"
        else RC=5; echo "!! judge pad=$S failed"; fi
      fi ;;
    *) echo "unknown step $S"; exit 2 ;;
  esac
done
P="$BASE$LS.json"; N="${BASE}_padnone$LS.json"; X="${BASE}_pad64$LS.json"
if [[ ",$STEPS," == *",plain,"* && ",$STEPS," == *",none,"* ]]; then
  python3 scripts/slotcheck/compare_preds.py "$P" "$N" --names plain wrapper_none --require_identical || RC=4
fi
if [[ ",$STEPS," == *",none,"* && ",$STEPS," == *",64,"* ]] && json_ok "$N" && json_ok "$X"; then
  python3 scripts/slotcheck/compare_preds.py "$X" "$N" --names pad64 padnone
  if [[ "$J" == "1" ]]; then python3 scripts/paired_judge.py "$X" "$N" --names pad64 padnone; fi
  python3 - "$X" "$N" "$CELL" <<'PY'
import json, sys
x, n = (json.load(open(p)) for p in sys.argv[1:3])
rx = {r["query_id"]: r for r in x["records"]}; rn = {r["query_id"]: r for r in n["records"]}
q = [k for k in rn if k in rx]
same = sum(rx[k]["prediction"] == rn[k]["prediction"] for k in q)
out = f"RESULT arm=P2b cell={sys.argv[3]} rows={len(q)} same_pred={same}"
if all(rx[k].get("judge") is not None and rn[k].get("judge") is not None for k in q):
    jx = sum(rx[k]["judge"] for k in q) / len(q); jn = sum(rn[k]["judge"] for k in q) / len(q)
    out += f" judge_pad64={jx:.4f} judge_padnone={jn:.4f} diff={jx - jn:+.4f} unparse64={x['summary'].get('judge_unparseable')} unparseNone={n['summary'].get('judge_unparseable')}"
mx = sum(rx[k].get("mems_used", 0) for k in q); mn = sum(rn[k].get("mems_used", 0) for k in q)
print(out + f" mems_ratio={mx / max(mn, 1):.4f}", flush=True)
PY
fi
echo "=== EXIT CODE $RC ==="; exit $RC
