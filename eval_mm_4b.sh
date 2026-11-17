#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/eval_mm_4b_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/eval_mm_4b_%j.out
#
# Multimodal doc-QA eval. MODE = pisco | base | pisco_text.
#   expmon launch --exp mmpisco --stage eval --parent <jid> eval_mm_4b.sh <CKPT_MODEL_DIR> [MODE=pisco] [LIMIT] [DATA] [IMAGE_RATE]
# Results -> <ckpt parent>/eval/mm_docqa_<MODE>[_r<RATE>].json  (base mode: expQ/base_qwen35_4b/eval/); judge-scored
# in place afterwards unless JUDGE=0.
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
export HF_HOME=/beegfs/scratch/user/hdejean/Reranking/cache
nvidia-smi -L

CKPT=${1:?model dir}
MODE=${2:-pisco}
LIMIT=${3:-}
DATA=${4:-/beegfs/scratch/user/hdejean/pisco/eval_data/mm_docqa_500}
RATE=${5:-}      # optional image_compr_rate override at eval (capacity probe); default = ckpt compr_rate
PX=${PX:-451584}    # image pixel budget (env); eval sets built at another budget need the matching PX
MINPX=${MINPX:-}    # lower bound of the budget (env); set it below 65536 for the equal-token low-resolution baseline
TOPK=${TOPK:-4}  # docs/frames per question (env); nav_ft_* windows carry 8 frames -> TOPK=8
RUN_DIR=$(dirname "$CKPT")
ADAPTERARG=""
if [[ "$MODE" == "base" && -n "$ADAPTER" ]]; then RUN_DIR=$(dirname "$ADAPTER"); ADAPTERARG="--base_adapter_path $ADAPTER"   # finetuned base (fair ceiling)
elif [[ "$MODE" == "base" ]]; then RUN_DIR=/beegfs/scratch/user/hdejean/pisco/expQ/base_qwen35_4b; fi
mkdir -p "$RUN_DIR/eval"
SUF=""; RATEARG=""; [[ -n "$RATE" ]] && { SUF="_r${RATE}"; RATEARG="--image_compr_rate $RATE"; }
[[ "$TOPK" != "4" ]] && SUF="${SUF}_k${TOPK}"
[[ "$PX" != "451584" ]] && SUF="${SUF}_px${PX}"
MINPXARG=""; [[ -n "$MINPX" ]] && MINPXARG="--image_min_pixels $MINPX"
RAWARG=""; [[ -n "${RAW:-}" ]] && { RAWARG="--raw_policy $RAW"; DML=${DML:-6144}; SUF="${SUF}_raw${RAW}"; }   # env RAW=gold|route1|route2|all: pages fed raw (selective expansion)
DMLARG=""; [[ -n "${DML:-}" ]] && DMLARG="--decoder_max_length $DML"   # env DML: decoder prompt limit (default 2048)
RANKARG=""; [[ -n "${RANK:-}" ]] && { RANKARG="--rank_scores $RANK"; SUF="${SUF}_rk$(basename $RANK .json)"; }   # env RANK=<json {query_id: [score per page]}>: external retriever (RAW=rank1|rank2, KEEPK)
DENSEARG=""; [[ -n "${DENSE:-}" ]] && { DENSEARG="--dense_rate $DENSE"; SUF="${SUF}_dense${DENSE}"; }   # env DENSE=R (with RAW=route1): the picked page re-compressed at rate R, not raw (arm G)
DROPARG=""; [[ "${DROP:-0}" == "1" ]] && { DROPARG="--drop_unexpanded"; SUF="${SUF}_drop"; }   # env DROP=1: non-expanded pages left out (retrieve-then-read)
KEEPARG=""; [[ -n "${KEEPK:-}" ]] && { KEEPARG="--keep_ranked $KEEPK"; SUF="${SUF}_keep${KEEPK}"; }   # env KEEPK=K (base mode): only the K pages RANK ranks first
XDARG=""; [[ -n "${XD:-}" ]] && { XDARG="--extra_distractors $XD"; SUF="${SUF}_xd${XD}"; }   # env XD=N: N more pages (other docs, shuffled); set TOPK=4+N
CSARG=""; [[ -n "${CSLICE:-}" ]] && CSARG="--compress_slice $CSLICE"   # env CSLICE=N: N chunks per compressor forward (exact; whole tours OOM in one call); no SUF
MQARG=""; [[ -n "${MQ:-}" ]] && { MQARG="--mem_quant $MQ"; SUF="${SUF}_mq${MQ}"; }   # env MQ=int8|fp8|int4: <MEM> storage precision
KEEPFARG=""; [[ -n "${KEEPF:-}" ]] && { KEEPFARG="--mem_keep_frac $KEEPF --mem_keep_mode ${KEEPM:-first}"; SUF="${SUF}_keep${KEEPF/./}${KEEPM:+_$KEEPM}"; }   # env KEEPF=0.75 [KEEPM=even]: keep the first 75% (or every k-th) of every chunk's <MEM>s (budget probe)
HSARG=""; [[ -n "${HS:-}" ]] && { HSARG="--head_scales $HS"; SUF="${SUF}_hs$(basename $(dirname $HS) | sed s/seal_//)"; }   # env HS=<expQ/probes/seal_<jid>/head_scales.json>: SEAL per-head scales
CHUNKARG=""; [[ "${CHUNK:-0}" == "1" ]] && { CHUNKARG="--chunk_docs"; SUF="${SUF}_chunk"; }   # env CHUNK=1: split long text documents into chunks instead of truncating
CMLARG=""; [[ -n "${CML:-}" ]] && { CMLARG="--compressor_max_length $CML"; SUF="${SUF}_cml${CML}"; }   # env CML=N: compressor chunk length (default 256); long-chunk probes
GOLDARG=""; [[ "${GOLD:-0}" == "1" ]] && { GOLDARG="--gold_only"; SUF="${SUF}_gold"; }   # env GOLD=1: distractor pages removed
STYLEARG=""; MNT=32; [[ "${STYLE:-brief}" == "expand" ]] && { STYLEARG="--answer_style expand"; MNT=256; SUF="${SUF}_expand"; }   # env STYLE=expand: evidence, then answer
MERGEARG=""; [[ "${MERGE:-0}" == "1" ]] && { MERGEARG="--merge_lora"; SUF="${SUF}_merged"; }   # fold LoRA (inference speed)
CFGARG=""; [[ -n "${CFG:-}" ]] && { for kv in $CFG; do CFGARG="$CFGARG --config_override $kv"; done; SUF="${SUF}_cfg"; }   # env CFG="mem_position_stride=8 compressor_extra_layers=[18]": PISCOConfig fields the ckpt lacks (shake-outs). Own suffix: on 2026-10-29 a 4-row CFG shake-out without one overwrote 174908's 1500-Q result files
NSAMP=${NSAMP:-0}   # env: G sampled answers/question + greedy confidence (mmrl headroom probe); judge scores them too
SAMPARG=""; JSAMP=""; [[ "$NSAMP" != "0" ]] && { SAMPARG="--num_samples $NSAMP --temperature ${TEMP:-1.0} --top_p ${TOPP:-0.95}"; JSAMP="--samples"; SUF="${SUF}_s${NSAMP}"; }
CKNAME=$(basename "$CKPT"); [[ "$CKNAME" != "model" ]] && SUF="${SUF}_${CKNAME}"   # intermediate checkpoint previews
DTAG=$(basename "$DATA"); [[ "$DTAG" != "mm_docqa_500" ]] && SUF="${SUF}_${DTAG}"      # other eval sets (nav_ft_val, ...)
OUTJ="$RUN_DIR/eval/mm_docqa_${MODE}${SUF}.json"
LIM=""; [[ -n "$LIMIT" ]] && { LIM="--limit $LIMIT"; OUTJ="${OUTJ%.json}_lim${LIMIT}.json"; }   # a partial run gets its own file
echo "FACTORS kind=eval_mm mode=$MODE ckpt=$(basename $RUN_DIR) data=$(basename $DATA) limit=${LIMIT:-all} image_rate=${RATE:-ckpt} topk=$TOPK max_pixels=$PX merged=${MERGE:-0} samples_per_q=$NSAMP raw_policy=${RAW:-none} answer_style=${STYLE:-brief} gold_only=${GOLD:-0} mem_quant=${MQ:-none} extra_distractors=${XD:-0} rank=$(basename ${RANK:-none} .json) drop=${DROP:-0} keep_ranked=${KEEPK:-0}${DENSE:+ dense_rate=$DENSE}"

# Requeue resume (gpu-be preempts): a step that THIS job already finished is not redone when SLURM requeues it.
# Markers $RUN_DIR/eval/.done/<json name>.<SLURM_JOB_ID>.{gen,judge}, written after each step succeeds; the job id
# survives a requeue, so a new launch (new id) always regenerates. Not under SLURM = no markers, old behaviour.
MARK=""; [[ -n "${SLURM_JOB_ID:-}" ]] && { mkdir -p "$RUN_DIR/eval/.done"; MARK="$RUN_DIR/eval/.done/$(basename "$OUTJ").$SLURM_JOB_ID"; }
json_ok() { python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$1" 2>/dev/null; }   # a kill mid-write leaves a cut file
if [[ -n "$MARK" && -f "$MARK.gen" && -f "$OUTJ" ]] && json_ok "$OUTJ"; then
  echo "[eval_mm_4b] requeue (restart ${SLURM_RESTART_COUNT:-?}): generation already done by job $SLURM_JOB_ID, reusing $OUTJ"
  RC=0
else
  rm -f "${MARK:-/nonexistent}.gen" "${MARK:-/nonexistent}.judge"
  pixi run python -u scripts/eval_mm_docqa.py --mode "$MODE" --checkpoint_path "$CKPT" \
    --data_path "$DATA" --output_path "$OUTJ" --topk_docs $TOPK --image_max_pixels $PX $MINPXARG --max_new_tokens $MNT $LIM $RATEARG $ADAPTERARG $MERGEARG $CFGARG $SAMPARG $CSARG $RAWARG $STYLEARG $GOLDARG $MQARG $KEEPFARG $CMLARG $HSARG $CHUNKARG $XDARG $DMLARG $RANKARG $DROPARG $KEEPARG $DENSEARG
  RC=$?
  [[ $RC -eq 0 && -n "$MARK" ]] && touch "$MARK.gen"
fi
# LLM-judge re-scoring of the same file, in place: EM/F1 under-count verbose answers. Env JPROMPT = rubric (default)
# | project | openeqa (LLM-Match 1-5 marks, for eval_data/openeqa_*); JMODEL = judge model (default Qwen/Qwen3-8B;
# openeqa sets use Qwen/Qwen3-14B, the campaign_eqa judge that passed its Gate J).
if [[ $RC -eq 0 && "${JUDGE:-1}" == "1" ]]; then
  if [[ -n "$MARK" && -f "$MARK.judge" ]]; then
    echo "[eval_mm_4b] requeue: judge already done by job $SLURM_JOB_ID, $OUTJ kept"
  elif pixi run python -u scripts/judge_mm_docqa.py "$OUTJ" $JSAMP --prompt ${JPROMPT:-rubric} --judge_model ${JMODEL:-Qwen/Qwen3-8B}; then
    [[ -n "$MARK" ]] && touch "$MARK.judge"
  else
    echo "!! judge failed (eval json kept)"
  fi
fi
echo "=== EXIT CODE $RC ==="; exit $RC
