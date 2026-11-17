#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=0:20:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/qafreeze_build_init_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/qafreeze_build_init_%j.out
#
# Campaign qafreeze, arm B: build the hybrid start model of arm Q for each seed given.
#   init_s<seed> = a copy of the reference finetune's model dir (R: QA-trained compressor/ + connector.pt + config.json)
#                  with decoder_state.pt REPLACED by the stage-2 decoder (expQ/archq3/pt_mm_5M_spread, R's own start).
#   PISCO.from_pretrained reads exactly decoder_state.pt, compressor/compressor_state.pt, connector.pt, config.json.
# Asserts (exit 4 on any failure, the dir is left as init_s<seed>.FAILED for inspection):
#   md5: decoder_state == stage-2's; compressor_state, connector, config == R's;
#   torch: swapped decoder_state has R's key names and shapes (and the stage-2 file's), same dtypes.
# Never overwrites: an existing init_s<seed> -> exit 3.
# Usage: expmon launch --exp qafreeze --stage build scripts/qafreeze/build_init.sbatch 42 43

source ~/.bashrc
ROOT=/beegfs/scratch/user/hdejean/pisco; cd $ROOT
S2=$ROOT/expQ/archq3/pt_mm_5M_spread/model
[ $# -ge 1 ] || { echo "usage: build_init.sbatch SEED [SEED ...]"; exit 2; }
echo "FACTORS kind=build_init seeds=$* decoder_from=pt_mm_5M_spread"
git rev-parse --short HEAD
mkdir -p $ROOT/expQ/qafreeze
RC=0
for SEED in "$@"; do
  case "$SEED" in
    42) R=$ROOT/expQ/ft_mm_4b_184778/model ;;
    43) R=$ROOT/expQ/ft_mm_4b_184780/model ;;
    44) R=$ROOT/expQ/freeze/ft_unfrozen_s44/model ;;
    *) echo "no reference for seed $SEED"; exit 2 ;;
  esac
  OUT=$ROOT/expQ/qafreeze/init_s$SEED
  if [ -e "$OUT" ]; then echo "refusing to overwrite $OUT"; exit 3; fi
  TMP=$OUT.building_$SLURM_JOB_ID
  echo "== seed $SEED: R=$R -> $OUT"
  cp -a "$R" "$TMP" || exit 4
  cp --preserve=mode "$S2/decoder_state.pt" "$TMP/decoder_state.pt" || exit 4
  OK=1
  chk() { a=$(md5sum < "$1" | cut -c1-32); b=$(md5sum < "$2" | cut -c1-32)
          if [ "$a" = "$b" ]; then echo "  md5 OK   $3 ${a:0:8}"; else echo "  md5 FAIL $3 $a vs $b"; OK=0; fi; }
  chk "$TMP/decoder_state.pt"                "$S2/decoder_state.pt"                "decoder_state.pt == stage-2"
  chk "$TMP/compressor/compressor_state.pt"  "$R/compressor/compressor_state.pt"  "compressor_state.pt == R"
  chk "$TMP/connector.pt"                    "$R/connector.pt"                    "connector.pt == R"
  chk "$TMP/config.json"                     "$R/config.json"                     "config.json == R"
  R_DEC="$R/decoder_state.pt" NEW_DEC="$TMP/decoder_state.pt" pixi run python - <<'PY' || OK=0
import os, sys, torch
a = torch.load(os.environ["R_DEC"], weights_only=True, map_location="cpu")
b = torch.load(os.environ["NEW_DEC"], weights_only=True, map_location="cpu")
ka, kb = set(a), set(b)
bad = sorted(ka ^ kb)
shape = [k for k in ka & kb if a[k].shape != b[k].shape or a[k].dtype != b[k].dtype]
same = sum(torch.equal(a[k], b[k]) for k in ka & kb)
print(f"  keys R {len(ka)}  swapped {len(kb)}  only-one-side {len(bad)}  shape/dtype mismatch {len(shape)}  "
      f"tensors identical to R {same}/{len(ka & kb)} (must be < all: the decoder was reset)")
for k in bad[:10] + shape[:10]: print("   ", k)
sys.exit(0 if not bad and not shape and same < len(ka & kb) else 1)
PY
  {
    echo "built by job $SLURM_JOB_ID on $(hostname) $(date -Is)"
    echo "R (compressor/, connector.pt, config.json, tokenizer): $R"
    echo "decoder_state.pt: $S2/decoder_state.pt"
  } > "$TMP/QAFREEZE_PROVENANCE.txt"
  if [ $OK = 1 ]; then mv "$TMP" "$OUT"; echo "  BUILT $OUT"; else mv "$TMP" "$OUT.FAILED"; echo "  FAILED seed $SEED"; RC=4; fi
done
echo "=== EXIT CODE $RC ==="; exit $RC
