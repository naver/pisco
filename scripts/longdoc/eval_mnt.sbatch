#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/expQ/longdoc/eval_mnt_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/expQ/longdoc/eval_mnt_%j.out
# Campaign longdoc, arm E: the shared eval launcher eval_mm_4b.sh with ONE change, the answer cap.
#   MNT=256 expmon launch --exp longdoc --stage E --env MNT=256 … scripts/longdoc/eval_mnt.sbatch <same args as eval_mm_4b.sh>
# Why: on LongDocURL the uncompressed decoder answers in full sentences and the shared 32-token cap cuts ~2/3 of its
# answers mid-sentence (jobs 191634 / 191635), so its score measures the cap, not the reading.
# How: at run time, a copy of the CURRENT eval_mm_4b.sh is made with (1) `MNT=32` -> `MNT=$MNT` and the output-name
# suffix `_mnt$MNT` (a new file name: never overwrites a recorded result), (2) `max_new_tokens=$MNT` in FACTORS.
# Everything else (env, row builders, judge, requeue markers) is the shared launcher byte for byte; both md5s are logged.
# The shared launcher itself is never edited (siblings pair on its md5).
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
[[ "${MNT:-}" =~ ^[0-9]+$ && "$MNT" != "32" ]] || { echo "eval_mnt: set env MNT to an integer other than 32 (32 = use eval_mm_4b.sh)"; exit 2; }
[[ -n "${STYLE:-}" && "$STYLE" != "brief" ]] && { echo "eval_mnt: STYLE=expand sets its own cap; refused"; exit 2; }
D=/beegfs/scratch/user/hdejean/pisco/expQ/longdoc/launchers; mkdir -p "$D"
DER="$D/eval_mm_4b_mnt.${SLURM_JOB_ID:-local}.sh"
sed -e 's/^STYLEARG=""; MNT=32; /STYLEARG=""; MNT=${MNT:?}; SUF="${SUF}_mnt${MNT}"; /' \
    -e 's/^echo "FACTORS kind=eval_mm mode=/echo "FACTORS kind=eval_mm max_new_tokens=$MNT mode=/' \
    eval_mm_4b.sh > "$DER"
n1=$(grep -cF 'STYLEARG=""; MNT=${MNT:?}; SUF="${SUF}_mnt${MNT}"; ' "$DER"); n2=$(grep -c 'FACTORS kind=eval_mm max_new_tokens=' "$DER")
nd=$(diff eval_mm_4b.sh "$DER" | grep -c '^>')
[[ "$n1" == 1 && "$n2" == 1 && "$nd" == 2 ]] || { echo "eval_mnt: substitution failed (n1=$n1 n2=$n2 changed=$nd); eval_mm_4b.sh changed shape"; exit 3; }
md5sum eval_mm_4b.sh "$DER" scripts/eval_mm_docqa.py scripts/judge_mm_docqa.py pisco/model.py
export MNT
bash "$DER" "$@"
