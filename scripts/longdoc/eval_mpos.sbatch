#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/expQ/longdoc/eval_mpos_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/expQ/longdoc/eval_mpos_%j.out
# Campaign longdoc, arm P (position probe): scripts/longdoc/eval_mnt.sbatch (answer cap MNT) with ONE more change:
# the generation step runs scripts/longdoc/eval_mpos.py (= scripts/eval_mm_docqa.py imported unchanged, plus env
# MPOS = a constant or a {query_id: N} json; see that file) instead of scripts/eval_mm_docqa.py.
#   expmon launch --exp longdoc --stage P --env MNT=512 --env MPOS=<N|table.json> [--env MPOS_REF=<recorded json>] …
#     scripts/longdoc/eval_mpos.sbatch <same args as eval_mm_4b.sh>
# The output name gains `_mnt$MNT_mpos<tag>` (tag = N, or the table's basename): never overwrites a recorded file.
# MPOS_REF makes the job exit 4 unless every prediction equals the reference's (the identity check).
# The shared launcher and eval script are never edited (siblings pair on their md5); both md5s are logged.
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
[[ "${MNT:-}" =~ ^[0-9]+$ && "$MNT" != "32" ]] || { echo "eval_mpos: set env MNT to an integer other than 32"; exit 2; }
[[ -n "${STYLE:-}" && "$STYLE" != "brief" ]] && { echo "eval_mpos: STYLE=expand sets its own cap; refused"; exit 2; }
[[ -n "${MPOS:-}" ]] || { echo "eval_mpos: set env MPOS (0 = identity)"; exit 2; }
if [[ "$MPOS" =~ ^[0-9]+$ ]]; then MTAG="$MPOS"; else [[ -f "$MPOS" ]] || { echo "eval_mpos: no file $MPOS"; exit 2; }; MTAG=$(basename "$MPOS" .json); fi
D=/beegfs/scratch/user/hdejean/pisco/expQ/longdoc/launchers; mkdir -p "$D"
DER="$D/eval_mm_4b_mpos.${SLURM_JOB_ID:-local}.sh"
sed -e 's/^STYLEARG=""; MNT=32; /STYLEARG=""; MNT=${MNT:?}; SUF="${SUF}_mnt${MNT}_mpos${MTAG:?}"; /' \
    -e 's/^echo "FACTORS kind=eval_mm mode=/echo "FACTORS kind=eval_mm max_new_tokens=$MNT mpos=$MTAG mode=/' \
    -e 's#^  pixi run python -u scripts/eval_mm_docqa.py #  pixi run python -u scripts/longdoc/eval_mpos.py #' \
    eval_mm_4b.sh > "$DER"
n1=$(grep -cF 'STYLEARG=""; MNT=${MNT:?}; SUF="${SUF}_mnt${MNT}_mpos${MTAG:?}"; ' "$DER"); n2=$(grep -c 'FACTORS kind=eval_mm max_new_tokens=' "$DER")
n3=$(grep -c '^  pixi run python -u scripts/longdoc/eval_mpos.py ' "$DER"); nd=$(diff eval_mm_4b.sh "$DER" | grep -c '^>')
[[ "$n1" == 1 && "$n2" == 1 && "$n3" == 1 && "$nd" == 3 ]] || { echo "eval_mpos: substitution failed (n1=$n1 n2=$n2 n3=$n3 changed=$nd); eval_mm_4b.sh changed shape"; exit 3; }
md5sum eval_mm_4b.sh "$DER" scripts/eval_mm_docqa.py scripts/longdoc/eval_mpos.py scripts/judge_mm_docqa.py pisco/model.py
export MNT MPOS MTAG MPOS_REF
bash "$DER" "$@"
