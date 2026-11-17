#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/expQ/longdoc/rejudge_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/expQ/longdoc/rejudge_%j.out
# Campaign longdoc, arm J: judge-sensitivity check. Re-scores COPIES of judged result files with the project
# (Yes/No) prompt of scripts/judge_mm_docqa.py; the original files are never touched.
#   expmon launch --exp longdoc --stage J --time … scripts/longdoc/rejudge.sbatch NAME FILE [FILE …]
# Copies: expQ/longdoc/rejudge/NAME/<dir above eval/>__<basename>. In each copy `judge` becomes the project-prompt
# verdict and the rubric verdict moves to `judge_rubric` (judge_mm_docqa.py keeps the previous prompt's scores).
# Then pair the copies with scripts/longdoc/length_effect.py exactly as the rubric pairs were.
# Requeue-safe: a copy already scored with the project prompt is skipped; any other copy is re-made from its original.
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
export HF_HOME=/beegfs/scratch/user/hdejean/Reranking/cache
nvidia-smi -L
NAME=$1; shift
[[ -z "$NAME" || $# -eq 0 ]] && { echo "usage: rejudge.sbatch NAME FILE [FILE …]"; exit 2; }
OUT=expQ/longdoc/rejudge/$NAME; mkdir -p "$OUT"
echo "FACTORS kind=rejudge arm=J name=$NAME prompt=project judge=Qwen/Qwen3-8B n_files=$#"
md5sum scripts/judge_mm_docqa.py scripts/judge_prompts.py
TODO=()
for f in "$@"; do
  [[ -f "$f" ]] || { echo "missing $f"; exit 2; }
  c="$OUT/$(basename "$(dirname "$(dirname "$f")")")__$(basename "$f")"
  if [[ -f "$c" ]] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$c'))['summary'].get('judge_prompt')=='project' else 1)"; then
    echo "done already: $c"; continue
  fi
  cp -f "$f" "$c"; echo "copy $f -> $c"; TODO+=("$c")
done
RC=0
[[ ${#TODO[@]} -gt 0 ]] && { pixi run python -u scripts/judge_mm_docqa.py "${TODO[@]}" --prompt project --judge_model Qwen/Qwen3-8B; RC=$?; }
echo "=== EXIT CODE $RC ==="; exit $RC
