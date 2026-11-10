#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/build_textvqa_ocr_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/build_textvqa_ocr_%j.out
# Arm F of campaign probes (CPU only): eval_data/probes_textvqa_ocr = the 500 TextVQA rows of mm_naturalqa_1500
# with their OCR tokens (lmms-lab/textvqa validation, streamed from the hub) as docs.
#   expmon launch --exp probes --stage build scripts/probes/build_textvqa_ocr.sbatch
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
echo "FACTORS kind=build_textvqa_ocr args=$(echo "$@" | tr ' ' ',')"
export HF_HOME=/beegfs/scratch/user/hdejean/Reranking/cache
curl -sS -o /dev/null -w "hub reachable: HTTP %{http_code}\n" --max-time 20 https://huggingface.co/api/datasets/lmms-lab/textvqa || echo "hub NOT reachable"
pixi run python -u scripts/probes/build_textvqa_ocr.py "$@"
RC=$?
echo "=== EXIT CODE $RC ==="
exit $RC
