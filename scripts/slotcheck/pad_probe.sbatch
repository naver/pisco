#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:45:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/expQ/slotcheck/pad_probe_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/expQ/slotcheck/pad_probe_%j.out
# Campaign slotcheck, arm P (check 2a): the same document pages compressed with 0 / 17 / 63 left pad tokens, the
# <MEM> vectors compared (cosine, max abs) against a same-input repeat (noise floor) and another page (planted).
# Exit 3 = a control or health failed. Usage: pad_probe.sbatch [N_PAGES=120] [N_BATCH4=30]
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
NP=${1:-120}; NB=${2:-30}
echo "FACTORS kind=probe arm=P2a n_pages=$NP n_batch4=$NB widths=0,17,63 model=ft_mm_4b_184778 set=mm_docqa_1500_1m"
echo "node $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
md5sum scripts/slotcheck/pad_probe.py scripts/eval_mm_docqa.py pisco/model.py pisco/mm_collator.py pisco/mm_utils.py | cut -c1-8,33-
OUT=/beegfs/scratch/user/hdejean/pisco/expQ/slotcheck/pad_probe_${SLURM_JOB_ID}
env CONDA_OVERRIDE_CUDA=12.7 pixi run python -u scripts/slotcheck/pad_probe.py --out "$OUT" --n_pages "$NP" --n_batch4 "$NB"
