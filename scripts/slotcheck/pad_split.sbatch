#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/expQ/slotcheck/pad_split_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/expQ/slotcheck/pad_split_%j.out
# Campaign slotcheck, arm P (check 2a split): which padding path moves the <MEM>s (1-row leak vs 2-row masked),
# with a forced-zeroing cause test and an fp32 pass (rounding or real). Exit 3 = the repeat control failed.
# Usage: pad_split.sbatch [N_PAGES=40]
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
NP=${1:-40}
echo "FACTORS kind=probe arm=P2a_split n_pages=$NP widths=17,63 passes=bf16,fp32 model=ft_mm_4b_184778 set=mm_docqa_1500_1m"
echo "node $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
md5sum scripts/slotcheck/pad_split.py scripts/eval_mm_docqa.py pisco/model.py pisco/mm_collator.py pisco/mm_utils.py | cut -c1-8,33-
OUT=/beegfs/scratch/user/hdejean/pisco/expQ/slotcheck/pad_split_${SLURM_JOB_ID}
env CONDA_OVERRIDE_CUDA=12.7 pixi run python -u scripts/slotcheck/pad_split.py --out "$OUT" --n_pages "$NP" --fp32
