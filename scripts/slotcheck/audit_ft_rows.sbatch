#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0:30:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/expQ/slotcheck/audit_ft_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/expQ/slotcheck/audit_ft_%j.out
# Campaign slotcheck, arm C (check 1, the feasibility gate): rows of the multimodal finetune sets whose answer page
# the collator's topk_docs=4 cap drops. CPU only, no model weights (tokenizers + image processor for the controls).
# Exit 3 = the gate or health failed (outputs written). Usage: audit_ft_rows.sbatch [N_CONTROL=50]
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
NC=${1:-50}
echo "FACTORS kind=audit arm=C n_control=$NC topk_docs=4 sets=docpages_20k_1m_teacher,natural_8k_teacher"
md5sum scripts/slotcheck/audit_ft_rows.py pisco/mm_collator.py pisco/collator.py | cut -c1-8,33-
OUT=/beegfs/scratch/user/hdejean/pisco/expQ/slotcheck/audit_ft_${SLURM_JOB_ID}
env CONDA_OVERRIDE_CUDA=12.7 pixi run python -u scripts/slotcheck/audit_ft_rows.py --out "$OUT" --n_control "$NC"
