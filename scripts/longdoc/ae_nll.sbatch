#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/expQ/longdoc/aenll_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/expQ/longdoc/aenll_%j.out
# Campaign longdoc, arm T: stage-1 autoencoding probe (scripts/longdoc/ae_nll.py) on the two longchunk stage-1 text
# models, C256 (188422) and L1024 (188466), the same held-out MS MARCO documents (rows from 1,000,000 on, >= 1,024 tokens).
#   expmon launch --exp longdoc --stage T --time … scripts/longdoc/ae_nll.sbatch [N_DOCS]
# N_DOCS 500 = the law's cell; a smaller N is plumbing. Output expQ/longdoc/ae/ae_nll_n<N>.json; pair with
# scripts/longdoc/ae_pair.py.
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
export HF_HOME=/beegfs/scratch/user/hdejean/Reranking/cache
nvidia-smi -L
R=/beegfs/scratch/user/hdejean/pisco; N=${1:-500}
OUT=$R/expQ/longdoc/ae/ae_nll_n${N}.json
echo "FACTORS kind=probe_ae models=c256,l1024 n_docs=$N data=msmarco_docs start_row=1000000"
[[ -e "$OUT" ]] && { echo "refusing to overwrite $OUT"; exit 3; }
pixi run python -u scripts/longdoc/ae_nll.py --n_docs $N --out $OUT \
  --model c256=$R/expQ/longchunk/pt_text_1M_cml256_188422/model \
  --model l1024=$R/expQ/longchunk/pt_text_1M_cml1024_188466/model
RC=$?; echo "=== EXIT CODE $RC ==="; exit $RC
