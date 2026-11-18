#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=3:20:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/expQ/spread/mech_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/expQ/spread/mech_%j.out
# Campaign spread, arm D (mechanism, no decoder question): runs the two recorded probes UNCHANGED, one model after
# another, with the args of the recorded stage-2 pair 183828-183832 (N_ROWS=40, all sets, REF_CKPT default):
#   expQ/archq2/probe_mixer.py     -> text ownership (text.ownership_own_wins), late-slot redundancy
#   expQ/archq2/probe_connector.py -> answer NLL on held-out finetuning rows (<set>.nll.as_is.mean)
# Only OUT changes: expQ/spread/mech/<name>_<jobid>/ holds probe_mixer.json + probe_connector.json (never writes under expQ/archq2).
#   expmon launch --exp spread --stage D … scripts/spread/mech_probes.sbatch <name>=<ckpt dir> [<name>=<ckpt dir> …]
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
export HF_HOME=/beegfs/scratch/user/hdejean/Reranking/cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export N_ROWS=40
nvidia-smi -L
echo "FACTORS kind=probe_mech n_rows=$N_ROWS models=$*"
md5sum expQ/archq2/probe_mixer.py expQ/archq2/probe_connector.py pisco/model.py
RC=0
for spec in "$@"; do
  NAME=${spec%%=*}; export CKPT=${spec#*=}
  D=/beegfs/scratch/user/hdejean/pisco/expQ/spread/mech/${NAME}_${SLURM_JOB_ID}; mkdir -p "$D"
  for P in mixer connector; do
    echo "=== $NAME $P ckpt=$CKPT start $(date +%T)"
    OUT=$D env CONDA_OVERRIDE_CUDA=12.7 pixi run python -u expQ/archq2/probe_$P.py || { echo "!!! $NAME $P failed"; RC=1; }
    ls "$D"/probe_$P.json 2>/dev/null
  done
done
echo "=== EXIT CODE $RC ==="; exit $RC
