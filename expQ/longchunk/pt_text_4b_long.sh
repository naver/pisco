#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=7-00:00:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/pt_text_long_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/pt_text_long_%j.out
#
# Campaign longchunk, stage 1 (text pretraining) with LONG compressor chunks on the spread <MEM> layout.
# Same recipe as qwen35_4b_pt.sh (Qwen3.5-4B/4B, pt_lora, rate 8, spread layout, stride 8, lr 5e-5,
# effective batch 192 docs = BS x 4 GPUs x ACCUM) except the two lengths, which are the campaign's factors:
#   CML = collator compressor_max_length (chunk length, tokens); the reference chain used 256
#   DML = collator decoder_max_length (text kept per document, the AE target); the reference used 512 (= 2 x CML)
# Usage: sbatch pt_text_4b_long.sh <SAMPLES> <CML> <DML> <BS> <ACCUM> [hydra overrides...]
#   C256 control on MS MARCO (same lengths as the reference chain 180486): 1000000 256 512 24 2
#   arm L1024:                                   1000000 1024 2048 6 8
#   arm L2048:                                   1000000 2048 4096 3 16
# Keep BS x ACCUM = 48 so the effective batch stays 192 documents (one variable: chunk length).
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/pisco
export HF_HOME=/beegfs/scratch/user/hdejean/Reranking/cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -L

SAMPLES=${1:?usage: sbatch pt_text_4b_long.sh SAMPLES CML DML BS ACCUM [hydra overrides...]}
CML=${2:?CML (compressor chunk length)}
DML=${3:?DML (decoder max length)}
BS=${4:?BS}
ACCUM=${5:?ACCUM}
shift 5; EXTRA="$@"
# env PT_DATA: the stage-1 text set (data.training_dataset; "path" or "path:N"). Default = the campaign's corpus,
# data/msmarco_docs (MS MARCO documents: 3.2M web pages, title + body, shuffled seed 42; built by job 188358).
# The recipe's original corpus is PT_DATA=EleutherAI/SmolLM2-135M-10B (too short for long chunks: 24% >= 1,024 tokens).
DATA=${PT_DATA:-/beegfs/scratch/user/hdejean/pisco/data/msmarco_docs}
MODEL=Qwen/Qwen3.5-4B
TAG=$((SAMPLES / 1000000))
OUT=${PT_OUT:-/beegfs/scratch/user/hdejean/pisco/expQ/longchunk/pt_text_${TAG}M_cml${CML}_${SLURM_JOB_ID}}
mkdir -p "$OUT"
echo "FACTORS kind=pt_text backbone=qwen3.5-4b data=$(basename "$DATA") samples=$SAMPLES cml=$CML dml=$DML bs=$BS accum=$ACCUM rate=8 layout=spread extra=$(echo $EXTRA | tr ' ' ',')"

PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')
echo "=== PISCO stage 1 4B/4B rate 8 spread: samples=$SAMPLES cml=$CML dml=$DML bs=$BS accum=$ACCUM -> $OUT port=$PORT ==="
pixi run accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
  --main_process_port $PORT pisco/train.py --config-name=pt_lora \
  compressor="$MODEL" decoder="$MODEL" \
  hydra.run.dir="$OUT" \
  model.init_args.config.compr_rate=8 \
  ++model.init_args.config.compressor_mem_layout=spread \
  ++model.init_args.config.mem_position_stride=8 \
  collator_kwargs.compressor_max_length=$CML \
  collator_kwargs.decoder_max_length=$DML \
  data.training_dataset="$DATA" \
  data.samples=$SAMPLES \
  hf_training.per_device_train_batch_size=$BS \
  hf_training.gradient_accumulation_steps=$ACCUM \
  hf_training.gradient_checkpointing=true \
  hf_training.learning_rate=5e-5 \
  hf_training.save_steps=2000 \
  hf_training.save_total_limit=2 \
  hf_training.eval_steps=2000 \
  +hf_training.optim=adamw_torch \
  +hf_training.ddp_find_unused_parameters=true $EXTRA
echo "=== EXIT CODE $? ==="
