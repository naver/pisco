#!/bin/bash
# Phase 2 (long-doc compression study): one grid cell = pretrain -> finetune for a
# given (compr_rate, compressor_max_length) on Qwen3-0.6B. Eval is run separately
# afterwards (scripts/eval_phase2_grid.py).
#
# Two hardware modes via GPU_KIND (default a100):
#   A100/H100 (native bf16, single GPU):
#     COMPR_RATE=32 CML=256 sbatch -p gpu --constraint=gpu_a100 phase2_cell.sbatch
#   V100 (no native bf16 -> fp16 + fp32 master weights; use a whole 4-GPU node via DDP):
#     GPU_KIND=v100 COMPR_RATE=32 CML=256 sbatch -p gpu-be --constraint=gpu_v100 \
#       --gres=gpu:4 --cpus-per-task=16 phase2_cell.sbatch
#   SMOKE=1 ... sbatch -p debug ...
#SBATCH -J p2cell
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -o logs/%j.out

set -euo pipefail
cd /beegfs/scratch/user/hdejean/pisco

R=${COMPR_RATE:?set COMPR_RATE}
C=${CML:?set CML}
SMOKE=${SMOKE:-0}
GPU_KIND=${GPU_KIND:-a100}
NGPU=${SLURM_GPUS_ON_NODE:-1}

ROOT=$(pwd)/phase2/r${R}_c${C}
PT=$ROOT/pt
FT=$ROOT/ft

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# How to launch train.py: plain python (1 GPU) or accelerate DDP (multi-GPU node).
if [ "$NGPU" -gt 1 ]; then
  PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
  LAUNCH="pixi run accelerate launch --multi_gpu --num_processes $NGPU --main_process_port $PORT"
else
  LAUNCH="pixi run python"
fi

if [ "$SMOKE" = "1" ]; then
  PT_SAMPLES=64; FT_SAMPLES=64
  PT_EXTRA="hf_training.per_device_train_batch_size=2 hf_training.gradient_accumulation_steps=1 +hf_training.max_steps=2 hf_training.warmup_steps=0 hf_training.logging_steps=1 hf_training.eval_steps=2 hf_training.save_steps=2 +hf_training.optim=adamw_torch"
  FT_EXTRA="$PT_EXTRA"
elif [ "$GPU_KIND" = "v100" ]; then
  PT_SAMPLES=${PT_SAMPLES:-100000}
  FT_SAMPLES=${FT_SAMPLES:-30000}
  # V100 (Volta, no native bf16): fp16 AMP + fp32 master weights (grad scaler needs
  # fp32 params) + small per-GPU batch (fp32 doubles memory, 32GB). DDP across the
  # node's GPUs recovers the throughput Volta loses vs A100. See CLAUDE.md Hardware.
  PREC="hf_training.bf16=False +hf_training.fp16=True ++model.init_args.config.torch_dtype=float32"
  # Option 2 (FP16_BASE=1): load the frozen base backbones in fp16 (trainable stays
  # fp32). Halves base-weight memory -> fits 4B+ on a 32GB V100. Saved into the
  # checkpoint config, so the finetune step inherits it. Lets you raise PT_BS.
  if [ "${FP16_BASE:-0}" = "1" ]; then
    PREC="$PREC ++model.init_args.config.frozen_base_dtype=float16"
    DEFAULT_PT_BS=4
  else
    DEFAULT_PT_BS=2
  fi
  PT_EXTRA="hf_training.per_device_train_batch_size=${PT_BS:-$DEFAULT_PT_BS} hf_training.gradient_accumulation_steps=$((64 / NGPU > 0 ? 64 / NGPU : 1)) $PREC"
  FT_EXTRA="$PREC"
else
  PT_SAMPLES=${PT_SAMPLES:-100000}
  FT_SAMPLES=${FT_SAMPLES:-30000}
  # A100/H100 native bf16 (config defaults). bs=8 x accum=16 (eff 128) fits 80GB.
  PT_EXTRA="hf_training.per_device_train_batch_size=${PT_BS:-8} hf_training.gradient_accumulation_steps=16"
  FT_EXTRA=""
fi

echo "=== Phase2 cell r=$R c=$C kind=$GPU_KIND ngpu=$NGPU smoke=$SMOKE pt=$PT_SAMPLES ft=$FT_SAMPLES ==="

# 1) Pretrain (sets compr_rate + chunk length; learns connector/LoRA at this budget)
$LAUNCH pisco/train.py --config-name=pt_lora \
  output_dir="$PT" \
  data.samples=$PT_SAMPLES \
  ++model.init_args.config.compr_rate=$R \
  ++collator_kwargs.compressor_max_length=$C \
  $PT_EXTRA

# 2) Finetune (RAG) from the pretrained checkpoint, chunking long docs
$LAUNCH pisco/train.py --config-name=finetune \
  output_dir="$FT" \
  model_name_or_path="$PT/model" \
  data.samples=$FT_SAMPLES \
  ++collator_kwargs.compressor_max_length=$C \
  ++collator_kwargs.chunk_docs=True \
  $FT_EXTRA

echo "=== CELL DONE r=$R c=$C  ft_model=$FT/model ==="
