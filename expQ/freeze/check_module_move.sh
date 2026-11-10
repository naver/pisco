#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu-be
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0:20:00
#SBATCH --output=/beegfs/scratch/user/hdejean/pisco/check_move_freeze_%j.out
#SBATCH --error=/beegfs/scratch/user/hdejean/pisco/check_move_freeze_%j.out
# Per-module weight movement of a (short) finetune vs its init: fraction of entries changed, relative Frobenius move,
# for LoRA-A / LoRA-B (decoder, compressor), connector linears, norm gain. CPU.  args: INIT_MODEL_DIR TRAINED_MODEL_DIR
source ~/.bashrc; cd /beegfs/scratch/user/hdejean/pisco
INIT=${1:?init model dir} TRAINED=${2:?trained model dir}
echo "FACTORS kind=check_module_move init=$(basename $(dirname $INIT)) trained=$(basename $(dirname $TRAINED))"
INIT=$INIT TRAINED=$TRAINED pixi run python - <<'PY'
import os, torch
from collections import defaultdict
def load(d, f):
    p = os.path.join(d, f)
    return torch.load(p, weights_only=True, map_location="cpu") if os.path.exists(p) else {}
def group(k):
    if k.startswith("norm"): return "connector.norm_gain"
    if k[0].isdigit(): return "connector.linear"
    if "lora_A" in k: return "lora_A"
    if "lora_B" in k: return "lora_B"
    if "token_adapter" in k or "trainable_tokens" in k: return "token_adapter"
    return "other"
for part, f in (("decoder", "decoder_state.pt"), ("compressor", "compressor/compressor_state.pt"), ("connector", "connector.pt")):
    a, b = load(os.environ["INIT"], f), load(os.environ["TRAINED"], f)
    if not a or not b:
        print(f"{part}: missing in init or trained"); continue
    acc = defaultdict(lambda: [0, 0, 0.0, 0.0, 0.0])   # n, changed, |d|^2, |x|^2, |x| sum for RMS
    for k in a:
        if k not in b or a[k].shape != b[k].shape or not a[k].is_floating_point(): continue
        x, y = a[k].float(), b[k].float()
        g = acc[f"{part}.{group(k)}"]
        g[0] += x.numel(); g[1] += int((x != y).sum()); g[2] += float((y - x).pow(2).sum()); g[3] += float(x.pow(2).sum())
    for g, (n, ch, d2, x2, _) in sorted(acc.items()):
        if n == 0: continue
        print(f"{g:28s} {n/1e6:8.2f}M entries  changed {100*ch/n:5.1f}%  moved {(d2/x2)**0.5 if x2 > 0 else float('nan'):.4f} of its norm  "
              f"RMS {(x2/n)**0.5:.5f}  trained dtype {b[next(iter(b))].dtype}")
PY
echo "=== EXIT CODE $? ==="
