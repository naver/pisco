"""SEAL-style probe (docs/literature_scan_2026-11.md rank 2, arXiv 2501.15225): with EVERYTHING frozen, learn one
scalar per attention head of the decoder's full-attention layers (Qwen3.5-4B: 8 layers x 16 heads = 128 numbers)
on a few hundred finetune rows, with the ordinary answer loss on the <MEM> prompt. The scale multiplies each head's
output before o_proj (a forward pre-hook, so no model code changes). Targets the "answer is in the slots, decoder
does not use it" gap. Writes <OUT>/head_scales.json for scripts/eval_mm_docqa.py --head_scales. Env: CKPT, TRAIN
(dataset dir, default the docpages finetune shard 0), N_TRAIN, N_HOLD, LR, EPOCHS, ACCUM, OUT.
"""
import json
import os
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import datasets  # noqa: E402

from pisco.model import PISCO  # noqa: E402
from pisco.mm_collator import MultimodalFineTuningCollator  # noqa: E402
from pisco.head_scales import install_head_scales  # noqa: E402

CKPT = os.environ.get("CKPT", "expQ/ft_mm_4b_183814/model")
TRAIN = os.environ.get("TRAIN", "data/mm_ft_docpages_20k_1m_teacher/shard_00")
N_TRAIN = int(os.environ.get("N_TRAIN", "200"))
N_HOLD = int(os.environ.get("N_HOLD", "100"))
LR = float(os.environ.get("LR", "1e-2"))
EPOCHS = int(os.environ.get("EPOCHS", "2"))
ACCUM = int(os.environ.get("ACCUM", "8"))
OUT = os.environ["OUT"]
dev = torch.device("cuda")
torch.manual_seed(0)
os.makedirs(OUT, exist_ok=True)

model = PISCO.from_pretrained(CKPT).to(dev).eval()
for p in model.parameters():
    p.requires_grad_(False)
scales, layers = install_head_scales(model.decoder, init=None)   # fresh ones, trainable
print(f"full-attention layers {layers}: {scales.numel()} scales", flush=True)
coll = MultimodalFineTuningCollator(
    compressor_tokenizer=model.compressor_tokenizer, decoder_tokenizer=model.decoder_tokenizer,
    compr_rate=model.compr_rate, compressor_max_length=256, decoder_max_length=1024,
    image_processor=model.image_processor, topk_docs=4, query_dependent=False, pad_to_multiple_of=None,
)
ds = datasets.load_from_disk(TRAIN)
ds = ds["train"] if "train" in getattr(ds, "keys", lambda: [])() else ds
perm = np.random.RandomState(0).permutation(len(ds))
tr_idx, ho_idx = perm[:N_TRAIN], perm[N_TRAIN:N_TRAIN + N_HOLD]


def loss_of(row, grad: bool):
    b = coll.torch_call([dict(row)])
    b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
    with torch.no_grad():   # the compressor and the connector are frozen: no graph needed there
        emb = model.compress(b["compressor_input_ids"], b["compressor_attention_mask"], pixel_values=b.get("pixel_values"),
                             image_grid_thw=b.get("image_grid_thw"), mm_token_type_ids=b.get("mm_token_type_ids"))
        embeds = model.replace_embeddings(emb, b["decoder_input_ids"])
    kw = {}
    pos = model.decoder_position_ids(b["decoder_input_ids"], b["decoder_attention_mask"])
    if pos is not None:
        kw["position_ids"] = pos
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        out = model.decoder(inputs_embeds=embeds, attention_mask=b["decoder_attention_mask"], labels=b["labels"], **kw)
    return out.loss


def holdout():
    with torch.no_grad():
        return float(np.mean([loss_of(ds[int(i)], False).item() for i in ho_idx]))


before = holdout()
print(f"held-out answer loss before: {before:.4f}", flush=True)
opt = torch.optim.Adam([scales], lr=LR)
step = 0
for ep in range(EPOCHS):
    order = np.random.RandomState(ep + 1).permutation(tr_idx)
    run = []
    for k, i in enumerate(order):
        loss = loss_of(ds[int(i)], True) / ACCUM
        loss.backward()
        run.append(float(loss) * ACCUM)
        if (k + 1) % ACCUM == 0:
            opt.step(); opt.zero_grad(); step += 1
            if step % 5 == 0:
                print(f"ep {ep} step {step} train loss {np.mean(run[-ACCUM * 5:]):.4f} scale range [{scales.min():.3f}, {scales.max():.3f}]", flush=True)
after = holdout()
print(f"held-out answer loss after: {after:.4f} (before {before:.4f}, delta {after - before:+.4f})", flush=True)
s = scales.detach().float().cpu()
json.dump({"ckpt": CKPT, "layers": layers, "scales": s.tolist(), "holdout_before": before, "holdout_after": after,
           "n_train": N_TRAIN, "lr": LR, "epochs": EPOCHS}, open(os.path.join(OUT, "head_scales.json"), "w"), indent=1)
flat = s.flatten()
top = flat.argsort()
print("most boosted heads (layer, head, scale):", [(layers[int(t) // s.shape[1]], int(t) % s.shape[1], round(float(flat[t]), 3)) for t in top[-6:].flip(0)])
print("most damped heads (layer, head, scale):", [(layers[int(t) // s.shape[1]], int(t) % s.shape[1], round(float(flat[t]), 3)) for t in top[:6]])
print("=== DONE ===")
