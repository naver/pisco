"""Routing probe (no training): can the decoder's own attention pick the page that holds the answer?
(docs/literature_scan_2026-11.md rank 4, AttentionRetriever 2602.12278.) For each question the all-<MEM> QA prompt is
run once with eager attention; at the last prompt position (the one that predicts the first answer token) the
attention mass of every head of every full-attention layer is summed over each document's <MEM>s. Picks compared,
pick@1 against `gold_positions`:
  digit    the stage-2 `find` route prompt (eval_mm_docqa.route_logprobs), the current router
  attn_*   attention mass per document: mean over all heads and layers; per layer; mean per <MEM> (length-normalised)
  heads    the best K heads chosen on one half of the questions, evaluated on the other half (and swapped)
Per-question head x document masses go to <OUT>/masses.npz for further analysis. Env: CKPT, DATA, N_ROWS, TOPK, PX, OUT.
"""
import json
import os
import pathlib
import re
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import datasets  # noqa: E402

import eval_mm_docqa as E  # noqa: E402
from pisco.model import PISCO  # noqa: E402
from pisco.mm_utils import collate_compressor  # noqa: E402

CKPT = os.environ.get("CKPT", "expQ/ft_mm_4b_183814/model")
DATA = os.environ["DATA"]
N = int(os.environ.get("N_ROWS", "0"))
TOPK = int(os.environ.get("TOPK", "10"))
PX = int(os.environ.get("PX", "1003520"))
OUT = os.environ["OUT"]
dev = torch.device("cuda")
os.makedirs(OUT, exist_ok=True)

model = PISCO.from_pretrained(CKPT, **({'image_max_pixels': PX} if PX else {})).to(dev).eval()   # same override path as the eval
coll = E.make_collator(model, topk=TOPK, compressor_max_length=256, image_compr_rate=None, query_dependent=False)
dtok = model.decoder_tokenizer
mem_tok = dtok.mem_token

# eager attention on the decoder so that the attention weights exist; hooks on every full-attention module
try:
    model.decoder.set_attn_implementation("eager")
except Exception:
    for m in model.decoder.modules():
        c = getattr(m, "config", None)
        if c is not None and hasattr(c, "_attn_implementation"):
            c._attn_implementation = "eager"
attn_mods = [m for m in model.decoder.modules() if type(m).__name__.endswith("Attention") and hasattr(m, "o_proj")]
captured = []


def grab(_m, _inp, out):
    w = out[1]
    if w is not None:
        captured.append(w[0, :, -1, :].float().cpu())   # (H, L): last query position over all keys


for m in attn_mods:
    m.register_forward_hook(grab)

ds = datasets.load_from_disk(DATA)
ds = ds["train"] if "train" in getattr(ds, "keys", lambda: [])() else ds
idx = list(range(len(ds))) if not N else list(np.random.RandomState(0).permutation(len(ds))[:N])
rows_out, masses, meta = [], [], []
with torch.inference_mode():
    for t, i in enumerate(idx):
        row = ds[int(i)]
        docs = [d if d else " " for d in row["docs"]][:TOPK]
        imgs = list(row.get("doc_images") or [])[:TOPK]
        gold = {int(g) for g in (row.get("gold_positions") or []) if int(g) < len(docs)}
        if not gold:
            continue
        query = coll.clean_text(row["query"])
        batch, background, _ = coll.example_chunks(docs, imgs, query)
        # <MEM>s per document from the background string ("Document j:" + its <MEM>s)
        parts = re.split(r"Document (\d+):", background)
        per_doc = {}
        for j in range(1, len(parts) - 1, 2):
            per_doc[int(parts[j])] = parts[j + 1].count(mem_tok)
        n_docs = len(docs)
        counts = [per_doc.get(j, 0) for j in range(n_docs)]
        assert sum(counts) == batch.total_mems, (counts, batch.total_mems)
        prompt = E._build_chat_prompt(dtok, background=background, question=query)
        dec = dtok(prompt, return_tensors="pt", add_special_tokens=False)
        ids = dec["input_ids"].to(dev)
        attn = dec["attention_mask"].to(dev)
        mem_pos = (ids[0] == dtok.mem_token_id).nonzero().squeeze(1).cpu()
        assert mem_pos.numel() == sum(counts)
        doc_of_mem = torch.repeat_interleave(torch.arange(n_docs), torch.tensor(counts))
        comp = collate_compressor([batch], coll, coll.vision_ids)
        comp = {k: v.to(dev) for k, v in comp.items()}
        emb = E.compress_sliced(model, comp, batch.is_image)
        embeds = model.replace_embeddings(emb, ids)
        captured.clear()
        E._reset_rope_deltas(model.decoder)
        model.decoder(inputs_embeds=embeds, attention_mask=attn, logits_to_keep=1, **E._pos_kwargs(model, ids, attn))
        assert captured, "no attention weights captured (eager attention not active?)"
        W = torch.stack(captured)                      # (layers, H, L)
        Wm = W[:, :, mem_pos]                          # (layers, H, n_mem)
        mass = torch.zeros(W.shape[0], W.shape[1], n_docs)
        mass.index_add_(2, doc_of_mem, Wm)             # sum over each document's <MEM>s
        lp = E.route_logprobs(model, background, emb, query, n_docs, dev) if n_docs <= 20 else None
        masses.append(mass.numpy().astype(np.float32))
        meta.append({"i": int(i), "n_docs": n_docs, "counts": counts, "gold": sorted(gold),
                     "digit_pick": int(np.argmax(lp)) if lp is not None else None, "source": row.get("source")})
        if (t + 1) % 100 == 0:
            print(f"{t + 1}/{len(idx)}", flush=True)

np.savez_compressed(os.path.join(OUT, "masses.npz"), **{f"q{k}": m for k, m in enumerate(masses)})
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"))


def pick_at1(score_fn):
    hits = [int(np.argmax(score_fn(m, md)) in md["gold"]) for m, md in zip(masses, meta)]
    return float(np.mean(hits)), hits


res = {"ckpt": CKPT, "data": DATA, "n": len(meta), "topk": TOPK}
digit = [md for md in meta if md["digit_pick"] is not None]
res["digit_pick@1"] = float(np.mean([md["digit_pick"] in md["gold"] for md in digit])) if digit else None
res["attn_mean_all_pick@1"], _ = pick_at1(lambda m, md: m.mean((0, 1)))
res["attn_mean_all_per_mem_pick@1"], _ = pick_at1(lambda m, md: m.mean((0, 1)) / np.maximum(np.array(md["counts"]), 1))
res["attn_per_layer_pick@1"] = [pick_at1(lambda m, md, l=l: m[l].mean(0))[0] for l in range(masses[0].shape[0])]
res["attn_last_layer_pick@1"] = res["attn_per_layer_pick@1"][-1]
# head selection with a split: rank heads by pick@1 on half A, use the top-K mean on half B, and swap
nq = len(meta)
half = np.arange(nq) % 2
L, H = masses[0].shape[:2]
head_hits = np.array([[[int(np.argmax(m[l, h]) in md["gold"]) for m, md in zip(masses, meta)] for h in range(H)] for l in range(L)])  # (L,H,nq)
for K in (1, 4, 16):
    accs = []
    for a in (0, 1):
        tr, te = half == a, half != a
        rank = head_hits[:, :, tr].mean(-1).ravel().argsort()[::-1][:K]
        sel = [(r // H, r % H) for r in rank]
        hits = [int(np.argmax(np.mean([m[l, h] for l, h in sel], 0)) in md["gold"]) for m, md, keep in zip(masses, meta, te) if keep]
        accs.append(float(np.mean(hits)))
    res[f"attn_top{K}_heads_pick@1_crossval"] = float(np.mean(accs))
best = head_hits.mean(-1)
res["best_single_head"] = {"layer_idx": int(best.argmax() // H), "head": int(best.argmax() % H), "pick@1_in_sample": float(best.max())}
res["random_pick@1"] = float(np.mean([len(md["gold"]) / md["n_docs"] for md in meta]))
by_src = {}
for s in sorted({md["source"] for md in meta}):
    sel = [k for k, md in enumerate(meta) if md["source"] == s]
    by_src[s] = {"n": len(sel),
                 "digit": float(np.mean([meta[k]["digit_pick"] in meta[k]["gold"] for k in sel if meta[k]["digit_pick"] is not None] or [np.nan])),
                 "attn_mean_all": float(np.mean([int(np.argmax(masses[k].mean((0, 1))) in meta[k]["gold"]) for k in sel]))}
res["by_source"] = by_src
json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
print("\n=== SUMMARY ===")
for k, v in res.items():
    if k != "by_source":
        print(f"{k}: {v}")
for s, v in by_src.items():
    print(f"  {s}: n={v['n']} digit {v['digit']:.3f} attn_mean_all {v['attn_mean_all']:.3f}")
print("=== DONE ===")
