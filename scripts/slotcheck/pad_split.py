"""Campaign slotcheck, arm P (check 2a, split): WHICH padding path moves the <MEM> vectors, and is it rounding?

The 120-page probe (pad_probe.py, job 191672) found left pads move a page's <MEM>s on two paths: a 1-row batch
(median cos .934) and a 4-page batch padded to 64 (median .997, min .64). The installed transformers zeroes pad
positions in Qwen3.5's linear-attention layers only when the batch has > 1 row (apply_mask_to_padding_states,
modeling_qwen3_5.py:198), and passes no mask at all when every position is real (_update_linear_attn_mask). This
probe separates the paths, page by page (each page compressed through the eval's own example_chunks + compress):
  ref     1 row, no pad                                    (the eval's single-page call)
  repeat  ref again                                        (bit floor; 0 by construction)
  twin0   2 rows [page, page], no pad (no mask reaches the linear layers)   (batch-size rounding floor)
  A{W}    1 row, W left pads, as installed (pads NOT zeroed in the linear layers: the leak)
  Afix{W} 1 row, W left pads, with the zeroing forced on for 1 row (patched in THIS process only)   (cause test)
  B{W}    2 rows [page, page], both W left pads (pads zeroed: the masked path)
Every condition is compared with ref (per <MEM>: cosine, max abs, rel L2). Then the same in fp32 (compressor +
connector cast to float32): if the masked path is rounding, B{W} and Afix{W} reach cos ~1 in fp32 while A{W} does not.
Reading (fixed before the run): Afix{W} ~ B{W} and A{W} far below both -> the 1-row leak is the HF condition;
B{W} at the twin0 floor in fp32 -> the masked path is rounding only.

  python scripts/slotcheck/pad_split.py --out expQ/slotcheck/pad_split_<job> [--n_pages 40] [--widths 17,63] [--fp32]
"""
import argparse, json, os, statistics, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
R42 = f"{ROOT}/expQ/ft_mm_4b_184778/model"
DATA = f"{ROOT}/eval_data/mm_docqa_1500_1m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=R42)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--n_pages", type=int, default=40)
    ap.add_argument("--widths", default="17,63")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--cml", type=int, default=256)
    ap.add_argument("--fp32", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    widths = [int(w) for w in a.widths.split(",")]

    import torch, datasets
    import transformers.models.qwen3_5.modeling_qwen3_5 as Q
    from pisco.model import PISCO
    from pisco.mm_utils import collate_compressor
    from eval_mm_docqa import make_collator

    orig_mask = Q.apply_mask_to_padding_states
    FORCE = {"on": False}

    def forced_mask(hidden_states, attention_mask):   # the installed rule, minus the "> 1 row" condition
        if FORCE["on"] and attention_mask is not None and attention_mask.shape[1] > 1:
            return (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)
        return orig_mask(hidden_states, attention_mask)
    Q.apply_mask_to_padding_states = forced_mask      # module global: GatedDeltaNet.forward looks it up per call

    dev = "cuda"
    t0 = time.time()
    model = PISCO.from_pretrained(a.checkpoint)
    model.to(dev).eval()
    print(f"[split] model loaded in {time.time() - t0:.0f}s; image_max_pixels={model.config.image_max_pixels}; "
          f"mem_layout={model.config.compressor_mem_layout}; attn={getattr(model.compressor.config, '_attn_implementation', None)}", flush=True)
    coll = make_collator(model, topk=a.topk, compressor_max_length=a.cml, image_compr_rate=None, query_dependent=False)
    pad_id = model.compressor_tokenizer.pad_token_id
    image_pad_id = coll.vision_ids[1]
    d = datasets.load_from_disk(a.data)
    ds = d["train"] if isinstance(d, datasets.DatasetDict) else d
    pages = []
    for i in range(len(ds)):
        r = ds[i]
        for j, img in enumerate((r.get("doc_images") or [])[: a.topk]):
            if img is not None:
                pages.append((i, j, img, r["query"]))
        if len(pages) >= a.n_pages:
            break
    pages = pages[: a.n_pages]
    comps = []
    for (i, j, img, q) in pages:
        b, _, _ = coll.example_chunks([""], [img], q)
        comps.append(collate_compressor([b], coll, coll.vision_ids))

    @torch.no_grad()
    def run(comp, pad, rows=1, force=False):
        ids, mask = comp["compressor_input_ids"], comp["compressor_attention_mask"]
        assert ids.shape[0] == 1 and bool(mask.all()), "one unpadded page per comp"
        if pad:
            ids = torch.cat([torch.full((1, pad), pad_id, dtype=ids.dtype), ids], 1)
            mask = torch.cat([torch.zeros((1, pad), dtype=mask.dtype), mask], 1)
        pv, grid = comp["pixel_values"], comp["image_grid_thw"]
        if rows > 1:   # the same page repeated: rows of input_ids, image_grid_thw and pixel_values stay in order
            ids, mask = ids.repeat(rows, 1), mask.repeat(rows, 1)
            pv, grid = pv.repeat(rows, 1), grid.repeat(rows, 1)
        mm = (ids == image_pad_id).long()
        FORCE["on"] = force
        try:
            e = model.compress(ids.to(dev), mask.to(dev), pixel_values=pv.to(dev), image_grid_thw=grid.to(dev),
                               mm_token_type_ids=mm.to(dev))
        finally:
            FORCE["on"] = False
        e = e.float().cpu()
        n = e.shape[0] // rows
        return e[:n]   # row 0's <MEM>s (rows come back in batch order)

    def compare(x, y):
        assert x.shape == y.shape, (x.shape, y.shape)
        cos = torch.nn.functional.cosine_similarity(x, y, dim=-1)
        return {"n": int(x.shape[0]), "cos_med": float(cos.median()), "cos_min": float(cos.min()),
                "maxabs": float((x - y).abs().max()), "rel_l2": float((x - y).norm() / x.norm().clamp_min(1e-12))}

    conds = ["repeat", "twin0"] + [f"{c}{w}" for w in widths for c in ("A", "Afix", "B")]

    def one_pass(tag):
        per = []
        t = time.time()
        for k, comp in enumerate(comps):
            ref = run(comp, 0)
            rec = {"page": k, "row": pages[k][0], "doc": pages[k][1], "tokens": int(comp["compressor_input_ids"].shape[1]),
                   "mems": int(ref.shape[0]), "repeat": compare(ref, run(comp, 0)), "twin0": compare(ref, run(comp, 0, rows=2))}
            for w in widths:
                rec[f"A{w}"] = compare(ref, run(comp, w))
                rec[f"Afix{w}"] = compare(ref, run(comp, w, force=True))
                rec[f"B{w}"] = compare(ref, run(comp, w, rows=2))
            per.append(rec)
            if k % 10 == 0:
                print(f"[split {tag}] page {k}: tokens {rec['tokens']} mems {rec['mems']} "
                      + " ".join(f"{c} {rec[c]['cos_med']:.5f}/{rec[c]['maxabs']:.3g}" for c in conds), flush=True)
        S = {}
        for c in conds:
            xs = [p[c] for p in per]
            S[c] = {"pages": len(xs), "cos_med_median": statistics.median(x["cos_med"] for x in xs),
                    "cos_min_min": min(x["cos_min"] for x in xs), "maxabs_max": max(x["maxabs"] for x in xs),
                    "maxabs_median": statistics.median(x["maxabs"] for x in xs),
                    "rel_l2_median": statistics.median(x["rel_l2"] for x in xs)}
            print(f"SUMMARY {tag} {c}: {json.dumps(S[c])}", flush=True)
        print(f"[split {tag}] {len(per)} pages in {time.time() - t:.0f}s", flush=True)
        return per, S

    res = {"checkpoint": a.checkpoint, "data": a.data, "widths": widths, "passes": {}}
    per, S = one_pass("bf16")
    res["passes"]["bf16"] = {"summary": S, "pages": per}
    if a.fp32:
        try:
            model.compressor.float(); model.connector.float()
            per32, S32 = one_pass("fp32")
            res["passes"]["fp32"] = {"summary": S32, "pages": per32}
        except Exception as ex:   # the bf16 read stands without it
            print(f"[split fp32] FAILED: {type(ex).__name__}: {ex}", flush=True)
            res["passes"]["fp32"] = {"error": f"{type(ex).__name__}: {ex}"}
    health = all(p["repeat"]["maxabs"] == 0.0 for p in per)
    res["health_repeat_exact"] = health
    with open(f"{a.out}/pad_split.json", "w") as f:
        json.dump(res, f, indent=1)

    def g(tag, c, k="cos_med_median"):
        s = res["passes"].get(tag, {}).get("summary")
        return f"{s[c][k]:.6f}" if s else "na"
    print("RESULT arm=P2a_split pages=%d repeat_exact=%s " % (len(per), health)
          + " ".join(f"{tag}_{c}_cos={g(tag, c)}" for tag in res["passes"] for c in ["twin0"] + [f"{x}{widths[-1]}" for x in ("A", "Afix", "B")]),
          flush=True)
    return 0 if health else 3


if __name__ == "__main__":
    sys.exit(main())
