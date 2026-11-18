"""Campaign slotcheck, arm P (check 2a): does left padding of the compressor input change the <MEM> vectors?

Training pads every compressor batch to a multiple of 64 (pisco/mm_collator.py, pad_to_multiple_of=64); the eval
passes None (scripts/eval_mm_docqa.py make_collator). The compressor is left padded. Here each page of the documents
eval set is compressed ALONE through the eval's own chunking (MultimodalFineTuningCollator.example_chunks, eval
kwargs) and the eval's compress call, with W extra left pad tokens (attention mask 0) in front, W in --widths.
Per page and width: per-<MEM> cosine to the unpadded vectors (connector output = what the decoder receives), max
absolute difference, relative L2. Controls in the same run:
  repeat: the same unpadded input compressed twice (the noise floor; a deterministic forward gives 1.0 / 0);
  other:  page k vs page k+1 (first min(n) <MEM>s), the planted difference: median cosine must fall below .99.
Also "batch4" (reported, not gating): the 4 pages of a question compressed together as the eval does (padding to the
longest page) vs the same batch padded to a multiple of 64 as training does.
Health: <MEM> count per page identical across widths.

  python scripts/slotcheck/pad_probe.py --out expQ/slotcheck/pad_probe_<job> [--n_pages 120] [--widths 0,17,63]
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
    ap.add_argument("--n_pages", type=int, default=120)
    ap.add_argument("--n_batch4", type=int, default=30)
    ap.add_argument("--widths", default="0,17,63")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--cml", type=int, default=256)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    widths = [int(w) for w in a.widths.split(",")]
    assert widths[0] == 0, "first width must be 0 (the unpadded reference)"

    import torch, datasets
    from pisco.model import PISCO
    from pisco.mm_utils import collate_compressor
    from eval_mm_docqa import make_collator
    dev = "cuda"
    t0 = time.time()
    model = PISCO.from_pretrained(a.checkpoint)
    model.to(dev).eval()
    print(f"[pad] model loaded in {time.time() - t0:.0f}s; image_max_pixels={model.config.image_max_pixels}; "
          f"mem_layout={getattr(model.config, 'compressor_mem_layout', None)}", flush=True)
    coll = make_collator(model, topk=a.topk, compressor_max_length=a.cml, image_compr_rate=None, query_dependent=False)
    assert coll.pad_to_multiple_of is None
    pad_id = model.compressor_tokenizer.pad_token_id
    image_pad_id = coll.vision_ids[1]
    d = datasets.load_from_disk(a.data)
    ds = d["train"] if isinstance(d, datasets.DatasetDict) else d

    pages = []   # (row, j, image, query)
    for i in range(len(ds)):
        r = ds[i]
        for j, img in enumerate((r.get("doc_images") or [])[: a.topk]):
            if img is not None:
                pages.append((i, j, img, r["query"]))
        if len(pages) >= a.n_pages + 1:
            break
    pages = pages[: a.n_pages + 1]   # +1: page k+1 is page k's planted "other"

    @torch.no_grad()
    def run(comp, pad):
        ids, mask = comp["compressor_input_ids"], comp["compressor_attention_mask"]
        if pad:
            ids = torch.cat([torch.full((ids.shape[0], pad), pad_id, dtype=ids.dtype), ids], 1)
            mask = torch.cat([torch.zeros((mask.shape[0], pad), dtype=mask.dtype), mask], 1)
        mm = (ids == image_pad_id).long()
        e = model.compress(ids.to(dev), mask.to(dev), pixel_values=comp["pixel_values"].to(dev),
                           image_grid_thw=comp["image_grid_thw"].to(dev), mm_token_type_ids=mm.to(dev))
        return e.float().cpu()

    def compare(x, y):
        n = min(x.shape[0], y.shape[0])
        x, y = x[:n], y[:n]
        cos = torch.nn.functional.cosine_similarity(x, y, dim=-1)
        return {"n": n, "cos_med": float(cos.median()), "cos_min": float(cos.min()),
                "maxabs": float((x - y).abs().max()), "rel_l2": float((x - y).norm() / x.norm().clamp_min(1e-12))}

    embs, per = [], []
    for k, (i, j, img, q) in enumerate(pages):
        b, _, _ = coll.example_chunks([""], [img], q)
        comp = collate_compressor([b], coll, coll.vision_ids)
        e0 = run(comp, 0)
        embs.append(e0)
        if k >= a.n_pages:
            break
        rec = {"row": i, "doc": j, "tokens": int(comp["compressor_input_ids"].shape[1]), "mems": {0: int(e0.shape[0])},
               "repeat": compare(e0, run(comp, 0))}
        for w in widths[1:]:
            ew = run(comp, w)
            rec["mems"][w] = int(ew.shape[0])
            rec[f"w{w}"] = compare(e0, ew)
        per.append(rec)
        if k % 20 == 0:
            print(f"[pad] page {k}: tokens {rec['tokens']} mems {rec['mems']} repeat {rec['repeat']} "
                  + " ".join(f"w{w} {rec[f'w{w}']}" for w in widths[1:]), flush=True)
    per = [rec for k, rec in enumerate(per) if k + 1 < len(embs)]
    for k, rec in enumerate(per):
        rec["other"] = compare(embs[k], embs[k + 1])

    # batch4: the eval's 4-page call (pad to longest) vs training's pad to a multiple of 64
    b4 = []
    for i in range(min(a.n_batch4, len(ds))):
        r = ds[i]
        b, _, _ = coll.example_chunks(list(r["docs"]), list(r.get("doc_images") or []), r["query"])
        if not b.input_ids:
            continue
        comp = collate_compressor([b], coll, coll.vision_ids)
        L = comp["compressor_input_ids"].shape[1]
        extra = (-L) % 64
        e0 = run(comp, 0)
        e64 = run(comp, extra)
        b4.append({"row": i, "chunks": len(b.input_ids), "longest": L, "extra_pad": extra, **compare(e0, e64)})

    def summ(key):
        xs = [p[key] for p in per]
        return {"pages": len(xs), "cos_med_median": statistics.median(x["cos_med"] for x in xs),
                "cos_min_min": min(x["cos_min"] for x in xs), "maxabs_max": max(x["maxabs"] for x in xs),
                "maxabs_median": statistics.median(x["maxabs"] for x in xs),
                "rel_l2_median": statistics.median(x["rel_l2"] for x in xs)}

    S = {k: summ(k) for k in ["repeat", "other"] + [f"w{w}" for w in widths[1:]]}
    floor = S["repeat"]["maxabs_max"]
    mems_same = all(len(set(p["mems"].values())) == 1 for p in per)
    controls_ok = S["repeat"]["cos_med_median"] >= 0.9999 and S["other"]["cos_med_median"] < 0.99
    inert = all(S[f"w{w}"]["cos_med_median"] >= 0.9999 and S[f"w{w}"]["maxabs_max"] <= 2 * floor for w in widths[1:])
    if b4:
        S["batch4"] = {"questions": len(b4), "cos_med_median": statistics.median(x["cos_med"] for x in b4),
                       "cos_min_min": min(x["cos_min"] for x in b4), "maxabs_max": max(x["maxabs"] for x in b4),
                       "extra_pad_median": statistics.median(x["extra_pad"] for x in b4)}
    res = {"checkpoint": a.checkpoint, "data": a.data, "widths": widths, "summary": S, "repeat_floor_maxabs": floor,
           "health_mems_same": mems_same, "controls_ok": controls_ok, "inert": inert and controls_ok and mems_same,
           "pages": per, "batch4": b4}
    with open(f"{a.out}/pad_probe.json", "w") as f:
        json.dump(res, f, indent=1)
    for k, v in S.items():
        print(f"SUMMARY {k}: {json.dumps(v)}", flush=True)
    print(f"RESULT arm=P2a pages={len(per)} controls_ok={controls_ok} mems_same={mems_same} floor_maxabs={floor:.3g} "
          f"inert={res['inert']} " + " ".join(f"w{w}_cos={S[f'w{w}']['cos_med_median']:.6f} w{w}_maxabs={S[f'w{w}']['maxabs_max']:.3g}"
                                              for w in widths[1:]), flush=True)
    return 0 if (controls_ok and mems_same) else 3


if __name__ == "__main__":
    sys.exit(main())
