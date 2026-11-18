"""Campaign slotcheck, arm C (check 1, the feasibility gate): in the multimodal finetune sets, how many rows lose
their answer page to the collator's `topk_docs=4` cap? CPU only, no model weights.

The finetune collator keeps `docs[:topk_docs]` (pisco/mm_collator.py, `example_chunks`), while the builder puts
gold + distractor pages in a shuffled order (scripts/build_mm_finetune.py) and the teacher label was written from up
to 8 pages (scripts/relabel_mm_teacher.py, --topk_docs 8). A row is
  - "dropped": EVERY gold position >= topk (the first gold page sits at index >= topk): the student never sees the
    answer page, yet is trained on the teacher's answer;
  - "partial": some gold position >= topk, at least one < topk (one answer page kept);
  - "kept": every gold position < topk;
  - "nogold": empty gold_positions (text_rag rows by construction) -> reported apart.
Counts come from the record (pyarrow, no image decode). Controls (the law's feasibility gate):
  (1) every image-source row carries docs, doc_images, source and a non-empty gold_positions: >= 95%;
  (2) on N sampled rows the committed collator's `example_chunks` (training kwargs) returns exactly
      min(len(docs), topk) "Document j:" blocks and as many image chunks; a PLANTED 6-doc row with golds at 4 and 5
      is counted "dropped" and also yields 4 blocks; planted negatives (golds [0, 5] -> partial, [1] -> kept).
Health: rows read = the set's row count; per-source counts sum to the total.

  python scripts/slotcheck/audit_ft_rows.py --out expQ/slotcheck/audit_ft_<job> [--n_control 50]
"""
import argparse, json, os, random, re, sys, time
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
R = f"{ROOT}/expQ/ft_mm_4b_184778/model"
PX = 1003520
# the reference finetune's data spec (expQ/ft_mm_4b_184778/training_config.yaml): "path:N" = first N rows
SETS = {
    "docpages_20k_1m_teacher": (f"{ROOT}/data/mm_ft_docpages_20k_1m_teacher", 20000),
    "natural_8k_teacher": (f"{ROOT}/data/mm_ft_natural_8k_teacher", 8000),
}
TOPK = 4          # collator_kwargs.topk_docs of the reference finetune
TRAIN_KW = dict(compressor_max_length=256, decoder_max_length=1024, topk_docs=TOPK, image_compr_rate=None,
                pad_to_multiple_of=64)   # training_config.yaml collator_kwargs + the class default padding
DOC_RE = re.compile(r"Document (\d+):")


def gold_status(n_docs, gold, topk=TOPK):
    """The ONE classification used for the counts and for the planted controls."""
    gold = [int(g) for g in (gold or [])]
    if not gold:
        return "nogold"
    if any(g < 0 or g >= n_docs for g in gold):
        return "badgold"
    if all(g >= topk for g in gold):
        return "dropped"
    if any(g >= topk for g in gold):
        return "partial"
    return "kept"


def load(path):
    import datasets
    d = datasets.load_from_disk(path)
    return d["train"] if isinstance(d, datasets.DatasetDict) else d


def image_counts(ds):
    """Non-null images per row, read from the arrow storage without decoding a single image."""
    import numpy as np
    import pyarrow.compute as pc
    if getattr(ds, "_indices", None) is not None:
        ds = ds.flatten_indices()
    col = ds.data.table.column("doc_images")
    out = []
    for ch in col.chunks:
        offs = ch.offsets.to_numpy(zero_copy_only=False)
        flat = ch.values
        isnull = pc.or_(flat.is_null(), pc.and_(pc.is_null(flat.field("bytes")), pc.is_null(flat.field("path"))))
        good = (~isnull.to_numpy(zero_copy_only=False)).astype(np.int64)
        cs = np.concatenate([[0], np.cumsum(good)])
        valid = ch.is_valid().to_numpy(zero_copy_only=False)
        for i in range(len(ch)):
            out.append(int(cs[offs[i + 1]] - cs[offs[i]]) if valid[i] else 0)
    return out


def count_set(name, path, cap):
    ds = load(path)
    n = len(ds)
    small = ds.select_columns(["docs", "gold_positions", "source", "query_id", "mistral_label", "gold_label"])
    t0 = time.time()
    nimg = image_counts(ds)
    assert len(nimg) == n, (len(nimg), n)
    per = defaultdict(lambda: defaultdict(int))
    hist_docs = defaultdict(Counter)
    hist_gold = defaultdict(Counter)
    examples = defaultdict(list)
    complete_img = defaultdict(int)
    i = 0
    for batch in small.iter(batch_size=2000):
        for docs, gold, src, qid, ml, gl in zip(batch["docs"], batch["gold_positions"], batch["source"],
                                                batch["query_id"], batch["mistral_label"], batch["gold_label"]):
            nd, ni = len(docs or []), nimg[i]
            kind = "image" if ni > 0 else "text"
            st = gold_status(nd, gold)
            for scope in (("all",) + (("trained",) if i < cap else ())):
                key = (scope, src or "<none>")
                c = per[key]
                c["rows"] += 1
                c[f"kind_{kind}"] += 1
                c["more_than_topk"] += int(nd > TOPK)
                c[st] += 1
                c["img_ne_docs"] += int(ni not in (0, nd))
                if st == "dropped":
                    c["dropped_teacher_eq_gold"] += int((ml or "").strip().lower() == (gl or "").strip().lower())
                if scope == "all":
                    hist_docs[src][nd] += 1
                    for g in gold or []:
                        hist_gold[src][int(g)] += 1
                    if kind == "image" and docs is not None and src and gold:
                        complete_img[src] += 1
                    if st == "dropped" and len(examples[src]) < 5:
                        examples[src].append({"row": i, "query_id": qid, "n_docs": nd, "gold": list(gold)})
            i += 1
    assert i == n, (i, n)
    tables = {}
    for (scope, src), c in sorted(per.items()):
        tables.setdefault(scope, {})[src] = dict(c)
    for scope, t in tables.items():
        tot = defaultdict(int)
        for c in t.values():
            for k, v in c.items():
                tot[k] += v
        t["TOTAL"] = dict(tot)
    img_rows = sum(c.get("kind_image", 0) for k, c in tables["all"].items() if k != "TOTAL")
    return {
        "set": name, "path": path, "rows_in_set": n, "trained_cap": cap, "rows_read": i, "secs": round(time.time() - t0, 1),
        "tables": tables, "n_docs_hist": {s: dict(sorted(h.items())) for s, h in hist_docs.items()},
        "gold_pos_hist": {s: dict(sorted(h.items())) for s, h in hist_gold.items()},
        "image_rows_complete": sum(complete_img.values()), "image_rows": img_rows,
        "dropped_examples": dict(examples),
    }


def controls(n_control, seed):
    from pisco.model import PISCO, PISCOConfig
    from pisco.mm_utils import load_image_processor
    from pisco.mm_collator import MultimodalFineTuningCollator
    cfg = PISCOConfig.from_pretrained(R)
    cfg.compressor_model_name = f"{R}/compressor"
    cfg.decoder_model_name = R
    ctok = PISCO.create_compressor_tokenizer(None, cfg)
    dtok = PISCO.create_decoder_tokenizer(None, cfg)
    ip = load_image_processor(f"{R}/compressor", min_pixels=cfg.image_min_pixels, max_pixels=PX)
    coll = MultimodalFineTuningCollator(compressor_tokenizer=ctok, decoder_tokenizer=dtok, compr_rate=cfg.compr_rate,
                                        image_processor=ip, **TRAIN_KW)
    assert coll.topk_docs == TOPK, coll.topk_docs

    def blocks(row):
        b, doc_text, _ = coll.example_chunks(list(row["docs"]), list(row["doc_images"] or []), row["query"])
        return len(DOC_RE.findall(doc_text)), sum(b.is_image), b.total_mems

    ds = load(SETS["docpages_20k_1m_teacher"][0])
    rng = random.Random(seed)
    nd = [len(d) for d in ds.select_columns(["docs"])["docs"]]
    big = [i for i, k in enumerate(nd) if k > TOPK]
    small = [i for i, k in enumerate(nd) if k <= TOPK]
    pick = rng.sample(big, min(len(big), n_control // 2))
    pick += rng.sample(small, min(len(small), n_control - len(pick)))
    rows, ok = [], 0
    for i in pick:
        r = ds[i]
        nb, nim, mems = blocks(r)
        exp = min(len(r["docs"]), TOPK)
        n_img_kept = sum(x is not None for x in (r["doc_images"] or [])[:exp])
        good = nb == exp and nim == n_img_kept
        ok += int(good)
        rows.append({"row": i, "n_docs": len(r["docs"]), "blocks": nb, "image_chunks": nim, "expected": exp,
                     "mems": mems, "ok": good})
    # planted rows, built from a real image row with >= 1 image
    base = ds[pick[0]]
    imgs = [x for x in base["doc_images"] if x is not None]
    six = [imgs[k % len(imgs)] for k in range(6)]
    planted = {}
    for name, gold, want in (("golds_4_5", [4, 5], "dropped"), ("golds_0_5", [0, 5], "partial"),
                             ("gold_1", [1], "kept")):
        row = {"docs": [""] * 6, "doc_images": six, "query": base["query"], "gold_positions": gold}
        st = gold_status(6, gold)
        nb, nim, _ = blocks(row)
        planted[name] = {"status": st, "want": want, "blocks": nb, "image_chunks": nim,
                         "ok": st == want and nb == TOPK and nim == TOPK}
    return {"sampled": len(rows), "sampled_ok": ok, "sampled_more_than_topk": sum(r["n_docs"] > TOPK for r in rows),
            "rows": rows, "planted": planted, "collator_kwargs": TRAIN_KW, "compr_rate": cfg.compr_rate}


def pct(a, b):
    return f"{100.0 * a / b:.2f}%" if b else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_control", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_controls", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {"topk_docs": TOPK, "sets": {}}
    for name, (path, cap) in SETS.items():
        print(f"== counting {name} ({path}, trained cap {cap})", flush=True)
        res["sets"][name] = count_set(name, path, cap)
    if not a.skip_controls:
        print("== controls: committed collator on sampled + planted rows", flush=True)
        res["controls"] = controls(a.n_control, a.seed)
    # gate inputs
    dp = res["sets"]["docpages_20k_1m_teacher"]
    g1 = dp["image_rows_complete"] / max(1, dp["image_rows"])
    health = {}
    for name, s in res["sets"].items():
        t = s["tables"]["all"]
        health[name] = {
            "rows_read_eq_set": s["rows_read"] == s["rows_in_set"],
            "sources_sum_to_total": sum(c["rows"] for k, c in t.items() if k != "TOTAL") == s["rows_in_set"],
            "status_sum_to_total": sum(t["TOTAL"].get(k, 0) for k in ("kept", "partial", "dropped", "nogold", "badgold"))
            == s["rows_in_set"],
        }
    gate = {"image_rows_complete_share": round(g1, 4), "gate1_pass": g1 >= 0.95}
    if "controls" in res:
        c = res["controls"]
        gate["gate2_pass"] = c["sampled_ok"] == c["sampled"] and c["sampled"] > 0 and all(p["ok"] for p in c["planted"].values())
    res["gate"], res["health"] = gate, health
    with open(f"{a.out}/audit.json", "w") as f:
        json.dump(res, f, indent=1)
    # human-readable table
    lines = [f"topk_docs={TOPK}; dropped = every gold position >= {TOPK}; partial = some gold >= {TOPK}, one kept"]
    for name, s in res["sets"].items():
        for scope in ("all", "trained"):
            t = s["tables"].get(scope)
            if not t:
                continue
            lines.append(f"\n## {name} [{scope} rows]  (set {s['rows_in_set']} rows, trained = first {s['trained_cap']})")
            lines.append("source | rows | image rows | >4 docs | dropped (all gold >= 4) | partial | kept | nogold | badgold | dropped, teacher==gold")
            for src, c in t.items():
                r = c["rows"]
                lines.append(f"{src} | {r} | {c.get('kind_image', 0)} | {c.get('more_than_topk', 0)} ({pct(c.get('more_than_topk', 0), r)}) | "
                             f"{c.get('dropped', 0)} ({pct(c.get('dropped', 0), r)}) | {c.get('partial', 0)} ({pct(c.get('partial', 0), r)}) | "
                             f"{c.get('kept', 0)} | {c.get('nogold', 0)} | {c.get('badgold', 0)} | {c.get('dropped_teacher_eq_gold', 0)}")
        lines.append(f"n_docs histogram: {json.dumps(s['n_docs_hist'])}")
        lines.append(f"gold position histogram: {json.dumps(s['gold_pos_hist'])}")
    lines.append(f"\ngate: {json.dumps(gate)}")
    lines.append(f"health: {json.dumps(health)}")
    if "controls" in res:
        c = res["controls"]
        lines.append(f"controls: sampled {c['sampled_ok']}/{c['sampled']} ok ({c['sampled_more_than_topk']} with > {TOPK} docs); "
                     f"planted {json.dumps(c['planted'])}")
    txt = "\n".join(lines)
    with open(f"{a.out}/summary.txt", "w") as f:
        f.write(txt + "\n")
    print(txt, flush=True)
    t = dp["tables"]["trained"]["TOTAL"]
    print(f"RESULT arm=C dropped_trained={t.get('dropped', 0)} partial_trained={t.get('partial', 0)} "
          f"rows_trained={t['rows']} gate1={gate['gate1_pass']} gate2={gate.get('gate2_pass')}", flush=True)
    healthy = all(all(h.values()) for h in health.values())
    # exit 3 = the feasibility gate or health failed (outputs are written): an afterok chain must not start
    return 0 if (healthy and gate["gate1_pass"] and gate.get("gate2_pass", True)) else 3


if __name__ == "__main__":
    sys.exit(main())
