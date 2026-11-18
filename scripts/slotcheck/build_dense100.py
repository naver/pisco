"""Campaign slotcheck, arm O (check 4): 100 dense document pages to overfit on + 100 held-out dense pages.

Source: data/mm_pretrain_pdfa_1m (page image + its text layer; the stage-2 image-AE format). Pages are taken from the
TAIL of the set (--scan rows), never a page whose docid is among the rows the stage-2 start model trained on
(the first --exclude_n rows of data/mm_pretrain_pdfa_1m_desc, expQ/archq3/pt_mm_5M_spread/training_config.yaml).
"Dense" = the most text tokens (compressor tokenizer) at or under --max_tokens, so the stage-2 AE target
(text[:ae_target_max_tokens = 768]) is the WHOLE page text; one page per docid. The 2 x --n picked pages are shuffled
(--seed) and split: first half = overfit set, second half = held-out set (same density by construction).

Writes (names fixed by the law, CAMPAIGN.md arm O):
  data/slotcheck_dense100<tag>_train      pretrain format (image, text, title, docid, source): the overfit training set
  eval_data/slotcheck_dense100<tag>       eval format for scripts/longdoc/mem_recall.py (doc_images=[page], gold [0])
  eval_data/slotcheck_dense100<tag>_heldout   the same for the held-out pages
  <out>/summary.json                      counts, token / word distribution, docids
  <out>/textlayer_{train,heldout}.json    each page's own text layer as a mem_recall record (the recall ceiling)
  The training set repeats the 100 pages --copies times (10 -> 1,000 rows; train.py's eval split takes 64 copies).
  python scripts/slotcheck/build_dense100.py --out expQ/slotcheck/dense100_<job> [--scan 60000] [--tag _smoke]
"""
import argparse, json, os, random, statistics, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=f"{ROOT}/data/mm_pretrain_pdfa_1m")
    ap.add_argument("--exclude", default=f"{ROOT}/data/mm_pretrain_pdfa_1m_desc")
    ap.add_argument("--exclude_n", type=int, default=80000)
    ap.add_argument("--scan", type=int, default=60000, help="rows read from the tail of --src")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_tokens", type=int, default=760, help="< ae_target_max_tokens 768: the target is the whole text")
    ap.add_argument("--min_tokens", type=int, default=300)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix of the written set names (a smoke writes _smoke)")
    ap.add_argument("--copies", type=int, default=10,
                    help="the training set holds each overfit page this many times: pisco/train.py moves "
                         "min(64, rows // 10) random rows to its eval split, which on 100 rows would drop 10 pages")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import datasets
    from datasets import Features, Image, Sequence, Value
    from transformers import AutoTokenizer
    t0 = time.time()

    def load(p):
        d = datasets.load_from_disk(p)
        return d["train"] if isinstance(d, datasets.DatasetDict) else d

    src, exc = load(a.src), load(a.exclude)
    trained_docids = set(exc.select(range(min(a.exclude_n, len(exc))))["docid"])
    lo = max(0, len(src) - a.scan)
    meta = src.select(range(lo, len(src))).select_columns(["text", "docid", "source"])
    texts, docids = meta["text"], meta["docid"]
    print(f"[dense] src rows {len(src)}, scanning {lo}..{len(src)}; trained docids excluded {len(trained_docids)} "
          f"({time.time() - t0:.0f}s)", flush=True)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    # tokens >= words on this tokenizer, so a page over max_tokens words cannot fit: drop it unread, tokenize the
    # rest in one batch (the densest pages by WORDS mostly exceed the token cap, so a top-k by words is not enough)
    cand = [k for k, t in enumerate(texts) if t and docids[k] not in trained_docids and len(t.split()) <= a.max_tokens]
    n_excl = sum(1 for d in docids if d in trained_docids)
    ntoks = [len(x) for x in tok([texts[k] for k in cand], add_special_tokens=False)["input_ids"]]
    n_tok = len(cand)
    pool = sorted(((n, len(texts[k].split()), k) for n, k in zip(ntoks, cand) if a.min_tokens <= n <= a.max_tokens),
                  reverse=True)
    picked, seen = [], set()
    for n, w, k in pool:
        if docids[k] in seen:
            continue
        picked.append((n, w, k)); seen.add(docids[k])
        if len(picked) == 2 * a.n:
            break
    assert len(picked) == 2 * a.n, f"only {len(picked)} pages in [{a.min_tokens}, {a.max_tokens}] tokens: raise --scan"
    rng = random.Random(a.seed)
    rng.shuffle(picked)
    halves = {"train": picked[: a.n], "heldout": picked[a.n:]}
    eval_feat = Features({"query_id": Value("string"), "source": Value("string"), "query": Value("string"),
                          "answers": Sequence(Value("string")), "docs": Sequence(Value("string")),
                          "doc_images": Sequence(Image()), "gold_positions": Sequence(Value("int64")),
                          "n_tokens": Value("int64"), "n_words": Value("int64")})
    summary = {"copies": a.copies, "src": a.src, "rows_src": len(src), "scan_from": lo, "rows_scanned": len(texts),
               "scanned_rows_excluded_as_trained": n_excl, "candidates": len(cand), "tokenized": n_tok,
               "max_tokens": a.max_tokens, "min_tokens": a.min_tokens, "seed": a.seed, "sets": {}}
    for name, rows in halves.items():
        idx = [lo + k for _, _, k in rows]
        sel = src.select(idx)
        assert not (set(sel["docid"]) & trained_docids)
        raw = sel.cast_column("image", Image(decode=False))["image"]   # original bytes: no JPEG re-encode
        imgs = [{"bytes": r["bytes"], "path": None} for r in raw]
        ev = datasets.Dataset.from_dict({
            "query_id": [f"dense_{d}" for d in sel["docid"]], "source": ["pdfa_dense"] * len(idx),
            "query": ["Transcribe the page."] * len(idx), "answers": [[] for _ in idx], "docs": [[""] for _ in idx],
            "doc_images": [[im] for im in imgs], "gold_positions": [[0] for _ in idx],
            "n_tokens": [n for n, _, _ in rows], "n_words": [w for _, w, _ in rows]}, features=eval_feat)
        ev_path = f"{ROOT}/eval_data/slotcheck_dense100{a.tag}" + ("" if name == "train" else "_heldout")
        ev.save_to_disk(ev_path)
        if name == "train":
            datasets.concatenate_datasets([sel] * a.copies).save_to_disk(f"{ROOT}/data/slotcheck_dense100{a.tag}_train")
        # ceiling control for the recall read: the page's own text layer (= the AE training target, <= 760 tokens)
        # as a "rebuild" in scripts/longdoc/mem_recall.py's record format, paired with recall_pair.py like any arm
        recs = [{"query_id": f"dense_{d}", "source": "pdfa_dense", "query": "Transcribe the page.", "answers": [],
                 "gold_page": 0, "text": t} for d, t in zip(sel["docid"], sel["text"])]
        with open(f"{a.out}/textlayer_{name}.json", "w") as f:
            json.dump({"args": {"arm": "textlayer", "data_path": ev_path}, "records": recs}, f, ensure_ascii=False)
        toks = [n for n, _, _ in rows]; words = [w for _, w, _ in rows]
        import io
        from PIL import Image as PILImage
        sizes = [PILImage.open(io.BytesIO(im["bytes"])).size for im in imgs]
        summary["sets"][name] = {"rows": len(idx), "eval_path": ev_path, "tokens_mean": statistics.mean(toks),
                                 "tokens_min": min(toks), "tokens_max": max(toks), "words_mean": statistics.mean(words),
                                 "px_mean": statistics.mean(w * h for w, h in sizes), "docids": list(sel["docid"])}
        print(f"[dense] {name}: {len(idx)} pages, tokens {min(toks)}-{max(toks)} (mean {statistics.mean(toks):.0f}), "
              f"words mean {statistics.mean(words):.0f}, px mean {summary['sets'][name]['px_mean']:.0f} -> {ev_path}",
              flush=True)
    overlap = set(summary["sets"]["train"]["docids"]) & set(summary["sets"]["heldout"]["docids"])
    summary["train_heldout_overlap"] = len(overlap)
    with open(f"{a.out}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    ok = not overlap and all(s["rows"] == a.n for s in summary["sets"].values())
    print(f"RESULT arm=O_build pages_train={summary['sets']['train']['rows']} pages_heldout={summary['sets']['heldout']['rows']} "
          f"overlap={len(overlap)} tokens_mean_train={summary['sets']['train']['tokens_mean']:.0f} "
          f"tokens_mean_heldout={summary['sets']['heldout']['tokens_mean']:.0f} excluded_trained={n_excl} "
          f"ok={ok} ({time.time() - t0:.0f}s)", flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
