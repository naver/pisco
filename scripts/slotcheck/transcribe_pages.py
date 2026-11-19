"""Campaign slotcheck, arm X (check 3): the uncompressed Qwen3.5-4B transcribes the pages of 500 document questions.

Questions: the first 167 / 167 / 166 rows per source (sources sorted, set order) of eval_data/mm_docqa_1500_1m; pages:
the first 4 (topk_docs = 4, the eval's cut) of each. A page that occurs several times (same pixels) is transcribed once:
key = md5 of its RGB bytes + size. Unique pages, sorted by key, go to shard k % n_shards. Prompt and generation are
TRANSCRIBE + run_ref of scripts/probe_mem_recall.py (imported, never edited; greedy).

Output (resumes on requeue): <out_dir>/shard<i>of<n><tag>.json {"records": [{key, qids, text, visual_tokens,
new_tokens, sec}]}; shard 0 also writes <out_dir>/questions<tag>.json {query_id: [page keys]} for build_text_sets.py.
Exit 4 when more than 2% of the shard's transcriptions are empty (law health) or nothing was written.
"""

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

PER_SOURCE = {"infovqa": 167, "mpdocvqa": 167, "slidevqa": 166}


def page_key(img):
    img = img.convert("RGB")
    return hashlib.md5(img.tobytes()).hexdigest()[:16] + f"_{img.size[0]}x{img.size[1]}"


def pick_questions(ds, topk, round_=1):
    """[(row index, query_id, [PIL image of each kept page])] for the 500 questions. round_ r = the r-th block of
    167 / 167 / 166 rows per source in set order (1 = the gating 500; 2 = the repeat's new, disjoint 500)."""
    seen, skip, out = {}, {}, []
    srcs = ds["source"]
    for i, src in enumerate(srcs):
        if skip.get(src, 0) < (round_ - 1) * PER_SOURCE.get(src, 0):
            skip[src] = skip.get(src, 0) + 1
            continue
        if seen.get(src, 0) >= PER_SOURCE.get(src, 0):
            continue
        row = ds[i]
        imgs = list(row.get("doc_images") or [])[:topk]
        assert imgs and all(x is not None for x in imgs), f"row {i}: a page without an image"
        seen[src] = seen.get(src, 0) + 1
        out.append((i, str(row["query_id"]), imgs))
    assert seen == PER_SOURCE, f"picked {seen}, expected {PER_SOURCE}"
    return out


def atomic_dump(obj, path):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=str(REPO / "eval_data/mm_docqa_1500_1m"))
    ap.add_argument("--out_dir", default=str(REPO / "expQ/slotcheck/transcribe"))
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n_shards", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="plumbing: the first N pages of the shard only")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--base_model_name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--image_max_pixels", type=int, default=1003520)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--round", type=int, default=1, help="question block (pick_questions); use its own --out_dir")
    args = ap.parse_args()

    import datasets
    import torch
    from transformers import AutoTokenizer
    from probe_mem_recall import TRANSCRIBE, run_ref  # noqa: F401  (TRANSCRIBE is the prompt run_ref uses)

    tag = f"_lim{args.limit}" if args.limit else ""
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"shard{args.shard}of{args.n_shards}{tag}.json")
    ds = datasets.load_from_disk(args.data_path)
    ds = ds["train"] if isinstance(ds, datasets.DatasetDict) else ds
    t0 = time.time()
    qs = pick_questions(ds, args.topk, args.round)
    pages, qmap = {}, {}   # key -> {"loc": (row, position) of its first occurrence, "qids"}; images not kept (RAM)
    for i, qid, imgs in qs:
        keys = []
        for j, img in enumerate(imgs):
            k = page_key(img)
            keys.append(k)
            pages.setdefault(k, {"loc": (i, j), "qids": []})["qids"].append(qid)
        qmap[qid] = keys
    n_slots = sum(len(v) for v in qmap.values())
    print(f"[transcribe] {len(qs)} questions, {n_slots} page slots, {len(pages)} unique pages "
          f"({time.time() - t0:.0f}s to hash); prompt: {TRANSCRIBE!r}", flush=True)
    if args.shard == 0 and not args.limit:
        atomic_dump(qmap, os.path.join(args.out_dir, "questions.json"))
    keys = sorted(pages)[args.shard::args.n_shards]
    if args.limit:
        keys = keys[: args.limit]
    done = {}
    if os.path.exists(out):
        done = {r["key"]: r for r in json.load(open(out))["records"]}
        print(f"[transcribe] resume: {len(done)} pages already in {out}", flush=True)
    todo = [k for k in keys if k not in done]
    print(f"[transcribe] shard {args.shard}/{args.n_shards}: {len(keys)} pages, {len(todo)} to do", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base_model_name)
    rows = (({"doc_images": [ds[pages[k]["loc"][0]]["doc_images"][pages[k]["loc"][1]]]}, 0) for k in todo)  # lazy
    gen = run_ref(args, rows, torch.device("cuda"))
    t_last, t_start = time.time(), time.time()
    for i, (k, (text, st)) in enumerate(zip(todo, gen)):
        now = time.time()
        n_new = len(tok(text, add_special_tokens=False)["input_ids"])
        done[k] = {"key": k, "qids": pages[k]["qids"], "text": text, "new_tokens": n_new,
                   "sec": round(now - t_last, 2), **st}
        t_last = now
        if (i + 1) % 20 == 0 or i + 1 == len(todo):
            atomic_dump({"args": vars(args), "records": list(done.values())}, out)
            print(f"[{i + 1}/{len(todo)}] {now - t_start:.0f}s, {(now - t_start) / (i + 1):.1f} s/page", flush=True)
    recs = [done[k] for k in keys if k in done]
    atomic_dump({"args": vars(args), "records": recs}, out)
    n = len(recs)
    empty = sum(not r["text"].strip() for r in recs)
    cap = sum(r["new_tokens"] >= args.max_new_tokens - 2 for r in recs)
    words = sum(len(r["text"].split()) for r in recs) / max(n, 1)
    secs = sorted(r["sec"] for r in recs)   # the first page's sec includes the model load: report the median
    spp = secs[len(secs) // 2] if secs else 0.0
    print(f"RESULT arm=X_transcribe shard={args.shard} n_shards={args.n_shards} limit={args.limit or 'all'} "
          f"pages={n} unique_total={len(pages)} page_slots={n_slots} empty={empty} at_cap={cap} "
          f"mean_words={words:.0f} max_tokens={max((r['new_tokens'] for r in recs), default=0)} "
          f"s_per_page_median={spp:.1f} wall_s={time.time() - t0:.0f}", flush=True)
    print(f"wrote {out}")
    sys.exit(4 if n == 0 or empty > 0.02 * n else 0)


if __name__ == "__main__":
    main()
