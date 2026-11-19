"""Campaign slotcheck, arm X (check 3): the three eval sets over the SAME 500 document questions (CPU).

  img      = the rows of eval_data/mm_docqa_1500_1m unchanged (baseline: image slots only; image bytes copied as is)
  text     = per question its 4 pages as TEXT documents: the uncompressed Qwen3.5-4B transcription of each page
             (expQ/slotcheck/transcribe/shard*of4.json, transcribe_pages.py); doc_images all None
  imgtext  = per page its image, then its transcription (8 documents; gold page g -> positions 2g and 2g + 1)
Questions and page keys: transcribe_pages.pick_questions + page_key (re-derived here and checked against
questions.json, so a transcription can never land on another page). No raw page enters any decoder prompt.

Audit (the law's health for arm X): every row goes through scripts/longdoc/audit_inputs.py `eval_row` (imported as a
tool, never edited) with the eval's own collator settings of the cell (CHUNK=1 -> chunk_docs=True, CML=256, TOPK 4/8):
<MEM>s per question, decoder prompt tokens, compressor tokens per chunk, answer string in the transcription (`in_raw`),
cut = answer in the transcription but not in what the compressor gets (must be 0). Writes <set>/dml.txt = max prompt +
256 (the law's DML), read by eval_x.sbatch. Exit 4 on any failed health check.

  python scripts/slotcheck/build_text_sets.py --out expQ/slotcheck/textsets_<job> [--limit 30 --tag _smoke]
"""
import argparse
import collections
import json
import os
import pathlib
import sys
import time
from multiprocessing import Pool

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "scripts/slotcheck"))
sys.path.insert(0, str(REPO / "scripts/longdoc"))
ED = REPO / "eval_data"
SETS = {"img": (4, 256, False), "text": (4, 256, True), "imgtext": (8, 256, True)}   # name: (TOPK, CML, CHUNK)


def log(*a):
    print(f"[build_text_sets {time.strftime('%H:%M:%S')}]", *a, flush=True)


def features():
    import datasets as D
    return D.Features({"docs": D.Sequence(D.Value("string")), "doc_images": D.Sequence(D.Image()),
                       "query": D.Value("string"), "mistral_label": D.Value("string"),
                       "answers": D.Sequence(D.Value("string")), "gold_positions": D.Sequence(D.Value("int32")),
                       "source": D.Value("string"), "query_id": D.Value("string")})


def _init_worker(tag):
    import audit_inputs as A
    A._setup()
    from pisco.mm_collator import MultimodalFineTuningCollator
    for name, (topk, cml, chunk) in SETS.items():
        A.VARIANTS[f"X_{name}{tag}"] = (f"slotcheck_docqa_{name}_500{tag}", topk, 0, cml, False, True)
        # the eval's make_collator with this cell's flags (audit_inputs' own _coll has chunk_docs=False)
        A.G["colls"][(topk, cml)] = MultimodalFineTuningCollator(
            compressor_tokenizer=A.G["ctok"], decoder_tokenizer=A.G["dtok"], compr_rate=A.G["cfg"].compr_rate,
            compressor_max_length=cml, image_processor=A.G["ip"], image_compr_rate=None, topk_docs=topk,
            query_dependent=False, pad_to_multiple_of=None, chunk_docs=chunk)
    # img and text share the key (4, 256): text's chunk_docs=True wins, a no-op on image-only rows (images are atomic)


def _audit(task):
    import audit_inputs as A
    return A.eval_row(task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="audit dir (audit.json)")
    ap.add_argument("--data_path", default=str(ED / "mm_docqa_1500_1m"))
    ap.add_argument("--tx_dir", default=str(REPO / "expQ/slotcheck/transcribe"))
    ap.add_argument("--n_shards", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="plumbing: the first N questions only (use with --tag)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--dml_cap", type=int, default=2048, help="health: max prompt + 256 must fit the eval default")
    args = ap.parse_args()
    assert not args.limit or args.tag, "a --limit build needs its own --tag (never overwrite the full sets)"
    import datasets as D
    dec = D.Image()
    from transcribe_pages import pick_questions, page_key, PER_SOURCE
    os.makedirs(args.out, exist_ok=True)
    fails = []
    # --- transcriptions
    tx = {}
    for s in range(args.n_shards):
        for r in json.load(open(f"{args.tx_dir}/shard{s}of{args.n_shards}.json"))["records"]:
            tx[r["key"]] = r
    qmap = json.load(open(f"{args.tx_dir}/questions.json"))
    log(f"{len(tx)} transcriptions from {args.n_shards} shards; {len(qmap)} questions in questions.json")
    # --- questions: re-derive and check the keys
    ds = D.load_from_disk(args.data_path)
    ds = ds["train"] if isinstance(ds, D.DatasetDict) else ds
    qs = pick_questions(ds, 4)
    raw = ds.cast_column("doc_images", D.Sequence(D.Image(decode=False)))
    if args.limit:
        qs = qs[: args.limit]
    rows = {"img": [], "text": [], "imgtext": []}
    stats = collections.Counter()
    for i, qid, imgs in qs:
        keys = [page_key(im) for im in imgs]
        if keys != qmap.get(qid):
            fails.append(f"key mismatch {qid}"); continue
        miss = [k for k in keys if k not in tx]
        if miss:
            fails.append(f"no transcription {qid} {miss}"); continue
        r = raw[i]
        bytes_imgs = list(r["doc_images"])[:4]
        # the bytes we copy decode to the very pixels that were transcribed
        if [page_key(dec.decode_example(b)) for b in bytes_imgs] != keys:   # the datasets decoder the eval uses
            fails.append(f"bytes != pixels {qid}"); continue
        base = {k: r[k] for k in ("query", "mistral_label", "answers", "source", "query_id")}
        gold = [int(g) for g in (r.get("gold_positions") or []) if int(g) < 4]
        texts = [tx[k]["text"] for k in keys]
        stats["empty"] += sum(not t.strip() for t in texts)
        stats["page_slots"] += len(texts)
        rows["img"].append(dict(r, gold_positions=[int(g) for g in (r.get("gold_positions") or [])]))
        rows["text"].append(dict(base, docs=texts, doc_images=[None] * len(texts), gold_positions=gold))
        d2, i2 = [], []
        for t, b in zip(texts, bytes_imgs):
            d2 += ["", t]; i2 += [b, None]
        rows["imgtext"].append(dict(base, docs=d2, doc_images=i2, gold_positions=sorted([2 * g for g in gold] + [2 * g + 1 for g in gold])))
    n_exp = args.limit or sum(PER_SOURCE.values())
    for name, rr in rows.items():
        if len(rr) != n_exp:
            fails.append(f"{name}: {len(rr)} rows, expected {n_exp}")
    # --- write (a set already complete is kept, never rebuilt; a partial one is set aside)
    for name, rr in rows.items():
        out = ED / f"slotcheck_docqa_{name}_500{args.tag}"
        if (out / "stats.json").exists():
            log(f"{out} already built (stats.json present): kept"); continue
        if out.exists():
            os.replace(out, f"{out}.partial_{time.strftime('%m%d%H%M')}")
        dset = D.Dataset.from_list(rr, features=features())
        D.DatasetDict({"train": dset}).save_to_disk(str(out), max_shard_size="1GB")
        back = D.load_from_disk(str(out))["train"]
        assert len(back) == len(rr) and back[0]["query_id"] == rr[0]["query_id"]
        json.dump({"rows": len(back), "by_source": dict(collections.Counter(back["source"]))}, open(out / "stats.json", "w"), indent=1)
        log(f"wrote {out}: {len(back)} rows")
    # --- audit through audit_inputs.eval_row with the cell's collator flags
    tasks = [(f"X_{n}{args.tag}", j) for n in SETS for j in range(len(rows[n]))]
    with Pool(args.workers, initializer=_init_worker, initargs=(args.tag,)) as pool:
        res = pool.map(_audit, tasks, chunksize=8)
    summ = {}
    for n in SETS:
        R = [x for x in res if x["v"] == f"X_{n}{args.tag}"]
        mem = [x["n_mem"] for x in R]; pr = [x["prompt"] for x in R]
        in_raw = sum(x["in_raw"] for x in R); cut = sum(x["cut_pisco"] for x in R)
        summ[n] = dict(rows=len(R), mems_total=sum(mem), mems_mean=round(sum(mem) / max(len(R), 1), 1), mems_max=max(mem),
                       prompt_max=max(pr), dml=max(pr) + 256, chunk_tok_max=max(x["chunk_tok_max"] for x in R),
                       chunks_mean=round(sum(x["n_chunks"] for x in R) / max(len(R), 1), 2),
                       answer_in_text=in_raw, cut=cut, gold_dropped=sum(x["gold_dropped"] for x in R),
                       by_source={s: dict(rows=sum(x["src"] == s for x in R),
                                          mems_mean=round(sum(x["n_mem"] for x in R if x["src"] == s) /
                                                          max(1, sum(x["src"] == s for x in R)), 1),
                                          answer_in_text=sum(x["in_raw"] for x in R if x["src"] == s))
                                  for s in sorted(PER_SOURCE)})
        if n != "img" and cut:
            fails.append(f"{n}: cut {cut} rows (answer in the transcription, not in the compressor's text)")
        if summ[n]["dml"] > args.dml_cap:
            fails.append(f"{n}: max prompt {max(pr)} + 256 > {args.dml_cap}")
        # a chunk's length counts its <MEM>s: CML text tokens + CML // 8 + 1 (rate 8, the law's; 192214 used CML + 1).
        # Only the text set is all-text; imgtext's chunks include atomic image spans, checked below by additivity.
        if n == "text" and summ[n]["chunk_tok_max"] > SETS[n][1] + SETS[n][1] // 8 + 1:
            fails.append(f"{n}: a text chunk of {summ[n]['chunk_tok_max']} tokens > CML {SETS[n][1]} + its <MEM>s")
    # imgtext = img pages + text pages of the same question: its <MEM>s per question must be exactly their sum
    per = {n: {x["qid"]: x["n_mem"] for x in res if x["v"] == f"X_{n}{args.tag}"} for n in SETS}
    bad = [q for q in per["imgtext"] if per["imgtext"][q] != per["img"].get(q, -1) + per["text"].get(q, -1)]
    if bad:
        fails.append(f"imgtext: <MEM>s != img + text on {len(bad)} questions (first {bad[:3]})")
    for n in SETS:
        out = ED / f"slotcheck_docqa_{n}_500{args.tag}"
        (out / "dml.txt").write_text(f"{summ[n]['dml']}\n")
    stats = dict(stats, questions=len(rows["img"]), empty_share=round(stats["empty"] / max(1, stats["page_slots"]), 4))
    if stats["empty_share"] > 0.02:
        fails.append(f"empty transcriptions {stats['empty']}/{stats['page_slots']} > 2%")
    import hashlib
    md5 = {p: hashlib.md5(open(REPO / p, "rb").read()).hexdigest()[:8] for p in
           ("scripts/longdoc/audit_inputs.py", "scripts/slotcheck/build_text_sets.py", "scripts/slotcheck/transcribe_pages.py",
            "pisco/mm_collator.py", "scripts/eval_mm_docqa.py")}
    json.dump(dict(summary=summ, stats=stats, fails=fails, md5=md5, rows=res), open(f"{args.out}/audit.json", "w"), indent=1)
    for n, s in summ.items():
        log(n, {k: v for k, v in s.items() if k != "by_source"})
    print("RESULT arm=X_build tag=" + (args.tag or "none") + f" questions={stats['questions']} empty={stats['empty']} "
          + " ".join(f"{n}_mems_mean={s['mems_mean']} {n}_prompt_max={s['prompt_max']} {n}_cut={s['cut']} "
                     f"{n}_answer_in_text={s['answer_in_text']}" for n, s in summ.items())
          + f" ok={not fails}", flush=True)
    for f in fails[:20]:
        print("FAIL", f)
    sys.exit(4 if fails else 0)


if __name__ == "__main__":
    main()
