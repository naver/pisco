"""Campaign longdoc, arm E soundness check 1: a CLOSED-BOOK set, every LongDocURL question ONCE, no document.

The pages and mixed sets ask each of their 1,407 questions at 10 / 20 / 40 pages; the text set asks 650 of them at
4k-32k tokens. Without a document the base prompt is the same for every bucket (gen_base: background "" when
docs[:TOPK] is empty), so one closed-book answer per question serves every set and bucket. This writes
eval_data/longdoc_closedbook: one row per base question id (`<id>` = query_id up to the last `__`), docs and
doc_images empty, gold_positions empty, source `longdocurl_cb`; query and answers copied from the first set holding the
id. Every query text and answer list is checked equal across the sets for the same id (printed; a mismatch is listed).
Run it through SLURM (scripts/longdoc/build_closedbook.sbatch); the eval cell then runs base mode with TOPK=0.
"""
import json, os, sys, collections

ROOT = "/beegfs/scratch/user/hdejean/pisco"
SETS = ["longdoc_pages_1500", "longdoc_mixed_1500", "longdoc_text_1500"]
OUT = f"{ROOT}/eval_data/longdoc_closedbook"


def base_id(qid):
    return qid.rsplit("__", 1)[0]


def main():
    import datasets as D
    if os.path.exists(f"{OUT}/stats.json"):
        print(f"{OUT} already built: kept"); return
    rows, order, seen_in = {}, [], collections.defaultdict(list)
    mism = []
    for s in SETS:
        ds = D.load_from_disk(f"{ROOT}/eval_data/{s}")["train"]
        ds = ds.select_columns(["query", "mistral_label", "answers", "source", "query_id"])
        n = 0
        for r in ds:
            b = base_id(r["query_id"]); n += 1
            seen_in[b].append(s)
            if b not in rows:
                rows[b] = dict(docs=[], doc_images=[], query=r["query"], mistral_label=r["mistral_label"],
                               answers=list(r["answers"]), gold_positions=[], source="longdocurl_cb", query_id=b)
                order.append(b)
            elif rows[b]["query"] != r["query"] or rows[b]["answers"] != list(r["answers"]):
                mism.append((b, r["source"]))
        print(f"{s}: {n} rows, {len({base_id(q) for q in ds['query_id']})} questions")
    per_set = {s: sum(1 for b in order if s in seen_in[b]) for s in SETS}
    feats = D.Features({"docs": D.Sequence(D.Value("string")), "doc_images": D.Sequence(D.Image()),
                        "query": D.Value("string"), "mistral_label": D.Value("string"),
                        "answers": D.Sequence(D.Value("string")), "gold_positions": D.Sequence(D.Value("int32")),
                        "source": D.Value("string"), "query_id": D.Value("string")})
    ds = D.Dataset.from_list([rows[b] for b in order], features=feats)
    D.DatasetDict({"train": ds}).save_to_disk(OUT)
    stats = {"kind": "closedbook", "from": SETS, "rows": len(order), "questions_in_pages_or_mixed": per_set[SETS[0]],
             "questions_in_text": per_set[SETS[2]], "query_or_answer_mismatch": len(mism), "mismatch_ids": mism[:50],
             "how": "one row per base question id, docs empty; eval base mode with TOPK=0 (closed-book)"}
    json.dump(stats, open(f"{OUT}/stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))
    return 0 if not mism else 1


if __name__ == "__main__":
    sys.exit(main())
