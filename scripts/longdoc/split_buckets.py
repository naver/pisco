"""Campaign longdoc, arm E: write one eval set per length bucket of the pages and mixed sets.

Why: a full uncompressed pages or mixed cell takes 8-10 GPU-h, and gpu-be preemption restarts `eval_mm_4b.sh` from
zero. One cell per bucket loses at most one bucket to a preemption (owner, 2026-11-19, option 1 of BLOCKED.md).
What: for SET in longdoc_pages_1500 / longdoc_mixed_1500, the rows whose `source` is `longdocurl_<b>` (b = p10 / p20 /
p40 for pages, m10 / m20 / m40 for mixed) go, in their original order and byte for byte (no image is decoded), to
eval_data/<SET>_<b>. Checks (exit 1 on any failure): every source row lands in exactly one bucket, the bucket's
query ids all end in `__<b>`, are unique, and equal the source's ids for that bucket in the same order; the three
buckets together give the source's row count. A bucket already written (stats.json present) is kept.
Run it through SLURM (scripts/longdoc/split_buckets.sbatch); CPU only.
"""
import json, os, sys

ROOT = "/beegfs/scratch/user/hdejean/pisco"
SETS = {"longdoc_pages_1500": ["p10", "p20", "p40"], "longdoc_mixed_1500": ["m10", "m20", "m40"]}


def main():
    import datasets as D
    D.disable_caching()
    bad = []
    for s, buckets in SETS.items():
        src = D.load_from_disk(f"{ROOT}/eval_data/{s}")["train"]
        srcs, qids = src["source"], src["query_id"]
        n_total = 0
        for b in buckets:
            idx = [i for i, x in enumerate(srcs) if x == f"longdocurl_{b}"]
            n_total += len(idx)
            out = f"{ROOT}/eval_data/{s}_{b}"
            want = [qids[i] for i in idx]
            if not all(q.endswith(f"__{b}") for q in want):
                bad.append(f"{s}_{b}: a query id does not end in __{b}")
            if len(set(want)) != len(want):
                bad.append(f"{s}_{b}: duplicate query ids")
            if os.path.exists(f"{out}/stats.json"):
                print(f"{out} already written: kept")
            else:
                sub = src.select(idx)
                D.DatasetDict({"train": sub}).save_to_disk(out, max_shard_size="1GB")
            got = D.load_from_disk(out)["train"]["query_id"]
            if got != want:
                bad.append(f"{s}_{b}: written query ids differ from the source order")
            st = {"kind": "bucket", "from": s, "bucket": b, "rows": len(got), "source_rows": len(qids),
                  "first_qid": got[0] if got else None, "how": "rows with source longdocurl_<bucket>, source order"}
            json.dump(st, open(f"{out}/stats.json", "w"), indent=1)
            print(json.dumps(st))
        if n_total != len(qids):
            bad.append(f"{s}: buckets hold {n_total} rows, source {len(qids)}")
    print("FAILURES:", bad if bad else "none")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
