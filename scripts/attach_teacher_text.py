"""Attach teacher descriptions (scripts/teacher_describe_vllm.py JSONL shards) to a stage-2 dataset as a
`teacher_text` column. Keeps the first --n rows (the rows a `path:N` training spec uses); rows without a
description get "" (the collator then falls back to image autoencoding for them).
  python scripts/attach_teacher_text.py --data data/mm_pretrain_natural --jsonl 'data/teacher_desc/natural_*.jsonl' \
      --n 50000 --out data/mm_pretrain_natural_desc
"""
import argparse
import glob
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--jsonl", required=True, help="glob of teacher_describe_vllm.py outputs")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_truncated_frac", type=float, default=0.25,
                    help="refuse if more than this share of descriptions hit max_tokens (prompt/length problem)")
    args = ap.parse_args()
    import datasets
    ds = datasets.load_from_disk(args.data)
    ds = ds["train"] if isinstance(ds, datasets.DatasetDict) else ds
    n = min(args.n or len(ds), len(ds))
    ds = ds.select(range(n))
    desc, n_len = {}, 0
    for path in sorted(glob.glob(args.jsonl)):
        for line in open(path):
            r = json.loads(line)
            if r["idx"] < n and r["teacher_text"]:
                desc[r["idx"]] = r["teacher_text"]
                n_len += r.get("finish") == "length"
    assert desc, f"no descriptions found in {args.jsonl}"
    frac = n_len / len(desc)
    print(f"[attach] {len(desc)}/{n} rows described; {frac:.1%} stopped at max_tokens; "
          f"mean {sum(len(t.split()) for t in desc.values()) / len(desc):.0f} words")
    assert frac <= args.max_truncated_frac, f"{frac:.1%} of descriptions truncated (> {args.max_truncated_frac:.0%})"
    ds = ds.add_column("teacher_text", [desc.get(i, "") for i in range(n)])
    datasets.DatasetDict({"train": ds}).save_to_disk(args.out)
    print(f"[attach] wrote {args.out}: {n} rows, columns {ds.column_names}")


if __name__ == "__main__":
    main()
