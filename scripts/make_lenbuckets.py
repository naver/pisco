"""
Phase 0 (long-doc compression study): split a QA eval set into document-length
buckets so document length becomes a controlled axis for eval_agent_qa.py.

Each output file has the same schema as the input and can be passed straight to
scripts/eval_agent_qa.py via --data_path. We tokenize the `trajectory` field with
the compressor backbone tokenizer (Qwen3-0.6B by default) so the bucket boundaries
match what the compressor actually sees.

Usage:
    pixi run python scripts/make_lenbuckets.py \
        --data_path /beegfs/scratch/user/rdeffaye/pisco/agentQA.json \
        --out_dir eval_data/agentqa_buckets \
        --field trajectory \
        --tokenizer Qwen/Qwen3-0.6B \
        --edges 400 800
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="/beegfs/scratch/user/rdeffaye/pisco/agentQA.json")
    ap.add_argument("--out_dir", default="eval_data/agentqa_buckets")
    ap.add_argument("--field", default="trajectory", help="text field to measure length on")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument(
        "--edges",
        type=int,
        nargs="+",
        default=[400, 800],
        help="ascending token-count cut points; N edges -> N+1 buckets",
    )
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    data = json.load(open(args.data_path))

    edges = sorted(args.edges)
    # Human-readable bucket names: short / <edges> / long
    names = []
    lo = 0
    for e in edges:
        names.append(f"{lo}_{e}")
        lo = e
    names.append(f"{lo}_inf")

    buckets = {n: [] for n in names}
    for ex in data:
        n_tok = len(tok(ex[args.field])["input_ids"])
        ex = {**ex, "_n_tokens": n_tok}
        # find bucket
        idx = 0
        while idx < len(edges) and n_tok >= edges[idx]:
            idx += 1
        buckets[names[idx]].append(ex)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"input: {len(data)} examples from {args.data_path}")
    for name, exs in buckets.items():
        out = os.path.join(args.out_dir, f"agentqa_{name}.json")
        json.dump(exs, open(out, "w"))
        tks = [e["_n_tokens"] for e in exs]
        rng = f"{min(tks)}-{max(tks)}" if tks else "-"
        print(f"  {name:>10}: n={len(exs):3d}  token range {rng:>12}  -> {out}")


if __name__ == "__main__":
    main()
