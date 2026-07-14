"""
Phase 0 upgrade (long-doc compression study): build a genuinely long-context QA
eval set from LongBench, in the schema scripts/eval_agent_qa.py expects.

agentQA tops out at ~2900 tokens with no quality breaking point. LongBench QA
subsets run 3k-18k tokens with short extractive answers, so document length finally
becomes a real variable. We map LongBench fields -> our schema and bucket by context
token length (measured with the compressor tokenizer):

    input    -> question
    context  -> trajectory      (the long document the compressor must compress)
    answers  -> ground_truth    (first reference; match metric is substring-based)

Note: THUDM/LongBench ships a loader *script* that modern `datasets` rejects;
use the parquet mirror `jannalu/LongBench` (same schema).

Usage:
    pixi run python scripts/make_longbench_eval.py \
        --subset multifieldqa_en --repo jannalu/LongBench \
        --out_dir eval_data/longbench_multifieldqa_en \
        --tokenizer Qwen/Qwen3-0.6B --edges 4000 8000 12000
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="multifieldqa_en",
                    help="LongBench QA subset (multifieldqa_en, 2wikimqa, hotpotqa, musique, qasper, narrativeqa)")
    ap.add_argument("--repo", default="jannalu/LongBench", help="parquet mirror of LongBench")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--edges", type=int, nargs="+", default=[4000, 8000, 12000],
                    help="ascending token cut points; N edges -> N+1 length buckets")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    out_dir = args.out_dir or f"eval_data/longbench_{args.subset}"
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = load_dataset(args.repo, args.subset, split="test")
    print(f"loaded {args.repo}:{args.subset}  n={len(ds)}")

    edges = sorted(args.edges)
    names, lo = [], 0
    for e in edges:
        names.append(f"{lo}_{e}")
        lo = e
    names.append(f"{lo}_inf")
    buckets = {n: [] for n in names}

    for ex in ds:
        ans = ex.get("answers") or [""]
        ground_truth = ans[0] if isinstance(ans, list) and ans else str(ans)
        n_tok = len(tok(ex["context"])["input_ids"])
        rec = {
            "question": ex["input"],
            "trajectory": ex["context"],   # harness compresses this field
            "ground_truth": ground_truth,
            "all_answers": ans,
            "_n_tokens": n_tok,
        }
        idx = 0
        while idx < len(edges) and n_tok >= edges[idx]:
            idx += 1
        buckets[names[idx]].append(rec)

    os.makedirs(out_dir, exist_ok=True)
    for name, exs in buckets.items():
        out = os.path.join(out_dir, f"{args.subset}_{name}.json")
        json.dump(exs, open(out, "w"))
        tks = [e["_n_tokens"] for e in exs]
        rng = f"{min(tks)}-{max(tks)}" if tks else "-"
        print(f"  {name:>10}: n={len(exs):3d}  token range {rng:>14}  -> {out}")


if __name__ == "__main__":
    main()
