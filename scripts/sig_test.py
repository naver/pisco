#!/usr/bin/env python
"""Paired significance test for two systems on LongBench or bergen-RAG.

Pairs the two systems on the SAME examples (LongBench: by question text, pooled
over length buckets; RAG: by q_id, pooled over datasets), then reports:
  - mean_A, mean_B, observed diff (B-A)
  - 95% CI of the diff via paired bootstrap
  - two-sided p-value via paired sign-flip permutation test
Stdlib only (random/statistics) so it runs under any python.

LongBench (per-example 'f1' or 'match' in the eval JSON's `samples`):
  python scripts/sig_test.py longbench --a expQ/158433 --b expQ/161788 \
      --metric f1 [--a_pat results_lb_ --b_pat results_lb_]
  # uncompressed baselines use --a_pat results_base_ (e.g. --a expQ/149314 --a_pat results_base_)

RAG (per-example 'M'|'EM'|'F1' in eval_dev_out.json):
  python scripts/sig_test.py rag --a mtv1 --b mt8 --metric M \
      --datasets kilt_nq kilt_triviaqa ... [--exp_dir .../expPISCO]
  # 'UNC4B' is a special prefix = fairR4b for kilt_* datasets, oodr_4b otherwise
"""
import argparse
import glob
import json
import math
import os
import statistics as st

BUCKETS = ["0_4000", "4000_8000", "8000_12000", "12000_inf"]
DATASETS7 = ["kilt_nq", "kilt_triviaqa", "kilt_hotpotqa", "2wikimultihopqa", "asqa", "popqa", "sciq"]


def load_longbench(run_dir, metric, pat):
    """question -> score, pooled over buckets."""
    out = {}
    for f in glob.glob(os.path.join(run_dir, "eval", f"{pat}*.json")):
        b = os.path.basename(f).replace(pat, "").replace(".json", "")
        if b not in BUCKETS:
            continue
        for s in json.load(open(f)).get("samples", []):
            out[s["question"]] = float(s[metric])
    return out


def rag_dir(exp_dir, prefix, ds):
    if prefix == "UNC4B":  # uncompressed 4B is split across two run prefixes
        prefix = "fairR4b" if ds.startswith("kilt") else "oodr_4b"
    return os.path.join(exp_dir, f"{prefix}_{ds}")


def load_rag(exp_dir, prefix, datasets, metric):
    """(dataset, q_id) -> score, pooled over datasets."""
    out = {}
    for ds in datasets:
        f = os.path.join(rag_dir(exp_dir, prefix, ds), "eval_dev_out.json")
        if not os.path.exists(f):
            print(f"  (skip {prefix}/{ds}: no file)")
            continue
        for ex in json.load(open(f)):
            out[(ds, ex.get("q_id"))] = float(ex.get(metric, 0.0))
    return out


def paired_stats(a, b):
    """Analytic paired test (two-sided) on per-example differences.
    z-based p-value + normal 95% CI; valid at these n (CLT). For binary metrics
    this is the large-sample equivalent of McNemar on the paired outcomes."""
    d = [bi - ai for ai, bi in zip(a, b)]
    n = len(d)
    obs = st.mean(d)
    sd = st.stdev(d) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    z = obs / se if se > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2)) if se > 0 else 1.0  # two-sided
    return {"n": n, "mean_a": st.mean(a), "mean_b": st.mean(b),
            "diff": obs, "ci95": (obs - 1.96 * se, obs + 1.96 * se), "p": p, "z": z}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    lb = sub.add_parser("longbench")
    lb.add_argument("--a", required=True); lb.add_argument("--b", required=True)
    lb.add_argument("--metric", default="f1", choices=["f1", "match"])
    lb.add_argument("--a_pat", default="results_lb_"); lb.add_argument("--b_pat", default="results_lb_")
    lb.add_argument("--root", default="/beegfs/scratch/user/hdejean/pisco")
    rag = sub.add_parser("rag")
    rag.add_argument("--a", required=True); rag.add_argument("--b", required=True)
    rag.add_argument("--metric", default="M", choices=["M", "EM", "F1"])
    rag.add_argument("--datasets", nargs="+", default=DATASETS7)
    rag.add_argument("--exp_dir", default="/beegfs/scratch/user/hdejean/bergen/expPISCO")
    args = ap.parse_args()

    if args.mode == "longbench":
        A = load_longbench(os.path.join(args.root, args.a), args.metric, args.a_pat)
        B = load_longbench(os.path.join(args.root, args.b), args.metric, args.b_pat)
        label = f"LongBench {args.metric}  A={args.a}  B={args.b}"
    else:
        A = load_rag(args.exp_dir, args.a, args.datasets, args.metric)
        B = load_rag(args.exp_dir, args.b, args.datasets, args.metric)
        label = f"RAG {args.metric}  A={args.a}  B={args.b}  ({len(args.datasets)} ds)"

    keys = sorted(set(A) & set(B), key=str)
    if not keys:
        raise SystemExit("no paired examples (keys did not match between A and B)")
    a = [A[k] for k in keys]
    b = [B[k] for k in keys]
    r = paired_stats(a, b)
    sig = "SIGNIFICANT" if r["p"] < 0.05 else "n.s."
    print(f"\n=== {label} ===")
    print(f"paired n = {r['n']}  (A had {len(A)}, B had {len(B)})")
    print(f"mean A = {r['mean_a']*100:.2f}   mean B = {r['mean_b']*100:.2f}   "
          f"diff (B-A) = {r['diff']*100:+.2f}")
    print(f"95% CI of diff = [{r['ci95'][0]*100:+.2f}, {r['ci95'][1]*100:+.2f}]  (x100)")
    print(f"paired-test p = {r['p']:.4f}  (z={r['z']:.2f})  -> {sig} at 0.05")


if __name__ == "__main__":
    main()
