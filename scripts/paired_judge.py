"""Paired comparison of two eval outputs (scripts/eval_mm_docqa.py JSON, judge-scored) on the SAME questions.
Prints, overall and per source: mean of A, mean of B, the difference A-B with its paired standard error, a 95%
paired-bootstrap interval, and for a 0/1 metric the exact McNemar test (two-sided binomial on the discordant pairs:
questions A gets right and B wrong, or the reverse). Stdlib only (safe on the login node: small JSON).
  python3 scripts/paired_judge.py A.json B.json [--metric judge] [--names A B] [--boot 2000]
Read it as: if the interval contains 0 (or p > 0.05) the two models are not shown to differ on this set.
"""
import argparse
import json
import math
import random
from collections import defaultdict


def records(path):
    return {r["query_id"]: r for r in json.load(open(path))["records"]}


def stats(pairs):
    d = [a - b for a, b in pairs]
    n = len(d)
    m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n) if n > 1 else float("nan")
    return sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n, m, se, n


def bootstrap_ci(pairs, boot, seed=0):
    """95% percentile interval of the mean paired difference, resampling questions."""
    rng = random.Random(seed)
    d = [a - b for a, b in pairs]
    n = len(d)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(boot))
    return means[int(0.025 * boot)], means[int(0.975 * boot) - 1]


def mcnemar(pairs):
    """Exact two-sided McNemar test for a 0/1 metric: (n A-only-right, n B-only-right, p). None if not binary."""
    if any(a not in (0.0, 1.0) or b not in (0.0, 1.0) for a, b in pairs):
        return None
    b_ = sum(1 for a, b in pairs if a == 1 and b == 0)
    c_ = sum(1 for a, b in pairs if a == 0 and b == 1)
    n = b_ + c_
    if n == 0:
        return b_, c_, 1.0
    k = min(b_, c_)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return b_, c_, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--metric", default="judge")
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    A, B = records(args.a), records(args.b)
    ids = [i for i in A if i in B and args.metric in A[i] and args.metric in B[i]]
    assert ids, f"no shared question ids with metric {args.metric!r}"
    groups = defaultdict(list)
    for i in ids:
        pair = (float(A[i][args.metric]), float(B[i][args.metric]))
        groups["ALL"].append(pair)
        groups[A[i].get("source", "?")].append(pair)
    na, nb = args.names
    print(f"{args.metric}: {na} = {args.a}\n{' ' * len(args.metric)}  {nb} = {args.b}")
    print("%-14s %5s %7s %7s %8s %7s %17s %7s %7s %8s" % ("source", "n", na[:7], nb[:7], "diff", "SE", "95% CI", "A-only", "B-only", "p McN"))
    for g in ["ALL"] + sorted(k for k in groups if k != "ALL"):
        a, b, m, se, n = stats(groups[g])
        lo, hi = bootstrap_ci(groups[g], args.boot)
        mc = mcnemar(groups[g])
        tail = "%7d %7d %8.3f" % mc if mc else "%7s %7s %8s" % ("-", "-", "-")
        print("%-14s %5d %7.3f %7.3f %+8.3f %7.3f [%+6.3f, %+6.3f] %s" % (g, n, a, b, m, se, lo, hi, tail))


if __name__ == "__main__":
    main()
