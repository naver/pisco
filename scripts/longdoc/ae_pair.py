"""Pair two models of a scripts/longdoc/ae_nll.py output per document (stdlib; safe on the login node).
For each condition: mean NLL of A and B (nats/token, lower = better), d = A - B, paired-bootstrap 95% CI, sign test
(two-sided binomial on the non-zero d). The self-pair (A with itself) must give d = 0 exactly.
  python3 scripts/longdoc/ae_pair.py <ae_nll.json> [--a l1024] [--b c256] [--boot 2000]
"""
import argparse
import json
import math
import random


def sign_p(pos, neg):
    n, k = pos + neg, min(pos, neg)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def ci(d, boot, seed=0):
    rng, n = random.Random(seed), len(d)
    m = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(boot))
    return m[int(0.025 * boot)], m[int(0.975 * boot) - 1]


ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--a", default="l1024")
ap.add_argument("--b", default="c256")
ap.add_argument("--boot", type=int, default=2000)
args = ap.parse_args()
res = json.load(open(args.path))
print(f"{len(res['docs'])} documents; NLL in nats per token (lower = better); d = {args.a} - {args.b}")
print(f"{'condition':<12} {'n':>4} {args.a:>8} {args.b:>8} {'d':>8} {'95% CI':>18} {'a<b':>5} {'a>b':>5} {'p sign':>8}  self-pair")
for c in res["conds"]:
    A = [r["nll"] for r in res["models"][args.a]["records"][c]]
    B = [r["nll"] for r in res["models"][args.b]["records"][c]]
    d = [x - y for x, y in zip(A, B)]
    lo, hi = ci(d, args.boot)
    better, worse = sum(x < 0 for x in d), sum(x > 0 for x in d)
    self_d = sum(x - x for x in A) / len(A)
    print(f"{c:<12} {len(d):>4} {sum(A)/len(A):>8.4f} {sum(B)/len(B):>8.4f} {sum(d)/len(d):>+8.4f} "
          f"[{lo:+.4f}, {hi:+.4f}] {better:>5} {worse:>5} {sign_p(better, worse):>8.2g}  {self_d:+.4f}")
