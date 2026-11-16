"""Pair two transcription-probe rebuilds per page against one reference (campaign longdoc, arm T). Stdlib only.

Word recall per page = share of the reference transcription's words found in the rebuilt text
(scripts/analyze_mem_recall.recall, the metric of architecture_review.md:100-105). Prints, per source and overall:
pages, mean recall of A and B, mean B - A with a paired bootstrap 95% CI, a sign test on the non-zero per-page
differences, and the answer-in-rebuild counts. A file paired with itself must give 0 (control).

  python3 -I scripts/longdoc/recall_pair.py --ref expQ/longdoc/recall/ref_n167.json \
      --a expQ/longdoc/recall/s2_c256_n167.json --b expQ/longdoc/recall/s2_l1024_n167.json
  python3 -I scripts/longdoc/recall_pair.py --ref expQ/archq/recall/ref.json --a expQ/archq/recall/ae_s2.json \
      --b expQ/longdoc/recall/v4kd_n120.json        # port validation: B must match A within 2 pt
"""
import argparse
import json
import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from analyze_mem_recall import contains, norm, recall  # noqa: E402


def load(p):
    return {r["query_id"]: r for r in json.load(open(p))["records"]}


def sign_p(pos, neg):
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot(d, reps=2000, seed=0):
    rng = random.Random(seed)
    n = len(d)
    ms = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return ms[int(.025 * reps)], ms[int(.975 * reps) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args()
    ref, a, b = load(args.ref), load(args.a), load(args.b)
    qs = [q for q in ref if q in a and q in b]
    print(f"ref {len(ref)} / A {len(a)} / B {len(b)} pages; paired on {len(qs)}")
    groups = {}
    for q in qs:
        rn = norm(ref[q]["text"])
        ra, rb = recall(rn, norm(a[q]["text"])), recall(rn, norm(b[q]["text"]))
        if ra is None or rb is None:
            continue
        ans = list(ref[q].get("answers") or [])
        row = (ra, rb, contains(norm(a[q]["text"]), ans), contains(norm(b[q]["text"]), ans))
        for g in (ref[q].get("source"), "ALL"):
            groups.setdefault(g, []).append(row)
    print(f"{'source':10} {'pages':>5} {'A recall':>8} {'B recall':>8} {'B-A':>7} {'95% CI':>17} {'B>A':>5} {'B<A':>5}"
          f" {'sign p':>8} {'ans A':>6} {'ans B':>6}")
    for g in sorted(groups, key=lambda s: (s == "ALL", s)):
        rows = groups[g]
        d = [rb - ra for ra, rb, _, _ in rows]
        lo, hi = boot(d)
        pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
        n = len(rows)
        print(f"{g:10} {n:5d} {sum(r[0] for r in rows) / n:8.3f} {sum(r[1] for r in rows) / n:8.3f} "
              f"{sum(d) / n:+7.3f} [{lo:+.3f}, {hi:+.3f}] {pos:5d} {neg:5d} {sign_p(pos, neg):8.2g} "
              f"{sum(r[2] for r in rows):6d} {sum(r[3] for r in rows):6d}")


if __name__ == "__main__":
    main()
