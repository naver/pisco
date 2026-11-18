"""Campaign spread, arm C: does the layout difference grow with input size? Stdlib only, reads judged eval jsons.

Per shared question: DiD = (S_big - T_big) - (S_small - T_small), S = spread, T = tail, same seed, same questions.
Prints per row: n, S-T at the big input, S-T at the small input, DiD, its 95% paired-bootstrap CI over questions
(scripts/paired_judge.py bootstrap_ci, never edited), and a self-check (DiD of a cell set with itself = 0.000).
Gate (CAMPAIGN.md arm C): DiD >= +.03 with the CI above 0 at both seeds -> "the layout pays at length".
  python3 scripts/spread/did_pairs.py --spec scripts/spread/did_pairs.tsv --out expQ/spread/did [--metric judge]
spec TSV: case, seed, S_big, T_big, S_small, T_small   ('#' lines are comments)
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from paired_judge import bootstrap_ci, stats  # noqa: E402


def recs(p):
    return {r["query_id"]: r for r in json.load(open(os.path.join(ROOT, p)))["records"]}


def did(paths, metric, boot):
    R = [recs(p) for p in paths]
    ids = sorted(set.intersection(*(set(i for i in r if metric in r[i]) for r in R)))
    v = [[float(r[i][metric]) for r in R] for i in ids]
    pairs = [(sb - tb, ss - ts) for sb, tb, ss, ts in v]
    big = stats([(sb, tb) for sb, tb, _, _ in v])[2]
    small = stats([(ss, ts) for _, _, ss, ts in v])[2]
    _, _, m, se, n = stats(pairs)
    lo, hi = bootstrap_ci(pairs, boot)
    return n, big, small, m, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="judge")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    L = [f"metric: {a.metric}", "",
         "| case | seed | n | S−T big ↑ | S−T 4 pages ↑ | DiD ↑ | 95% CI | gate (≥ +.03, CI > 0) | self-check |",
         "|---|---|---|---|---|---|---|---|---|"]
    ok = True
    for line in open(os.path.join(ROOT, a.spec)):
        if line.startswith("#") or not line.strip():
            continue
        c, seed, sb, tb, ss, ts = line.rstrip("\n").split("\t")[:6]
        n, big, small, m, lo, hi = did([sb, tb, ss, ts], a.metric, a.boot)
        z = did([sb, sb, ss, ss], a.metric, 200)[3]
        ok &= z == 0.0
        L.append(f"| {c} | {seed} | {n} | {big:+.3f} | {small:+.3f} | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
                 f"{'met' if m >= 0.03 and lo > 0 else 'not met'} | {z:.3f} {'ok' if z == 0.0 else 'FAIL'} |")
    os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
    name = "did.md" if a.metric == "judge" else f"did_{a.metric}.md"
    open(os.path.join(ROOT, a.out, name), "w").write("\n".join(L) + "\n")
    print("\n".join(L) + f"\nRESULT did_controls={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
