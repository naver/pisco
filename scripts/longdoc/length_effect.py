"""Paired contrasts of judged eval files (campaign longdoc, arms J / L / U). Stdlib only.

Per question (paired on query_id, only questions present in every file): d = sum_k sign_k * score_k, where each
term is `+FILE` or `-FILE`. Examples:
  two-file difference (g10 - ten chunks):   length_effect.py +g10.json -ten.json
  Delta_len (arm L):  +R_long.json -U_long.json -R_short.json +U_short.json
  dilution D (arm U): +eighth_long.json -plain_long.json -eighth_short.json +plain_short.json
Prints n, mean d, paired bootstrap 95% CI (10,000 resamples, seed 0), sign test on non-zero d (two-sided, exact).
--metric judge (default) | em | match | anls | f1. --json writes the numbers.
"""
import argparse, json, math, random, sys


def load(path, metric):
    recs = json.load(open(path))["records"]
    out = {}
    for r in recs:
        v = r.get(metric)
        if v is None:
            continue
        out[str(r["query_id"])] = float(v)
    return out, len(recs)


def sign_test(pos, neg):
    n, k = pos + neg, min(pos, neg)
    if n == 0:
        return 1.0
    # two-sided exact binomial, p = 0.5, summed in log space
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    return min(1.0, 2 * sum(math.exp(x) for x in logs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="judge")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--json", default=None)
    argv = sys.argv[1:]   # +FILE / -FILE terms are positional (argparse would read -FILE as an option)
    a = ap.parse_args([x for k, x in enumerate(argv) if x.startswith("--") or (k and argv[k - 1].startswith("--"))])
    a.terms = [x for k, x in enumerate(argv) if x[:1] in "+-" and not x.startswith("--") and not (k and argv[k - 1].startswith("--"))]
    assert a.terms, "give +FILE / -FILE terms"
    terms = []
    for t in a.terms:
        assert t[0] in "+-", f"term must start with + or -: {t}"
        sc, n = load(t[1:], a.metric)
        terms.append((1.0 if t[0] == "+" else -1.0, t[1:], sc, n))
    common = set(terms[0][2])
    for _, _, sc, _ in terms[1:]:
        common &= set(sc)
    qs = sorted(common)
    d = [sum(s * sc[q] for s, _, sc, _ in terms) for q in qs]
    n = len(d)
    mean = sum(d) / n if n else float("nan")
    rng = random.Random(0)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(a.boot)) if n else []
    lo, hi = (boots[int(.025 * a.boot)], boots[int(.975 * a.boot) - 1]) if boots else (float("nan"),) * 2
    pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
    p = sign_test(pos, neg)
    means = {f: sum(sc[q] for q in qs) / n for _, f, sc, _ in terms} if n else {}
    res = dict(metric=a.metric, n=n, mean=mean, ci95=[lo, hi], pos=pos, neg=neg, sign_p=p,
               files=[dict(sign=s, file=f, n_rows=nr, mean_on_common=means.get(f)) for s, f, _, nr in terms])
    for t in res["files"]:
        print(f"  {'+' if t['sign'] > 0 else '-'} {t['mean_on_common']:.4f}  (rows {t['n_rows']})  {t['file']}")
    print(f"d = {mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  n={n}  non-zero +{pos}/-{neg}  sign p={p:.3g}  ({a.metric})")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
