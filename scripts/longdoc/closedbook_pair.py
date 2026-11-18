#!/usr/bin/env python3
"""Arm E soundness check 1: uncompressed WITH the document vs closed-book, per bucket (stdlib only).

The closed-book file holds every arm E question once (query_id without the `__<bucket>` suffix); a
with-document file holds rows `<question id>__<bucket>`. Each row is paired with the closed-book row of
its question. Per bucket: n, judge with doc, closed-book judge, diff (pt), paired bootstrap 95% CI,
discordant counts (doc right & cb wrong / doc wrong & cb right), and the law's verdict (diff >= 20 pt).

usage: closedbook_pair.py CLOSEDBOOK.json DOC.json [DOC.json ...] [--metric judge] [--json OUT]
"""
import argparse
import json
import random
from collections import defaultdict


def load(path):
    return json.load(open(path))["records"]


def boot(d, n_boot=10000, seed=0):
    rng = random.Random(seed)
    n = len(d)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("closedbook")
    ap.add_argument("docs", nargs="+")
    ap.add_argument("--metric", default="judge")
    ap.add_argument("--bar", type=float, default=0.20)
    ap.add_argument("--json")
    a = ap.parse_args()
    cb = {r["query_id"]: float(r[a.metric]) for r in load(a.closedbook)}
    out = []
    for f in a.docs:
        rows = load(f)
        by = defaultdict(list)
        missing = 0
        for r in rows:
            qid, _, bucket = r["query_id"].rpartition("__")
            if qid not in cb:
                missing += 1
                continue
            by[bucket].append((float(r[a.metric]), cb[qid]))
        if missing:
            raise SystemExit(f"{f}: {missing} rows have no closed-book question")
        print(f"== {f}  ({len(rows)} rows, metric {a.metric})")
        print(f"{'bucket':>8} {'n':>5} {'with doc':>9} {'closed':>7} {'diff pt':>8} {'95% CI':>16} {'doc+/cb+':>9}  verdict (bar {a.bar*100:.0f} pt)")
        for b in sorted(by, key=lambda s: (len(s), s)) + ["ALL"]:
            p = [x for v in by.values() for x in v] if b == "ALL" else by[b]
            d = [x - y for x, y in p]
            m = sum(d) / len(d)
            lo, hi = boot(d)
            wd = sum(x for x, _ in p) / len(p)
            wc = sum(y for _, y in p) / len(p)
            up = sum(1 for x, y in p if x > y)
            dn = sum(1 for x, y in p if x < y)
            ok = m >= a.bar
            print(f"{b:>8} {len(p):>5} {wd:>9.3f} {wc:>7.3f} {m*100:>+8.1f} [{lo*100:+6.1f},{hi*100:+6.1f}] {up:>4}/{dn:<4}  {'PASS' if ok else 'FAIL'}")
            out.append(dict(file=f, bucket=b, n=len(p), with_doc=wd, closed=wc, diff=m, ci=[lo, hi], doc_only=up, cb_only=dn, pass_=ok))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
