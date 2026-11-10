"""Page picking without training (campaign probes, arm D): fuse the digit route prompt with the decoder's attention mass.

Input: one or more `expQ/probes/attn_route_<jid>/` dirs written by probe_attn_route.py AFTER 2026-11-11 (meta.json rows
carry `digit_lp`, the full digit log-probs, and `query_id`). Pickers, pick@1 against `gold`:
  digit       argmax of the digit log-probs (the current router)
  attn        argmax of the attention mass, mean over all heads and layers (known-positive control: .885 on retr10, 183814)
  product     argmax of log softmax(digit) + log attention share (both are distributions over the pages)
  rrf         reciprocal-rank fusion, 1/(60 + rank), ties broken by attention
  per_source  digit or attn per source, chosen on the other half of the questions (2-fold by row parity), never in-sample
Every picker is paired with `digit` on the same questions: difference, 95% bootstrap CI (2000 resamples, seed 0) and the
exact McNemar p. Writes <dir>/fusion.json and <dir>/fused_<picker>.json ({query_id: [score per page]}, the eval's RANK=).
Usage: python scripts/probes/route_fusion.py <attn_route dir> [...]   (CPU, seconds; run through SLURM gres=gpu:0)
"""
import json
import math
import os
import sys

import numpy as np


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(h, ref, seed=0, n_boot=2000):
    h, ref = np.asarray(h, float), np.asarray(ref, float)
    d = h - ref
    rng = np.random.default_rng(seed)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    b, c = int(((h == 1) & (ref == 0)).sum()), int(((h == 0) & (ref == 1)).sum())
    return {"diff": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "mcnemar_p": mcnemar_p(b, c), "wins": b, "losses": c}


def rank_desc(x):
    r = np.empty(len(x), int)
    r[np.argsort(-np.asarray(x), kind="stable")] = np.arange(1, len(x) + 1)
    return r


def scores(lp, mass):
    lp = np.asarray(lp, float)
    att = mass.mean((0, 1)).astype(float)
    share = att / max(att.sum(), 1e-12)
    log_sm = lp - np.logaddexp.reduce(lp)
    rrf = 1.0 / (60 + rank_desc(lp)) + 1.0 / (60 + rank_desc(att)) + 1e-9 * share   # tie -> attention
    return {"digit": lp, "attn": att, "product": log_sm + np.log(np.maximum(share, 1e-12)), "rrf": rrf}


def run(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    assert meta and "digit_lp" in meta[0], f"{d}: meta.json has no digit_lp (re-run probe_attn_route.py, 2026-11-11+)"
    z = np.load(os.path.join(d, "masses.npz"))
    keep = [k for k, md in enumerate(meta) if md["digit_lp"] is not None]
    S = {k: scores(meta[k]["digit_lp"], z[f"q{k}"]) for k in keep}
    hit = {p: np.array([int(int(np.argmax(S[k][p])) in meta[k]["gold"]) for k in keep])
           for p in ["digit", "attn", "product", "rrf"]}
    src = np.array([meta[k].get("source") or "-" for k in keep])
    half = np.array([meta[k]["i"] % 2 for k in keep])
    ps = np.zeros(len(keep), int)   # per-source choice, cross-validated
    for s in set(src):
        for f in (0, 1):
            train, test = (src == s) & (half != f), (src == s) & (half == f)
            best = "attn" if hit["attn"][train].mean() > hit["digit"][train].mean() else "digit"
            ps[test] = hit[best][test]
    hit["per_source"] = ps
    res = {"dir": d, "n": len(keep), "n_skipped": len(meta) - len(keep),
           "pick@1": {p: float(h.mean()) for p, h in hit.items()},
           "vs_digit": {p: paired(h, hit["digit"]) for p, h in hit.items() if p != "digit"},
           "by_source": {s: {"n": int((src == s).sum()), **{p: float(h[src == s].mean()) for p, h in hit.items()}}
                         for s in sorted(set(src))}}
    json.dump(res, open(os.path.join(d, "fusion.json"), "w"), indent=1)
    for p in ["attn", "product", "rrf"]:
        json.dump({meta[k]["query_id"]: [float(x) for x in S[k][p]] for k in keep},
                  open(os.path.join(d, f"fused_{p}.json"), "w"))
    print(f"== {d}  n={res['n']} (skipped {res['n_skipped']})")
    for p, v in res["pick@1"].items():
        c = res["vs_digit"].get(p)
        tail = (f"  vs digit {c['diff']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] McNemar p={c['mcnemar_p']:.3g}"
                f" (+{c['wins']}/-{c['losses']})") if c else ""
        print(f"  pick@1 {p:<10} {v:.3f}{tail}")
    for s, v in res["by_source"].items():
        print(f"  {s:<10} n={v['n']} " + " ".join(f"{p} {v[p]:.3f}" for p in hit))


def selftest(src, out):
    """Plumbing check on an OLD probe dir (no digit_lp): synthetic log-probs with argmax == the saved digit_pick must give
    back the saved digit and attention pick@1 exactly."""
    os.makedirs(out, exist_ok=True)
    meta = json.load(open(os.path.join(src, "meta.json")))
    for md in meta:
        n, p = md["n_docs"], md["digit_pick"]
        md["digit_lp"] = None if p is None else [math.log(0.5 if j == p else 0.5 / (n - 1)) for j in range(n)]
        md["query_id"] = str(md["i"])
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"))
    if not os.path.exists(os.path.join(out, "masses.npz")):
        os.symlink(os.path.abspath(os.path.join(src, "masses.npz")), os.path.join(out, "masses.npz"))
    want = json.load(open(os.path.join(src, "summary.json")))
    run(out)
    got = json.load(open(os.path.join(out, "fusion.json")))["pick@1"]
    assert abs(got["digit"] - want["digit_pick@1"]) < 1e-9, (got["digit"], want["digit_pick@1"])
    assert abs(got["attn"] - want["attn_mean_all_pick@1"]) < 1e-9, (got["attn"], want["attn_mean_all_pick@1"])
    print(f"SELFTEST PASS: digit {got['digit']:.4f} attn {got['attn']:.4f} reproduce {src}/summary.json")


if __name__ == "__main__":
    if sys.argv[1] == "--selftest":
        selftest(sys.argv[2], sys.argv[3])
    else:
        for d in sys.argv[1:]:
            run(d)
