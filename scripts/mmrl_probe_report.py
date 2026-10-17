#!/usr/bin/env python
"""mmrl headroom report for a sampled eval file (eval_mm_docqa.py --num_samples G, judged with --samples).

Answers three questions per source:
  1. headroom   pass@G (any of G samples judged correct) vs greedy judge. RL / rejection sampling can
                only reinforce answers the model already produces sometimes; pass@G - greedy bounds it.
  2. RAFT yield fraction of questions with a judge-correct sample that also FINISHED (emitted a stop
                token before max_new_tokens) = rows a rejection-sampling diet would gain.
  3. confidence AUROC of greedy confidence (mean token log-prob, first-token prob, sample agreement)
                for "greedy is right" (.5 = chance). With --base (uncompressed answers on the same
                question ids), the fallback curve: send the least-confident x% to the raw pages.

  python scripts/mmrl_probe_report.py expQ/ft_mm_4b_172671/eval/mm_docqa_pisco_s8.json \
      [--base expQ/base_qwen35_4b/eval/mm_docqa_base.json]
"""

import argparse
import collections
import json
import os

import numpy as np


def auroc(scores, labels):
    """P(score of a positive > score of a negative), ties count half (Mann-Whitney U)."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):   # average ranks over ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(allv)); r[order] = ranks
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def norm(s):
    return " ".join((s or "").lower().strip().rstrip(".").split())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("probe")
    ap.add_argument("--base", default=None, help="uncompressed eval json on the same question ids (fallback curve)")
    args = ap.parse_args()

    recs = [r for r in json.load(open(args.probe))["records"] if r.get("samples")]
    assert recs and all("samples_judge" in r and "judge" in r for r in recs), \
        "probe file must be judged with scripts/judge_mm_docqa.py --samples"
    G = len(recs[0]["samples"])
    for r in recs:
        r["agree"] = float(np.mean([norm(s) == norm(r["prediction"]) for s in r["samples"]]))
        r["raft"] = int(any(j and f for j, f in zip(r["samples_judge"], r["samples_finished"])))
    groups = collections.defaultdict(list)
    for r in recs:
        groups["ALL"].append(r); groups[r.get("source", "?")].append(r)

    print(f"{args.probe}  (G={G}, n={len(recs)})\n")
    print(f"{'source':12s}{'n':>5}{'greedy':>8}{'sample':>8}{'pass@G':>8}{'headroom':>10}"
          f"{'wrong->any':>11}{'RAFT':>7}{'AUC lp':>8}{'AUC p1':>8}{'AUC agr':>8}")
    out = {"probe": args.probe, "G": G, "by_source": {}}
    for src in ["ALL"] + sorted(k for k in groups if k != "ALL"):
        rs = groups[src]
        g = np.array([r["judge"] for r in rs]); pg = np.array([float(any(r["samples_judge"])) for r in rs])
        wrong = [r for r in rs if not r["judge"]]
        row = {"n": len(rs), "greedy": float(g.mean()), "sample": float(np.mean([np.mean(r["samples_judge"]) for r in rs])),
               "pass@G": float(pg.mean()), "headroom": float(pg.mean() - g.mean()),
               "wrong_recoverable": float(np.mean([any(r["samples_judge"]) for r in wrong])) if wrong else float("nan"),
               "raft_yield": float(np.mean([r["raft"] for r in rs])),
               "auroc_logprob": auroc([r["logprob_mean"] for r in rs], g),
               "auroc_pfirst": auroc([r["p_first"] for r in rs], g),
               "auroc_agree": auroc([r["agree"] for r in rs], g)}
        out["by_source"][src] = row
        print(f"{src:12s}{row['n']:>5}{row['greedy']:>8.3f}{row['sample']:>8.3f}{row['pass@G']:>8.3f}"
              f"{row['headroom']:>+10.3f}{row['wrong_recoverable']:>11.3f}{row['raft_yield']:>7.3f}"
              f"{row['auroc_logprob']:>8.3f}{row['auroc_pfirst']:>8.3f}{row['auroc_agree']:>8.3f}")
    print("\ngreedy/sample/pass@G = judge accuracy; headroom = pass@G - greedy; wrong->any = greedy-wrong questions"
          "\nwith >=1 correct sample; RAFT = has a correct AND finished sample; AUC = AUROC for 'greedy is right'.")

    if args.base:
        brecs = json.load(open(args.base))["records"]
        base = {r["query_id"]: r for r in brecs}
        assert None not in base and len(base) == len(brecs), "base file: query_id missing or not unique"
        assert len({r["query_id"] for r in recs}) == len(recs), "probe file: query_id not unique"
        rs = [r for r in recs if r["query_id"] in base and "judge" in base[r["query_id"]]]
        print(f"\nfallback to raw pages for the least-confident x% ({len(rs)} questions with an uncompressed answer)")
        print(f"{'x':>6}" + "".join(f"{c:>12}" for c in ("logprob", "agreement", "random")))
        p, b = np.array([r["judge"] for r in rs]), np.array([base[r["query_id"]]["judge"] for r in rs])
        rng = np.random.default_rng(0)
        curves = {}
        for name, key in (("logprob", lambda r: r["logprob_mean"]), ("agreement", lambda r: r["agree"] + 1e-6 * r["logprob_mean"])):
            order = np.argsort([key(r) for r in rs], kind="mergesort")
            curves[name] = order
        curves["random"] = rng.permutation(len(rs))
        out["fallback"] = {}
        for x in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0):
            k = int(round(x * len(rs)))
            accs = {}
            for name, order in curves.items():
                use_base = np.zeros(len(rs), bool); use_base[order[:k]] = True
                accs[name] = float(np.where(use_base, b, p).mean())
            out["fallback"][x] = accs
            print(f"{x:>6.0%}" + "".join(f"{accs[c]:>12.3f}" for c in ("logprob", "agreement", "random")))
        print(f"oracle (best of both per question): {float(np.maximum(p, b).mean()):.3f}"
              f"   compressed {p.mean():.3f}   uncompressed {b.mean():.3f}")
        out["oracle"] = float(np.maximum(p, b).mean())

    path = os.path.splitext(args.probe)[0] + "_mmrl_report.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
