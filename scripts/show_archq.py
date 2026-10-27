"""archq probe table: every judged eval of one checkpoint on one eval set, paired with a reference file.

One row per eval json (label = file name minus the common prefix/suffix): judge per source, all, mean decoder prompt
tokens, route hit rate (raw_policy route1), and the paired difference to --ref on the same questions (bootstrap 95%
CI + exact McNemar, scripts/paired_judge.py). Stdlib only.

  python3 scripts/show_archq.py --ref expQ/ft_mm_4b_173550/eval/mm_docqa_pisco_px1003520_mm_docqa_1500_1m.json \
      expQ/ft_mm_4b_173550/eval/mm_docqa_pisco_*_mm_docqa_1500_1m.json expQ/base_qwen35_4b/eval/mm_docqa_base_px1003520*_mm_docqa_1500_1m.json
"""
import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paired_judge import bootstrap_ci, mcnemar  # noqa: E402


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--metric", default="judge")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    ref = {r["query_id"]: r for r in json.load(open(args.ref))["records"]}
    files = [args.ref] + [f for f in dict.fromkeys(args.files) if os.path.abspath(f) != os.path.abspath(args.ref)]
    srcs = sorted({r.get("source") for r in ref.values()})
    print(f"reference: {args.ref}\n")
    print("| arm | n | " + " | ".join(srcs) + " | all | diff vs ref (95% CI) | McNemar p | prompt tokens | route hit |")
    print("|---|---|" + "---|" * len(srcs) + "---|---|---|---|---|")
    for f in files:
        recs = [r for r in json.load(open(f))["records"] if r.get(args.metric) is not None]
        label = os.path.basename(f).replace(".json", "").replace("mm_docqa_", "").replace("_mm_docqa_1500_1m", "")
        label = f"{os.path.basename(os.path.dirname(os.path.dirname(f)))}/{label}"
        by = {s: mean([r[args.metric] for r in recs if r.get("source") == s]) for s in srcs}
        pairs = [(r[args.metric], ref[r["query_id"]][args.metric]) for r in recs
                 if r["query_id"] in ref and ref[r["query_id"]].get(args.metric) is not None]
        diff = ci = p = None
        if pairs and f != args.ref:
            diff = sum(a - b for a, b in pairs) / len(pairs)
            ci = bootstrap_ci(pairs, args.boot)
            mc = mcnemar(pairs)
            p = mc[2] if mc else None
        tok = mean([r.get("decoder_prompt_tokens", r.get("prompt_tokens")) for r in recs])
        hit = mean([r.get("route_hit") for r in recs])
        d = "-" if diff is None else f"{diff:+.3f} ({ci[0]:+.3f}, {ci[1]:+.3f})"
        print(f"| {label} | {len(recs)} | " + " | ".join(fmt(by[s]) for s in srcs)
              + f" | {fmt(mean([r[args.metric] for r in recs]))} | {d} | {fmt(p)} | {fmt(tok, 0)} | {fmt(hit, 2)} |")


if __name__ == "__main__":
    main()
