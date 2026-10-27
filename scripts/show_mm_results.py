#!/usr/bin/env python3
"""Multimodal doc-QA results table: every mm_docqa*.json under expQ/, one row per (model, checkpoint,
image rate), columns = metrics per dataset, ALWAYS next to the uncompressed baseline restricted to the
SAME questions (previews run on a 100-question subset; the baseline row is recomputed on those ids).

  python scripts/show_mm_results.py [--metric judge|em|anls|match ...] [--ci]
Metrics: judge = Qwen3-8B Yes/No judge (project prompt); em = normalised exact match; anls = DocVQA
ANLS; match = gold contained in the answer. '-' = not scored yet (judge runs after each eval).
--ci adds, per metric, the PAIRED difference model - baseline on the shared questions with its 95% interval
(1.96 x paired standard error): an interval containing 0 means "not shown to differ from the uncompressed model".
For two PISCO arms against each other use scripts/paired_judge.py.
"""
import argparse
import collections
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "expQ/base_qwen35_4b/eval/mm_docqa_base.json")
METRICS = ["judge", "em", "anls", "match"]


def load(path):
    d = json.load(open(path))
    return d["records"]


def agg(recs, metric):
    by = collections.defaultdict(list)
    for r in recs:
        if metric in r:
            by[r.get("source", "?")].append(float(r[metric]))
            by["all"].append(float(r[metric]))
    return {k: sum(v) / len(v) for k, v in by.items() if v}


def paired_delta(recs, base, metric):
    """{source: (mean difference model - baseline, half-width of the 95% interval)} on the shared question ids."""
    b = {r["query_id"]: r for r in base}
    by = collections.defaultdict(list)
    for r in recs:
        o = b.get(r["query_id"])
        if o is not None and metric in r and metric in o:
            d = float(r[metric]) - float(o[metric])
            by[r.get("source", "?")].append(d)
            by["all"].append(d)
    out = {}
    for k, d in by.items():
        n = len(d)
        m = sum(d) / n
        se = (sum((x - m) ** 2 for x in d) / (n - 1) / n) ** 0.5 if n > 1 else float("nan")
        out[k] = (m, 1.96 * se)
    return out


def label(path):
    run = path.split("/expQ/")[1].split("/")[0]
    fn = os.path.basename(path)
    ck = re.search(r"_(checkpoint-\d+)", fn)
    rate = re.search(r"_r(\d+)", fn)
    mode = "base" if "_base" in fn else ("pisco_text" if "pisco_text" in fn else "pisco")
    return run, (ck.group(1) if ck else "final"), (rate.group(1) if rate else "8"), mode


def setting(path):
    """(pixel budget, docs per question) an eval file was run with, from the eval_mm_4b.sh filename suffixes."""
    fn = os.path.basename(path)
    px = re.search(r"_px(\d+)", fn)
    k = re.search(r"_k(\d+)", fn)
    return (int(px.group(1)) if px else 451584, int(k.group(1)) if k else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", nargs="+", default=METRICS)
    ap.add_argument("--ci", action="store_true", help="add the paired model-baseline difference with its 95%% interval")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(ROOT, "expQ/*/eval/mm_docqa*.json")))
    # uncompressed-baseline files, grouped by eval setting (pixel budget, docs per question): the 1M-px doc set and
    # the k8 nav set REUSE the query ids of their default-setting twins, so a flat id map lets one overwrite the other
    base_files = sorted(glob.glob(os.path.join(ROOT, "expQ/base_qwen35_4b/eval/mm_docqa_base*.json")))
    base_by = collections.defaultdict(dict)
    for f in base_files:
        base_by[setting(f)].update({r["query_id"]: r for r in load(f)})

    rows = []
    for f in files:
        if f in base_files:
            continue
        if f.endswith("_report.json"):  # mmrl probe summaries (scripts/mmrl_probe_report.py), no per-question records
            continue
        recs = load(f)
        run, ck, rate, mode = label(f)
        ids = [r["query_id"] for r in recs]
        n = len(recs)
        srcs = sorted(set(r.get("source", "?") for r in recs))
        base = base_by.get(setting(f), {})
        subset_base = [base[i] for i in ids if i in base]
        rows.append((run, ck, rate, mode, n, "+".join(srcs), recs, subset_base))

    # one block per evaluated file, baseline recomputed on the same question ids
    cell_w = 12 + (16 if args.ci else 0)
    print(f"{'RUN':34s} {'CKPT':16s} {'RATE':5s} {'N':>4s} {'DATASETS':22s} | " + " ".join(
        f"{m:>5s}/{'base':>5s}" + (f" {'diff±95%':>14s} " if args.ci else "") for m in args.metric))
    print("-" * (34 + 17 + 6 + 5 + 23 + 3 + cell_w * len(args.metric)))
    for run, ck, rate, mode, n, srcs, recs, sb in sorted(rows, key=lambda r: (r[0], r[1], r[2])):
        per_src = sorted(set(r.get("source", "?") for r in recs)) + (["all"] if len(set(r.get("source") for r in recs)) > 1 else [])
        for src in per_src:
            cells = []
            for m in args.metric:
                a = agg(recs, m).get(src)
                b = agg(sb, m).get(src) if sb else None
                cell = (f"{a:5.3f}/{b:5.3f}" if a is not None and b is not None else
                        (f"{a:5.3f}/  -  " if a is not None else "  -  /  -  "))
                if args.ci:
                    dl = paired_delta(recs, sb, m).get(src) if sb else None
                    cell += f" {dl[0]:+6.3f} ±{dl[1]:5.3f} " if dl else " " * 16
                cells.append(cell)
            name = f"{run} ({mode})" if mode != "pisco" else run
            print(f"{name:34s} {ck:16s} {rate:5s} {sum(1 for r in recs if r.get('source') == src) if src != 'all' else n:4d} {src:22s} | " + " ".join(cells))
    print(f"\nbaseline = uncompressed Qwen3.5-4B reading the raw images, same prompt and same eval setting "
          f"(pixel budget, docs per question); {len(base_files)} baseline file(s)")
    for (px, k), base in sorted(base_by.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        full = agg(list(base.values()), "judge")
        print(f"  px {px} k {k or 'default'}: " + ", ".join(f"{s} {v:.3f}" for s, v in sorted(full.items()) if s != "all"))


if __name__ == "__main__":
    main()
