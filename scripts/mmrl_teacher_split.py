#!/usr/bin/env python
"""Split a multimodal finetune set by whether its teacher label is right, and build the mmrl arms.

DEX-Comp (arXiv 2609.05152) stage 1 trains only on questions the uncompressed teacher answers
correctly (S+), judged by an LLM against gold, and keeps the teacher-failed ones (S-) for RL.
Our teacher set (`mistral_label` = Qwen3.5-4B on the raw pages) copies wrong answers on ~25-50%
of the short-gold rows (string match), so the same split applies here.

  dump   dataset -> records json in the eval_mm_docqa format (prediction = teacher label),
         ready for `scripts/judge_mm_docqa.py`
  build  judged json -> three datasets:
           --out_pd     S+ only                                   (DEX-Comp "pure distillation")
           --out_fix    S+ teacher labels + S- with the gold answer, short-gold sources only
                        (docmatix gold is a full sentence, the style that made answers verbose
                        in mmdiag, so its S- rows are dropped)
           --out_probe  a random S- sample for rollouts (can a model trained on S+ solve them?)

  python scripts/mmrl_teacher_split.py dump  --data data/mm_ft_docpages_20k_teacher --out X.json
  python scripts/judge_mm_docqa.py X.json --batch_size 64
  python scripts/mmrl_teacher_split.py build --data data/mm_ft_docpages_20k_teacher --judged X.json \
      --out_pd data/mmrl_ft_20k_pd --out_fix data/mmrl_ft_20k_fix --out_probe data/mmrl_sminus_probe_2k
"""

import argparse
import ast
import collections
import json
import os
import pathlib
import random
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import datasets  # noqa: E402

from pisco.metrics import anls_single  # noqa: E402
from pisco.reward import containment_em  # noqa: E402

LONG_GOLD_SOURCES = {"docmatix"}   # gold = an extracted sentence; never used as a target


def flat_answers(answers):
    out = []
    for a in answers if isinstance(answers, (list, tuple)) else [answers]:
        if a is None:
            continue
        a = str(a).strip()
        if a.startswith("[") and a.endswith("]"):
            try:
                v = ast.literal_eval(a)
                if isinstance(v, (list, tuple)):
                    out.extend(str(x) for x in v)
                    continue
            except (ValueError, SyntaxError):
                pass
        out.append(a)
    return [a for a in out if a]


def load(path):
    d = datasets.load_from_disk(path)
    return d["train"] if isinstance(d, datasets.DatasetDict) else d


def dump(args):
    ds = load(args.data).select_columns(["query", "answers", "source", "query_id", "mistral_label"])
    recs = []
    for i, r in enumerate(ds):
        golds = flat_answers(r["answers"])
        t = r["mistral_label"] or ""
        recs.append({"idx": i, "query_id": r["query_id"], "source": r["source"], "query": r["query"],
                     "answers": golds, "prediction": t, "containment": float(containment_em(t, golds)),
                     "anls": max((anls_single(t, g) for g in golds), default=0.0)})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": {"n": len(recs), "data": args.data}, "records": recs}, f, indent=1, ensure_ascii=False)
    print(f"[dump] {len(recs)} records -> {args.out}")


def save(ds, path):
    datasets.DatasetDict({"train": ds}).save_to_disk(path)
    print(f"[build] {len(ds)} rows -> {path}")


def build(args):
    recs = json.load(open(args.judged))["records"]
    assert all("judge" in r for r in recs), "run scripts/judge_mm_docqa.py on the dump first"
    ds = load(args.data)
    assert len(ds) == len(recs), f"dataset has {len(ds)} rows, judged file {len(recs)}"

    pos = [r["idx"] for r in recs if r["judge"] == 1]
    neg = [r["idx"] for r in recs if r["judge"] == 0]
    fix_gold = [r["idx"] for r in recs if r["judge"] == 0 and r["source"] not in LONG_GOLD_SOURCES and r["answers"]]

    by = collections.defaultdict(lambda: collections.Counter())
    for r in recs:
        c = by[r["source"]]
        c["n"] += 1; c["judge"] += r["judge"]; c["containment"] += r["containment"]
        c["judge_and_containment"] += r["judge"] * r["containment"]
    print(f"\n{'source':18s}{'rows':>7}{'judge S+':>10}{'contain':>9}{'both':>7}")
    for s, c in sorted(by.items(), key=lambda kv: -kv[1]["n"]):
        n = c["n"]
        print(f"{s:18s}{n:>7}{c['judge'] / n:>10.3f}{c['containment'] / n:>9.3f}{c['judge_and_containment'] / n:>7.3f}")
    print(f"\nS+ {len(pos)}  S- {len(neg)}  (S- with short gold, re-targeted in the fix arm: {len(fix_gold)})")

    stats = {"data": args.data, "judged": args.judged, "n": len(recs), "s_plus": len(pos), "s_minus": len(neg),
             "fix_gold": len(fix_gold), "by_source": {s: dict(c) for s, c in by.items()}}

    if args.out_pd:
        pd = ds.select(pos)
        save(pd.add_column("label_origin", ["teacher"] * len(pd)), args.out_pd)
    if args.out_fix:
        # chartqa-style numeric gold carries a trailing period ("0.46."); the teacher style has none.
        # Only numbers are stripped: "U.S.A." / "Inc." keep theirs.
        gold_of = {r["idx"]: re.sub(r"(?<=[\d%])\.$", "", r["answers"][0]) for r in recs if r["answers"]}
        idx = sorted(pos + fix_gold)
        fx = ds.select(idx)
        gold_set = set(fix_gold)
        labels = [gold_of[i] if i in gold_set else lab for i, lab in zip(idx, fx["mistral_label"])]
        fx = fx.remove_columns("mistral_label").add_column("mistral_label", labels)
        save(fx.add_column("label_origin", ["gold" if i in gold_set else "teacher" for i in idx]), args.out_fix)
    if args.out_probe or args.out_rest:
        rng = random.Random(args.seed)
        pick = sorted(rng.sample(neg, min(args.probe_n, len(neg))))
        if args.out_probe:
            save(ds.select(pick), args.out_probe)
            stats["probe"] = {"n": len(pick), "by_source": dict(collections.Counter(recs[i]["source"] for i in pick))}
        if args.out_rest:   # every other S- row (same seed => complement of the probe sample)
            rest = sorted(set(neg) - set(pick))
            save(ds.select(rest), args.out_rest)
            stats["rest"] = {"n": len(rest), "by_source": dict(collections.Counter(recs[i]["source"] for i in rest))}
    if not (args.out_pd or args.out_fix or args.out_probe):
        return   # rest-only call: keep the stats file of the full build
    with open(os.path.splitext(args.judged)[0] + "_split_stats.json", "w") as f:
        json.dump(stats, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--data", required=True)
    d.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--data", required=True)
    b.add_argument("--judged", required=True)
    b.add_argument("--out_pd")
    b.add_argument("--out_fix")
    b.add_argument("--out_probe")
    b.add_argument("--out_rest", help="S- rows NOT in the probe sample (same --probe_n/--seed)")
    b.add_argument("--probe_n", type=int, default=2000)
    b.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dump(args) if args.cmd == "dump" else build(args)


if __name__ == "__main__":
    main()
