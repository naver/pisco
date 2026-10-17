#!/usr/bin/env python
"""Rejection-sampling (RAFT) diet for multimodal PISCO from a sampled probe on teacher-failed rows.

Input: a probe json from `eval_mm_docqa.py --num_samples G` run on teacher-failed rows (S-) and judged with
`judge_mm_docqa.py --samples`. For every S- row where some sample was judged correct AND finished (emitted a stop
token), the shortest such sample becomes the new `mistral_label` (label_origin = "raft"); the system prompt asks
for brevity, and long samples tend to hit gold inside chatter. Non-answers ("the documents do not specify ...")
are never kept unless the gold is one too: the judge sometimes accepts them. Rows with no usable sample are dropped.
Output = the base diet (e.g. the judge-correct S+ set) + the recovered rows, same schema.

  python scripts/mmrl_raft_build.py --probe <probe json> --rows data/mmrl_sminus_probe_2k \
      --base data/mmrl_ft_20k_pd --out data/mmrl_ft_20k_pd_raft
"""

import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import datasets  # noqa: E402

from pisco.reward import is_refusal  # noqa: E402


def load(path):
    d = datasets.load_from_disk(path)
    return d["train"] if isinstance(d, datasets.DatasetDict) else d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", required=True, action="append", help="repeatable; pairs with --rows in order")
    ap.add_argument("--rows", required=True, action="append",
                    help="the dataset each probe was run on (S- rows, with images); repeatable")
    ap.add_argument("--base", required=True, help="diet the recovered rows are added to (e.g. the S+ set)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert len(args.probe) == len(args.rows), "give one --rows per --probe"
    recs = [r for p in args.probe for r in json.load(open(p))["records"]]
    assert recs and all("samples_judge" in r for r in recs), "probe must be judged with judge_mm_docqa.py --samples"
    parts = [load(p) for p in args.rows]
    rows = parts[0] if len(parts) == 1 else datasets.concatenate_datasets(parts)
    qids = rows["query_id"]
    assert len(set(qids)) == len(qids) and len({r["query_id"] for r in recs}) == len(recs), "query_id not unique"
    pos = {q: i for i, q in enumerate(qids)}

    winners, per, n_refusal = {}, collections.defaultdict(lambda: [0, 0]), 0
    for r in recs:
        gold_refuses = any(is_refusal(a) for a in r["answers"])
        good = [s for s, j, f in zip(r["samples"], r["samples_judge"], r["samples_finished"]) if j and f and s.strip()]
        kept = [s for s in good if gold_refuses or not is_refusal(s)]
        n_refusal += len(good) - len(kept)
        good = kept
        per[r["source"]][0] += 1
        if good:
            winners[r["query_id"]] = min(good, key=len)
            per[r["source"]][1] += 1
    print(f"{'source':18s}{'probed':>8}{'recovered':>11}")
    for s, (n, k) in sorted(per.items(), key=lambda kv: -kv[1][0]):
        print(f"{s:18s}{n:>8}{k:>11}  ({k / n:.1%})")
    print(f"[raft] dropped {n_refusal} judged-correct samples that are non-answers (gold is not one)")

    idx = sorted(pos[q] for q in winners)
    rec = rows.select(idx)
    labels = [winners[q] for q in rec["query_id"]]
    rec = rec.remove_columns("mistral_label").add_column("mistral_label", labels)
    rec = rec.remove_columns([c for c in ("label_origin",) if c in rec.column_names])
    rec = rec.add_column("label_origin", ["raft"] * len(rec))

    base = load(args.base)
    if "label_origin" not in base.column_names:
        base = base.add_column("label_origin", ["teacher"] * len(base))
    rec = rec.select_columns(base.column_names).cast(base.features)
    out = datasets.concatenate_datasets([base, rec])
    datasets.DatasetDict({"train": out}).save_to_disk(args.out)
    words = sorted(len(l.split()) for l in labels)
    print(f"\n[raft] {len(base)} base + {len(rec)} recovered = {len(out)} rows -> {args.out}")
    if words:
        print(f"[raft] recovered target length: median {words[len(words) // 2]} words, p90 {words[int(.9 * len(words))]}")


if __name__ == "__main__":
    main()
