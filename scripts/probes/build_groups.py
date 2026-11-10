"""Arm B of campaign probes: grouped HotpotQA sets that split the long-chunk gain into its causes.

Background: under the spread layout, joining all 10 passages of a question into ONE chunk (g10) beats ten
separate chunks (g1) by +.037 judge accuracy, with no training. Candidate causes: (a) the gold passages see each
other inside the compressor, (b) fewer chunk boundaries / fewer "+1" slots, (c) passage order. Each set below keeps
the same passages and changes only which ones share a chunk:

  gold_joined   the gold passages joined into one chunk (at the first gold's place), each distractor alone.
                Slot total = g1's minus (n_gold - 1): tests (a) at g1's budget.
  distr_joined  the distractors joined into one chunk (at the first distractor's place), each gold alone.
                Slot total close to g10's: tests (b) with the golds kept apart.
  g10_shuf      all 10 joined like g10, passage order shuffled (seeded by query_id): tests (c).
  pred<K>_<picker>  NO ORACLE: the K passages the model itself ranks first joined into one chunk (at the first one's
                place, original order), the others alone. Ranks come from an attention-route dir (--route_dir, written
                by probe_attn_route.py on the g1 layout + route_fusion.py): picker `digit` = meta.json digit_lp,
                `attn` / `product` / `rrf` = fused_<picker>.json. Tests whether a two-pass pipeline (compress
                separately, rank, re-compress the top K together) gets the gold_joined gain without knowing the golds.
                Prints, per picker and K, the share of rows whose top K holds every gold passage.

gold_positions are remapped to the chunks that hold a gold passage. Every other column is copied.

--check rebuilds the existing ad hoc g10 set (2026-11-06) from the base set and must match it row for row
(it also detects the separator g10 used); nothing is written if it fails.

Run through SLURM only (scripts/probes/build_groups.sbatch), never on the login node.
"""
import argparse
import json
import os
import random
import sys
import zlib
from collections import Counter

from datasets import Dataset, DatasetDict, load_from_disk

SEP_CANDIDATES = [" \n\n", "\n\n", "\n", " ", ""]


def load_train(path):
    d = load_from_disk(path)
    return d["train"] if isinstance(d, DatasetDict) else d


def regroup(row, groups, sep):
    """groups: list of lists of original passage indices, in chunk order."""
    docs = list(row["docs"])
    imgs = list(row.get("doc_images") or [])
    gold = {int(g) for g in (row.get("gold_positions") or [])}
    out = dict(row)
    out["docs"] = [sep.join(docs[i] for i in g) for g in groups]
    if imgs:
        for g in groups:
            if len(g) > 1:
                assert all(imgs[i] is None for i in g), "cannot join an image page"
        out["doc_images"] = [imgs[g[0]] if len(g) == 1 else None for g in groups]
    out["gold_positions"] = [n for n, g in enumerate(groups) if gold & set(g)]
    return out


def groups_for(kind, row, top=None):
    n = len(row["docs"])
    gold = sorted({int(g) for g in (row.get("gold_positions") or [])})
    if kind == "g10":
        return [list(range(n))]
    if kind == "g10_shuf":
        order = list(range(n))
        random.Random(zlib.crc32(str(row["query_id"]).encode())).shuffle(order)
        return [order]
    if kind in ("gold_joined", "distr_joined") or kind.startswith("pred"):
        if kind.startswith("pred"):
            joined = sorted(top)
        else:
            joined = gold if kind == "gold_joined" else [i for i in range(n) if i not in gold]
        if len(joined) <= 1:
            return [[i] for i in range(n)]  # nothing to join (logged in the summary)
        out, placed = [], False
        for i in range(n):
            if i in joined:
                if not placed:
                    out.append(list(joined)); placed = True
            else:
                out.append([i])
        return out
    raise ValueError(kind)


def check_g10(base, g10):
    assert len(base) == len(g10), (len(base), len(g10))
    r0, g0 = base[0], g10[0]
    for sep in SEP_CANDIDATES:
        if len(g0["docs"]) == 1 and g0["docs"][0] == sep.join(r0["docs"]):
            break
    else:
        print("[check] FAIL: no separator reproduces g10 row 0")
        print("[check] g10 docs[0][:400] =", repr(g0["docs"][0][:400]))
        print("[check] base docs[0][:200] =", repr(r0["docs"][0][:200]), " docs[1][:200] =", repr(r0["docs"][1][:200]))
        return None
    bad = 0
    for i in range(len(base)):
        a = regroup(base[i], groups_for("g10", base[i]), sep)
        b = g10[i]
        if a["docs"] != list(b["docs"]) or a["gold_positions"] != list(b["gold_positions"]) or a["query_id"] != b["query_id"]:
            bad += 1
            if bad <= 3:
                print(f"[check] row {i} differs: gold {a['gold_positions']} vs {list(b['gold_positions'])}")
    print(f"[check] separator={sep!r}: {len(base) - bad}/{len(base)} rows identical to the existing g10 set")
    return sep if bad == 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="eval_data/text_hotpotqa_1500")
    ap.add_argument("--g10", default="eval_data/text_hotpotqa_1500_g10")
    ap.add_argument("--out_prefix", default="eval_data/probes_hotpot_")
    ap.add_argument("--kinds", default="gold_joined,distr_joined,g10_shuf")
    ap.add_argument("--route_dir", default=None, help="attention-route dir for pred<K>_<picker> kinds")
    args = ap.parse_args()

    base = load_train(args.base)
    print(f"[base] {len(base)} rows, columns {base.column_names}")
    ng = Counter(len(r) for r in base["gold_positions"])
    nd = Counter(len(r) for r in base["docs"])
    print(f"[base] gold passages per row {dict(ng)}; passages per row {dict(nd)}")
    sep = check_g10(base, load_train(args.g10))
    if sep is None:
        print("[check] g10 not reproduced: nothing written"); sys.exit(1)

    summary = {"base": args.base, "separator": sep, "rows": len(base), "gold_per_row": dict(ng), "sets": {}}
    ranks = {}
    if any(k.startswith("pred") for k in args.kinds.split(",")):
        meta = json.load(open(os.path.join(args.route_dir, "meta.json")))
        by_qid = {str(m["query_id"]): m for m in meta}
        for r in base:  # the route dir must be this base set, row for row
            m = by_qid[str(r["query_id"])]
            assert sorted(int(g) for g in m["gold"]) == sorted(int(g) for g in r["gold_positions"]), r["query_id"]
            assert int(m["n_docs"]) == len(r["docs"]), r["query_id"]
        ranks["digit"] = {q: m["digit_lp"] for q, m in by_qid.items()}
        for pk in ("attn", "product", "rrf"):
            f = os.path.join(args.route_dir, f"fused_{pk}.json")
            if os.path.exists(f):
                ranks[pk] = {str(q): v for q, v in json.load(open(f)).items()}
        for pk, rk in ranks.items():
            for K in (1, 2, 3, 4):
                full = sum(1 for r in base if {int(g) for g in r["gold_positions"]} <=
                           set(sorted(range(len(r["docs"])), key=lambda i: -rk[str(r["query_id"])][i])[:K]))
                print(f"[recall] picker={pk} K={K}: top K holds every gold on {full}/{len(base)} rows = {full / len(base):.3f}")
                summary.setdefault("recall_all_golds", {})[f"{pk}@{K}"] = full / len(base)
    for kind in args.kinds.split(","):
        rows, nchunks, unjoined = [], Counter(), 0
        for r in base:
            top = None
            if kind.startswith("pred"):
                K, pk = kind[4:].split("_", 1)
                sc = ranks[pk][str(r["query_id"])]
                top = sorted(range(len(r["docs"])), key=lambda i: -sc[i])[:int(K)]
            groups = groups_for(kind, r, top)
            if all(len(g) == 1 for g in groups):
                unjoined += 1
            new = regroup(r, groups, sep)
            # every passage kept exactly once
            assert sorted(i for g in groups for i in g) == list(range(len(r["docs"])))
            assert sum(len(d) for d in new["docs"]) == sum(len(d) for d in r["docs"]) + len(sep) * (len(r["docs"]) - len(groups))
            assert new["gold_positions"], f"row {r['query_id']}: no gold chunk"
            if kind.startswith("pred"):
                assert len(groups) == len(r["docs"]) - len(top) + 1
            rows.append(new); nchunks[len(groups)] += 1
        ds = Dataset.from_list(rows, features=base.features)
        out = f"{args.out_prefix}{kind}"
        assert not os.path.exists(out), f"{out} exists: never overwrite an eval set"
        DatasetDict({"train": ds}).save_to_disk(out)
        summary["sets"][kind] = {"path": out, "rows": len(ds), "chunks_per_row": dict(sorted(nchunks.items())),
                                 "rows_left_unjoined": unjoined}
        print(f"[{kind}] {len(ds)} rows -> {out}; chunks per row {dict(sorted(nchunks.items()))}; unjoined {unjoined}")
    print("SUMMARY " + json.dumps(summary))


if __name__ == "__main__":
    main()
