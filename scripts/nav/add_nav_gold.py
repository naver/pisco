"""Add gold frame positions to a nav memory-QA set (scripts/nav/build_nav_datasets.py wrote gold_positions = []), so
route-then-expand on frames can be scored. Each row's frames are matched by content (md5 of the stored JPEG bytes) to
its episode's frames (query_id = "<episode>-w<w>-q<k>"); the asked categories are parsed from the question templates
and looked up in the episode's per-frame semantic objects, with the builder's rule (area >= --min_area):
  presence  Yes -> every frame showing the object; No -> none (no page to route to)
  order     the first-sight frames of both objects
  position  the first-sight frame of the object
Every stored answer is recomputed from the metadata and must match (else the row is reported and the run fails).

  python scripts/nav/add_nav_gold.py --data data/nav_ft_val_v2 --episodes 'data/nav_hm3d_val/shard_*/*' \
      --out eval_data/nav_ft_val_v2_gold
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from collections import Counter

PRESENCE = re.compile(r"^Did you see a (.+) during the walk\? Answer yes or no\.$")
ORDER = re.compile(r"^Which did you see first, the (.+) or the (.+)\?$")
POSITION = re.compile(r"^When you first saw the (.+), was it on your left, in the center, or on your right\? ")


def first_seen(objs, cat, min_area):
    for i, f in enumerate(objs):
        for o in f:
            if o["category"] == cat and o["area"] >= min_area:
                return i, o
    return None, None


def gold_of(row, objs, min_area):
    """(gold positions, recomputed answer) of one row; objs = the row's frames' object lists."""
    q, kind = row["query"], row["source"]
    if kind == "nav_presence":
        c = PRESENCE.match(q).group(1)
        gold = [i for i, f in enumerate(objs) if any(o["category"] == c and o["area"] >= min_area for o in f)]
        anywhere = any(o["category"] == c for f in objs for o in f)
        return gold, "Yes" if gold else ("No" if not anywhere else "?")
    if kind == "nav_order":
        a, b = ORDER.match(q).groups()
        ia, _ = first_seen(objs, a, min_area)
        ib, _ = first_seen(objs, b, min_area)
        return [ia, ib], (a if ia < ib else b)
    if kind == "nav_position":
        c = POSITION.match(q).group(1)
        i, o = first_seen(objs, c, min_area)
        return [i], o["pos"]
    raise ValueError(f"unknown source {kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--episodes", required=True, help="glob of episode dirs (each with meta.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_area", type=float, default=0.02, help="the builder's --min_area")
    args = ap.parse_args()

    import datasets
    eps = {os.path.basename(d): d for d in glob.glob(args.episodes) if os.path.exists(os.path.join(d, "meta.json"))}
    cache = {}

    def episode(name):
        if name not in cache:
            d = eps[name]
            m = json.load(open(os.path.join(d, "meta.json")))
            h = {hashlib.md5(open(os.path.join(d, f["file"]), "rb").read()).hexdigest(): i
                 for i, f in enumerate(m["frames"])}
            cache[name] = (m, h)
        return cache[name]

    bad, kinds, n_gold = [], Counter(), Counter()

    def label(row):
        m, h = episode(row["query_id"].rsplit("-w", 1)[0])
        idx = [h[hashlib.md5(im["bytes"]).hexdigest()] for im in row["doc_images"]]
        gold, ans = gold_of(row, [m["frames"][i]["objects"] for i in idx], args.min_area)
        if ans != row["mistral_label"] or any(g is None for g in gold):
            bad.append((row["query_id"], row["query"], row["mistral_label"], ans, gold))
        kinds[row["source"]] += 1
        n_gold[(row["source"], len(set(gold)))] += 1
        row["gold_positions"] = sorted(set(g for g in gold if g is not None))
        return row

    ds = datasets.load_from_disk(args.data)
    splits = ds if isinstance(ds, datasets.DatasetDict) else {"_": ds}
    out = {}
    for name, d in splits.items():
        feats = d.features
        raw = d.cast_column("doc_images", datasets.Sequence(datasets.Image(decode=False)))
        out[name] = raw.map(label, load_from_cache_file=False).cast_column("doc_images", feats["doc_images"])
    print("rows per kind:", dict(kinds))
    print("gold frames per row (kind, n):", dict(sorted(n_gold.items())))
    if bad:
        print(f"!! {len(bad)} rows whose answer does not match the metadata, e.g.", bad[:5])
        return 1
    res = datasets.DatasetDict(out) if isinstance(ds, datasets.DatasetDict) else out["_"]
    res.save_to_disk(args.out)
    print(f"wrote {args.out}; every stored answer recomputed from the metadata")
    return 0


if __name__ == "__main__":
    sys.exit(main())
