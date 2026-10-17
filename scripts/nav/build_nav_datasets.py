"""Turn rendered HM3D episodes (scripts/nav/render_hm3d.py output) into PISCO training rows.

Two products, both from the per-frame semantic ground truth:
1. nav PRETRAIN rows {image, text}: one row per frame, text = templated scene description from the GT
   objects ("a kitchen view with a window on the left, a dining chair and a dining table on the right, …").
   Used by MultimodalPretrainingCollator (image-AE / image-continuation) so the compressor learns frame→text.
2. nav MEMORY-QA rows {docs, doc_images, query, mistral_label, answers, source, ...}: a window of
   --frames frames from one episode (evenly subsampled) + a question about the walk, answered from GT:
     presence   "Did you see a X during the walk? Answer yes or no."
     order      "Which did you see first, the X or the Y?"
     position   "When you first saw the X, was it on your left, in the center, or on your right?"
     final      "What was the most prominent object in view at the end of the walk?"
     room-ish   "How many different kinds of furniture did you see? …" (skipped: too ambiguous)
   Used by MultimodalFineTuningCollator (docs = frames, 'Document j:' = frame j, image_compr_rate 16 → 25 MEM/frame).
   Same schema as the doc-QA sets, so eval_mm_docqa.py / judge / show_mm_results.py work unchanged.

  python scripts/nav/build_nav_datasets.py --episodes 'data/nav_hm3d_train/shard_*/*' --out_pretrain data/nav_pretrain \
      --out_qa data/nav_ft_train --frames 8 --qa_per_episode 6
"""

import argparse
import glob
import json
import os
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from mm_data_utils import ShardWriter, image_bytes_feature  # noqa: E402

FURNITURE_HINT = {"chair", "table", "couch", "sofa", "bed", "cabinet", "shelf", "desk", "stool", "bench", "dresser"}


def pos_phrase(p):
    return {"left": "on the left", "center": "in the center", "right": "on the right"}[p]


def describe(objects, k=6):
    objs = objects[:k]
    if not objs:
        return "An empty room with bare walls and floor."
    groups = {}
    for o in objs:
        groups.setdefault(o["pos"], []).append(o["category"])
    parts = []
    for p in ("left", "center", "right"):
        if p in groups:
            names = groups[p]
            lst = ", ".join(names[:-1]) + (" and " if len(names) > 1 else "") + names[-1]
            parts.append(f"{lst} {pos_phrase(p)}")
    return "A view of a room with " + "; ".join(parts) + "."


def first_seen(frames, cat, min_area=0.0):
    for i, f in enumerate(frames):
        for o in f["objects"]:
            if o["category"] == cat and o["area"] >= min_area:
                return i, o
    return None, None


def episode_questions(frames, scene_cats, rng, n, min_area=0.02):
    """Questions only about objects that are clearly visible (>= min_area of a frame) somewhere in the
    window; 'No' answers only for categories absent from EVERY frame at any size. v2: no 'final' question
    (GT pixel area != perceived prominence: the uncompressed model scored .12 on it)."""
    seen = []
    for f in frames:
        for o in f["objects"]:
            if o["area"] >= min_area and o["category"] not in seen:
                seen.append(o["category"])
    seen_any = {o["category"] for f in frames for o in f["objects"]}
    unseen = [c for c in scene_cats if c not in seen_any]
    qs = []
    # presence (balanced)
    for _ in range(max(1, n // 3)):
        if seen and rng.random() < 0.5:
            c = rng.choice(seen); qs.append((f"Did you see a {c} during the walk? Answer yes or no.", "Yes", "presence"))
        elif unseen:
            c = rng.choice(unseen); qs.append((f"Did you see a {c} during the walk? Answer yes or no.", "No", "presence"))
    # order
    if len(seen) >= 2:
        for _ in range(max(1, n // 4)):
            a, b = rng.sample(seen, 2)
            ia, _ = first_seen(frames, a, min_area); ib, _ = first_seen(frames, b, min_area)
            if ia == ib:
                continue
            first = a if ia < ib else b
            qs.append((f"Which did you see first, the {a} or the {b}?", first, "order"))
    # position at first sight (single-word answer so EM/F1 are meaningful)
    for c in rng.sample(seen, min(len(seen), max(1, n // 4))):
        i, o = first_seen(frames, c, min_area)
        qs.append((f"When you first saw the {c}, was it on your left, in the center, or on your right? "
                   f"Answer with left, center or right.", o["pos"], "position"))
    rng.shuffle(qs)
    return qs[:n]


def qa_features():
    import datasets
    return datasets.Features({
        "docs": datasets.Sequence(datasets.Value("string")),
        "doc_images": datasets.Sequence(datasets.Image()),
        "query": datasets.Value("string"),
        "mistral_label": datasets.Value("string"),
        "answers": datasets.Sequence(datasets.Value("string")),
        "gold_positions": datasets.Sequence(datasets.Value("int32")),
        "source": datasets.Value("string"),
        "query_id": datasets.Value("string"),
    })


def pt_features():
    import datasets
    return datasets.Features({"image": datasets.Image(), "text": datasets.Value("string"),
                              "title": datasets.Value("string"), "docid": datasets.Value("string"),
                              "source": datasets.Value("string")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True, help="glob of episode dirs (each with meta.json)")
    ap.add_argument("--out_pretrain", default=None)
    ap.add_argument("--out_qa", default=None)
    ap.add_argument("--frames", type=int, default=8, help="frames per QA window")
    ap.add_argument("--qa_per_episode", type=int, default=6)
    ap.add_argument("--windows_per_episode", type=int, default=2)
    ap.add_argument("--pretrain_stride", type=int, default=2, help="keep every k-th frame for pretrain rows")
    ap.add_argument("--min_area", type=float, default=0.02, help="min frame share for an object to be asked about")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    eps = sorted(d for d in glob.glob(args.episodes) if os.path.exists(os.path.join(d, "meta.json")))
    print(f"[nav] {len(eps)} episodes")
    # scene-level category pools (for 'No' answers): union over the scene's episodes
    scene_cats = {}
    metas = {}
    for d in eps:
        m = json.load(open(os.path.join(d, "meta.json"))); metas[d] = m
        pool = scene_cats.setdefault(m["scene"], set())
        for f in m["frames"]:
            pool.update(o["category"] for o in f["objects"])

    wp = ShardWriter(args.out_pretrain, pt_features(), shard_size=5000) if args.out_pretrain else None
    wq = ShardWriter(args.out_qa, qa_features(), shard_size=1000) if args.out_qa else None
    n_pt = n_qa = 0
    for d in eps:
        m = metas[d]; frames = m["frames"]
        imgs = [open(os.path.join(d, f["file"]), "rb").read() for f in frames]
        if wp:
            for i in range(0, len(frames), args.pretrain_stride):
                wp.add({"image": image_bytes_feature(imgs[i]), "text": describe(frames[i]["objects"]),
                        "title": m["episode"], "docid": f"{m['episode']}-{i:03d}", "source": "hm3d_nav"}); n_pt += 1
        if wq:
            for w in range(args.windows_per_episode):
                L = len(frames)
                span = rng.randint(max(args.frames, L // 2), L)       # window covers half to all of the walk
                start = rng.randint(0, L - span)
                idx = [start + round(j * (span - 1) / (args.frames - 1)) for j in range(args.frames)]
                win = [frames[i] for i in idx]
                for k, (q, a, kind) in enumerate(episode_questions(win, sorted(scene_cats[m["scene"]]), rng, args.qa_per_episode, args.min_area)):
                    wq.add({"docs": [""] * len(idx), "doc_images": [image_bytes_feature(imgs[i]) for i in idx],
                            "query": q, "mistral_label": a, "answers": [a], "gold_positions": [],
                            "source": f"nav_{kind}", "query_id": f"{m['episode']}-w{w}-q{k}"}); n_qa += 1
    if wp:
        wp.finalize()
    if wq:
        wq.finalize()
    print(f"[nav] pretrain rows {n_pt}, qa rows {n_qa}")


if __name__ == "__main__":
    main()
