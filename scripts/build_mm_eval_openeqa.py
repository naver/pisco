"""Build OpenEQA EM-EQA eval sets (HM3D half: 557 human-written questions over 63 house tours) for
scripts/eval_mm_docqa.py, in the build_mm_eval.py schema: docs (empty strings), doc_images (the tour frames),
query, mistral_label (the reference answer), answers (reference + extra_answers), gold_positions (none: OpenEQA
has no frame labels), source (the question category), query_id (OpenEQA question_id).

One set per k: the k frames evenly spaced over the tour, first and last included (OpenEQA's own sampling rule,
the same `uniform_indices` as the silver-nav campaign_eqa `uniform` arm), or the whole tour (k = all). With
`--frames_from`, the k frames are instead the first k of a RETRIEVED ranking per question (the RAG setting:
retriever picks, compressor compresses): either a campaign_eqa cell json (rows[].frames, the Qwen3-VL-Embedding-2B
top-15 of the `bank` arm) or a {question_id: [frame stems, best first]} json (scripts/openeqa_retrieve.py). Frames are
resized once per tour to the pixel budget and stored as JPEG, like every other mm eval set. A sidecar
<out>/frames.json records, per question, the tour, its length and the frame indices used.

  python scripts/build_mm_eval_openeqa.py --k 16,32,64,all --out 'eval_data/openeqa_hm3d_k{k}'

Data: /beegfs/scratch/user/hdejean/open-eqa (questions + frames; the frames come from the HF mirror
the-Lin/openeqa hm3d-v0.zip, fetched by openeqa_fetch.sh). Regime label for every number: "OpenEQA EM-EQA HM3D
557 Q, memory only" (no walking).
"""

import argparse
import json
import os
import pathlib
import sys
from collections import Counter
from multiprocessing import Pool

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_mm_finetune import ft_features  # noqa: E402
from mm_data_utils import ShardWriter, image_bytes_feature, prepare_page  # noqa: E402

ROOT = "/beegfs/scratch/user/hdejean/open-eqa"
DATASET = f"{ROOT}/data/open-eqa-v0.json"
FRAME_ROOT = f"{ROOT}/data/frames/hm3d-v0"
IMG_EXT = (".jpg", ".jpeg", ".png")


def load_questions(path=DATASET):
    data = json.load(open(path))
    return [d for d in data if str(d.get("episode_history", "")).startswith("hm3d-v0/")]


def frame_paths(episode_dir):
    """The tour's RGB frames in order. OpenEQA names them <step>-rgb.png/jpg; other images in the dir (depth,
    semantic) are left out when an -rgb set exists."""
    fs = sorted(f for f in os.listdir(episode_dir) if f.lower().endswith(IMG_EXT))
    rgb = [f for f in fs if "-rgb" in f]
    fs = rgb or fs
    return [os.path.join(episode_dir, f) for f in fs]


def uniform_indices(n_frames, k):
    """k frame indices evenly spaced over the tour, first and last included (OpenEQA's rule; the campaign's)."""
    if k is None or k >= n_frames:
        return list(range(n_frames))
    import numpy as np
    return [int(round(x)) for x in np.linspace(0, n_frames - 1, k)]


def load_ranking(path):
    """{question_id: [frame stems, best first]} from a campaign_eqa cell json (rows[].frames are paths, .jpg there,
    .png here: matched by stem) or from a ranking json written by scripts/openeqa_retrieve.py."""
    d = json.load(open(path))
    if isinstance(d, dict) and "rows" in d:
        return {r["question_id"]: [os.path.splitext(os.path.basename(p))[0] for p in r["frames"]] for r in d["rows"]}
    d = d.get("ranking", d)
    return {q: [os.path.splitext(os.path.basename(p))[0] for p in v] for q, v in d.items()}


def _prep(args):
    path, max_pixels, min_pixels = args
    return prepare_page(open(path, "rb").read(), max_pixels=max_pixels, min_pixels=min_pixels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", default="16,32,64,all", help="comma list of frame counts; 'all' = the whole tour")
    ap.add_argument("--out", default="eval_data/openeqa_hm3d_k{k}", help="output dir pattern, {k} replaced")
    ap.add_argument("--max_pixels", type=int, default=451584)
    ap.add_argument("--min_pixels", type=int, default=65536)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit_tours", type=int, default=0, help="first N tours only (smoke)")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--frame_root", default=FRAME_ROOT)
    ap.add_argument("--frames_from", default=None, help="ranked frames per question (campaign cell json or ranking json)")
    args = ap.parse_args()

    ks = [None if k.strip() == "all" else int(k) for k in args.k.split(",") if k.strip()]
    qs = load_questions(args.dataset)
    ranked = load_ranking(args.frames_from) if args.frames_from else None
    if ranked:
        assert all(k is not None for k in ks), "--frames_from needs explicit k values"
        missing = [q["question_id"] for q in qs if q["question_id"] not in ranked]
        assert not missing, f"{len(missing)} questions without a ranking, e.g. {missing[:3]}"
    by_tour = {}
    for q in qs:
        by_tour.setdefault(q["episode_history"].split("/", 1)[1], []).append(q)
    tours = sorted(by_tour)
    if args.limit_tours:
        tours = tours[: args.limit_tours]
    print(f"{len(qs)} questions over {len(by_tour)} tours; building k = {args.k} for {len(tours)} tours")

    writers = {k: ShardWriter(args.out.format(k="all" if k is None else k), ft_features(), shard_size=200) for k in ks}
    side = {k: {} for k in ks}
    lengths, cats, n_rows = [], Counter(), Counter()
    with Pool(args.workers) as pool:
        for ti, tour in enumerate(tours):
            paths = frame_paths(os.path.join(args.frame_root, tour))
            assert paths, f"no frames under {args.frame_root}/{tour}"
            lengths.append(len(paths))
            stem_idx = {os.path.splitext(os.path.basename(p))[0]: i for i, p in enumerate(paths)}
            if ranked:   # only the frames some question of this tour retrieves get resized
                need = sorted({stem_idx[st] for q in by_tour[tour] for st in ranked[q["question_id"]][: max(ks)]})
                prepped = dict(zip(need, pool.map(_prep, [(paths[i], args.max_pixels, args.min_pixels) for i in need], chunksize=4)))
                frames = [prepped.get(i) for i in range(len(paths))]
            else:
                frames = pool.map(_prep, [(p, args.max_pixels, args.min_pixels) for p in paths], chunksize=4)
            for k in ks:
                for q in by_tour[tour]:
                    if ranked:
                        idx = [stem_idx[st] for st in ranked[q["question_id"]][:k]]
                        assert len(idx) == k, f"{q['question_id']}: {len(idx)} ranked frames, need {k}"
                    else:
                        idx = uniform_indices(len(paths), k)
                    cats[q["category"]] += 1
                    writers[k].add({
                        "docs": [""] * len(idx),
                        "doc_images": [image_bytes_feature(frames[i]) for i in idx],
                        "query": q["question"],
                        "mistral_label": q["answer"],
                        "answers": [q["answer"]] + list(q.get("extra_answers") or []),
                        "gold_positions": [],
                        "source": q["category"],
                        "query_id": q["question_id"],
                    })
                    side[k][q["question_id"]] = {"tour": tour, "n_tour_frames": len(paths), "indices": idx,
                                                 "frames": [os.path.basename(paths[i]) for i in idx]}
                    n_rows[k] += 1
            kb = sum(len(b) for b in frames if b) / 1024 / max(1, sum(1 for b in frames if b))
            print(f"[{ti + 1}/{len(tours)}] {tour}: {len(paths)} frames, {kb:.0f} KB/frame, "
                  f"{len(by_tour[tour])} questions", flush=True)
    for k in ks:
        writers[k].finalize()
        out = pathlib.Path(args.out.format(k="all" if k is None else k))
        json.dump(side[k], open(out / "frames.json", "w"))
    mems16 = [(n * (448 // 16 + 1)) for n in lengths]
    print(f"tour length: min {min(lengths)} mean {sum(lengths) / len(lengths):.0f} max {max(lengths)} frames; "
          f"whole tour at rate 16 (29 <MEM>/frame): mean {sum(mems16) / len(mems16):.0f}, max {max(mems16)} <MEM>")
    print("questions per category (per k):", dict(cats.most_common()) if len(ks) == 1 else
          {c: n // len(ks) for c, n in cats.items()})
    print("rows per set:", {("all" if k is None else k): n for k, n in n_rows.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
