"""The dense index of the LVBench videos, built once: <MEM> embeddings (decoder space, after the connector) of EVERY
frame at 1 frame / 5 s, plus CLIP ViT-L/14-336 frame features, plus CLIP's top minutes per question (the retriever
picks the candidates; the compressed model only reads).
  --step mems   per video <vid>.pt {"mems": bf16 [sum_n, H], "offsets": long [n_frames+1]} and <vid>_clip.pt fp16 [n, 768]
                (shard by video with --shard/--nshards)
  --step cands  cands.json {question uid or EWB qid: [minutes ranked by CLIP, top 10]}; dense = max over the minute's
                12 frames of the frame-query cosine; LVBench questions use the stem (options removed)

  python scripts/video/cache_mems.py --step mems --checkpoint_path expQ/ft_mm_4b_174908/model --image_compr_rate 16 \
      --shard 0 --nshards 4
  python scripts/video/cache_mems.py --step cands
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from route_windows import clip_frame_feats, clip_text_feats, compress_frames, load_clip, load_pisco  # noqa: E402
from zoom_route import FPM  # noqa: E402

CACHE = "data/video/lvbench_cache"


def load_cache(cache_dir, vid, device):
    """{frame index: <MEM> embeddings} view over the cached tensor of one video."""
    d = torch.load(f"{cache_dir}/{vid}.pt", map_location=device)
    mems, offs = d["mems"], d["offsets"].tolist()
    return {k: mems[a:z] for k, (a, z) in enumerate(zip(offs[:-1], offs[1:]))}


def step_mems(args, device):
    n_frames = json.load(open(f"{args.frames_dir}/frames.json"))
    vids = sorted(n_frames)[args.shard::args.nshards][: args.limit]
    out = pathlib.Path(args.cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    todo = [v for v in vids if not (out / f"{v}.pt").exists() or not (out / f"{v}_clip.pt").exists()]
    print(f"[cache] shard {args.shard}/{args.nshards}: {len(vids)} videos, {len(todo)} to do", flush=True)
    if not todo:
        return
    model, coll = load_pisco(args, device)
    clip, _ = load_clip(device)
    for v in todo:
        t = time.perf_counter()
        ks = list(range(n_frames[v]))
        if not (out / f"{v}.pt").exists():
            emb = compress_frames(model, coll, args.frames_dir, v, ks, args.batch, device)
            sizes = [emb[k].shape[0] for k in ks]
            torch.save({"mems": torch.cat([emb[k] for k in ks]).to(torch.bfloat16).cpu(),
                        "offsets": torch.tensor(np.cumsum([0] + sizes), dtype=torch.long),
                        "image_compr_rate": args.image_compr_rate, "checkpoint": args.checkpoint_path},
                       out / f"{v}.pt.tmp")
            (out / f"{v}.pt.tmp").rename(out / f"{v}.pt")
            del emb
        if not (out / f"{v}_clip.pt").exists():
            feats = clip_frame_feats(clip, args.frames_dir, v, ks, args.batch, device)
            torch.save(torch.stack([feats[k] for k in ks]).to(torch.float16).cpu(), out / f"{v}_clip.pt")
        print(f"[cache] {v}: {len(ks)} frames  {time.perf_counter() - t:.0f}s", flush=True)


def step_cands(args, device):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from mcq_levels import load_questions
    n_frames = json.load(open(f"{args.frames_dir}/frames.json"))
    qs, n_min = load_questions(f"{args.frames_dir}/frames.json")
    ewb = [r for r in json.load(open(args.ewb)) if r["video_id"] in n_frames]
    clip, tok = load_clip(device)
    cands = {}
    for v in sorted(n_frames):
        feats = torch.load(f"{args.cache_dir}/{v}_clip.pt", map_location=device).float()   # [n, 768]
        nm = n_min[v]
        items = [(q["uid"], q["stem"]) for q in qs if q["video_id"] == v] + \
                [(r["qid"], r["question"]) for r in ewb if r["video_id"] == v]
        if not items:
            continue
        txt = clip_text_feats(clip, tok, [t for _, t in items], device)   # [m, 768]
        sims = (txt @ feats.T)[:, : nm * FPM].reshape(len(items), nm, FPM).amax(-1)   # dense: max over the minute
        for (uid, _), row in zip(items, sims):
            cands[uid] = [int(j) for j in torch.argsort(row, descending=True)[:10].tolist()]
    json.dump(cands, open(f"{args.cache_dir}/cands.json", "w"))
    # sanity: EWB recall of CLIP's top-3 on these candidates
    hit = [any(m * 60 < r["correct_interval"][1] and (m + 1) * 60 > r["correct_interval"][0] for m in cands[r["qid"]][:3])
           for r in ewb if r["qid"] in cands]
    print(f"[cands] {len(cands)} queries; EWB CLIP recall@3 over the whole video: {np.mean(hit):.3f} (n={len(hit)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["mems", "cands"], required=True)
    ap.add_argument("--frames_dir", default="data/video/lvbench_frames")
    ap.add_argument("--cache_dir", default=CACHE)
    ap.add_argument("--ewb", default="expQ/video/ewb_test.json")
    ap.add_argument("--checkpoint_path", default=None)
    ap.add_argument("--image_compr_rate", type=int, default=None)
    ap.add_argument("--image_max_pixels", type=int, default=451584)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    (step_mems if args.step == "mems" else step_cands)(args, "cuda")
    return 0


if __name__ == "__main__":
    sys.exit(main())
