"""Search-only test of route-then-expand on hour-long video (ExtremeWhenBench step V1). Per query: 10 one-minute
windows of one video (scripts/video/build_ewb_windows.py), 12 frames each. Every window is one document,
"Document j:" + the <MEM>s of its frames, and the model's own routing scores rank the windows
(eval_mm_docqa.route_logprobs, the call that routes pages and text). --scorer clip ranks the same windows by the
paper's retriever: CLIP ViT-L/14-336, max over the window's frames of the frame-query cosine. Frames are compressed
once per video and reused by all its queries (an offline index). Writes per-query scores + recall@1/@3.

  python scripts/video/route_windows.py --scorer pisco --checkpoint_path expQ/ft_mm_4b_174908/model \
      --image_compr_rate 16 --windows expQ/video/ewb_lvbench_w10.json --frames_dir data/video/lvbench_frames \
      --out expQ/video/route_174908_r16.json
"""

import argparse
import json
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from eval_mm_docqa import make_collator, route_logprobs  # noqa: E402
from pisco.mm_utils import MMChunkBatch, collate_compressor  # noqa: E402

CLIP = "openai/clip-vit-large-patch14-336"
CLIP_MEAN, CLIP_STD = np.array([0.48145466, 0.4578275, 0.40821073]), np.array([0.26862954, 0.26130258, 0.27577711])


def frame(frames_dir, vid, k):
    return Image.open(f"{frames_dir}/{vid}/{k + 1:05d}.jpg").convert("RGB")


def window_frames(q):
    f = q["frames_per_window"]
    return [list(range(m * f, m * f + f)) for m in q["windows"]]


def by_video(items):
    g = defaultdict(list)
    for q in items:
        g[q["video_id"]].append(q)
    return g


def load_pisco(args, device):
    from pisco.model import PISCO
    kw = {"image_max_pixels": args.image_max_pixels} if args.image_max_pixels else {}
    model = PISCO.from_pretrained(args.checkpoint_path, **kw).to(device).eval()
    coll = make_collator(model, topk=1, compressor_max_length=256, image_compr_rate=args.image_compr_rate,
                         query_dependent=False)
    return model, coll


@torch.inference_mode()
def compress_frames(model, coll, frames_dir, vid, ks, batch, device, emb=None):
    """{frame index: its <MEM> embeddings} for the frames `ks` of video `vid` (frames already in `emb` are skipped)."""
    emb = {} if emb is None else emb
    ks = [k for k in ks if k not in emb]
    for i in range(0, len(ks), batch):
        part, full, sizes = ks[i:i + batch], MMChunkBatch(), []
        for k in part:
            b = coll.example_chunks([" "], [frame(frames_dir, vid, k)], None)[0]
            full.extend(b)
            sizes.append(b.total_mems)
        comp = {n: v.to(device) for n, v in collate_compressor([full], coll, coll.vision_ids).items()}
        e = model.compress(comp["compressor_input_ids"], comp["compressor_attention_mask"],
                           pixel_values=comp.get("pixel_values"), image_grid_thw=comp.get("image_grid_thw"),
                           mm_token_type_ids=comp.get("mm_token_type_ids"))
        offs = np.cumsum([0] + sizes)
        for k, a, z in zip(part, offs[:-1], offs[1:]):
            emb[k] = e[a:z]
    return emb


def load_clip(device):
    from transformers import AutoTokenizer, CLIPModel
    model = CLIPModel.from_pretrained(CLIP, dtype=torch.float16).to(device).eval()
    return model, AutoTokenizer.from_pretrained(CLIP)


def clip_pixels(im, size):
    """CLIP preprocessing (shortest side -> size, bicubic, centre crop, normalise), without torchvision."""
    w, h = im.size
    s = size / min(w, h)
    im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = im.size
    im = im.crop(((w - size) // 2, (h - size) // 2, (w - size) // 2 + size, (h - size) // 2 + size))
    return ((np.asarray(im, dtype=np.float32) / 255 - CLIP_MEAN) / CLIP_STD).transpose(2, 0, 1)


@torch.inference_mode()
def clip_frame_feats(model, frames_dir, vid, ks, batch, device):
    size, feats = model.config.vision_config.image_size, {}
    for i in range(0, len(ks), batch):
        part = ks[i:i + batch]
        pv = torch.tensor(np.stack([clip_pixels(frame(frames_dir, vid, k), size) for k in part]), dtype=torch.float16,
                          device=device)
        f = model.visual_projection(model.vision_model(pixel_values=pv).pooler_output)
        for k, x in zip(part, torch.nn.functional.normalize(f.float(), dim=-1)):
            feats[k] = x
    return feats


@torch.inference_mode()
def clip_text_feats(model, tok, texts, device):
    enc = tok(texts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
    return torch.nn.functional.normalize(model.text_projection(model.text_model(**enc).pooler_output).float(), dim=-1)


def pisco_scores(args, items, device):
    model, coll = load_pisco(args, device)
    mem = model.decoder_tokenizer.mem_token
    for vid, qs in by_video(items).items():
        t = time.perf_counter()
        ks = sorted({k for q in qs for w in window_frames(q) for k in w})
        emb = compress_frames(model, coll, args.frames_dir, vid, ks, args.batch, device)
        t_index = time.perf_counter() - t
        for q in qs:
            wins = window_frames(q)
            bg = "".join(f"Document {j}:" + mem * sum(emb[k].shape[0] for k in w) for j, w in enumerate(wins))
            t = time.perf_counter()
            lp = route_logprobs(model, bg, torch.cat([emb[k] for w in wins for k in w]), coll.clean_text(q["question"]),
                                len(wins), device)
            yield q, lp, {"mems": sum(emb[k].shape[0] for w in wins for k in w), "route_s": time.perf_counter() - t,
                          "index_s_per_frame": t_index / len(ks)}


def clip_scores(args, items, device):
    model, tok = load_clip(device)
    for vid, qs in by_video(items).items():
        ks = sorted({k for q in qs for w in window_frames(q) for k in w})
        feats = clip_frame_feats(model, args.frames_dir, vid, ks, args.batch, device)
        txt = clip_text_feats(model, tok, [q["question"] for q in qs], device)
        for q, tv in zip(qs, txt):
            yield q, [max(float(feats[k] @ tv) for k in w) for w in window_frames(q)], {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", choices=["pisco", "clip"], required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint_path", default=None)
    ap.add_argument("--image_compr_rate", type=int, default=None, help="default: the checkpoint's")
    ap.add_argument("--image_max_pixels", type=int, default=451584)
    ap.add_argument("--batch", type=int, default=24, help="frames per compressor / CLIP forward")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    items = json.load(open(args.windows))[: args.limit]
    gen = pisco_scores if args.scorer == "pisco" else clip_scores
    recs, t0 = [], time.time()
    for q, sc, st in gen(args, items, "cuda"):
        order = sorted(range(len(sc)), key=lambda j: -sc[j])
        rank = min(order.index(g) for g in q["gold"])
        recs.append({"qid": q["qid"], "video_id": q["video_id"], "category": q["category"], "gold": q["gold"],
                     "scores": sc, "rank": rank, **st})
        if len(recs) % 50 == 0:
            print(f"[{len(recs)}/{len(items)}] recall@1 {np.mean([r['rank'] < 1 for r in recs]):.3f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    summ = {f"recall@{k}": float(np.mean([r["rank"] < k for r in recs])) for k in (1, 3)}
    for c in sorted({r["category"] for r in recs}):
        rc = [r for r in recs if r["category"] == c]
        summ[f"recall@1/{c}"] = float(np.mean([r["rank"] < 1 for r in rc]))
        summ[f"n/{c}"] = len(rc)
    summ.update(n=len(recs), scorer=args.scorer, checkpoint=args.checkpoint_path, image_compr_rate=args.image_compr_rate,
                image_max_pixels=args.image_max_pixels, windows=args.windows)
    if recs and "mems" in recs[0]:
        summ.update(mean_mems=float(np.mean([r["mems"] for r in recs])),
                    route_s_mean=float(np.mean([r["route_s"] for r in recs])))
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summ, "records": recs}, open(args.out, "w"))
    print(json.dumps({k: v for k, v in summ.items() if not k.startswith("n/")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
