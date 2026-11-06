"""Whole-video zoom test (ExtremeWhenBench; the hierarchy with no training). Level 0: every minute of the video is one
document, shown by `--l0_frames` of its 12 frames (default 2, at 15 s and 45 s); the model ranks ALL minutes from
their <MEM>s (route_logprobs) and keeps its top `--top`. Level 1: those minutes with all 12 frames; the model picks
one. A minute is right when it overlaps the event's interval. --scorer clip ranks every minute by the paper's
retriever (max frame-query cosine over the minute's 12 frames = dense, or over the level-0 frames only = sparse) and
writes its top minutes; give that file to a pisco run with --clip_ranking and the model's level 1 also zooms into
CLIP's top minutes (CLIP searches, the model zooms). Arm "both" (default on): both levels in place, every minute
coarse and the top minutes dense, the pick over ALL minutes (final_both_all; can recover from a wrong level 0) and
restricted to the top minutes (final_both_top). Queries: the 500 of scripts/video/build_ewb_windows.py.

  python scripts/video/zoom_route.py --scorer clip --out expQ/video/results/zoom_clip.json
  python scripts/video/zoom_route.py --scorer pisco --checkpoint_path expQ/ft_mm_4b_174908/model --image_compr_rate 16 \
      --clip_ranking expQ/video/results/zoom_clip.json --out expQ/video/results/zoom_174908_r16.json
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from route_windows import (by_video, clip_frame_feats, clip_text_feats, compress_frames, load_clip,  # noqa: E402
                           load_pisco, route_logprobs)

FPM = 12   # frames per minute (frame k = t 5k s)


def right(m, interval):
    s, e = interval
    return m * 60 < e and (m + 1) * 60 > s


def best_rank(scores, interval):
    order = sorted(range(len(scores)), key=lambda j: -scores[j])
    return min(r for r, m in enumerate(order) if right(m, interval))


def l0_offsets(n):
    return [round((i + 0.5) * FPM / n - 0.5) for i in range(n)]   # 2 -> frames 3 and 9 (t 15 s, 45 s)


def pisco_run(args, items, n_min, device):
    model, coll = load_pisco(args, device)
    mem = model.decoder_tokenizer.mem_token
    clip_top = {}
    if args.clip_ranking:
        clip_top = {r["qid"]: r["top"] for r in json.load(open(args.clip_ranking))["records"]}
    offs = l0_offsets(args.l0_frames)
    for vid, qs in by_video(items).items():
        nm, t = n_min[vid], time.perf_counter()
        emb = compress_frames(model, coll, args.frames_dir, vid, [m * FPM + o for m in range(nm) for o in offs],
                              args.batch, device)
        t_index = time.perf_counter() - t

        def route(cands, frames_of, question):
            ks = [k for m in cands for k in frames_of(m)]
            compress_frames(model, coll, args.frames_dir, vid, ks, args.batch, device, emb)
            bg = "".join(f"Document {j}:" + mem * sum(emb[k].shape[0] for k in frames_of(m)) for j, m in enumerate(cands))
            with torch.inference_mode():
                return route_logprobs(model, bg, torch.cat([emb[k] for k in ks]), question, len(cands), device)

        dense = lambda m: range(m * FPM, m * FPM + FPM)
        for q in qs:
            question = coll.clean_text(q["question"])
            t = time.perf_counter()
            lp0 = route(range(nm), lambda m: [m * FPM + o for o in offs], question)
            t0 = time.perf_counter() - t
            top = [int(j) for j in np.argsort(lp0)[::-1][: args.top]]
            lp1 = route(top, dense, question)
            rec = {"lp0": lp0, "top": top, "lp1": lp1, "final": top[int(np.argmax(lp1))], "l0_s": t0,
                   "index_s": t_index, "l0_mems": sum(emb[m * FPM + o].shape[0] for m in range(nm) for o in offs)}
            if not args.no_both:   # both levels in place: every minute coarse, the top minutes dense, pick over all
                tops = set(top)
                t = time.perf_counter()
                lpb = route(range(nm), lambda m: dense(m) if m in tops else [m * FPM + o for o in offs], question)
                rec.update(lp_both=lpb, final_both_all=int(np.argmax(lpb)),
                           final_both_top=top[int(np.argmax([lpb[m] for m in top]))], both_s=time.perf_counter() - t)
            if q["qid"] in clip_top:
                ct = clip_top[q["qid"]][: args.top]
                lph = route(ct, dense, question)
                rec.update(clip_top=ct, lp_hybrid=lph, final_hybrid=ct[int(np.argmax(lph))])
            yield q, rec


def clip_run(args, items, n_min, device):
    model, tok = load_clip(device)
    offs = l0_offsets(args.l0_frames)
    for vid, qs in by_video(items).items():
        nm = n_min[vid]
        feats = clip_frame_feats(model, args.frames_dir, vid, list(range(nm * FPM)), args.batch, device)
        txt = clip_text_feats(model, tok, [q["question"] for q in qs], device)
        for q, tv in zip(qs, txt):
            fs = [float(feats[k] @ tv) for k in range(nm * FPM)]
            dense = [max(fs[m * FPM:(m + 1) * FPM]) for m in range(nm)]
            sparse = [max(fs[m * FPM + o] for o in offs) for m in range(nm)]
            yield q, {"dense": dense, "sparse": sparse, "top": [int(j) for j in np.argsort(dense)[::-1][:10]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", choices=["pisco", "clip"], required=True)
    ap.add_argument("--windows", default="expQ/video/ewb_lvbench_w10.json", help="the queries (their windows unused)")
    ap.add_argument("--frames_dir", default="data/video/lvbench_frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint_path", default=None)
    ap.add_argument("--image_compr_rate", type=int, default=None)
    ap.add_argument("--image_max_pixels", type=int, default=451584)
    ap.add_argument("--l0_frames", type=int, default=2)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--clip_ranking", default=None)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_both", action="store_true", help="skip the both-levels-in-place arm")
    args = ap.parse_args()

    items = json.load(open(args.windows))[: args.limit]
    n_min = {v: n // FPM for v, n in json.load(open(f"{args.frames_dir}/frames.json")).items()}
    gen = pisco_run if args.scorer == "pisco" else clip_run
    recs, t0 = [], time.time()
    for q, r in gen(args, items, n_min, "cuda"):
        iv, nm = q["interval"], n_min[q["video_id"]]
        r.update(qid=q["qid"], video_id=q["video_id"], category=q["category"], interval=iv, n_minutes=nm,
                 chance=sum(right(m, iv) for m in range(nm)) / nm)
        if args.scorer == "pisco":
            r.update(l0_rank=best_rank(r["lp0"], iv), hit=right(r["final"], iv))
            if "final_hybrid" in r:
                r["hit_hybrid"] = right(r["final_hybrid"], iv)
            if "lp_both" in r:
                r.update(hit_both_all=right(r["final_both_all"], iv), hit_both_top=right(r["final_both_top"], iv),
                         both_rank=best_rank(r["lp_both"], iv))
        else:
            r.update(rank_dense=best_rank(r["dense"], iv), rank_sparse=best_rank(r["sparse"], iv))
        recs.append(r)
        if len(recs) % 25 == 0:
            key = "l0_rank" if args.scorer == "pisco" else "rank_dense"
            print(f"[{len(recs)}/{len(items)}] top-{args.top} {np.mean([x[key] < args.top for x in recs]):.3f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    m = lambda f: float(np.mean([f(x) for x in recs]))
    summ = {"n": len(recs), "scorer": args.scorer, "checkpoint": args.checkpoint_path, "rate": args.image_compr_rate,
            "l0_frames": args.l0_frames, "top": args.top, "chance@1": m(lambda x: x["chance"]),
            "mean_minutes": m(lambda x: x["n_minutes"])}
    for k in (1, 3, 5, 10):
        if args.scorer == "pisco":
            summ[f"l0_recall@{k}"] = m(lambda x: x["l0_rank"] < k)
        else:
            summ[f"clip_dense_recall@{k}"] = m(lambda x: x["rank_dense"] < k)
            summ[f"clip_sparse_recall@{k}"] = m(lambda x: x["rank_sparse"] < k)
    if args.scorer == "pisco":
        summ.update({"zoom_recall@1": m(lambda x: x["hit"]), "l0_mems_mean": m(lambda x: x["l0_mems"]),
                     "l0_route_s_mean": m(lambda x: x["l0_s"])})
        if recs and "hit_hybrid" in recs[0]:
            summ["clip_then_model_zoom_recall@1"] = m(lambda x: x["hit_hybrid"])
        if recs and "lp_both" in recs[0]:
            summ.update({"both_all_recall@1": m(lambda x: x["hit_both_all"]),
                         "both_top_recall@1": m(lambda x: x["hit_both_top"]),
                         "both_recall@3": m(lambda x: x["both_rank"] < 3), "both_s_mean": m(lambda x: x["both_s"])})
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summ, "records": recs}, open(args.out, "w"))
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
