"""Compression cost per image, split into vision tower and language model, by compressor size.

Timing only: compute depends on shapes, not on trained weights, so plain Qwen3.5 checkpoints stand in for the PISCO
compressor. Input = one real eval page at each pixel budget: <|vision_start|> <|image_pad|>*N <|vision_end|> followed
by N//rate+1 placeholder <MEM> slots, the compressor forward as PISCO.compress runs it (logits_to_keep=1, no cache).

  python scripts/bench_compress_cost.py --models Qwen/Qwen3.5-0.8B-Base,Qwen/Qwen3.5-2B,Qwen/Qwen3.5-4B \
      --data eval_data/mm_docqa_1500_1m --out expQ/archq/bench/compress_cost.json
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pisco.mm_utils import image_to_span, load_image_processor  # noqa: E402


def timed(fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Qwen/Qwen3.5-0.8B-Base,Qwen/Qwen3.5-2B,Qwen/Qwen3.5-4B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--pixels", default="1003520,451584")
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--rate", type=int, default=8)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import AutoModelForImageTextToText
    import datasets
    ds = datasets.load_from_disk(args.data)
    ds = ds["train"] if isinstance(ds, datasets.DatasetDict) else ds
    page = ds[0]["doc_images"][0]
    res = []
    for name in args.models.split(","):
        model = AutoModelForImageTextToText.from_pretrained(name, torch_dtype=torch.bfloat16,
                                                            attn_implementation="sdpa").cuda().eval()
        cfg = model.config
        vids = (int(cfg.vision_start_token_id), int(cfg.image_token_id), int(cfg.vision_end_token_id))
        n_vis = sum(p.numel() for n, p in model.named_parameters() if ".visual." in n or n.startswith("visual."))
        n_all = sum(p.numel() for p in model.parameters())
        for px in [int(x) for x in args.pixels.split(",")]:
            proc = load_image_processor(name, max_pixels=px)
            span, pv, grid = image_to_span(page, proc, vids)
            n_mem = (len(span) - 2) // args.rate + 1
            ids1 = list(span) + [int(cfg.vision_end_token_id)] * n_mem   # placeholder <MEM> slots (shape only)
            for B in [int(b) for b in args.batches.split(",")]:
                ids = torch.tensor([ids1] * B, device="cuda")
                pvs = torch.from_numpy(np.concatenate([pv] * B, 0)).cuda()
                thw = torch.tensor([grid] * B, dtype=torch.long, device="cuda")
                tt = (ids == vids[1]).long()
                torch.cuda.reset_peak_memory_stats()
                t_vis = timed(lambda: model.model.get_image_features(pixel_values=pvs, image_grid_thw=thw), args.reps)
                t_all = timed(lambda: model(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pvs,
                                            image_grid_thw=thw, mm_token_type_ids=tt, logits_to_keep=1,
                                            use_cache=False, output_hidden_states=True), args.reps)
                r = {"model": name, "params_b": n_all / 1e9, "vision_params_b": n_vis / 1e9, "max_pixels": px,
                     "visual_tokens": len(span) - 2, "mems": n_mem, "batch": B,
                     "ms_per_image_total": 1e3 * t_all / B, "ms_per_image_vision": 1e3 * t_vis / B,
                     "ms_per_image_lm": 1e3 * (t_all - t_vis) / B, "images_per_s": B / t_all,
                     "peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
                res.append(r)
                print(json.dumps(r), flush=True)
        del model
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"gpu": torch.cuda.get_device_name(0), "results": res}, open(args.out, "w"), indent=1)
    print("| model | params (B) | vision (B) | pixels | visual tok | batch | ms/image | vision ms | LM ms | images/s |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in res:
        print(f"| {r['model'].split('/')[-1]} | {r['params_b']:.2f} | {r['vision_params_b']:.2f} | {r['max_pixels']} "
              f"| {r['visual_tokens']} | {r['batch']} | {r['ms_per_image_total']:.1f} | {r['ms_per_image_vision']:.1f} "
              f"| {r['ms_per_image_lm']:.1f} | {r['images_per_s']:.1f} |")


if __name__ == "__main__":
    main()
