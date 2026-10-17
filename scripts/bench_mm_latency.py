"""Latency / throughput benchmark: PISCO (precomputed <MEM> embeddings) vs the uncompressed VLM reading raw images.

What the eval timings cannot show: at batch size 1 the generation loop dominates, so an 11x shorter prompt
barely moves the wall clock. Here we measure, for K images per query and batch size B:
  * prefill  — one forward pass over the prompt (= time to first token), median over reps
  * generate — full generate() with EXACTLY G new tokens (min_new_tokens == max_new_tokens), median
  * qps      — B / generate time
  * peak GPU memory
PISCO compression is done ONCE per image up front (as a RAG index would) and its per-image cost is reported
separately. The PISCO decoder LoRA is merged for a fair per-token cost.

  python scripts/bench_mm_latency.py --mode base  --checkpoint_path <pisco ckpt (for the decoder name)> --data <ds> --out x.json
  python scripts/bench_mm_latency.py --mode pisco --checkpoint_path <pisco ckpt> --data <ds> --image_compr_rate 16 --out y.json
"""
import argparse
import json
import pathlib
import statistics
import sys
import time

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from pisco.model import PISCO, PISCOConfig, merge_lora_inplace  # noqa: E402
from pisco.mm_utils import collate_compressor, get_vision_token_ids, load_image_processor, segments_to_chunks  # noqa: E402
from eval_mm_docqa import _Pad, _build_chat_prompt  # noqa: E402
from relabel_mm_teacher import build_prompt  # noqa: E402


def sync():
    torch.cuda.synchronize()


def timed(fn, reps):
    fn()  # warmup
    sync()
    ts = []
    for _ in range(reps):
        sync(); t = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def load_rows(path, n):
    import datasets
    ds = datasets.load_from_disk(path)
    if isinstance(ds, datasets.DatasetDict):
        ds = ds["train"]
    rows = [ds[i] for i in range(min(n, len(ds)))]
    pool = [img for r in rows for img in (r.get("doc_images") or []) if img is not None]
    return rows, pool


def query_images(pool, qi, K):
    return [pool[(qi * K + j) % len(pool)] for j in range(K)]


def forward_prefill(model, **kw):
    """Time to first token = generate exactly ONE token (prompt forward + one sampling step). Same code path as the
    full generate for both modes; a bare forward(use_cache=False) breaks Qwen3.5's hybrid attention on inputs_embeds."""
    return model.generate(**kw, do_sample=False, top_p=None, max_new_tokens=1, min_new_tokens=1)


def bench_base(args, rows, pool, device):
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    name = args.base_model_name or PISCOConfig.from_pretrained(args.checkpoint_path).decoder_model_name
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    proc = load_image_processor(name, max_pixels=args.image_max_pixels)
    cfg = model.config
    vision_ids = (int(cfg.vision_start_token_id), int(cfg.image_token_id), int(cfg.vision_end_token_id))
    eos = [i for i in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
           if isinstance(i, int) and i >= 0]

    def make_batch(K, B):
        enc_ids, pvs, grids = [], [], []
        for qi in range(B):
            imgs = query_images(pool, qi, K)
            prompt, pv, grid = build_prompt(tok, [""] * K, imgs, rows[qi % len(rows)]["query"], proc, vision_ids)
            enc_ids.append(tok(prompt, add_special_tokens=False)["input_ids"]); pvs += pv; grids += grid
        enc = tok.pad({"input_ids": enc_ids}, padding="longest", return_tensors="pt")
        ids = enc["input_ids"]
        return dict(input_ids=ids.to(device), attention_mask=enc["attention_mask"].to(device),
                    pixel_values=torch.from_numpy(np.concatenate(pvs, 0)).to(device),
                    image_grid_thw=torch.tensor(grids, dtype=torch.long, device=device),
                    mm_token_type_ids=(ids == vision_ids[1]).long().to(device)), int(ids.shape[1])

    def run(K, B):
        kw, L = make_batch(K, B)
        with torch.inference_mode():
            pre = timed(lambda: forward_prefill(model, **kw), args.reps)
            gen = timed(lambda: model.generate(**kw, do_sample=False, top_p=None, max_new_tokens=args.gen_tokens,
                                               min_new_tokens=args.gen_tokens, eos_token_id=eos), args.reps)
        return L, pre, gen

    return run, {"model": name, "offline_compress_ms_per_image": None}


def bench_pisco(args, rows, pool, device):
    model = PISCO.from_pretrained(args.checkpoint_path, image_max_pixels=args.image_max_pixels).to(device).eval()
    model.decoder = merge_lora_inplace(model.decoder)        # fold LoRA (transformers-integrated PEFT has no merge_and_unload)
    model.compressor = merge_lora_inplace(model.compressor)
    tok, dtok = model.compressor_tokenizer, model.decoder_tokenizer
    dtok.padding_side = "left"
    vision_ids = model.vision_token_ids or get_vision_token_ids(tok)
    pad = _Pad(tok, model.image_processor)

    # ---- offline: one compressor pass per image (the "index"), embeddings kept on GPU
    embs, ts = [], []
    with torch.inference_mode():
        for img in pool:
            b = segments_to_chunks([("image", img)], compressor_tokenizer=tok, image_processor=model.image_processor,
                                   compr_rate=model.compr_rate, image_compr_rate=args.image_compr_rate,
                                   text_chunker=lambda ids: [ids], vision_ids=vision_ids)
            comp = {k: v.to(device) for k, v in collate_compressor([b], pad, vision_ids).items()}
            sync(); t = time.perf_counter()
            e = model.compress(comp["compressor_input_ids"], comp["compressor_attention_mask"],
                               pixel_values=comp.get("pixel_values"), image_grid_thw=comp.get("image_grid_thw"),
                               mm_token_type_ids=comp.get("mm_token_type_ids"))
            sync(); ts.append(time.perf_counter() - t)
            embs.append(e.detach())
    offline_ms = 1000 * statistics.median(ts[1:] if len(ts) > 1 else ts)
    print(f"[pisco] offline compression: {offline_ms:.1f} ms/image, {embs[0].shape[0]} MEM/image")

    def make_batch(K, B):
        enc_ids, chunks = [], []
        for qi in range(B):
            idx = [(qi * K + j) % len(pool) for j in range(K)]
            background = "".join(f"Document {j}:" + dtok.mem_token * embs[i].shape[0] for j, i in enumerate(idx))
            prompt = _build_chat_prompt(dtok, background=background, question=rows[qi % len(rows)]["query"])
            enc_ids.append(dtok(prompt, add_special_tokens=False)["input_ids"]); chunks += [embs[i] for i in idx]
        enc = dtok.pad({"input_ids": enc_ids}, padding="longest", return_tensors="pt")
        ids = enc["input_ids"].to(device)
        compressed = torch.cat(chunks, 0)
        return ids, enc["attention_mask"].to(device), compressed

    def run(K, B):
        ids, attn, compressed = make_batch(K, B)
        with torch.inference_mode():
            embeds = model.replace_embeddings(compressed, ids)   # the lookup a RAG system does per query
            pre = timed(lambda: forward_prefill(model.decoder, inputs_embeds=embeds, attention_mask=attn), args.reps)
            gen = timed(lambda: model.decoder.generate(inputs_embeds=embeds, attention_mask=attn, do_sample=False, top_p=None,
                                                       max_new_tokens=args.gen_tokens, min_new_tokens=args.gen_tokens,
                                                       eos_token_id=model.decoder_eos_ids()), args.reps)
        return int(ids.shape[1]), pre, gen

    return run, {"model": args.checkpoint_path, "offline_compress_ms_per_image": offline_ms,
                 "mem_per_image": int(embs[0].shape[0]), "lora_merged": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["base", "pisco"], required=True)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--base_model_name", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_rows", type=int, default=64)
    ap.add_argument("--ks", default="4,8,16")
    ap.add_argument("--batch_sizes", default="1,8,32")
    ap.add_argument("--gen_tokens", type=int, default=32)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--image_compr_rate", type=int, default=None)
    ap.add_argument("--image_max_pixels", type=int, default=451584)
    args = ap.parse_args()
    device = torch.device("cuda")
    rows, pool = load_rows(args.data, args.n_rows)
    print(f"[bench] {len(rows)} rows, image pool {len(pool)}, gpu {torch.cuda.get_device_name(0)}")
    run, meta = (bench_base if args.mode == "base" else bench_pisco)(args, rows, pool, device)
    results = []
    for K in [int(k) for k in args.ks.split(",")]:
        for B in [int(b) for b in args.batch_sizes.split(",")]:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                L, pre, gen = run(K, B)
                r = {"mode": args.mode, "K": K, "B": B, "prompt_tokens": L, "prefill_ms": 1000 * pre,
                     "generate_ms": 1000 * gen, "qps": B / gen, "ms_per_query": 1000 * gen / B,
                     "peak_gb": torch.cuda.max_memory_allocated() / 2**30}
            except torch.cuda.OutOfMemoryError:
                r = {"mode": args.mode, "K": K, "B": B, "oom": True}
                torch.cuda.empty_cache()
            results.append(r)
            print(json.dumps(r), flush=True)
    out = {"meta": {**meta, **vars(args), "gpu": torch.cuda.get_device_name(0)}, "results": results}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[bench] wrote {args.out}")


if __name__ == "__main__":
    main()
