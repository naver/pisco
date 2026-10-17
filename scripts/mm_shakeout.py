"""Stage-0 shake-out for multimodal PISCO: no training, one GPU, ~10 minutes.

Loads a (text-trained) PISCO checkpoint whose backbones are Qwen3.5 VLMs and checks, each
printing PASS/FAIL:
  1. offline image processor shapes (448x448 -> (784,1536), grid [1,28,28]; big page capped)
  2. vision token ids == compressor config ids; <MEM> id; vocab size
  3. one-sample compress() with an image -> (n_mem, hidden) finite, non-constant
  4. mixed batch: a text chunk's MEM embeddings are unchanged by an image neighbour (M-RoPE
     under left padding) -> cosine > 0.99
  5. fwd+bwd memory probe at several batch sizes (image-AE shaped batches), for the
     requested train_vision_lora policy
  6. generate() smoke
  7. datasets schema: Sequence(Image()) with None survives save_to_disk/load_from_disk

Usage:
  python scripts/mm_shakeout.py --checkpoint expQ/qwen35_4b_pt_5M_143228/model \
      [--batch_sizes 2,4,8] [--train_vision_lora merger] [--image_max_pixels 451584]
"""

import argparse
import math
import os
import sys
import tempfile
import time
import traceback

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pisco.model import PISCO  # noqa: E402
from pisco.mm_utils import (  # noqa: E402
    collate_compressor,
    get_vision_token_ids,
    load_image_processor,
    segments_to_chunks,
)
from pisco.collator_utils import add_memory_tokens_to_inputs  # noqa: E402

RESULTS = []
PROBE_LINES = []


def report(name, ok, detail=""):
    RESULTS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if name.startswith("fwd+bwd"):
        PROBE_LINES.append(f"{name}: {'ok' if ok else 'FAIL'} {detail}")


def synth_page(w, h, seed=0):
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    y = 10
    while y < h - 20:
        d.text((10, y), " ".join(f"word{rng.integers(0, 999)}" for _ in range(max(3, w // 60))), fill=(0, 0, 0))
        y += 18
    return img


class _Pad:
    """Minimal stand-in exposing compressor_pad + image_processor for collate_compressor."""

    def __init__(self, tok, proc):
        self.tok, self.image_processor = tok, proc

    def compressor_pad(self, input_ids):
        return self.tok.pad({"input_ids": input_ids}, padding="longest", return_tensors="pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch_sizes", default="2,4,8")
    ap.add_argument("--train_vision_lora", default="merger")
    ap.add_argument("--image_max_pixels", type=int, default=451584)
    ap.add_argument("--decoder_len", type=int, default=1024)
    ap.add_argument("--gradient_checkpointing", action="store_true",
                    help="enable grad checkpointing on both backbones before the memory probe (as training does)")
    ap.add_argument("--probe_repeats", type=int, default=2,
                    help="fwd+bwd repetitions per batch size; the LAST timing is reported (first = Triton warm-up)")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}  ckpt={args.checkpoint}")

    # ---- 1. processor
    try:
        proc = load_image_processor("Qwen/Qwen3.5-4B", max_pixels=args.image_max_pixels)
        o = proc(images=[synth_page(448, 448)], return_tensors="np")
        ok1 = tuple(o["pixel_values"].shape) == (784, 1536) and o["image_grid_thw"].tolist() == [[1, 28, 28]]
        o2 = proc(images=[synth_page(1240, 1754)], return_tensors="np")
        g = o2["image_grid_thw"][0]
        n_big = int(g[0] * g[1] * g[2] // 4)
        ok1b = n_big * 1024 <= args.image_max_pixels + 32 * 32 * 4
        report("processor shapes", ok1 and ok1b, f"448->{o['image_grid_thw'].tolist()}  A4@150dpi->{n_big} tokens")
    except Exception:
        traceback.print_exc(); report("processor shapes", False)
        return finish()

    # ---- load model
    t0 = time.time()
    model = PISCO.from_pretrained(
        args.checkpoint, train_vision_lora=args.train_vision_lora, image_max_pixels=args.image_max_pixels
    )
    model.to(dev)
    print(f"model loaded in {time.time() - t0:.0f}s; image_processor={'yes' if model.image_processor else 'NO'}")
    report("model has image processor", model.image_processor is not None)
    tok = model.compressor_tokenizer

    # ---- 2. token ids
    try:
        vs, ip, ve = get_vision_token_ids(tok)
        cfg = model.compressor.config
        ok = (vs, ip, ve) == (int(cfg.vision_start_token_id), int(cfg.image_token_id), int(cfg.vision_end_token_id))
        report("vision token ids", ok, f"ids={(vs, ip, ve)} mem={tok.mem_token_id} vocab={len(tok)}")
    except Exception:
        traceback.print_exc(); report("vision token ids", False); return finish()

    pad = _Pad(tok, model.image_processor)
    vision_ids = (vs, ip, ve)
    rate = model.compr_rate

    # ---- 3. one-sample compress
    try:
        b = segments_to_chunks([("image", synth_page(448, 448))], compressor_tokenizer=tok,
                               image_processor=model.image_processor, compr_rate=rate,
                               text_chunker=lambda i: [i], vision_ids=vision_ids)
        batch = collate_compressor([b], pad, vision_ids)
        batch = {k: v.to(dev) for k, v in batch.items()}
        with torch.no_grad():
            emb = model.compress(batch["compressor_input_ids"], batch["compressor_attention_mask"],
                                 pixel_values=batch["pixel_values"], image_grid_thw=batch["image_grid_thw"],
                                 mm_token_type_ids=batch["mm_token_type_ids"])
        n_mem = b.n_mems[0]
        var = emb.float().std(0).mean().item()
        ok = emb.shape[0] == n_mem and torch.isfinite(emb).all().item() and var > 1e-4
        report("one-image compress", ok, f"emb={tuple(emb.shape)} expected n_mem={n_mem} ({198}//{rate}+1) row-std={var:.3g}")
    except Exception:
        traceback.print_exc(); report("one-image compress", False)

    # ---- 4. mixed-batch equivalence
    try:
        text = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris. " * 3
        bt = segments_to_chunks([("text", text)], compressor_tokenizer=tok, image_processor=None,
                                compr_rate=rate, text_chunker=lambda i: [i], vision_ids=vision_ids)
        bi = segments_to_chunks([("image", synth_page(448, 448, seed=3))], compressor_tokenizer=tok,
                                image_processor=model.image_processor, compr_rate=rate,
                                text_chunker=lambda i: [i], vision_ids=vision_ids)
        alone = collate_compressor([bt], pad, vision_ids)
        mixed = collate_compressor([bt, bi], pad, vision_ids)
        with torch.no_grad():
            e_alone = model.compress(alone["compressor_input_ids"].to(dev), alone["compressor_attention_mask"].to(dev))
            m = {k: v.to(dev) for k, v in mixed.items()}
            e_mixed = model.compress(m["compressor_input_ids"], m["compressor_attention_mask"],
                                     pixel_values=m["pixel_values"], image_grid_thw=m["image_grid_thw"],
                                     mm_token_type_ids=m["mm_token_type_ids"])
        n_t = bt.n_mems[0]
        cos = torch.nn.functional.cosine_similarity(e_alone.float(), e_mixed[:n_t].float(), dim=-1)
        report("mixed-batch equivalence (left pad + M-RoPE)", bool(cos.min() > 0.99),
               f"min cos={cos.min().item():.4f} mean={cos.mean().item():.4f} (text chunk padded from {alone['compressor_input_ids'].shape[1]} to {mixed['compressor_input_ids'].shape[1]})")
    except Exception:
        traceback.print_exc(); report("mixed-batch equivalence", False)

    # ---- 5. memory probe (image-AE shaped batches)
    dtok = model.decoder_tokenizer
    if args.gradient_checkpointing:
        for m in (model.compressor, model.decoder):
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("[probe] gradient checkpointing enabled on compressor + decoder")
    for bs in [int(x) for x in args.batch_sizes.split(",") if x]:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            model.train()
            batches = [segments_to_chunks([("image", synth_page(1000, 1400, seed=i))], compressor_tokenizer=tok,
                                          image_processor=model.image_processor, compr_rate=rate,
                                          text_chunker=lambda i: [i], vision_ids=vision_ids) for i in range(bs)]
            comp = collate_compressor(batches, pad, vision_ids)
            dec_texts = [dtok.ae_token + dtok.mem_token * b.n_mems[0] + dtok.bos_token +
                         " ".join(f"tok{j}" for j in range(args.decoder_len - b.n_mems[0] - 40)) + dtok.eos_token for b in batches]
            dec = dtok(dec_texts, return_tensors="pt", padding="longest", add_special_tokens=False,
                       max_length=args.decoder_len, truncation=True)
            labels = dec["input_ids"].clone(); labels[labels == dtok.mem_token_id] = -100
            labels[labels == dtok.pad_token_id] = -100
            inputs = dict(compressor_input_ids=comp["compressor_input_ids"], compressor_attention_mask=comp["compressor_attention_mask"],
                          decoder_input_ids=dec["input_ids"], decoder_attention_mask=dec["attention_mask"], labels=labels,
                          pixel_values=comp["pixel_values"], image_grid_thw=comp["image_grid_thw"], mm_token_type_ids=comp["mm_token_type_ids"])
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            times = []
            for _ in range(max(1, args.probe_repeats)):
                torch.cuda.synchronize(); t = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(**inputs)
                out["loss"].backward()
                torch.cuda.synchronize(); times.append(time.time() - t)
                model.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated() / 2**30
            report(f"fwd+bwd bs={bs}", math.isfinite(out["loss"].item()),
                   f"loss={out['loss'].item():.3f} peak={peak:.1f} GiB  {times[-1]:.1f}s (first {times[0]:.1f}s)  "
                   f"gc={'on' if args.gradient_checkpointing else 'off'}  compressor rows {tuple(comp['compressor_input_ids'].shape)} decoder {tuple(dec['input_ids'].shape)}")
        except torch.cuda.OutOfMemoryError:
            report(f"fwd+bwd bs={bs}", False, "CUDA OOM")
            model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        except Exception:
            traceback.print_exc(); report(f"fwd+bwd bs={bs}", False)
    model.eval()

    # ---- 6. generate smoke
    try:
        b = segments_to_chunks([("image", synth_page(448, 448, seed=7))], compressor_tokenizer=tok,
                               image_processor=model.image_processor, compr_rate=rate,
                               text_chunker=lambda i: [i], vision_ids=vision_ids)
        comp = collate_compressor([b], pad, vision_ids)
        prompt = dtok.ae_token + dtok.mem_token * b.n_mems[0] + dtok.bos_token
        dec = dtok([prompt], return_tensors="pt", add_special_tokens=False)
        mi = {"compressor_input_ids": comp["compressor_input_ids"], "compressor_attention_mask": comp["compressor_attention_mask"],
              "decoder_input_ids": dec["input_ids"], "decoder_attention_mask": dec["attention_mask"],
              "pixel_values": comp["pixel_values"], "image_grid_thw": comp["image_grid_thw"], "mm_token_type_ids": comp["mm_token_type_ids"]}
        mi = {k: v.to(dev) for k, v in mi.items()}
        with torch.no_grad():
            txt = model.generate(mi, max_new_tokens=24)
        report("generate smoke", isinstance(txt, list) and len(txt) == 1, repr(txt[0][:120]))
    except Exception:
        traceback.print_exc(); report("generate smoke", False)

    # ---- 7. datasets schema round-trip
    try:
        import datasets
        feats = datasets.Features({"docs": datasets.Sequence(datasets.Value("string")),
                                   "doc_images": datasets.Sequence(datasets.Image()),
                                   "query": datasets.Value("string"), "mistral_label": datasets.Value("string")})
        ds = datasets.Dataset.from_dict({"docs": [["", "text doc"]], "doc_images": [[synth_page(64, 64), None]],
                                         "query": ["q"], "mistral_label": ["a"]}, features=feats)
        with tempfile.TemporaryDirectory() as td:
            ds.save_to_disk(td); back = datasets.load_from_disk(td)
            row = back[0]
            ok = row["doc_images"][1] is None and isinstance(row["doc_images"][0], Image.Image)
        report("datasets Sequence(Image()) with None round-trip", ok)
    except Exception:
        traceback.print_exc(); report("datasets Sequence(Image()) with None round-trip", False,
                                     "-> fall back to blank JPEG + doc_has_image flags")

    return finish()


def finish():
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print("\n=== MM SHAKEOUT SUMMARY ===")
    for name, ok in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"=== {len(RESULTS) - n_fail}/{len(RESULTS)} passed ===")
    # expmon profile contract: the fwd+bwd probe lines, between these markers
    print("=== MM PROFILE SUMMARY ===")
    for line in PROBE_LINES:
        print(line)
    print("=== END MM PROFILE SUMMARY ===")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
