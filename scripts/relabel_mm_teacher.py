"""Teacher-relabel a multimodal finetune set: replace `mistral_label` by the UNCOMPRESSED backbone's
own answer (Qwen3.5-4B reading the raw pages with the exact finetuning prompt).

Why: text PISCO's finetune targets were the backbone's answers on the same documents (distillation),
so the compressed model learns to reproduce what the decoder would say with full context. Our first
multimodal finetune used dataset gold answers instead (human/GPT style the decoder never produces).
This script builds the distillation targets. Dataset gold is kept in `answers` / `gold_label`.

Sharded for parallel GPU jobs:
  python scripts/relabel_mm_teacher.py --src data/mm_ft_docpages_20k --out data/mm_ft_docpages_20k_teacher \
      --shard 0 --n_shards 4 [--model Qwen/Qwen3.5-4B] [--max_new_tokens 48] [--batch_size 4]
  ... then --merge to concatenate the shard outputs into the final dataset dir.
"""

import argparse
import os
import pathlib
import sys
import time

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from pisco.collator import FineTuningCollator  # noqa: E402
from pisco.mm_utils import image_to_span, load_image_processor  # noqa: E402

_D = {k: v.default for k, v in __import__("inspect").signature(FineTuningCollator.__init__).parameters.items()
      if k in ("system_prompt", "user_prompt")}


def build_prompt(tok, docs, imgs, query, proc, vision_ids):
    vs, ip, ve = vision_ids
    background, pvs, grids = "", [], []
    for j, d in enumerate(docs):
        img = imgs[j] if j < len(imgs) else None
        if img is not None:
            span, pv, grid = image_to_span(img, proc, vision_ids)
            pvs.append(pv); grids.append(grid)
            background += f"Document {j}:" + tok.convert_ids_to_tokens(vs) + tok.convert_ids_to_tokens(ip) * (len(span) - 2) + tok.convert_ids_to_tokens(ve)
        else:
            background += f"Document {j}:" + (d or "")
    messages = [{"role": "system", "content": _D["system_prompt"]},
                {"role": "user", "content": _D["user_prompt"].replace("[documents]", background).replace("[question]", query)}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return prompt, pvs, grids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--topk_docs", type=int, default=8)
    ap.add_argument("--image_max_pixels", type=int, default=451584)
    ap.add_argument("--merge", action="store_true", help="concatenate <out>/shard_* into <out> (no GPU)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import datasets
    if args.merge:
        parts = [datasets.load_from_disk(str(p)) for p in sorted(pathlib.Path(args.out).glob("shard_*"))]
        full = datasets.concatenate_datasets(parts)
        datasets.DatasetDict({"train": full}).save_to_disk(args.out)
        n_same = sum(1 for r in full.select(range(0, len(full), max(1, len(full) // 2000))) if r["mistral_label"].strip().lower() == r["gold_label"].strip().lower())
        print(f"merged {len(parts)} shards -> {args.out}: {len(full)} rows; teacher == gold on ~{n_same} of the sampled rows")
        return

    try:
        ds = datasets.load_from_disk(args.src)["train"]
    except KeyError:
        ds = datasets.load_from_disk(args.src)
    idx = list(range(args.shard, len(ds), args.n_shards))
    if args.limit:
        idx = idx[: args.limit]
    ds = ds.select(idx)
    print(f"[relabel] shard {args.shard}/{args.n_shards}: {len(ds)} rows")

    from transformers import AutoModelForImageTextToText, AutoTokenizer
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    proc = load_image_processor(args.model, max_pixels=args.image_max_pixels)
    cfg = model.config
    vision_ids = (int(cfg.vision_start_token_id), int(cfg.image_token_id), int(cfg.vision_end_token_id))
    eos = [i for i in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
           if isinstance(i, int) and i >= 0]

    teacher = []
    t0 = time.time()
    with torch.inference_mode():
        for i, row in enumerate(ds):
            docs = list(row["docs"])[: args.topk_docs]
            imgs = list(row.get("doc_images") or [])[: args.topk_docs]
            prompt, pvs, grids = build_prompt(tok, docs, imgs, row["query"], proc, vision_ids)
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
            kwargs = dict(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
            if pvs:
                kwargs.update(pixel_values=torch.from_numpy(np.concatenate(pvs, 0)).to(device),
                              image_grid_thw=torch.tensor(grids, dtype=torch.long, device=device),
                              mm_token_type_ids=(enc["input_ids"] == vision_ids[1]).long().to(device))
            out = model.generate(**kwargs, do_sample=False, top_p=None, max_new_tokens=args.max_new_tokens, eos_token_id=eos)
            teacher.append(tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip())
            if (i + 1) % 100 == 0:
                print(f"[relabel] {i + 1}/{len(ds)}  {time.time() - t0:.0f}s  e.g. {teacher[-1][:60]!r}", flush=True)

    ds = ds.add_column("gold_label", ds["mistral_label"])
    ds = ds.remove_columns("mistral_label").add_column("mistral_label", teacher)
    out = pathlib.Path(args.out) / f"shard_{args.shard:02d}"
    ds.save_to_disk(str(out))
    print(f"[relabel] wrote {out} ({len(ds)} rows, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
