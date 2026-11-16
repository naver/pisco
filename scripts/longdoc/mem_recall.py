"""Stride-aware port of scripts/probe_mem_recall.py (campaign longdoc, arm T): are a page's words in its <MEM>s?

The legacy probe calls `model.decoder.generate` without `position_ids`, so it is legacy-only (HANDOFF.md
don't-break list): on a checkpoint with mem_position_stride > 1 every <MEM> must take `stride` position ids.
This port imports the legacy row picker and reference arm UNCHANGED and re-implements only the AE arm, adding the
checkpoint's own <MEM> position ids (PISCO.decoder_position_ids; None on a legacy checkpoint, so the legacy path
is byte-for-byte the old one). Same args, same output schema, plus `stride` per record.

  --arm ref  the uncompressed backbone transcribes the first gold page from its pixels (reference text)
  --arm ae   a stage-2 PISCO checkpoint rebuilds the page text from its <MEM>s (<AE> <MEM>*n <bos> -> text)

Validation (law): on the legacy checkpoint expQ/pt_mm_v4kd_4b/model, 120 pages per source, word recall vs the
recorded ref.json must be 70.5% (recomputed from expQ/archq/recall/ae_s2.json) within +-2 pt.
Pairing two rebuilds per page: scripts/longdoc/recall_pair.py.
"""

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from pisco.model import PISCO, merge_lora_inplace  # noqa: E402
from pisco.mm_utils import collate_compressor  # noqa: E402
from eval_mm_docqa import _reset_rope_deltas, make_collator  # noqa: E402
from probe_mem_recall import pick_rows, run_ref  # noqa: E402  (legacy code, imported, never edited)


@torch.inference_mode()
def run_ae(args, rows, device):
    over = {"image_max_pixels": args.image_max_pixels}
    model = PISCO.from_pretrained(args.checkpoint_path, **over)
    model.to(device).eval()
    model.decoder = merge_lora_inplace(model.decoder)
    model.compressor = merge_lora_inplace(model.compressor)
    stride = int(getattr(model.config, "mem_position_stride", 1) or 1)
    print(f"[probe] mem_position_stride={stride}", flush=True)
    coll = make_collator(model, topk=1, compressor_max_length=256, image_compr_rate=args.image_compr_rate,
                         query_dependent=False)
    dtok = model.decoder_tokenizer
    eos = model.decoder_eos_ids()
    for row, g in rows:
        batch, _, _ = coll.example_chunks([" "], [row["doc_images"][g]], None)
        comp = {k: v.to(device) for k, v in collate_compressor([batch], coll, coll.vision_ids).items()}
        emb = model.compress(comp["compressor_input_ids"], comp["compressor_attention_mask"],
                             pixel_values=comp["pixel_values"], image_grid_thw=comp["image_grid_thw"],
                             mm_token_type_ids=comp["mm_token_type_ids"])
        prompt = dtok.ae_token + dtok.mem_token * batch.total_mems + dtok.bos_token
        ids = dtok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        attn = torch.ones_like(ids)
        embeds = model.replace_embeddings(emb, ids)
        extra = {}
        pos = model.decoder_position_ids(ids, attn)
        if pos is not None:   # spread layout: HF generate continues each row from its last given id (+1 per step)
            _reset_rope_deltas(model.decoder)
            extra["position_ids"] = pos
        out = model.decoder.generate(inputs_embeds=embeds, attention_mask=attn, do_sample=False,
                                     top_p=None, max_new_tokens=args.max_new_tokens, eos_token_id=eos, **extra)
        yield dtok.batch_decode(out, skip_special_tokens=True)[0].strip(), {"mems": batch.total_mems, "stride": stride}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["ref", "ae"], required=True)
    ap.add_argument("--checkpoint_path", help="ae: PISCO checkpoint with an AE-trained decoder (stage-2)")
    ap.add_argument("--base_model_name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--per_source", type=int, default=120)
    ap.add_argument("--image_max_pixels", type=int, default=1003520)
    ap.add_argument("--image_compr_rate", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=768, help="768 = the stage-2 AE target length")
    ap.add_argument("--limit", type=int, default=0, help="plumbing: stop after this many pages (0 = all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert args.arm == "ref" or args.checkpoint_path, "--arm ae needs --checkpoint_path"

    import datasets
    device = torch.device("cuda")
    ds = datasets.load_from_disk(args.data_path)
    ds = ds["train"] if isinstance(ds, datasets.DatasetDict) else ds
    rows = pick_rows(ds, args.per_source)
    if args.limit:
        rows = rows[:args.limit]
    print(f"[probe] {len(rows)} pages, arm={args.arm}", flush=True)
    gen = run_ref(args, rows, device) if args.arm == "ref" else run_ae(args, rows, device)
    records, t0 = [], time.time()
    for i, ((row, g), (text, st)) in enumerate(zip(rows, gen)):
        records.append({"query_id": row.get("query_id"), "source": row.get("source"), "query": row["query"],
                        "answers": list(row.get("answers") or []), "gold_page": g, "text": text, **st})
        if (i + 1) % 20 == 0:
            print(f"[{i + 1}/{len(rows)}] {time.time() - t0:.0f}s, mean "
                  f"{np.mean([len(r['text'].split()) for r in records]):.0f} words", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "records": records}, f, indent=1, ensure_ascii=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
