"""Is the answer still in the <MEM>s? Transcription probe of the doc eval pages (compression capacity vs. its use).

For the first gold page of the first N questions per source of a doc eval set:
  --arm ref  the uncompressed backbone transcribes the page from its pixels (the reference text)
  --arm ae   a PISCO checkpoint trained on image autoencoding (stage-2) rebuilds the page text from its <MEM>s alone,
             in the stage-2 AE format (<AE> <MEM>*n <bos> -> text, MultimodalPretrainingCollator); --compressor_from
             puts another checkpoint's compressor + connector (e.g. the finetuned model's) in front of the same decoder
scripts/analyze_mem_recall.py then asks, per question: is the gold answer in the reference? in the rebuilt text?
and crosses that with the finetuned model's judged answers (answer rebuilt but QA wrong = the decoder does not use it;
answer in the reference but not rebuilt = lost in compression).

  python scripts/probe_mem_recall.py --arm ae --checkpoint_path expQ/pt_mm_v4kd_4b/model \
      --data_path eval_data/mm_docqa_1500_1m --per_source 120 --out expQ/archq/recall/ae_s2.json
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
sys.path.insert(0, str(REPO / "scripts"))

from pisco.model import PISCO, eos_token_ids, merge_lora_inplace  # noqa: E402
from pisco.mm_utils import collate_compressor, image_span_text, image_to_span, merge_size_of, n_image_tokens  # noqa: E402
from eval_mm_docqa import make_collator  # noqa: E402

TRANSCRIBE = ("Transcribe all the text of this image in reading order, including every number, label and table cell. "
              "Output plain text only.")


def pick_rows(ds, per_source):
    seen, rows = {}, []
    for i, src in enumerate(ds["source"]):
        if seen.get(src, 0) >= per_source:
            continue
        row = ds[i]
        gold = [int(g) for g in (row.get("gold_positions") or [])]
        imgs = row.get("doc_images") or []
        if not gold or gold[0] >= len(imgs) or imgs[gold[0]] is None:
            continue
        seen[src] = seen.get(src, 0) + 1
        rows.append((row, gold[0]))
    return rows


@torch.inference_mode()
def run_ref(args, rows, device):
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from pisco.mm_utils import load_image_processor
    tok = AutoTokenizer.from_pretrained(args.base_model_name, padding_side="left")
    base = AutoModelForImageTextToText.from_pretrained(args.base_model_name, torch_dtype=torch.bfloat16,
                                                       attn_implementation="sdpa").to(device).eval()
    proc = load_image_processor(args.base_model_name, max_pixels=args.image_max_pixels)
    cfg = base.config
    vision_ids = (int(cfg.vision_start_token_id), int(cfg.image_token_id), int(cfg.vision_end_token_id))
    merge = merge_size_of(proc)
    for row, g in rows:
        _, pv, grid = image_to_span(row["doc_images"][g], proc, vision_ids)
        msgs = [{"role": "user", "content": image_span_text(n_image_tokens(grid, merge)) + "\n" + TRANSCRIBE}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        out = base.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                            pixel_values=torch.from_numpy(pv).to(device),
                            image_grid_thw=torch.tensor([grid], dtype=torch.long, device=device),
                            mm_token_type_ids=(ids == vision_ids[1]).long(),
                            do_sample=False, top_p=None, max_new_tokens=args.max_new_tokens,
                            eos_token_id=eos_token_ids(tok))
        yield tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)[0].strip(), \
            {"visual_tokens": n_image_tokens(grid, merge)}


@torch.inference_mode()
def run_ae(args, rows, device):
    over = {"image_max_pixels": args.image_max_pixels}
    model = PISCO.from_pretrained(args.checkpoint_path, **over)
    if args.compressor_from:   # another checkpoint's compressor + connector, the AE decoder kept
        other = PISCO.from_pretrained(args.compressor_from, **over)
        model.compressor, model.connector = other.compressor, other.connector
        del other
    model.to(device).eval()
    model.decoder = merge_lora_inplace(model.decoder)
    model.compressor = merge_lora_inplace(model.compressor)
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
        embeds = model.replace_embeddings(emb, ids)
        out = model.decoder.generate(inputs_embeds=embeds, attention_mask=torch.ones_like(ids), do_sample=False,
                                     top_p=None, max_new_tokens=args.max_new_tokens, eos_token_id=eos)
        yield dtok.batch_decode(out, skip_special_tokens=True)[0].strip(), {"mems": batch.total_mems}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["ref", "ae"], required=True)
    ap.add_argument("--checkpoint_path", help="ae: PISCO checkpoint with an AE-trained decoder (stage-2)")
    ap.add_argument("--compressor_from", default=None, help="ae: take compressor + connector from this checkpoint")
    ap.add_argument("--base_model_name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--per_source", type=int, default=120)
    ap.add_argument("--image_max_pixels", type=int, default=1003520)
    ap.add_argument("--image_compr_rate", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=768, help="768 = the stage-2 AE target length")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert args.arm == "ref" or args.checkpoint_path, "--arm ae needs --checkpoint_path"

    import datasets
    device = torch.device("cuda")
    ds = datasets.load_from_disk(args.data_path)
    ds = ds["train"] if isinstance(ds, datasets.DatasetDict) else ds
    rows = pick_rows(ds, args.per_source)
    print(f"[probe] {len(rows)} pages, arm={args.arm}", flush=True)
    gen = run_ref(args, rows, device) if args.arm == "ref" else run_ae(args, rows, device)
    records, t0 = [], time.time()
    for i, ((row, g), (text, st)) in enumerate(zip(rows, gen)):
        records.append({"query_id": row.get("query_id"), "source": row.get("source"), "query": row["query"],
                        "answers": list(row.get("answers") or []), "gold_page": g, "text": text, **st})
        if (i + 1) % 20 == 0:
            print(f"[{i + 1}/{len(rows)}] {time.time() - t0:.0f}s, mean {np.mean([len(r['text'].split()) for r in records]):.0f} words",
                  flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "records": records}, f, indent=1, ensure_ascii=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
