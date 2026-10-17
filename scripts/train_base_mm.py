"""LoRA-finetune the UNCOMPRESSED VLM backbone (Qwen3.5-4B) on a multimodal RAG set with RAW page/frame
images — the fair ceiling for a PISCO finetune on the same rows (same prompt, same labels, same LoRA
rank/lr), i.e. "what does the decoder reach when it sees the pixels instead of <MEM> embeddings".

Rows: docs (List[str]), doc_images (List[Image|None]), query, mistral_label — the finetune schema.
Prompt = relabel_mm_teacher.build_prompt (== FineTuningCollator prompt == eval_mm_docqa base mode), label =
mistral_label + <|im_end|>, loss on the label only. Vision policy mirrors PISCOConfig.train_vision_lora="merger"
(ViT LoRA frozen, merger + LM LoRA trained). Resumable (checkpoint-N in --out, HF Trainer).

  python scripts/train_base_mm.py --data data/nav_ft_train_v2 --out expQ/ft_nav_base_4b --topk_docs 8
  → <out>/model = PEFT adapter dir (+ base_mm.json), evaluated with
    scripts/eval_mm_docqa.py --mode base --base_adapter_path <out>/model ...
"""
import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from pisco.mm_utils import load_image_processor  # noqa: E402
from relabel_mm_teacher import build_prompt  # noqa: E402


class BaseMMCollator:
    def __init__(self, tok, proc, vision_ids, *, topk_docs: int, max_length: int):
        self.tok, self.proc, self.vision_ids = tok, proc, vision_ids
        self.topk_docs, self.max_length = topk_docs, max_length
        self.im_end = tok.convert_tokens_to_ids("<|im_end|>")
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else self.im_end

    def __call__(self, rows):
        ids_l, lab_l, pvs, grids = [], [], [], []
        for row in rows:
            docs = list(row["docs"])[: self.topk_docs]
            imgs = list(row.get("doc_images") or [])[: self.topk_docs]
            prompt, pv, grid = build_prompt(self.tok, docs, imgs, row["query"], self.proc, self.vision_ids)
            p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
            a_ids = self.tok(str(row["mistral_label"]).strip(), add_special_tokens=False)["input_ids"] + [self.im_end]
            ids = p_ids + a_ids
            labels = [-100] * len(p_ids) + a_ids
            if len(ids) > self.max_length:   # never cut into an image span: refuse instead of corrupting the row
                n_pad_tokens = sum(1 for t in p_ids if t == self.vision_ids[1])
                raise ValueError(f"row too long ({len(ids)} > {self.max_length}; {n_pad_tokens} visual tokens) — raise --max_length")
            ids_l.append(ids); lab_l.append(labels); pvs += pv; grids += grid
        T = max(len(x) for x in ids_l)
        input_ids = torch.full((len(rows), T), self.pad_id, dtype=torch.long)
        labels = torch.full((len(rows), T), -100, dtype=torch.long)
        attn = torch.zeros((len(rows), T), dtype=torch.long)
        for i, (ids, lab) in enumerate(zip(ids_l, lab_l)):   # right padding for training
            input_ids[i, : len(ids)] = torch.tensor(ids); labels[i, : len(lab)] = torch.tensor(lab); attn[i, : len(ids)] = 1
        out = {"input_ids": input_ids, "attention_mask": attn, "labels": labels}
        if pvs:
            out.update(pixel_values=torch.from_numpy(np.concatenate(pvs, 0)),
                       image_grid_thw=torch.tensor(grids, dtype=torch.long),
                       mm_token_type_ids=(input_ids == self.vision_ids[1]).long())
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--topk_docs", type=int, default=8)
    ap.add_argument("--image_max_pixels", type=int, default=451584)
    ap.add_argument("--max_length", type=int, default=8192)
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--vision_lora", choices=["all", "merger", "none"], default="merger")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=64)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--save_steps", type=int, default=50)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    import datasets
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer, Trainer, TrainingArguments

    ds = datasets.load_from_disk(args.data)
    if isinstance(ds, datasets.DatasetDict):
        ds = ds["train"]
    if args.samples:
        ds = ds.select(range(min(len(ds), args.samples)))
    ds = ds.shuffle(seed=42)
    print(f"[base-ft] {len(ds)} rows from {args.data}")

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="right")
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    cfg = model.config
    vision_ids = (int(cfg.vision_start_token_id), int(cfg.image_token_id), int(cfg.vision_end_token_id))
    proc = load_image_processor(args.model, max_pixels=args.image_max_pixels)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    # Same LoRA shape as PISCO's decoder/compressor adapters (r, alpha=2r, all-linear, dropout .1).
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=args.lora_r, lora_alpha=2 * args.lora_r,
                                             target_modules="all-linear", lora_dropout=0.1))
    if args.vision_lora != "all":   # mirror PISCO.train_vision_lora: ViT frozen, merger (+LM) trained
        n_frozen = n_kept = 0
        for name, p in model.named_parameters():
            if ".visual." not in name or not p.requires_grad:
                continue
            if args.vision_lora == "merger" and ".visual.merger." in name:
                n_kept += 1
            else:
                p.requires_grad = False; n_frozen += 1
        print(f"[base-ft] vision_lora={args.vision_lora}: froze {n_frozen} vision tensors, kept {n_kept}")
    model.print_trainable_parameters()

    collator = BaseMMCollator(tok, proc, vision_ids, topk_docs=args.topk_docs, max_length=args.max_length)
    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs, per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum, learning_rate=args.lr, warmup_ratio=0.05, weight_decay=0.1,
        max_grad_norm=1.0, bf16=True, logging_steps=10, save_strategy="steps", save_steps=args.save_steps,
        save_total_limit=2, report_to="tensorboard", remove_unused_columns=False, dataloader_num_workers=args.num_workers,
        optim="adamw_torch", lr_scheduler_type="linear", gradient_checkpointing=False,  # enabled on the model above
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    resume = any(p.name.startswith("checkpoint-") for p in pathlib.Path(args.out).glob("checkpoint-*"))
    print(f"[base-ft] resume_from_checkpoint={resume}")
    trainer.train(resume_from_checkpoint=resume or None)

    out = pathlib.Path(args.out) / "model"
    trainer.model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    json.dump({"base_model": args.model, "image_max_pixels": args.image_max_pixels, "topk_docs": args.topk_docs,
               "lora_r": args.lora_r, "vision_lora": args.vision_lora, "lr": args.lr, "rows": len(ds)},
              open(out / "base_mm.json", "w"), indent=1)
    print(f"[base-ft] saved adapter -> {out}")


if __name__ == "__main__":
    main()
