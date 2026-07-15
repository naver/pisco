#!/usr/bin/env python
"""Batched decode throughput on a REAL RAG dataset (varied inputs), PISCO vs uncompressed.

For N real (docs, query) examples: PISCO precomputes MEMs per example (offline); the decoder
then runs batched over cached MEM embeddings (left-padded to batch max). The uncompressed
baseline runs batched over the full concatenated documents. Sweep batch size -> seq/s + max
batch. Compression is NOT counted (precomputed/cached), matching the paper's serving setup.
"""
import argparse, os, time
import torch
from pisco.collator_utils import add_memory_tokens_to_inputs, chunk_list
from scripts.eval_agent_qa import _build_chat_prompt
from pisco.model import PISCO
from transformers import AutoTokenizer
from datasets import load_dataset


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_gen(model, kwargs, n_new, repeats=3):
    model.generate(**kwargs, do_sample=False, top_p=None, max_new_tokens=n_new, min_new_tokens=n_new)
    sync(); t = time.perf_counter()
    for _ in range(repeats):
        model.generate(**kwargs, do_sample=False, top_p=None, max_new_tokens=n_new, min_new_tokens=n_new)
    sync(); return (time.perf_counter() - t) / repeats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="expQ/149907/model")
    ap.add_argument("--dataset", default="maxoul/pisco_finetuning_data")
    ap.add_argument("--n_examples", type=int, default=512)
    ap.add_argument("--n_docs", type=int, default=5)
    ap.add_argument("--doc_len", type=int, default=128)
    ap.add_argument("--n_new", type=int, default=32)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64, 256, 512])
    args = ap.parse_args()
    dev = torch.device("cuda")

    m = PISCO.from_pretrained(args.checkpoint); m.to(dev); m.eval()
    if hasattr(m.decoder, "merge_and_unload"):
        m.decoder = m.decoder.merge_and_unload()
    ctok, dtok = m.compressor_tokenizer, m.decoder_tokenizer
    btok = AutoTokenizer.from_pretrained(m.config.decoder_model_name, padding_side="left")
    if btok.pad_token_id is None:
        btok.pad_token_id = btok.eos_token_id
    H = m.decoder.get_input_embeddings().weight.shape[1]
    pad_emb = m.decoder.get_input_embeddings()(
        torch.tensor([dtok.pad_token_id or dtok.eos_token_id], device=dev)).detach()  # [1,H]

    ds = load_dataset(args.dataset, split="train", streaming=True)
    pisco_embeds, base_ids = [], []          # per-example (CPU for embeds)
    plens, blens = [], []
    with torch.inference_mode():
        for ex in ds:
            if len(pisco_embeds) >= args.n_examples:
                break
            docs = list(ex["docs"])[: args.n_docs]
            query = ex["query"]
            # each doc -> <=doc_len tokens (compressor side)
            chunks = []
            for d in docs:
                ids = ctok(d, add_special_tokens=False, truncation=True, max_length=args.doc_len)["input_ids"]
                if ids:
                    chunks.append(ids)
            if not chunks:
                continue
            chunks_mem, n_mems = add_memory_tokens_to_inputs(chunks, ctok, m.compr_rate)
            total_mems = int(sum(n_mems))
            padded = ctok.pad([{"input_ids": c} for c in chunks_mem], return_tensors="pt")
            emb = m.compress(padded["input_ids"].to(dev), padded["attention_mask"].to(dev))
            prompt = _build_chat_prompt(dtok, background=dtok.mem_token * total_mems, question=query)
            dec_ids = dtok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
            e1 = m.replace_embeddings(emb, dec_ids)[0]      # [L,H]
            pisco_embeds.append(e1.to("cpu")); plens.append(e1.shape[0])
            # base: full concatenated docs (each truncated to doc_len) + query
            base_ctx = "\n\n".join(ctok.decode(ctok(d, add_special_tokens=False, truncation=True,
                                    max_length=args.doc_len)["input_ids"]) for d in docs)
            bp = _build_chat_prompt(btok, background=base_ctx, question=query)
            base_ids.append(bp); blens.append(len(btok(bp, add_special_tokens=False)["input_ids"]))

    N = len(pisco_embeds)
    print(f"loaded {N} real RAG examples from {args.dataset}")
    print(f"PISCO decoder len: mean {sum(plens)/N:.0f} (max {max(plens)})   "
          f"base len: mean {sum(blens)/N:.0f} (max {max(blens)})  compr_rate={m.compr_rate}")
    print(f"{'batch':>6} {'pisco_s/it':>11} {'base_s/it':>10} {'pisco_seq/s':>12} {'base_seq/s':>11} {'speedup':>8}")

    def pisco_batch(B):
        embs = pisco_embeds[:B]; Lmax = max(e.shape[0] for e in embs)
        x = pad_emb.expand(B, Lmax, H).clone()
        am = torch.zeros(B, Lmax, device=dev)
        for i, e in enumerate(embs):
            L = e.shape[0]; x[i, Lmax - L:, :] = e.to(dev); am[i, Lmax - L:] = 1
        return {"inputs_embeds": x, "attention_mask": am}

    def base_batch(B):
        enc = btok(base_ids[:B], return_tensors="pt", add_special_tokens=False, padding=True)
        return {k: v.to(dev) for k, v in enc.items()}

    for B in args.batches:
        if B > N:
            break
        row = {}
        for name, mk in [("pisco", pisco_batch), ("base", base_batch)]:
            try:
                with torch.inference_mode():
                    row[name] = time_gen(m.decoder, mk(B), args.n_new)
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                row[name] = None; torch.cuda.empty_cache()
        p, b = row.get("pisco"), row.get("base")
        def f(x, d=3): return f"{x:.{d}f}" if isinstance(x, (int, float)) else "OOM"
        pt = B / p if p else None; bt = B / b if b else None
        sp = b / p if (p and b) else None
        print(f"{B:>6} {f(p):>11} {f(b):>10} {f(pt,1):>12} {f(bt,1):>11} {f(sp,2):>8}")
    print("\n(real RAG inputs, batched with left-padding; compression precomputed/cached; OOM=didn't fit)")


if __name__ == "__main__":
    main()
