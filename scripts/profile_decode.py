#!/usr/bin/env python
"""Attribute decode-step GPU time to attention vs FFN(mlp) vs LM-head, to check how big the
248k-vocab LM-head really is. Module-level CUDA-event timing over a 32-token generate.
"""
import argparse, time
import torch
from pisco.collator_utils import add_memory_tokens_to_inputs, chunk_list
from scripts.eval_agent_qa import _build_chat_prompt
from pisco.model import PISCO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="expQ/149907/model")
    ap.add_argument("--context_file", default="eval_data/longbench_multifieldqa_en/multifieldqa_en_0_4000.json")
    ap.add_argument("--n_docs", type=int, default=5)
    ap.add_argument("--doc_len", type=int, default=128)
    ap.add_argument("--n_new", type=int, default=32)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 64])
    args = ap.parse_args()
    dev = torch.device("cuda")

    m = PISCO.from_pretrained(args.checkpoint); m.to(dev); m.eval()
    if hasattr(m.decoder, "merge_and_unload"):
        m.decoder = m.decoder.merge_and_unload()
    dec = m.decoder
    cfg = dec.config
    vocab = getattr(cfg, "vocab_size", None) or getattr(getattr(cfg, "text_config", cfg), "vocab_size", None)

    # discover modules by class/shape (block-level, so no double counting)
    cats = {"attention": [], "mlp": [], "lm_head": []}
    for name, mod in dec.named_modules():
        cn = mod.__class__.__name__
        if isinstance(mod, torch.nn.Linear) and mod.out_features == vocab:
            cats["lm_head"].append(mod)
        elif cn.endswith("MLP"):
            cats["mlp"].append(mod)
        elif cn.endswith("Attention"):
            cats["attention"].append(mod)
    print("modules found:", {k: len(v) for k, v in cats.items()}, "vocab=", vocab)

    events = {k: [] for k in cats}

    def mk_pre():
        def pre(mod, inp):
            s = torch.cuda.Event(enable_timing=True); s.record(); mod._pstart = s
        return pre

    def mk_post(k):
        def post(mod, inp, out):
            e = torch.cuda.Event(enable_timing=True); e.record()
            events[k].append((mod._pstart, e))
        return post

    for k, mods in cats.items():
        for mod in mods:
            mod.register_forward_pre_hook(mk_pre()); mod.register_forward_hook(mk_post(k))

    # build one PISCO decoder input (short context)
    ctok, dtok = m.compressor_tokenizer, m.decoder_tokenizer
    import json
    raw = json.load(open(args.context_file))[0]["trajectory"]
    ids = ctok(raw, add_special_tokens=False)["input_ids"][: args.n_docs * args.doc_len]
    chunks = chunk_list(ids, chunk_length=args.doc_len, chunk_overlap=0)[: args.n_docs]
    chunks_mem, n_mems = add_memory_tokens_to_inputs(chunks, ctok, m.compr_rate)
    with torch.inference_mode():
        padded = ctok.pad([{"input_ids": c} for c in chunks_mem], return_tensors="pt")
        emb = m.compress(padded["input_ids"].to(dev), padded["attention_mask"].to(dev))
        prompt = _build_chat_prompt(dtok, background=dtok.mem_token * int(sum(n_mems)),
                                    question="What is the main topic?")
        dec_ids = dtok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
        embeds1 = m.replace_embeddings(emb, dec_ids)  # [1,L,H]

    for B in args.batches:
        for k in events:
            events[k].clear()
        kw = {"inputs_embeds": embeds1.repeat(B, 1, 1),
              "attention_mask": torch.ones(B, embeds1.shape[1], device=dev)}
        with torch.inference_mode():
            dec.generate(**kw, do_sample=False, top_p=None, max_new_tokens=args.n_new, min_new_tokens=args.n_new)  # warmup
            for k in events:
                events[k].clear()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            dec.generate(**kw, do_sample=False, top_p=None, max_new_tokens=args.n_new, min_new_tokens=args.n_new)
            torch.cuda.synchronize(); total_ms = (time.perf_counter() - t0) * 1000
        sums = {k: sum(s.elapsed_time(e) for s, e in evs) for k, evs in events.items()}
        acc = sum(sums.values())
        print(f"\n=== batch={B}  total_generate={total_ms:.0f}ms  (module-attributed {acc:.0f}ms) ===")
        for k in ["attention", "mlp", "lm_head"]:
            print(f"  {k:10} {sums[k]:8.0f} ms  = {100*sums[k]/total_ms:5.1f}% of total   ({100*sums[k]/acc:5.1f}% of attributed)")
        print(f"  {'other/norms/embed/overhead':10} {total_ms-acc:8.0f} ms  = {100*(total_ms-acc)/total_ms:5.1f}% of total")


if __name__ == "__main__":
    main()
