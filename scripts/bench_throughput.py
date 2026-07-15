#!/usr/bin/env python
"""Batched decode throughput: PISCO (precomputed MEMs) vs uncompressed, reproducing the
PISCO paper's Table 3 regime — short context (5 docs x 128 tok), 32-tok output, sweep batch
size. Reports seq/s and the max batch that fits (the paper's real lever).

Compression is precomputed ONCE (offline), then the decoder runs batched over cached MEM
embeddings; the uncompressed baseline runs batched over the full 640-token context.
"""
import argparse, json, time
import torch
from pisco.collator_utils import add_memory_tokens_to_inputs, chunk_list
from scripts.eval_agent_qa import _build_chat_prompt
from pisco.model import PISCO
from transformers import AutoModelForCausalLM, AutoTokenizer


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_generate(model, kwargs, n_new, repeats=3):
    # warmup
    model.generate(**kwargs, do_sample=False, top_p=None,
                   max_new_tokens=n_new, min_new_tokens=n_new)
    sync()
    t = time.perf_counter()
    for _ in range(repeats):
        model.generate(**kwargs, do_sample=False, top_p=None,
                       max_new_tokens=n_new, min_new_tokens=n_new)
    sync()
    return (time.perf_counter() - t) / repeats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="expQ/161788/model")
    ap.add_argument("--n_docs", type=int, default=5)
    ap.add_argument("--doc_len", type=int, default=128)
    ap.add_argument("--n_new", type=int, default=32)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64, 256, 1024])
    ap.add_argument("--context_file", default="eval_data/longbench_multifieldqa_en/multifieldqa_en_0_4000.json")
    args = ap.parse_args()
    dev = torch.device("cuda")

    # --- load PISCO, merge LoRA (deployable) ---
    m = PISCO.from_pretrained(args.checkpoint); m.to(dev); m.eval()
    if hasattr(m.decoder, "merge_and_unload"):
        m.decoder = m.decoder.merge_and_unload()
    ctok, dtok = m.compressor_tokenizer, m.decoder_tokenizer

    # --- build a short context: n_docs x doc_len tokens from a real doc ---
    raw = json.load(open(args.context_file))[0]["trajectory"]
    ctx_ids = ctok(raw, add_special_tokens=False)["input_ids"][: args.n_docs * args.doc_len]
    chunks = chunk_list(ctx_ids, chunk_length=args.doc_len, chunk_overlap=0)[: args.n_docs]
    chunks_mem, n_mems = add_memory_tokens_to_inputs(chunks, ctok, m.compr_rate)
    total_mems = int(sum(n_mems))
    query = "According to the documents, what is the main topic?"

    # --- PISCO: precompute MEMs ONCE (offline), build 1 decoder example ---
    with torch.inference_mode():
        padded = ctok.pad([{"input_ids": c} for c in chunks_mem], return_tensors="pt")
        emb = m.compress(padded["input_ids"].to(dev), padded["attention_mask"].to(dev))
        prompt = _build_chat_prompt(dtok, background=dtok.mem_token * total_mems, question=query)
        dec_ids = dtok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
        pisco_embeds_1 = m.replace_embeddings(emb, dec_ids)  # [1, L, H]
    pisco_L = pisco_embeds_1.shape[1]

    # --- base: full 640-tok context, 1 example ---
    btok = AutoTokenizer.from_pretrained(m.config.decoder_model_name, padding_side="left")
    if btok.pad_token_id is None:
        btok.pad_token_id = btok.eos_token_id
    base_text = ctok.decode(ctx_ids, skip_special_tokens=True)
    base_prompt = _build_chat_prompt(btok, background=base_text, question=query)
    base_ids_1 = btok(base_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
    base_L = base_ids_1.shape[1]
    base_model = m.decoder  # SAME merged decoder weights -> fair (base just reads tokens, no MEMs)

    print(f"PISCO decoder input len = {pisco_L} (mems={total_mems})   base input len = {base_L} tok")
    print(f"{'batch':>6} {'pisco_s/it':>11} {'base_s/it':>10} {'pisco_seq/s':>12} {'base_seq/s':>11} {'speedup':>8}")
    rows = []
    for B in args.batches:
        res = {"B": B}
        for name, fn in [("pisco", None), ("base", None)]:
            try:
                with torch.inference_mode():
                    if name == "pisco":
                        kw = {"inputs_embeds": pisco_embeds_1.repeat(B, 1, 1),
                              "attention_mask": torch.ones(B, pisco_L, device=dev)}
                        model = m.decoder
                    else:
                        kw = {"input_ids": base_ids_1.repeat(B, 1),
                              "attention_mask": torch.ones(B, base_L, device=dev)}
                        model = base_model
                    t = time_generate(model, kw, args.n_new)
                res[name] = t
                res[name + "_tps"] = B / t
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                res[name] = None; res[name + "_tps"] = None
                torch.cuda.empty_cache()
        p, b = res.get("pisco"), res.get("base")
        sp = (b / p) if (p and b) else None
        def f(x, d=3): return f"{x:.{d}f}" if isinstance(x, (int, float)) else "OOM"
        print(f"{B:>6} {f(p):>11} {f(b):>10} {f(res.get('pisco_tps'),1):>12} {f(res.get('base_tps'),1):>11} {f(sp,2):>8}")
        rows.append(res)
    print("\n(seq/s = throughput; speedup = base_s/it / pisco_s/it at same batch; OOM = didn't fit)")


if __name__ == "__main__":
    main()
