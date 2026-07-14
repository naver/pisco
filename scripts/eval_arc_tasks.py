#!/usr/bin/env python
"""Direct capability check: can a PISCO checkpoint SUMMARIZE and TRANSLATE (not just QA)?

Runs generation on WikiSum (summarization) and short_translation examples from the ARC zip,
using the NEUTRAL system prompt (matching how the per-task-prompt ARC finetune was trained).
Scores summarization with ROUGE-L (inline LCS, no dep); prints translation samples + token overlap.
Usage: python scripts/eval_arc_tasks.py --checkpoint expQ/155363/model --n 60
"""
import argparse, json, zipfile, io, torch
from pisco.model import PISCO
from pisco.collator_utils import add_memory_tokens_to_inputs, chunk_list

ZIP = "/beegfs/scratch/user/hdejean/arc_ft_data/ARC-Encoder_ft.zip"
NEUTRAL = "You are a helpful assistant. Follow the instruction using the provided text."

def read_n(name, n):
    out = []
    with zipfile.ZipFile(ZIP) as z, z.open(f"ARC-Encoder_ft/{name}.jsonl") as f:
        for line in io.TextIOWrapper(f, encoding="utf-8"):
            if line.strip():
                out.append(json.loads(line))
            if len(out) >= n:
                break
    return out

def lcs(a, b):
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            dp[i][j] = dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[len(a)][len(b)]

def rougeL_f1(pred, ref):
    p, r = pred.split(), ref.split()
    if not p or not r: return 0.0
    l = lcs(p, r)
    prec, rec = l/len(p), l/len(r)
    return 0.0 if prec+rec == 0 else 2*prec*rec/(prec+rec)

@torch.inference_mode()
def gen(model, dev, doc, query, cml=512, max_new=128):
    ct, dt = model.compressor_tokenizer, model.decoder_tokenizer
    ids = ct(doc, add_special_tokens=False, truncation=False)["input_ids"]
    chunks = chunk_list(ids, chunk_length=cml, chunk_overlap=0)[:16]
    chunks, n_mems = add_memory_tokens_to_inputs(chunks, ct, model.compr_rate)
    background = dt.mem_token * int(sum(n_mems))
    msgs = [{"role":"system","content":NEUTRAL},
            {"role":"user","content":f"\n\nBackground:{background}\n Question: {query}"}]
    prompt = dt.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    di = dt(prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=2048)
    # trim compressor mems to match decoder mems that survived truncation
    dmems = int((di["input_ids"]==dt.mem_token_id).sum())
    ci = ct.pad({"input_ids": chunks}, padding="longest", return_tensors="pt")
    emb = model.compress(ci["input_ids"].to(dev), ci["attention_mask"].to(dev))
    de = model.replace_embeddings(emb, di["input_ids"].to(dev))
    out = model.decoder.generate(inputs_embeds=de, attention_mask=di["attention_mask"].to(dev),
                                 do_sample=False, top_p=None, max_new_tokens=max_new)
    t = dt.batch_decode(out, skip_special_tokens=True)[0]
    return t.split("</think>",1)[1].strip() if "</think>" in t else t.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="PISCO checkpoint (pisco mode)")
    ap.add_argument("--mode", choices=["pisco", "base"], default="pisco")
    ap.add_argument("--base_model_name", default=None, help="HF model for uncompressed base mode, e.g. Qwen/Qwen3.5-4B")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--cml", type=int, default=512)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "base":
        # UNCOMPRESSED ceiling: feed the FULL document text to the decoder, no compression.
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = args.base_model_name
        btok = AutoTokenizer.from_pretrained(name)
        bmodel = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto").to(dev).eval()
        print(f"loaded BASE (uncompressed) {name}")
        @torch.inference_mode()
        def gen_fn(doc, query, max_new=128):
            msgs = [{"role":"system","content":NEUTRAL},
                    {"role":"user","content":f"\n\nBackground:\n{doc}\n Question: {query}"}]
            prompt = btok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ins = btok(prompt, return_tensors="pt", truncation=True, max_length=8192).to(dev)
            out = bmodel.generate(**ins, do_sample=False, top_p=None, max_new_tokens=max_new)
            t = btok.decode(out[0][ins["input_ids"].shape[1]:], skip_special_tokens=True)
            return t.split("</think>",1)[1].strip() if "</think>" in t else t.strip()
    else:
        model = PISCO.from_pretrained(args.checkpoint).to(dev).eval()
        print(f"loaded {args.checkpoint} compr_rate={model.compr_rate}")
        def gen_fn(doc, query, max_new=128):
            return gen(model, dev, doc, query, cml=args.cml, max_new=max_new)

    print("\n===== SUMMARIZATION (WikiSum, ROUGE-L) =====")
    rs = []
    ws = read_n("wikisum", args.n)
    for i, ex in enumerate(ws):
        pred = gen_fn(ex["passage"], ex["question"])
        rs.append(rougeL_f1(pred, ex["answer"]))
        if i < 3:
            print(f"[{i}] PRED: {pred[:200]}\n    REF : {ex['answer'][:200]}\n    RL={rs[-1]:.3f}")
    print(f"WikiSum ROUGE-L (n={len(rs)}): {sum(rs)/len(rs):.3f}")

    print("\n===== TRANSLATION (short_translation) =====")
    ts, ov = read_n("short_translation", min(args.n, 20)), []
    for i, ex in enumerate(ts):
        pred = gen_fn(ex["passage"], ex["question"])
        rp, rr = set(pred.split()), set(ex["answer"].split())
        ov.append(len(rp & rr)/max(1, len(rr)))
        if i < 4:
            print(f"[{i}] Q: {ex['question']}\n    PRED: {pred[:160]}\n    REF : {ex['answer'][:160]}\n    tok-overlap={ov[-1]:.2f}")
    print(f"Translation token-overlap w/ ref (n={len(ov)}): {sum(ov)/len(ov):.3f}")

if __name__ == "__main__":
    main()
