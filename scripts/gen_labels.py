#!/usr/bin/env python
"""Regenerate short/raw QA answers into complete, self-contained SENTENCE labels.

Why: PISCO stops cleanly only when trained on uniform sentence-style labels (like KILT's
mistral_label). ARC's raw spans ("hamlet", "3") broke stopping (2a: 89% runaway turns). We rephrase
each (question, gold-answer) into one grounded sentence — the gold answer is already correct, so this
is a cheap "declarativization", not fact generation (docs NOT fed to the label LLM -> fast & faithful).

Input:  an ARC jsonl (question, passages, answer) from ARC-Encoder_ft.zip.
Output: jsonl rows in PISCO schema {docs, query, mistral_label, source} at --out.

Usage: python scripts/gen_labels.py --source freebaseqa --n 12000 --out arc_ft_data/regen/freebaseqa.jsonl
"""
import argparse, json, zipfile, io, os, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ZIP = "/beegfs/scratch/user/hdejean/arc_ft_data/ARC-Encoder_ft.zip"
SYS = "You rewrite short answers into one complete, self-contained sentence."
USER = ("Question: {q}\nReference answer: {a}\n\nWrite the answer to the question as ONE complete, "
        "self-contained declarative sentence that includes the reference answer verbatim. "
        "Output only the sentence, nothing else.")

def read_src(name, n):
    out = []
    with zipfile.ZipFile(ZIP) as z, z.open(f"ARC-Encoder_ft/{name}.jsonl") as f:
        for line in io.TextIOWrapper(f, encoding="utf-8"):
            if line.strip():
                out.append(json.loads(line))
            if len(out) >= n:
                break
    return out

def clean(t):
    if "</think>" in t:
        t = t.split("</think>", 1)[1]
    # kill any runaway chat-turn continuation (the exact failure we are preventing)
    for m in ["\nuser", "\nassistant", "\nUser", "\nAssistant", "<|im_end|>", "<think>"]:
        i = t.find(m)
        if i != -1:
            t = t[:i]
    return t.strip().strip('"').strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto").to(dev).eval()
    print(f"gen_labels: source={args.source} n={args.n} model={args.model}")

    rows = read_src(args.source, args.n)
    def gold_of(o):
        a = o["answer"]
        return (a[0] if isinstance(a, list) and a else a) if not isinstance(a, str) else a
    def docs_of(o):
        return o["passages"] if "passages" in o else [o.get("passage", "")]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    kept = 0; fell_back = 0
    with open(args.out, "w") as fout:
        for s in range(0, len(rows), args.batch_size):
            batch = rows[s:s + args.batch_size]
            prompts = []
            for o in batch:
                msgs = [{"role": "system", "content": SYS},
                        {"role": "user", "content": USER.format(q=o["question"], a=gold_of(o))}]
                prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                               add_generation_prompt=True, enable_thinking=False))
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(dev)
            with torch.inference_mode():
                out = model.generate(**enc, do_sample=False, top_p=None, max_new_tokens=64,
                                     pad_token_id=tok.pad_token_id)
            gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for o, g in zip(batch, gen):
                gold = str(gold_of(o)); label = clean(g)
                # Guard for STOPPING (the point of this script): accept any fluent, non-degenerate
                # sentence. Strict gold-containment was too aggressive (numeric answers: "3" vs
                # "three" -> false reject -> raw short fallback, reintroducing the short-span risk).
                # The prompt anchors the gold verbatim + greedy decode -> the model stays faithful
                # (freebaseqa: 0% fallback). Fall back only if empty / too short / not sentence-like.
                sentence_like = len(label) >= 12 and (label[-1] in ".!?" or len(label.split()) >= 4)
                if not sentence_like:
                    label = gold if gold.endswith((".", "!", "?")) else gold + "."
                    fell_back += 1
                fout.write(json.dumps({"docs": docs_of(o), "query": o["question"],
                                       "mistral_label": label, "source": args.source}) + "\n")
                kept += 1
            if s % (args.batch_size * 20) == 0:
                print(f"  {kept}/{len(rows)} (fallbacks={fell_back})", flush=True)
    print(f"DONE {args.source}: wrote {kept}, fallbacks={fell_back} ({100*fell_back/max(1,kept):.0f}%) -> {args.out}")

if __name__ == "__main__":
    main()
