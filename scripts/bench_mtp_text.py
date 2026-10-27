"""Speed-up of MTP speculative decoding on Qwen3.5-4B with token prompts (vLLM), for long answers.

vLLM 0.19.1 refuses MTP with prompt embeddings (the draft head needs the prompt's token ids; <MEM> positions have
none), so PISCO cannot use it yet. This measures what it would give once supported: the same decoder family, real
questions from a doc eval set turned into open-ended prompts ("Explain ... in detail"), N requests at once, G tokens.
Runs in the vLLM env.

  python scripts/bench_mtp_text.py --data eval_data/mm_docqa_1500_1m --spec 0 --out expQ/archq/bench/mtp_text_off.json
"""
import argparse
import json
import os
import time

import torch
from vllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--spec", type=int, default=0, help="MTP draft tokens (0 = off)")
    ap.add_argument("--ns", default="1,16,64")
    ap.add_argument("--gs", default="64,256")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import datasets
    ds = datasets.load_from_disk(args.data)
    ds = ds["train"] if isinstance(ds, datasets.DatasetDict) else ds
    queries = ds.select_columns(["query"])["query"][:256]
    extra = {"speculative_config": {"method": "mtp", "num_speculative_tokens": args.spec}} if args.spec else {}
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096, seed=0,
              enable_prefix_caching=False, limit_mm_per_prompt={"image": 0, "video": 0}, disable_log_stats=False, **extra)
    tok = llm.get_tokenizer()

    def prompt(q):
        msgs = [{"role": "user", "content": f"Explain in detail what one needs to know to answer this question: {q}"}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    prompts = [prompt(q) for q in queries]
    llm.generate(prompts[:4], SamplingParams(temperature=0.0, max_tokens=16), use_tqdm=False)   # warmup
    res = []
    for N in [int(n) for n in args.ns.split(",")]:
        for G in [int(g) for g in args.gs.split(",")]:
            sp = SamplingParams(temperature=0.0, max_tokens=G, min_tokens=G, ignore_eos=True)
            llm.generate(prompts[:min(N, 4)], sp, use_tqdm=False)
            torch.cuda.synchronize(); t = time.perf_counter()
            outs = llm.generate(prompts[:N], sp, use_tqdm=False)
            torch.cuda.synchronize(); dt = time.perf_counter() - t
            r = {"spec": args.spec, "N": N, "G": G, "wall_s": dt, "out_tok_per_s": N * G / dt,
                 "ms_per_token_per_request": 1e3 * dt / G}
            res.append(r)
            print(json.dumps(r), flush=True)
    metrics = {}
    try:
        metrics = {m.name: getattr(m, "value", getattr(m, "values", None)) for m in llm.get_metrics() if "spec_decode" in m.name}
    except Exception as e:   # introspection only
        metrics = {"error": repr(e)}
    print(f"[mtp] metrics {json.dumps(metrics, default=str)}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"model": args.model, "spec": args.spec, "metrics": metrics, "results": res}, open(args.out, "w"), indent=1,
              default=str)


if __name__ == "__main__":   # vLLM spawns its engine process, which re-imports this module
    main()
