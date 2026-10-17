"""SIMULATED compression in vLLM: the plain Qwen3.5-4B fed random-token prompts of the RAW length vs the PISCO length.
Decode/prefill compute depends only on prompt length (same architecture, PISCO's LoRA merges away), so this measures
the serving-speed effect of compression with vLLM's batching, without PISCO code. The vision encoder is not run in
either arm (raw = "ViT features already cached"); scripts/bench_vllm.py --arm raw_pixels adds that cost.

Lengths per K (images/query) are read from the HF benchmark (expQ/bench/<tag>_base.json / <tag>_pisco_*.json, B=1 row),
i.e. the real tokenized prompt lengths: e.g. nav K=8 -> 3231 raw vs 295 PISCO tokens.

  python scripts/bench_vllm_sim.py --raw_json expQ/bench/nav_base.json --pisco_json expQ/bench/nav_pisco_r16.json \
      --out expQ/bench/vllm/nav_sim.json
"""
import argparse
import json
import os
import time

import torch
from vllm import LLM, SamplingParams


def lengths(path):
    d = json.load(open(path))
    return {r["K"]: r["prompt_tokens"] for r in d["results"] if r["B"] == 1 and not r.get("oom")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--raw_json", required=True)
    ap.add_argument("--pisco_json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ns", default="1,16,64,256,1024")
    ap.add_argument("--gs", default="1,4,32")
    ap.add_argument("--max_num_seqs", type=int, default=1024)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    args = ap.parse_args()
    L_raw, L_pisco = lengths(args.raw_json), lengths(args.pisco_json)
    Ks = sorted(set(L_raw) & set(L_pisco))
    print(f"[sim] lengths raw {L_raw} pisco {L_pisco}", flush=True)

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
              max_num_seqs=args.max_num_seqs, enable_prefix_caching=False, seed=0,
              limit_mm_per_prompt={"image": 0, "video": 0})   # text only: no vision encoder, no mm profiling
    g = torch.Generator().manual_seed(0)

    def reqs(L, N):   # distinct random prompts so nothing can be shared between requests
        return [{"prompt_token_ids": torch.randint(1000, 150000, (L,), generator=g).tolist()} for _ in range(N)]

    def sp(G):
        return SamplingParams(temperature=0.0, max_tokens=G, min_tokens=G, ignore_eos=True)

    llm.generate(reqs(512, 8), sp(4), use_tqdm=False)   # warmup
    results = []
    for K in Ks:
        for N in [int(n) for n in args.ns.split(",")]:
            for G in [int(x) for x in args.gs.split(",")]:
                for arm, L in (("raw", L_raw[K]), ("pisco", L_pisco[K])):
                    llm.generate(reqs(L, min(N, 4)), sp(G), use_tqdm=False)   # untimed: compile/tune this shape first
                    rq = reqs(L, N)
                    torch.cuda.synchronize(); t = time.perf_counter()
                    outs = llm.generate(rq, sp(G), use_tqdm=False)
                    torch.cuda.synchronize(); dt = time.perf_counter() - t
                    assert all(len(o.outputs[0].token_ids) == G for o in outs)
                    r = {"arm": arm, "K": K, "prompt_len": L, "N": N, "G": G, "wall_s": dt, "req_per_s": N / dt,
                         "out_tok_per_s": N * G / dt}
                    results.append(r)
                    print(json.dumps(r), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"model": args.model, "lengths": {"raw": L_raw, "pisco": L_pisco}, "results": results},
              open(args.out, "w"), indent=1)
    print(f"[sim] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
