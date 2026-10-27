"""vLLM throughput/latency benchmark: PISCO decoder on precomputed <MEM> prompt embeddings vs raw Qwen3.5-4B on images.
Runs in the vLLM env (pixi --manifest-path .../Reranking/vllm_env/pixi.toml); no pisco import. Inputs come from
scripts/vllm_prep_pisco.py.

Arms (one per process):
  pisco       exported text-only decoder (Qwen3_5ForCausalLM), prompt_embeds = compressed pages/frames
  raw_lmonly  SAME decoder, prompt_embeds tiled to the raw-image prompt length: the LM cost of the raw context with the
              vision encoder cost removed (as if ViT features were cached). Timing only, outputs meaningless.
  raw_pixels  Qwen/Qwen3.5-4B VLM with the real images; the vision encoder runs per request. Every image is made unique
              (one pixel changed) so vLLM's multimodal caches cannot reuse work across requests.
For each K (images/query), N (requests submitted together, vLLM schedules them with continuous batching) and G
(generated tokens, ignore_eos so all arms generate exactly G): wall time of llm.generate -> requests/s, output tokens/s;
G=1 gives time to first token for the whole batch. --acc writes greedy predictions for an accuracy-parity check.
"""
import argparse
import json
import os
import pathlib
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # registry entry below must live in the engine process

import torch  # noqa: E402
from vllm import LLM, ModelRegistry, SamplingParams  # noqa: E402

# vLLM ships a text-only Qwen3.5 class but does not register it, and it lacks the hybrid-model hooks (IsHybrid + the
# gated-delta-net state shape/dtype/copy methods) that the image class has. Without them vLLM cannot size the
# attention and linear-attention cache pages together ("page size ... not divisible"). Borrow them from the image class.
from vllm.model_executor.models import config as _mcfg  # noqa: E402
from vllm.model_executor.models.interfaces import IsHybrid  # noqa: E402
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration as _VLM  # noqa: E402


class PiscoQwen3_5ForCausalLM(Qwen3_5ForCausalLM, IsHybrid):
    get_mamba_state_dtype_from_config = classmethod(_VLM.get_mamba_state_dtype_from_config.__func__)
    get_mamba_state_shape_from_config = classmethod(_VLM.get_mamba_state_shape_from_config.__func__)
    get_mamba_state_copy_func = classmethod(_VLM.get_mamba_state_copy_func.__func__)


ModelRegistry.register_model("Qwen3_5ForCausalLM", PiscoQwen3_5ForCausalLM)
_mcfg.MODELS_CONFIG_MAP["Qwen3_5ForCausalLM"] = _mcfg.MODELS_CONFIG_MAP["Qwen3_5ForConditionalGeneration"]


def sp(G):
    return SamplingParams(temperature=0.0, max_tokens=G, min_tokens=G, ignore_eos=True)


def unique_image(img, salt):
    im = img.convert("RGB").copy()
    w, h = im.size
    px = im.load()
    r, g, b = px[salt % w, (salt // w) % h]
    px[salt % w, (salt // w) % h] = ((r + 1 + salt) % 256, g, b)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["pisco", "raw_lmonly", "raw_pixels"], required=True)
    ap.add_argument("--model", required=True, help="exported decoder dir (pisco/raw_lmonly) or Qwen/Qwen3.5-4B (raw_pixels)")
    ap.add_argument("--inputs", required=True, help="bench_<tag>.pt from vllm_prep_pisco.py")
    ap.add_argument("--acc", default=None, help="acc_<tag>.pt: write greedy predictions (pisco arm)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="4,8,16")
    ap.add_argument("--ns", default="1,16,64,256")
    ap.add_argument("--gs", default="1,4,32")
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--kv_cache_dtype", default="auto", help="fp8: quantised KV cache (the 8 full-attention layers)")
    ap.add_argument("--mamba_ssm_cache_dtype", default="auto",
                    help="float16: half-size linear-attention state (Qwen3.5 stores it in float32)")
    ap.add_argument("--attention_backend", default=None, help="e.g. TRITON_ATTN (fp8 KV on A100: FA2 has no fp8)")
    ap.add_argument("--spec_mtp", type=int, default=0,
                    help="speculative decoding with the checkpoint's MTP head, N draft tokens (scripts/add_mtp_head.py)")
    args = ap.parse_args()

    B = torch.load(args.inputs, weights_only=False)
    common = dict(dtype="bfloat16", gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
                  enable_prefix_caching=False, seed=0, kv_cache_dtype=args.kv_cache_dtype,
                  mamba_ssm_cache_dtype=args.mamba_ssm_cache_dtype)
    if args.attention_backend:
        common["attention_backend"] = args.attention_backend
    if args.spec_mtp:
        common["speculative_config"] = {"method": "mtp", "num_speculative_tokens": args.spec_mtp}
    if args.arm == "raw_pixels":
        llm = LLM(model=args.model, limit_mm_per_prompt={"image": max(int(k) for k in args.ks.split(","))},
                  mm_processor_cache_gb=0, **common)
        import datasets
        ds = datasets.load_from_disk(B["data"])
        ds = ds["train"] if isinstance(ds, datasets.DatasetDict) else ds
        pool = [ds[r]["doc_images"][i] for r, i in B["pool_refs"]]
    else:
        llm = LLM(model=args.model, enable_prompt_embeds=True, **common)
    try:
        sc = llm.llm_engine.vllm_config.scheduler_config
        print(f"[vllm] arm={args.arm} model={args.model} max_num_seqs={sc.max_num_seqs} "
              f"max_num_batched_tokens={sc.max_num_batched_tokens}", flush=True)
    except Exception as e:   # introspection only
        print(f"[vllm] arm={args.arm} model={args.model} (scheduler config unavailable: {e})", flush=True)

    salt = [0]

    def make(K, N):
        d = B[K]
        reqs = []
        for qi in range(N):
            if args.arm == "pisco":
                reqs.append({"prompt_embeds": d["embeds"][qi]})
            elif args.arm == "raw_lmonly":
                e = d["embeds"][qi]
                L = d["raw_len"][qi]
                reqs.append({"prompt_embeds": e.repeat((L + e.shape[0] - 1) // e.shape[0], 1)[:L].contiguous()})
            else:
                imgs = []
                for i in d["image_refs"][qi]:
                    salt[0] += 1
                    imgs.append(unique_image(pool[i], salt[0]))
                reqs.append({"prompt": d["raw_prompts"][qi], "multi_modal_data": {"image": imgs}})
        return reqs

    results = []
    if args.acc:
        A = torch.load(args.acc, weights_only=False)
        t = time.perf_counter()
        outs = llm.generate([{"prompt_embeds": e} for e in A["embeds"]], SamplingParams(temperature=0.0, max_tokens=32))
        dt = time.perf_counter() - t
        preds = [{"query_id": q, "prediction": o.outputs[0].text.strip()} for q, o in zip(A["query_ids"], outs)]
        json.dump({"data": A["data"], "wall_s": dt, "records": preds}, open(args.out.replace(".json", "_acc.json"), "w"), indent=1)
        print(f"[acc] {len(preds)} predictions in {dt:.1f} s ({len(preds)/dt:.1f} q/s, whole set in one call)", flush=True)

    llm.generate(make(int(args.ks.split(",")[0]), 4), sp(4))   # warmup (graphs, kernels)
    for K in [int(k) for k in args.ks.split(",")]:
        for N in [int(n) for n in args.ns.split(",")]:
            for G in [int(g) for g in args.gs.split(",")]:
                llm.generate(make(K, min(N, 4)), sp(G), use_tqdm=False)   # untimed: compile/tune this shape first
                reqs = make(K, N)
                torch.cuda.synchronize(); t = time.perf_counter()
                outs = llm.generate(reqs, sp(G), use_tqdm=False)
                torch.cuda.synchronize(); dt = time.perf_counter() - t
                plen = [len(o.prompt_token_ids) if o.prompt_token_ids is not None else None for o in outs]
                r = {"arm": args.arm, "K": K, "N": N, "G": G, "wall_s": dt, "req_per_s": N / dt,
                     "out_tok_per_s": N * G / dt, "prompt_len": plen[0] if plen[0] is not None else int(B[K]["embeds"][0].shape[0]
                     if args.arm == "pisco" else B[K]["raw_len"][0]),
                     "gen_check": len(outs[0].outputs[0].token_ids)}
                results.append(r)
                print(json.dumps(r), flush=True)
    info = {"kv_cache_dtype": args.kv_cache_dtype, "spec_mtp": args.spec_mtp,
            "mamba_ssm_cache_dtype": args.mamba_ssm_cache_dtype, "attention_backend": args.attention_backend}
    try:   # KV capacity and speculative-decoding counters (introspection only)
        cc = llm.llm_engine.vllm_config.cache_config
        info.update(num_gpu_blocks=cc.num_gpu_blocks, block_size=cc.block_size)
        info["metrics"] = {m.name: getattr(m, "value", None) for m in llm.get_metrics()
                           if "spec_decode" in m.name or "gpu_cache" in m.name or "kv_cache" in m.name}
    except Exception as e:
        info["introspection_error"] = repr(e)
    print(f"[vllm] {json.dumps(info)}", flush=True)
    json.dump({"arm": args.arm, "model": args.model, "inputs": args.inputs, **info, "results": results},
              open(args.out, "w"), indent=1)
    print(f"[vllm] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
