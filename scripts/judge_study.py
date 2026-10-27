"""Run one judge condition over the judge-reliability pool (scripts/build_judge_pool.py) with vLLM.

  python scripts/judge_study.py --pool expQ/judge_study/pool.json --model Qwen/Qwen3-8B \
      [--think] [--prompt project|rubric] --out expQ/judge_study/q8_project.json

Conditions
  --prompt project  the project's Yes/No prompt (judge_teacher.judge_prompt, bergen's), verdict = "yes" in reply
  --prompt rubric   an explicit rubric; the judge answers with a JSON {"verdict": "yes"|"no", "reason": "..."}
  --think           Qwen3 thinking mode on (the final answer after </think> is parsed); off = the current judge
Output: {"model", "prompt", "think", "verdicts": {id: 0/1/None}, "raw": {id: reply}, "n_unparseable"}.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from judge_prompts import PROMPTS, build_prompt, parse_verdict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", choices=PROMPTS, default="project")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--max_tokens", type=int, default=None)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    pool = json.load(open(args.pool))
    if args.limit:
        pool = pool[: args.limit]
    msgs = [build_prompt(args.prompt, it["query"], it["answers"], it["prediction"]) for it in pool]
    max_tokens = args.max_tokens or (1024 if args.think else 64)
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
              seed=0, limit_mm_per_prompt={"image": 0, "video": 0} if "gemma" in args.model.lower() or "Ministral" in args.model else None)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    kw = {}
    if "qwen3" in args.model.lower():
        kw["chat_template_kwargs"] = {"enable_thinking": bool(args.think)}
    outs = llm.chat(msgs, sp, use_tqdm=True, **kw)
    verdicts, raw, weird, truncated = {}, {}, 0, 0
    for it, o in zip(pool, outs):
        text = o.outputs[0].text
        v = parse_verdict(text, args.prompt)
        if v is None:
            weird += 1
        if o.outputs[0].finish_reason == "length":
            truncated += 1
        verdicts[str(it["id"])] = v
        raw[str(it["id"])] = text[-400:]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"model": args.model, "prompt": args.prompt, "think": bool(args.think), "n": len(pool),
               "n_unparseable": weird, "n_truncated": truncated, "verdicts": verdicts, "raw": raw},
              open(args.out, "w"), indent=0, ensure_ascii=False)
    yes = sum(1 for v in verdicts.values() if v == 1)
    print(f"[judge_study] {args.model} prompt={args.prompt} think={args.think}: {len(pool)} items, "
          f"yes rate {yes / len(pool):.3f}, unparseable {weird}, truncated {truncated} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
