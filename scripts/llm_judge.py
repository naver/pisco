#!/usr/bin/env python
"""LLM-as-a-judge (OpenAI) for bergen RAG predictions.

Scores each prediction for *semantic correctness* (Yes/No) against the gold
answer(s) — a fairer measure than EM/F1 for verbose generators like PISCO.

Reads a bergen run's `eval_dev_out.json` (a list of dicts with `question`,
`label` [gold answers], `response` [prediction]), asks an OpenAI model to judge
each, and writes `eval_dev_llmjudge.json` next to it: {model, n, accuracy,
cost, per_example}.

Reuses the exact judge prompt from bergen's models/evaluators/openai.py.

Usage:
  python scripts/llm_judge.py --exp_dir /beegfs/.../bergen/expPISCO \
      --prefix mt8 --datasets kilt_nq sciq [--model gpt-4o-mini] [--limit 20]

The key must be in $OPENAI_API_KEY; the corporate https_proxy is honored by the
openai SDK's httpx client automatically.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai

# input $/1K tokens, output $/1K tokens — extend as needed. Judge output is tiny.
PRICING = {
    "gpt-4o-mini": (0.00015, 0.00060),
    "gpt-4o":      (0.0025,  0.0100),
    "gpt-4.1-mini":(0.0004,  0.0016),
    "gpt-4.1":     (0.0020,  0.0080),
    "gpt-5":       (0.00125, 0.0100),
    "gpt-5-mini":  (0.00025, 0.0020),
    "gpt-5-nano":  (0.00005, 0.0004),
}


def cost_of(usage, model):
    p = PRICING.get(model)
    if p is None:
        return 0.0
    return round(usage.prompt_tokens * p[0] / 1000 + usage.completion_tokens * p[1] / 1000, 6)


def judge_prompt(question, gold_answers, prediction):
    # gold_answers is a list of acceptable answers; show them all
    gold = " | ".join(str(a) for a in gold_answers) if isinstance(gold_answers, list) else str(gold_answers)
    return [
        {"role": "system", "content": "You are an evaluation tool. Just answer by {Yes} or {No}."},
        {"role": "user", "content": (
            "Here is a question, a golden answer and an AI-generated answer. Can you judge "
            "whether the AI-generated answer is correct according to the question and golden "
            "answer, simply answer {Yes} or {No}.\n"
            f" Question: {question}. \ngolden answer: {gold} \n Generated answer: {prediction}.\nResponse:"
        )},
    ]


def judge_one(client, model, ex, create_kwargs):
    """Judge a single example; returns (score, weird, cost, raw). Retries transient errors."""
    msgs = judge_prompt(ex.get("question", ""), ex.get("label", []), ex.get("response", ""))
    for attempt in range(5):
        try:
            r = client.chat.completions.create(model=model, messages=msgs, **create_kwargs)
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    text = (r.choices[0].message.content or "").lower()
    score = 1 if "yes" in text else 0
    weird = 1 if ("yes" not in text and "no" not in text) else 0
    return score, weird, cost_of(r.usage, model), text.strip()[:40]


def score_run(client, model, examples, create_kwargs, limit=None, workers=16, budget=None):
    """budget: dict {'spent': float, 'ceiling': float} shared across datasets/models.
    Stops submitting once budget['spent'] >= ceiling; returns partial + hit_ceiling flag."""
    items = examples[:limit] if limit else examples
    results = {}
    hit_ceiling = False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for i, ex in enumerate(items):
            if budget and budget["spent"] >= budget["ceiling"]:
                hit_ceiling = True
                break
            futures[pool.submit(judge_one, client, model, ex, create_kwargs)] = i
        for fut in as_completed(futures):
            i = futures[fut]
            out = fut.result()
            results[i] = out
            if budget is not None:
                budget["spent"] += out[2]
            if len(results) % 100 == 0:
                acc = sum(r[0] for r in results.values()) / len(results)
                sp = f" spent=${budget['spent']:.2f}" if budget else ""
                print(f"  [{len(results)}/{len(futures)}] acc={acc:.3f}{sp}", flush=True)

    idx = sorted(results)
    scores = [results[i][0] for i in idx]
    weird = sum(results[i][1] for i in idx)
    total_cost = sum(results[i][2] for i in idx)
    per_ex = [{"q_id": items[i].get("q_id"), "score": results[i][0], "judge_raw": results[i][3]} for i in idx]
    acc = sum(scores) / len(scores) if scores else 0.0
    return {"n": len(scores), "accuracy": round(acc, 4), "weird": weird,
            "cost_usd": round(total_cost, 4), "per_example": per_ex, "hit_ceiling": hit_ceiling}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", default="/beegfs/scratch/user/hdejean/bergen/expPISCO")
    ap.add_argument("--prefix", required=True, help="run prefix, e.g. mt8 (dir = <prefix>_<dataset>)")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--limit", type=int, default=None, help="judge only first N examples (cheap test)")
    ap.add_argument("--workers", type=int, default=16, help="concurrent API calls")
    ap.add_argument("--out_name", default=None, help="default: eval_dev_llmjudge_<modeltag>.json")
    ap.add_argument("--max_cost", type=float, default=25.0, help="HARD ceiling ($) across this + shared budget_file")
    ap.add_argument("--budget_file", default=None, help="shared JSON ledger so multiple runs share one ceiling")
    ap.add_argument("--reasoning_effort", default="minimal", help="for gpt-5/o* reasoning models")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")
    client = openai.OpenAI()  # reads OPENAI_API_KEY + honors https_proxy via httpx

    # reasoning models (gpt-5*, o1/o3/o4*) reject temperature and take reasoning_effort
    is_reasoning = args.model.startswith("gpt-5") or args.model[:2] in ("o1", "o3", "o4")
    create_kwargs = {"reasoning_effort": args.reasoning_effort} if is_reasoning else {"temperature": 0}

    modeltag = args.model.replace("/", "_").replace(".", "")
    out_name = args.out_name or f"eval_dev_llmjudge_{modeltag}.json"

    # shared budget ledger (global hard ceiling)
    spent0 = 0.0
    if args.budget_file and os.path.exists(args.budget_file):
        try:
            spent0 = json.load(open(args.budget_file)).get("spent", 0.0)
        except Exception:
            spent0 = 0.0
    budget = {"spent": spent0, "ceiling": args.max_cost}
    print(f"model={args.model} kwargs={create_kwargs} out={out_name} "
          f"budget: spent=${spent0:.2f} ceiling=${args.max_cost:.2f}", flush=True)

    summary = {}
    for ds in args.datasets:
        if budget["spent"] >= budget["ceiling"]:
            print(f"!! CEILING ${budget['ceiling']} reached (spent ${budget['spent']:.2f}) — stopping before {ds}", flush=True)
            break
        run_dir = os.path.join(args.exp_dir, f"{args.prefix}_{ds}")
        out_json = os.path.join(run_dir, "eval_dev_out.json")
        if not os.path.exists(out_json):
            print(f"SKIP {args.prefix}_{ds}: no {out_json}", flush=True)
            continue
        examples = json.load(open(out_json))
        print(f"=== {args.prefix}_{ds}: {len(examples)} examples, model={args.model}"
              f"{' (limit '+str(args.limit)+')' if args.limit else ''} ===", flush=True)
        res = score_run(client, args.model, examples, create_kwargs,
                        limit=args.limit, workers=args.workers, budget=budget)
        res["model"] = args.model
        res["prefix"] = args.prefix
        res["dataset"] = ds
        save_path = os.path.join(run_dir, out_name)
        json.dump(res, open(save_path, "w"), indent=2)
        summary[ds] = {"acc": res["accuracy"], "n": res["n"], "cost": res["cost_usd"]}
        print(f"  -> acc={res['accuracy']} n={res['n']} cost=${res['cost_usd']} weird={res['weird']}"
              f"  saved {save_path}", flush=True)
        if args.budget_file:  # persist shared spend after each dataset
            json.dump({"spent": round(budget["spent"], 4)}, open(args.budget_file, "w"))
        if res.get("hit_ceiling"):
            print(f"!! CEILING ${budget['ceiling']} hit mid-dataset ({ds}) — stopping", flush=True)
            break

    print("\n=== SUMMARY (LLM-judge accuracy) ===")
    for ds, v in summary.items():
        print(f"  {ds:18s} acc={v['acc']:.3f}  n={v['n']}  cost=${v['cost']}")
    if summary:
        tot = sum(v["cost"] for v in summary.values())
        print(f"  {'MEAN':18s} acc={sum(v['acc'] for v in summary.values())/len(summary):.3f}"
              f"  total_cost=${tot:.3f}")


if __name__ == "__main__":
    main()
