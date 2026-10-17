#!/usr/bin/env python
"""Judge the same teacher answers with an OpenAI model, to cross-check the local judge.

Companion to scripts/judge_teacher.py. Reads that script's per-example dump (so the
rows are identical, matched on q_id), asks an OpenAI model the SAME Yes/No question,
and prints a three-way comparison: containment-EM vs local judge vs API judge.

Point of the exercise: containment-EM was shown to track gold ANSWER LENGTH rather than
teacher correctness (job 163367), and the local Qwen3-8B judge disagreed with containment
in both directions on triviaqa. With three independent scorers on the same rows, the odd
one out is identifiable. Calibrate on ~1200 rows for a few cents before relabelling 453K.

Needs OPENAI_API_KEY exported (note: ~/.bashrc sets it WITHOUT export, so
`source ~/.bashrc && export OPENAI_API_KEY` is required before running).

Usage:
  python scripts/judge_teacher_openai.py --local_json outputs/judge_teacher_<jid>.json \
      --model gpt-4.1 --output_path outputs/judge_3way.json
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openai

# $ per 1M tokens (input, output). Verify against the current OpenAI dashboard.
PRICING_PER_M = {
    "gpt-4o-mini": (0.15, 0.60), "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00), "gpt-4o": (2.50, 10.00),
}


def judge_prompt(question, gold_answers, prediction):
    """Verbatim from scripts/llm_judge.py (bergen models/evaluators/openai.py)."""
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


def judge_one(client, model, rec):
    msgs = judge_prompt(rec["query"], rec["gold"], rec["teacher"])
    for attempt in range(5):
        try:
            r = client.chat.completions.create(model=model, messages=msgs, max_tokens=5)
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    txt = (r.choices[0].message.content or "").lower()
    return (1 if "yes" in txt else 0,
            1 if ("yes" not in txt and "no" not in txt) else 0,
            r.usage.prompt_tokens, r.usage.completion_tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_json", required=True, help="output of scripts/judge_teacher.py")
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit_per_source", type=int, default=None)
    ap.add_argument("--output_path", default="outputs/judge_3way.json")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not exported. ~/.bashrc sets it without `export`; "
                         "run: source ~/.bashrc && export OPENAI_API_KEY")

    local = json.load(open(args.local_json))["by_source"]
    client = openai.OpenAI()

    out, tot_in, tot_out, weird = {}, 0, 0, 0
    print(f"{'source':<16}{'n':>5}{'contain%':>10}{'local%':>9}{'api%':>8}   role")
    for src, blob in local.items():
        recs = blob.get("per_example") or []
        if args.limit_per_source:
            recs = recs[: args.limit_per_source]
        if not recs:
            continue
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            res = list(pool.map(lambda r: judge_one(client, args.model, r), recs))
        api = [x[0] for x in res]
        weird += sum(x[1] for x in res)
        tot_in += sum(x[2] for x in res); tot_out += sum(x[3] for x in res)
        c = [r["containment"] for r in recs]
        l = [r["judge"] for r in recs]
        n = len(recs)
        print(f"{src:<16}{n:>5}{100*sum(c)/n:>9.1f}%{100*sum(l)/n:>8.1f}%{100*sum(api)/n:>7.1f}%"
              f"   {'CONTROL' if blob.get('control') else 'under test'}")
        out[src] = dict(n=n, containment=sum(c), local=sum(l), api=sum(api),
                        control=blob.get("control"),
                        per_example=[dict(q_id=r["q_id"], query=r["query"], gold=r["gold"],
                                          teacher=r["teacher"], containment=r["containment"],
                                          local=r["judge"], api=a)
                                     for r, a in zip(recs, api)])

    pin, pout = PRICING_PER_M.get(args.model, (0, 0))
    cost = tot_in / 1e6 * pin + tot_out / 1e6 * pout
    print(f"\nunparseable: {weird}   tokens in/out: {tot_in}/{tot_out}   cost: ${cost:.4f}")
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    json.dump({"model": args.model, "cost_usd": cost, "tokens_in": tot_in,
               "tokens_out": tot_out, "by_source": out}, open(args.output_path, "w"), indent=2)
    print(f"wrote {args.output_path}")


if __name__ == "__main__":
    main()
