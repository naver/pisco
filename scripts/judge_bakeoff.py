#!/usr/bin/env python
"""Compare several OpenAI judge models on identical rows, against containment + a local judge.

There is no ground truth here, so "better judge" is argued from two signals:
  * agreement with containment on CONTROL sources (short, single-alias gold, where
    containment is known reliable) -- a good judge must not drift there;
  * behaviour on triviaqa, where containment is known to over-count via alias soup
    (29.7 gold aliases/row): a good judge should sit BELOW containment, but not as far
    below as an over-strict one.

Also reports measured token usage per model so cost can be priced from the dashboard --
model pricing is deliberately NOT hardcoded, since the gpt-5.x line postdates what this
script's author could verify.

Usage:
  python scripts/judge_bakeoff.py --local_json outputs/judge_teacher_<jid>.json \
     --models gpt-5.4-mini,gpt-5.4,gpt-5.6-sol --limit_per_source 200
"""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import openai

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def judge_prompt(question, gold, prediction):
    g = " | ".join(str(a) for a in gold) if isinstance(gold, list) else str(gold)
    return [
        {"role": "system", "content": "You are an evaluation tool. Just answer by {Yes} or {No}."},
        {"role": "user", "content": (
            "Here is a question, a golden answer and an AI-generated answer. Can you judge "
            "whether the AI-generated answer is correct according to the question and golden "
            "answer, simply answer {Yes} or {No}.\n"
            f" Question: {question}. \ngolden answer: {g} \n Generated answer: {prediction}.\nResponse:"
        )},
    ]


# Param style differs across families; probe once per model, then reuse.
_STYLES = [
    {"max_tokens": 8},
    {"max_completion_tokens": 16},
    {"max_completion_tokens": 512, "reasoning_effort": "low"},
]


def probe_style(client, model):
    msgs = judge_prompt("capital of France", ["Paris"], "The capital is Paris.")
    for kw in _STYLES:
        try:
            r = client.chat.completions.create(model=model, messages=msgs, **kw)
            if (r.choices[0].message.content or "").strip():
                return kw
        except Exception:
            continue
    raise SystemExit(f"no working parameter style for {model}")


def judge_one(client, model, kw, rec):
    msgs = judge_prompt(rec["query"], rec["gold"], rec["teacher"])
    for attempt in range(5):
        try:
            r = client.chat.completions.create(model=model, messages=msgs, **kw)
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    t = (r.choices[0].message.content or "").lower()
    u = r.usage
    rt = getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
    return (1 if "yes" in t else 0, 1 if ("yes" not in t and "no" not in t) else 0,
            u.prompt_tokens, u.completion_tokens, rt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_json", required=True)
    ap.add_argument("--prior_json", default=None, help="earlier run to merge (e.g. gpt-4.1 3-way)")
    ap.add_argument("--models", required=True)
    ap.add_argument("--limit_per_source", type=int, default=200)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--output_path", default="outputs/judge_bakeoff.json")
    args = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("export OPENAI_API_KEY first (~/.bashrc sets it without export)")

    local = json.load(open(args.local_json))["by_source"]
    prior = json.load(open(args.prior_json))["by_source"] if args.prior_json else {}
    client = openai.OpenAI()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    rows = {s: (b.get("per_example") or [])[: args.limit_per_source] for s, b in local.items()}
    rows = {s: r for s, r in rows.items() if r}
    controls = {s for s, b in local.items() if b.get("control")}

    scores, usage = {}, {}
    for m in models:
        kw = probe_style(client, m)
        print(f"[bakeoff] {m}: params {list(kw)}", flush=True)
        scores[m], usage[m] = {}, [0, 0, 0]
        for s, recs in rows.items():
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                res = list(pool.map(lambda r: judge_one(client, m, kw, r), recs))
            scores[m][s] = [x[0] for x in res]
            usage[m][0] += sum(x[2] for x in res)
            usage[m][1] += sum(x[3] for x in res)
            usage[m][2] += sum(x[4] for x in res)
        print(f"[bakeoff] {m}: done", flush=True)

    hdr = f"{'source':<16}{'n':>5}{'contain':>9}{'qwen8b':>8}"
    if prior: hdr += f"{'gpt-4.1':>9}"
    for m in models: hdr += f"{m[:11]:>12}"
    print("\n" + hdr + "   role")
    for s, recs in rows.items():
        n = len(recs)
        c = 100*sum(r["containment"] for r in recs)/n
        q = 100*sum(r["judge"] for r in recs)/n
        line = f"{s:<16}{n:>5}{c:>8.1f}%{q:>7.1f}%"
        if prior:
            pe = {x["q_id"]: x["api"] for x in prior.get(s, {}).get("per_example", [])}
            hits = [pe[r["q_id"]] for r in recs if r["q_id"] in pe]
            line += f"{100*sum(hits)/len(hits):>8.1f}%" if hits else f"{'-':>9}"
        for m in models:
            line += f"{100*sum(scores[m][s])/n:>11.1f}%"
        print(line + f"   {'CONTROL' if s in controls else ''}")

    print(f"\n{'model':<14}{'in tok':>10}{'out tok':>10}{'reason tok':>12}{'out/row':>9}")
    ntot = sum(len(r) for r in rows.values())
    for m in models:
        i, o, rt = usage[m]
        print(f"{m:<14}{i:>10}{o:>10}{rt:>12}{o/ntot:>9.1f}")
    print(f"\nrows judged per model: {ntot}  (price from your dashboard: no rates hardcoded)")

    json.dump({"models": models, "usage": usage,
               "scores": {m: {s: v for s, v in d.items()} for m, d in scores.items()},
               "rows": {s: [r["q_id"] for r in recs] for s, recs in rows.items()}},
              open(args.output_path, "w"), indent=2)
    print(f"wrote {args.output_path}")


if __name__ == "__main__":
    main()
