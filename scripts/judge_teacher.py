#!/usr/bin/env python
"""Judge the cached teacher answers in the PISCO finetuning data with a local LLM.

Why: `containment_em` decides which rows are teacher-correct (S+) vs teacher-failed (S-),
but it can only fire when the gold string appears verbatim. On sources whose gold is a
full extracted sentence (wikiqa, much of msmarco) a correct paraphrase scores 0, so the
"46% of targets are wrong" figure is untrustworthy exactly where S- is largest. Token-F1
was tried (job 163353) and is worse -- it penalises length mismatch in both directions.

This asks a local instruct model the same Yes/No question the project's OpenAI judge asks
(prompt copied verbatim from scripts/llm_judge.py:judge_prompt, so numbers stay comparable
with the bergen LLM-judge column), and reports judge-vs-containment per source.

CALIBRATION IS PART OF THE RUN: pass control sources whose gold is a short span (nq_open,
triviaqa), where containment is known-good. If the judge roughly reproduces containment
there, its msmarco/wikiqa numbers can be trusted; if it says Yes to nearly everything, it
is too lenient and the msmarco number means nothing.

Usage:
  python scripts/judge_teacher.py --sources msmarco,wikiqa --controls nq_open,triviaqa \
      --n_per_source 300 --judge_model Qwen/Qwen3-8B --output_path outputs/judge_teacher.json
"""

import argparse
import collections
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

from pisco.reward import containment_em, source_of


from judge_prompts import project_prompt as judge_prompt  # noqa: E402  (one copy of the prompt, scripts/judge_prompts.py)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="maxoul/pisco_finetuning_data")
    p.add_argument("--split", default="train")
    p.add_argument("--sources", default="msmarco,wikiqa",
                   help="Sources under investigation (containment suspected unreliable).")
    p.add_argument("--controls", default="nq_open,triviaqa",
                   help="Short-gold sources used to calibrate the judge against containment.")
    p.add_argument("--n_per_source", type=int, default=300)
    p.add_argument("--judge_model", default="Qwen/Qwen3-8B")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_path", default="outputs/judge_teacher.json")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wanted = [s for s in (args.sources.split(",") + args.controls.split(",")) if s]
    controls = set(args.controls.split(",")) if args.controls else set()

    print(f"[judge] loading {args.dataset}")
    ds = datasets.load_dataset(args.dataset, split=args.split)

    per_source = collections.defaultdict(list)
    for ex in ds:
        src = source_of(ex.get("q_id"))
        if src in wanted and len(per_source[src]) < args.n_per_source:
            per_source[src].append(ex)
        if all(len(per_source[s]) >= args.n_per_source for s in wanted):
            break
    for s in wanted:
        print(f"[judge]   {s}: {len(per_source[s])} rows")

    print(f"[judge] loading judge {args.judge_model}")
    tok = AutoTokenizer.from_pretrained(args.judge_model, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device).eval()

    rows, results = [], {}
    for src in wanted:
        for ex in per_source[src]:
            rows.append((src, ex))

    prompts = [
        tok.apply_chat_template(
            judge_prompt(ex.get("query", ""), ex.get("labels", []), ex.get("mistral_label", "")),
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for _, ex in rows
    ]

    verdicts, weird = [], 0
    with torch.inference_mode():
        for i in range(0, len(prompts), args.batch_size):
            chunk = prompts[i : i + args.batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            for t in tok.batch_decode(gen, skip_special_tokens=True):
                low = t.lower()
                if "yes" in low:
                    verdicts.append(1)
                elif "no" in low:
                    verdicts.append(0)
                else:
                    verdicts.append(0); weird += 1
            if (i // args.batch_size) % 10 == 0:
                print(f"[judge] {min(i+args.batch_size, len(prompts))}/{len(prompts)}", flush=True)

    print(f"\n[judge] unparseable verdicts: {weird}")
    print(f"\n{'source':<14}{'n':>5}{'contain%':>10}{'judge%':>9}{'delta':>8}"
          f"{'J+/C-':>8}{'J-/C+':>8}   role")
    for src in wanted:
        idx = [k for k, (s, _) in enumerate(rows) if s == src]
        if not idx:
            continue
        c = [1 if containment_em(rows[k][1].get("mistral_label"), rows[k][1].get("labels")) else 0 for k in idx]
        j = [verdicts[k] for k in idx]
        n = len(idx)
        jc = sum(1 for a, b in zip(j, c) if a == 1 and b == 0)   # judge rescues
        cj = sum(1 for a, b in zip(j, c) if a == 0 and b == 1)   # judge overrules
        role = "CONTROL" if src in controls else "under test"
        print(f"{src:<14}{n:>5}{100*sum(c)/n:>9.1f}%{100*sum(j)/n:>8.1f}%"
              f"{100*(sum(j)-sum(c))/n:>+7.1f}{jc:>8}{cj:>8}   {role}")
        results[src] = dict(n=n, containment=sum(c), judge=sum(j),
                            judge_yes_contain_no=jc, judge_no_contain_yes=cj,
                            control=src in controls)
        # ALL per-example verdicts, keyed by q_id, so a second judge (e.g. an API
        # model) can be joined against this one row-by-row. An aggregate delta
        # cannot say which scorer is the wrong one.
        results[src]["per_example"] = [
            dict(q_id=rows[k][1].get("q_id"), query=rows[k][1].get("query"),
                 gold=rows[k][1].get("labels"),
                 teacher=(rows[k][1].get("mistral_label") or "")[:400],
                 judge=verdicts[k], containment=c_k)
            for k, c_k in zip(idx, c)
        ]

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump({"judge_model": args.judge_model, "n_per_source": args.n_per_source,
                   "unparseable": weird, "by_source": results}, f, indent=2)
    print(f"\n[judge] wrote {args.output_path}")
    print("\n[judge] READ THE CONTROLS FIRST: if judge% tracks contain% on the control rows,")
    print("[judge] the under-test numbers are trustworthy; if judge% is near 100 everywhere,")
    print("[judge] the judge is too lenient and nothing here can be believed.")


if __name__ == "__main__":
    main()
