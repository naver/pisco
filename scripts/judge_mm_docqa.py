"""LLM-judge scoring for multimodal doc-QA eval files (scripts/eval_mm_docqa.py output).

Exact match / F1 under-count verbose-but-correct answers ("Rituximab was approved first." vs gold
"Rituximab"). This re-scores each record with the project's Yes/No judge prompt (verbatim from
scripts/judge_teacher.py / llm_judge.py / bergen) using a local instruct model, writes a `judge`
field (1/0) per record and `judge` / `judge/<source>` means into the summary, IN PLACE.

  python scripts/judge_mm_docqa.py expQ/base_qwen35_4b/eval/mm_docqa_base.json [more.json ...] \
      [--judge_model Qwen/Qwen3-8B] [--batch_size 16] [--samples]

--samples also judges each record's `samples` list (eval_mm_docqa.py --num_samples) into `samples_judge`,
and adds `pass@G_judge` (any sample judged correct) and `sample_judge` (mean over samples) to the summary.
Identical (question, answer) pairs are judged once.
"""

import argparse
import collections
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from judge_teacher import judge_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--judge_model", default="Qwen/Qwen3-8B")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--samples", action="store_true", help="also judge each record's `samples` list")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[judge] loading {args.judge_model}")
    tok = AutoTokenizer.from_pretrained(args.judge_model, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32
    ).to(device).eval()

    def judge_all(prompts, tag):
        verdicts, weird = [], 0
        with torch.inference_mode():
            for i in range(0, len(prompts), args.batch_size):
                enc = tok(prompts[i : i + args.batch_size], return_tensors="pt", padding=True).to(device)
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
                gen = out[:, enc["input_ids"].shape[1]:]
                for t in tok.batch_decode(gen, skip_special_tokens=True):
                    t = t.strip().lower()
                    if "yes" in t and "no" not in t:
                        verdicts.append(1)
                    elif "no" in t:
                        verdicts.append(0)
                    else:
                        verdicts.append(0); weird += 1
                if (i // args.batch_size) % 10 == 0:
                    print(f"[judge] {tag} {min(i + args.batch_size, len(prompts))}/{len(prompts)}", flush=True)
        return verdicts, weird

    def chat(r, pred):
        return tok.apply_chat_template(judge_prompt(r["query"], r["answers"], pred),
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)

    for path in args.files:
        data = json.load(open(path))
        recs = data["records"]
        verdicts, weird = judge_all([chat(r, r["prediction"]) for r in recs], os.path.basename(path))
        agg = collections.defaultdict(list)
        for r, v in zip(recs, verdicts):
            r["judge"] = v
            agg["judge"].append(v); agg[f"judge/{r.get('source', '?')}"].append(v)
        if args.samples:
            keys, raw = {}, []   # (record idx, normalised answer) -> position; raw = first original text per key
            for i, r in enumerate(recs):
                for s in r.get("samples") or []:
                    if (i, s.strip().lower()) not in keys:
                        keys[(i, s.strip().lower())] = len(keys)
                        raw.append((i, s))
            sv, sw = judge_all([chat(recs[i], s) for i, s in raw], f"{os.path.basename(path)} samples")
            uniq = raw
            weird += sw
            print(f"[judge] samples: {sum(len(r.get('samples') or []) for r in recs)} judged as {len(uniq)} unique")
            for i, r in enumerate(recs):
                if not r.get("samples"):
                    continue
                r["samples_judge"] = [sv[keys[(i, s.strip().lower())]] for s in r["samples"]]
                src = r.get("source", "?")
                hit, mean = float(any(r["samples_judge"])), float(np.mean(r["samples_judge"]))
                agg["pass@G_judge"].append(hit); agg[f"pass@G_judge/{src}"].append(hit)
                agg["sample_judge"].append(mean); agg[f"sample_judge/{src}"].append(mean)
        data["summary"].update({k: float(np.mean(v)) for k, v in agg.items()})
        data["summary"]["judge_model"] = args.judge_model
        data["summary"]["judge_unparseable"] = weird
        with open(path, "w") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
        s = data["summary"]
        print(f"=== {path}")
        for k in sorted(k for k in s if k.startswith(("judge", "pass@", "sample_judge")) or k in ("em", "anls", "match")):
            if isinstance(s[k], float):
                print(f"  {k:28s} {s[k]:.4f}")
        print(f"  unparseable: {weird}")


if __name__ == "__main__":
    main()
