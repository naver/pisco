#!/usr/bin/env python
"""Build a weighted finetuning mix in PISCO's FineTuningCollator schema
({docs:[str], query:str, mistral_label:str}).

Sources:
  - any ARC jsonl basename (read from ARC-Encoder_ft.zip): adversarialqa, drop, wikisum, ...
  - "kilt"  -> our existing maxoul/pisco_finetuning_data (already in-schema)

Spec format:  "src:weight,src:weight,..."  (weights are relative; sampled proportionally).
For a target --total N, source i contributes round(w_i/sum_w * N) rows (sampled WITH
replacement if the source is smaller than its quota). Output shuffled, saved as
DatasetDict({"train": ...}) for train.py's load_from_disk(path)["train"].
"""
import argparse, json, zipfile, io, random
from datasets import Dataset, DatasetDict, load_dataset

ZIP = "/beegfs/scratch/user/hdejean/arc_ft_data/ARC-Encoder_ft.zip"

# Per-task system prompt: QA sources keep PISCO's extract-and-answer-briefly prompt; everything
# else (summarization, paraphrase, translation) gets a neutral prompt so the record's own
# instruction (in `query`) drives behavior instead of a conflicting "answer briefly" order.
QA_PROMPT = ("You are a helpful assistant. Your task is to extract relevant information from "
             "provided documents and to answer to questions as briefly as possible.")
NEUTRAL_PROMPT = "You are a helpful assistant. Follow the instruction using the provided text."
QA_SOURCES = {"kilt", "freebaseqa", "drop", "msmarco", "adversarialqa", "sciq", "asqa"}

def system_prompt_for(src):
    base = src.split(":", 1)[1] if src.startswith("regen:") else src
    return QA_PROMPT if base in QA_SOURCES else NEUTRAL_PROMPT

def to_pisco(rec):
    if "passages" in rec:      docs = rec["passages"]
    elif "passage" in rec:     docs = [rec["passage"]]
    else:                      docs = rec.get("docs", [])
    q = rec.get("question", rec.get("query", ""))
    ans = rec.get("answer", rec.get("mistral_label", ""))
    if isinstance(ans, list):  ans = ans[0] if ans else ""
    return {"docs": list(docs), "query": q, "mistral_label": ans}

REGEN_DIR = "/beegfs/scratch/user/hdejean/arc_ft_data/regen"

def load_source(src, need, rng):
    if src == "kilt":
        # stream our KILT QA; take a bit more than needed, then sample
        ds = load_dataset("maxoul/pisco_finetuning_data", split="train", streaming=True)
        pool = []
        for ex in ds:
            pool.append({"docs": list(ex["docs"]), "query": ex["query"],
                         "mistral_label": ex["mistral_label"]})
            if len(pool) >= max(need, 1) * 2 and len(pool) >= 5000:
                break
    elif src.startswith("regen:"):
        # pre-generated sentence labels (scripts/gen_labels.py), already in PISCO schema
        pool = []
        with open(f"{REGEN_DIR}/{src.split(':',1)[1]}.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    pool.append({"docs": r["docs"], "query": r["query"],
                                 "mistral_label": r["mistral_label"]})
    else:
        pool = []
        with zipfile.ZipFile(ZIP) as z:
            with z.open(f"ARC-Encoder_ft/{src}.jsonl") as f:
                for line in io.TextIOWrapper(f, encoding="utf-8"):
                    line = line.strip()
                    if line:
                        pool.append(to_pisco(json.loads(line)))
    if not pool:
        return []
    if len(pool) >= need:
        return rng.sample(pool, need)
    return [pool[rng.randrange(len(pool))] for _ in range(need)]  # oversample w/ replacement

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--total", type=int, default=100000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # split on the LAST colon so "regen:freebaseqa:6" -> ("regen:freebaseqa", 6)
    pairs = [(s.rsplit(":", 1)[0], float(s.rsplit(":", 1)[1])) for s in args.spec.split(",")]
    wsum = sum(w for _, w in pairs)
    rows = []
    for src, w in pairs:
        need = round(w / wsum * args.total)
        got = load_source(src, need, rng)
        sp = system_prompt_for(src)
        for r in got:
            r["source"] = src
            r["system_prompt"] = sp
        rows.extend(got)
        print(f"  {src:30} w={w:<5} -> {len(got)} rows")
    rng.shuffle(rows)
    d = Dataset.from_list(rows)
    DatasetDict({"train": d}).save_to_disk(args.out)
    print(f"saved {len(d)} rows -> {args.out}")

if __name__ == "__main__":
    main()
