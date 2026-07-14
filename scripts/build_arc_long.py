#!/usr/bin/env python
"""Adapt ARC-Encoder long-document finetuning data -> PISCO FineTuningCollator schema.

ARC jsonl rows:  {passage:str, question:str, answer:str}
PISCO rows:      {docs:[str], query:str, mistral_label:str}   (keys read in collator.py:545-547)

Long-document family in kyutai/ARC_finetuning: wikisum (genuine long docs, ~1.4k tok),
dialogsum (dialogue summary). samsum/parasci are short, excluded from the "long" set.
Saved as a DatasetDict({"train": ...}) so train.py's load_from_disk(path)["train"] path works.
"""
import argparse, json, zipfile, io
from datasets import Dataset, DatasetDict

ZIP = "/beegfs/scratch/user/hdejean/arc_ft_data/ARC-Encoder_ft.zip"

def read_jsonl(name):
    with zipfile.ZipFile(ZIP) as z:
        with z.open(f"ARC-Encoder_ft/{name}.jsonl") as f:
            for line in io.TextIOWrapper(f, encoding="utf-8"):
                line = line.strip()
                if line:
                    yield json.loads(line)

def to_pisco(rec, source):
    # passage(str) -> single doc; question -> query; answer(str|list) -> mistral_label
    ans = rec["answer"]
    if isinstance(ans, list):
        ans = ans[0] if ans else ""
    return {"docs": [rec["passage"]], "query": rec["question"],
            "mistral_label": ans, "source": source}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["wikisum", "dialogsum"],
                    help="ARC long-doc files to include")
    ap.add_argument("--out", default="/beegfs/scratch/user/hdejean/arc_ft_data/arc_long_hf")
    args = ap.parse_args()

    rows = []
    for ds in args.datasets:
        n0 = len(rows)
        for rec in read_jsonl(ds):
            try:
                rows.append(to_pisco(rec, ds))
            except KeyError:
                pass
        print(f"  {ds}: +{len(rows)-n0} rows")
    d = Dataset.from_list(rows)
    DatasetDict({"train": d}).save_to_disk(args.out)
    print(f"saved {len(d)} rows -> {args.out}")
    print("sample:", {k: (v[:80] if isinstance(v, str) else v) for k, v in d[0].items()})

if __name__ == "__main__":
    main()
