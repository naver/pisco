"""Arm F of campaign probes: the TextVQA rows of eval_data/mm_naturalqa_1500 with their OCR tokens as `docs`.

The photo set stores docs=[""] (build_mm_natural_qa.py line 82), so `--mode pisco_text` on it compresses nothing.
This copies its TextVQA rows unchanged (image, question, answers, query_id) and puts the OCR tokens of the same
lmms-lab/textvqa validation row in docs[0]. query_id "textvqa:<i>" is the row's index in the validation stream;
every question is checked against the stream, and the build aborts on any mismatch.

    python scripts/probes/build_textvqa_ocr.py [--src eval_data/mm_naturalqa_1500] [--out eval_data/probes_textvqa_ocr]
"""
import argparse
import sys

import datasets
import numpy as np


def norm(s):
    return " ".join(str(s).lower().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="eval_data/mm_naturalqa_1500")
    ap.add_argument("--out", default="eval_data/probes_textvqa_ocr")
    args = ap.parse_args()

    src = datasets.load_from_disk(args.src)["train"]
    keep = [j for j, s in enumerate(src["source"]) if s == "textvqa"]
    rows = src.select(keep)
    want = {int(q.split(":")[1]): j for j, q in enumerate(rows["query_id"])}
    print(f"[ocr] {len(rows)} textvqa rows in {args.src}; stream indices {min(want)}..{max(want)}", flush=True)

    stream = datasets.load_dataset("lmms-lab/textvqa", split="validation", streaming=True)
    stream = stream.select_columns(["question", "ocr_tokens"])
    ocr, bad = {}, []
    for i, r in enumerate(stream):
        if i in want:
            j = want[i]
            if norm(r["question"]) != norm(rows[j]["query"]):
                bad.append((i, r["question"], rows[j]["query"]))
            ocr[j] = " ".join(t for t in (r["ocr_tokens"] or []) if t)
        if i >= max(want):
            break
    if bad or len(ocr) != len(rows):
        print(f"[ocr] ABORT: {len(bad)} question mismatches, {len(ocr)}/{len(rows)} rows found; first: {bad[:3]}")
        sys.exit(1)

    out = rows.map(lambda r, j: {"docs": [ocr[j]]}, with_indices=True)
    n_tok = np.array([len(ocr[j].split()) for j in range(len(rows))])
    hit = [any(norm(a) and norm(a) in norm(ocr[j]) for a in rows[j]["answers"]) for j in range(len(rows))]
    print(f"[ocr] questions matched {len(rows)}/{len(rows)}; OCR tokens per row mean {n_tok.mean():.1f}, "
          f"median {np.median(n_tok):.0f}, max {n_tok.max()}, empty {(n_tok == 0).sum()}; "
          f"a gold answer appears verbatim in the OCR text on {np.mean(hit):.3f} of rows", flush=True)
    datasets.DatasetDict({"train": out}).save_to_disk(args.out)
    print(f"[ocr] wrote {args.out}: {len(out)} rows, columns {out.column_names}")


if __name__ == "__main__":
    main()
