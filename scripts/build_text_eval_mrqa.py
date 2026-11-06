"""Build the MRQA out-of-domain TEXT eval set (the benchmark of "Position IDs Matter", arXiv 2409.14364, Table 4):
one passage per question, reading comprehension. Source: mrqa-workshop/mrqa `test` split (ungated), whose six
subsets are the out-of-domain ones: BioASQ, DROP, DuoRC.ParaphraseRC, RACE, RelationExtraction, TextbookQA.
Passages range from a paragraph (DROP, RE) to pages (TextbookQA, RACE, DuoRC), so the set also serves the
long-chunk question. Same row schema as every eval set (`doc_images` empty; `gold_positions` = [0]), so
scripts/eval_mm_docqa.py runs it unchanged (TOPK >= 1). `source` = mrqa_<subset>.

  python scripts/build_text_eval_mrqa.py --out eval_data/text_mrqa_ood_1500 --per_subset 250
"""

import argparse
import pathlib
import random
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_mm_finetune import ft_features  # noqa: E402
from mm_data_utils import ShardWriter  # noqa: E402

OOD = ["BioASQ", "DROP", "DuoRC.ParaphraseRC", "RACE", "RelationExtraction", "TextbookQA"]
MARKUP = re.compile(r"\[(PAR|DOC|TLE|SEP)\]")


def clean_context(c: str) -> str:
    c = MARKUP.sub("\n", c)
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n", c)).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_subset", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import datasets
    ds = datasets.load_dataset("mrqa-workshop/mrqa", split="test")
    rng = random.Random(args.seed)
    by = {s: [] for s in OOD}
    for i, s in enumerate(ds["subset"]):
        if s in by:
            by[s].append(i)
    writer = ShardWriter(args.out, ft_features(), shard_size=500)
    for s in OOD:
        idx = sorted(rng.sample(by[s], min(args.per_subset, len(by[s]))))
        lens = []
        for i in idx:
            o = ds[i]
            ctx = clean_context(o["context"])
            answers = [a for a in o["answers"] if a and a.strip()] or [""]
            lens.append(len(ctx.split()))
            writer.add({
                "docs": [ctx], "doc_images": [],
                "query": o["question"].strip(), "mistral_label": answers[0], "answers": answers,
                "gold_positions": [0], "source": f"mrqa_{s.split('.')[0]}", "query_id": o["qid"],
            })
        lens.sort()
        print(f"{s}: {len(idx)} of {len(by[s])} questions; passage words p50 {lens[len(lens) // 2]} "
              f"p90 {lens[int(0.9 * len(lens))]} max {lens[-1]}", flush=True)
    writer.finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
