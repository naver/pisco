"""Build the WHOLE-DOCUMENT eval set: the questions of an existing doc-QA eval set, each with every page of its
original document (in reading order) instead of gold + 3 distractors. Query, answers and query_id are copied from the
base set, so results pair with it question by question.

Sources (rows of the base set; others are skipped):
  mpdocvqa: lmms-lab-encoder/MP-DocVQA `val` (ungated, streamed). VisRAG query-id = "<page id>.jpg-<n>"; the original
            row with the same question whose answer page (page_ids[answer_page_idx]) is that page gives the
            question's page window (1-20 pages).

  python scripts/build_mm_eval_longdoc.py --base eval_data/mm_docqa_1500_1m --out eval_data/mm_docqa_longdoc_1m \
      --max_pixels 1003520
"""

import argparse
import ast
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_mm_finetune import ft_features  # noqa: E402
from mm_data_utils import ShardWriter, image_bytes_feature, prepare_page  # noqa: E402


def norm_q(q: str) -> str:
    return " ".join(str(q).split()).lower()


def mpdocvqa_rows(base_rows, max_pixels):
    import datasets
    want = {}                                   # (question, answer page id) -> base row
    for r in base_rows:
        page = r["query_id"].rsplit(".jpg-", 1)[0]
        want[(norm_q(r["query"]), page)] = r
    print(f"[mpdocvqa] {len(want)} base questions to match", flush=True)
    found = set()
    for i, o in enumerate(datasets.load_dataset("lmms-lab-encoder/MP-DocVQA", split="val", streaming=True)):
        pages = ast.literal_eval(o["page_ids"]) if isinstance(o["page_ids"], str) else list(o["page_ids"])
        gold = int(o["answer_page_idx"])
        key = (norm_q(o["question"]), pages[gold])
        if key in want and key not in found:
            imgs = [o[f"image_{k + 1}"] for k in range(len(pages))]
            assert all(im is not None for im in imgs), f"{o['questionId']}: missing page image"
            r = want[key]
            found.add(key)
            yield {
                "docs": [""] * len(pages),
                "doc_images": [image_bytes_feature(prepare_page(im, max_pixels=max_pixels)) for im in imgs],
                "query": r["query"], "mistral_label": r["mistral_label"], "answers": list(r["answers"]),
                "gold_positions": [gold], "source": "mpdocvqa", "query_id": r["query_id"],
            }
        if (i + 1) % 500 == 0:
            print(f"[mpdocvqa] {i + 1} val rows seen, {len(found)} matched", flush=True)
        if len(found) == len(want):
            break
    print(f"[mpdocvqa] matched {len(found)}/{len(want)}; unmatched e.g. "
          f"{[k for k in want if k not in found][:3]}", flush=True)


BUILDERS = {"mpdocvqa": mpdocvqa_rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="eval set whose questions are re-used (load_from_disk)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", default="mpdocvqa")
    ap.add_argument("--max_pixels", type=int, default=1003520)
    args = ap.parse_args()

    import datasets
    base = datasets.load_from_disk(args.base)
    base = base["train"] if "train" in getattr(base, "keys", lambda: [])() else base
    base = base.select_columns(["query", "mistral_label", "answers", "source", "query_id"])
    writer = ShardWriter(args.out, ft_features(), shard_size=200)
    n_pages = Counter()
    for src in args.sources.split(","):
        rows = [r for r in base if r["source"] == src]
        for row in BUILDERS[src](rows, args.max_pixels):
            writer.add(row)
            n_pages[len(row["doc_images"])] += 1
    writer.finalize()
    print("pages per question:", dict(sorted(n_pages.items())))
    print("buckets:", {b: sum(v for k, v in n_pages.items() if lo <= k <= hi)
                       for b, (lo, hi) in {"1-4": (1, 4), "5-10": (5, 10), "11-20": (11, 20)}.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
