"""Build the multimodal document-QA EVAL set (same row schema as the finetune set).

Sources: the VisRAG retrieval test sets (openbmb/VisRAG-Ret-Test-*; ungated; each has `corpus`
(corpus-id, image), `queries` (query-id, query, answer, options, is_numerical) and `qrels`):
  slidevqa 556 Q / 1284 slides · infovqa 718 Q / 459 pages · mpdocvqa 591 Q / 741 pages ·
  arxivqa 816 / 8066 · chartqa 63 / 500 · plotqa 863 / 9593
Each row = gold page(s) + `--distractors` other pages of the same corpus (same deck/document prefix
when ids allow), shuffled.

  python scripts/build_mm_eval.py --out eval_data/mm_docqa_500 --per_source slidevqa=200,infovqa=200,mpdocvqa=100
"""

import argparse
import ast
import pathlib
import random
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_mm_finetune import ft_features  # noqa: E402
from mm_data_utils import ShardWriter, image_bytes_feature, prepare_page  # noqa: E402

VISRAG = {
    "slidevqa": "openbmb/VisRAG-Ret-Test-SlideVQA",
    "infovqa": "openbmb/VisRAG-Ret-Test-InfoVQA",
    "mpdocvqa": "openbmb/VisRAG-Ret-Test-MP-DocVQA",
    "arxivqa": "openbmb/VisRAG-Ret-Test-ArxivQA",
    "chartqa": "openbmb/VisRAG-Ret-Test-ChartQA",
    "plotqa": "openbmb/VisRAG-Ret-Test-PlotQA",
}


def parse_answers(a) -> list:
    """VisRAG stores `answer` as a string; InfoVQA/MP-DocVQA rows are a stringified list "['2003']"."""
    if isinstance(a, (list, tuple)):
        return [str(x) for x in a]
    a = str(a).strip()
    if a.startswith("[") and a.endswith("]"):
        try:
            v = ast.literal_eval(a)
            if isinstance(v, (list, tuple)):
                return [str(x) for x in v]
        except (ValueError, SyntaxError):
            pass
    return [a]


def visrag_rows(name, n, distractors, rng, max_pixels, max_gold=2, exclude=()):
    import datasets
    repo = VISRAG[name]
    queries = list(datasets.load_dataset(repo, "queries", split="train"))
    qrels = datasets.load_dataset(repo, "qrels", split="train")
    rel = defaultdict(list)
    for r in qrels:
        if int(r["score"]) > 0:
            rel[str(r["query-id"])].append(str(r["corpus-id"]))
    rng.shuffle(queries)
    chosen = [q for q in queries if rel.get(str(q["query-id"])) and str(q.get("answer") or "").strip()
              and str(q["query-id"]) not in exclude][: n]
    need = set(d for q in chosen for d in rel[str(q["query-id"])][:max_gold])
    # resize every page lazily: gold pages always, others only as distractor candidates (cap the pool)
    corpus = datasets.load_dataset(repo, "corpus", split="train")
    jpg = {}
    pool_cap = max(2000, 20 * len(chosen))
    for i, row in enumerate(corpus):
        cid = str(row["corpus-id"])
        if cid in need or len(jpg) < pool_cap:
            try:
                jpg[cid] = prepare_page(row["image"], max_pixels=max_pixels)
            except Exception as e:
                print(f"  skip {cid}: {e}")
        if (i + 1) % 1000 == 0:
            print(f"[{name}] corpus {i + 1}/{len(corpus)} seen, {len(jpg)} resized", flush=True)
    all_ids = list(jpg)
    for q in chosen:
        gold = [d for d in rel[str(q["query-id"])] if d in jpg][:max_gold]
        if not gold:
            continue
        prefix = gold[0].rsplit("_", 1)[0] if "_" in gold[0] else None
        same = [d for d in all_ids if prefix and d.startswith(prefix + "_") and d not in gold]
        pool = same if len(same) >= distractors else [d for d in all_ids if d not in gold]
        neg = rng.sample(pool, min(distractors, len(pool)))
        docs = gold + neg
        rng.shuffle(docs)
        yield {
            "docs": [""] * len(docs),
            "doc_images": [image_bytes_feature(jpg[d]) for d in docs],
            "query": q["query"],
            "mistral_label": parse_answers(q["answer"])[0],
            "answers": parse_answers(q["answer"]),
            "gold_positions": [i for i, d in enumerate(docs) if d in set(gold)],
            "source": name,
            "query_id": str(q["query-id"]),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_source", default="slidevqa=200,infovqa=200,mpdocvqa=100")
    ap.add_argument("--distractors", type=int, default=3)
    ap.add_argument("--max_pixels", type=int, default=451584)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extend", default=None,
                    help="an existing eval set: its rows are copied verbatim (same ids, pages, distractors) and only "
                         "the missing questions per source are added, so results stay comparable on the old ids")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    plan = dict((k, int(v)) for k, v in (kv.split("=") for kv in args.per_source.split(",") if kv))
    writer = ShardWriter(args.out, ft_features(), shard_size=1000)
    have = defaultdict(set)
    if args.extend:
        import datasets
        old = datasets.load_from_disk(args.extend)
        old = old["train"] if "train" in getattr(old, "keys", lambda: [])() else old
        old = old.cast_column("doc_images", datasets.Sequence(datasets.Image(decode=False)))
        for r in old:
            writer.add(dict(r))
            have[r["source"]].add(str(r["query_id"]))
        print(f"=== kept {len(old)} rows from {args.extend}: " + ", ".join(f"{k} {len(v)}" for k, v in have.items()))
    for src, n in plan.items():
        n_new = n - len(have[src])
        if n_new <= 0:
            continue
        print(f"=== {src}: {n_new} new questions from {VISRAG[src]} (target {n})")
        k = 0
        for row in visrag_rows(src, n_new, args.distractors, rng, args.max_pixels, exclude=have[src]):
            writer.add(row); k += 1
        print(f"=== {src}: {k} rows")
    writer.finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
