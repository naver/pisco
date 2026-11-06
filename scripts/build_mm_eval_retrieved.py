"""Build RETRIEVED-DISTRACTOR eval sets: the questions of an existing doc-QA eval set (VisRAG sources), each with its
gold page(s) plus the other pages of the same corpus that a visual retriever (ColQwen2) ranks highest, shuffled.
Query, answers, query_id and the gold rule (first 2 qrels pages, as scripts/build_mm_eval.py) are the base set's, so
results pair with it question by question: only the distractors change (random / same-document -> retrieved).

Phase 1 (GPU): rank each source's whole corpus for its questions with late interaction (MaxSim); the top `--keep`
ids, their scores and the gold ranks go to `--ranking` (JSON; reused if present). Phase 2 (CPU): one eval set per
`--pages` value, written to `--out` with "{pages}" replaced.

  python scripts/build_mm_eval_retrieved.py --base eval_data/mm_docqa_1500_1m \
      --ranking expQ/archq/colqwen2_ranking.json --out eval_data/mm_docqa_retr{pages}_1m --pages 4,10
"""

import argparse
import json
import os
import pathlib
import random
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_mm_eval import VISRAG  # noqa: E402
from build_mm_finetune import ft_features  # noqa: E402
from mm_data_utils import ShardWriter, image_bytes_feature, prepare_page  # noqa: E402

RETRIEVER = "vidore/colqwen2-v1.0-hf"


def qrels(repo):
    import datasets
    rel = defaultdict(list)
    for r in datasets.load_dataset(repo, "qrels", split="train"):
        if int(r["score"]) > 0:
            rel[str(r["query-id"])].append(str(r["corpus-id"]))
    return rel


def load_retriever(device):
    import torch
    from transformers import ColQwen2ForRetrieval, ColQwen2Processor
    model = ColQwen2ForRetrieval.from_pretrained(RETRIEVER, dtype=torch.bfloat16).to(device).eval()
    try:
        proc = ColQwen2Processor.from_pretrained(RETRIEVER)
    except Exception as e:  # the fast image processor needs torchvision (see pisco.mm_utils.load_image_processor)
        from transformers import AutoTokenizer
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
        print(f"[retr] ColQwen2Processor.from_pretrained failed ({type(e).__name__}); PIL image processor instead")
        proc = ColQwen2Processor(image_processor=Qwen2VLImageProcessorPil.from_pretrained(RETRIEVER),
                                 tokenizer=AutoTokenizer.from_pretrained(RETRIEVER))
    return model, proc


def rank_source(name, base_rows, model, proc, *, keep, batch, device, smoke):
    import datasets
    import torch
    repo = VISRAG[name]
    rel = qrels(repo)
    corpus = datasets.load_dataset(repo, "corpus", split="train")
    if smoke:
        corpus = corpus.select(range(min(smoke, len(corpus))))
        base_rows = base_rows[:smoke]
    ids = [str(x) for x in corpus["corpus-id"]]
    page_embs = []
    with torch.inference_mode():
        for i in range(0, len(corpus), batch):
            imgs = [im.convert("RGB") for im in corpus[i:i + batch]["image"]]
            b = proc.process_images(images=imgs, return_tensors="pt").to(device)
            e = model(**b).embeddings
            m = b["attention_mask"].bool()
            page_embs.extend(e[k][m[k]] for k in range(len(imgs)))   # drop padding vectors; stays on the GPU
            if (i // batch) % 20 == 0:
                print(f"[retr] {name}: {i + len(imgs)}/{len(corpus)} pages embedded", flush=True)
        q_embs = []
        for i in range(0, len(base_rows), batch):
            b = proc.process_queries(text=[r["query"] for r in base_rows[i:i + batch]], return_tensors="pt").to(device)
            e = model(**b).embeddings
            m = b["attention_mask"].bool()
            q_embs.extend(e[k][m[k]] for k in range(e.shape[0]))
        scores = proc.score_retrieval(q_embs, page_embs, batch_size=64, output_dtype=torch.float32)
    out = {}
    for r, s in zip(base_rows, scores):
        order = torch.argsort(s, descending=True).tolist()
        gold = rel[r["query_id"]][:2]
        rank_of = {ids[j]: k for k, j in enumerate(order)}
        out[r["query_id"]] = {"top": [[ids[j], round(float(s[j]), 3)] for j in order[:keep]],
                              "gold": gold, "gold_rank": [rank_of.get(g) for g in gold]}
    r1 = sum(1 for v in out.values() if v["gold_rank"] and v["gold_rank"][0] == 0) / max(1, len(out))
    r10 = sum(1 for v in out.values() if any(x is not None and x < 10 for x in v["gold_rank"])) / max(1, len(out))
    print(f"[retr] {name}: {len(out)} questions, {len(ids)} pages; gold at rank 1 {r1:.1%}, in top 10 {r10:.1%}",
          flush=True)
    return out


def build_sets(base_rows_by_src, ranking, pages_list, out_pattern, max_pixels, seed):
    import datasets
    writers = {p: ShardWriter(out_pattern.format(pages=p), ft_features(), shard_size=500) for p in pages_list}
    for name, rows in base_rows_by_src.items():
        need = set()
        for r in rows:
            v = ranking[name][r["query_id"]]
            need.update(v["gold"])
            need.update(d for d, _ in v["top"][: max(pages_list) + 2])
        jpg = {}
        for row in datasets.load_dataset(VISRAG[name], "corpus", split="train"):
            cid = str(row["corpus-id"])
            if cid in need:
                jpg[cid] = prepare_page(row["image"], max_pixels=max_pixels)
        print(f"[build] {name}: {len(jpg)} pages resized", flush=True)
        for p in pages_list:
            rng = random.Random(seed)
            for r in rows:
                v = ranking[name][r["query_id"]]
                gold = [g for g in v["gold"] if g in jpg]
                neg = [d for d, _ in v["top"] if d not in set(v["gold"]) and d in jpg][: p - len(gold)]
                docs = gold + neg
                rng.shuffle(docs)
                writers[p].add({
                    "docs": [""] * len(docs),
                    "doc_images": [image_bytes_feature(jpg[d]) for d in docs],
                    "query": r["query"], "mistral_label": r["mistral_label"], "answers": list(r["answers"]),
                    "gold_positions": [i for i, d in enumerate(docs) if d in set(gold)],
                    "source": name, "query_id": r["query_id"],
                })
    for w in writers.values():
        w.finalize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ranking", required=True, help="ranking JSON (written by phase 1, reused if present)")
    ap.add_argument("--out", default=None, help='output path pattern with "{pages}"; omit to rank only')
    ap.add_argument("--pages", default="4,10")
    ap.add_argument("--keep", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_pixels", type=int, default=1003520)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0, help="rank only the first N pages / questions per source")
    args = ap.parse_args()

    import datasets
    base = datasets.load_from_disk(args.base)
    base = base["train"] if "train" in getattr(base, "keys", lambda: [])() else base
    base = base.select_columns(["query", "mistral_label", "answers", "source", "query_id"])
    by_src = defaultdict(list)
    for r in base:
        by_src[r["source"]].append(r)
    if os.path.exists(args.ranking):
        ranking = json.load(open(args.ranking))
        print(f"[retr] ranking reused from {args.ranking}")
    else:
        model, proc = load_retriever("cuda")
        ranking = {n: rank_source(n, rows, model, proc, keep=args.keep, batch=args.batch, device="cuda",
                                  smoke=args.smoke) for n, rows in by_src.items()}
        if args.smoke:
            print("[retr] smoke: ranking not saved")
            return 0
        json.dump(ranking, open(args.ranking, "w"))
        print(f"[retr] wrote {args.ranking}")
    if args.out:
        build_sets(by_src, ranking, [int(x) for x in args.pages.split(",")], args.out, args.max_pixels, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
