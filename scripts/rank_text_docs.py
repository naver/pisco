"""External text retrievers over each question's own candidate documents, for the retrieve-then-read baselines of
scripts/eval_mm_docqa.py (--rank_scores, --raw_policy rankK, --keep_ranked): BM25 (IDF over every document of the
eval set) and a dense retriever (BGE, query instruction, CLS pooling, cosine). Writes <out_prefix>_<name>.json =
{query_id: [score per document]} and prints how often each ranks a gold document first / both gold in its top 2.

  python scripts/rank_text_docs.py --data eval_data/text_hotpotqa_1500 --out_prefix expQ/archq/rank_hotpot
"""

import argparse
import json
import math
import re
import sys
from collections import Counter

DENSE = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def words(t):
    return re.findall(r"\w+", t.lower())


def bm25(rows, k1=1.2, b=0.75):
    docs = [words(d) for r in rows for d in r["docs"]]
    df = Counter(w for d in docs for w in set(d))
    n, avg = len(docs), sum(map(len, docs)) / len(docs)
    idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
    out = {}
    for r in rows:
        q = words(r["query"])
        sc = []
        for d in r["docs"]:
            tf, dl = Counter(words(d)), len(words(d))
            sc.append(sum(idf.get(w, 0) * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * dl / avg)) for w in q if w in tf))
        out[r["query_id"]] = sc
    return out


def dense(rows, batch=64):
    import torch
    from transformers import AutoModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(DENSE)
    model = AutoModel.from_pretrained(DENSE).to(dev).eval()

    def enc(texts):
        vs = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch):
                x = tok(texts[i:i + batch], padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
                vs.append(torch.nn.functional.normalize(model(**x).last_hidden_state[:, 0].float(), dim=-1).cpu())
        return torch.cat(vs)

    q = enc([QUERY_PREFIX + r["query"] for r in rows])
    d = enc([t for r in rows for t in r["docs"]])
    out, k = {}, 0
    for i, r in enumerate(rows):
        n = len(r["docs"])
        out[r["query_id"]] = (d[k:k + n] @ q[i]).tolist()
        k += n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()
    import datasets
    ds = datasets.load_from_disk(args.data)
    ds = ds["train"] if "train" in getattr(ds, "keys", lambda: [])() else ds
    rows = list(ds.select_columns(["query_id", "query", "docs", "gold_positions"]))
    for name, fn in [("bm25", bm25), ("bge", dense)]:
        sc = fn(rows)
        json.dump(sc, open(f"{args.out_prefix}_{name}.json", "w"))
        top = {r["query_id"]: sorted(range(len(sc[r["query_id"]])), key=lambda j: -sc[r["query_id"]][j]) for r in rows}
        p1 = sum(top[r["query_id"]][0] in set(r["gold_positions"]) for r in rows) / len(rows)
        both2 = sum(set(r["gold_positions"]) <= set(top[r["query_id"]][:2]) for r in rows) / len(rows)
        print(f"{name}: gold first {p1:.3f}, all gold in top 2 {both2:.3f} -> {args.out_prefix}_{name}.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
