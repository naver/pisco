"""Build the multimodal RAG finetuning set: page images as retrieved documents + QA.

Sources (ungated; the OpenDocVQA page corpus is gated on the Hub, so it is not used):
  - HuggingFaceM4/Docmatix  (config `images`): rows {images: [page...], texts: [{user, assistant, source}]}
    multi-page PDF renders with GPT-generated QA (MIT). Streamed shard by shard.
  - HuggingFaceM4/the_cauldron (configs docvqa, infographic_vqa, chartqa, ...): rows
    {images: [img], texts: [{user, assistant, source}]} — the TRAIN splits of DocVQA / InfographicVQA /
    ChartQA, i.e. in-domain for the VisRAG test sets used by build_mm_eval.py.

Each output row is a retrieval-style RAG example: the gold page(s) + distractor pages from OTHER
documents of the same source (shuffled) as `doc_images`; `docs` = "" (image docs); `mistral_label` =
the answer; `answers` kept for eval-style scoring. The user turn of the_cauldron/Docmatix often ends
with an instruction ("Answer briefly."); it is stripped to the bare question.

  python scripts/build_mm_finetune.py --out data/mm_ft_docpages_20k --n 20000 --distractors 3 \
      --mix docmatix=10000,docvqa=7000,infographic_vqa=2000,chartqa=1000
  # v2 (60K+, multi-page gold, six cauldron subsets, text RAG rehearsal rows from maxoul/pisco_finetuning_data):
  python scripts/build_mm_finetune.py --out data/mm_ft_docpages_65k --n 65000 --distractors 3 --max_gold_pages 4 \
      --mix docmatix=30000,docvqa=10000,chartqa=6000,infographic_vqa=2000,tat_qa=2000,visualmrc=3000,ai2d=2000,text_rag=10000
"""

import argparse
import pathlib
import random
import re
import sys
from collections import deque

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mm_data_utils import ShardWriter, image_bytes_feature, prepare_page  # noqa: E402

DOCMATIX = "HuggingFaceM4/Docmatix"
CAULDRON = "HuggingFaceM4/the_cauldron"
TEXT_RAG = "maxoul/pisco_finetuning_data"   # text PISCO finetune set: docs (list[str]), query, mistral_label
_INSTR_WORDS = re.compile(r"\b(answer|brief|briefly|succinct|concise|short|single word|one word|phrase|respond|reply)\b", re.I)


def _is_instruction_line(line: str) -> bool:
    """Trailing lines like 'Make the answer very short.' / 'Be succinct.' / 'Answer briefly.'"""
    line = line.strip()
    return bool(line) and "?" not in line and len(line) <= 60 and bool(_INSTR_WORDS.search(line))


def ft_features():
    import datasets
    return datasets.Features({
        "docs": datasets.Sequence(datasets.Value("string")),
        "doc_images": datasets.Sequence(datasets.Image()),
        "query": datasets.Value("string"),
        "mistral_label": datasets.Value("string"),
        "answers": datasets.Sequence(datasets.Value("string")),
        "gold_positions": datasets.Sequence(datasets.Value("int32")),
        "source": datasets.Value("string"),
        "query_id": datasets.Value("string"),
    })


def clean_question(q: str) -> str:
    q = q.replace("<image>", "").strip()
    lines = [l for l in q.split("\n") if l.strip()]
    while lines and _is_instruction_line(lines[-1]):
        lines.pop()
    q = " ".join(l.strip() for l in lines)
    # the_cauldron also appends the instruction on the same line: "...? Answer briefly."
    if "?" in q:
        head, tail = q.rsplit("?", 1)
        if _is_instruction_line(tail):
            q = head + "?"
    return q.strip()


def usable_qa(q: str, a: str, max_answer_chars: int) -> bool:
    if len(q) < 12 or not re.search(r"[A-Za-z]{3}", q):      # e.g. Docmatix "Q4:" stubs
        return False
    if not a or len(a) > max_answer_chars:
        return False
    return True


class DistractorPool:
    """Ring buffer of recently seen (resized) pages per source, to sample negatives from other docs."""

    def __init__(self, size=400):
        self.buf = deque(maxlen=size)

    def add(self, doc_key, jpg):
        self.buf.append((doc_key, jpg))

    def sample(self, k, exclude_key, rng):
        cands = [j for key, j in self.buf if key != exclude_key]
        return rng.sample(cands, min(k, len(cands)))


def iter_docqa_rows(repo, config, *, source, n, qa_per_doc, max_gold_pages, distractors, max_pixels, rng,
                    pool_warmup=20, max_answer_chars=160):
    """Stream `repo`/`config` and yield finetune rows. Works for Docmatix and the_cauldron (same schema)."""
    import datasets
    ds = datasets.load_dataset(repo, config, split="train", streaming=True)
    pool = DistractorPool()
    out, seen = 0, 0
    for row in ds:
        seen += 1
        imgs = row.get("images") or []
        texts = row.get("texts") or []
        if not imgs or not texts:
            continue
        try:
            pages = [prepare_page(im, max_pixels=max_pixels) for im in imgs[:max_gold_pages]]
        except Exception as e:
            print(f"  skip row {seen}: {type(e).__name__}: {e}")
            continue
        key = f"{source}:{seen}"
        if len(pool.buf) >= pool_warmup:
            qas = list(texts)
            rng.shuffle(qas)
            for t in qas[:qa_per_doc]:
                q, a = clean_question(t.get("user") or ""), (t.get("assistant") or "").strip()
                if not usable_qa(q, a, max_answer_chars):
                    continue
                neg = pool.sample(distractors, key, rng)
                docs = pages + neg
                order = list(range(len(docs)))
                rng.shuffle(order)
                docs = [docs[i] for i in order]
                gold = [j for j, i in enumerate(order) if i < len(pages)]
                yield {
                    "docs": [""] * len(docs),
                    "doc_images": [image_bytes_feature(d) for d in docs],
                    "query": q,
                    "mistral_label": a,
                    "answers": [a],
                    "gold_positions": gold,
                    "source": source,
                    "query_id": f"{source}-{seen}-{qas.index(t)}",
                }
                out += 1
                if out >= n:
                    return
        for p in pages:
            pool.add(key, p)
        if seen % 500 == 0:
            print(f"[{source}] seen {seen} docs, wrote {out} rows", flush=True)


def iter_text_rag_rows(*, n, topk, rng, seed):
    """Text RAG rows (no images) so the multimodal finetune rehearses text compression.
    Same schema; doc_images are all None, gold_positions unknown (left empty)."""
    import datasets
    ds = datasets.load_dataset(TEXT_RAG, split="train", streaming=True).shuffle(seed=seed, buffer_size=5000)
    out = 0
    for i, r in enumerate(ds):
        docs = [d for d in (r.get("docs") or []) if isinstance(d, str) and d.strip()][:topk]
        q, a = (r.get("query") or "").strip(), (r.get("mistral_label") or "").strip()
        if not docs or not q or not a:
            continue
        yield {
            "docs": docs,
            "doc_images": [None] * len(docs),
            "query": q,
            "mistral_label": a,
            "answers": [a],
            "gold_positions": [],
            "source": "text_rag",
            "query_id": f"text_rag-{i}",
        }
        out += 1
        if out >= n:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--mix", default="docmatix=10000,docvqa=7000,infographic_vqa=2000,chartqa=1000")
    ap.add_argument("--distractors", type=int, default=3)
    ap.add_argument("--max_gold_pages", type=int, default=2, help="Docmatix docs: first N pages kept as gold")
    ap.add_argument("--qa_per_doc", type=int, default=2)
    ap.add_argument("--max_answer_chars", type=int, default=160, help="drop verbose QA pairs (Docmatix answers are sentences)")
    ap.add_argument("--text_topk", type=int, default=4, help="text_rag rows: number of text docs kept")
    ap.add_argument("--max_pixels", type=int, default=451584)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    plan = dict((k, int(v)) for k, v in (kv.split("=") for kv in args.mix.split(",") if kv))
    writer = ShardWriter(args.out, ft_features(), shard_size=2000)
    total = 0
    for src, n in plan.items():
        if n <= 0 or total >= args.n:
            continue
        n = min(n, args.n - total)
        if src == "text_rag":
            print(f"=== text_rag: {TEXT_RAG}, target {n} rows (top-{args.text_topk} text docs, no images)")
            k = 0
            for row in iter_text_rag_rows(n=n, topk=args.text_topk, rng=rng, seed=args.seed):
                writer.add(row); k += 1
            total += k
            print(f"=== text_rag: wrote {k} rows (total {total})")
            continue
        repo, config = (DOCMATIX, "images") if src == "docmatix" else (CAULDRON, src)
        print(f"=== {src}: {repo}/{config}, target {n} rows")
        k = 0
        for row in iter_docqa_rows(repo, config, source=src, n=n, qa_per_doc=args.qa_per_doc,
                                   max_gold_pages=args.max_gold_pages if src == "docmatix" else 1,
                                   distractors=args.distractors, max_pixels=args.max_pixels, rng=rng,
                                   max_answer_chars=args.max_answer_chars):
            writer.add(row); k += 1
        total += k
        print(f"=== {src}: wrote {k} rows (total {total})")
    writer.finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
