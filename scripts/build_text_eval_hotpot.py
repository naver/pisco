"""Build the TEXT doc-QA eval set from HotpotQA distractor (hotpotqa/hotpot_qa, `validation`, ungated): each question
with its 10 Wikipedia paragraphs, i.e. the 2 supporting paragraphs (gold) + 8 distractors that bigram TF-IDF retrieved
for the question. Same row schema as the multimodal eval sets (text documents: `doc_images` empty), so
scripts/eval_mm_docqa.py runs it in every mode, route-then-expand included (`gold_positions` = the 2 supporting
paragraphs). `source` = hotpot_bridge / hotpot_comparison (comparison questions need both paragraphs).

  python scripts/build_text_eval_hotpot.py --out eval_data/text_hotpotqa_1500 --n 1500
"""

import argparse
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_mm_finetune import ft_features  # noqa: E402
from mm_data_utils import ShardWriter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import datasets
    ds = datasets.load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    rng = random.Random(args.seed)
    idx = sorted(rng.sample(range(len(ds)), min(args.n, len(ds))))
    writer = ShardWriter(args.out, ft_features(), shard_size=500)
    n_gold, n_docs = [], []
    for i in idx:
        o = ds[i]
        sup = set(o["supporting_facts"]["title"])
        paras = [(t, f"{t}: " + "".join(s).strip()) for t, s in zip(o["context"]["title"], o["context"]["sentences"])]
        rng.shuffle(paras)
        gold = [j for j, (t, _) in enumerate(paras) if t in sup]
        n_gold.append(len(gold)); n_docs.append(len(paras))
        writer.add({
            "docs": [p for _, p in paras], "doc_images": [],
            "query": o["question"], "mistral_label": o["answer"], "answers": [o["answer"]],
            "gold_positions": gold, "source": f"hotpot_{o['type']}", "query_id": o["id"],
        })
    writer.finalize()
    print(f"{len(idx)} questions; docs per question {min(n_docs)}-{max(n_docs)}; gold per question "
          f"{dict((k, n_gold.count(k)) for k in sorted(set(n_gold)))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
