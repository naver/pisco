"""Build the multiple-choice IMAGE eval set used by "Position IDs Matter" (arXiv 2409.14364, Table 6) and the
VoCo-LLaMA line: ScienceQA-IMG (derek-thomas/ScienceQA `test`, rows with an image) and MMBench-EN dev
(lmms-lab/MMBench_EN `dev`). Both ungated. One image per question; the options go into the query text and the
model is asked for the letter. Same row schema as the other eval sets, so scripts/eval_mm_docqa.py runs it
unchanged; `answers` = [letter, "letter. text", text] so EM / judge accept any of the three. `source` = sqa_img |
mmbench. Pages are resized to --max_pixels like every other set.

  python scripts/build_mm_eval_mcq.py --out eval_data/mm_mcq_1000 --per_source sqa_img=500,mmbench=500
"""

import argparse
import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_mm_finetune import ft_features  # noqa: E402
from mm_data_utils import ShardWriter, image_bytes_feature, prepare_page  # noqa: E402

LETTERS = "ABCDEFGH"


def mcq_query(question: str, options, hint: str = "") -> str:
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    ctx = f"Context: {hint.strip()}\n" if hint and hint.strip() else ""
    return f"{ctx}Question: {question.strip()}\nOptions:\n{opts}\nAnswer with the letter of the correct option."


def row(src, qid, img, question, options, ans_idx, hint, max_pixels):
    letter, text = LETTERS[ans_idx], str(options[ans_idx])
    return {
        "docs": [""], "doc_images": [image_bytes_feature(prepare_page(img, max_pixels=max_pixels))],
        "query": mcq_query(question, options, hint), "mistral_label": f"{letter}. {text}",
        "answers": [letter, f"{letter}. {text}", text], "gold_positions": [0], "source": src, "query_id": f"{src}_{qid}",
    }


def is_nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip().lower() in ("nan", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_source", default="sqa_img=500,mmbench=500")
    ap.add_argument("--max_pixels", type=int, default=1003520)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    plan = dict((k, int(v)) for k, v in (kv.split("=") for kv in args.per_source.split(",")))
    rng = random.Random(args.seed)

    import datasets
    writer = ShardWriter(args.out, ft_features(), shard_size=500)
    if plan.get("sqa_img"):
        ds = datasets.load_dataset("derek-thomas/ScienceQA", split="test")
        ds = ds.cast_column("image", datasets.Image(decode=False))
        with_img = [i for i, im in enumerate(ds["image"]) if im is not None]
        idx = sorted(rng.sample(with_img, min(plan["sqa_img"], len(with_img))))
        print(f"sqa_img: {len(idx)} of {len(with_img)} image questions in the test split", flush=True)
        for i in idx:
            o = ds[i]
            writer.add(row("sqa_img", i, o["image"]["bytes"], o["question"], o["choices"], int(o["answer"]), o.get("hint") or "", args.max_pixels))
    if plan.get("mmbench"):
        ds = datasets.load_dataset("lmms-lab/MMBench_EN", split="dev")
        ds = ds.cast_column("image", datasets.Image(decode=False))
        idx = sorted(rng.sample(range(len(ds)), min(plan["mmbench"], len(ds))))
        print(f"mmbench: {len(idx)} of {len(ds)} dev questions", flush=True)
        for i in idx:
            o = ds[i]
            options = [o[k] for k in "ABCD" if not is_nan(o.get(k))]
            ans = "ABCD".index(str(o["answer"]).strip()[0])
            if ans >= len(options):
                continue
            writer.add(row("mmbench", o["index"], o["image"]["bytes"], o["question"], options, ans, o.get("hint") or "", args.max_pixels))
    writer.finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
