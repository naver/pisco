"""Natural-image QA for the GENERIC compressor (single image per question, the mm_docqa schema:
docs=[""], doc_images=[img], query, answers, mistral_label, source, query_id).

  eval  -> eval_data/mm_naturalqa_500: held-out VAL/TEST sets whose photos are NOT stage-2 pretraining photos
           (stage-2 natural = localized narratives on COCO / Open Images / Flickr30k / ADE20k):
           lmms-lab/VizWiz-VQA (photos by blind users), xai-org/RealworldQA (real-world scenes),
           lmms-lab/textvqa validation (scene text; Open Images -> mild image-overlap risk, noted).
  ft    -> data/mm_ft_natural_8k: the_cauldron vqav2 / textvqa / aokvqa TRAIN splits (gold answers; teacher labels
           are added afterwards with relabel_mm_4b.sh, as for the doc finetune set).
  probe -> print a few rows of every source (schema check), write nothing.
Photos are resized to --max_pixels (451584) with the shared prepare_page.
  python scripts/build_mm_natural_qa.py --what eval --out eval_data/mm_naturalqa_500
"""
import argparse
import os
import io
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mm_data_utils import image_bytes_feature, prepare_page  # noqa: E402

EVAL_SOURCES = {   # name -> (repo, config, split, n)
    "vizwiz": ("lmms-lab/VizWiz-VQA", None, "val", 150),
    "realworldqa": ("xai-org/RealworldQA", None, "test", 150),
    "textvqa": ("lmms-lab/textvqa", None, "validation", 200),
}
FT_SOURCES = {     # the_cauldron configs (train splits) -> n
    "vqav2": 3000, "textvqa": 2500, "aokvqa": 2500,
}


def first_image(row):
    for k in ("image", "images", "img"):
        v = row.get(k)
        if isinstance(v, list):
            v = v[0] if v else None
        if v is not None:
            return v
    return None


def answers_of(row):
    """list of gold answer strings, whatever the source schema."""
    for k in ("answers", "answer", "multiple_choice_answer", "label"):
        v = row.get(k)
        if v is None:
            continue
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            out = [a.get("answer") if isinstance(a, dict) else a for a in v]
            return [str(a) for a in out if a not in (None, "")]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["probe", "eval", "ft"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_pixels", type=int, default=451584)
    args = ap.parse_args()
    import datasets
    from build_mm_finetune import clean_question
    feats = datasets.Features({
        "docs": datasets.Sequence(datasets.Value("string")), "doc_images": datasets.Sequence(datasets.Image()),
        "query": datasets.Value("string"), "answers": datasets.Sequence(datasets.Value("string")),
        "mistral_label": datasets.Value("string"), "gold_positions": datasets.Sequence(datasets.Value("int32")),
        "source": datasets.Value("string"), "query_id": datasets.Value("string")})
    rows = []

    def add(src, i, img, q, answers):
        jpg = prepare_page(img, max_pixels=args.max_pixels, crop_aspect=None)
        rows.append({"docs": [""], "doc_images": [image_bytes_feature(jpg)], "query": q, "answers": answers,
                     "mistral_label": answers[0], "gold_positions": [0], "source": src, "query_id": f"{src}:{i}"})

    if args.what in ("probe", "eval"):
        for name, (repo, cfg, split, n) in EVAL_SOURCES.items():
            ds = datasets.load_dataset(repo, cfg, split=split, streaming=True)
            got = 0
            for i, row in enumerate(ds):
                if args.what == "probe":
                    print(name, {k: (type(v).__name__ if not isinstance(v, (str, int, float)) else str(v)[:80])
                                 for k, v in row.items()})
                    if i >= 1:
                        break
                    continue
                img, q, ans = first_image(row), row.get("question") or row.get("query"), answers_of(row)
                if img is None or not q or not ans:
                    continue
                if name == "vizwiz" and (row.get("category") == "unanswerable"
                                         or sum(a.lower() == "unanswerable" for a in ans) * 2 >= len(ans)):
                    continue
                ans = [a for a in ans if a.lower() != "unanswerable"] or ans
                add(name, i, img, q.strip(), ans)
                got += 1
                if got >= n:
                    break
            print(f"[natural-qa] {name}: {got if args.what == 'eval' else 'probed'}", flush=True)
    if args.what in ("probe", "ft"):
        for cfg, n in FT_SOURCES.items():
            ds = datasets.load_dataset("HuggingFaceM4/the_cauldron", cfg, split="train", streaming=True)
            got = 0
            for i, row in enumerate(ds):
                imgs, texts = row.get("images") or [], row.get("texts") or []
                if args.what == "probe":
                    print(cfg, len(imgs), texts[:1])
                    if i >= 1:
                        break
                    continue
                if len(imgs) != 1 or not texts:
                    continue
                t = texts[0]
                q, a = clean_question(t.get("user") or ""), (t.get("assistant") or "").strip()
                if not q or not a or len(a) > 80:
                    continue
                add(cfg, i, imgs[0], q, [a])
                got += 1
                if got >= n:
                    break
            print(f"[natural-qa] cauldron/{cfg}: {got if args.what == 'ft' else 'probed'}", flush=True)
    if args.what == "probe":
        return 0
    ds = datasets.Dataset.from_list(rows, features=feats)
    datasets.DatasetDict({"train": ds}).save_to_disk(args.out)
    print(f"[natural-qa] wrote {args.out}: {len(ds)} rows {dict(__import__('collections').Counter(ds['source']))}")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc or 0)   # streamed-download threads kept the probe alive for hours after main() returned
