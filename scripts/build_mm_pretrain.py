"""Build the stage-2 multimodal pretraining set: (page image, page text) pairs.

Source: Tevatron/wiki-ss-corpus (1.27M Wikipedia screenshots WITH the page text; a
save_to_disk arrow dataset of 809 shards, ~480 MB each). Shards are downloaded one at a time
(hf_hub_download), every row is resized to the Qwen pixel budget and stored as JPEG, and the
raw shard is deleted — peak scratch usage stays ~1 GB above the output.

  python scripts/build_mm_pretrain.py --out data/mm_pretrain_160k --n 160000 \
      [--shards_seed 42] [--max_pixels 451584] [--crop_aspect 1.6] [--min_words 30]

Output: DatasetDict{'train'} with columns image (Image), text, title, docid, source.
(pixparse/pdfa-eng-wds, planned as a second source, ships PDFs not renders; no pdftoppm /
pypdfium2 in the env, so it is left out of v1.)
"""

import argparse
import os
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mm_data_utils import ShardWriter, image_bytes_feature, prepare_page  # noqa: E402

REPO = "Tevatron/wiki-ss-corpus"
N_SHARDS = 809


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=160000)
    ap.add_argument("--shards_seed", type=int, default=42, help="shard order (random subset of the corpus)")
    ap.add_argument("--max_pixels", type=int, default=451584)
    ap.add_argument("--crop_aspect", type=float, default=1.6,
                    help="keep only the top w*crop_aspect rows of taller screenshots (None/0 = no crop)")
    ap.add_argument("--min_words", type=int, default=30)
    ap.add_argument("--max_chars", type=int, default=6000)
    ap.add_argument("--keep_raw", action="store_true", help="do not delete downloaded shards")
    args = ap.parse_args()

    import datasets
    from huggingface_hub import hf_hub_download

    feats = datasets.Features({
        "image": datasets.Image(),
        "text": datasets.Value("string"),
        "title": datasets.Value("string"),
        "docid": datasets.Value("string"),
        "source": datasets.Value("string"),
    })
    writer = ShardWriter(args.out, feats, shard_size=5000)
    if writer.n_written >= args.n:
        print("already complete"); writer.finalize(); return 0

    order = list(range(N_SHARDS))
    random.Random(args.shards_seed).shuffle(order)
    # Resume: skip the shards whose rows are already written (approximate: by count).
    rows_per_shard_est = 1568
    skip = writer.n_written // rows_per_shard_est
    crop = args.crop_aspect if args.crop_aspect and args.crop_aspect > 0 else None
    n_skipped_text = 0

    for k, shard_idx in enumerate(order[skip:], start=skip):
        if writer.n_written >= args.n:
            break
        fname = f"train/data-{shard_idx:05d}-of-{N_SHARDS:05d}.arrow"
        print(f"[wiki-ss] shard {k + 1}: downloading {fname}", flush=True)
        path = hf_hub_download(REPO, fname, repo_type="dataset")
        ds = datasets.Dataset.from_file(path)
        if not isinstance(ds.features.get("image"), datasets.Image):
            ds = ds.cast_column("image", datasets.Image())
        for row in ds:
            text = (row.get("text") or "").strip()
            if len(text.split()) < args.min_words:
                n_skipped_text += 1
                continue
            try:
                jpg = prepare_page(row["image"], max_pixels=args.max_pixels, crop_aspect=crop)
            except Exception as e:
                print(f"  skip {row.get('docid')}: {type(e).__name__}: {e}")
                continue
            writer.add({
                "image": image_bytes_feature(jpg),
                "text": text[: args.max_chars],
                "title": row.get("title") or "",
                "docid": str(row.get("docid")),
                "source": "wiki-ss",
            })
            if writer.n_written + len(writer.buf) >= args.n:
                break
        del ds
        if not args.keep_raw:
            try:
                real = os.path.realpath(path)
                os.remove(real)
                if os.path.islink(path):
                    os.remove(path)
            except OSError as e:
                print(f"  could not delete {path}: {e}")
        print(f"[wiki-ss] rows so far {writer.n_written + len(writer.buf)} (skipped short-text {n_skipped_text})", flush=True)

    writer.finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
