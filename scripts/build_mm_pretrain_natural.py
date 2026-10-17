"""Stage-2 NATURAL-IMAGE data: photos + a detailed description, for a generic (not document-only) image compressor.
Source: HuggingFaceM4/the_cauldron, config `localized_narratives` (COCO / Open Images / Flickr30k / ADE20k photos with
human spoken descriptions, ~40 words: "In this image we can see a person standing on the grass, ..."). The description
is the image-autoencoding target, as the page text is for document pages. Rows {image, text, title, docid, source},
the same schema as data/mm_pretrain_* so the sets can be mixed with train.py's `path:N,path:N` spec.

Photos are resized to --max_pixels (451584 by default: photos do not need the 1M budget dense pages need) and stored as
JPEG. Every stored size is a fixed point of smart_resize, also under a larger processor budget, so photos and 1M-px
pages can share one model config. Resumable (ShardWriter + stream skip).
  python scripts/build_mm_pretrain_natural.py --out data/mm_pretrain_natural --n 60000
"""
import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mm_data_utils import ShardWriter, image_bytes_feature, prepare_page  # noqa: E402

CAULDRON = "HuggingFaceM4/the_cauldron"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--configs", default="localized_narratives", help="the_cauldron configs: a,b or a=N,b=M")
    ap.add_argument("--max_pixels", type=int, default=451584)
    ap.add_argument("--min_words", type=int, default=15)
    ap.add_argument("--max_chars", type=int, default=3000)
    ap.add_argument("--probe", type=int, default=0, help="print N rows and exit (no writing)")
    args = ap.parse_args()
    import datasets
    feats = datasets.Features({"image": datasets.Image(), "text": datasets.Value("string"),
                               "title": datasets.Value("string"), "docid": datasets.Value("string"),
                               "source": datasets.Value("string")})
    # "--configs a,b" splits --n evenly; "--configs a=27000,b=10000" sets each config's rows (a config with fewer
    # usable rows simply ends early).
    specs = [c.strip() for c in args.configs.split(",") if c.strip()]
    if all("=" in c for c in specs):
        configs = [c.split("=")[0] for c in specs]
        targets = [int(c.split("=")[1]) for c in specs]
    else:
        configs = specs
        per = args.n // len(configs)
        targets = [per] * (len(configs) - 1) + [args.n - per * (len(configs) - 1)]
    writer = None if args.probe else ShardWriter(args.out, feats, shard_size=5000)
    done_before = 0 if args.probe else writer.n_written
    starts = [sum(targets[:i]) for i in range(len(targets))]
    for ci, cfg in enumerate(configs):
        target = targets[ci]
        already = max(0, min(target, done_before - starts[ci]))   # rows of this config written by a previous run
        if already >= target:
            continue
        ds = datasets.load_dataset(CAULDRON, cfg, split="train", streaming=True)
        n_cfg, n_seen, n_short = already, 0, 0
        for row in ds:
            n_seen += 1
            if n_seen <= already:           # resume: these source rows were consumed before (approximate if
                continue                    # rows were skipped as short; harmless duplicates at worst)
            imgs, texts = row.get("images") or [], row.get("texts") or []
            if len(imgs) != 1 or not texts:
                continue
            desc = " ".join((texts[0].get("assistant") or "").split())
            if len(desc.split()) < args.min_words:
                n_short += 1
                continue
            if args.probe:
                print(n_seen, imgs[0].size, "|", texts[0].get("user", "")[:50].replace("\n", " "), "|",
                      len(desc.split()), "words:", desc[:200])
                if n_seen >= args.probe:
                    return 0
                continue
            try:
                jpg = prepare_page(imgs[0], max_pixels=args.max_pixels, crop_aspect=None)
            except Exception as e:
                print(f"  skip row {n_seen}: {type(e).__name__}: {e}")
                continue
            writer.add({"image": image_bytes_feature(jpg), "text": desc[: args.max_chars], "title": "",
                        "docid": f"{cfg}:{n_seen}", "source": f"cauldron/{cfg}"})
            n_cfg += 1
            if n_cfg % 5000 == 0:
                print(f"[natural] {cfg}: {n_cfg}/{target} (seen {n_seen}, too short {n_short})", flush=True)
            if n_cfg >= target:
                break
        print(f"[natural] {cfg}: done {n_cfg} rows (seen {n_seen}, too short {n_short})", flush=True)
    writer.finalize()
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc or 0)   # streamed-download threads kept the probe alive for hours after main() returned
