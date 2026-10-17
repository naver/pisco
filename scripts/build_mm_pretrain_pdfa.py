"""Dense-page stage-2 data: pixparse/pdfa-eng-wds (18.7M PDF pages with OCR words/lines, webdataset tars).

Why: the stage-2 compressor only saw Wikipedia screenshots (large type) and fails on dense pages
(mpdocvqa judge .45 vs .86 uncompressed; neither 4x MEM nor more finetune data helps). pdfa gives real
PDF pages WITH their text for free -> image-autoencoding targets on exactly the pages that fail.

Each sample = one PDF (`pdf` bytes) + `json` with `pages[i].lines` (reading-order text). We render up to
`--pages_per_doc` pages with pypdfium2 at a scale that lands near the pixel budget, keep pages with
>= --min_words, resize/JPEG with the shared prepare_page, store {image, text, title, docid, source}.

Sharded over the 1,800 tars for parallel CPU jobs:
  python scripts/build_mm_pretrain_pdfa.py --out data/mm_pretrain_pdfa --n 25000 --shard 0 --n_shards 4
then  --merge  (concatenate shard_* dirs into <out>).
"""

import argparse
import os
import io
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mm_data_utils import ShardWriter, image_bytes_feature, prepare_page  # noqa: E402

REPO = "pixparse/pdfa-eng-wds"
N_TARS = 1800


def page_text(meta, i):
    try:
        pg = meta["pages"][i]
    except (KeyError, IndexError, TypeError):
        return ""
    lines = pg.get("lines") or {}
    txt = lines.get("text") if isinstance(lines, dict) else None
    if not txt:
        words = pg.get("words") or {}
        txt = words.get("text") if isinstance(words, dict) else None
    if isinstance(txt, list):
        return "\n".join(str(t) for t in txt)
    return str(txt or "")


def render(pdf_bytes, page_idx, target_px):
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(pdf_bytes)
    try:
        if page_idx >= len(doc):
            return None
        page = doc[page_idx]
        w, h = page.get_size()                      # points (1/72 in)
        scale = math.sqrt(target_px * 1.3 / max(1.0, w * h))   # render ~30% above budget, prepare_page downsizes
        img = page.render(scale=scale).to_pil().convert("RGB")
        return img
    finally:
        doc.close()


def features():
    import datasets
    return datasets.Features({
        "image": datasets.Image(), "text": datasets.Value("string"), "title": datasets.Value("string"),
        "docid": datasets.Value("string"), "source": datasets.Value("string"),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=25000, help="pages for THIS shard")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--pages_per_doc", type=int, default=2)
    ap.add_argument("--min_words", type=int, default=80, help="dense pages only")
    ap.add_argument("--max_chars", type=int, default=6000)
    ap.add_argument("--max_pixels", type=int, default=451584)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    import datasets
    out = pathlib.Path(args.out)
    if args.merge:
        parts = [datasets.load_from_disk(str(p))["train"] if (p / "dataset_dict.json").exists() else datasets.load_from_disk(str(p))
                 for p in sorted(out.glob("shard_*"))]
        full = datasets.concatenate_datasets(parts)
        datasets.DatasetDict({"train": full}).save_to_disk(str(out))
        print(f"merged {len(parts)} shards -> {out}: {len(full)} rows")
        return

    tars = [f"pdfa-eng-train-{i:04d}.tar" for i in range(args.shard, N_TARS, args.n_shards)]
    urls = [f"https://huggingface.co/datasets/{REPO}/resolve/main/{t}" for t in tars]
    # One stream PER TAR: a network hiccup inside a tar ("tarfile.ReadError: unexpected end of data", shards 1/3 of the
    # 1M rebuild) only costs the rest of that tar instead of the whole shard. Start from an EMPTY shard dir: the loop
    # does not skip rows a previous run already wrote.
    writer = ShardWriter(str(out / f"shard_{args.shard:02d}"), features(), shard_size=2500)
    assert writer.n_written == 0, f"shard_{args.shard:02d} already has {writer.n_written} rows: delete it to rebuild"
    kept = skipped = seen = 0
    for url in urls:
        if kept >= args.n:
            break
        try:
            ds = datasets.load_dataset("webdataset", data_files={"train": [url]}, split="train", streaming=True)
            for ex in ds:
                seen += 1
                pdf, meta = ex.get("pdf"), ex.get("json")
                if pdf is None or not isinstance(meta, dict):
                    continue
                for i in range(args.pages_per_doc):
                    txt = page_text(meta, i).strip()
                    if len(txt.split()) < args.min_words:
                        skipped += 1
                        continue
                    try:
                        img = render(pdf, i, args.max_pixels)
                        if img is None:
                            break
                        jpg = prepare_page(img, max_pixels=args.max_pixels)
                    except Exception as e:
                        print(f"  render fail {ex.get('__key__')} p{i}: {type(e).__name__}: {str(e)[:60]}")
                        break
                    writer.add({"image": image_bytes_feature(jpg), "text": txt[: args.max_chars],
                                "title": str(ex.get("__key__", "")), "docid": f"{ex.get('__key__', '')}-p{i}", "source": "pdfa"})
                    kept += 1
                if seen % 200 == 0:
                    print(f"[pdfa shard {args.shard}] docs {seen} kept pages {kept} skipped sparse {skipped}", flush=True)
                if kept >= args.n:
                    break
        except Exception as e:
            print(f"  tar failed {url.rsplit('/', 1)[-1]}: {type(e).__name__}: {str(e)[:80]} -- skipping the rest of it", flush=True)
    writer.finalize()
    print(f"[pdfa shard {args.shard}] done: {kept} pages from {seen} docs")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)   # streamed-download threads can keep the interpreter alive after main() returns
