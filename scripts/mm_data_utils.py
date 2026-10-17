"""Shared helpers for building multimodal PISCO datasets (page images + text).

Pages are resized ONCE here with the same `smart_resize` rule the Qwen image processor
applies at training time (dims multiple of 32, pixel count within [min, max]) and stored as
JPEG, so the on-the-fly resize is a no-op and the arrow files stay small (~90 KB/page at
451584 px, q85).
"""

import io
import json
import os
import pathlib
import sys
from typing import Dict, Iterable, List, Optional

from PIL import Image, ImageOps

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from pisco.mm_utils import DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS  # noqa: E402

FACTOR = 32  # patch 16 x spatial merge 2


def smart_resize(height: int, width: int, factor: int = FACTOR,
                 min_pixels: int = DEFAULT_MIN_PIXELS, max_pixels: int = DEFAULT_MAX_PIXELS):
    """Qwen2-VL smart_resize (same maths as transformers' implementation)."""
    import math
    try:
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import smart_resize as _sr
        return _sr(height, width, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
    except Exception:
        pass
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def prepare_page(img, max_pixels: int = DEFAULT_MAX_PIXELS, min_pixels: int = DEFAULT_MIN_PIXELS,
                 quality: int = 85, crop_aspect: Optional[float] = None) -> bytes:
    """PIL/bytes -> resized RGB JPEG bytes. `crop_aspect` keeps only the top `w*crop_aspect`
    rows of very tall screenshots (a 451584-px budget makes a 1:5 page unreadable)."""
    if isinstance(img, (bytes, bytearray)):
        img = Image.open(io.BytesIO(img))
    elif isinstance(img, dict) and "bytes" in img:
        img = Image.open(io.BytesIO(img["bytes"]))
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if crop_aspect is not None and h > w * crop_aspect:
        img = img.crop((0, 0, w, int(w * crop_aspect)))
        w, h = img.size
    h2, w2 = smart_resize(h, w, min_pixels=min_pixels, max_pixels=max_pixels)
    if (w2, h2) != (w, h):
        img = img.resize((w2, h2), Image.BICUBIC)
    # the stored geometry must be a fixed point of smart_resize
    assert smart_resize(h2, w2, min_pixels=min_pixels, max_pixels=max_pixels) == (h2, w2), (h2, w2)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def image_bytes_feature(b: Optional[bytes]):
    """Value for a datasets.Image() column (None stays None)."""
    return None if b is None else {"bytes": b, "path": None}


class ShardWriter:
    """Accumulate rows, flush every `shard_size` to <out>/shards/NNNNN as a datasets.Dataset,
    then `finalize()` concatenates into <out>/ (DatasetDict with a 'train' split). Resumable:
    existing shards are kept and counted."""

    def __init__(self, out_dir: str, features, shard_size: int = 5000):
        import datasets
        self.datasets = datasets
        self.out = pathlib.Path(out_dir)
        self.shard_dir = self.out / "shards"
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.features = features
        self.shard_size = shard_size
        self.buf: List[dict] = []
        self.existing = sorted(p for p in self.shard_dir.iterdir() if p.is_dir() and (p / "state.json").exists())
        self.n_written = sum(json.load(open(p / "state.json")).get("_num_rows", 0) if "_num_rows" in json.load(open(p / "state.json")) else len(self.datasets.load_from_disk(str(p))) for p in self.existing)
        self.next_idx = len(self.existing)
        if self.existing:
            print(f"[shards] resuming: {len(self.existing)} shards / {self.n_written} rows already in {self.shard_dir}")

    def add(self, row: dict) -> None:
        self.buf.append(row)
        if len(self.buf) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        cols = {k: [r[k] for r in self.buf] for k in self.features}
        ds = self.datasets.Dataset.from_dict(cols, features=self.features)
        path = self.shard_dir / f"{self.next_idx:05d}"
        ds.save_to_disk(str(path))
        self.n_written += len(self.buf)
        print(f"[shards] wrote {path.name} ({len(self.buf)} rows, total {self.n_written})", flush=True)
        self.next_idx += 1
        self.buf = []

    def finalize(self) -> None:
        self.flush()
        parts = [self.datasets.load_from_disk(str(p)) for p in sorted(self.shard_dir.iterdir()) if p.is_dir()]
        full = self.datasets.concatenate_datasets(parts)
        dd = self.datasets.DatasetDict({"train": full})
        dd.save_to_disk(str(self.out))
        print(f"[shards] final dataset: {len(full)} rows -> {self.out}")
        du = sum(f.stat().st_size for f in self.out.rglob("*") if f.is_file() and "shards" not in f.parts) / 2**30
        print(f"[shards] size on disk (without shards/): {du:.2f} GiB; `rm -r {self.shard_dir}` once verified")
