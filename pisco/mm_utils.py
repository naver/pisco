"""
PISCO
Copyright (c) 2026-present NAVER Corp.
All Rights Reserved.

Multimodal plumbing for PISCO: turning images into compressor-side "image spans"
(`<|vision_start|> <|image_pad|>*N <|vision_end|>`) that get trailing `<MEM>` tokens
exactly like a text chunk, plus the batching of pixel tensors next to the padded ids.

Design notes
- The compressor backbone (Qwen3.5) is a native VLM: the HF forward scatters the vision
  tower's output into the `<|image_pad|>` slots before the language model runs. So an
  image span is just another compressor chunk whose `<MEM>` hidden states we read out.
- An image span is ATOMIC: it is never chunked or truncated (cutting it makes the number of
  `<|image_pad|>` tokens disagree with `image_grid_thw` and the HF model raises).
- `compressor_max_length` governs text chunks only.
- The number of `<MEM>` tokens for an image follows the text rule, `len(span) // rate + 1`,
  so the decoder keeps seeing "1 MEM ~ `compr_rate` compressor tokens". `image_compr_rate`
  overrides the rate for images only (ablation knob).
- Contract that must hold everywhere: the row order of image chunks in the padded batch ==
  the concatenation order of `pixel_values` == the row order of `image_grid_thw`.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

MM_TOKENS = {
    "vision_start": "<|vision_start|>",
    "image_pad": "<|image_pad|>",
    "vision_end": "<|vision_end|>",
}

DEFAULT_MIN_PIXELS = 65536      # 256 x 256
DEFAULT_MAX_PIXELS = 451584     # official Qwen-VL finetune default, ~440 visual tokens


def load_image_processor(source: Optional[str], min_pixels: int = DEFAULT_MIN_PIXELS,
                         max_pixels: int = DEFAULT_MAX_PIXELS):
    """
    Qwen2-VL family image processor, PIL/numpy backend (no torchvision dependency, works
    offline). `AutoProcessor` is deliberately NOT used: it drags in the video processor,
    which needs torchvision. Tries the snapshot's preprocessor_config.json first, then
    falls back to the known Qwen3.5 constants.
    """
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
        Qwen2VLImageProcessorPil,
    )

    kwargs = dict(min_pixels=min_pixels, max_pixels=max_pixels)
    proc = None
    if source:
        try:
            proc = Qwen2VLImageProcessorPil.from_pretrained(source, **kwargs)
            print(f"[mm] image processor loaded from {source}")
        except Exception as e:  # missing preprocessor_config.json, offline miss, ...
            print(f"[mm] no image processor config at {source} ({type(e).__name__}); using Qwen3.5 defaults")
    if proc is None:
        proc = Qwen2VLImageProcessorPil(
            patch_size=16, merge_size=2, temporal_patch_size=2,
            image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5],
            **kwargs,
        )
    return proc


def get_vision_token_ids(tokenizer) -> Tuple[int, int, int]:
    """(vision_start, image_pad, vision_end) ids; raises if the tokenizer lacks them."""
    ids = tokenizer.convert_tokens_to_ids(
        [MM_TOKENS["vision_start"], MM_TOKENS["image_pad"], MM_TOKENS["vision_end"]]
    )
    unk = getattr(tokenizer, "unk_token_id", None)
    if any(i is None or i == unk for i in ids):
        raise ValueError(f"tokenizer has no Qwen vision tokens: {ids}")
    return int(ids[0]), int(ids[1]), int(ids[2])


def merge_size_of(image_processor) -> int:
    return int(getattr(image_processor, "merge_size", 2))


def n_image_tokens(grid_thw: Sequence[int], merge_size: int = 2) -> int:
    t, h, w = (int(x) for x in grid_thw)
    return t * h * w // (merge_size ** 2)


def to_rgb(image) -> "PIL.Image.Image":
    from PIL import Image, ImageOps

    if isinstance(image, dict) and "bytes" in image:   # undecoded datasets.Image row
        import io
        image = Image.open(io.BytesIO(image["bytes"]))
    if not isinstance(image, Image.Image):
        raise TypeError(f"expected a PIL image, got {type(image)}")
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def image_to_span(image, image_processor, vision_ids: Tuple[int, int, int]):
    """
    One image -> (span_ids, pixel_values (P, 1536) float32 numpy, grid_thw [t, h, w]).
    span_ids = [vision_start] + [image_pad] * N + [vision_end], N = t*h*w / merge^2.
    """
    vs, ip, ve = vision_ids
    out = image_processor(images=[to_rgb(image)], return_tensors="np")
    pixel_values = np.asarray(out["pixel_values"], dtype=np.float32)
    grid = [int(x) for x in np.asarray(out["image_grid_thw"]).reshape(-1)[:3]]
    n = n_image_tokens(grid, merge_size_of(image_processor))
    assert pixel_values.shape[0] == n * merge_size_of(image_processor) ** 2, (
        pixel_values.shape, grid)
    span = [vs] + [ip] * n + [ve]
    return span, pixel_values, grid


@dataclass
class MMChunkBatch:
    """Compressor-side chunks of ONE example (text chunks and image chunks interleaved in
    segment order), with the pixel tensors of the image chunks in the same order."""
    input_ids: List[List[int]] = field(default_factory=list)   # <MEM>s already appended
    n_mems: List[int] = field(default_factory=list)
    is_image: List[bool] = field(default_factory=list)
    pixel_values: List[np.ndarray] = field(default_factory=list)
    image_grid_thw: List[List[int]] = field(default_factory=list)

    def extend(self, other: "MMChunkBatch") -> None:
        self.input_ids.extend(other.input_ids)
        self.n_mems.extend(other.n_mems)
        self.is_image.extend(other.is_image)
        self.pixel_values.extend(other.pixel_values)
        self.image_grid_thw.extend(other.image_grid_thw)

    @property
    def total_mems(self) -> int:
        return int(sum(self.n_mems))


Segment = Tuple[str, Union[str, "PIL.Image.Image"]]   # ("text", str) | ("image", PIL)


def segments_to_chunks(
    segments: Sequence[Segment],
    *,
    compressor_tokenizer,
    image_processor,
    compr_rate: int,
    text_chunker: Callable[[List[int]], List[List[int]]],
    vision_ids: Tuple[int, int, int],
    image_compr_rate: Optional[int] = None,
    clean_text: Optional[Callable[[str], str]] = None,
    image_prefix_ids: Optional[List[int]] = None,
) -> MMChunkBatch:
    """
    Build the compressor chunks of one example from an ordered list of segments.
    - text: clean -> tokenize (plain, no chat template, like PretrainingCollator) ->
      `text_chunker` (caller decides: random chunks, truncation, sliding window) ->
      `add_memory_tokens_to_inputs`.
    - image: `image_to_span` -> one atomic chunk -> `add_memory_tokens_to_inputs` at
      `image_compr_rate or compr_rate`. `image_prefix_ids` (e.g. a query, OSCAR-style) are
      prepended to the span and count toward its MEM budget.
    """
    from pisco.collator_utils import add_memory_tokens_to_inputs

    out = MMChunkBatch()
    for kind, payload in segments:
        if kind == "text":
            text = clean_text(payload) if clean_text else payload
            ids = compressor_tokenizer(
                text, padding="do_not_pad", return_tensors=None, truncation=False
            )["input_ids"]
            chunks = [c for c in text_chunker(ids) if len(c) > 0] or [ids]
            chunks, n_mems = add_memory_tokens_to_inputs(chunks, compressor_tokenizer, compr_rate)
            out.input_ids.extend(chunks)
            out.n_mems.extend(int(n) for n in n_mems)
            out.is_image.extend([False] * len(chunks))
        elif kind == "image":
            if image_processor is None:
                raise ValueError("image segment given but the collator has no image_processor")
            span, pv, grid = image_to_span(payload, image_processor, vision_ids)
            if image_prefix_ids:
                span = list(image_prefix_ids) + span
            chunks, n_mems = add_memory_tokens_to_inputs(
                [span], compressor_tokenizer, image_compr_rate or compr_rate
            )
            out.input_ids.extend(chunks)
            out.n_mems.append(int(n_mems[0]))
            out.is_image.append(True)
            out.pixel_values.append(pv)
            out.image_grid_thw.append(grid)
        else:
            raise ValueError(f"unknown segment kind {kind!r}")
    return out


def collate_compressor(batches: Sequence[MMChunkBatch], collator, vision_ids: Tuple[int, int, int]) -> dict:
    """
    Pad all chunks of all examples (left padding, via `collator.compressor_pad`) and, when
    any image chunk is present, add the three multimodal tensors the Qwen3.5 forward needs.
    Returns ONLY the two text keys for an image-free batch, so text-only training keeps the
    exact legacy batch layout.
    """
    all_ids: List[List[int]] = []
    pixel_values: List[np.ndarray] = []
    grids: List[List[int]] = []
    for b in batches:
        all_ids.extend(b.input_ids)
        pixel_values.extend(b.pixel_values)
        grids.extend(b.image_grid_thw)

    pad_mult = getattr(collator, "pad_to_multiple_of", None)
    if pad_mult:
        padded = collator.compressor_tokenizer.pad(
            {"input_ids": all_ids}, padding="longest", pad_to_multiple_of=pad_mult, return_tensors="pt"
        )
    else:
        padded = collator.compressor_pad(all_ids)
    out = {
        "compressor_input_ids": padded["input_ids"],
        "compressor_attention_mask": padded["attention_mask"],
    }
    if not pixel_values:
        return out

    _, image_pad_id, _ = vision_ids
    mm_token_type_ids = (padded["input_ids"] == image_pad_id).long()
    grid_t = torch.tensor(grids, dtype=torch.long)
    merge = merge_size_of(getattr(collator, "image_processor", None))
    expected = int((grid_t.prod(-1) // (merge ** 2)).sum())
    got = int(mm_token_type_ids.sum())
    assert expected == got, f"image_pad tokens {got} != grid tokens {expected}"
    out["pixel_values"] = torch.from_numpy(np.concatenate(pixel_values, axis=0))
    out["image_grid_thw"] = grid_t
    out["mm_token_type_ids"] = mm_token_type_ids
    return out


MM_BATCH_KEYS = ("pixel_values", "image_grid_thw", "mm_token_type_ids")
