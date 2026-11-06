"""Per-head output scales on the decoder's full-attention layers (SEAL, arXiv 2501.15225; probe of 2026-11-06).

`install_head_scales(decoder, init)` puts a forward pre-hook on every full-attention `o_proj`: the head outputs
(B, L, H * head_dim) are viewed as (B, L, H, head_dim) and multiplied by one scalar per head before the projection.
With `init=None` the scales start at 1 and are trainable (the probe); with a saved JSON (`load_head_scales`) they are
applied at inference (eval_mm_docqa --head_scales). Qwen3.5's linear-attention layers are left alone.
"""
import json
from typing import List, Optional, Tuple

import torch
from torch import nn


def full_attention_modules(decoder: nn.Module) -> List[Tuple[int, nn.Module]]:
    """(layer index, attention module) for every module that owns an `o_proj` (the full-attention layers)."""
    out = []
    for name, m in decoder.named_modules():
        if hasattr(m, "o_proj") and hasattr(m, "head_dim") and type(m).__name__.endswith("Attention"):
            li = getattr(m, "layer_idx", None)
            if li is None:   # fall back to the index in the module path (…layers.N.self_attn)
                parts = name.split(".")
                li = int(parts[parts.index("layers") + 1]) if "layers" in parts else len(out)
            out.append((int(li), m))
    out.sort(key=lambda t: t[0])
    return out


def install_head_scales(decoder: nn.Module, init: Optional[torch.Tensor] = None):
    """Returns (scales Parameter of shape (n_layers, n_heads), layer indices). Hooks stay registered on `decoder`."""
    mods = full_attention_modules(decoder)
    assert mods, "no full-attention module with o_proj found"
    n_heads = [m.o_proj.in_features // m.head_dim for _, m in mods]
    assert len(set(n_heads)) == 1, n_heads
    H = n_heads[0]
    dev = next(decoder.parameters()).device
    if init is None:
        scales = nn.Parameter(torch.ones(len(mods), H, device=dev, dtype=torch.float32))
    else:
        scales = nn.Parameter(init.to(dev, torch.float32), requires_grad=False)
    for r, (_, m) in enumerate(mods):
        def pre(mod, inp, r=r, hd=m.head_dim):
            x = inp[0]
            s = scales[r].to(x.dtype)
            y = (x.view(*x.shape[:-1], -1, hd) * s.view(*([1] * (x.dim() - 1)), -1, 1)).reshape(x.shape)
            return (y,) + tuple(inp[1:])
        m.o_proj.register_forward_pre_hook(pre)
    return scales, [li for li, _ in mods]


def load_head_scales(decoder: nn.Module, path: str):
    """Apply the scales saved by scripts/probe_seal.py; returns the Parameter."""
    j = json.load(open(path))
    scales, layers = install_head_scales(decoder, init=torch.tensor(j["scales"]))
    assert layers == list(j["layers"]), (layers, j["layers"])
    return scales
