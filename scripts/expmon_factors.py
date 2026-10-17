#!/usr/bin/env python3
"""EXP_FACTORS_CMD hook for expmon: `<launcher> <argv...>` -> `key=val` lines.

Maps the positional arguments of the repo's launchers to named factors so `expmon runs/grid`
can compare runs by what they tested. Unknown launchers print nothing (expmon then shows a
bare jobid). Hydra overrides given after the positionals (pt_mm_4b.sh) are passed through
as `k=v` when they look like `a.b.c=value`.
"""
import os
import re
import sys

POSITIONALS = {
    # launcher: (list of factor names in positional order, defaults)
    "pt_mm_4b.sh": (["samples", "bs", "accum", "vision_lora", "init"],
                    ["160000", "8", "8", "merger", "qwen35_4b_pt_5M_143228"]),
    "ft_mm_4b.sh": (["init", "samples", "bs", "accum"], [None, "20000", "2", "32"]),
    "eval_mm_4b.sh": (["ckpt", "mode", "limit", "data", "image_rate"], [None, "pisco", "all", "mm_docqa_500", "ckpt"]),
    "mm_shakeout.sh": (["ckpt", "batch_sizes", "vision_lora", "extra"],
                       ["qwen35_4b_pt_5M_143228", "2,4,8", "merger", ""]),
    "smoke_mm.sh": (["pt_data", "ft_data", "eval_data"], ["mm_pretrain_160k", "mm_ft_docpages_20k", "mm_docqa_500"]),
    "build_mm_data.sh": (["what", "n_pretrain"], ["all", "160000"]),
    "pt_causal_4b4b.sh": (["samples", "bs", "accum"], ["500000", "28", "2"]),
}
KIND = {"pt_mm_4b.sh": "pt_mm", "ft_mm_4b.sh": "ft_mm", "eval_mm_4b.sh": "eval_mm", "mm_shakeout.sh": "shakeout",
        "smoke_mm.sh": "smoke_mm", "build_mm_data.sh": "data", "pt_causal_4b4b.sh": "pt_causal",
        "mmrl_split_4b.sh": "mmrl_split"}
# Launcher knobs passed as environment variables (expmon runs this hook in the launching shell).
ENV_FACTORS = {
    "ft_mm_4b.sh": {"FT_DATA": "ft_data"},
    "eval_mm_4b.sh": {"TOPK": "topk", "PX": "max_pixels", "NSAMP": "samples_per_q", "MERGE": "merged"},
}


def short(v: str) -> str:
    """Paths -> the run/model name they point at (…/expQ/<run>/model -> <run>)."""
    if v and "/" in v:
        parts = [p for p in v.rstrip("/").split("/") if p]
        if parts and parts[-1] in ("model",) and len(parts) > 1:
            return parts[-2]
        return parts[-1] if parts else v
    return v


def main(argv):
    if not argv:
        return 0
    launcher = os.path.basename(argv[0])
    args = argv[1:]
    print(f"launcher={launcher}")
    if launcher in KIND:
        print(f"kind={KIND[launcher]}")
    for var, name in ENV_FACTORS.get(launcher, {}).items():
        if os.environ.get(var):
            print(f"{name}={short(os.environ[var])}")
    spec = POSITIONALS.get(launcher)
    if not spec:
        return 0
    names, defaults = spec
    pos = [a for a in args if not re.match(r"^\+?[\w.]+=", a)]
    overrides = [a for a in args if re.match(r"^\+?[\w.]+=", a)]
    for i, name in enumerate(names):
        val = pos[i] if i < len(pos) else defaults[i]
        if val is None or val == "":
            continue
        print(f"{name}={short(str(val))}")
    for ov in overrides:
        k, v = ov.lstrip("+").split("=", 1)
        print(f"{k.split('.')[-1]}={v}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
