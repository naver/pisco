#!/usr/bin/env python3
"""Display main parameters of PISCO checkpoints under one or more dirs.

A PISCO checkpoint is detected by the presence of `model/config.json` with
`model_type == "PISCO"`. By default scans `exp4/` and `expft/`.

Usage:
    python scripts/show_pisco_params.py
    python scripts/show_pisco_params.py exp4 expft some/other/dir
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


FIELDS = [
    ("decoder_model_name", "decoder"),
    ("compressor_model_name", "compressor"),
    ("compr_rate", "rate"),
    ("compressor_mlp_hidden_dim", "mlp"),
    ("freeze_decoder", "frz_dec"),
    ("lora_decoder", "lora_dec"),
    ("lora_r_decoder", "r_dec"),
    ("lora_compressor", "lora_comp"),
    ("lora_r_compressor", "r_comp"),
]


def find_pisco_configs(root: Path):
    """Yield (checkpoint_dir, config_dict) for every PISCO model dir under root."""
    for config_path in sorted(root.rglob("config.json")):
        try:
            cfg = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if cfg.get("model_type") != "PISCO":
            continue
        yield config_path.parent, cfg


def load_training_data_info(ckpt_dir: Path) -> tuple[str, str]:
    """Read `training_config.yaml` from the experiment dir; return (dataset, samples).

    The yaml lives one level above the model/ dir. We only need two scalar fields,
    so we line-parse instead of pulling in PyYAML.
    """
    # search up to 3 parents (handles model/, checkpoint-NNN/, etc.)
    candidates = [ckpt_dir] + list(ckpt_dir.parents)[:3]
    for d in candidates:
        f = d / "training_config.yaml"
        if not f.exists():
            continue
        text = f.read_text()
        dataset = _grep_yaml_scalar(text, "training_dataset")
        samples = _grep_yaml_scalar(text, "samples")
        return (dataset or "-", samples or "-")
    return ("-", "-")


def _grep_yaml_scalar(text: str, key: str) -> str | None:
    m = re.search(rf"^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    if not m:
        return None
    return m.group(1).strip().strip("'\"")


def short(name):
    """Trim model paths for display: keep only the trailing segment(s)."""
    if not name:
        return "-"
    if "/" in name and not name.startswith("/"):
        return name  # HF id like "Qwen/Qwen3-0.6B" — already short
    return os.path.basename(name.rstrip("/")) or name


def render_table(headers, rows):
    """Print rows as a framed ASCII table. headers and each row are lists of strings."""
    cols = list(zip(headers, *rows))
    widths = [max(len(str(c)) for c in col) for col in cols]

    def hsep():
        return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def fmt_row(cells):
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"

    print(hsep())
    print(fmt_row(headers))
    print(hsep())
    for r in rows:
        print(fmt_row(r))
    print(hsep())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "roots",
        nargs="*",
        default=["exp4", "expft"],
        help="Directories to scan (default: exp4 expft).",
    )
    ap.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Also list intermediate checkpoint-NNN dirs, not just the final model/.",
    )
    ap.add_argument(
        "--full-paths",
        action="store_true",
        help="Show full model_name paths instead of trimmed basenames.",
    )
    args = ap.parse_args()

    rows = []
    for root in args.roots:
        root_path = Path(root)
        if not root_path.exists():
            print(f"WARNING: {root} does not exist, skipping", file=sys.stderr)
            continue
        for ckpt_dir, cfg in find_pisco_configs(root_path):
            if not args.include_checkpoints and ckpt_dir.name.startswith("checkpoint-"):
                continue
            rows.append((ckpt_dir, cfg))

    if not rows:
        print("No PISCO checkpoints found.")
        return

    fmt = short if not args.full_paths else (lambda x: "-" if x is None else str(x))

    headers = ["checkpoint"] + [h for _, h in FIELDS] + ["dataset", "samples"]
    table_rows = []
    for ckpt_dir, cfg in rows:
        # strip the trailing /model from the path for readability
        label = str(ckpt_dir.parent if ckpt_dir.name == "model" else ckpt_dir)
        row = [label]
        for key, _ in FIELDS:
            v = cfg.get(key, "?")
            if key.endswith("_model_name"):
                v = fmt(v)
            row.append(v)
        dataset, samples = load_training_data_info(ckpt_dir)
        row.append(dataset)
        row.append(samples)
        table_rows.append(row)

    render_table(headers, table_rows)


if __name__ == "__main__":
    main()
