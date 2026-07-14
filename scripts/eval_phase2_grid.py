"""
Phase 2 (long-doc compression study): evaluate the trained compr_rate x chunk_length
grid across document-length buckets, producing the quality-vs-compression frontier.

Each cell lives at phase2/r{R}_c{C}/ft/model. We evaluate every cell at ITS trained
chunk length (compressor_max_length=C, overlap=0) across the length buckets, so the
table answers: for a given document length, how high can compr_rate go before QA
collapses, and does a larger trained chunk help? The uncompressed base ceiling is
read from the Phase 1 sweep (or recomputed once here if --with_base).

Usage (on a GPU):
    pixi run python scripts/eval_phase2_grid.py \
        --phase2_dir phase2 --buckets_dir eval_data/agentqa_buckets \
        --out_csv outputs/phase2_grid.csv
"""

import argparse
import csv
import glob
import os
import re
from typing import Any, Dict, List

import torch

from pisco.metrics import f1_single, match_single
from pisco.model import PISCO
from scripts.eval_agent_qa import _generate_with_pisco, _load_json


CELL_RE = re.compile(r"r(\d+)_c(\d+)")


def _eval_bucket(model, examples, *, device, cml, overlap, dml, max_new_tokens):
    ms, f1s, ratios, covs = [], [], [], []
    for ex in examples:
        pred, stats = _generate_with_pisco(
            model,
            trajectory_text=str(ex.get("trajectory", "")),
            question=str(ex.get("question", "")),
            device=device,
            compressor_max_length=cml,
            decoder_max_length=dml,
            max_new_tokens=max_new_tokens,
            chunk_overlap=overlap,
        )
        gt = str(ex.get("ground_truth", ""))
        ms.append(float(match_single(pred, gt)))
        f1s.append(float(f1_single(pred, gt)[0]))
        ratios.append(stats["effective_ratio"])
        covs.append(stats["coverage"])
    n = max(1, len(examples))
    return {
        "match": sum(ms) / n,
        "f1": sum(f1s) / n,
        "effective_ratio": sum(ratios) / n,
        "coverage": sum(covs) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2_dir", default="phase2")
    ap.add_argument("--buckets_dir", default="eval_data/agentqa_buckets")
    ap.add_argument("--out_csv", default="outputs/phase2_grid.csv")
    ap.add_argument("--dml", type=int, default=4096)
    ap.add_argument("--overlap", type=int, default=0, help="held at 0 for the clean rate/chunk frontier")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    bucket_files = sorted(glob.glob(os.path.join(args.buckets_dir, "*.json")))
    buckets = {os.path.basename(f).replace(".json", ""): _load_json(f) for f in bucket_files}
    print("buckets:", {k: len(v) for k, v in buckets.items()})

    cells = sorted(glob.glob(os.path.join(args.phase2_dir, "r*_c*", "ft", "model")))
    print(f"found {len(cells)} trained cells")

    rows: List[Dict[str, Any]] = []
    for cell in cells:
        m = CELL_RE.search(cell)
        if not m:
            print(f"  skip (unparseable): {cell}")
            continue
        rate, cml = int(m.group(1)), int(m.group(2))
        try:
            model = PISCO.from_pretrained(cell)
            model.to(device)
            model.eval()
        except Exception as e:
            print(f"  FAILED to load {cell}: {e}")
            continue
        for bname, exs in buckets.items():
            r = _eval_bucket(model, exs, device=device, cml=cml, overlap=args.overlap,
                             dml=args.dml, max_new_tokens=args.max_new_tokens)
            row = {"rate": rate, "cml": cml, "bucket": bname, "n": len(exs), **r}
            rows.append(row)
            print(f"[r={rate:2d} c={cml:4d}] {bname:16s} match={r['match']:.3f} "
                  f"f1={r['f1']:.3f} ratio={r['effective_ratio']:.1f} cov={r['coverage']:.2f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    fields = ["rate", "cml", "bucket", "n", "match", "f1", "effective_ratio", "coverage"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})
    print(f"\nwrote {len(rows)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
