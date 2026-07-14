"""
Phase 1 (long-doc compression study): eval-time knob sweep on a fixed PISCO
checkpoint, across document-length buckets, reusing the generation + metric code
in scripts/eval_agent_qa.py. Loads the (large) checkpoint ONCE and loops over all
configs in-process, so we don't pay the model-load cost per config.

For each (bucket x compressor_max_length x chunk_overlap x decoder_max_length) cell
it reports QA match/F1 plus compression diagnostics (effective_ratio, coverage,
frac_examples_trimmed). A `--mode base` row per bucket gives the uncompressed
ceiling so QA can be read as a fraction of it.

Usage (on a GPU):
    pixi run python scripts/sweep_lenbuckets.py \
        --checkpoint_path expft/pt_95370/model \
        --buckets_dir eval_data/agentqa_buckets \
        --out_csv outputs/phase1_sweep.csv
"""

import argparse
import csv
import glob
import os
from typing import Any, Dict, List

import torch

from pisco.metrics import f1_single, match_single
from pisco.model import PISCO, PISCOConfig
from scripts.eval_agent_qa import (  # reuse, single source of truth
    _generate_with_base_decoder,
    _generate_with_pisco,
    _load_json,
)


def _eval_bucket_pisco(model, examples, *, device, cml, dml, overlap, max_new_tokens):
    ms, f1s, ratios, covs, trimmed = [], [], [], [], 0
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
        trimmed += int(stats["mems_trimmed"] > 0)
    n = max(1, len(examples))
    return {
        "match": sum(ms) / n,
        "f1": sum(f1s) / n,
        "effective_ratio": sum(ratios) / n,
        "coverage": sum(covs) / n,
        "frac_trimmed": trimmed / n,
    }


def _eval_bucket_base(model, tok, examples, *, device, dml, max_new_tokens):
    ms, f1s = [], []
    for ex in examples:
        pred = _generate_with_base_decoder(
            model,
            tok,
            trajectory_text=str(ex.get("trajectory", "")),
            question=str(ex.get("question", "")),
            device=device,
            max_new_tokens=max_new_tokens,
            decoder_max_length=dml,
        )
        gt = str(ex.get("ground_truth", ""))
        ms.append(float(match_single(pred, gt)))
        f1s.append(float(f1_single(pred, gt)[0]))
    n = max(1, len(examples))
    return {"match": sum(ms) / n, "f1": sum(f1s) / n}


def main():
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--buckets_dir", default="eval_data/agentqa_buckets")
    ap.add_argument("--out_csv", default="outputs/phase1_sweep.csv")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--cml_list", type=int, nargs="+", default=[128, 256, 512, 1024])
    ap.add_argument("--overlap_list", type=int, nargs="+", default=[0, 16, 64])
    ap.add_argument("--dml_list", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--with_base", action="store_true", help="also run the uncompressed baseline per bucket")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    bucket_files = sorted(glob.glob(os.path.join(args.buckets_dir, "*.json")))
    buckets = {os.path.basename(f).replace(".json", ""): _load_json(f) for f in bucket_files}
    print("buckets:", {k: len(v) for k, v in buckets.items()})

    rows: List[Dict[str, Any]] = []

    # ---- PISCO sweep (model loaded once) ----
    model = PISCO.from_pretrained(args.checkpoint_path)
    model.to(device)
    model.eval()

    # 1-D sweeps around a center point to keep cell count tractable:
    #   - compressor_max_length sweep (overlap=0, dml=max)
    #   - chunk_overlap sweep (cml=center, dml=max)
    #   - decoder_max_length sweep (cml=center, overlap=0)
    dml_max = max(args.dml_list)
    cml_center = args.cml_list[0]
    cells = set()
    for cml in args.cml_list:
        cells.add((cml, 0, dml_max))
    for ov in args.overlap_list:
        cells.add((cml_center, ov, dml_max))
    for dml in args.dml_list:
        cells.add((cml_center, 0, dml))

    for bname, exs in buckets.items():
        for (cml, ov, dml) in sorted(cells):
            r = _eval_bucket_pisco(
                model, exs, device=device, cml=cml, dml=dml, overlap=ov,
                max_new_tokens=args.max_new_tokens,
            )
            row = {"mode": "pisco", "bucket": bname, "n": len(exs),
                   "cml": cml, "overlap": ov, "dml": dml, **r}
            rows.append(row)
            print(f"[pisco] {bname} cml={cml} ov={ov} dml={dml} -> "
                  f"match={r['match']:.3f} f1={r['f1']:.3f} ratio={r['effective_ratio']:.1f} "
                  f"cov={r['coverage']:.2f} trim={r['frac_trimmed']:.2f}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- uncompressed baseline (optional) ----
    if args.with_base:
        cfg = PISCOConfig.from_pretrained(args.checkpoint_path)
        tok = AutoTokenizer.from_pretrained(cfg.decoder_model_name, padding_side="left")
        if tok.pad_token_id is None:
            tok.pad_token_id = int(tok.eos_token_id)
        dtype = torch.bfloat16 if device.type == "cuda" else None
        try:
            base = AutoModelForImageTextToText.from_pretrained(
                cfg.decoder_model_name, torch_dtype=dtype
            ).to(device)
        except Exception as e:
            print(f"AutoModelForImageTextToText failed ({e}); falling back to AutoModelForCausalLM")
            base = AutoModelForCausalLM.from_pretrained(
                cfg.decoder_model_name, torch_dtype=dtype
            ).to(device)
        base.eval()
        for bname, exs in buckets.items():
            r = _eval_bucket_base(base, tok, exs, device=device, dml=dml_max,
                                  max_new_tokens=args.max_new_tokens)
            rows.append({"mode": "base", "bucket": bname, "n": len(exs),
                         "cml": "", "overlap": "", "dml": dml_max, **r})
            print(f"[base ] {bname} dml={dml_max} -> match={r['match']:.3f} f1={r['f1']:.3f}")

    # ---- write CSV ----
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    fields = ["mode", "bucket", "n", "cml", "overlap", "dml",
              "match", "f1", "effective_ratio", "coverage", "frac_trimmed"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})
    print(f"\nwrote {len(rows)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
