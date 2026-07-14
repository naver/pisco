#!/usr/bin/env python3
"""Results table for agentic-QA evals (scripts/eval_agent_qa.py outputs).

Scans EXP_OUT/<run>/eval/results_*.json (default EXP_OUT=expQ) and prints one row
per eval, with the backbone read from the run's saved PISCO config. Fixed schema:

    RUN | MODEL | CKPT | N | MATCH | F1 | COMPR

Wired as EXP_RESULT_CMD so `expmon results` shows it. Pure stdlib.
"""
import glob
import json
import os
import sys

EXP_OUT = os.environ.get("EXP_OUT", "expQ")


def _backbone(run_dir: str) -> str:
    """compressor->decoder short names from the run's model/config.json."""
    cfg_path = os.path.join(run_dir, "model", "config.json")
    if not os.path.exists(cfg_path):
        # fall back to any checkpoint config
        cands = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*", "config.json")))
        cfg_path = cands[-1] if cands else None
    if not cfg_path or not os.path.exists(cfg_path):
        return "?"
    try:
        c = json.load(open(cfg_path))
    except Exception:
        return "?"
    short = lambda n: (n or "?").split("/")[-1].replace("Qwen3.5-", "")
    comp = short(c.get("compressor_model_name"))
    dec = short(c.get("decoder_model_name"))
    bidi = "·bidi" if c.get("bidirectional") else ""
    return f"{comp}->{dec}{bidi}"


def _dataset_tag(payload: dict) -> str:
    """Short dataset/bucket label from data_path.
    longbench .../multifieldqa_en_4000_8000.json -> 'mfqa 4-8k'; agentQA.json -> 'agentQA'."""
    dp = (payload.get("data_path") or "").rstrip("/")
    base = os.path.basename(dp)[:-len(".json")] if dp.endswith(".json") else dp
    if "multifieldqa_en_" in base:
        rng = base.split("multifieldqa_en_")[-1]            # e.g. 4000_8000 / 12000_inf
        lo, _, hi = rng.partition("_")
        k = lambda s: ("inf" if s == "inf" else f"{int(s)//1000}k")
        return f"mfqa {k(lo)}-{k(hi)}"
    return base or "?"


def _ckpt_tag(payload: dict, fname: str) -> str:
    cp = payload.get("checkpoint_path") or ""
    if "checkpoint-" in cp:
        return "ck-" + cp.rstrip("/").split("checkpoint-")[-1]
    if cp.rstrip("/").endswith("model"):
        return "final"
    # else infer from filename results_<tag>.json
    base = os.path.basename(fname)[len("results_"):-len(".json")]
    return base or "?"


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(EXP_OUT, "*", "eval", "results_*.json"))):
        run = f.split(os.sep)[-3]
        try:
            p = json.load(open(f))
        except Exception:
            continue
        m = p.get("metrics", {})
        comp = p.get("compression") or {}
        ratio = comp.get("effective_ratio") if isinstance(comp, dict) else None
        mems = comp.get("avg_mems_used") if isinstance(comp, dict) else None
        mode = p.get("mode", "pisco")
        is_base = mode == "base"
        rows.append({
            "run": run,
            "model": _backbone(os.path.join(EXP_OUT, run)),
            "mode": mode,
            "data": _dataset_tag(p),
            # base mode does no compression -> chunk/rate are meaningless there
            "chunk": "-" if is_base else p.get("compressor_max_length", "-"),
            "rate": "-" if is_base else p.get("compr_rate", "-"),
            "ckpt": _ckpt_tag(p, f),
            "n": p.get("n_examples", "-"),
            "match": m.get("match"),
            "f1": m.get("f1"),
            "mems": mems,
            "compr": f"{ratio:.1f}x" if isinstance(ratio, (int, float)) else "-",
        })

    if not rows:
        print(f"No evals found under {EXP_OUT}/*/eval/results_*.json")
        return

    # group by model+ckpt, then by dataset/bucket ascending (length order)
    def _bucket_key(d):
        # mfqa N-…  -> sort by N; non-mfqa first
        if d.startswith("mfqa "):
            lo = d.split()[1].split("-")[0].replace("k", "")
            return (1, int(lo) if lo.isdigit() else 0)
        return (0, 0)
    # group by model+ckpt, then bucket length, then chunk size (128 vs 256 adjacent)
    # group by model+ckpt, then bucket length, then chunk size, then rate
    # group by model+ckpt, then bucket length, then mode (base first), chunk, rate
    rows.sort(key=lambda r: (r["run"], r["ckpt"], _bucket_key(r["data"]), r["mode"] != "base",
                             r["chunk"] if isinstance(r["chunk"], int) else 0,
                             r["rate"] if isinstance(r["rate"], int) else 0))
    hdr = f"{'RUN':>8} | {'MODEL':<16} | {'MODE':<5} | {'DATA':<12} | {'CHUNK':>5} | {'RATE':>4} | {'CKPT':>8} | {'N':>3} | {'MATCH':>6} | {'F1':>6} | {'MEMS':>5} | {'COMPR':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        mt = f"{r['match']:.4f}" if isinstance(r["match"], (int, float)) else "-"
        f1 = f"{r['f1']:.4f}" if isinstance(r["f1"], (int, float)) else "-"
        mems = f"{r['mems']:.0f}" if isinstance(r["mems"], (int, float)) else "-"
        print(f"{r['run']:>8} | {r['model']:<16} | {r['mode']:<5} | {r['data']:<12} | {str(r['chunk']):>5} | {str(r['rate']):>4} | {r['ckpt']:>8} | {str(r['n']):>3} | {mt:>6} | {f1:>6} | {mems:>5} | {r['compr']:>6}")


if __name__ == "__main__":
    sys.exit(main())
