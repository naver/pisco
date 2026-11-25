#!/usr/bin/env python3
"""Campaign longdoc, arm P (position probe): `scripts/eval_mm_docqa.py` run UNCHANGED, with one extra knob from env.

MPOS = an integer N, or a JSON file {query_id: N} (`scripts/longdoc/mpos_map.py`). The decoder position ids of the
first <MEM> and of every token after it (the other <MEM>s, the question, the generated answer: HF generate continues
from the last prompt id + 1) are raised by N; the text before the first <MEM> keeps its ids. Slot count, slot order,
embeddings and every other setting are untouched, so a 10-page document can be read at the positions a 40-page one
uses. MPOS unset or 0 = eval_mm_docqa.py byte for byte (the shared file is imported, never edited: siblings keep
pairing on its md5).
MPOS_REF = a recorded result json: after the run, every prediction must equal the reference's for the same
query_id, else exit 4 (the identity check that gates the probe cells via `--dependency afterok`).
Writes `<output>.mpos.json` = the offset applied per query_id. All argv go to eval_mm_docqa.main().
"""
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import eval_mm_docqa as E  # noqa: E402
import torch  # noqa: E402

SPEC = os.environ.get("MPOS", "0").strip() or "0"
TABLE = None if SPEC.lstrip("-").isdigit() else json.load(open(SPEC))
CUR = {"n": 0}
APPLIED = {}

_orig_pos, _orig_gen = E._pos_kwargs, E.gen_pisco


def _offset(row) -> int:
    if TABLE is None:
        return int(SPEC)
    qid = str(row.get("query_id"))
    if qid not in TABLE:
        raise KeyError(f"MPOS table {SPEC} has no offset for {qid}")
    return int(TABLE[qid])


def _pos_kwargs(model, ids, attn, mm=None):
    out = _orig_pos(model, ids, attn, mm)
    n = CUR["n"]
    if not n:
        return out
    if not out:
        raise RuntimeError("MPOS needs the spread <MEM> layout (mem_position_stride > 1)")
    pos = out["position_ids"]
    is_mem = (ids == model.decoder_tokenizer.mem_token_id) & (attn > 0)
    after = (torch.cumsum(is_mem.long(), -1) > 0).to(pos.dtype) * n
    return {"position_ids": pos + (after if pos.dim() == 2 else after.unsqueeze(0))}


def gen_pisco(model, coll, row, **kw):
    CUR["n"] = _offset(row)
    APPLIED[str(row.get("query_id"))] = CUR["n"]
    try:
        return _orig_gen(model, coll, row, **kw)
    finally:
        CUR["n"] = 0


E._pos_kwargs, E.gen_pisco = _pos_kwargs, gen_pisco


def main():
    print(f"[eval_mpos] MPOS={SPEC} ({'per-question table' if TABLE is not None else 'constant'})", flush=True)
    E.main()
    out = sys.argv[sys.argv.index("--output_path") + 1]
    json.dump({"mpos": SPEC, "n_rows": len(APPLIED), "offsets": APPLIED}, open(out + ".mpos.json", "w"))
    vals = sorted(APPLIED.values())
    if vals:
        print(f"[eval_mpos] {len(vals)} rows; offset min {vals[0]} p50 {vals[len(vals) // 2]} max {vals[-1]}")
    ref = os.environ.get("MPOS_REF")
    if ref:
        mine = {r["query_id"]: r["prediction"] for r in json.load(open(out))["records"]}
        theirs = {r["query_id"]: r["prediction"] for r in json.load(open(ref))["records"]}
        missing = [q for q in mine if q not in theirs]
        diff = [q for q in mine if q in theirs and mine[q] != theirs[q]]
        print(f"[eval_mpos] identity vs {ref}: {len(mine)} rows, {len(missing)} missing in ref, {len(diff)} differ")
        if missing or diff or not mine:
            print(f"[eval_mpos] IDENTITY FAILED, first differing: {(diff or missing)[:5]}")
            sys.exit(4)
        print("[eval_mpos] IDENTITY OK")


if __name__ == "__main__":
    main()
