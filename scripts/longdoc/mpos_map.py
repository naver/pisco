#!/usr/bin/env python3
"""Campaign longdoc, arm P: per-question position offsets N = (decoder position span of the LONG bucket's prompt)
- (span of the SHORT bucket's prompt), from the arm E input audit (job 191516, `rows.jsonl`, every row, CML 1,024).
Stdlib only (runs on the frontend in a second). Writes {short query_id: N} as JSON and prints min / p50 / max.

    python3 scripts/longdoc/mpos_map.py expQ/longdoc/audit_191516/rows.jsonl E_pages_cml1024 p10 p40 OUT.json
"""
import json
import statistics
import sys


def main():
    rows_path, variant, short, long_, out = sys.argv[1:6]
    span = {}
    for line in open(rows_path):
        r = json.loads(line)
        if r["v"] == variant:
            span[r["qid"]] = r["span"]
    offsets = {}
    for qid, s in span.items():
        if qid.endswith("__" + short):
            other = qid[: -len(short)] + long_
            if other not in span:
                sys.exit(f"no {long_} row for {qid}")
            offsets[qid] = span[other] - s
    if not offsets:
        sys.exit(f"no {variant} rows ending __{short}")
    vals = sorted(offsets.values())
    assert min(vals) > 0, "a long prompt with a span not above the short one"
    json.dump(offsets, open(out, "w"), indent=0, sort_keys=True)
    print(f"{len(vals)} questions; N min {vals[0]} p50 {statistics.median(vals):.0f} max {vals[-1]} -> {out}")


if __name__ == "__main__":
    main()
