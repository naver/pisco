#!/usr/bin/env python3
"""Health of arm E result files (run with the pixi python, `-I`; a few seconds per file on the frontend).

Per file: rows, rows judged, judge_unparseable, judge, answers at the cap (Qwen3.5 tokenizer, padding and
truncation OFF, `len(ids) >= cap - 2`), prompt / decoder-prompt max, `mems_used` total (pisco files), and the
judge per source bucket. Compare `mems_used` with the audit (expQ/longdoc/audit_191516/rows.jsonl) by hand.

usage: .pixi/envs/default/bin/python -I scripts/longdoc/cell_health.py [--cap 512] [--json OUT] FILE...
"""
import argparse
import collections
import json

import tokenizers

TOK = "expQ/ft_mm_4b_184778/model/tokenizer.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--json")
    a = ap.parse_args()
    t = tokenizers.Tokenizer.from_file(TOK)
    t.no_padding()
    t.no_truncation()
    out = {}
    for f in a.files:
        d = json.load(open(f))
        r, sm = d["records"], d["summary"]
        cap = sum(1 for x in r if len(t.encode(x["prediction"]).ids) >= a.cap - 2)
        by = collections.defaultdict(list)
        for x in r:
            by[x["source"]].append(x.get("judge", 0) or 0)
        pk = "decoder_prompt_tokens" if "decoder_prompt_tokens" in r[0] else "prompt_tokens"
        res = dict(
            n=len(r),
            judged=sum(1 for x in r if x.get("judge") in (0, 1)),
            unparseable=sm.get("judge_unparseable"),
            judge=round(sm.get("judge", float("nan")), 4),
            at_cap=cap,
            at_cap_pct=round(100 * cap / len(r), 2),
            prompt_max=max(x.get(pk, 0) for x in r),
            mems_used=sum(x.get("mems_used", 0) for x in r) if "mems_used" in r[0] else None,
            buckets={k: (len(v), round(sum(v) / len(v), 4)) for k, v in sorted(by.items())},
        )
        out[f] = res
        print(f.split("/")[-1], json.dumps(res))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
