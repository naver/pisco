"""Answer-level identity of two eval outputs of the SAME model on the SAME set (a re-run check).
Stdlib only, small JSON (login-node safe). Prints one markdown row per pair:
  label | n common | answers differ | judge A | judge B | judge flips (A-only right / B-only right)
  python3 scripts/spread/compare_answers.py LABEL A.json B.json [LABEL A.json B.json ...]
"""
import json
import sys


def records(path):
    return {r["query_id"]: r for r in json.load(open(path))["records"]}


def row(label, pa, pb):
    a, b = records(pa), records(pb)
    ids = sorted(set(a) & set(b))
    n = len(ids)
    diff = sum(1 for i in ids if a[i]["prediction"].strip() != b[i]["prediction"].strip())
    ja = sum(a[i]["judge"] for i in ids) / n
    jb = sum(b[i]["judge"] for i in ids) / n
    a_only = sum(1 for i in ids if a[i]["judge"] == 1 and b[i]["judge"] == 0)
    b_only = sum(1 for i in ids if a[i]["judge"] == 0 and b[i]["judge"] == 1)
    return f"| {label} | {n} | {diff} | {ja:.3f} | {jb:.3f} | {a_only + b_only} ({a_only}/{b_only}) | `{pa}` | `{pb}` |"


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) % 3:
        sys.exit("usage: LABEL A.json B.json [LABEL A.json B.json ...]")
    print("| pair | n | answers differ | judge A | judge B | judge flips (A-only/B-only) | A | B |")
    print("|---|---|---|---|---|---|---|---|")
    for k in range(0, len(args), 3):
        print(row(*args[k:k + 3]))
