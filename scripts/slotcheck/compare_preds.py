"""Campaign slotcheck: per-question comparison of two eval_mm_docqa result files (stdlib only).

Prints rows, identical predictions, mems_used totals (and the ratio), judge coverage / unparseable / mean when present.
--require_identical: exit 4 unless every shared query_id has the same prediction AND the same mems_used AND both files
hold the same query_ids (the wrapper invariant of CAMPAIGN.md "## Patchable").
  python3 scripts/slotcheck/compare_preds.py A.json B.json [--names A B] [--require_identical]
"""
import argparse, json, sys


def load(p):
    d = json.load(open(p))
    return d.get("summary", {}), {r["query_id"]: r for r in d["records"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    ap.add_argument("--require_identical", action="store_true")
    a = ap.parse_args()
    (sa, ra), (sb, rb) = load(a.a), load(a.b)
    common = [q for q in ra if q in rb]
    same_pred = sum(ra[q]["prediction"] == rb[q]["prediction"] for q in common)
    same_mems = sum(ra[q].get("mems_used") == rb[q].get("mems_used") for q in common)
    ma = sum(ra[q].get("mems_used", 0) for q in common); mb = sum(rb[q].get("mems_used", 0) for q in common)
    print(f"[compare] {a.names[0]} rows {len(ra)} | {a.names[1]} rows {len(rb)} | shared {len(common)}")
    print(f"[compare] identical predictions {same_pred} / {len(common)}; identical mems_used {same_mems} / {len(common)}; "
          f"mems_used {ma} vs {mb} (ratio {mb / max(ma, 1):.4f})")
    for n, s, r in ((a.names[0], sa, ra), (a.names[1], sb, rb)):
        js = [x["judge"] for x in r.values() if x.get("judge") is not None]
        if js:
            print(f"[compare] {n}: judge on {len(js)} / {len(r)} rows, unparseable {s.get('judge_unparseable')}, "
                  f"judge mean {sum(js) / len(js):.4f}, em {s.get('em')}")
    ok = same_pred == len(common) and same_mems == len(common) and set(ra) == set(rb)
    if a.require_identical:
        print(f"INVARIANT {'PASS' if ok else 'FAIL'}: {a.names[0]} vs {a.names[1]}", flush=True)
        return 0 if ok else 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
