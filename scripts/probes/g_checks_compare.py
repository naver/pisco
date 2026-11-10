"""Arm G checks after a --dense_rate patch (run by g_checks.sbatch, stdlib only). Pass = the law's criteria:
plain 300-Q regression EM within .01 of the recorded file (184781's first 300); RAW=route1 DENSE=8 control >= 97% of
answers identical to it, judge within .01, same slot total. Exit 1 on any failure."""
import json
import sys

P = "expQ/probes"
REC = "expQ/ft_mm_4b_184778/eval/mm_docqa_pisco_px1003520_mm_docqa_1500_1m.json"
FILES = {
    "regression": f"{P}/reg_g_plain2/eval/mm_docqa_pisco_px1003520_mm_docqa_1500_1m_lim300.json",
    "control": f"{P}/g_dense8_ctrl2/eval/mm_docqa_pisco_px1003520_rawroute1_dense8_mm_docqa_1500_1m_lim300.json",
    "pre-patch control (188525)": f"{P}/g_dense8_ctrl/eval/attempt_h016_0644_mm_docqa_pisco_px1003520_rawroute1_dense8_mm_docqa_1500_1m_lim300.json",
}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


rec = {r["query_id"]: r for r in json.load(open(REC))["records"]}
ok = True
for name, f in FILES.items():
    try:
        rows = json.load(open(f))["records"]
    except FileNotFoundError:
        print(f"{name}: MISSING {f}")
        ok = ok and name.startswith("pre-patch")
        continue
    base = [rec[r["query_id"]] for r in rows]
    same = sum(r["prediction"] == b["prediction"] for r, b in zip(rows, base))
    em, em0 = mean([r["em"] for r in rows]), mean([b["em"] for b in base])
    j, j0 = mean([r.get("judge") for r in rows]), mean([b.get("judge") for b in base])
    m, m0 = sum(r.get("mems_used", 0) for r in rows), sum(b.get("mems_used", 0) for b in base)
    print(f"{name}: n={len(rows)} identical {same}/{len(rows)}  EM {em:.4f} vs {em0:.4f}  judge {j:.4f} vs {j0:.4f}  "
          f"slots {m} vs {m0}")
    if name == "regression":
        ok = ok and len(rows) == 300 and abs(em - em0) <= 0.01
    elif name == "control":
        ok = ok and len(rows) == 300 and same >= 0.97 * len(rows) and abs(j - j0) <= 0.01 and m == m0
print("G CHECKS", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
