"""Build the judge-reliability pool: (question, gold answers, prediction) items taken from existing eval outputs
(scripts/eval_mm_docqa.py JSON) of several models per family, strong and weak, deduplicated by (question, prediction).
Stdlib only (small JSON: safe on the login node).

  python3 scripts/build_judge_pool.py --out expQ/judge_study/pool.json

Each item: id, family, source, query_id, query, answers, prediction, models (which eval files gave it), em, anls,
match, judge_stored (the Qwen3-8B verdict recorded in the eval file, when all files agree).
"""
import argparse
import json
import os
import re
import string
from collections import defaultdict

E = "/beegfs/scratch/user/hdejean/pisco/expQ/"
FAMILIES = {
    "doc": [
        ("base_1m", E + "base_qwen35_4b/eval/mm_docqa_base_px1003520_mm_docqa_500_1m.json"),
        ("pisco_173550", E + "ft_mm_4b_173550/eval/mm_docqa_pisco_px1003520_mm_docqa_500_1m.json"),
        ("pisco_173703", E + "ft_mm_4b_173703/eval/mm_docqa_pisco_px1003520_mm_docqa_500_1m.json"),
        ("pisco_171654", E + "ft_mm_4b_171654/eval/mm_docqa_pisco.json"),
    ],
    "natural": [
        ("base", E + "base_qwen35_4b/eval/mm_docqa_base_mm_naturalqa_500.json"),
        ("pisco_173550", E + "ft_mm_4b_173550/eval/mm_docqa_pisco_mm_naturalqa_500.json"),
        ("pisco_173539", E + "ft_mm_4b_173539/eval/mm_docqa_pisco_mm_naturalqa_500.json"),
        ("pisco_172671", E + "ft_mm_4b_172671/eval/mm_docqa_pisco_mm_naturalqa_500.json"),
    ],
    "nav": [
        ("base_ft", E + "ft_nav_base_4b/eval/mm_docqa_base_k8_nav_ft_val_v2.json"),
        ("base_zeroshot", E + "base_qwen35_4b/eval/mm_docqa_base_k8_nav_ft_val_v2.json"),
        ("pisco_nav", E + "ft_nav_4b/eval/mm_docqa_pisco_r16_k8_nav_ft_val_v2.json"),
        ("pisco_nav_from_doc", E + "ft_nav_from_doc_4b/eval/mm_docqa_pisco_r16_k8_nav_ft_val_v2.json"),
    ],
}


def norm(s):
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    items = {}
    for fam, files in FAMILIES.items():
        for tag, path in files:
            if not os.path.exists(path):
                print(f"!! missing {path}")
                continue
            recs = json.load(open(path))["records"]
            for r in recs:
                key = (fam, str(r["query_id"]), norm(r["prediction"]))
                if key not in items:
                    items[key] = {"family": fam, "source": r.get("source"), "query_id": str(r["query_id"]),
                                  "query": r["query"], "answers": r["answers"], "prediction": r["prediction"],
                                  "models": [], "em": r.get("em"), "anls": r.get("anls"), "match": r.get("match"),
                                  "judge_stored": set()}
                it = items[key]
                it["models"].append(tag)
                if "judge" in r:
                    it["judge_stored"].add(int(r["judge"]))
    out = []
    for i, it in enumerate(items.values()):
        js = it.pop("judge_stored")
        it["judge_stored"] = js.pop() if len(js) == 1 else None
        it["id"] = i
        out.append(it)
    per = defaultdict(int)
    for it in out:
        per[(it["family"], it["source"])] += 1
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=0, ensure_ascii=False)
    print(f"{len(out)} items -> {args.out}")
    for k in sorted(per):
        print(f"  {k[0]:8s} {k[1]:14s} {per[k]}")
    n_num = sum(1 for it in out if re.fullmatch(r"[\d.,%$ ]+", str(it["answers"][0]).strip() or "x"))
    print(f"  numeric-gold items: {n_num}; em=1 items: {sum(1 for it in out if it['em'] == 1)}")


if __name__ == "__main__":
    main()
