"""OpenEQA EM-EQA (HM3D, 557 Q) results: import the silver-nav campaign_eqa cells so they can be judged with the same
judge as ours, and tabulate LLM-Match per arm, overall and per category, paired against a reference arm. Stdlib only.

  import   campaign cell jsons (fmnav/eqa/em_answer.py: rows[{question_id, prediction}]) -> eval_mm_docqa.py-format
           jsons (records[{query_id, source, query, answers, prediction}]) under --out, ready for
           `scripts/judge_mm_docqa.py --prompt openeqa --judge_model Qwen/Qwen3-14B <files>`.
  table    judged jsons -> LLM-Match (= 100 x mean judge), SR (share of marks > 3, the Sequential-EQA rule), per
           category; with --ref: paired difference per question, 95% paired bootstrap CI and the exact sign test on
           the non-zero differences (the campaign's test), on the questions both files hold.

  python scripts/openeqa_results.py import /beegfs/scratch/user/hdejean/silver-nav/results/eqa/em_uniform_k15_327340.json ... --out expQ/openeqa/campaign
  python scripts/openeqa_results.py table expQ/openeqa/*.json --ref expQ/base_qwen35_4b/eval/mm_docqa_base_k16_openeqa_hm3d_k16.json
"""

import argparse
import json
import math
import os
import random

DATASET = "/beegfs/scratch/user/hdejean/open-eqa/data/open-eqa-v0.json"   # as scripts/build_mm_eval_openeqa.py


def load_questions(path=DATASET):
    return [d for d in json.load(open(path)) if str(d.get("episode_history", "")).startswith("hm3d-v0/")]


def do_import(args):
    by_id = {q["question_id"]: q for q in load_questions(args.dataset)}
    os.makedirs(args.out, exist_ok=True)
    for src in args.files:
        cell = json.load(open(src))
        recs = []
        for r in cell["rows"]:
            q = by_id[r["question_id"]]
            recs.append({"query_id": q["question_id"], "source": q["category"], "query": q["question"],
                         "answers": [q["answer"]] + list(q.get("extra_answers") or []),
                         "prediction": r.get("prediction", ""), "n_frames_shown": r.get("n_frames_shown")})
        summary = {"n": len(recs), "mode": "campaign_eqa", "arm": cell.get("arm"), "k": cell.get("k"),
                   "answerer": cell.get("answerer"), "campaign_file": src, "regime": cell.get("regime")}
        out = os.path.join(args.out, os.path.basename(src))
        json.dump({"summary": summary, "records": recs}, open(out, "w"), indent=1, ensure_ascii=False)
        print(f"{out}: {len(recs)} rows, arm {cell.get('arm')} k {cell.get('k')} answerer {cell.get('answerer')}")


def sign_test(deltas):
    """Exact two-sided sign test on the non-zero paired differences: (wins, losses, p)."""
    w = sum(1 for d in deltas if d > 0)
    l_ = sum(1 for d in deltas if d < 0)
    n = w + l_
    if n == 0:
        return w, l_, 1.0
    k = min(w, l_)
    return w, l_, min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot_ci(deltas, boot=2000, seed=0):
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(boot))
    return means[int(0.025 * boot)], means[int(0.975 * boot) - 1]


def judged(path):
    d = json.load(open(path))
    recs = {r["query_id"]: r for r in d["records"] if "judge" in r}
    return d["summary"], recs


def do_table(args):
    ref = judged(args.ref)[1] if args.ref else None
    cats = sorted({r["source"] for f in args.files for r in judged(f)[1].values()})
    short = {c: "".join(w[0] for w in c.split()) for c in cats}
    print("LLM-Match = 100 x mean((mark-1)/4); SR = % of marks > 3. Judge: see each file's summary.judge_model.")
    print("categories: " + ", ".join(f"{short[c]} = {c}" for c in cats))
    head = f"{'file':60s} {'n':>4s} {'LLM-M':>6s} {'SR':>5s} " + " ".join(f"{short[c]:>5s}" for c in cats)
    if ref:
        head += f"  {'vs ref':>7s} {'95% CI':>16s} {'sign test (w/l, p)':>20s}"
    print(head)
    for f in args.files:
        s, recs = judged(f)
        if not recs:
            print(f"{os.path.basename(f):60s} not judged"); continue
        v = [r["judge"] for r in recs.values()]
        lm = 100 * sum(v) / len(v)
        sr = 100 * sum(1 for x in v if x > 0.5 + 1e-9) / len(v)
        per = {}
        for c in cats:
            vc = [r["judge"] for r in recs.values() if r["source"] == c]
            per[c] = 100 * sum(vc) / len(vc) if vc else float("nan")
        line = f"{os.path.basename(f)[:60]:60s} {len(v):4d} {lm:6.1f} {sr:5.1f} " + " ".join(f"{per[c]:5.1f}" for c in cats)
        if ref:
            ids = sorted(set(recs) & set(ref))
            d = [100 * (recs[i]["judge"] - ref[i]["judge"]) for i in ids]
            if d:
                lo, hi = boot_ci(d, args.boot)
                w, l_, p = sign_test(d)
                line += f"  {sum(d) / len(d):+7.2f} [{lo:+6.2f}, {hi:+6.2f}] {w:4d}/{l_:<4d} p={p:.3f} (n {len(d)})"
        print(line + f"   judge={s.get('judge_model', '?')}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("import"); a.add_argument("files", nargs="+"); a.add_argument("--out", required=True)
    a.add_argument("--dataset", default=DATASET)
    t = sub.add_parser("table"); t.add_argument("files", nargs="+"); t.add_argument("--ref", default=None)
    t.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    (do_import if args.cmd == "import" else do_table)(args)


if __name__ == "__main__":
    main()
