"""Analyse the judge-reliability study: agreement between judge conditions, error rates on objective anchors,
and whether the campaign's conclusions depend on the judge. Stdlib only (login-node safe).

  python3 scripts/judge_agreement.py --pool expQ/judge_study/pool.json expQ/judge_study/*.json \
      [--sheet expQ/judge_study/human_sheet.md --sheet_n 150]

Anchors (no human needed):
  sure-yes : em == 1 (the prediction equals a gold answer after normalisation)   -> a judge must say yes
  sure-no  : gold is a number and the prediction holds numbers, none equal to any gold number -> must say no
Conclusions re-scored under every judge: per family, the models' mean verdict, so rank flips are visible.
--sheet writes a Markdown table of items where the judges disagree (stratified by family/source), with an empty
`human` column to fill in (1 correct / 0 wrong); scripts/judge_agreement.py --human <filled sheet> scores the judges.
"""
import argparse
import json
import random
import re
import string
from collections import defaultdict
from itertools import combinations


def norm(s):
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


NUM = re.compile(r"-?\d+(?:[.,]\d+)?")


def numbers(s):
    return {x.replace(",", "") for x in NUM.findall(str(s))}


def anchors(it):
    if it.get("em") == 1:
        return "sure_yes"
    golds = [str(a) for a in it["answers"]]
    if golds and all(NUM.fullmatch(g.strip().replace("%", "").replace("$", "").strip() or "x") for g in golds):
        gn = set().union(*(numbers(g) for g in golds))
        pn = numbers(it["prediction"])
        if pn and not (pn & gn):
            return "sure_no"
    return None


def kappa(a, b):
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan"), po


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("judges", nargs="*")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--sheet_n", type=int, default=150)
    ap.add_argument("--human", default=None, help="a filled human sheet (Markdown) to score the judges against")
    args = ap.parse_args()
    pool = {str(it["id"]): it for it in json.load(open(args.pool))}
    J = {}
    for p in args.judges:
        d = json.load(open(p))
        name = f"{d['model'].split('/')[-1]}|{d['prompt']}|{'think' if d['think'] else 'direct'}"
        J[name] = {k: v for k, v in d["verdicts"].items()}
        print(f"{name:40s} n={d['n']} unparseable={d['n_unparseable']} truncated={d.get('n_truncated', 0)}")
    stored = {k: it["judge_stored"] for k, it in pool.items() if it.get("judge_stored") is not None}
    J["stored Qwen3-8B (eval files)"] = stored
    names = list(J)
    ids = [k for k in pool if all(J[n].get(k) is not None for n in names)]
    print(f"\n{len(ids)} items judged by every condition (pool {len(pool)})")

    print("\n== pairwise agreement (upper: % agree, lower: kappa) ==")
    w = max(len(n) for n in names)
    print(" " * w + "".join(f"  J{i:<4d}" for i in range(len(names))))
    for i, a in enumerate(names):
        row = []
        for j, b in enumerate(names):
            if i == j:
                row.append("   -  ")
            else:
                k, po = kappa([J[a][x] for x in ids], [J[b][x] for x in ids])
                row.append(f" {po:5.3f}" if j > i else f" {k:5.3f}")
        print(f"{a:{w}s}" + "".join(row))
    for i, n in enumerate(names):
        print(f"  J{i} = {n}   yes rate {sum(J[n][x] for x in ids) / len(ids):.3f}")

    print("\n== objective anchors: error rate of each judge ==")
    anc = defaultdict(list)
    for x in ids:
        a = anchors(pool[x])
        if a:
            anc[a].append(x)
    print(f"  sure_yes items {len(anc['sure_yes'])}, sure_no items {len(anc['sure_no'])}")
    for n in names:
        fn = sum(1 for x in anc["sure_yes"] if J[n][x] == 0)
        fp = sum(1 for x in anc["sure_no"] if J[n][x] == 1)
        print(f"  {n:40s} says NO on sure-yes: {fn:4d} ({fn / max(1, len(anc['sure_yes'])):.1%})   "
              f"says YES on sure-no: {fp:4d} ({fp / max(1, len(anc['sure_no'])):.1%})")

    print("\n== conclusions under each judge: mean verdict per model (items of that model only) ==")
    fams = sorted({pool[x]["family"] for x in ids})
    for fam in fams:
        models = sorted({m for x in ids if pool[x]["family"] == fam for m in pool[x]["models"]})
        print(f"  [{fam}] " + " ".join(f"{m:>18s}" for m in models))
        for n in names:
            vals = []
            for m in models:
                xs = [x for x in ids if pool[x]["family"] == fam and m in pool[x]["models"]]
                vals.append(sum(J[n][x] for x in xs) / len(xs) if xs else float("nan"))
            print(f"    {n:38s} " + " ".join(f"{v:18.3f}" for v in vals))
    # per-family disagreement rate between the current judge and the others
    print("\n== disagreement with the stored judge, per family/source ==")
    cur = "stored Qwen3-8B (eval files)"
    for n in names:
        if n == cur:
            continue
        per = defaultdict(lambda: [0, 0])
        for x in ids:
            k = (pool[x]["family"], pool[x]["source"])
            per[k][1] += 1
            per[k][0] += int(J[n][x] != J[cur][x])
        print(f"  {n:40s} " + "  ".join(f"{k[1]}:{a / b:.0%}" for k, (a, b) in sorted(per.items())))

    if args.human:
        rows = [l for l in open(args.human) if l.startswith("|") and not l.startswith("|--") and not l.startswith("| id")]
        hum = {}
        for l in rows:
            cells = [c.strip() for c in l.strip().strip("|").split("|")]
            if cells and cells[-1] in ("0", "1"):
                hum[cells[0]] = int(cells[-1])
        print(f"\n== accuracy against {len(hum)} human labels ==")
        for n in names:
            xs = [x for x in hum if x in J[n] and J[n][x] is not None]
            acc = sum(J[n][x] == hum[x] for x in xs) / max(1, len(xs))
            print(f"  {n:40s} {acc:.3f} (n={len(xs)})")

    if args.sheet:
        dis = [x for x in ids if len({J[n][x] for n in names}) > 1]
        random.seed(0)
        random.shuffle(dis)
        by = defaultdict(list)
        for x in dis:
            by[(pool[x]["family"], pool[x]["source"])].append(x)
        per = max(1, args.sheet_n // max(1, len(by)))
        pick = [x for k in sorted(by) for x in by[k][:per]]
        with open(args.sheet, "w") as f:
            f.write(f"# Judge disagreement sheet ({len(pick)} of {len(dis)} disagreeing items; fill `human`: 1 correct, 0 wrong)\n\n")
            f.write("Judges: " + "; ".join(f"J{i}={n}" for i, n in enumerate(names)) + "\n\n")
            f.write("| id | family/source | question | gold | prediction | " + " ".join(f"J{i}" for i in range(len(names))) + " | human |\n")
            f.write("|---|---|---|---|---|" + "---|" * 1 + "---|\n")
            for x in pick:
                it = pool[x]
                q = it["query"].replace("|", "/").replace("\n", " ")[:160]
                g = " / ".join(str(a) for a in it["answers"])[:80].replace("|", "/")
                p = it["prediction"].replace("|", "/").replace("\n", " ")[:160]
                js = " ".join(str(J[n][x]) for n in names)
                f.write(f"| {x} | {it['family']}/{it['source']} | {q} | {g} | {p} | {js} |  |\n")
        print(f"\nwrote {args.sheet}: {len(pick)} items to label by hand (of {len(dis)} disagreements)")


if __name__ == "__main__":
    main()
