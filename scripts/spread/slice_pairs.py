"""Campaign spread, arm A: case audit + slice power over RECORDED judged eval files (stdlib only, no model).

For every pair of scripts/spread/audit_pairs.tsv (A = spread layout, B = tail layout, same questions):
  (1) re-derivation: A-B judge accuracy vs the recorded value (+-.003) for role=gate pairs;
  (2) no-op control: every file paired with itself must give 0.000 with 0 discordant pairs;
  (3) inventory: rows, judge on every row, unparseable, judge prompt, mems_used total, the job that wrote the file
      (`wrote <path>` lines of the eval logs), its node(s) (sacct), node image, git sha (expmon manifest);
  slices: source, question type, answer kind / length, mems_used and decoder prompt quartiles, n_images. Per slice: n,
      A, B, A-B, paired bootstrap 95% CI, McNemar b/c/p, both right / both wrong, uncompressed accuracy on the same
      rows, and the resolving power (2.8 x paired SE = the smallest difference this slice detects at 80% power).
      A slice with n < 300 or b + c < 40 is "not resolvable here" (CAMPAIGN.md gates).
The paired statistics are those of scripts/paired_judge.py (imported, never edited).

  python3 scripts/spread/slice_pairs.py --spec scripts/spread/audit_pairs.tsv --out expQ/spread/audit [--boot 2000]
"""
import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from paired_judge import bootstrap_ci, mcnemar, stats  # noqa: E402

TOL = 0.003
MIN_N, MIN_DISC = 300, 40
EL97 = {f"homer-{i:03d}" for i in range(5, 10)}
# node agreement on image inputs, campaign_probes/CAMPAIGN.md:70-95: homer-008 deviates; 005/006/009 untested
NODE_NOTE = {"homer-008": "IMAGE-OUTLIER", "homer-005": "untested", "homer-006": "untested(text drift suspect)",
             "homer-009": "untested"}


def load(path):
    d = json.load(open(os.path.join(ROOT, path)))
    return d.get("summary", {}), {r["query_id"]: r for r in d["records"]}


def fnum(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def qtype(q):
    w = (q or "").strip().lower().split()
    if not w:
        return "other"
    two = " ".join(w[:2])
    if two in ("how many", "how much"):
        return "how many/much"
    if w[0] in ("what", "which", "who", "when", "where", "why", "how"):
        return w[0]
    if w[0] in ("is", "are", "does", "do", "did", "can", "was", "were", "has", "have", "will", "should", "could"):
        return "yes/no"
    return "other"


NUM = re.compile(r"^[\s$€£¥]*[-+]?\d[\d,.]*\s*(%|percent|million|billion|thousand|k|m|bn)?\s*$", re.I)


def first_answer(a):
    if isinstance(a, list):
        return str(a[0]) if a else ""
    s = str(a)
    if s.startswith("["):
        try:
            import ast
            v = ast.literal_eval(s)
            return str(v[0]) if v else ""
        except Exception:
            pass
    return s


def answer_kind(a):
    s = first_answer(a)
    if NUM.match(s):
        return "numeric"
    n = len(s.split())
    return "text 1 word" if n <= 1 else ("text 2-3 words" if n <= 3 else "text 4+ words")


def quartile_labels(vals):
    """value -> 'Qk [lo,hi]' by the quartiles of vals (ties kept in one bin)."""
    s = sorted(vals)
    if not s:
        return lambda v: "?"
    cuts = [s[int(len(s) * q)] for q in (0.25, 0.5, 0.75)]
    edges = [s[0]] + cuts + [s[-1]]

    def lab(v):
        k = sum(1 for c in cuts if v > c)   # 0..3
        return f"Q{k + 1} ({edges[k]:g}-{edges[k + 1]:g}]" if k else f"Q1 [{edges[0]:g}-{edges[1]:g}]"
    return lab


def slice_row(pairs, base, boot, seed=0):
    a, b, m, se, n = stats(pairs)
    mc = mcnemar(pairs)
    bb, cc, p = mc if mc else (None, None, None)
    lo, hi = bootstrap_ci(pairs, boot, seed) if boot and n > 1 else (float("nan"), float("nan"))
    both1 = sum(1 for x, y in pairs if x == 1 and y == 1) / n
    both0 = sum(1 for x, y in pairs if x == 0 and y == 0) / n
    disc = (bb or 0) + (cc or 0)
    out = {"n": n, "A": a, "B": b, "diff": m, "se": se, "ci": [lo, hi], "b_A_only": bb, "c_B_only": cc, "p_mcnemar": p,
           "discordant": disc, "both_right": both1, "both_wrong": both0, "power_2p8se": 2.8 * se if se == se else None,
           "resolvable": n >= MIN_N and disc >= MIN_DISC}
    if base:
        out["base"] = sum(base) / len(base)
        out["base_n"] = len(base)
    return out


def writer_index():
    """relative result path -> [jobid, ...] from `wrote <path>` lines of every *_<jobid>.out log (repo root + expQ/**)."""
    idx = defaultdict(list)
    logs = glob.glob(os.path.join(ROOT, "*_*.out")) + glob.glob(os.path.join(ROOT, "expQ", "**", "*_*.out"), recursive=True)
    if not logs:
        return idx
    for chunk in range(0, len(logs), 500):
        r = subprocess.run(["grep", "-H", "-E", "^wrote .*\\.json"] + logs[chunk:chunk + 500], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            log, _, rest = line.partition(":")
            m = re.search(r"_(\d+)\.out$", log)
            path = rest.split(None, 1)[1].strip() if " " in rest else ""
            k = path.find("expQ/")
            if m and k >= 0:
                idx[path[k:]].append(m.group(1))
    return idx


def sacct_nodes(jobids):
    """jobid -> [(state, node, start, end), ...] every allocation attempt (-D), oldest first."""
    out = defaultdict(list)
    if not jobids:
        return out
    r = subprocess.run(["sacct", "-j", ",".join(sorted(set(jobids))), "-X", "-D", "-n", "-P", "-S", "2026-09-01",
                        "-o", "JobID,State,NodeList,Start,End"], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        f = line.split("|")
        if len(f) >= 5:
            out[f[0].split("_")[0]].append((f[1], f[2], f[3], f[4]))
    return out


def manifest_git():
    g = {}
    p = os.path.join(ROOT, ".expmon", "runs.jsonl")
    if os.path.exists(p):
        for line in open(p):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("jobid") and r.get("git"):
                g[str(r["jobid"])] = r["git"]
    return g


def inventory(path, summ, recs, widx, nodes, git):
    jobs = widx.get(path, [])
    # the file on disk was written by the LAST writer job (highest id that has a `wrote` line)
    job = max(jobs, key=int) if jobs else None
    att = nodes.get(job, []) if job else []
    node = att[-1][1] if att else "?"
    judged = sum(1 for r in recs.values() if "judge" in r)
    mems = sum(int(fnum(r.get("mems_used"), 0)) for r in recs.values())
    return {"path": path, "rows": len(recs), "judged": judged, "unparseable": summ.get("judge_unparseable"),
            "judge_prompt": summ.get("judge_prompt"), "judge": summ.get("judge"), "mems_used_total": mems,
            "writer_jobs": jobs, "writer": job, "attempts": [a[1] for a in att], "node": node,
            "image": "el9_7" if node in EL97 else ("el9_8" if node.startswith("homer-") else "?"),
            "node_note": NODE_NOTE.get(node, "agrees" if node.startswith("homer-") else "?"),
            "git": git.get(job, "?") if job else "?",
            "end": att[-1][3] if att else "?"}


EQ_BAND = 0.02


def pooled(cells, boot):
    """Seed-pooled paired test (owner REQUEST 2026-11-18, option A). cells = [(A_path, B_path), ...] one per seed.
    Per shared question: A-B averaged over seeds; 95% paired-bootstrap CI over questions. equal = that CI inside
    +-EQ_BAND AND every seed's own A-B inside +-EQ_BAND; better / worse stay per-seed (gates unchanged)."""
    recs = [(load(a)[1], load(b)[1]) for a, b in cells]
    ids = sorted(set.intersection(*(set(i for i in A if i in B and "judge" in A[i] and "judge" in B[i]) for A, B in recs)))
    per_seed = [stats([(fnum(A[i]["judge"]), fnum(B[i]["judge"])) for i in ids])[2] for A, B in recs]
    d = [(sum(fnum(A[i]["judge"]) - fnum(B[i]["judge"]) for A, B in recs) / len(recs), 0.0) for i in ids]
    _, _, m, se, n = stats(d)
    lo, hi = bootstrap_ci(d, boot)
    eq = -EQ_BAND < lo and hi < EQ_BAND and all(abs(x) < EQ_BAND for x in per_seed)
    return {"n": n, "per_seed": per_seed, "pooled": m, "se": se, "ci": [lo, hi], "width": hi - lo, "equal": eq}


def pooled_main(spec, out, boot):
    """spec TSV: case, expect (pooled A-B within +-TOL, '-' none), A_s42, B_s42, A_s43, B_s43; controls: expect, self-pair 0."""
    L = ["| case | n | per seed A-B ↑ | pooled A-B ↑ | 95% CI | width | equal (±2 pt) | control |", "|---|---|---|---|---|---|---|---|"]
    ok = True
    for line in open(os.path.join(ROOT, spec)):
        if line.startswith("#") or not line.strip():
            continue
        c, exp, a1, b1, a2, b2 = line.rstrip("\n").split("\t")[:6]
        r, z = pooled([(a1, b1), (a2, b2)], boot), pooled([(a1, a1), (a2, a2)], 200)
        ctl = z["pooled"] == 0.0 and (exp == "-" or abs(r["pooled"] - float(exp)) <= TOL)
        ok &= ctl
        L.append(f"| {c} | {r['n']} | {' / '.join(f'{x:+.3f}' for x in r['per_seed'])} | {r['pooled']:+.3f} | [{r['ci'][0]:+.3f}, "
                 f"{r['ci'][1]:+.3f}] | {r['width']:.3f} | {'EQUAL' if r['equal'] else 'no'} | self-pair {z['pooled']:.3f}"
                 f"{'' if exp == '-' else f', expect {float(exp):+.3f}'} {'ok' if ctl else 'FAIL'} |")
    os.makedirs(os.path.join(ROOT, out), exist_ok=True)
    open(os.path.join(ROOT, out, "pooled.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L) + f"\nRESULT pooled_controls={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooled", help="TSV of seed pairs: run ONLY the seed-pooled test (see pooled_main)")
    ap.add_argument("--spec", default="scripts/spread/audit_pairs.tsv")
    ap.add_argument("--out", default="expQ/spread/audit")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--slice_boot", type=int, default=1000)
    ap.add_argument("--no_inventory", action="store_true", help="skip log grep / sacct (fast local test)")
    args = ap.parse_args()
    if args.pooled:
        return pooled_main(args.pooled, args.out, args.boot)
    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)

    spec = []
    for line in open(os.path.join(ROOT, args.spec)):
        if line.startswith("#") or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        spec.append(dict(zip(["case", "role", "budget", "seed", "set", "expect", "A", "B", "base"], f)))

    files = sorted({s[k] for s in spec for k in ("A", "B", "base") if s[k] != "-"})
    data = {p: load(p) for p in files}
    widx = {} if args.no_inventory else writer_index()
    writers = [max(widx[p], key=int) for p in files if widx.get(p)]
    nodes = {} if args.no_inventory else sacct_nodes(writers)
    git = manifest_git()
    inv = {p: inventory(p, *data[p], widx, nodes, git) for p in files}

    report = {"inventory": inv, "cases": [], "selfpair": {}, "gate": {}}
    # (2) no-op control
    for p in files:
        recs = data[p][1]
        pairs = [(fnum(r["judge"]), fnum(r["judge"])) for r in recs.values() if "judge" in r]
        a, b, m, se, n = stats(pairs)
        mc = mcnemar(pairs)
        report["selfpair"][p] = {"n": n, "diff": m, "discordant": (mc[0] + mc[1]) if mc else None}
    ok2 = all(v["diff"] == 0.0 and (v["discordant"] in (0, None)) for v in report["selfpair"].values())

    gate1 = []
    for s in spec:
        (sa, A), (sb, B) = data[s["A"]], data[s["B"]]
        base = data[s["base"]][1] if s["base"] != "-" else {}
        ids = [i for i in A if i in B and "judge" in A[i] and "judge" in B[i]]
        if not ids:
            report["cases"].append({**s, "error": "no shared judged ids"})
            continue
        lab_m = quartile_labels([fnum(B[i].get("mems_used"), 0) for i in ids])
        lab_p = quartile_labels([fnum(B[i].get("decoder_prompt_tokens"), 0) for i in ids])
        keys = {
            "ALL": lambda i: "ALL",
            "source": lambda i: A[i].get("source", "?"),
            "question type": lambda i: qtype(A[i].get("query")),
            "answer kind": lambda i: answer_kind(A[i].get("answers")),
            "mems_used (tail file)": lambda i: lab_m(fnum(B[i].get("mems_used"), 0)),
            "decoder prompt tokens (tail)": lambda i: lab_p(fnum(B[i].get("decoder_prompt_tokens"), 0)),
            "n_images": lambda i: str(A[i].get("n_images", "?")),
        }
        case = {**s, "n_shared": len(ids), "n_A": len(A), "n_B": len(B),
                "mems_A": sum(int(fnum(A[i].get("mems_used"), 0)) for i in ids),
                "mems_B": sum(int(fnum(B[i].get("mems_used"), 0)) for i in ids), "slices": {}}
        for kname, kf in keys.items():
            groups, bgroups = defaultdict(list), defaultdict(list)
            for i in ids:
                g = kf(i)
                groups[g].append((fnum(A[i]["judge"]), fnum(B[i]["judge"])))
                if i in base and "judge" in base[i]:
                    bgroups[g].append(fnum(base[i]["judge"]))
            nb = args.boot if kname == "ALL" else args.slice_boot
            case["slices"][kname] = {g: slice_row(v, bgroups.get(g), nb) for g, v in sorted(groups.items())}
        allr = case["slices"]["ALL"]["ALL"]
        if s["role"] == "gate" and s["expect"] != "-":
            exp = float(s["expect"])
            ok = abs(allr["diff"] - exp) <= TOL
            gate1.append(ok)
            case["rederive"] = {"expect": exp, "got": allr["diff"], "ok": ok}
        report["cases"].append(case)

    ok1 = bool(gate1) and all(gate1)
    health = inv.get("expQ/ft_mm_4b_184778/eval/mm_docqa_pisco_px1003520_mm_docqa_1500_1m.json", {})
    okh = (health.get("rows") == 1500 and health.get("judged") == 1500 and health.get("unparseable") in (0, None)
           and health.get("mems_used_total") == 646042)
    report["gate"] = {"rederive_ok": ok1, "selfpair_ok": ok2, "health_184778_docs_ok": okh,
                      "feasibility": "PASS" if (ok1 and ok2 and okh) else "FAIL"}
    json.dump(report, open(os.path.join(ROOT, args.out, "audit.json"), "w"), indent=1)

    # ---------- human tables ----------
    L = []
    L.append("# Arm A — case audit of the recorded spread vs tail files (judge accuracy, 1 = correct; A = spread, B = tail)\n")
    L.append(f"FEASIBILITY GATE: {report['gate']['feasibility']}  (1) re-derivation {'PASS' if ok1 else 'FAIL'}  "
             f"(2) self-pair {'PASS' if ok2 else 'FAIL'}  health 184778 docs {'PASS' if okh else 'FAIL'}\n")
    L.append("## (1) Re-derivation of the recorded differences (tolerance +-.003)\n")
    L.append("| case | expected A-B | re-derived A-B ↑ | 95% CI | McNemar b/c, p | ok |\n|---|---|---|---|---|---|")
    for c in report["cases"]:
        if "rederive" in c:
            r = c["slices"]["ALL"]["ALL"]
            L.append(f"| {c['case']} | {c['rederive']['expect']:+.3f} | {r['diff']:+.3f} | [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] "
                     f"| {r['b_A_only']}/{r['c_B_only']}, {r['p_mcnemar']:.3g} | {'yes' if c['rederive']['ok'] else 'NO'} |")
    L.append("\n## (3) Inventory: every file, its health and where it ran\n")
    L.append("| file | rows | judged | unparseable | prompt | judge ↑ | mems_used total | writer job | node (attempts) | image | node check | git |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p, v in inv.items():
        jd = f"{v['judge']:.3f}" if isinstance(v["judge"], float) else "?"
        L.append(f"| {p.replace('expQ/', '')} | {v['rows']} | {v['judged']} | {v['unparseable']} | {v['judge_prompt']} | {jd} "
                 f"| {v['mems_used_total']:,} | {v['writer']} | {v['node']} ({','.join(v['attempts'])}) | {v['image']} | {v['node_note']} | {v['git']} |")
    L.append("\n## Cases and slices (A-B paired; power = 2.8 x paired SE = smallest difference the slice detects at 80% power)\n")
    for c in report["cases"]:
        if "error" in c:
            L.append(f"### {c['case']}: {c['error']}\n")
            continue
        ia, ib = inv[c["A"]], inv[c["B"]]
        same = ia["node"] == ib["node"] or ("IMAGE-OUTLIER" not in (ia["node_note"], ib["node_note"])
                                             and "untested" not in ia["node_note"] + ib["node_note"])
        L.append(f"### {c['case']} — role {c['role']}, budget {c['budget']}, seed {c['seed']}, set {c['set']}")
        L.append(f"A = {c['A']} ({ia['node']}, {ia['git']}) · B = {c['B']} ({ib['node']}, {ib['git']}) · shared rows "
                 f"{c['n_shared']} (A {c['n_A']}, B {c['n_B']}) · mems_used A {c['mems_A']:,} / B {c['mems_B']:,} "
                 f"(ratio {c['mems_A'] / max(c['mems_B'], 1):.4f}) · same-node rule: {'OK' if same else 'BLOCKED or untested'}\n")
        L.append("| slice | group | n | A ↑ | B ↑ | A-B ↑ | 95% CI | b/c | p McN | both right | both wrong | uncompressed ↑ | power (pt) | resolvable |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for kname, groups in c["slices"].items():
            for g, r in groups.items():
                base = f"{r['base']:.3f}" if "base" in r else "-"
                pw = f"{100 * r['power_2p8se']:.1f}" if r["power_2p8se"] is not None else "-"
                L.append(f"| {kname} | {g} | {r['n']} | {r['A']:.3f} | {r['B']:.3f} | {r['diff']:+.3f} | [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] "
                         f"| {r['b_A_only']}/{r['c_B_only']} | {r['p_mcnemar']:.3g} | {r['both_right']:.2f} | {r['both_wrong']:.2f} | {base} | {pw} "
                         f"| {'yes' if r['resolvable'] else 'not resolvable here'} |")
        L.append("")
    open(os.path.join(ROOT, args.out, "audit.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:4 + sum(1 for c in report["cases"] if "rederive" in c) + 3]))
    print(f"RESULT feasibility={report['gate']['feasibility']} rederive={int(ok1)} selfpair={int(ok2)} health={int(okh)} "
          f"cases={len(report['cases'])} files={len(files)}")
    print(f"wrote {os.path.join(args.out, 'audit.json')} and audit.md")
    return 0 if report["gate"]["feasibility"] == "PASS" else 3   # nonzero: jobs chained afterok on the gate never start


if __name__ == "__main__":
    sys.exit(main())
