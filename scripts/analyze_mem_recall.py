"""Join the transcription probe (scripts/probe_mem_recall.py) with judged QA evals: where is a doc answer lost?

Per question (first gold page): ans_ref = a gold answer appears in the reference transcription (pixels, uncompressed
backbone); ans_ae = it appears in the text rebuilt from the page's <MEM>s. Among questions the finetuned PISCO gets
wrong while the answer is in the reference:
  rebuilt     -> the vectors carry the answer, the QA decoder does not use it (training / RL lever)
  not rebuilt -> the answer did not survive compression (capacity / rate / budget lever)
Word and number recall compare the whole rebuilt page with the reference. Stdlib only.

  python3 scripts/analyze_mem_recall.py --ref expQ/archq/recall/ref.json --ae expQ/archq/recall/ae_s2.json \
      [--ae2 expQ/archq/recall/ae_ftcomp.json] --pisco <ft eval json> --base <base eval json>
"""
import argparse
import json
import re
from collections import Counter, defaultdict


def norm(s):
    return " ".join(re.sub(r"[^0-9a-z]+", " ", str(s).lower()).split())


def contains(text_n, answers):
    for a in answers:
        a = norm(a)
        if a and re.search(r"(?<![0-9a-z])" + re.escape(a) + r"(?![0-9a-z])", text_n):
            return True
    return False


def recall(ref_n, hyp_n, keep=lambda w: True):
    r = Counter(w for w in ref_n.split() if keep(w))
    h = Counter(w for w in hyp_n.split() if keep(w))
    tot = sum(r.values())
    return (sum(min(c, h[w]) for w, c in r.items()) / tot) if tot else None


def judged(path):
    d = json.load(open(path))
    return {r["query_id"]: r.get("judge") for r in d["records"]}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ae", required=True)
    ap.add_argument("--ae2", default=None, help="a second rebuild (e.g. the finetuned compressor), same pages")
    ap.add_argument("--pisco", required=True, help="judged eval json of the finetuned PISCO")
    ap.add_argument("--base", required=True, help="judged eval json of the uncompressed backbone")
    args = ap.parse_args()
    ref = {r["query_id"]: r for r in json.load(open(args.ref))["records"]}
    aes = [("ae", {r["query_id"]: r for r in json.load(open(args.ae))["records"]})]
    if args.ae2:
        aes.append(("ae2", {r["query_id"]: r for r in json.load(open(args.ae2))["records"]}))
    pj, bj = judged(args.pisco), judged(args.base)
    is_num = lambda w: any(c.isdigit() for c in w)  # noqa: E731

    for tag, ae in aes:
        rows = defaultdict(list)
        for q, r in ref.items():
            if q not in ae:
                continue
            ref_n, ae_n = norm(r["text"]), norm(ae[q]["text"])
            rows[r["source"]].append(dict(
                w=recall(ref_n, ae_n), num=recall(ref_n, ae_n, is_num),
                in_ref=contains(ref_n, r["answers"]), in_ae=contains(ae_n, r["answers"]),
                p=pj.get(q), b=bj.get(q), ref_words=len(ref_n.split()), ae_words=len(ae_n.split())))
        print(f"\n== {tag}: {args.ae if tag == 'ae' else args.ae2}")
        print("| source | n | ref words | rebuilt words | word recall | number recall | answer in ref | in rebuilt, given in ref "
              "| PISCO judge | base judge | PISCO wrong & ans in ref | of which rebuilt (use) | not rebuilt (lost) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for src in sorted(rows) + ["ALL"]:
            R = [x for s in rows for x in rows[s]] if src == "ALL" else rows[src]
            inref = [x for x in R if x["in_ref"]]
            wrong = [x for x in inref if x["p"] == 0]
            use = sum(x["in_ae"] for x in wrong)
            print(f"| {src} | {len(R)} | {mean([x['ref_words'] for x in R]):.0f} | {mean([x['ae_words'] for x in R]):.0f} "
                  f"| {mean([x['w'] for x in R]):.2f} | {mean([x['num'] for x in R]):.2f} "
                  f"| {len(inref) / max(len(R), 1):.2f} | {mean([float(x['in_ae']) for x in inref]):.2f} "
                  f"| {mean([x['p'] for x in R]):.3f} | {mean([x['b'] for x in R]):.3f} "
                  f"| {len(wrong)} | {use} | {len(wrong) - use} |")
        # does rebuilding predict QA success? (answer-in-ref questions only)
        inref = [x for s in rows for x in rows[s] if x["in_ref"] and x["p"] is not None]
        for v in (True, False):
            sub = [x["p"] for x in inref if x["in_ae"] == v]
            print(f"  PISCO judge when the answer is {'rebuilt' if v else 'NOT rebuilt'}: {mean(sub):.3f} (n={len(sub)})")


if __name__ == "__main__":
    main()
