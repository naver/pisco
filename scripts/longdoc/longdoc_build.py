"""Campaign longdoc, arm E: long-document eval sets (owner, 2026-11-17). Run through build_long.sbatch (CPU job).

Source: LongDocURL (dengchao/LongDocURL, Apache-2.0): 2,325 questions on 396 PDFs of 40-150 pages, evidence pages
annotated. Checked and not used (sizes / licences, 2026-11-17): QuALITY (CC BY 4.0, ~2k dev Q, 2k-8k-token articles:
cannot fill 16k / 32k), Qasper (CC BY 4.0, 1,005 val Q, 3k-10k-token papers: same), NarrativeQA (Apache-2.0, reaches
32k but no evidence location, so no nested windows; famous stories invite closed-book answers), LooGLE (CC BY-SA 4.0,
only >= 24k-word documents), MMLongBench-Doc (data CC BY-NC 4.0 research only, 1,091 Q on 135 PDFs; a second source
if LongDocURL runs short).

One question pool, nested windows of CONSECUTIVE pages of the question's own PDF that contain every evidence page
(evidence span <= 8 pages), so every question is asked at every length (paired length effect, like arm L):
  pages : page images, windows of 10 / 20 / 40 pages                       -> eval_data/longdoc_pages_1500
  text  : the PDF text layer, one doc per page (a page over PIECE tokens is split into consecutive pieces, so the
          compressor gets it whole at CML >= 1,024), windows grown to >= 4k / 8k / 16k / 32k tokens; questions with
          Text / Layout evidence only, every evidence page with a text layer  -> eval_data/longdoc_text_1500
  mixed : the pages windows; each page is its text layer or its image (p = .5, fixed per (doc, page), so the nested
          windows agree; a page without a text layer stays an image)    -> eval_data/longdoc_mixed_1500
  needle: a synthetic page "the access code of the <name> archive is <6 digits>" inserted at 10 / 50 / 90% depth of
          a real window, per modality and bucket, exact scoring        -> eval_data/longdoc_needle_{text,pages,mixed}
Rows are interleaved by question (q1 at every bucket, then q2 ...): --limit N keeps N / n_buckets whole questions.
The bucket is in `source` (longdocurl_p10 / _t4k / _m20, needle_<kind>_<bucket>_d<depth>) and the query_id suffix.
Schema = the existing eval sets (docs, doc_images, query, mistral_label, answers, gold_positions, source, query_id).
"""
import argparse, collections, hashlib, json, math, os, random, re, sys, tarfile, time, unicodedata
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ED = f"{ROOT}/eval_data"
SRC = f"{ED}/longdoc_src"
R = f"{ROOT}/expQ/ft_mm_4b_184778/model"
REPO = "dengchao/LongDocURL"
PX = 1003520
PIECE = 1000
PAGE_B = (10, 20, 40)
TOK_B = (4000, 8000, 16000, 32000)
DEPTHS = (0.1, 0.5, 0.9)
NAMES = ["Alder", "Birchwood", "Cedarholm", "Dunmore", "Elmstead", "Fairhaven", "Glenrock", "Hartwell", "Ironvale",
         "Juniper", "Kestrel", "Larkspur", "Marlowe", "Northgate", "Oakridge", "Pemberton", "Quarrington", "Redfern",
         "Stonebridge", "Thornbury", "Upton", "Valemont", "Westbrook", "Yarrowdale", "Zephyr"]
TOK = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def tok():
    if "t" not in TOK:
        from transformers import AutoTokenizer
        TOK["t"] = AutoTokenizer.from_pretrained(f"{R}/compressor")
    return TOK["t"]


def norm(s):
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def clean(t):
    t = (t or "").replace("\r\n", "\n").replace("\r", "\n")
    t = "".join(c for c in t if c == "\n" or unicodedata.category(c)[0] != "C")
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\n\s*\n+", "\n\n", t).strip()


# ---------------------------------------------------------------- source
def fetch(need_all):
    from huggingface_hub import hf_hub_download
    os.makedirs(f"{SRC}/pdf", exist_ok=True)
    meta = hf_hub_download(REPO, "LongDocURL_public.jsonl", repo_type="dataset", local_dir=f"{SRC}/hf")
    rows = [json.loads(l) for l in open(meta)]
    need = {str(r["doc_no"]) for r in rows} if need_all else set()
    have = lambda: {f[:-4] for f in os.listdir(f"{SRC}/pdf") if f.endswith(".pdf")}
    if need - have():
        tgz = hf_hub_download(REPO, "pdf_files.tar.gz", repo_type="dataset", local_dir=f"{SRC}/hf")
        got = have()
        log(f"extracting {len(need - got)} pdfs from {tgz}")
        with tarfile.open(tgz, "r|gz") as tf:
            for m in tf:
                b = os.path.basename(m.name)
                if m.isfile() and b.endswith(".pdf") and b[:-4] in need and b[:-4] not in got:
                    with tf.extractfile(m) as fi, open(f"{SRC}/pdf/{b}.part", "wb") as fo:
                        fo.write(fi.read())
                    os.replace(f"{SRC}/pdf/{b}.part", f"{SRC}/pdf/{b}")
    miss = need - have()
    log(f"meta {len(rows)} questions, {len(need)} docs, pdfs missing {len(miss)}")
    return rows, miss


def doc_text(doc):
    """Per-page text layer (cached): list of strings."""
    os.makedirs(f"{SRC}/text", exist_ok=True)
    p = f"{SRC}/text/{doc}.json"
    if os.path.exists(p):
        return json.load(open(p))
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(f"{SRC}/pdf/{doc}.pdf")
    out = []
    for i in range(len(pdf)):
        pg = pdf[i]
        try:
            tp = pg.get_textpage()
            out.append(clean(tp.get_text_range()))
            tp.close()
        except Exception:
            out.append("")
        pg.close()
    pdf.close()
    json.dump(out, open(p + ".part", "w"))
    os.replace(p + ".part", p)
    return out


def page_path(doc, i):
    return f"{SRC}/pages/{doc}/{i}.jpg"


def render(task):
    """Render the listed 0-based pages of one PDF to JPEG at <= PX pixels (cached)."""
    doc, pages = task
    todo = [i for i in sorted(pages) if not os.path.exists(page_path(doc, i))]
    if not todo:
        return doc, 0
    import pypdfium2 as pdfium
    os.makedirs(f"{SRC}/pages/{doc}", exist_ok=True)
    pdf = pdfium.PdfDocument(f"{SRC}/pdf/{doc}.pdf")
    for i in todo:
        pg = pdf[i]
        w, h = pg.get_size()
        s = min(6.0, math.sqrt(PX / max(1.0, w * h)) * 0.99)
        img = pg.render(scale=s).to_pil().convert("RGB")
        img.save(page_path(doc, i) + ".part.jpg", quality=90)
        os.replace(page_path(doc, i) + ".part.jpg", page_path(doc, i))
        pg.close()
    pdf.close()
    return doc, len(todo)


def page_tokens(task):
    doc = task
    tx = doc_text(doc)
    return doc, [len(tok()(t, add_special_tokens=False)["input_ids"]) if t else 0 for t in tx]


def pieces(t):
    """Split a text into consecutive pieces of <= PIECE tokens (char offsets of the tokenizer)."""
    enc = tok()(t, add_special_tokens=False, return_offsets_mapping=True)
    off = enc["offset_mapping"]
    if len(off) <= PIECE:
        return [t]
    out = []
    for k in range(0, len(off), PIECE):
        a = off[k][0] if k else 0
        b = off[min(k + PIECE, len(off)) - 1][1] if k + PIECE < len(off) else len(t)
        out.append(t[a:b].strip())
    return [p for p in out if p]


# ---------------------------------------------------------------- questions and windows
def answers(r):
    a = r.get("answer")
    s = ", ".join(str(x) for x in a) if isinstance(a, list) else ("" if a is None else str(a))
    return [s.strip()] if s.strip() else []


def page_windows(n, lo, hi, rng):
    """Nested windows {W: (s, e)} of consecutive pages containing [lo, hi] (0-based inclusive)."""
    cur, out = (lo, hi + 1), {}
    for W in PAGE_B:
        W2 = min(W, n)
        need = W2 - (cur[1] - cur[0])
        amin, amax = max(0, need - (n - cur[1])), min(need, cur[0])
        a = rng.randint(amin, amax)
        cur = (cur[0] - a, cur[1] + need - a)
        out[W] = cur
    return out


def token_windows(ntok, lo, hi, rng):
    """Nested windows {T: (s, e)} grown from [lo, hi] until >= T tokens; the left share of the added tokens follows a
    per-question fraction f ~ U(0, 1) (the evidence lands at a random depth). Buckets the document cannot reach are absent."""
    n, f = len(ntok), rng.random()
    s, e = lo, hi + 1
    tot, left, added, out = sum(ntok[s:e]), 0, 0, {}
    for T in TOK_B:
        while tot < T and (s > 0 or e < n):
            go_left = s > 0 and (e >= n or left < f * (added + 1))
            if go_left:
                s -= 1; tot += ntok[s]; left += ntok[s]; added += ntok[s]
            else:
                tot += ntok[e]; added += ntok[e]; e += 1
        if tot < T:
            break
        out[T] = (s, e)
    return out


def is_text_page(doc, i, txt):
    h = int(hashlib.md5(f"{doc}:{i}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < 0.5 and len(txt) >= 100


def img_entry(doc, i):
    return {"bytes": open(page_path(doc, i), "rb").read(), "path": None}


def text_docs(doc, idx, tx, gold_pages):
    docs, gold = [], []
    for i in idx:
        if len(tx[i]) < 20:
            continue
        for p in pieces(tx[i]):
            if i in gold_pages:
                gold.append(len(docs))
            docs.append(p)
    return docs, gold


def mixed_docs(doc, idx, tx, gold_pages):
    docs, imgs, gold = [], [], []
    for i in idx:
        if is_text_page(doc, i, tx[i]):
            parts = pieces(tx[i])
            for p in parts:
                if i in gold_pages:
                    gold.append(len(docs))
                docs.append(p); imgs.append(None)
        else:
            if i in gold_pages:
                gold.append(len(docs))
            docs.append(""); imgs.append(img_entry(doc, i))
    return docs, imgs, gold


def row(qid, src, query, ans, docs, imgs, gold):
    return dict(docs=docs, doc_images=imgs, query=query, mistral_label=ans[0], answers=ans,
                gold_positions=[int(g) for g in gold], source=src, query_id=qid)


# ---------------------------------------------------------------- needle
def needle_image(sentence, rng):
    from PIL import Image, ImageDraw, ImageFont
    W, H = 850, 1100
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=34)
        lines, cur = [], ""
        for w in sentence.split():
            if d.textlength(cur + " " + w, font=font) > W - 120 and cur:
                lines.append(cur); cur = w
            else:
                cur = (cur + " " + w).strip()
        lines.append(cur)
        y = rng.randint(120, H - 120 - 50 * len(lines))
        for k, l in enumerate(lines):
            d.text((60, y + 50 * k), l, fill="black", font=font)
    except Exception:   # no FreeType: bitmap font on a small canvas, upscaled
        small = Image.new("RGB", (W // 3, H // 3), "white")
        ds = ImageDraw.Draw(small)
        words, lines, cur = sentence.split(), [], ""
        for w in words:
            if len(cur) + len(w) > 40 and cur:
                lines.append(cur); cur = w
            else:
                cur = (cur + " " + w).strip()
        lines.append(cur)
        y = rng.randint(40, H // 3 - 40 - 14 * len(lines))
        for k, l in enumerate(lines):
            ds.text((20, y + 14 * k), l, fill="black")
        img = small.resize((W, H))
    import io
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=92)
    return {"bytes": b.getvalue(), "path": None}


# ---------------------------------------------------------------- main
FEATS = None


def features():
    import datasets as D
    return D.Features({"docs": D.Sequence(D.Value("string")), "doc_images": D.Sequence(D.Image()),
                       "query": D.Value("string"), "mistral_label": D.Value("string"),
                       "answers": D.Sequence(D.Value("string")), "gold_positions": D.Sequence(D.Value("int32")),
                       "source": D.Value("string"), "query_id": D.Value("string")})


def write(name, gen, stats):
    import datasets as D
    out = f"{ED}/{name}"
    if os.path.exists(f"{out}/stats.json"):
        log(f"{out} already built (stats.json present): kept, not rebuilt")
        return
    if os.path.exists(out):   # a partial write (preempted job): set aside, never deleted
        os.replace(out, f"{out}.partial_{time.strftime('%m%d%H%M')}")
    os.makedirs(f"{SRC}/cache", exist_ok=True)
    ds = D.Dataset.from_generator(gen, features=features(), cache_dir=f"{SRC}/cache", writer_batch_size=16)
    D.DatasetDict({"train": ds}).save_to_disk(out, max_shard_size="1GB")
    back = D.load_from_disk(out)["train"]
    r0 = back[0]
    assert len(r0["docs"]) == len(r0["doc_images"]) or not r0["doc_images"], (len(r0["docs"]), len(r0["doc_images"]))
    stats["rows"] = len(back)
    stats["by_source"] = dict(collections.Counter(back["source"]))
    json.dump(stats, open(f"{out}/stats.json", "w"), indent=1)
    log(f"wrote {out}: {len(back)} rows {stats['by_source']}")


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="all", help="comma list of pages,text,mixed,needle (or all)")
    ap.add_argument("--max_q", type=int, default=1500, help="questions per set (each asked at every bucket)")
    ap.add_argument("--per_doc", type=int, default=6, help="at most this many questions per PDF")
    ap.add_argument("--needle_docs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--suffix", default="", help="appended to every set name (smoke builds)")
    a = ap.parse_args(argv)
    kinds = {"pages", "text", "mixed", "needle"} if a.kinds == "all" else set(a.kinds.split(","))
    t0 = time.time()
    meta, miss = fetch(True)
    by_doc = collections.defaultdict(list)
    for r in meta:
        by_doc[str(r["doc_no"])].append(r)
    docs = sorted(d for d in by_doc if d not in miss)
    with Pool(a.workers) as pool:
        ntok = dict(pool.imap_unordered(page_tokens, docs, chunksize=4))
    texts = {d: doc_text(d) for d in docs}
    log(f"text layers + token counts for {len(docs)} docs ({time.time() - t0:.0f}s)")
    # evidence page convention: 1-based if the answer is found more often on page e-1 than on page e
    h1 = h0 = 0
    for d in docs:
        for r in by_doc[d]:
            if set(r.get("evidence_sources") or []) != {"Text"} or not answers(r):
                continue
            na = norm(answers(r)[0])
            for e in r["evidence_pages"]:
                h1 += 0 <= e - 1 < len(texts[d]) and na in norm(texts[d][e - 1])
                h0 += 0 <= e < len(texts[d]) and na in norm(texts[d][e])
    base = 1 if h1 >= h0 else 0
    log(f"evidence page convention: answer on page e-1: {h1}, on page e: {h0} -> {base}-based")
    rng = random.Random(a.seed)
    pool_q = []
    for d in docs:
        n = len(texts[d])
        qs = []
        for r in by_doc[d]:
            ev = [e - base for e in (r.get("evidence_pages") or [])]
            if not ev or not answers(r) or str(r.get("answer_format")) == "None":
                continue
            if min(ev) < 0 or max(ev) >= n or max(ev) - min(ev) + 1 > 8 or n < max(PAGE_B):
                continue
            qs.append((r, ev))
        rng.shuffle(qs)
        pool_q += [(d, r, ev) for r, ev in qs[: a.per_doc]]
    rng.shuffle(pool_q)
    pq = pool_q[: a.max_q]
    tq = [(d, r, ev) for d, r, ev in pool_q
          if set(r.get("evidence_sources") or []) <= {"Text", "Layout"} and all(len(texts[d][e]) >= 100 for e in ev)]
    log(f"usable: {len(pool_q)} questions on {len({x[0] for x in pool_q})} docs; pages/mixed take {len(pq)}; "
        f"text-evidence {len(tq)}")
    # windows (fixed per question, seeded by its id)
    pw = {r["question_id"]: page_windows(len(texts[d]), min(ev), max(ev), random.Random(f"{a.seed}:{r['question_id']}"))
          for d, r, ev in pq}
    tw = {r["question_id"]: token_windows(ntok[d], min(ev), max(ev), random.Random(f"{a.seed}:t:{r['question_id']}"))
          for d, r, ev in tq}
    tq = [x for x in tq if len(tw[x[1]["question_id"]]) == len(TOK_B)] + \
         [x for x in tq if len(tw[x[1]["question_id"]]) < len(TOK_B)]
    tq = tq[: a.max_q]
    # needle windows: docs not limited to the question pool
    nrng = random.Random(f"{a.seed}:needle")
    ndocs = [d for d in docs if len(texts[d]) >= max(PAGE_B) and sum(ntok[d]) >= max(TOK_B)]
    ndocs = nrng.sample(ndocs, min(a.needle_docs, len(ndocs)))
    # render every page any image set needs
    need = collections.defaultdict(set)
    if kinds & {"pages", "mixed", "needle"}:
        for d, r, ev in pq:
            s, e = pw[r["question_id"]][max(PAGE_B)]
            need[d].update(range(s, e))
        for d in ndocs:
            need[d].update(range(len(texts[d])))
        with Pool(a.workers) as pool:
            done = sum(k for _, k in pool.imap_unordered(render, list(need.items())))
        log(f"rendered {done} new pages ({sum(len(v) for v in need.values())} needed, {time.time() - t0:.0f}s)")
    sfx = a.suffix
    q_of = lambda r: r["question"].strip()

    if "pages" in kinds:
        def gen_pages():
            for d, r, ev in pq:
                for W, (s, e) in pw[r["question_id"]].items():
                    idx = list(range(s, e))
                    yield row(f"{r['question_id']}__p{W}", f"longdocurl_p{W}", q_of(r), answers(r),
                              [""] * len(idx), [img_entry(d, i) for i in idx], [idx.index(x) for x in sorted(set(ev))])
        write(f"longdoc_pages_1500{sfx}", gen_pages, dict(kind="pages", source=REPO, buckets_pages=PAGE_B, n_q=len(pq),
                                                         n_docs=len({x[0] for x in pq}), evidence_base=base, px=PX))
    if "mixed" in kinds:
        st = collections.defaultdict(list)

        def gen_mixed():
            for d, r, ev in pq:
                for W, (s, e) in pw[r["question_id"]].items():
                    dd, im, g = mixed_docs(d, list(range(s, e)), texts[d], set(ev))
                    st[W].append(sum(x is None for x in im) / max(1, len(im)))
                    yield row(f"{r['question_id']}__m{W}", f"longdocurl_m{W}", q_of(r), answers(r), dd, im, g)
        stats = dict(kind="mixed", source=REPO, buckets_pages=PAGE_B, n_q=len(pq), evidence_base=base, piece_tokens=PIECE)
        write(f"longdoc_mixed_1500{sfx}", gen_mixed, stats)
    if "text" in kinds:
        toks, ain = collections.defaultdict(list), collections.Counter()

        def gen_text():
            for d, r, ev in tq:
                na = norm(answers(r)[0])
                for T, (s, e) in tw[r["question_id"]].items():
                    dd, g = text_docs(d, list(range(s, e)), texts[d], set(ev))
                    toks[T].append(sum(ntok[d][s:e]))
                    ain[T] += na in norm(" ".join(dd))
                    yield row(f"{r['question_id']}__t{T // 1000}k", f"longdocurl_t{T // 1000}k", q_of(r), answers(r),
                              dd, [], g)
        stats = dict(kind="text", source=REPO, buckets_tokens=TOK_B, n_q=len(tq), evidence_base=base, piece_tokens=PIECE)
        write(f"longdoc_text_1500{sfx}", gen_text, stats)
        stats["tokens"] = {T: dict(n=len(v), p50=pct(v, .5), max=max(v)) for T, v in toks.items()}
        stats["answer_in_text"] = dict(ain)
        json.dump(stats, open(f"{ED}/longdoc_text_1500{sfx}/stats.json", "w"), indent=1)
    if "needle" in kinds:
        for mod in ("text", "pages", "mixed"):
            def gen_needle(mod=mod):
                k = 0
                for di, d in enumerate(ndocs):
                    r2 = random.Random(f"{a.seed}:needle:{mod}:{d}")
                    n = len(texts[d])
                    buckets = TOK_B if mod == "text" else PAGE_B
                    for B in buckets:
                        if mod == "text":
                            mid = r2.randrange(n)
                            win = token_windows(ntok[d], mid, mid, r2).get(B)
                            if win is None:
                                continue
                        else:
                            st0 = r2.randint(0, n - B)
                            win = (st0, st0 + B)
                        for dep in DEPTHS:
                            name, code = f"{NAMES[k % len(NAMES)]} {r2.randint(2, 99)}", f"{r2.randint(100000, 999999)}"
                            k += 1
                            sent = f"Note for the record: the access code of the {name} archive is {code}."
                            idx = list(range(*win))
                            if mod == "text":
                                dd, _ = text_docs(d, idx, texts[d], set())
                                im = []
                            elif mod == "pages":
                                dd, im = [""] * len(idx), [img_entry(d, i) for i in idx]
                            else:
                                dd, im, _ = mixed_docs(d, idx, texts[d], set())
                            j = int(round(dep * len(dd)))
                            as_text = mod == "text" or (mod == "mixed" and k % 2 == 0)
                            dd.insert(j, sent if as_text else "")
                            if mod != "text":
                                im.insert(j, None if as_text else needle_image(sent, r2))
                            tag = f"{B // 1000}k" if mod == "text" else f"p{B}"
                            yield row(f"needle_{mod}_{d}_{tag}_d{int(dep * 100)}", f"needle_{mod}_{tag}_d{int(dep * 100)}",
                                      f"What is the access code of the {name} archive?", [code], dd, im, [j])
            write(f"longdoc_needle_{mod}{sfx}", gen_needle, dict(kind=f"needle_{mod}", docs=ndocs, depths=DEPTHS))
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
