"""Arm A (campaign longdoc): what the compressor and the decoder actually receive, per eval variant and per training
stage. CPU only, no model weights: R's tokenizers + image processor and the eval's / trainer's own row builders.

Eval variants (pisco + base mode): <MEM>s per question, compressor tokens per chunk, decoder prompt tokens, decoder
position span (stride-aware), cut rate (gold answer in the raw text but not in what the compressor gets; or a gold
page dropped by TOPK), parity (pisco and base get the same documents in the same order), out-of-range share vs the
finetune sample. Training stages: the collator of each stage run on sampled rows (same kwargs as training_config.yaml).
Health: R's plain 4-page document cell must predict 646,042 <MEM>s (campaign_probes/STATE.md:42).

  python scripts/longdoc/audit_inputs.py --out expQ/longdoc/audit_<job> [--limit 20] [--workers 16]
"""
import argparse, hashlib, json, os, random, sys, time, traceback
from collections import defaultdict
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
R = f"{ROOT}/expQ/ft_mm_4b_184778/model"
ED = f"{ROOT}/eval_data"
LC = f"{ROOT}/expQ/longchunk"
PX = 1003520
REF = {  # recorded files with mems_used per row (health / cross-check; mems depend on tokenizer, CML, rate, PX only)
    "docs_p4": f"{ROOT}/expQ/ft_mm_4b_184778/eval/mm_docqa_pisco_px1003520_mm_docqa_1500_1m.json",
    "claim_g10_cml4096": f"{ROOT}/expQ/ft_mm_4b_183538/eval/mm_docqa_pisco_k10_cml4096_text_hotpotqa_1500_g10.json",
    "claim_mrqa_cml4096": f"{ROOT}/expQ/ft_mm_4b_183538/eval/mm_docqa_pisco_px1003520_cml4096_text_mrqa_ood_1500.json",
    "claim_longdoc_k20": f"{ROOT}/expQ/ft_mm_4b_183538/eval/mm_docqa_pisco_k20_px1003520_mm_docqa_longdoc_1m.json",
}
# name: (set, TOPK, XD, CML, GOLD, gating)
VARIANTS = {
    "docs_p4": ("mm_docqa_1500_1m", 4, 0, 256, False, True),
    "docs_p10": ("mm_docqa_1500_1m", 10, 6, 256, False, True),
    "docs_p20": ("mm_docqa_1500_1m", 20, 16, 256, False, True),
    "docs_p4_gold": ("mm_docqa_1500_1m", 4, 0, 256, True, True),
    "text_n10": ("text_hotpotqa_1500", 10, 0, 4096, False, True),
    "text_n20": ("text_hotpotqa_1500", 20, 10, 4096, False, True),
    "text_n40": ("text_hotpotqa_1500", 40, 30, 4096, False, True),
    "text_n10_gold": ("text_hotpotqa_1500", 10, 0, 4096, True, True),
    "text_n10_cml256": ("text_hotpotqa_1500", 10, 0, 256, False, False),
    "claim_g10_cml4096": ("text_hotpotqa_1500_g10", 10, 0, 4096, False, False),
    "claim_g10_cml256": ("text_hotpotqa_1500_g10", 10, 0, 256, False, False),
    "claim_mrqa_cml4096": ("text_mrqa_ood_1500", 4, 0, 4096, False, False),
    "claim_mrqa_cml256": ("text_mrqa_ood_1500", 4, 0, 256, False, False),
    "claim_longdoc_k20": ("mm_docqa_longdoc_1m", 20, 0, 256, False, False),
}
G = {}


def _setup():
    if G:
        return
    import torch
    from pisco.model import PISCO, PISCOConfig, mem_spread_position_ids
    from pisco.mm_utils import load_image_processor
    cfg = PISCOConfig.from_pretrained(R)
    dec_name = cfg.decoder_model_name
    cfg.compressor_model_name = f"{R}/compressor"
    cfg.decoder_model_name = R
    G["ctok"] = PISCO.create_compressor_tokenizer(None, cfg)
    G["dtok"] = PISCO.create_decoder_tokenizer(None, cfg)
    G["ip"] = load_image_processor(f"{R}/compressor", min_pixels=cfg.image_min_pixels, max_pixels=PX)
    G["bip"] = load_image_processor(dec_name, max_pixels=PX)    # base mode: eval builds it from the decoder name
    G["same_ip"] = (getattr(G["ip"], "min_pixels", None), getattr(G["ip"], "max_pixels", None)) == \
                   (getattr(G["bip"], "min_pixels", None), getattr(G["bip"], "max_pixels", None))
    G["cfg"], G["torch"], G["spread"] = cfg, torch, mem_spread_position_ids
    G["stride"] = int(getattr(cfg, "mem_position_stride", 1) or 1)
    G["ds"], G["colls"], G["by_source"] = {}, {}, {}


def _ds(name):
    if name not in G["ds"]:
        import datasets
        d = datasets.load_from_disk(f"{ED}/{name}")
        G["ds"][name] = d["train"] if isinstance(d, datasets.DatasetDict) else d
    return G["ds"][name]


def _coll(topk, cml):
    if (topk, cml) not in G["colls"]:
        from pisco.mm_collator import MultimodalFineTuningCollator
        G["colls"][(topk, cml)] = MultimodalFineTuningCollator(
            compressor_tokenizer=G["ctok"], decoder_tokenizer=G["dtok"], compr_rate=G["cfg"].compr_rate,
            compressor_max_length=cml, image_processor=G["ip"], image_compr_rate=None, topk_docs=topk,
            query_dependent=False, pad_to_multiple_of=None, chunk_docs=False)
    return G["colls"][(topk, cml)]


def _span(ids):
    torch = G["torch"]
    t = torch.tensor([ids]); m = torch.ones_like(t)
    if G["stride"] <= 1:
        return len(ids)
    return int(G["spread"](t, m, G["dtok"].mem_token_id, G["stride"]).max()) + 1


def _doc_key(d, img):
    if img is not None:
        try:
            return "img:" + hashlib.md5(img.tobytes()[:65536] + str(img.size).encode()).hexdigest()[:16]
        except Exception:
            return f"img:{id(img)}"
    return "txt:" + hashlib.md5((d or "").encode()).hexdigest()[:16]


def eval_row(task):
    vname, i = task
    _setup()
    from eval_mm_docqa import _build_chat_prompt, with_distractors, gold_only_row
    from pisco.metrics import normalize
    from pisco.mm_utils import image_to_span, n_image_tokens, merge_size_of, get_vision_token_ids
    setn, topk, xd, cml, gold, _ = VARIANTS[vname]
    ds = _ds(setn)
    row0 = ds[i]
    row = row0
    if xd:
        if setn not in G["by_source"]:
            bs = defaultdict(list)
            for j, s in enumerate(ds["source"]):
                bs[s].append(j)
            G["by_source"][setn] = bs
        row = with_distractors(ds, i, row, G["by_source"][setn], xd)
    if gold:
        row = gold_only_row(row)
    coll, dtok, ctok = _coll(topk, cml), G["dtok"], G["ctok"]
    # --- pisco mode: exactly gen_pisco's prompt
    docs = [d if d else " " for d in row["docs"]]
    imgs = list(row.get("doc_images") or [])
    query = coll.clean_text(row["query"])
    batch, background, _ = coll.example_chunks(docs, imgs, query)
    ids = dtok(_build_chat_prompt(dtok, background=background, question=query), add_special_tokens=False)["input_ids"]
    n_mem = sum(1 for t in ids if t == dtok.mem_token_id)
    assert n_mem == batch.total_mems, (n_mem, batch.total_mems)
    chunk_toks = [len(c) for c in batch.input_ids]
    p_keys = [_doc_key(d, imgs[j] if j < len(imgs) else None) for j, d in enumerate(list(row["docs"])[:coll.topk_docs])]
    # --- base mode: exactly gen_base's inputs
    bdocs, bimgs = list(row["docs"])[:topk], list(row.get("doc_images") or [])[:topk]
    b_keys = [_doc_key(d, bimgs[j] if j < len(bimgs) else None) for j, d in enumerate(bdocs)]
    merge = merge_size_of(G["bip"]); vis = get_vision_token_ids(dtok)
    bg, k_img = "", 0
    from pisco.mm_utils import image_span_text
    for j, d in enumerate(bdocs):
        img = bimgs[j] if j < len(bimgs) else None
        if img is not None:
            grid = batch.image_grid_thw[k_img] if G["same_ip"] else image_to_span(img, G["bip"], vis)[2]
            k_img += 1
            bg += f"Document {j}:" + image_span_text(n_image_tokens(grid, merge))
        else:
            bg += f"Document {j}:" + (d or "")
    b_tokens = len(dtok(_build_chat_prompt(dtok, background=bg, question=row["query"]), add_special_tokens=False)["input_ids"])
    # --- cut: gold answer in the question's own raw text docs, not in the compressor's text (or a gold page past TOPK)
    answers = [a for a in (row0.get("answers") or [row0.get("mistral_label", "")]) if isinstance(a, str) and a.strip()]
    na = [normalize(a) for a in answers]
    na = [a for a in na if a]
    raw_txt = normalize(" ".join(d or "" for d in row0["docs"]))
    comp_txt = normalize(" ".join(ctok.decode([t for t in c if t != ctok.mem_token_id]) for c, im in
                                  zip(batch.input_ids, batch.is_image) if not im))
    base_txt = normalize(" ".join(d or "" for j, d in enumerate(bdocs) if not (j < len(bimgs) and bimgs[j] is not None)))
    in_raw = any(a in raw_txt for a in na)
    gold_pos = [int(g) for g in (row.get("gold_positions") or [])]
    gold_dropped = bool(gold_pos) and all(g >= topk for g in gold_pos)
    text_trunc = any(len(ctok(d, add_special_tokens=False)["input_ids"]) > cml for j, d in enumerate(list(row["docs"])[:topk])
                     if not (j < len(imgs) and imgs[j] is not None))
    return dict(v=vname, i=i, qid=row0.get("query_id"), src=row0.get("source"), n_mem=n_mem, n_chunks=len(chunk_toks),
                chunk_tok_max=max(chunk_toks) if chunk_toks else 0, chunk_toks=chunk_toks, prompt=len(ids), span=_span(ids),
                base_prompt=b_tokens, parity=p_keys == b_keys, n_docs=len(bdocs), in_raw=in_raw,
                cut_pisco=in_raw and not any(a in comp_txt for a in na), cut_base=in_raw and not any(a in base_txt for a in na),
                gold_dropped=gold_dropped, text_trunc=text_trunc)


STAGES = {  # name: training_config.yaml of the run
    "stage1_C256": f"{LC}/pt_text_1M_cml256_188422/training_config.yaml",
    "stage1_L1024": f"{LC}/pt_text_1M_cml1024_188466/training_config.yaml",
    "stage2_C256": f"{LC}/pt_mm_cml256/training_config.yaml",
    "finetune_C256": f"{LC}/ft_cml256_s42/training_config.yaml",
}


def _stage_ds(tc):
    import datasets
    specs = [t.strip() for t in str(tc["data"]["training_dataset"]).split(",") if t.strip()]
    parts = []
    for spec in specs:   # same as pisco/train.py
        path, _, cap = spec.rpartition(":") if ":" in spec and spec.rsplit(":", 1)[-1].isdigit() else (spec, "", "")
        d = datasets.load_from_disk(path)
        if isinstance(d, datasets.DatasetDict):
            d = d["train"]
        parts.append(d.select(range(min(len(d), int(cap)))) if cap else d)
    ds = parts[0] if len(parts) == 1 else datasets.concatenate_datasets(parts).shuffle(seed=42)
    return ds.select(range(min(len(ds), int(tc["data"]["samples"]))))


def stage_rows(task):
    sname, idxs = task
    _setup()
    import yaml, importlib
    torch = G["torch"]
    tc = yaml.safe_load(open(STAGES[sname]))
    mod, _, cls = tc["collator_class"].rpartition(".")
    C = getattr(importlib.import_module(mod), cls)
    kw = dict(tc.get("collator_kwargs") or {})
    import inspect
    if "image_processor" in inspect.signature(C.__init__).parameters:
        kw["image_processor"] = G["ip"]
    if sname not in G:
        G[sname] = (C(compressor_tokenizer=G["ctok"], decoder_tokenizer=G["dtok"], compr_rate=G["cfg"].compr_rate, **kw),
                    _stage_ds(tc))
    coll, ds = G[sname]
    out = []
    for i in idxs:
        random.seed(i); torch.manual_seed(i)
        import numpy as np; np.random.seed(i % (2 ** 31))
        try:
            b = coll.torch_call([ds[int(i)]])
        except Exception as e:
            out.append(dict(v=sname, i=int(i), err=repr(e)[:200])); continue
        d = b["decoder_input_ids"][0]; m = b["decoder_attention_mask"][0]
        ids = d[m.bool()].tolist()
        cm = b["compressor_attention_mask"]
        out.append(dict(v=sname, i=int(i), n_mem=sum(1 for t in ids if t == G["dtok"].mem_token_id), prompt=len(ids),
                        span=_span(ids), n_chunks=int(cm.shape[0]), chunk_toks=[int(x) for x in cm.sum(1).tolist()]))
    return out


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


def summ(rows, key):
    xs = [r[key] for r in rows if key in r]
    return dict(p50=pct(xs, .5), p99=pct(xs, .99), max=max(xs) if xs else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="rows per variant (smoke); 0 = all")
    ap.add_argument("--stage_rows", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    t0 = time.time()
    _setup()   # once, inherited by the forked workers
    import datasets
    tasks = []
    for v, (setn, *_r) in VARIANTS.items():
        d = datasets.load_from_disk(f"{ED}/{setn}")
        n = len(d["train"] if isinstance(d, datasets.DatasetDict) else d)
        tasks += [(v, i) for i in range(min(n, a.limit) if a.limit else n)]
    stage_tasks = []
    for s, path in STAGES.items():
        import yaml
        tc = yaml.safe_load(open(path))
        n = min(int(tc["data"]["samples"]), 10 ** 9)
        k = min(a.limit or a.stage_rows, a.stage_rows)
        idx = sorted(random.Random(0).sample(range(n), k)) if n > k else list(range(n))
        stage_tasks += [(s, idx[j:j + 50]) for j in range(0, len(idx), 50)]
    print(f"[audit] {len(tasks)} eval rows, {len(stage_tasks)} stage batches, {a.workers} workers", flush=True)
    rows, srows, errors = [], [], []
    with Pool(a.workers, initializer=_setup) as pool:
        for k, r in enumerate(pool.imap_unordered(_safe_eval, tasks, chunksize=8)):
            (errors if "err" in r else rows).append(r)
            if (k + 1) % 2000 == 0:
                print(f"[audit] eval {k + 1}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
        for rs in pool.imap_unordered(stage_rows, stage_tasks):
            for r in rs:
                (errors if "err" in r else srows).append(r)
    print(f"[audit] done rows in {time.time() - t0:.0f}s; {len(errors)} errors", flush=True)
    by = defaultdict(list)
    for r in rows + srows:
        by[r["v"]].append(r)
    ft = by.get("finetune_C256", [])
    ft_max_mem = max((r["n_mem"] for r in ft), default=None)
    ft_max_span = max((r["span"] for r in ft), default=None)
    report = {"meta": dict(R=R, PX=PX, stride=G.get("stride"), limit=a.limit, stage_rows=a.stage_rows,
                           finetune_max_mem=ft_max_mem, finetune_max_span=ft_max_span, n_errors=len(errors),
                           errors=errors[:20], wall_s=round(time.time() - t0)), "variants": {}, "stages": {}}
    for v in VARIANTS:
        rs = by.get(v, [])
        if not rs:
            continue
        n = len(rs)
        s = dict(n=n, set=VARIANTS[v][0], topk=VARIANTS[v][1], xd=VARIANTS[v][2], cml=VARIANTS[v][3], gold=VARIANTS[v][4],
                 gating=VARIANTS[v][5], mems=summ(rs, "n_mem"), mems_total=sum(r["n_mem"] for r in rs),
                 chunk_tokens=dict(p50=pct([t for r in rs for t in r["chunk_toks"]], .5),
                                   p99=pct([t for r in rs for t in r["chunk_toks"]], .99),
                                   max=max(r["chunk_tok_max"] for r in rs)),
                 prompt=summ(rs, "prompt"), span=summ(rs, "span"), base_prompt=summ(rs, "base_prompt"),
                 share_prompt_gt2048=sum(r["prompt"] > 2048 for r in rs) / n,
                 share_base_prompt_gt2048=sum(r["base_prompt"] > 2048 for r in rs) / n,
                 answer_in_raw=sum(r["in_raw"] for r in rs),
                 cut_pisco=sum(r["cut_pisco"] for r in rs), cut_base=sum(r["cut_base"] for r in rs),
                 gold_dropped=sum(r["gold_dropped"] for r in rs),
                 cut_rate=(sum(r["cut_pisco"] or r["gold_dropped"] for r in rs)) / n,
                 parity=sum(r["parity"] for r in rs) / n, text_trunc_share=sum(r["text_trunc"] for r in rs) / n)
        if ft_max_mem is not None:
            s["out_of_range_share"] = sum(r["n_mem"] > ft_max_mem or r["span"] > ft_max_span for r in rs) / n
        if v in REF and os.path.exists(REF[v]):
            ref = {str(x["query_id"]): x.get("mems_used") for x in json.load(open(REF[v]))["records"]}
            mine = {str(r["qid"]): r["n_mem"] for r in rs}
            common = [q for q in mine if q in ref]
            s["ref"] = dict(file=REF[v], n_common=len(common), n_equal=sum(mine[q] == ref[q] for q in common),
                            ref_total_common=sum(ref[q] for q in common), mine_total_common=sum(mine[q] for q in common))
        report["variants"][v] = s
    for sname in STAGES:
        rs = by.get(sname, [])
        if rs:
            ct = [t for r in rs for t in r["chunk_toks"]]
            report["stages"][sname] = dict(n=len(rs), mems=summ(rs, "n_mem"), prompt=summ(rs, "prompt"), span=summ(rs, "span"),
                                           chunks=summ(rs, "n_chunks"),
                                           chunk_tokens=dict(p50=pct(ct, .5), p99=pct(ct, .99), max=max(ct) if ct else None))
    h = report["variants"].get("docs_p4", {})
    report["health"] = dict(docs_p4_mems_total=h.get("mems_total"), expected=646042,
                            pass_=(h.get("mems_total") == 646042) if not a.limit else None)
    json.dump(report, open(f"{a.out}/audit.json", "w"), indent=1)
    with open(f"{a.out}/rows.jsonl", "w") as f:
        for r in rows + srows:
            r = dict(r); r.pop("chunk_toks", None); f.write(json.dumps(r) + "\n")
    print(json.dumps({k: report[k] for k in ("meta", "health")}, indent=1))
    hdr = "| variant | n | <MEM>/q p50/p99/max | chunk tok p50/max | prompt p50/p99/max | span p50/p99/max | base prompt p99/max | cut | gold drop | parity | out-of-range |"
    print(hdr)
    for v, s in report["variants"].items():
        print(f"| {v} | {s['n']} | {s['mems']['p50']}/{s['mems']['p99']}/{s['mems']['max']} | {s['chunk_tokens']['p50']}/{s['chunk_tokens']['max']} "
              f"| {s['prompt']['p50']}/{s['prompt']['p99']}/{s['prompt']['max']} | {s['span']['p50']}/{s['span']['p99']}/{s['span']['max']} "
              f"| {s['base_prompt']['p99']}/{s['base_prompt']['max']} | {s['cut_rate']:.4f} ({s['cut_pisco']}) | {s['gold_dropped']} "
              f"| {s['parity']:.4f} | {s.get('out_of_range_share', float('nan')):.4f} |")
    for sname, s in report["stages"].items():
        print(f"| {sname} | {s['n']} | {s['mems']['p50']}/{s['mems']['p99']}/{s['mems']['max']} | {s['chunk_tokens']['p50']}/{s['chunk_tokens']['max']} "
              f"| {s['prompt']['p50']}/{s['prompt']['p99']}/{s['prompt']['max']} | {s['span']['p50']}/{s['span']['p99']}/{s['span']['max']} | | | | | |")


def _safe_eval(task):
    try:
        return eval_row(task)
    except Exception as e:
        return dict(v=task[0], i=task[1], err=repr(e)[:300], tb=traceback.format_exc()[-600:])


if __name__ == "__main__":
    main()
