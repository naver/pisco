"""Multimodal document-QA eval: compressed PISCO vs. the uncompressed VLM reading raw pages.

Data: a datasets dir (build_mm_eval.py / build_mm_finetune.py schema): docs, doc_images,
query, mistral_label, answers, gold_positions, source.

Modes
  pisco       page images -> compressor -> <MEM> -> decoder (what we train)
  base        the decoder backbone (a VLM) reads the raw page images, same prompt  (ceiling)
  pisco_text  text PISCO on `docs` text (only meaningful when rows carry OCR text)

  python scripts/eval_mm_docqa.py --mode pisco --checkpoint_path expQ/<run>/model \
      --data_path eval_data/mm_docqa_500 --output_path expQ/<run>/eval/mm_docqa_pisco.json [--limit 50]
Metrics: EM / F1 / ANLS / containment-match, max over `answers`; compress/decode timings.

The pisco modes build their compressor chunks and "Document j:" background through
MultimodalFineTuningCollator.example_chunks, i.e. exactly as finetuning does (text docs truncated to
--compressor_max_length, clean_text, the OSCAR query prefix with --query_dependent). A prompt longer than
--decoder_max_length is an error, not a silent trim: right truncation would cut the question.
"""

import argparse
import json
import os
import pathlib
import re
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from pisco import mm_tasks  # noqa: E402
from pisco.metrics import anls_single, em_single, f1_single, match_single, normalize  # noqa: E402
from pisco.model import PISCO, PISCOConfig, eos_token_ids, merge_lora_inplace  # noqa: E402
from pisco.mm_collator import MultimodalFineTuningCollator  # noqa: E402
from pisco.mm_utils import (MMChunkBatch, collate_compressor, image_span_text, image_to_span, merge_size_of,  # noqa: E402
                            n_image_tokens)
from eval_agent_qa import _count_mem_tokens  # noqa: E402
from pisco.collator import FineTuningCollator  # noqa: E402

_FT_DEFAULTS = {k: v.default for k, v in __import__("inspect").signature(FineTuningCollator.__init__).parameters.items()
                if k in ("system_prompt", "user_prompt")}


# --answer_style expand: copy the evidence first, then answer (RLComp's "re-expansion", tested without RL).
EXPAND_SYSTEM_PROMPT = (
    "You are a helpful assistant. Your task is to extract relevant information from provided documents and to answer "
    "questions. First copy, word for word, the text of the documents that contains the answer, after 'Evidence:'. "
    "Then give the final answer, as briefly as possible, after 'Answer:'."
)


def final_answer(text: str) -> str:
    """The part after the last 'Answer:' of an expand-style output (the whole text when there is none)."""
    i = text.rfind("Answer:")
    return text[i + len("Answer:"):].strip() if i >= 0 else text.strip()


def _build_chat_prompt(tokenizer, *, background: str, question: str, system: str = None) -> str:
    """EXACTLY the finetuning prompt (FineTuningCollator defaults), as a generation prompt.
    eval_agent_qa's variant inserts a newline after 'Background:' that training never saw."""
    messages = [
        {"role": "system", "content": system or _FT_DEFAULTS["system_prompt"]},
        {"role": "user", "content": _FT_DEFAULTS["user_prompt"].replace("[documents]", background).replace("[question]", question)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def make_collator(model: PISCO, *, topk: int, compressor_max_length: int, image_compr_rate, query_dependent: bool):
    """The finetuning collator, used here only for example_chunks (chunking / truncation / cleaning)."""
    return MultimodalFineTuningCollator(
        compressor_tokenizer=model.compressor_tokenizer, decoder_tokenizer=model.decoder_tokenizer,
        compr_rate=model.compr_rate, compressor_max_length=compressor_max_length,
        image_processor=model.image_processor, image_compr_rate=image_compr_rate,
        topk_docs=topk, query_dependent=query_dependent, pad_to_multiple_of=None,
    )

TIMINGS = defaultdict(list)


def with_distractors(ds, i, row, by_source, n_extra) -> Dict[str, Any]:
    """The row with n_extra more pages, each the first page of a distant row of the same source whose id names another
    document (first 12 characters differ), then all pages shuffled (seeded by query_id), gold_positions remapped."""
    import random
    idx = by_source[row.get("source")]
    p, key = idx.index(i), str(row.get("query_id"))[:12]
    docs, imgs = list(row["docs"]), list(row.get("doc_images") or [])
    imgs += [None] * (len(docs) - len(imgs))
    t = len(idx) // 2
    while len(docs) < len(row["docs"]) + n_extra and t < len(idx) // 2 + len(idx):
        j = idx[(p + t) % len(idx)]
        t += 1
        if j == i or str(ds[j].get("query_id"))[:12] == key:
            continue
        other = ds[j]
        docs.append(other["docs"][0]); imgs.append((other.get("doc_images") or [None])[0])
    order = list(range(len(docs)))
    random.Random(str(row.get("query_id"))).shuffle(order)
    out = dict(row)
    out["docs"], out["doc_images"] = [docs[o] for o in order], [imgs[o] for o in order]
    out["gold_positions"] = [order.index(int(g)) for g in (row.get("gold_positions") or [])]
    return out


def gold_only_row(row) -> Dict[str, Any]:
    """The row with its distractor pages removed (gold pages kept in order, gold_positions renumbered)."""
    gold = sorted({int(g) for g in (row.get("gold_positions") or [])})
    imgs = list(row.get("doc_images") or [])
    out = dict(row)
    out["docs"] = [row["docs"][g] for g in gold]
    out["doc_images"] = [imgs[g] if g < len(imgs) else None for g in gold]
    out["gold_positions"] = list(range(len(gold)))
    return out


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _flatten_answers(answers) -> List[str]:
    """Defensive: expand stringified lists ("['2003']") left by older eval builds."""
    import ast
    out = []
    for a in answers or []:
        if a is None:
            continue
        a = str(a).strip()
        if a.startswith("[") and a.endswith("]"):
            try:
                v = ast.literal_eval(a)
                if isinstance(v, (list, tuple)):
                    out.extend(str(x) for x in v); continue
            except (ValueError, SyntaxError):
                pass
        out.append(a)
    return out or [""]


def score(pred: str, answers: List[str]) -> Dict[str, float]:
    answers = [a for a in _flatten_answers(answers) if normalize(a)]
    if not answers:   # no usable gold: nothing can be right (an empty gold would match everything)
        return {"em": 0.0, "f1": 0.0, "anls": 0.0, "match": 0.0}
    return {
        "em": max(em_single(pred, a) for a in answers),
        # f1_single does not normalise (pisco.metrics legacy); use the same normalisation as EM
        "f1": max(f1_single(normalize(pred), normalize(a))[0] for a in answers),
        "anls": max(anls_single(pred, a) for a in answers),
        "match": max(match_single(pred, a) for a in answers),
    }


def _greedy_confidence(out, eos_ids) -> Dict[str, float]:
    """Log-probabilities of the greedy answer tokens (up to and including the first stop token)."""
    seq = out.sequences[0, -len(out.logits):]
    lps = []
    for t, logits in enumerate(out.logits):
        tok = int(seq[t])
        lps.append(float(torch.log_softmax(logits[0].float(), -1)[tok]))
        if tok in eos_ids:
            break
    return {"logprob_sum": float(np.sum(lps)), "logprob_mean": float(np.mean(lps)),
            "p_first": float(np.exp(lps[0]))}


def quantize_mems(emb: torch.Tensor, kind: str) -> torch.Tensor:
    """Storage probe: the <MEM> vectors stored at lower precision, then restored (per-vector absmax for int8/int4)."""
    if kind == "fp8":
        return emb.to(torch.float8_e4m3fn).to(emb.dtype)
    q = {"int8": 127, "int4": 7}[kind]
    scale = emb.float().abs().amax(-1, keepdim=True).clamp_min(1e-8) / q
    return ((emb.float() / scale).round().clamp(-q, q) * scale).to(emb.dtype)


@torch.inference_mode()
def gen_pisco(model: PISCO, coll: MultimodalFineTuningCollator, row, *, device, decoder_max_length, max_new_tokens,
              text_only=False, num_samples=0, temperature=1.0, top_p=0.95, system=None, mem_quant=None):
    dtok = model.decoder_tokenizer
    docs = [d if d else " " for d in row["docs"]]
    imgs = [] if text_only else list(row.get("doc_images") or [])
    query = coll.clean_text(row["query"])
    batch, background, _ = coll.example_chunks(docs, imgs, query)
    prompt = _build_chat_prompt(dtok, background=background, question=query, system=system)
    dec = dtok(prompt, return_tensors="pt", add_special_tokens=False, padding=False)
    total = batch.total_mems
    got = _count_mem_tokens(dec["input_ids"], dtok.mem_token_id)
    assert got == total, f"{got} <MEM> in the prompt, {total} chunks' worth"
    if dec["input_ids"].shape[1] > decoder_max_length:
        # Right truncation would drop the question and could cut an image's <MEM>s while its pixels stay.
        raise RuntimeError(f"prompt of {dec['input_ids'].shape[1]} tokens exceeds --decoder_max_length "
                           f"{decoder_max_length} (query_id {row.get('query_id')}, {total} <MEM>): raise it, "
                           "or lower --topk_docs / --image_compr_rate")
    comp = collate_compressor([batch], coll, coll.vision_ids)
    comp = {k: v.to(device) for k, v in comp.items()}
    _sync(); t = time.perf_counter()
    emb = model.compress(comp["compressor_input_ids"], comp["compressor_attention_mask"],
                         pixel_values=comp.get("pixel_values"), image_grid_thw=comp.get("image_grid_thw"),
                         mm_token_type_ids=comp.get("mm_token_type_ids"))
    _sync(); TIMINGS["compress"].append(time.perf_counter() - t)
    if mem_quant:
        emb = quantize_mems(emb, mem_quant)
    embeds = model.replace_embeddings(emb, dec["input_ids"].to(device))
    attn = dec["attention_mask"].to(device)
    eos_ids = model.decoder_eos_ids()
    _sync(); t = time.perf_counter()
    if num_samples:   # same greedy tokens, plus the per-step logits for a confidence score
        res = model.decoder.generate(inputs_embeds=embeds, attention_mask=attn, do_sample=False, top_p=None,
                                     max_new_tokens=max_new_tokens, eos_token_id=eos_ids,
                                     return_dict_in_generate=True, output_logits=True)
        out = res.sequences
    else:
        out = model.decoder.generate(inputs_embeds=embeds, attention_mask=attn,
                                     do_sample=False, top_p=None, max_new_tokens=max_new_tokens,
                                     eos_token_id=eos_ids)
    _sync(); TIMINGS["decode"].append(time.perf_counter() - t)
    text = dtok.batch_decode(out, skip_special_tokens=True)[0].strip()
    stats = {"mems_used": got, "n_images": int(sum(batch.is_image)),
             "decoder_prompt_tokens": int(dec["input_ids"].shape[1])}
    if num_samples:   # G sampled answers from the same <MEM> context (RL / rejection-sampling headroom)
        stats.update(_greedy_confidence(res, set(eos_ids)))
        sids = model.decoder.generate(inputs_embeds=embeds, attention_mask=attn, do_sample=True,
                                      temperature=temperature, top_p=top_p, top_k=0,   # nucleus only (no HF top-k 50)
                                      num_return_sequences=num_samples,
                                      max_new_tokens=max_new_tokens, eos_token_id=eos_ids)
        stats["samples"] = [s.strip() for s in dtok.batch_decode(sids, skip_special_tokens=True)]
        # a sample that never emitted a stop token was cut at max_new_tokens (unusable as a target)
        stats["samples_finished"] = [bool(any(int(x) in eos_ids for x in r)) for r in sids]
    return text, stats


ROUTE_QUESTION = 'Which document contains the answer to this question: "{q}"? Answer with the document number.'


def _reset_rope_deltas(module):
    """Qwen3.5 caches `rope_deltas` from its last image forward and reuses them for an embeds-only forward."""
    for m in module.modules():
        if hasattr(m, "rope_deltas"):
            m.rope_deltas = None


@torch.inference_mode()
def route_logprobs(model: PISCO, background: str, emb, query: str, n_docs: int, device) -> List[float]:
    """log p("Document j") over the n_docs candidates, from the all-<MEM> context: the stage-2 `find` task
    (pisco.mm_tasks) asked about the answer instead of a snippet. One forward: candidates differ in their last token."""
    dtok = model.decoder_tokenizer
    msgs = [{"role": "system", "content": mm_tasks.FIND_SYSTEM_PROMPT},
            {"role": "user", "content": mm_tasks.user_content(background, ROUTE_QUESTION.format(q=query))}]
    prompt = dtok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    cands = [dtok(prompt + mm_tasks.find_answer(j), add_special_tokens=False)["input_ids"] for j in range(n_docs)]
    assert all(len(c) == len(cands[0]) and c[:-1] == cands[0][:-1] for c in cands), "candidates differ before the digit"
    ids = torch.tensor([cands[0][:-1]], device=device)
    _reset_rope_deltas(model.decoder)
    logits = model.decoder(inputs_embeds=model.replace_embeddings(emb, ids), attention_mask=torch.ones_like(ids),
                           logits_to_keep=1).logits[0, -1].float()
    return torch.log_softmax(logits[[c[-1] for c in cands]], -1).tolist()


@torch.inference_mode()
def gen_pisco_mixed(model: PISCO, coll: MultimodalFineTuningCollator, row, *, device, decoder_max_length,
                    max_new_tokens, raw_policy):
    """Selective expansion (REFRAG-style): some pages reach the decoder as raw image spans, the others as <MEM>s.
    raw_policy: gold = the gold pages raw (oracle upper bound); routeK = the K pages the model itself ranks highest
    (route_logprobs on the all-<MEM> context) raw; all = every page raw, through the PISCO decoder.
    The <MEM> vectors enter through a hook on the decoder's input-embedding layer, so Qwen's own forward places the
    vision features and builds the M-RoPE positions from input_ids."""
    dtok = model.decoder_tokenizer
    docs = [d if d else " " for d in row["docs"]][: coll.topk_docs]
    imgs = list(row.get("doc_images") or [])
    query = coll.clean_text(row["query"])
    per_doc = [coll.example_chunks([d], imgs[j:j + 1], query)[0] for j, d in enumerate(docs)]
    full = MMChunkBatch()
    for b in per_doc:
        full.extend(b)
    comp = {k: v.to(device) for k, v in collate_compressor([full], coll, coll.vision_ids).items()}
    _sync(); t = time.perf_counter()
    emb = model.compress(comp["compressor_input_ids"], comp["compressor_attention_mask"],
                         pixel_values=comp.get("pixel_values"), image_grid_thw=comp.get("image_grid_thw"),
                         mm_token_type_ids=comp.get("mm_token_type_ids"))
    _sync(); TIMINGS["compress"].append(time.perf_counter() - t)
    offs = np.cumsum([0] + [b.total_mems for b in per_doc])
    gold = {int(g) for g in (row.get("gold_positions") or [])}
    stats = {}
    if raw_policy == "gold":
        raw = {g for g in gold if g < len(docs)}
    elif raw_policy == "all":
        raw = set(range(len(docs)))
    else:   # routeK: the K pages the model ranks highest
        bg = "".join(f"Document {j}:" + dtok.mem_token * b.total_mems for j, b in enumerate(per_doc))
        lp = route_logprobs(model, bg, emb, query, len(docs), device)
        picks = [int(j) for j in np.argsort(lp)[::-1][: int(raw_policy[len("route"):])]]
        raw = set(picks)
        stats.update(route_logprobs=lp, route_pick=picks[0], route_picks=picks, route_hit=float(picks[0] in gold))
    raw = {j for j in raw if per_doc[j].is_image and per_doc[j].is_image[0]}
    merge = merge_size_of(model.image_processor)
    background, pvs, grids, keep = "", [], [], []
    for j, b in enumerate(per_doc):
        if j in raw:
            background += f"Document {j}:" + image_span_text(n_image_tokens(b.image_grid_thw[0], merge))
            pvs.append(b.pixel_values[0]); grids.append(list(b.image_grid_thw[0]))
        else:
            background += f"Document {j}:" + dtok.mem_token * b.total_mems
            keep.append(emb[offs[j]:offs[j + 1]])
    prompt = _build_chat_prompt(dtok, background=background, question=query)
    dec = dtok(prompt, return_tensors="pt", add_special_tokens=False, padding=False)
    ids = dec["input_ids"].to(device)
    if ids.shape[1] > decoder_max_length:
        raise RuntimeError(f"prompt of {ids.shape[1]} tokens exceeds --decoder_max_length {decoder_max_length} "
                           f"(query_id {row.get('query_id')}, {len(raw)} raw pages)")
    sel = torch.cat(keep) if keep else emb[:0]
    mem_id, ip = dtok.mem_token_id, coll.vision_ids[1]
    assert int((ids == mem_id).sum()) == sel.shape[0], "decoder <MEM> count != compressed vectors kept"
    assert int((ids == ip).sum()) == sum(n_image_tokens(g, merge) for g in grids), "template altered an image span"
    kwargs = dict(input_ids=ids, attention_mask=dec["attention_mask"].to(device))
    if pvs:
        kwargs.update(pixel_values=torch.from_numpy(np.concatenate(pvs, 0)).to(device),
                      image_grid_thw=torch.tensor(grids, dtype=torch.long, device=device),
                      mm_token_type_ids=(ids == ip).long())
    fired = []

    def put_mems(_mod, inp, out):   # prefill only: decode steps carry no <MEM>
        m = inp[0] == mem_id
        if not m.any():
            return None
        assert int(m.sum()) == sel.shape[0], "a forward with a different <MEM> count reached the embedding hook"
        out = out.clone()
        out[m] = sel.to(out.dtype)
        fired.append(int(m.sum()))
        return out

    h = model.decoder.get_input_embeddings().register_forward_hook(put_mems)
    try:
        _reset_rope_deltas(model.decoder)
        _sync(); t = time.perf_counter()
        out = model.decoder.generate(**kwargs, do_sample=False, top_p=None, max_new_tokens=max_new_tokens,
                                     eos_token_id=model.decoder_eos_ids())
        _sync(); TIMINGS["decode"].append(time.perf_counter() - t)
    finally:
        h.remove()
    assert sel.shape[0] == 0 or fired, "the embedding hook never fired: <MEM> vectors were not injected"
    text = dtok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)[0].strip()
    stats.update(mems_used=int(sel.shape[0]), n_images=len(per_doc), n_raw=len(raw),
                 decoder_prompt_tokens=int(ids.shape[1]))
    return text, stats


@torch.inference_mode()
def gen_base(base, proc, tok, row, *, device, topk, max_new_tokens, decoder_max_length):
    cfg = base.config
    vision_ids = (int(cfg.vision_start_token_id), int(cfg.image_token_id), int(cfg.vision_end_token_id))
    ip = vision_ids[1]
    merge = merge_size_of(proc)
    docs = list(row["docs"])[:topk]
    imgs = list(row.get("doc_images") or [])[:topk]
    background, pvs, grids = "", [], []
    for j, d in enumerate(docs):
        img = imgs[j] if j < len(imgs) else None
        if img is not None:
            _, pv, grid = image_to_span(img, proc, vision_ids)
            pvs.append(pv); grids.append(grid)
            background += f"Document {j}:" + image_span_text(n_image_tokens(grid, merge))
        else:
            background += f"Document {j}:" + (d or "")
    prompt = _build_chat_prompt(tok, background=background, question=row["query"])
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"]
    n_pad_tokens = int((ids == ip).sum())
    expected = sum(n_image_tokens(g, merge) for g in grids)
    assert n_pad_tokens == expected, f"template altered the image_pad run: {n_pad_tokens} vs {expected}"
    kwargs = dict(input_ids=ids.to(device), attention_mask=enc["attention_mask"].to(device))
    if pvs:
        kwargs.update(pixel_values=torch.from_numpy(np.concatenate(pvs, 0)).to(device),
                      image_grid_thw=torch.tensor(grids, dtype=torch.long, device=device),
                      mm_token_type_ids=(ids == ip).long().to(device))
    _sync(); t = time.perf_counter()
    out = base.generate(**kwargs, do_sample=False, top_p=None, max_new_tokens=max_new_tokens,
                        eos_token_id=eos_token_ids(tok))
    _sync(); TIMINGS["decode"].append(time.perf_counter() - t)
    text = tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)[0].strip()
    return text, {"n_images": len(pvs), "prompt_tokens": int(ids.shape[1]), "visual_tokens": expected}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pisco", "base", "pisco_text"], default="pisco")
    ap.add_argument("--checkpoint_path", default=None)
    ap.add_argument("--base_model_name", default=None, help="base mode: VLM to load (default: the ckpt's decoder)")
    ap.add_argument("--base_adapter_path", default=None, help="base mode: PEFT adapter dir from scripts/train_base_mm.py")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--topk_docs", type=int, default=4)
    ap.add_argument("--compressor_max_length", type=int, default=256)
    ap.add_argument("--decoder_max_length", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--image_compr_rate", type=int, default=None)
    ap.add_argument("--image_max_pixels", type=int, default=None,
                    help="pixel budget per page; default: the checkpoint's own (base mode: Qwen3.5 default)")
    ap.add_argument("--query_dependent", action="store_true", help="OSCAR checkpoint: query prefix on the compressor side")
    ap.add_argument("--image_min_pixels", type=int, default=None,
                    help="lower bound of the pixel budget (default: the checkpoint's / processor's, 65536); needed when "
                         "--image_max_pixels is set below it, e.g. the equal-token low-resolution baseline")
    ap.add_argument("--merge_lora", action="store_true")
    ap.add_argument("--mem_quant", choices=["int8", "fp8", "int4"], default=None,
                    help="pisco mode: store the <MEM> vectors at this precision (storage probe, quantize_mems)")
    ap.add_argument("--extra_distractors", type=int, default=0,
                    help="add N pages of other documents of the same source, then shuffle (with_distractors); "
                         "set --topk_docs to the new page count")
    ap.add_argument("--gold_only", action="store_true", help="drop the distractor pages (both modes): distractor interference")
    ap.add_argument("--answer_style", choices=["brief", "expand"], default="brief",
                    help="pisco mode: expand = copy the evidence, then answer (EXPAND_SYSTEM_PROMPT); prediction = the "
                         "text after 'Answer:', the full output kept as prediction_raw")
    ap.add_argument("--raw_policy", choices=["none", "gold", "route1", "route2", "all"], default="none",
                    help="pisco mode: pages fed raw (image spans) instead of <MEM>s: gold pages (oracle), the page the "
                         "model routes to, or all (selective-expansion probe, gen_pisco_mixed)")
    ap.add_argument("--num_samples", type=int, default=0,
                    help="pisco modes: also draw G sampled answers per question + greedy log-prob confidence")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rescore", default=None, help="re-score the records of an existing output json (no model run)")
    ap.add_argument("--cut_at_user", action="store_true",
                    help="rescore: truncate predictions at a hallucinated next turn ('\\nuser\\n'), for files generated before <|im_end|> was a stop token")
    args = ap.parse_args()

    if args.rescore:
        with open(args.rescore) as f:
            old = json.load(f)
        agg = defaultdict(list)
        for r in old["records"]:
            if args.cut_at_user:
                r["prediction_raw"] = r.get("prediction_raw", r["prediction"])
                r["prediction"] = re.split(r"\n(?:user|assistant|system)\n", r["prediction_raw"], 1)[0].strip()
            sc = score(r["prediction"], r["answers"]); r.update(sc)
            for k, v in sc.items():
                agg[k].append(v); agg[f"{k}/{r.get('source', '?')}"].append(v)
        old["summary"].update({k: float(np.mean(v)) for k, v in agg.items()})
        with open(args.output_path, "w") as f:
            json.dump(old, f, indent=1, ensure_ascii=False)
        for k in sorted(old["summary"]):
            if isinstance(old["summary"][k], float): print(f"  {k:28s} {old['summary'][k]:.4f}")
        return

    import datasets
    torch.manual_seed(args.seed)
    assert not (args.num_samples and args.mode == "base"), "--num_samples is implemented for the pisco modes"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ds = datasets.load_from_disk(args.data_path)["train"]
    except KeyError:
        ds = datasets.load_from_disk(args.data_path)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"[eval] {len(ds)} rows from {args.data_path}; mode={args.mode}")

    model = base = proc = tok = None
    if args.mode in ("pisco", "pisco_text"):
        assert args.checkpoint_path, "--checkpoint_path required"
        overrides = {k: v for k, v in (("image_max_pixels", args.image_max_pixels),
                                       ("image_min_pixels", args.image_min_pixels)) if v}
        model = PISCO.from_pretrained(args.checkpoint_path, **overrides)
        model.to(device).eval()
        if args.merge_lora:   # fold LoRA into the weights: same outputs, ~1.8x faster per decoded token
            model.decoder = merge_lora_inplace(model.decoder)
            model.compressor = merge_lora_inplace(model.compressor)
        assert model.image_processor is not None, "checkpoint has no image processor"
        args.image_max_pixels = int(model.config.image_max_pixels)
        coll = make_collator(model, topk=args.topk_docs, compressor_max_length=args.compressor_max_length,
                             image_compr_rate=args.image_compr_rate, query_dependent=args.query_dependent)
        print(f"[eval] image_max_pixels={args.image_max_pixels} (checkpoint: {not overrides})")
    else:
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        from pisco.mm_utils import load_image_processor
        name = args.base_model_name or PISCOConfig.from_pretrained(args.checkpoint_path).decoder_model_name
        tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        base = AutoModelForImageTextToText.from_pretrained(name, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        if args.base_adapter_path:   # fair ceiling: the backbone LoRA-finetuned on raw images (scripts/train_base_mm.py)
            from peft import PeftModel
            base = PeftModel.from_pretrained(base, args.base_adapter_path).merge_and_unload()
            print(f"[base] loaded + merged adapter {args.base_adapter_path}")
        base.to(device).eval()
        proc = load_image_processor(name, **{k: v for k, v in (("max_pixels", args.image_max_pixels),
                                                               ("min_pixels", args.image_min_pixels)) if v})
        if not args.image_max_pixels:   # record the processor's budget (attribute name differs across versions)
            size = getattr(proc, "size", None) or {}
            mp = getattr(proc, "max_pixels", None) or size.get("longest_edge") or size.get("max_pixels")
            args.image_max_pixels = int(mp) if mp else None
        print(f"[base] {name}")

    records, agg = [], defaultdict(list)
    t0 = time.time()
    by_source = defaultdict(list)
    if args.extra_distractors:
        for j, src in enumerate(ds["source"]):
            by_source[src].append(j)
    for i, row in enumerate(ds):
        if args.extra_distractors:
            row = with_distractors(ds, i, row, by_source, args.extra_distractors)
        if args.gold_only:
            row = gold_only_row(row)
        if args.mode == "base":
            pred, stats = gen_base(base, proc, tok, row, device=device, topk=args.topk_docs,
                                   max_new_tokens=args.max_new_tokens, decoder_max_length=args.decoder_max_length)
        elif args.raw_policy != "none":
            pred, stats = gen_pisco_mixed(model, coll, row, device=device, decoder_max_length=args.decoder_max_length,
                                          max_new_tokens=args.max_new_tokens, raw_policy=args.raw_policy)
            if "route_hit" in stats:
                agg["route_hit"].append(stats["route_hit"]); agg[f"route_hit/{row.get('source', '?')}"].append(stats["route_hit"])
        else:
            pred, stats = gen_pisco(model, coll, row, device=device,
                                    decoder_max_length=args.decoder_max_length, max_new_tokens=args.max_new_tokens,
                                    text_only=(args.mode == "pisco_text"),
                                    num_samples=args.num_samples, temperature=args.temperature, top_p=args.top_p,
                                    system=EXPAND_SYSTEM_PROMPT if args.answer_style == "expand" else None,
                                    mem_quant=args.mem_quant)
            if args.answer_style == "expand":
                stats["prediction_raw"], pred = pred, final_answer(pred)
        answers = list(row.get("answers") or [row.get("mistral_label", "")])
        sc = score(pred, answers)
        if "samples" in stats:
            ss = [score(s, answers) for s in stats["samples"]]
            stats["samples_em"] = [s["em"] for s in ss]
            stats["samples_anls"] = [s["anls"] for s in ss]
            hit = float(any(s["anls"] >= 0.5 for s in ss))
            agg["pass@G_anls50"].append(hit); agg[f"pass@G_anls50/{row.get('source', '?')}"].append(hit)
        for k, v in sc.items():
            agg[k].append(v); agg[f"{k}/{row.get('source', '?')}"].append(v)
        records.append({"query_id": row.get("query_id"), "source": row.get("source"), "query": row["query"],
                        "answers": answers, "prediction": pred, **sc, **stats})
        if (i + 1) % 10 == 0:
            print(f"[{i + 1}/{len(ds)}] em={np.mean(agg['em']):.3f} f1={np.mean(agg['f1']):.3f} "
                  f"anls={np.mean(agg['anls']):.3f} match={np.mean(agg['match']):.3f}  {time.time() - t0:.0f}s", flush=True)

    summary = {k: float(np.mean(v)) for k, v in agg.items()}
    summary.update({"n": len(records), "mode": args.mode, "checkpoint": args.checkpoint_path,
                    "topk_docs": args.topk_docs, "image_compr_rate": args.image_compr_rate,
                    "image_max_pixels": args.image_max_pixels, "image_min_pixels": args.image_min_pixels,
                    "query_dependent": args.query_dependent, "raw_policy": args.raw_policy,
                    "answer_style": args.answer_style, "gold_only": args.gold_only, "mem_quant": args.mem_quant,
                    "extra_distractors": args.extra_distractors,
                    "num_samples": args.num_samples,
                    "temperature": args.temperature, "top_p": args.top_p, "top_k": 0,
                    **{f"time_{k}_mean_s": float(np.mean(v)) for k, v in TIMINGS.items()}})
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=1, ensure_ascii=False)
    print("=== MM DOCQA SUMMARY ===")
    for k in sorted(summary):
        if isinstance(summary[k], float):
            print(f"  {k:28s} {summary[k]:.4f}")
    print(f"wrote {args.output_path}")


if __name__ == "__main__":
    main()
