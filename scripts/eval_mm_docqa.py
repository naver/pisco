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

from pisco.metrics import anls_single, em_single, f1_single, match_single, normalize  # noqa: E402
from pisco.model import PISCO, PISCOConfig, merge_lora_inplace  # noqa: E402
from pisco.mm_utils import collate_compressor, get_vision_token_ids, image_to_span, segments_to_chunks  # noqa: E402
from eval_agent_qa import SYSTEM_PROMPT, _adjust_compressor_chunks_to_target_mems, _count_mem_tokens  # noqa: E402
from pisco.collator import FineTuningCollator  # noqa: E402

_FT_DEFAULTS = {k: v.default for k, v in __import__("inspect").signature(FineTuningCollator.__init__).parameters.items()
                if k in ("system_prompt", "user_prompt")}


def _build_chat_prompt(tokenizer, *, background: str, question: str) -> str:
    """EXACTLY the finetuning prompt (FineTuningCollator defaults), as a generation prompt.
    eval_agent_qa's variant inserts a newline after 'Background:' that training never saw."""
    messages = [
        {"role": "system", "content": _FT_DEFAULTS["system_prompt"]},
        {"role": "user", "content": _FT_DEFAULTS["user_prompt"].replace("[documents]", background).replace("[question]", question)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

TIMINGS = defaultdict(list)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class _Pad:
    def __init__(self, tok, proc):
        self.tok, self.image_processor = tok, proc

    def compressor_pad(self, ids):
        return self.tok.pad({"input_ids": ids}, padding="longest", return_tensors="pt")


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
    answers = _flatten_answers(answers)
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


@torch.inference_mode()
def gen_pisco(model: PISCO, row, *, device, topk, compressor_max_length, decoder_max_length, max_new_tokens,
              image_compr_rate, text_only=False, num_samples=0, temperature=1.0, top_p=0.95):
    tok, dtok = model.compressor_tokenizer, model.decoder_tokenizer
    vision_ids = model.vision_token_ids or get_vision_token_ids(tok)
    pad = _Pad(tok, model.image_processor)
    docs = list(row["docs"])[:topk]
    imgs = list(row.get("doc_images") or [])[:topk]
    segs = []
    for j, d in enumerate(docs):
        img = imgs[j] if j < len(imgs) else None
        if img is not None and not text_only:
            segs.append(("image", img))
        else:
            segs.append(("text", d if d else " "))
    from pisco.collator_utils import chunk_list
    chunker = lambda ids: chunk_list(ids, chunk_length=compressor_max_length, chunk_overlap=0) or [ids]  # noqa: E731
    # one "Document j:" header per segment (a text doc may span several chunks), built the
    # same way as MultimodalFineTuningCollator: one segments_to_chunks call per document.
    from pisco.mm_utils import MMChunkBatch
    batch, background = MMChunkBatch(), ""
    for j, seg in enumerate(segs):
        b = segments_to_chunks([seg], compressor_tokenizer=tok, image_processor=model.image_processor,
                               compr_rate=model.compr_rate, image_compr_rate=image_compr_rate,
                               text_chunker=chunker, vision_ids=vision_ids)
        batch.extend(b)
        background += f"Document {j}:" + dtok.mem_token * b.total_mems
    prompt = _build_chat_prompt(dtok, background=background, question=row["query"])
    dec = dtok(prompt, return_tensors="pt", add_special_tokens=False, max_length=decoder_max_length,
               truncation=True, padding=False)
    total = batch.total_mems
    got = _count_mem_tokens(dec["input_ids"], dtok.mem_token_id)
    if got != total:
        batch.input_ids = _adjust_compressor_chunks_to_target_mems(batch.input_ids, mem_token_id=tok.mem_token_id,
                                                                   target_total_mems=got)
    comp = collate_compressor([batch], pad, vision_ids)
    comp = {k: v.to(device) for k, v in comp.items()}
    _sync(); t = time.perf_counter()
    emb = model.compress(comp["compressor_input_ids"], comp["compressor_attention_mask"],
                         pixel_values=comp.get("pixel_values"), image_grid_thw=comp.get("image_grid_thw"),
                         mm_token_type_ids=comp.get("mm_token_type_ids"))
    _sync(); TIMINGS["compress"].append(time.perf_counter() - t)
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
    stats = {"mems_used": got, "mems_trimmed": total - got, "n_images": int(sum(batch.is_image)),
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


@torch.inference_mode()
def gen_base(base, proc, tok, row, *, device, topk, max_new_tokens, decoder_max_length):
    cfg = base.config
    vision_ids = (int(cfg.vision_start_token_id), int(cfg.image_token_id), int(cfg.vision_end_token_id))
    vs, ip, ve = vision_ids
    docs = list(row["docs"])[:topk]
    imgs = list(row.get("doc_images") or [])[:topk]
    background, pvs, grids = "", [], []
    for j, d in enumerate(docs):
        img = imgs[j] if j < len(imgs) else None
        if img is not None:
            span, pv, grid = image_to_span(img, proc, vision_ids)
            pvs.append(pv); grids.append(grid)
            background += f"Document {j}:" + tok.convert_ids_to_tokens(vs) + tok.convert_ids_to_tokens(ip) * (len(span) - 2) + tok.convert_ids_to_tokens(ve)
        else:
            background += f"Document {j}:" + (d or "")
    prompt = _build_chat_prompt(tok, background=background, question=row["query"])
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"]
    n_pad_tokens = int((ids == ip).sum())
    expected = sum(len(p) // 4 for p in pvs)
    assert n_pad_tokens == expected, f"template altered the image_pad run: {n_pad_tokens} vs {expected}"
    kwargs = dict(input_ids=ids.to(device), attention_mask=enc["attention_mask"].to(device))
    if pvs:
        kwargs.update(pixel_values=torch.from_numpy(np.concatenate(pvs, 0)).to(device),
                      image_grid_thw=torch.tensor(grids, dtype=torch.long, device=device),
                      mm_token_type_ids=(ids == ip).long().to(device))
    _sync(); t = time.perf_counter()
    eos = [i for i in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
           if isinstance(i, int) and i >= 0 and i != tok.unk_token_id]
    out = base.generate(**kwargs, do_sample=False, top_p=None, max_new_tokens=max_new_tokens, eos_token_id=eos)
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
    ap.add_argument("--image_max_pixels", type=int, default=451584)
    ap.add_argument("--merge_lora", action="store_true")
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
        model = PISCO.from_pretrained(args.checkpoint_path, image_max_pixels=args.image_max_pixels)
        model.to(device).eval()
        if args.merge_lora:   # fold LoRA into the weights: same outputs, ~1.8x faster per decoded token
            model.decoder = merge_lora_inplace(model.decoder)
            model.compressor = merge_lora_inplace(model.compressor)
        assert model.image_processor is not None or args.mode == "pisco_text", "checkpoint has no image processor"
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
        proc = load_image_processor(name, max_pixels=args.image_max_pixels)
        print(f"[base] {name}")

    records, agg = [], defaultdict(list)
    t0 = time.time()
    for i, row in enumerate(ds):
        if args.mode == "base":
            pred, stats = gen_base(base, proc, tok, row, device=device, topk=args.topk_docs,
                                   max_new_tokens=args.max_new_tokens, decoder_max_length=args.decoder_max_length)
        else:
            pred, stats = gen_pisco(model, row, device=device, topk=args.topk_docs,
                                    compressor_max_length=args.compressor_max_length,
                                    decoder_max_length=args.decoder_max_length, max_new_tokens=args.max_new_tokens,
                                    image_compr_rate=args.image_compr_rate, text_only=(args.mode == "pisco_text"),
                                    num_samples=args.num_samples, temperature=args.temperature, top_p=args.top_p)
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
                    "image_max_pixels": args.image_max_pixels, "num_samples": args.num_samples,
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
