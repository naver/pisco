#!/usr/bin/env python3
import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pisco.collator_utils import add_memory_tokens_to_inputs, chunk_list
from pisco.metrics import f1_single, match_single
from pisco.model import PISCO, PISCOConfig


SYSTEM_PROMPT = (
    "You are a helpful assistant. Your task is to extract relevant information from "
    "provided documents and to answer to questions as briefly as possible."
)


def _maybe_mkdir_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _parse_trajectory_jsonl(s: str) -> List[Dict[str, str]]:
    """
    `trajectory` is a string containing newline-separated JSON objects:
    {"role": "...", "content": "..."}\n{"role": "...", "content": "..."}\n...
    """
    msgs: List[Dict[str, str]] = []
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # Some datasets contain minor JSON issues in the `content` field
            # (e.g. `There"s ...` instead of `There's ...`), which breaks line-based JSONL parsing.
            # Heuristic: convert an unescaped quote between two letters into an apostrophe.
            fixed = re.sub(r"([A-Za-z])\"([A-Za-z])", r"\1'\2", line)
            obj = json.loads(fixed)
        role = str(obj.get("role", ""))
        content = str(obj.get("content", ""))
        msgs.append({"role": role, "content": content})
    return msgs


def _trajectory_to_text(msgs: List[Dict[str, str]], *, max_chars: Optional[int]) -> str:
    parts: List[str] = []
    for m in msgs:
        role = m.get("role", "").strip().lower()
        content = m.get("content", "")
        if role == "user":
            prefix = "User"
        elif role == "assistant":
            prefix = "Assistant"
        elif role == "system":
            prefix = "System"
        else:
            prefix = role or "Unknown"
        parts.append(f"{prefix}: {content}")
    text = "\n".join(parts)
    if max_chars is not None and len(text) > max_chars:
        text = text[-max_chars:]
    return text


def _build_chat_prompt(tokenizer, *, background: str, question: str) -> str:
    user_prompt = f"\n\nBackground:{background}\n Question: {question}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    chat_template_kwargs = {"enable_thinking": False}

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True, 
            **chat_template_kwargs
    )

    # Fallback for tokenizers without chat templates
    return f"{SYSTEM_PROMPT}\n\n{user_prompt}\nAnswer:"


def _count_mem_tokens(input_ids: torch.Tensor, mem_token_id: int) -> int:
    return int((input_ids == mem_token_id).sum().item())


def _adjust_compressor_chunks_to_target_mems(
    compressor_chunks: List[List[int]],
    *,
    mem_token_id: int,
    target_total_mems: int,
) -> List[List[int]]:
    """
    Ensure the compressor side contains exactly `target_total_mems` occurrences of `mem_token_id`.
    We only trim from the end, and only remove `<MEM>` tokens (which are always appended at end of chunks).
    """
    if target_total_mems < 0:
        raise ValueError("target_total_mems must be >= 0")

    def chunk_mem_count(chunk: List[int]) -> int:
        return int(sum(1 for x in chunk if x == mem_token_id))

    total = sum(chunk_mem_count(c) for c in compressor_chunks)
    if total == target_total_mems:
        return compressor_chunks
    if total < target_total_mems:
        raise ValueError(f"Decoder expects {target_total_mems} mems, compressor has only {total}.")

    out = [list(c) for c in compressor_chunks]
    while out and total > target_total_mems:
        last = out[-1]
        last_mems = chunk_mem_count(last)
        if total - last_mems >= target_total_mems:
            out.pop()
            total -= last_mems
            continue

        # Need to remove only a subset of mem tokens from the end of the last chunk
        to_remove = total - target_total_mems
        removed = 0
        i = len(last) - 1
        while i >= 0 and removed < to_remove:
            if last[i] == mem_token_id:
                last.pop(i)
                removed += 1
            i -= 1
        total -= removed
        out[-1] = last

    if total != target_total_mems:
        raise RuntimeError(f"Failed to adjust mems: got {total}, expected {target_total_mems}")
    return out


@dataclass
class EvalResult:
    question: str
    trajectory: str
    ground_truth: str
    prediction: str
    match: float
    f1: float


@torch.inference_mode()
def _generate_with_pisco(
    model: PISCO,
    *,
    trajectory_text: str,
    question: str,
    device: torch.device,
    compressor_max_length: int,
    decoder_max_length: int,
    max_new_tokens: int,
) -> str:
    # Compressor side: tokenize trajectory, chunk, append MEMs, then pad
    compressor_tok = model.compressor_tokenizer
    decoder_tok = model.decoder_tokenizer

    comp_ids: List[int] = compressor_tok(
        trajectory_text, add_special_tokens=False, truncation=False
    )["input_ids"]
    chunks = chunk_list(comp_ids, chunk_length=compressor_max_length, chunk_overlap=0)
    chunks_with_mems, n_mems = add_memory_tokens_to_inputs(
        chunks, compressor_tok, model.compr_rate
    )
    total_mems = int(sum(n_mems))

    background = decoder_tok.mem_token * total_mems
    prompt = _build_chat_prompt(decoder_tok, background=background, question=question)

    decoder_inputs = decoder_tok(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        max_length=decoder_max_length,
        truncation=True,
        padding=False,
    )
    decoder_input_ids = decoder_inputs["input_ids"]
    decoder_attention_mask = decoder_inputs["attention_mask"]

    # If truncation removed some MEM tokens, trim compressor side to match.
    mem_token_id = decoder_tok.mem_token_id
    decoder_mems = _count_mem_tokens(decoder_input_ids, mem_token_id)
    if decoder_mems != total_mems:
        chunks_with_mems = _adjust_compressor_chunks_to_target_mems(
            chunks_with_mems, mem_token_id=compressor_tok.mem_token_id, target_total_mems=decoder_mems
        )
        total_mems = decoder_mems

    compressor_inputs = compressor_tok.pad(
        {"input_ids": chunks_with_mems}, padding="longest", return_tensors="pt"
    )

    compressor_input_ids = compressor_inputs["input_ids"].to(device)
    compressor_attention_mask = compressor_inputs["attention_mask"].to(device)
    decoder_input_ids = decoder_input_ids.to(device)
    decoder_attention_mask = decoder_attention_mask.to(device)

    # Compression + embedding replacement
    embeddings = model.compress(compressor_input_ids, compressor_attention_mask)
    dec_inputs_embeds = model.replace_embeddings(embeddings, decoder_input_ids)

    output_ids = model.decoder.generate(
        inputs_embeds=dec_inputs_embeds,
        attention_mask=decoder_attention_mask,
        do_sample=False,
        top_p=None,
        max_new_tokens=max_new_tokens,
    )
    decoded = decoder_tok.batch_decode(output_ids, skip_special_tokens=True)[0]
    return decoded


@torch.inference_mode()
def _generate_with_base_decoder(
    model,
    tokenizer,
    *,
    trajectory_text: str,
    question: str,
    device: torch.device,
    max_new_tokens: int,
    decoder_max_length: int,
) -> str:
    background = "\n\n" + trajectory_text
    prompt = _build_chat_prompt(tokenizer, background=background, question=question)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        max_length=decoder_max_length,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output_ids = model.generate(
        **inputs,
        do_sample=False,
        top_p=None,
        max_new_tokens=max_new_tokens,
    )
    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return decoded


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PISCO checkpoint on agentic QA data.")
    parser.add_argument("--data_path", type=str, default="/beegfs/scratch/user/rdeffaye/pisco/agentQA.json")
    parser.add_argument("--output_path", type=str, default="outputs/eval_agent_qa.json")

    parser.add_argument("--mode", type=str, choices=["pisco", "base"], required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to PISCO checkpoint (required for pisco mode; used to infer base decoder model name in base mode).")
    parser.add_argument("--base_model_name", type=str, default=None, help="Override base decoder model name (base mode only).")

    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples.")
    parser.add_argument("--batch_size", type=int, default=1, help="Currently only batch_size=1 is supported.")

    parser.add_argument("--compressor_max_length", type=int, default=128)
    parser.add_argument("--decoder_max_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--trajectory_max_chars", type=int, default=None, help="If set, keep only the last N characters of the trajectory text (helps fit context).")

    parser.add_argument("--device", type=str, default=None, help="e.g. cuda, cuda:0, cpu. Default: cuda if available else cpu.")
    parser.add_argument("--attn_implementation", type=str, default=None, help="Override attention implementation (PISCO mode).")
    args = parser.parse_args()

    if args.batch_size != 1:
        raise ValueError("Only --batch_size=1 is supported for now.")

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    data = _load_json(args.data_path)
    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array.")
    if args.limit is not None:
        data = data[: args.limit]
    typed_data: List[Dict[str, Any]] = []
    for ex in data:
        if not isinstance(ex, dict):
            raise ValueError(f"Expected each dataset entry to be an object, got {type(ex)}")
        typed_data.append(ex)
    data = typed_data

    if args.mode == "pisco" and not args.checkpoint_path:
        raise ValueError("--checkpoint_path is required for --mode pisco")

    pisco_model: Optional[PISCO] = None
    base_model: Optional[AutoModelForCausalLM] = None
    base_tok: Any = None

    if args.mode == "pisco":
        pisco_model = PISCO.from_pretrained(
            args.checkpoint_path,
            load_decoder=True,
            attn_implementation=args.attn_implementation,
        )
        pisco_model.to(device)
        pisco_model.eval()
    else:
        if args.base_model_name:
            decoder_model_name = args.base_model_name
        else:
            if not args.checkpoint_path:
                raise ValueError(
                    "In base mode you must provide --checkpoint_path (to infer decoder model name) "
                    "or specify --base_model_name explicitly."
                )
            cfg = PISCOConfig.from_pretrained(args.checkpoint_path)
            decoder_model_name = cfg.decoder_model_name

        base_tok = cast(
            Any,
            AutoTokenizer.from_pretrained(decoder_model_name, padding_side="left"),
        )
        if base_tok.pad_token_id is None:
            if base_tok.eos_token_id is None:
                raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")
            base_tok.pad_token_id = int(base_tok.eos_token_id)

        base_model = AutoModelForCausalLM.from_pretrained(
            decoder_model_name,
            dtype=torch.bfloat16 if device.type == "cuda" else None,
            device_map=None,
        )
        assert base_model is not None
        base_model.to(device)  # type: ignore[arg-type]
        base_model.eval()

    results: List[EvalResult] = []

    for i, ex in enumerate(data):
        question = str(ex.get("question", ""))
        ground_truth = str(ex.get("ground_truth", ""))
        traj_raw = str(ex.get("trajectory", ""))

        # msgs = _parse_trajectory_jsonl(traj_raw)
        # trajectory_text = _trajectory_to_text(msgs, max_chars=args.trajectory_max_chars)

        if args.mode == "pisco":
            assert pisco_model is not None
            pred = _generate_with_pisco(
                pisco_model,
                trajectory_text=traj_raw,
                question=question,
                device=device,
                compressor_max_length=args.compressor_max_length,
                decoder_max_length=args.decoder_max_length,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            assert base_model is not None and base_tok is not None
            pred = _generate_with_base_decoder(
                base_model,
                base_tok,
                trajectory_text=traj_raw,
                question=question,
                device=device,
                max_new_tokens=args.max_new_tokens,
                decoder_max_length=args.decoder_max_length,
            )

        m = float(match_single(pred, ground_truth))
        f1, _, _ = f1_single(pred, ground_truth)
        f1 = float(f1)

        results.append(
            EvalResult(
                question=question,
                trajectory=traj_raw,
                ground_truth=ground_truth,
                prediction=pred,
                match=m,
                f1=f1,
            )
        )

        if (i + 1) % 10 == 0 or (i + 1) == len(data):
            avg_match = sum(r.match for r in results) / len(results)
            avg_f1 = sum(r.f1 for r in results) / len(results)
            print(f"[{i+1}/{len(data)}] match={avg_match:.4f} f1={avg_f1:.4f}")

    avg_match = sum(r.match for r in results) / max(1, len(results))
    avg_f1 = sum(r.f1 for r in results) / max(1, len(results))

    payload = {
        "mode": args.mode,
        "data_path": args.data_path,
        "checkpoint_path": args.checkpoint_path,
        "base_model_name": args.base_model_name,
        "compressor_max_length": args.compressor_max_length,
        "decoder_max_length": args.decoder_max_length,
        "max_new_tokens": args.max_new_tokens,
        "trajectory_max_chars": args.trajectory_max_chars,
        "n_examples": len(results),
        "metrics": {"match": avg_match, "f1": avg_f1},
        "samples": [
            {
                "question": r.question,
                "ground_truth": r.ground_truth,
                "prediction": r.prediction,
                "match": r.match,
                "f1": r.f1,
            }
            for r in results
        ],
    }

    _maybe_mkdir_parent(args.output_path)
    with open(args.output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("\nSummary")
    print(f"- n_examples: {len(results)}")
    print(f"- match: {avg_match:.4f}")
    print(f"- f1: {avg_f1:.4f}")
    print(f"- wrote: {args.output_path}")


if __name__ == "__main__":
    main()

