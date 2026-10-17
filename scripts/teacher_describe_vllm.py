"""Teacher descriptions for stage-2 `desc` targets: the UNCOMPRESSED Qwen3.5-4B describes each image (vLLM env).
The prompt is built from pisco/mm_tasks.py (pure Python) exactly as MultimodalPretrainingCollator builds the `desc`
chat for one image: system DESC_SYSTEM_PROMPT, user "Background:Document 0:<image> Question: Describe Document 0 in
detail.", generation prompt with an empty think block (enable_thinking=False). The full training chat renders as
that generation prompt + answer + <|im_end|> (checked with the real tokenizer), so the text written here is a
valid target AND the KL teacher's prompt matches it token for token.

Rows [start, end) of a datasets dir; output JSONL {"idx", "teacher_text", "n_tokens", "finish"}; resumable
(indices already in --out are skipped); images are processed in blocks so 1M-px pages never pile up in RAM.
  python scripts/teacher_describe_vllm.py --data data/mm_pretrain_natural --start 0 --end 25000 \
      --out data/teacher_desc/natural_00000.jsonl
"""
import argparse
import json
import os
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pisco import mm_tasks  # noqa: E402  (pure Python)

VISION_SPAN = "<|vision_start|><|image_pad|><|vision_end|>"   # one pad per image: vLLM expands it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--block", type=int, default=1024, help="rows per llm.generate call")
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_mem", type=float, default=0.88)
    args = ap.parse_args()

    import datasets
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    ds = datasets.load_from_disk(args.data)
    ds = ds["train"] if isinstance(ds, datasets.DatasetDict) else ds
    end = min(args.end if args.end is not None else len(ds), len(ds))
    done = set()
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        for line in open(out_path):
            try:
                done.add(json.loads(line)["idx"])
            except Exception:
                pass
    todo = [i for i in range(args.start, end) if i not in done]
    print(f"[teacher] {args.data} rows [{args.start},{end}): {len(todo)} to describe ({len(done)} already done)", flush=True)
    if not todo:
        return 0

    tok = AutoTokenizer.from_pretrained(args.model)
    msgs = [{"role": "system", "content": mm_tasks.DESC_SYSTEM_PROMPT},
            {"role": "user", "content": mm_tasks.user_content("Document 0:" + VISION_SPAN, mm_tasks.desc_query(0))}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    print("[teacher] prompt:", repr(prompt), flush=True)

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
              limit_mm_per_prompt={"image": 1}, mm_processor_cache_gb=0, enable_prefix_caching=False, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    t0, n_done = time.time(), 0
    with open(out_path, "a") as f:
        for b in range(0, len(todo), args.block):
            idx = todo[b : b + args.block]
            reqs = [{"prompt": prompt, "multi_modal_data": {"image": ds[i]["image"].convert("RGB")}} for i in idx]
            outs = llm.generate(reqs, sp, use_tqdm=False)
            for i, o in zip(idx, outs):
                c = o.outputs[0]
                text = c.text.replace("<think>", "").replace("</think>", "").strip()
                f.write(json.dumps({"idx": i, "teacher_text": text, "n_tokens": len(c.token_ids),
                                    "finish": c.finish_reason}) + "\n")
            f.flush()
            n_done += len(idx)
            dt = time.time() - t0
            print(f"[teacher] {n_done}/{len(todo)} in {dt:.0f}s ({n_done / dt:.1f} img/s); e.g. "
                  f"{outs[0].outputs[0].text[:120]!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
