"""Stage B test: train the READER only (decoder LoRA; compressor and connector frozen) on cached <MEM>s of hour-long
videos, in the hierarchy's layouts. Rows (videos of the training folds only; fold(v) = index of v in the sorted video
ids % nfolds, the held-out fold is --fold):
  LVBench MCQ (data/video/lvbench_qa):  layouts both / fine / coarse (fine only for local questions), candidates =
      CLIP top-3 (cands.json) + the gold minutes, answer = the letter.
  ExtremeWhenBench queries (expQ/video/ewb_test.json): "which minute" pointing rows, layouts both / fine, candidates =
      CLIP top-3 + the gold minute, answer = "Document <minute>" (the find/route chat of eval_mm_docqa.route_logprobs).
Layouts: coarse = every minute as 2 frames (t 15 s, 45 s); fine = the candidate minutes, 12 frames, nothing else;
both = every minute coarse, the candidates dense in place. Labels "Document <minute>" everywhere.
One forward per row (inputs_embeds via PISCO.replace_embeddings), loss on the answer tokens only, grad accumulation,
AdamW + cosine. Saves a PISCO checkpoint (save_pretrained) to --out/model; evaluate it with mcq_levels.py --eval_fold.

  python scripts/video/train_reader.py --checkpoint_path expQ/ft_mm_4b_174908/model --fold 0 --out expQ/video/reader_f0
"""

import argparse
import json
import math
import pathlib
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cache_mems import load_cache  # noqa: E402
from mcq_levels import gold_minutes, load_questions  # noqa: E402
from route_windows import load_pisco  # noqa: E402
from zoom_route import FPM, l0_offsets  # noqa: E402
from eval_mm_docqa import ROUTE_QUESTION, _build_chat_prompt, _reset_rope_deltas  # noqa: E402
from pisco import mm_tasks  # noqa: E402


def ewb_rows(path, n_min):
    rows = []
    for r in json.load(open(path)):
        if r["video_id"] not in n_min:
            continue
        s, e = r["correct_interval"]
        gold = [m for m in range(n_min[r["video_id"]]) if m * 60 < max(e, s + 1) and (m + 1) * 60 > s]
        rows.append({"uid": r["qid"], "video_id": r["video_id"], "question": r["question"], "gold": gold, "kind": "ewb"})
    return rows


def build_rows(args, qs, n_min, cands):
    allv = sorted(n_min)
    train_v = {v for v in allv if allv.index(v) % args.nfolds != args.fold}
    rows = []
    for q in qs:
        if q["video_id"] not in train_v:
            continue
        nm = n_min[q["video_id"]]
        gold = gold_minutes(q["ref"], nm)
        top = [m for m in cands[q["uid"]] if m < nm][: args.top]
        cand = sorted(set(top) | set(gold if len(gold) <= args.top else []))
        layouts = ["both", "coarse"] + (["fine"] if q["local"] else [])
        for lay in layouts:
            rows.append({"video_id": q["video_id"], "kind": "mcq", "layout": lay, "cands": cand,
                         "question": q["question"] + "\nAnswer with the letter of the correct option.",
                         "answer": q["answer"]})
    for r in ewb_rows(args.ewb, n_min):
        if r["video_id"] not in train_v or not r["gold"]:
            continue
        nm = n_min[r["video_id"]]
        top = [m for m in cands[r["uid"]] if m < nm][: args.top]
        g = r["gold"][0]
        cand = sorted(set(top) | {g})
        for lay in ("both", "fine"):
            rows.append({"video_id": r["video_id"], "kind": "ewb", "layout": lay, "cands": cand,
                         "question": r["question"], "answer": mm_tasks.find_answer(g)})
    return rows


def build_example(model, coll, row, emb, nm, offs, device):
    """(input_ids, labels, mems) for one row."""
    dtok = model.decoder_tokenizer
    mem = dtok.mem_token
    dense = lambda m: list(range(m * FPM, m * FPM + FPM))
    sparse = lambda m: [m * FPM + o for o in offs]
    cset = set(row["cands"])
    if row["layout"] == "fine":
        minutes, frames_of = row["cands"], dense
    elif row["layout"] == "coarse":
        minutes, frames_of = range(nm), sparse
    else:
        minutes, frames_of = range(nm), (lambda m: dense(m) if m in cset else sparse(m))
    bg, parts = "", []
    for m in minutes:
        ks = frames_of(m)
        bg += f"Document {m}:" + mem * sum(emb[k].shape[0] for k in ks)
        parts.extend(emb[k] for k in ks)
    question = coll.clean_text(row["question"])
    if row["kind"] == "mcq":
        prompt = _build_chat_prompt(dtok, background=bg, question=question)
    else:
        msgs = [{"role": "system", "content": mm_tasks.FIND_SYSTEM_PROMPT},
                {"role": "user", "content": mm_tasks.user_content(bg, ROUTE_QUESTION.format(q=question))}]
        prompt = dtok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    p_ids = dtok(prompt, add_special_tokens=False)["input_ids"]
    full = dtok(prompt + row["answer"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    assert full[: len(p_ids)] == p_ids, "the answer re-tokenizes the prompt"
    ids = torch.tensor([full], device=device)
    labels = torch.full_like(ids, -100)
    labels[0, len(p_ids):] = ids[0, len(p_ids):]
    mems = torch.cat(parts)
    assert int((ids == dtok.mem_token_id).sum()) == mems.shape[0]
    return ids, labels, mems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames_dir", default="data/video/lvbench_frames")
    ap.add_argument("--cache_dir", default="data/video/lvbench_cache")
    ap.add_argument("--ewb", default="expQ/video/ewb_test.json")
    ap.add_argument("--image_compr_rate", type=int, default=16)
    ap.add_argument("--image_max_pixels", type=int, default=451584)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--nfolds", type=int, default=3)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--l0_frames", type=int, default=2)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--max_tokens", type=int, default=7000)
    ap.add_argument("--limit", type=int, default=None, help="rows per epoch (smoke)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda"
    qs, n_min = load_questions(f"{args.frames_dir}/frames.json")
    cands = json.load(open(f"{args.cache_dir}/cands.json"))
    rows = build_rows(args, qs, n_min, cands)
    random.shuffle(rows)
    rows = rows[: args.limit]
    kinds = {}
    for r in rows:
        kinds[(r["kind"], r["layout"])] = kinds.get((r["kind"], r["layout"]), 0) + 1
    print(f"[reader] fold {args.fold}/{args.nfolds}: {len(rows)} rows/epoch {dict(sorted(kinds))}", flush=True)

    model, coll = load_pisco(args, device)
    for p in model.compressor.parameters():
        p.requires_grad_(False)
    for p in model.connector.parameters():
        p.requires_grad_(False)
    params = [p for p in model.decoder.parameters() if p.requires_grad]
    print(f"[reader] trainable decoder params: {sum(p.numel() for p in params) / 1e6:.1f} M", flush=True)
    model.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.decoder.train()
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    total = math.ceil(len(rows) * args.epochs / args.accum)
    warm = max(1, int(total * args.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    offs = l0_offsets(args.l0_frames)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.jsonl", "a")
    step, seen, t0, losses, skipped = 0, 0, time.time(), [], 0
    cache_v, emb = None, None
    n_rows = int(len(rows) * args.epochs)
    order = []
    while len(order) < n_rows:   # epochs, each a fresh shuffle, grouped by video for cache reuse
        ep = rows[:]
        random.shuffle(ep)
        ep.sort(key=lambda r: r["video_id"])   # stable: random within a video, videos in order
        order.extend(ep)
    order = order[:n_rows]
    for i, row in enumerate(order):
        if row["video_id"] != cache_v:
            emb, cache_v = load_cache(args.cache_dir, row["video_id"], device), row["video_id"]
        ids, labels, mems = build_example(model, coll, row, emb, n_min[row["video_id"]], offs, device)
        if ids.shape[1] > args.max_tokens:
            skipped += 1
            continue
        _reset_rope_deltas(model.decoder)
        emb_in = model.replace_embeddings(mems, ids)
        out_ = model.decoder(inputs_embeds=emb_in, attention_mask=torch.ones_like(ids), labels=labels)
        (out_.loss / args.accum).backward()
        losses.append(float(out_.loss))
        seen += 1
        if seen % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == total:
                rec = {"step": step, "of": total, "loss": float(np.mean(losses[-args.accum * 10:])), "lr": sched.get_last_lr()[0],
                       "rows": seen, "skipped": skipped, "s": round(time.time() - t0)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n"); log.flush()
    model.decoder.eval()
    model.save_pretrained(str(out / "model"))
    json.dump({"args": vars(args), "rows_per_epoch": len(rows), "kinds": {f"{k[0]}/{k[1]}": v for k, v in kinds.items()},
               "steps": step, "skipped": skipped, "final_loss": float(np.mean(losses[-50:])) if losses else None},
              open(out / "train_summary.json", "w"), indent=1)
    print(f"[reader] saved {out / 'model'}; {step} steps, {skipped} rows skipped (> {args.max_tokens} tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
