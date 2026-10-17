#!/usr/bin/env python
"""
Rejection-sampling finetuning (RAFT / STaR) data generation for PISCO.

Motivation (DEX-Comp, arXiv 2609.05152): PISCO finetunes on `mistral_label`, the
uncompressed teacher's answer -- including on the ~46% of queries where that answer
does not contain the gold. On those, the current recipe trains the compressed model
to reproduce a wrong answer. This script instead lets the *compressed* model try the
query itself, keeps only the attempts that hit gold, and writes those out as new
training targets.

That is the cheap half of DEX-Comp's "Hard Exploration" stage: same rollouts, same
binary reward, but no policy gradient -- the kept samples are just SFT targets, so
the existing finetune pipeline consumes the output unchanged.

Usage (rollouts):
    python scripts/raft_rollout.py --checkpoint_path <ckpt> \
        --output_path /path/to/raft_data --group_size 8 --limit 20000

Then finetune on the result:
    python pisco/train.py --config-name=finetune \
        ++model_name_or_path=<ckpt> ++data.training_dataset=/path/to/raft_data
"""

import argparse
import collections
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets

from pisco.collator import RolloutCollator
from pisco.model import PISCO, PISCOConfig
from pisco.reward import is_correct, source_of, teacher_is_correct


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    src = p.add_argument_group("model")
    src.add_argument("--checkpoint_path", type=str, default=None,
                     help="PISCO checkpoint to roll out from.")
    src.add_argument("--decoder_model_name", type=str, default=None,
                     help="Build an untrained PISCO instead of loading a checkpoint (smoke tests only).")
    src.add_argument("--compressor_model_name", type=str, default=None)
    src.add_argument("--compr_rate", type=int, default=16)
    src.add_argument("--device", type=str, default=None)
    src.add_argument("--dtype", type=str, default=None, choices=["float32", "bfloat16", "float16"])

    dat = p.add_argument_group("data")
    dat.add_argument("--dataset", type=str, default="maxoul/pisco_finetuning_data")
    dat.add_argument("--split", type=str, default="train")
    dat.add_argument("--output_path", type=str, required=True)
    dat.add_argument("--limit", type=int, default=None,
                     help="Cap the number of teacher-failed queries to roll out on.")
    dat.add_argument("--topk_docs", type=int, default=5)
    dat.add_argument("--compressor_max_length", type=int, default=128)
    dat.add_argument("--decoder_max_length", type=int, default=512)
    dat.add_argument("--query_dependent", action="store_true", help="OSCAR-style rollouts.")
    dat.add_argument("--sources", type=str, default=None,
                     help="Comma-separated q_id prefixes to keep (e.g. msmarco,wikiqa).")
    dat.add_argument("--exclude_sources", type=str, default=None,
                     help="Comma-separated q_id prefixes to drop.")

    gen = p.add_argument_group("sampling")
    gen.add_argument("--group_size", type=int, default=8, help="G rollouts per query.")
    gen.add_argument("--temperature", type=float, default=1.0)
    gen.add_argument("--top_p", type=float, default=0.95)
    gen.add_argument("--max_new_tokens", type=int, default=64)
    gen.add_argument("--batch_size", type=int, default=8, help="Queries per batch (x G sequences).")
    gen.add_argument("--seed", type=int, default=0)

    sco = p.add_argument_group("scoring")
    sco.add_argument("--score_mode", type=str, default="containment",
                     choices=["containment", "f1", "either"])
    sco.add_argument("--f1_threshold", type=float, default=0.6)
    sco.add_argument("--allow_truncated", action="store_true",
                     help="Keep samples cut off at max_new_tokens. Off by default: a truncated "
                          "answer ends mid-word and is a bad SFT target even when it contains gold.")
    sco.add_argument("--include_teacher_correct", action="store_true",
                     help="Also emit the S+ rows unchanged, giving a complete training diet.")
    return p.parse_args()


def load_model(args, device):
    if args.checkpoint_path:
        model = PISCO.from_pretrained(args.checkpoint_path)
    else:
        if not args.decoder_model_name:
            raise SystemExit("need --checkpoint_path or --decoder_model_name")
        cfg = PISCOConfig(
            decoder_model_name=args.decoder_model_name,
            compressor_model_name=args.compressor_model_name or args.decoder_model_name,
            compr_rate=args.compr_rate,
            torch_dtype=args.dtype or ("bfloat16" if device.type == "cuda" else "float32"),
            attn_implementation="sdpa",
        )
        model = PISCO(cfg)
    model.to(device)
    model.eval()
    return model


def clean_completion(text: str) -> str:
    """Drop a leading <think>...</think> block (Qwen3) and surrounding whitespace."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"[raft] loading dataset {args.dataset}")
    ds = (datasets.load_from_disk(args.dataset) if os.path.exists(args.dataset)
          else datasets.load_dataset(args.dataset, split=args.split))
    if isinstance(ds, datasets.DatasetDict):
        ds = ds[args.split]

    score_kw = dict(mode=args.score_mode, f1_threshold=args.f1_threshold)

    print(f"[raft] splitting {len(ds)} rows into teacher-correct / teacher-failed")
    tc_flags = [teacher_is_correct(ex, **score_kw) for ex in ds]
    neg_idx = [i for i, ok in enumerate(tc_flags) if not ok]
    pos_idx = [i for i, ok in enumerate(tc_flags) if ok]
    print(f"[raft]   S+ {len(pos_idx)}  S- {len(neg_idx)}")

    keep = set(args.sources.split(",")) if args.sources else None
    drop = set(args.exclude_sources.split(",")) if args.exclude_sources else set()
    if keep is not None or drop:
        def wanted(i):
            src = source_of(ds[i].get("q_id"))
            return (keep is None or src in keep) and src not in drop
        neg_idx = [i for i in neg_idx if wanted(i)]
        pos_idx = [i for i in pos_idx if wanted(i)]
        print(f"[raft]   after source filter (keep={args.sources} drop={args.exclude_sources}): "
              f"S+ {len(pos_idx)}  S- {len(neg_idx)}")
        per = collections.Counter()
        for i in neg_idx:
            per[source_of(ds[i].get("q_id"))] += 1
        for i in pos_idx:
            per.setdefault(source_of(ds[i].get("q_id")), 0)
        pos_per = collections.Counter(source_of(ds[i].get("q_id")) for i in pos_idx)
        print(f"[raft]   teacher-correct rate under score_mode={args.score_mode}:")
        for src in sorted(per):
            n = per[src] + pos_per[src]
            print(f"[raft]     {src:<16} S+ {pos_per[src]:>6} / {n:<6} = {100*pos_per[src]/max(1,n):5.1f}%")
    if args.limit:
        neg_idx = neg_idx[: args.limit]
        print(f"[raft]   capped S- to {len(neg_idx)}")

    model = load_model(args, device)
    collator = RolloutCollator(
        compressor_tokenizer=model.compressor_tokenizer,
        decoder_tokenizer=model.decoder_tokenizer,
        compr_rate=model.compr_rate,
        compressor_max_length=args.compressor_max_length,
        decoder_max_length=args.decoder_max_length,
        topk_docs=args.topk_docs,
        query_dependent=args.query_dependent,
    )

    G = args.group_size
    kept, n_by, solved_by, hits_by = [], collections.Counter(), collections.Counter(), collections.Counter()
    n_truncated_correct = 0   # correct samples thrown away for being cut off
    eos_id = model.decoder_tokenizer.eos_token_id
    t0 = time.time()

    with torch.inference_mode():
        for start in range(0, len(neg_idx), args.batch_size):
            chunk_idx = neg_idx[start : start + args.batch_size]
            chunk = [ds[i] for i in chunk_idx]
            for ex in chunk:
                ex.setdefault("mistral_label", "")

            batch = collator.torch_call(chunk)
            batch = {k: v.to(device) for k, v in batch.items()}

            outs, out_ids = model.generate(
                batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=G,
                return_ids=True,
            )
            assert len(outs) == len(chunk) * G, f"expected {len(chunk)*G} rollouts, got {len(outs)}"
            # A sample that never emitted eos hit the max_new_tokens wall: it ends
            # mid-word, so it is unusable as a training target however it scores.
            finished = (out_ids == eos_id).any(dim=1).tolist()

            for j, ex in enumerate(chunk):
                src = source_of(ex.get("q_id"))
                n_by[src] += 1
                cands = [clean_completion(c) for c in outs[j * G : (j + 1) * G]]
                done = finished[j * G : (j + 1) * G]
                good_all = [c for c in cands if c and is_correct(c, ex.get("labels"), **score_kw)]
                good = [c for c, fin in zip(cands, done)
                        if c and (fin or args.allow_truncated)
                        and is_correct(c, ex.get("labels"), **score_kw)]
                n_truncated_correct += len(good_all) - len(good)
                hits_by[src] += len(good)
                if not good:
                    continue
                solved_by[src] += 1
                # Prefer the shortest correct sample: the system prompt asks for brevity,
                # and long samples tend to hit gold incidentally inside chatter.
                winner = min(good, key=len)
                kept.append({
                    "q_id": ex.get("q_id"),
                    "query": ex.get("query"),
                    "docs": ex.get("docs"),
                    "labels": ex.get("labels"),
                    "mistral_label": winner,
                    "n_correct": len(good),
                    "group_size": G,
                    "origin": "raft",
                })

            done = start + len(chunk)
            if done % (args.batch_size * 10) == 0 or done >= len(neg_idx):
                rate = done / max(1e-9, time.time() - t0)
                print(f"[raft] {done}/{len(neg_idx)} queries  kept={len(kept)}  {rate:.1f} q/s", flush=True)

    n_q = sum(n_by.values())
    n_solved = sum(solved_by.values())
    print(f"\n[raft] rolled out {n_q} teacher-failed queries at G={G}")
    print(f"[raft] recovered {n_solved} ({100 * n_solved / max(1, n_q):.2f}% pass@{G})")
    if not args.allow_truncated:
        print(f"[raft] dropped {n_truncated_correct} correct-but-truncated samples "
              f"(hit max_new_tokens={args.max_new_tokens}; --allow_truncated keeps them)")
    print(f"\n{'source':<16}{'queries':>9}{'solved':>9}{'pass@G':>9}{'per-sample':>12}")
    for src, n in n_by.most_common():
        print(f"{src:<16}{n:>9}{solved_by[src]:>9}"
              f"{100 * solved_by[src] / n:>8.1f}%{100 * hits_by[src] / (n * G):>11.1f}%")

    out_rows = list(kept)
    if args.include_teacher_correct:
        for i in pos_idx:
            ex = ds[i]
            out_rows.append({
                "q_id": ex.get("q_id"), "query": ex.get("query"), "docs": ex.get("docs"),
                "labels": ex.get("labels"), "mistral_label": ex.get("mistral_label"),
                "n_correct": 0, "group_size": 0, "origin": "teacher",
            })
        print(f"\n[raft] + {len(pos_idx)} teacher-correct rows -> {len(out_rows)} total")

    if not out_rows:
        print("[raft] nothing kept; not writing a dataset")
        return

    out = datasets.Dataset.from_list(out_rows)
    out.save_to_disk(args.output_path)
    stats = {
        "queries": n_q, "solved": n_solved, "group_size": G,
        "score_mode": args.score_mode, "kept": len(kept), "total_rows": len(out_rows),
        "by_source": {s: {"queries": n_by[s], "solved": solved_by[s], "samples_correct": hits_by[s]}
                      for s in n_by},
    }
    with open(os.path.join(args.output_path, "raft_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[raft] wrote {len(out_rows)} rows -> {args.output_path}")


if __name__ == "__main__":
    main()
