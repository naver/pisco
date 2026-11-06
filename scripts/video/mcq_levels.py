"""Does the whole-video coarse layer help the reader? LVBench multiple-choice questions on the 67 EWB videos
(data/video/lvbench_qa, 1,026 Q), answered from <MEM>s of the video's frames under five contexts:
  coarse     every minute as `--l0_frames` frames (default 2 of the 12 at 5 s spacing): the whole video in one context
  fine       the top-`--top` minutes of the model's own level-0 routing, all 12 frames, nothing else
  both       every minute coarse, the top minutes dense IN PLACE (same "Document <minute>" labels)
  both_gold  every minute coarse, the minutes of LVBench's time_reference dense (oracle pick; only when <= --top minutes)
  (coarse is also the uniform-sampling baseline: 1 frame / 30 s)
  blind      no video at all (question only): the answer-prior control. --only_arms blind skips the index and routing.
With --asr_dir (scripts/video/transcribe.py output), the transcript layer: each minute's speech (segments by midpoint)
is a text document compressed to <MEM>s and placed in the minute's slot after its frames, same "Document <minute>" label:
  asr         transcript only (minutes with speech)
  coarse_asr  coarse frames + transcript per minute
  both_asr    both + transcript per minute
The answer is scored by log p(letter) over the question's options in one forward (no generation). A question is
"local" when its time_reference spans <= 120 s, "global" otherwise (summarization, reasoning, long events).
Shard by video: --shard i --nshards n. --summarize merges shard files and prints accuracy per arm x locality.
--cache_dir reads the per-frame <MEM>s from scripts/video/cache_mems.py instead of compressing; --cands uses CLIP's
ranked minutes (cands.json) as the candidates instead of the model's own routing (retriever searches, model reads).
--eval_fold k --nfolds 3 keeps only the videos of fold k (sorted video ids, index % nfolds), the held-out videos of
the reader trained by scripts/video/train_reader.py --fold k.

  python scripts/video/mcq_levels.py --checkpoint_path expQ/ft_mm_4b_174908/model --image_compr_rate 16 \
      --shard 0 --nshards 6 --out expQ/video/results/mcq_174908_r16.s0.json
  python scripts/video/mcq_levels.py --summarize expQ/video/results/mcq_174908_r16.s*.json
"""

import argparse
import glob
import json
import pathlib
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from route_windows import compress_frames, load_pisco, route_logprobs  # noqa: E402
from zoom_route import FPM, l0_offsets  # noqa: E402
from eval_mm_docqa import _build_chat_prompt, _reset_rope_deltas  # noqa: E402

QA = "data/video/lvbench_qa/data/train-00000-of-00001.parquet"
TS = re.compile(r"(\d{1,3}):(\d{2})(?::(\d{2}))?")
OPT = re.compile(r"^\(([A-F])\)\s*(.*)$")


def seconds(m):
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(s) if s else int(h) * 60 + int(mi)


def parse_ref(ref):
    """time_reference 'MM:SS-MM:SS' (minutes may exceed 59) -> (start_s, end_s) or None."""
    ts = [seconds(m) for m in TS.finditer(str(ref))]
    return (min(ts[0], ts[-1]), max(ts[0], ts[-1])) if len(ts) >= 2 else None


def parse_question(text):
    stem, letters = [], []
    for line in text.split("\n"):
        m = OPT.match(line.strip())
        (letters if m else stem).append(m.group(1) if m else line)
    return "\n".join(stem).strip(), letters


def load_questions(frames_json):
    import pandas as pd
    n_min = {v: n // FPM for v, n in json.load(open(frames_json)).items()}
    df = pd.read_parquet(QA)
    df = df[df.key.isin(n_min)]
    qs = []
    for r in df.itertuples():
        stem, letters = parse_question(r.question)
        if len(letters) < 2 or r.answer not in letters:
            continue
        ref = parse_ref(r.time_reference)
        qs.append({"uid": r.uid, "video_id": r.key, "question": r.question, "stem": stem, "letters": letters,
                   "answer": r.answer, "qtype": "|".join(r.question_type), "ref": ref,
                   "span_s": (ref[1] - ref[0]) if ref else None,
                   "local": bool(ref and ref[1] - ref[0] <= 120), "n_minutes": n_min[r.key]})
    return qs, n_min


def minute_texts(asr_dir, vid, nm):
    """{minute: transcript text} from the video's Whisper segments (a segment belongs to the minute of its midpoint)."""
    p = pathlib.Path(asr_dir) / f"{vid}.json"
    if not p.exists():
        return {}
    out = {}
    for seg in json.load(open(p))["segments"]:
        m = int(((seg["start"] + seg["end"]) / 2) // 60)
        if 0 <= m < nm:
            out[m] = (out.get(m, "") + " " + seg["text"]).strip()
    return out


@torch.inference_mode()
def compress_texts(model, coll, texts, batch, device):
    """[<MEM> embeddings per text document] (each text may span several compressor chunks)."""
    from pisco.mm_utils import MMChunkBatch, collate_compressor
    out = []
    for i in range(0, len(texts), batch):
        part, full, sizes = texts[i:i + batch], MMChunkBatch(), []
        for t in part:
            b = coll.example_chunks([t], [], None)[0]
            full.extend(b)
            sizes.append(b.total_mems)
        comp = {n: v.to(device) for n, v in collate_compressor([full], coll, coll.vision_ids).items()}
        e = model.compress(comp["compressor_input_ids"], comp["compressor_attention_mask"])
        offs = np.cumsum([0] + sizes)
        out.extend(e[a:z] for a, z in zip(offs[:-1], offs[1:]))
    return out


def gold_minutes(ref, nm):
    if not ref:
        return []
    s, e = ref
    return [m for m in range(nm) if m * 60 < max(e, s + 1) and (m + 1) * 60 > s]


@torch.inference_mode()
def letter_logprobs(model, bg, embcat, question, letters, device):
    dtok = model.decoder_tokenizer
    prompt = _build_chat_prompt(dtok, background=bg, question=question + "\nAnswer with the letter of the correct option.")
    ids = torch.tensor([dtok(prompt, add_special_tokens=False)["input_ids"]], device=device)
    assert int((ids == dtok.mem_token_id).sum()) == embcat.shape[0], "decoder <MEM> count != vectors"
    _reset_rope_deltas(model.decoder)
    logits = model.decoder(inputs_embeds=model.replace_embeddings(embcat, ids), attention_mask=torch.ones_like(ids),
                           logits_to_keep=1).logits[0, -1].float()
    toks = []
    for l in letters:
        t = dtok(l, add_special_tokens=False)["input_ids"]
        assert len(t) == 1, f"letter {l!r} is not one token"
        toks.append(t[0])
    return torch.log_softmax(logits[toks], -1).tolist(), int(ids.shape[1])


def run(args, device):
    qs, n_min = load_questions(f"{args.frames_dir}/frames.json")
    allv = sorted(n_min)
    if args.eval_fold is not None:
        qs = [q for q in qs if allv.index(q["video_id"]) % args.nfolds == args.eval_fold]
    vids = sorted({q["video_id"] for q in qs})[args.shard::args.nshards]
    cands = json.load(open(args.cands)) if args.cands else None
    qs = [q for q in qs if q["video_id"] in vids][: args.limit]
    print(f"[mcq] shard {args.shard}/{args.nshards}: {len(vids)} videos, {len(qs)} questions", flush=True)
    model, coll = load_pisco(args, device)
    mem = model.decoder_tokenizer.mem_token
    offs = l0_offsets(args.l0_frames)
    sparse = lambda m: [m * FPM + o for o in offs]
    dense = lambda m: list(range(m * FPM, m * FPM + FPM))
    recs, t_all = [], time.time()
    for vid in vids:
        nm, emb = n_min[vid], {}
        if args.cache_dir:
            from cache_mems import load_cache
            emb = load_cache(args.cache_dir, vid, device)
        elif args.only_arms != ["blind"]:
            compress_frames(model, coll, args.frames_dir, vid, [k for m in range(nm) for k in sparse(m)], args.batch,
                            device, emb)
        if args.only_arms == ["blind"]:
            for q in [x for x in qs if x["video_id"] == vid]:
                hid = model.decoder.get_input_embeddings()(torch.zeros(1, 1, dtype=torch.long, device=device)).shape[-1]
                e = torch.empty(0, hid, device=device)
                lp, n_tok = letter_logprobs(model, "", e, coll.clean_text(q["question"]), q["letters"], device)
                pred = q["letters"][int(np.argmax(lp))]
                recs.append(dict(q, blind={"lp": lp, "pred": pred, "correct": pred == q["answer"], "tokens": n_tok}))
            print(f"[{len(recs)}/{len(qs)}] blind {np.mean([r['blind']['correct'] for r in recs]):.3f}", flush=True)
            continue
        texts = minute_texts(args.asr_dir, vid, nm) if args.asr_dir else {}
        asr_emb = dict(zip(sorted(texts), compress_texts(model, coll, [texts[m] for m in sorted(texts)], args.batch,
                                                          device))) if texts else {}
        none = lambda m: []

        def context(minutes, frames_of, asr=False):
            ks = [k for m in minutes for k in frames_of(m)]
            compress_frames(model, coll, args.frames_dir, vid, ks, args.batch, device, emb)
            bg, parts = "", []
            for m in minutes:
                group = [emb[k] for k in frames_of(m)] + ([asr_emb[m]] if asr and m in asr_emb else [])
                if group:
                    bg += f"Document {m}:" + mem * sum(g.shape[0] for g in group)
                    parts.extend(group)
            return bg, torch.cat(parts)

        for q in [x for x in qs if x["video_id"] == vid]:
            t = time.perf_counter()
            question = coll.clean_text(q["question"])
            gold = gold_minutes(q["ref"], nm)
            if cands is not None:   # the retriever's ranking
                ranked = [m for m in cands[q["uid"]] if m < nm]
                top = ranked[: args.top]
                lp0 = [-(ranked.index(m) if m in ranked else nm) for m in range(nm)]
            else:   # the model's own level-0 routing
                bg0, e0 = context(range(nm), sparse)
                lp0 = route_logprobs(model, bg0, e0, coll.clean_text(q["stem"]), nm, device)
                top = [int(j) for j in np.argsort(lp0)[::-1][: args.top]]
            mixed = lambda m, s=set(top): dense(m) if m in s else sparse(m)
            arms = {"coarse": (range(nm), sparse, False), "fine": (sorted(top), dense, False),
                    "both": (range(nm), mixed, False)}
            if 0 < len(gold) <= args.top:
                arms["both_gold"] = (range(nm), lambda m, s=set(gold): dense(m) if m in s else sparse(m), False)
            if asr_emb:
                arms.update({"asr": (range(nm), none, True), "coarse_asr": (range(nm), sparse, True),
                             "both_asr": (range(nm), mixed, True)})
            rec = dict(q, top=top, gold_minutes=gold, lp0_gold_rank=(min(sorted(range(nm), key=lambda j: -lp0[j]).index(g)
                                                                        for g in gold) if gold else None),
                       asr_minutes=len(asr_emb), asr_chars=sum(len(t) for t in texts.values()))
            if args.only_arms:
                arms = {a: v for a, v in arms.items() if a in args.only_arms}
            for arm, (minutes, fo, asr) in arms.items():
                bg, e = context(minutes, fo, asr)
                lp, n_tok = letter_logprobs(model, bg, e, question, q["letters"], device)
                pred = q["letters"][int(np.argmax(lp))]
                rec[arm] = {"lp": lp, "pred": pred, "correct": pred == q["answer"], "tokens": n_tok}
            rec["q_s"] = time.perf_counter() - t
            recs.append(rec)
            if len(recs) % 10 == 0:
                acc = {a: np.mean([r[a]["correct"] for r in recs if a in r]) for a in ("coarse", "fine", "both")}
                print(f"[{len(recs)}/{len(qs)}] " + " ".join(f"{a} {v:.3f}" for a, v in acc.items())
                      + f"  {time.time() - t_all:.0f}s", flush=True)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"args": vars(args), "records": recs}, open(args.out, "w"))
    summarize([args.out])


def summarize(files):
    recs = [r for f in files for r in json.load(open(f))["records"]]
    arms = [a for a in ["blind", "coarse", "fine", "both", "both_gold", "asr", "coarse_asr", "both_asr"] if any(a in r for r in recs)]
    print(f"n = {len(recs)} questions, {len({r['video_id'] for r in recs})} videos"
          + (f", {sum(1 for r in recs if r.get('asr_minutes'))} with a transcript" if "asr" in arms else ""))
    print(f"{'subset':<28}{'n':>6}" + "".join(f"{a:>12}" for a in arms) + f"{'pick@top':>10}")
    for name, sel in [("all", lambda r: True), ("local (ref <= 2 min)", lambda r: r["local"]),
                      ("global (ref > 2 min)", lambda r: not r["local"]),
                      ("summarization|reasoning", lambda r: "summarization" in r["qtype"] or "reasoning" in r["qtype"]),
                      ("key information retrieval", lambda r: "key information" in r["qtype"]),
                      ("with transcript", lambda r: bool(r.get("asr_minutes")))]:
        rs = [r for r in recs if sel(r)]
        row = f"{name:<28}{len(rs):>6}"
        for a in arms:
            xs = [r[a]["correct"] for r in rs if a in r]
            row += f"{np.mean(xs):>9.3f}/{len(xs):<3d}" if xs else f"{'-':>12}"
        pk = [r["lp0_gold_rank"] < len(r["top"]) for r in rs if r.get("lp0_gold_rank") is not None and r["local"]]
        row += f"{np.mean(pk):>10.3f}" if pk else f"{'-':>10}"
        print(row)
    toks = {a: np.mean([r[a]["tokens"] for r in recs if a in r]) for a in arms if any(a in r for r in recs)}
    print("mean prompt tokens:", {a: round(v) for a, v in toks.items()}, " chance:",
          round(float(np.mean([1 / len(r["letters"]) for r in recs])), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", nargs="*", default=None)
    ap.add_argument("--frames_dir", default="data/video/lvbench_frames")
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoint_path", default=None)
    ap.add_argument("--image_compr_rate", type=int, default=None)
    ap.add_argument("--image_max_pixels", type=int, default=451584)
    ap.add_argument("--l0_frames", type=int, default=2)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--asr_dir", default=None, help="transcripts from scripts/video/transcribe.py: adds the asr arms")
    ap.add_argument("--only_arms", nargs="*", default=None, help="run only these arms (blind = question-only control)")
    ap.add_argument("--cache_dir", default=None, help="per-frame <MEM> cache (scripts/video/cache_mems.py)")
    ap.add_argument("--cands", default=None, help="cands.json: CLIP-ranked minutes per question (retriever candidates)")
    ap.add_argument("--eval_fold", type=int, default=None)
    ap.add_argument("--nfolds", type=int, default=3)
    args = ap.parse_args()
    if args.summarize is not None:
        summarize(sorted(f for pat in args.summarize for f in glob.glob(pat)))
        return 0
    assert args.out and args.checkpoint_path
    run(args, "cuda")
    return 0


if __name__ == "__main__":
    sys.exit(main())
