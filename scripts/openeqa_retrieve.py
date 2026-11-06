"""Rank the frames of each OpenEQA HM3D tour for each question with Qwen3-VL-Embedding-2B (the retriever of the
silver-nav campaign_eqa `bank` arm): cosine between the question embedding and every frame embedding of its tour.
Last-token pooling of the final hidden state, L2-normalised (the model is a decoder used as an embedder). Images go
through the PIL image processor (no torchvision in this env); the image span is written into the chat text by hand.

Writes {"ranking": {question_id: [frame stems, best first]}, "scores": {...}, "overlap_top15": ...} to --out; the
builder (scripts/build_mm_eval_openeqa.py --frames_from) turns it into retr-k eval sets. --compare: a campaign cell
json whose rows[].frames hold its retriever's top-15, to report the top-15 overlap (same model, maybe other prompts).

  python scripts/openeqa_retrieve.py --out expQ/openeqa/ranking_qwen3vlemb2b.json \
      --compare /beegfs/scratch/user/hdejean/silver-nav/results/eqa/em_bank_k15_327431.json
"""

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pisco.mm_utils import image_span_text, load_image_processor, merge_size_of, n_image_tokens  # noqa: E402
from build_mm_eval_openeqa import FRAME_ROOT, frame_paths, load_questions  # noqa: E402

REPO = "Qwen/Qwen3-VL-Embedding-2B"
# asymmetric instructions, in the spirit of silver-nav/navmem/encode.py (the campaign's exact eqa prompts are not on
# this cluster): the question asks about something, the frame shows a view of the house
QUERY_PROMPT = "Retrieve the view of the house that answers this question."
IMAGE_PROMPT = "Describe the objects visible in this view of a house."


class Embedder:
    def __init__(self, repo=REPO, max_pixels=451584, min_pixels=65536, batch=16, device="cuda"):
        from PIL import Image
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
        self.Image = Image
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.tok.padding_side = "right"
        self.proc = load_image_processor(repo, max_pixels=max_pixels, min_pixels=min_pixels)
        self.merge = merge_size_of(self.proc)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(repo, dtype=torch.bfloat16).to(device).eval()
        self.device, self.batch = device, batch

    def _chat(self, system, user):
        return self.tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                            tokenize=False, add_generation_prompt=False)

    @torch.inference_mode()
    def _pool(self, texts, pixel_values=None, grids=None):
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        extra = {}
        if pixel_values is not None:
            extra = dict(pixel_values=pixel_values.to(self.device, torch.bfloat16), image_grid_thw=grids.to(self.device))
        out = self.model(**enc, **extra, output_hidden_states=True, logits_to_keep=1, use_cache=False)
        h = out.hidden_states[-1]
        last = enc["attention_mask"].sum(1) - 1
        v = h[torch.arange(h.shape[0], device=h.device), last].float()
        return torch.nn.functional.normalize(v, dim=-1).cpu().numpy()

    def texts(self, strings):
        out = []
        for i in range(0, len(strings), self.batch):
            out.append(self._pool([self._chat(QUERY_PROMPT, s) for s in strings[i:i + self.batch]]))
        return np.concatenate(out)

    def images(self, paths):
        out = []
        for i in range(0, len(paths), self.batch):
            ims = [self.Image.open(p).convert("RGB") for p in paths[i:i + self.batch]]
            feats = self.proc(images=ims, return_tensors="pt")
            grids = feats["image_grid_thw"]
            texts = [self._chat(IMAGE_PROMPT, image_span_text(n_image_tokens(g, self.merge))) for g in grids]
            out.append(self._pool(texts, feats["pixel_values"], grids))
        return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=64)
    ap.add_argument("--compare", default=None, help="campaign cell json with rows[].frames (top-15 overlap)")
    ap.add_argument("--frame_root", default=FRAME_ROOT)
    ap.add_argument("--limit_tours", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    qs = load_questions()
    by_tour = {}
    for q in qs:
        by_tour.setdefault(q["episode_history"].split("/", 1)[1], []).append(q)
    tours = sorted(by_tour)[: args.limit_tours or None]
    emb = Embedder(batch=args.batch)
    ranking, scores, t0 = {}, {}, time.time()
    for ti, tour in enumerate(tours):
        paths = frame_paths(os.path.join(args.frame_root, tour))
        stems = [os.path.splitext(os.path.basename(p))[0] for p in paths]
        fv = emb.images(paths)
        qv = emb.texts([q["question"] for q in by_tour[tour]])
        sim = qv @ fv.T
        for q, row in zip(by_tour[tour], sim):
            order = np.argsort(-row, kind="stable")[: args.top]
            ranking[q["question_id"]] = [stems[i] for i in order]
            scores[q["question_id"]] = [float(row[i]) for i in order]
        print(f"[{ti + 1}/{len(tours)}] {tour}: {len(paths)} frames, {len(by_tour[tour])} questions, "
              f"{time.time() - t0:.0f}s", flush=True)
    out = {"model": REPO, "query_prompt": QUERY_PROMPT, "image_prompt": IMAGE_PROMPT, "top": args.top,
           "ranking": ranking, "scores": scores}
    if args.compare:
        camp = {r["question_id"]: [os.path.splitext(os.path.basename(p))[0] for p in r["frames"]]
                for r in json.load(open(args.compare))["rows"]}
        ov = [len(set(ranking[q][:15]) & set(camp[q][:15])) / 15 for q in ranking if q in camp]
        top1 = [float(ranking[q][0] == camp[q][0]) for q in ranking if q in camp]
        out["overlap_top15"] = float(np.mean(ov)); out["same_top1"] = float(np.mean(top1))
        print(f"top-15 overlap with {os.path.basename(args.compare)}: {np.mean(ov):.3f}; same top-1 {np.mean(top1):.3f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
