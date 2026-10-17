"""Label-free probe: how well do NAVIGATION frames (Habitat renders) survive compression?

1. The uncompressed backbone (Qwen3.5-4B, raw image) lists the objects in each frame -> reference.
2. Yes/no questions per frame: up to K positives (objects it listed) and K negatives (objects listed
   for OTHER frames, absent from this frame's list and its description).
3. Answered by: the uncompressed model on the raw frame (consistency ceiling), and a multimodal PISCO
   checkpoint at several image compression rates (eval-time image_compr_rate override).
Reports yes/no accuracy (positives / negatives / all) per rate, and object-list recall of the
reference objects in PISCO's own list. Same single-document RAG prompt as the doc-QA eval.

  python scripts/probe_mm_frames.py --checkpoint_path expQ/ft_mm_4b_171745/model \
      --frames_glob '/beegfs/.../clips_test/*/frames/*.png' --every 2 --rates 8,16,32,64 \
      --output_path expQ/probe_habitat/probe_171745.json
"""

import argparse
import glob
import json
import os
import pathlib
import random
import re
import sys
from collections import defaultdict

import torch
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from eval_mm_docqa import gen_base, gen_pisco  # noqa: E402
from pisco.model import PISCO  # noqa: E402
from pisco.mm_utils import load_image_processor  # noqa: E402

LIST_Q = "List the distinct objects visible in the image, as a comma-separated list of short nouns."
STOP = {"wall", "walls", "floor", "ceiling", "image", "picture", "room", "light", "shadow", "corner", "surface", "area"}


def parse_objects(text):
    items = re.split(r",|\n|;| and ", text.lower())
    out = []
    for it in items:
        it = re.sub(r"^[\s\-\*\d\.\)]+", "", it).strip(" .")
        it = re.sub(r"^(a|an|the|some)\s+", "", it)
        if 2 < len(it) <= 30 and it not in STOP and it not in out:
            out.append(it)
    return out


def yes(text):
    t = text.strip().lower()
    return t.startswith("yes") or t.startswith("there is") or t.startswith("there are")


def mentions(obj, text):
    head = obj.split()[-1]
    return re.search(r"\b" + re.escape(head.rstrip("s")) + r"s?\b", text.lower()) is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--frames_glob", required=True)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--rates", default="8,16,32,64")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--image_max_pixels", type=int, default=451584)
    args = ap.parse_args()
    rng = random.Random(0)
    dev = torch.device("cuda")
    frames = sorted(glob.glob(args.frames_glob))[:: args.every]
    print(f"[probe] {len(frames)} frames")

    # ---- uncompressed reference
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from pisco.model import PISCOConfig
    name = PISCOConfig.from_pretrained(args.checkpoint_path).decoder_model_name
    btok = AutoTokenizer.from_pretrained(name, padding_side="left")
    base = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    proc = load_image_processor(name, max_pixels=args.image_max_pixels)

    def ask_base(img, q, n=48):
        row = {"docs": [""], "doc_images": [img], "query": q}
        return gen_base(base, proc, btok, row, device=dev, topk=1, max_new_tokens=n, decoder_max_length=4096)[0]

    recs = []
    for f in frames:
        img = Image.open(f).convert("RGB")
        lst = ask_base(img, LIST_Q, 64)
        recs.append({"frame": f, "ref_list_raw": lst, "ref_objects": parse_objects(lst)})
    pool = sorted({o for r in recs for o in r["ref_objects"]})
    for r in recs:
        pos = r["ref_objects"][: args.k]
        negs = [o for o in pool if o not in r["ref_objects"] and not mentions(o, r["ref_list_raw"])]
        neg = rng.sample(negs, min(args.k, len(negs)))
        r["questions"] = [(f"Is there a {o} in the image? Answer yes or no.", 1, o) for o in pos] + \
                         [(f"Is there a {o} in the image? Answer yes or no.", 0, o) for o in neg]
        img = Image.open(r["frame"]).convert("RGB")
        r["base_yesno"] = [int(yes(ask_base(img, q, 8))) for q, _, _ in r["questions"]]
    del base; torch.cuda.empty_cache()

    # ---- compressed, several rates
    model = PISCO.from_pretrained(args.checkpoint_path, image_max_pixels=args.image_max_pixels).to(dev).eval()
    rates = [int(x) for x in args.rates.split(",")]
    for rate in rates:
        for r in recs:
            img = Image.open(r["frame"]).convert("RGB")
            def ask(q, n):
                row = {"docs": [""], "doc_images": [img], "query": q}
                return gen_pisco(model, row, device=dev, topk=1, compressor_max_length=256, decoder_max_length=2048,
                                 max_new_tokens=n, image_compr_rate=rate)
            lst, st = ask(LIST_Q, 64)
            r[f"r{rate}_list"] = lst
            r[f"r{rate}_mems"] = st["mems_used"]
            r[f"r{rate}_yesno"] = [int(yes(ask(q, 8)[0])) for q, _, _ in r["questions"]]
        print(f"[probe] rate {rate} done", flush=True)

    # ---- summary
    def acc(key, which):
        hits = n = 0
        for r in recs:
            for (q, lab, o), p in zip(r["questions"], r[key]):
                if which == "all" or (which == "pos") == (lab == 1):
                    hits += int(p == lab); n += 1
        return hits / max(1, n), n
    summ = {"n_frames": len(recs), "n_ref_objects_mean": sum(len(r["ref_objects"]) for r in recs) / len(recs)}
    print(f"\n=== FRAME PROBE ({len(recs)} frames, {args.checkpoint_path}) ===")
    print(f"{'reader':26s} {'MEM/frame':>9s} {'yes/no all':>10s} {'positives':>10s} {'negatives':>10s} {'obj recall':>10s}")
    for key, label, mem in [("base_yesno", "uncompressed (raw frame)", "-")] + \
            [(f"r{rt}_yesno", f"PISCO rate {rt}", f"{sum(r[f'r{rt}_mems'] for r in recs) / len(recs):.0f}") for rt in rates]:
        a, n = acc(key, "all"); p, _ = acc(key, "pos"); ng, _ = acc(key, "neg")
        if key == "base_yesno":
            rec = 1.0
        else:
            rt = key[1:].split("_")[0]
            tot = sum(len(r["ref_objects"]) for r in recs) or 1
            rec = sum(mentions(o, r[f"r{rt}_list"]) for r in recs for o in r["ref_objects"]) / tot
        summ[label] = {"mem_per_frame": mem, "yesno_all": a, "yesno_pos": p, "yesno_neg": ng, "obj_recall": rec, "n_q": n}
        print(f"{label:26s} {mem:>9s} {a:10.3f} {p:10.3f} {ng:10.3f} {rec:10.3f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    json.dump({"summary": summ, "records": recs}, open(args.output_path, "w"), indent=1)
    print(f"wrote {args.output_path}")


if __name__ == "__main__":
    main()
