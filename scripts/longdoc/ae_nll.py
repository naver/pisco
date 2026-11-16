"""Stage-1 autoencoding probe (campaign longdoc, arm T): did long-chunk stage 1 learn a length the later stages never use?

Each stage-1 text model (C256 / L1024, longchunk) rebuilds held-out MS MARCO documents from their <MEM>s, in the
stage-1 AE format of pisco.collator.PretrainingCollator (<AE> <MEM>*n <bos> text <eos>, loss after the last <MEM>).
Rows from 1,000,000 on (training used rows [0, 1M), pisco/train.py `select(range(samples))`), kept when the
compressor tokenizer gives >= 1,024 tokens. Per document, the target is a prefix of its first 1,024 tokens:
  s256        one chunk of 256 tokens -> the same 256 tokens            (the chunk length later stages use)
  l1024       one chunk of 1,024 tokens -> the same 1,024 tokens        (the length only L1024 trained on)
  l1024_4x256 four chunks of 256 tokens -> the same 1,024 tokens        (chunk length varied, target fixed)
  wrong256    the PREVIOUS document's 256-token <MEM>s -> this document's 256 tokens (control: must be far worse)
Output: per model, per condition, per document the mean per-token NLL (nats) and the token count.
Pair two models per document with scripts/longdoc/ae_pair.py (paired bootstrap CI + sign test).
  python scripts/longdoc/ae_nll.py --model c256=<dir> --model l1024=<dir> --n_docs 500 --out <json>
"""

import argparse
import json
import pathlib
import sys
import time

import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pisco.collator import PretrainingCollator  # noqa: E402
from pisco.collator_utils import add_memory_tokens_to_inputs, mask_before_mem  # noqa: E402
from pisco.model import PISCO  # noqa: E402

CONDS = ["s256", "l1024", "l1024_4x256", "wrong256"]


def pick_docs(path, start, n_docs, min_tokens, tok, clean):
    import datasets
    ds = datasets.load_from_disk(path)
    if isinstance(ds, datasets.DatasetDict):
        ds = ds["train"]
    docs, i = [], start
    while len(docs) < n_docs and i < len(ds):
        row = ds[i]
        ids = tok(clean(row["text"]), add_special_tokens=False)["input_ids"]
        if len(ids) >= min_tokens:
            docs.append({"row": i, "id": row["id"], "ids": ids[:min_tokens]})
        i += 1
    print(f"[ae] {len(docs)} docs >= {min_tokens} tokens from rows {start}..{i - 1}", flush=True)
    return docs


@torch.inference_mode()
def nll(model, coll, chunks, target_ids, device):
    comp_ids, n_mems = add_memory_tokens_to_inputs(chunks, coll.compressor_tokenizer, coll.compr_rate)
    comp = coll.compressor_pad(comp_ids)
    dtok = coll.decoder_tokenizer
    text = coll.compressor_tokenizer.decode(target_ids)
    dec_text = dtok.ae_token + dtok.mem_token * sum(n_mems) + dtok.bos_token + text + dtok.eos_token
    dec = dtok([dec_text], return_tensors="pt", add_special_tokens=False)
    labels = mask_before_mem(coll.mask_special_tokens(dec["input_ids"].clone()), mem_token_id=dtok.mem_token_id)
    coll.assert_consistent_n_mems(comp, dec)
    out = model(compressor_input_ids=comp["input_ids"].to(device),
                compressor_attention_mask=comp["attention_mask"].to(device),
                decoder_input_ids=dec["input_ids"].to(device),
                decoder_attention_mask=dec["attention_mask"].to(device),
                labels=labels.to(device))
    n_tok = int((labels[:, 1:] != -100).sum())
    return float(out["loss"]), n_tok, sum(n_mems)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="name=checkpoint_dir (repeatable)")
    ap.add_argument("--data_path", default=str(REPO / "data/msmarco_docs"))
    ap.add_argument("--start_row", type=int, default=1_000_000)
    ap.add_argument("--n_docs", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_path = pathlib.Path(args.out)
    if out_path.exists():
        sys.exit(f"refusing to overwrite {out_path}")
    device = "cuda"
    res = {"args": vars(args), "conds": CONDS, "docs": None, "models": {}}
    docs = None
    for spec in args.model:
        name, path = spec.split("=", 1)
        t0 = time.time()
        model = PISCO.from_pretrained(path)
        model.to(device).eval()
        stride = int(getattr(model.config, "mem_position_stride", 1) or 1)
        coll = PretrainingCollator(model.compressor_tokenizer, model.decoder_tokenizer, model.compr_rate,
                                   compressor_max_length=1024, decoder_max_length=4096)
        if docs is None:   # same documents for every model (the tokenizers are the same backbone)
            docs = pick_docs(args.data_path, args.start_row, args.n_docs, 1024, model.compressor_tokenizer,
                             coll.clean_text)
            res["docs"] = [{"row": d["row"], "id": d["id"]} for d in docs]
        rec = {c: [] for c in CONDS}
        for k, d in enumerate(docs):
            ids = d["ids"]
            prev = docs[k - 1]["ids"]   # k = 0 takes the last document
            for c, chunks, tgt in [("s256", [ids[:256]], ids[:256]),
                                   ("l1024", [ids], ids),
                                   ("l1024_4x256", [ids[j:j + 256] for j in range(0, 1024, 256)], ids),
                                   ("wrong256", [prev[:256]], ids[:256])]:
                loss, n_tok, mems = nll(model, coll, chunks, tgt, device)
                rec[c].append({"nll": loss, "tokens": n_tok, "mems": mems})
            if k % 50 == 0:
                print(f"[ae] {name} doc {k}: " + " ".join(f"{c}={rec[c][-1]['nll']:.3f}" for c in CONDS), flush=True)
        means = {c: sum(r["nll"] for r in rec[c]) / len(rec[c]) for c in CONDS}
        print(f"[ae] {name} stride={stride} mean nll: {means}  ({time.time() - t0:.0f} s)", flush=True)
        res["models"][name] = {"path": path, "stride": stride, "mean_nll": means, "records": rec}
        del model
        torch.cuda.empty_cache()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out_path, "w"))
    print(f"[ae] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
