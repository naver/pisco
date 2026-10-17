"""Prepare a vLLM run of the PISCO decoder (pisco env, one GPU).

vLLM's Qwen3.5 VLM class needs token ids to build its 3-axis rotary positions, so it cannot take prompt embeddings.
For text-only input the three axes are equal and M-RoPE == ordinary RoPE, so the decoder is exported as vLLM's
(unregistered) text-only `Qwen3_5ForCausalLM`: LoRA merged, language-model weights renamed `model.*`, vision tower
dropped, `mrope_*` removed from rope_parameters. The embedding matrix is exported WITH the TrainableTokens rows
(<MEM>/<AE>) applied; the head is tied to it, as in HF.

Writes:
  <export_dir>/                     config.json + model.safetensors + tokenizer  (vLLM model dir)
  <inputs>/bench_<tag>.pt           {K: {"embeds": [T_i x H bf16], "raw_prompts": [str], "raw_len": [int],
                                          "image_refs": [[pool idx]]}, "pool_refs": [(row, img)], "data": path}
  <inputs>/acc_<tag>.pt             {"embeds": [...], "query_ids": [...], "data": path}  (eval-identical embeddings)

  python scripts/vllm_prep_pisco.py --checkpoint_path expQ/ft_nav_4b/model --export_dir expQ/ft_nav_4b/vllm_decoder \
     --bench_data data/nav_ft_val_v2 --acc_data data/nav_ft_val_v2 --acc_topk 8 --image_compr_rate 16 --tag nav \
     --inputs expQ/bench/vllm_inputs
"""
import argparse
import json
import os
import pathlib
import shutil
import sys

import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from pisco.model import PISCO, PISCOConfig, merge_lora_inplace  # noqa: E402
from pisco.mm_utils import collate_compressor, get_vision_token_ids, load_image_processor, segments_to_chunks  # noqa: E402
import eval_mm_docqa as E  # noqa: E402
from relabel_mm_teacher import build_prompt  # noqa: E402


def export_decoder(model: PISCO, export_dir: pathlib.Path, base_name: str):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file
    export_dir.mkdir(parents=True, exist_ok=True)
    dec = merge_lora_inplace(model.decoder)
    # effective input-embedding matrix, TrainableTokens rows included
    emb_layer = dec.get_input_embeddings()
    n_vocab = emb_layer.weight.shape[0] if hasattr(emb_layer, "weight") else emb_layer.base_layer.weight.shape[0]
    dev = next(dec.parameters()).device
    with torch.inference_mode():
        emb = torch.cat([emb_layer(torch.arange(i, min(i + 8192, n_vocab), device=dev)) for i in range(0, n_vocab, 8192)])
    out = {"model.embed_tokens.weight": emb.to(torch.bfloat16).cpu().contiguous()}
    for k, v in dec.state_dict().items():
        if ".visual." in k or "lora_" in k or "trainable_tokens" in k or "embed_tokens" in k or k.startswith("lm_head"):
            continue
        assert k.startswith("model.language_model."), k
        nk = "model." + k[len("model.language_model."):].replace(".base_layer.", ".")
        out[nk] = v.to(torch.bfloat16).cpu().contiguous()
    save_file(out, str(export_dir / "model.safetensors"), metadata={"format": "pt"})
    cfg = json.load(open(hf_hub_download(base_name, "config.json")))
    cfg["architectures"] = ["Qwen3_5ForCausalLM"]
    t = cfg["text_config"]
    t["vocab_size"] = int(n_vocab)
    t["rope_parameters"] = {k: v for k, v in t["rope_parameters"].items() if not k.startswith("mrope")}
    json.dump(cfg, open(export_dir / "config.json", "w"), indent=1)
    model.decoder_tokenizer.save_pretrained(str(export_dir))
    print(f"[export] {len(out)} tensors, vocab {n_vocab}, {sum(v.numel() for v in out.values())/1e9:.2f}B params -> {export_dir}")
    return dec


def bench_inputs(model: PISCO, args, base_tok, base_proc, base_vision_ids):
    import datasets
    ds = datasets.load_from_disk(args.bench_data)
    ds = ds["train"] if isinstance(ds, datasets.DatasetDict) else ds
    rows = [ds[i] for i in range(min(64, len(ds)))]
    pool, pool_refs = [], []
    for ri, r in enumerate(rows):
        for ii, img in enumerate(r.get("doc_images") or []):
            if img is not None:
                pool.append(img); pool_refs.append((ri, ii))
    tok, dtok = model.compressor_tokenizer, model.decoder_tokenizer
    vision_ids = model.vision_token_ids or get_vision_token_ids(tok)
    pad = E._Pad(tok, model.image_processor)
    embs = []
    with torch.inference_mode():
        for img in pool:
            b = segments_to_chunks([("image", img)], compressor_tokenizer=tok, image_processor=model.image_processor,
                                   compr_rate=model.compr_rate, image_compr_rate=args.image_compr_rate,
                                   text_chunker=lambda ids: [ids], vision_ids=vision_ids)
            comp = {k: v.to("cuda") for k, v in collate_compressor([b], pad, vision_ids).items()}
            embs.append(model.compress(comp["compressor_input_ids"], comp["compressor_attention_mask"],
                                       pixel_values=comp.get("pixel_values"), image_grid_thw=comp.get("image_grid_thw"),
                                       mm_token_type_ids=comp.get("mm_token_type_ids")))
    vs, ip, ve = base_vision_ids
    single_span = base_tok.convert_ids_to_tokens(vs) + base_tok.convert_ids_to_tokens(ip) + base_tok.convert_ids_to_tokens(ve)
    out = {"pool_refs": pool_refs, "data": args.bench_data, "mem_per_image": int(embs[0].shape[0])}
    for K in [int(k) for k in args.ks.split(",")]:
        d = {"embeds": [], "raw_prompts": [], "raw_len": [], "image_refs": []}
        for qi in range(args.n_queries):
            idx = [(qi * K + j) % len(pool) for j in range(K)]
            q = rows[qi % len(rows)]["query"]
            background = "".join(f"Document {j}:" + dtok.mem_token * embs[i].shape[0] for j, i in enumerate(idx))
            ids = dtok(E._build_chat_prompt(dtok, background=background, question=q), return_tensors="pt",
                       add_special_tokens=False)["input_ids"].to("cuda")
            with torch.inference_mode():
                pe = model.replace_embeddings(torch.cat([embs[i] for i in idx]), ids)[0]
            d["embeds"].append(pe.to(torch.bfloat16).cpu())
            # raw-image prompt for vLLM: ONE <|image_pad|> per image (vLLM expands it), same template
            raw_bg = "".join(f"Document {j}:" + single_span for j in range(K))
            d["raw_prompts"].append(E._build_chat_prompt(base_tok, background=raw_bg, question=q))
            full, _, _ = build_prompt(base_tok, [""] * K, [pool[i] for i in idx], q, base_proc, base_vision_ids)
            d["raw_len"].append(len(base_tok(full, add_special_tokens=False)["input_ids"]))
            d["image_refs"].append(idx)
        out[K] = d
        print(f"[bench] K={K}: pisco len {d['embeds'][0].shape[0]}, raw len {d['raw_len'][0]}")
    return out


def acc_inputs(model: PISCO, args):
    """Embeddings built by eval_mm_docqa.gen_pisco itself: its decoder.generate call is intercepted."""
    import datasets
    ds = datasets.load_from_disk(args.acc_data)
    ds = ds["train"] if isinstance(ds, datasets.DatasetDict) else ds
    captured = {}

    class _Stop(Exception):
        pass

    real_generate = model.decoder.generate

    def grab(*a, inputs_embeds=None, **kw):
        captured["e"] = inputs_embeds[0].to(torch.bfloat16).cpu()
        raise _Stop

    model.decoder.generate = grab
    out = {"embeds": [], "query_ids": [], "data": args.acc_data, "topk": args.acc_topk}
    with torch.inference_mode():
        for i in range(min(len(ds), args.acc_limit or len(ds))):
            row = ds[i]
            try:
                E.gen_pisco(model, row, device=torch.device("cuda"), topk=args.acc_topk, compressor_max_length=256,
                            decoder_max_length=2048, max_new_tokens=32, image_compr_rate=args.image_compr_rate)
            except _Stop:
                pass
            out["embeds"].append(captured.pop("e")); out["query_ids"].append(row.get("query_id", i))
    model.decoder.generate = real_generate
    print(f"[acc] {len(out['embeds'])} prompts, mean len {sum(e.shape[0] for e in out['embeds'])/len(out['embeds']):.0f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--export_dir", required=True)
    ap.add_argument("--bench_data", required=True)
    ap.add_argument("--acc_data", default=None)
    ap.add_argument("--acc_topk", type=int, default=4)
    ap.add_argument("--acc_limit", type=int, default=None)
    ap.add_argument("--image_compr_rate", type=int, default=None)
    ap.add_argument("--ks", default="4,8,16")
    ap.add_argument("--n_queries", type=int, default=256)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--inputs", required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    model = PISCO.from_pretrained(args.checkpoint_path).to("cuda").eval()
    base_name = PISCOConfig.from_pretrained(args.checkpoint_path).decoder_model_name
    base_tok = AutoTokenizer.from_pretrained(base_name)
    base_proc = load_image_processor(base_name)
    base_vision_ids = get_vision_token_ids(base_tok)
    inputs = pathlib.Path(args.inputs); inputs.mkdir(parents=True, exist_ok=True)
    if args.acc_data:   # BEFORE the merge: embeddings exactly as the HF eval builds them
        torch.save(acc_inputs(model, args), inputs / f"acc_{args.tag}.pt")
    torch.save(bench_inputs(model, args, base_tok, base_proc, base_vision_ids), inputs / f"bench_{args.tag}.pt")
    if not (pathlib.Path(args.export_dir) / "model.safetensors").exists():
        export_decoder(model, pathlib.Path(args.export_dir), base_name)
    print("[prep] done")


if __name__ == "__main__":
    main()
