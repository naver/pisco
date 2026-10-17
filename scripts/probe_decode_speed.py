"""Check pisco.model.merge_lora_inplace: outputs must match the unmerged decoder, speed must match the raw backbone.
Why is the PISCO decoder slower PER TOKEN than the raw backbone? The decoder carries a transformers-integrated
PEFT adapter (`add_adapter`), so `merge_and_unload` does not exist and the rank-64 LoRA path runs on every linear at
every step. Test: merge_adapter() (fold weights, keep wrappers) and a full unwrap (replace each LoRA layer by its
base_layer) -> pisco.model.merge_lora_inplace.
  python scripts/probe_decode_speed.py --checkpoint_path expQ/ft_nav_4b/model
"""
import argparse, pathlib, statistics, sys, time
import torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pisco.model import PISCO, PISCOConfig, merge_lora_inplace  # noqa: E402


def sync(): torch.cuda.synchronize()


def gen_ms(model, reps=3, **kw):
    def f(): return model.generate(**kw, do_sample=False, top_p=None, max_new_tokens=33, min_new_tokens=33)
    f(); sync(); ts = []
    for _ in range(reps):
        sync(); t = time.perf_counter(); f(); sync(); ts.append(time.perf_counter() - t)
    def f1(): return model.generate(**kw, do_sample=False, top_p=None, max_new_tokens=1, min_new_tokens=1)
    f1(); sync(); t = time.perf_counter(); f1(); sync(); t1 = time.perf_counter() - t
    return 1000 * (statistics.median(ts) - t1) / 32, 1000 * t1


def describe(m, tag):
    dt = {str(p.dtype) for p in m.parameters()}
    lora = sum(1 for n, _ in m.named_modules() if "lora" in n.lower())
    tt = sum(1 for n, _ in m.named_modules() if "trainable_token" in n.lower())
    print(f"[{tag}] class={type(m).__name__} dtypes={dt} lora_modules={lora} trainable_token_wrappers={tt} "
          f"attn={getattr(m.config, '_attn_implementation', '?')} n_params={sum(p.numel() for p in m.parameters())/1e9:.3f}B")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint_path", required=True); args = ap.parse_args()
    dev = torch.device("cuda")
    text = "Background: " + "the quick brown fox jumps over the lazy dog. " * 25 + "\nQuestion: what jumps?\nAnswer:"
    import json as _json
    msgs = [{"role": "system", "content": "You answer from the background."}, {"role": "user", "content": text}]

    model = PISCO.from_pretrained(args.checkpoint_path).to(dev).eval()
    dtok = model.decoder_tokenizer; dtok.padding_side = "left"
    text = dtok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = dtok(text, return_tensors="pt", add_special_tokens=False).to(dev)
    kw = dict(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    with torch.inference_mode():
        describe(model.decoder, "as loaded (LoRA active)")
        ref_logits = model.decoder(**kw).logits[0, -1].float()
        ref = model.decoder.generate(**kw, do_sample=False, top_p=None, max_new_tokens=16, min_new_tokens=16)
        print("  %.1f ms/token, ttft %.0f ms" % gen_ms(model.decoder, **kw))
        print("  text:", repr(dtok.decode(ref[0, enc['input_ids'].shape[1]:])))
        model.decoder = merge_lora_inplace(model.decoder)
        describe(model.decoder, "merge_lora_inplace (each LoraLayer merged, then unwrapped)")
        print("  %.1f ms/token, ttft %.0f ms" % gen_ms(model.decoder, **kw))
        out = model.decoder.generate(**kw, do_sample=False, top_p=None, max_new_tokens=16, min_new_tokens=16)
        print("  same output as unmerged:", bool((out == ref).all()), " max|dlogit| %.4f" % (model.decoder(**kw).logits[0, -1].float() - ref_logits).abs().max().item())
        print("  text:", repr(dtok.decode(out[0, enc['input_ids'].shape[1]:])))
        # the PISCO path (inputs_embeds via replace_embeddings) still works after unwrap
        emb = model.decoder.get_input_embeddings()(enc["input_ids"])
        print("  embeds: %.1f ms/token, ttft %.0f ms" % gen_ms(model.decoder, inputs_embeds=emb, attention_mask=enc["attention_mask"]))
    del model; torch.cuda.empty_cache()

    from transformers import AutoModelForImageTextToText, AutoTokenizer
    name = PISCOConfig.from_pretrained(args.checkpoint_path).decoder_model_name
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    base = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(dev)
    with torch.inference_mode():
        describe(base, "raw backbone")
        print("  %.1f ms/token, ttft %.0f ms" % gen_ms(base, input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]))


if __name__ == "__main__":
    main()
