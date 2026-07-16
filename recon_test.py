"""Autoencoding reconstruction test: does a PISCO backbone's COMPRESSION work?
Compress each doc, prompt the decoder with the training-time AE prefix
(<AE> <MEM>*n <bos>), greedily generate, and compare to the original.
Good compression -> reconstruction closely matches the source.

Run: pixi run python recon_test.py  ['label=ckpt' ...]
"""
import sys
import torch
from pisco.model import PISCO
from pisco.collator import PretrainingCollator

DEFAULT = [
    ("Qwen3-8B/8B 135866 (mine)", "/beegfs/scratch/user/hdejean/pisco/expQ/135866/model"),
    ("Qwen2.5-7B expQ (baseline)", "/beegfs/scratch/user/hdejean/pisco/expQ/expQ/model"),
]

DOCS = [
    "The capital of France is Paris, a city famous for the Eiffel Tower and the Louvre museum.",
    "Photosynthesis is the process by which green plants convert sunlight, water and carbon dioxide into glucose and oxygen.",
    "Albert Einstein developed the theory of relativity, one of the two pillars of modern physics.",
]


def parse(argv):
    if not argv:
        return DEFAULT
    out = []
    for a in argv:
        lbl, path = a.split("=", 1) if "=" in a else (a, a)
        out.append((lbl, path))
    return out


@torch.no_grad()
def run(name, ckpt):
    print(f"\n================ {name} ================")
    print(f"[load] {ckpt}")
    pisco = PISCO.from_pretrained(ckpt, load_decoder=True).cuda().eval()
    coll = PretrainingCollator(
        pisco.compressor_tokenizer, pisco.decoder_tokenizer, pisco.compr_rate,
        compressor_max_length=512, decoder_max_length=2048, ae_ratio=1.0,
    )
    dt = pisco.decoder_tokenizer
    bos = dt.bos_token or ""
    for doc in DOCS:
        ids = pisco.compressor_tokenizer(doc)["input_ids"]
        comp_ids, decoder_text = coll.prepare_for_autoencoding(text=doc, text_ids=ids)
        n_mem = decoder_text.count(dt.mem_token)
        prompt = dt.ae_token + dt.mem_token * n_mem + bos  # <AE><MEM>*n<bos?>
        comp_pad = coll.compressor_pad(comp_ids)
        dec = dt(prompt, return_tensors="pt", add_special_tokens=False)
        mi = {
            "compressor_input_ids": comp_pad["input_ids"].cuda(),
            "compressor_attention_mask": comp_pad["attention_mask"].cuda(),
            "decoder_input_ids": dec["input_ids"].cuda(),
            "decoder_attention_mask": dec["attention_mask"].cuda(),
        }
        recon = pisco.generate(mi, max_new_tokens=96)[0]
        print(f"\n  [{n_mem} mem tokens]")
        print(f"  ORIG : {doc}")
        print(f"  RECON: {recon.strip()[:300]}")
    del pisco
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for name, ckpt in parse(sys.argv[1:]):
        try:
            run(name, ckpt)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[ERROR] {name}: {type(e).__name__}: {e}")
