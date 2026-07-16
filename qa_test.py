"""QA test: does a PISCO backbone COMPRESS a text and ANSWER a question about it?

The texts use FICTIONAL / COUNTERFACTUAL facts that are NOT in any LLM's world
knowledge (made-up names/numbers, and one fact that contradicts reality). This
way a correct answer can ONLY come from reading the compressed text, not from
the decoder's parametric memory.

To prove that, every question is asked TWICE:
  - WITH  : compressed text given as <MEM> tokens   (should be CORRECT)
  - NOCTX : no text at all, only the question        (should be WRONG / "I don't know")
A healthy, faithful backbone: WITH passes, NOCTX fails.

Run: pixi run python qa_test.py  ['label=ckpt' ...]   (submit via qa_test.sh on slurm)
"""
import sys
import torch
from pisco.model import PISCO
from pisco.collator_utils import add_memory_tokens_to_inputs, chunk_list

SYSTEM_PROMPT = (
    "You are a helpful assistant. Your task is to extract relevant information from "
    "provided documents and to answer to questions as briefly as possible. "
    "If the answer is not in the documents, say you don't know."
)

DEFAULT = [
    ("Qwen2.5-7B expQ (working baseline)", "/beegfs/scratch/user/hdejean/pisco/expQ/expQ/model"),
    ("Ministral-8B 134552 (working)", "/beegfs/scratch/user/hdejean/pisco/exp4/pt_lora_ministral-8_8_134552/model"),
    ("Qwen3-8B/8B 135866 (suspect collapse)", "/beegfs/scratch/user/hdejean/pisco/expQ/135866/model"),
]

# Each: (text to compress, question, expected substring, must-NOT-contain-if-guessing).
# Facts are invented or counterfactual so they CANNOT be answered from memory.
QA = [
    (
        "The Zorbian Treaty was signed in 1847 by Queen Helvina of Brundlewick, ending "
        "the twelve-year Salt War. Under its terms the port city of Marnach was handed "
        "to the Kingdom of Ostreth, and all tariffs on glassware were abolished.",
        "Who signed the Zorbian Treaty, and which city was handed to Ostreth?",
        "marnach",  # also expect 'helvina'
    ),
    (
        "Dr. Quillon Vasterby discovered the mineral florbinite in 2019 inside the Karst "
        "caves of Tellurion. Florbinite glows pale green only when exposed to moonlight, "
        "and engineers use it to calibrate deep-sea sonar arrays.",
        "What color does florbinite glow, and what is it used for?",
        "green",  # also expect 'sonar'
    ),
    (
        # Counterfactual: a made-up tall building. Reciting memory cannot give 712m/copper/Drax.
        "In the records of New Avalon, the Aurelian Spire was designed by the architect "
        "Penelope Drax in 1962. It stands exactly 712 metres tall and is built entirely "
        "from polished copper, making it the tallest copper structure ever raised.",
        "Who designed the Aurelian Spire and how tall is it?",
        "712",  # also expect 'drax'
    ),
]


def parse(argv):
    if not argv:
        return DEFAULT
    out = []
    for a in argv:
        lbl, path = a.split("=", 1) if "=" in a else (a, a)
        out.append((lbl, path))
    return out


def build_prompt(decoder_tok, background, question):
    user = f"\n\nBackground:{background}\n Question: {question}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    if getattr(decoder_tok, "chat_template", None):
        return decoder_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    return f"{SYSTEM_PROMPT}\n{user}\nAnswer:"


@torch.inference_mode()
def answer_with_ctx(pisco, text, question, max_new_tokens=64):
    """Compress `text` into <MEM> tokens and answer using ONLY the compressed form."""
    ctok, dtok = pisco.compressor_tokenizer, pisco.decoder_tokenizer
    comp_ids = ctok(text, add_special_tokens=False, truncation=False)["input_ids"]
    chunks = chunk_list(comp_ids, chunk_length=512, chunk_overlap=0)
    chunks_with_mems, n_mems = add_memory_tokens_to_inputs(chunks, ctok, pisco.compr_rate)
    total_mems = int(sum(n_mems))

    prompt = build_prompt(dtok, dtok.mem_token * total_mems, question)
    dec = dtok(prompt, return_tensors="pt", add_special_tokens=False)

    comp_pad = ctok.pad({"input_ids": chunks_with_mems}, padding="longest", return_tensors="pt")
    embeddings = pisco.compress(comp_pad["input_ids"].cuda(), comp_pad["attention_mask"].cuda())
    dec_embeds = pisco.replace_embeddings(embeddings, dec["input_ids"].cuda())
    out_ids = pisco.decoder.generate(
        inputs_embeds=dec_embeds,
        attention_mask=dec["attention_mask"].cuda(),
        do_sample=False, top_p=None, max_new_tokens=max_new_tokens,
    )
    return dtok.batch_decode(out_ids, skip_special_tokens=True)[0].strip(), total_mems


@torch.inference_mode()
def answer_no_ctx(pisco, question, max_new_tokens=64):
    """Control: ask the SAME question with NO text at all (no <MEM> tokens)."""
    dtok = pisco.decoder_tokenizer
    prompt = build_prompt(dtok, "", question)
    dec = dtok(prompt, return_tensors="pt", add_special_tokens=False)
    out_ids = pisco.decoder.generate(
        input_ids=dec["input_ids"].cuda(),
        attention_mask=dec["attention_mask"].cuda(),
        do_sample=False, top_p=None, max_new_tokens=max_new_tokens,
    )
    full = dtok.batch_decode(out_ids[:, dec["input_ids"].shape[1]:], skip_special_tokens=True)
    return full[0].strip()


@torch.no_grad()
def run(name, ckpt):
    print(f"\n================ {name} ================")
    print(f"[load] {ckpt}")
    pisco = PISCO.from_pretrained(ckpt, load_decoder=True).cuda().eval()
    print(f"[ok] compr_rate={pisco.compr_rate}")
    faithful = 0
    for text, q, expect in QA:
        pred, nmem = answer_with_ctx(pisco, text, q)
        ctrl = answer_no_ctx(pisco, q)
        with_ok = expect.lower() in pred.lower()
        ctrl_leak = expect.lower() in ctrl.lower()  # model already knew -> bad test item
        good = with_ok and not ctrl_leak
        faithful += good
        print(f"\n  Q       : {q}")
        print(f"  EXPECT  : contains {expect!r} (fictional/counterfactual)")
        print(f"  [{nmem} mem tokens]")
        print(f"  WITH ctx: {pred[:280]}")
        print(f"  NO ctx  : {ctrl[:280]}")
        verdict = ("USES COMPRESSION" if good
                   else "LEAK: knew w/o text" if (with_ok and ctrl_leak)
                   else "FAIL: wrong even with text")
        print(f"  -> {verdict}")
    print(f"\n  >>> {name}: {faithful}/{len(QA)} answered correctly FROM the compressed text")
    del pisco
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for name, ckpt in parse(sys.argv[1:]):
        try:
            run(name, ckpt)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[ERROR] {name}: {type(e).__name__}: {e}")
