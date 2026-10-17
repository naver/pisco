# Latency and throughput of multimodal PISCO

Measured 2026-10-15/16 on one A100-SXM4-80GB, Qwen3.5-4B backbone, bf16.
Question: how much faster is answering from compressed `<MEM>` vectors than from the raw images, and when does
compression stop paying?

## Bottom line

- **Compression is done once per image, offline** (about 75 ms per image on HF), like building a RAG index.
  It is not counted in the query-time numbers below.
- **Short answers, any load:** the first token arrives 2.6 to 9 times sooner for a single request, and 9 to 22 times
  sooner for 32 requests at once (HF).
- **Many requests at once (vLLM):** PISCO serves 9 to 19 times more requests per second than raw images, and
  5 to 10 times more than raw prompts whose image features are already cached.
- **Long answers, one request at a time:** the gain disappears. It falls under 20% once the answer is longer than
  about 1/30 of the prompt tokens saved: about 70 tokens with 8 frames or pages, about 190 with 16.
- **Accuracy is unchanged by serving with vLLM:** the same answers as HF on 919/928 nav and 476/500 doc questions.

## Setup

**Prompt lengths.** Raw = the uncompressed model reading the images. PISCO = `<MEM>` vectors (25 per nav frame,
55 per document page at the 451K-pixel budget).

| setting | images per question | raw tokens | PISCO tokens |
|---|---|---|---|
| nav frames | 4 / 8 / 16 | 1647 / 3231 / 6405 | 179 / 295 / 533 |
| doc pages | 4 / 8 / 16 | 1815 / 3567 / 7061 | 299 / 535 / 1011 |

**Model detail that matters.** Qwen3.5 is a hybrid model: only 8 of its 32 layers keep a context that grows with the
prompt (the KV cache); the other 24 keep a fixed-size state per request. Compression shrinks the first kind only.

**What each phase costs.**
- *Reading the prompt* (prefill) is real computation, proportional to the number of prompt tokens.
- *Writing the answer* goes one token at a time. Each step reads all model weights from GPU memory once (shared by the
  whole batch) plus every request's stored context. With one request the weights dominate, so prompt length barely
  matters. With many requests the stored contexts dominate, and they are 11 times smaller with PISCO.
- *GPU memory* limits how many requests run together; smaller contexts let more of them fit.

## Results with HuggingFace generate

Script `scripts/bench_mm_latency.py` (launcher `bench_mm_4b.sh`), results `expQ/bench/{nav,doc}_{base,pisco_*}.json`.
LoRA merged, exactly 32 generated tokens. Raw timings include the vision encoder.

| setting | batch | first token, raw → PISCO | requests/s, raw → PISCO | peak GPU memory, raw → PISCO |
|---|---|---|---|---|
| nav, 8 frames | 1 | 262 → 55 ms | 0.79 → 0.91 | 9.1 → 18.2 GB |
| nav, 8 frames | 32 | 8.2 → 0.48 s | 3.2 → 19.2 | 30.0 → 19.5 GB |
| nav, 16 frames | 32 | 17.6 → 0.80 s | 1.6 → 15.9 | 50.9 → 20.8 GB |
| doc, 4 pages | 1 | 160 → 53 ms | 0.88 → 0.93 | 8.8 → 18.3 GB |
| doc, 16 pages | 32 | 19.9 → 1.5 s | 1.4 → 12.0 | 55.3 → 23.4 GB |

PISCO's 18 GB floor is the compressor and the decoder both loaded. With offline compression, a query server only
holds the decoder (about 9 GB).

HF spends about 35 ms per generated token, mostly Python overhead; vLLM spends about 7 ms. Use vLLM for serving
numbers.

## Results with vLLM (real PISCO)

Scripts `scripts/vllm_prep_pisco.py` + `scripts/bench_vllm.py` (launchers `vllm_prep_4b.sh`, `bench_vllm_4b.sh`),
results `expQ/bench/vllm/{nav,doc}_{pisco,raw_lmonly,raw_pixels}.json`. 32 generated tokens; N requests submitted
together, scheduled by vLLM's continuous batching.

Three arms:
- **raw images:** Qwen3.5-4B with the images; the vision encoder runs for every request (each image made unique so
  no cache helps).
- **raw, features cached:** the same decoder fed prompts of the raw length; the vision encoder cost removed.
- **PISCO:** the PISCO decoder fed the compressed prompts.

Requests per second:

| setting | requests at once | raw images | raw, features cached | PISCO |
|---|---|---|---|---|
| nav, 8 frames | 1 | 2.0 | 2.8 | 3.9 |
| nav, 8 frames | 64 | 3.9 | 7.2 | 54.9 |
| nav, 8 frames | 256 | 3.7 | 6.9 | 58.5 |
| nav, 16 frames | 64 | 1.9 | 3.5 | 36.0 |
| doc, 4 pages | 256 | 6.6 | 12.0 | 58.8 |
| doc, 16 pages | 64 | 1.8 | 3.2 | 21.5 |

Raw throughput stops growing from about 64 requests: the GPU is saturated reading prompts (about 22,000 prompt
tokens per second). Running the vision encoder per request roughly halves raw throughput again.

**Accuracy check.** Same eval sets, greedy decoding: vLLM and HF give identical answers on 919/928 nav and 476/500
doc questions; the differences are near-ties from numerical noise.

## When compression stops paying

Script `scripts/bench_vllm_sim.py` (plain Qwen3.5-4B in vLLM, random prompts at the raw and at the PISCO length:
compute depends only on length), analysis `scripts/show_vllm_glen.py`, results `expQ/bench/vllm/nav_sim_glen.json`.

How many times faster PISCO is, nav with 8 frames, by answer length and number of requests at once:

| requests at once | 8 tokens (a few words) | 32 (a sentence) | 128 (a paragraph) | 512 (a page) | 2048 (four pages) |
|---|---|---|---|---|---|
| 1 | 2.2× | 1.4× | 1.1× | 1.0× | 1.0× |
| 16 | 8.1× | 5.0× | 2.5× | 1.5× | 1.2× |
| 64 | 9.5× | 7.7× | 4.5× | 2.3× | 1.6× |
| 256 | 9.3× | 8.5× | 6.1× | 3.2× | 1.8× |

Answer length after which PISCO is less than 20% faster:

| requests at once | 4 frames | 8 frames | 16 frames |
|---|---|---|---|
| 1 | 30 tokens | 72 tokens | 177 tokens |
| 16 | 670 | 2160 | never* |
| 64 | 5300 | never* | never* |
| 256 | never* | never* | never* |

\* never within the fitted model; the measured gain at 2048 tokens stays at 1.2× or more.

Doc pages (`expQ/bench/vllm/doc_sim_glen_partial.json`, recovered from the job log: the run hit its time limit before
the batched 16-page cases). Answer length after which PISCO is less than 20% faster:

| requests at once | 4 pages | 8 pages | 16 pages |
|---|---|---|---|
| 1 | 31 tokens | 76 tokens | 195 tokens |
| 16 | 690 | 2250 | not run |
| 64 | 6100 | never* | not run |
| 256 | never* | never* | not run |

Doc gains for short answers are smaller than nav (6 to 7× instead of 9 to 12× at 256 requests) because a page keeps
55 `<MEM>` vectors and a frame 25, but the break-even points are almost the same.

**Rule of thumb for one request.** What matters is how many prompt tokens compression saves, not the ratio. On this
setup, generating one token takes as long as reading about 160 prompt tokens (7.1 ms against about 44 µs per prompt
token). So PISCO stays at least 20% faster while the answer is shorter than about 1/30 of the tokens saved:
- 8 frames save about 2900 tokens → about 100 tokens of answer by the rule, 72 measured;
- 16 pages save about 6000 tokens → about 200 by the rule, 195 measured.

For very long answers the gain settles at 1.0× for one request and 1.2 to 2.3× for 256 requests. On a model with
full attention in every layer the long-answer gain should be larger (all layers would keep a growing context); not
measured here.

## Running PISCO in vLLM

No change to vLLM 0.19.1 is needed (env: `Reranking/vllm_env`).

1. **Export the decoder** (`scripts/vllm_prep_pisco.py`): merge the LoRA, keep the language-model weights renamed
   `model.*`, drop the vision tower, write a text-only config without the `mrope_*` rotary keys. vLLM's image class
   for Qwen3.5 refuses prompt embeddings (it needs token ids to place image positions); for text-only input the
   two position schemes are identical.
2. **Register the text-only class with the hybrid hooks** (`scripts/bench_vllm.py`): `Qwen3_5ForCausalLM` ships with
   vLLM but is unregistered and lacks the hooks that size the attention and linear-attention caches together.
   They are borrowed from the image class.
3. **Send prompt embeddings:** compress each image offline, build the chat prompt with `<MEM>` placeholders, replace
   them with the compressed vectors (`PISCO.replace_embeddings`), pass `{"prompt_embeds": tensor}`.

Launcher settings: `C_INCLUDE_PATH=/usr/local/cuda/include` (Triton compiles a launcher that needs `cuda.h`) and
`VLLM_DISABLE_COMPILE_CACHE=1` (a graph compiled for token ids was reused for embeddings and crashed).

## Pitfalls found

- **LoRA must be merged for speed.** PISCO's adapters are transformers-integrated, without `merge_and_unload`; the old
  `--merge_lora` flag silently did nothing (56 instead of 31 ms per token on HF). Use
  `pisco.model.merge_lora_inplace`: same greedy answers (198/200 identical on nav), base-model speed.
- **Warm up every shape.** Qwen3.5's Triton kernels compile per new shape; the first call can take seconds.
  All benchmarks run one untimed call per configuration.

## Reproduce

```bash
expmon launch --exp mmpisco --stage bench bench_mm_4b.sh            # HF
expmon launch --exp mmpisco --stage bench vllm_prep_4b.sh           # export + prompt embeddings
expmon launch --exp mmpisco --stage bench bench_vllm_4b.sh          # vLLM, three arms + accuracy check
SUF=_glen MML=8704 expmon launch --exp mmpisco --stage bench bench_vllm_sim_4b.sh 1,16,64,256 1,8,32,128,512,2048
python3 scripts/show_vllm_glen.py expQ/bench/vllm/nav_sim_glen.json
```
