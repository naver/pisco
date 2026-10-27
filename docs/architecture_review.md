# Is PISCO's design right? Literature review and probes (archq, October 2026)

Questions from the owner (2026-10-21): is the current architecture OK? Should it be redesigned from scratch? What are
its limits (originally: latency)? Can it be combined with other approaches (KV cache, ...)?

Method: four literature reviews (text compression, visual-token compression, KV cache, limits of soft compression;
about 100 papers from 2023 to October 2026, arXiv ids checked) and 25 cheap jobs on the best model
`expQ/ft_mm_4b_173550/model` (inference only, no training), expmon campaign `archq`. All accuracy numbers: 1,500
document questions (500 InfographicVQA, 500 MP-DocVQA, 500 SlideVQA), rubric judge, paired with the uncompressed
Qwen3.5-4B on the same questions. Context: [multimodal_results.md](multimodal_results.md), [latency.md](latency.md).

## Short answers

| question | answer |
|---|---|
| Is the architecture OK? | **Yes.** The best 2025-26 systems put compressed vectors where PISCO does (the decoder's input embeddings). On infographics PISCO keeps more (56%) than every method that compresses without the question (35-45%). |
| Redesign from scratch? | **No, extend it.** One change at query time closes the document gap without training: the model points to the page that holds the answer, and that one page is given raw. Accuracy then equals the uncompressed model (.801 vs .809) with 22-36% of its prompt tokens. |
| Main limits | (1) dense pages lose content in compression; (2) on multi-page documents the decoder fails to use what the vectors hold; (3) speed gains need long inputs with short answers or many requests, and Qwen3.5's fixed per-request state caps them; (4) one compression rate; (5) a 4B compressor and 0.6 MB of vectors per page. |
| Combine with KV-cache methods? | A little. A float16 recurrent state is a free +9-14% for long answers. fp8 KV and MTP speculative decoding are blocked on this setup (A100, vLLM 0.19.1). A full-attention decoder gains much more from compression than the hybrid Qwen3.5. |

## Probes

### 1. Expand the right page: the document gap closes

The decoder gets every page as `<MEM>`s except one, which it gets as the raw image. "Self-routed": the PISCO model
first answers "Which document contains the answer to this question?" from the `<MEM>`s alone (the stage-2 `find`
task, never trained on questions), then the chosen page is expanded. Jobs 174849-174851, 174869-174874.

| pages per question | PISCO, all `<MEM>` | PISCO, self-routed page raw | PISCO, gold page raw | uncompressed | right page picked | prompt tokens: routed / raw |
|---|---|---|---|---|---|---|
| 4 | .578 | **.801** | .811 | .809 | 98% | 1,264 / 3,505 |
| 8 | .574 | **.803** | - | .807 | 97% | 1,711 / 6,945 |
| 10 | .585 | **.796** | - | .801 | 97% | 1,934 / 8,665 |

- Self-routed minus uncompressed: −.007 (95% CI −.021 to +.007, p = .35) at 4 pages; −.005 (−.021 to +.012, p = .64)
  at 10 pages. No difference can be detected.
- The PISCO decoder reads raw pages although finetuning never showed it one: all 4 pages raw gives .804 (base .809).
- Per source at 4 pages: infographics .706 (uncompressed .726), multi-page .916 (.926), slides .782 (.774).
- Same-document distractors: a quarter of the 4-page set already has its 3 distractors from the gold page's own deck
  or document (slides 315/500, multi-page 64/500; infographics are single pages). On those 379 questions
  route-then-expand with the connector norm scores .818 vs uncompressed .805 (+.013, CI −.024 to +.050); routing drops
  from 99% to 95% (multi-page 92%, −.031 on 64 Q, CI −.109 to +.047). Split: `expQ/archq/hardneg_split.py`.
- Untested: retrieved negatives, and more than 10 candidates (the routing score reads one digit). VisRAG's corpora hold
  at most 9 slides per deck and 15 pages per document, so a whole-document test needs the original SlideVQA /
  MP-DocVQA documents.

Speed (vLLM simulation at these prompt lengths, A100, job 174875). Route-then-expand = one short routing pass plus the
answer pass:

| pages | requests at once | raw | all `<MEM>` (71% of accuracy) | route-then-expand (99%) |
|---|---|---|---|---|
| 4 | 1 (first token) | 1× | 4.4× | 1.6× |
| 4 | 256 (32-token answers) | 1× | 5.9× | 2.0× |
| 8 | 1 (first token) | 1× | 5.7× | 2.3× |
| 8 | 256 (32-token answers) | 1× | 6.7× | 2.7× |

The vision encoder is not in these numbers; raw runs it on every page, route-then-expand on one page, so the real gain
over raw is larger. The expanded page must be available at query time: either the image (vision encoder at query
time, about 25 ms per page) or its cached vision features (about 5 MB per page).

### 2. Where the answer gets lost: compression or use?

Each gold page is transcribed twice: from its pixels by the uncompressed model (reference), and from its `<MEM>`s alone
by the stage-2 model (its image-autoencoding task). Questions PISCO answers wrong although the answer is in the
reference, 360 pages (jobs 174859-174861):

| source | words of the page rebuilt from `<MEM>`s | answer still in the `<MEM>`s ("use" failure) | answer lost in compression |
|---|---|---|---|
| infographics | 58% | 17 | **25** |
| multi-page documents | 83% | **27** | 6 |
| slides | 70% | 10 | 8 |
| all | 71% | 54 | 39 |

- Infographics mostly lose content: a capacity problem.
- Multi-page documents mostly keep it and the decoder fails to use it: a training problem.
- When the answer is rebuilt PISCO is right 71% of the time, when it is not 52%.
- With the finetuned compressor in front of the same decoder: 47 use failures, 46 losses (same picture).

### 3. Other cheap probes (4 pages, paired with the best model .578)

| change | result | reading |
|---|---|---|
| prompt "copy the evidence, then answer" | .603 (+.025, CI +.005 to +.045) | the use problem is partly a prompting problem; RL on this format is the next step |
| only the gold page (no distractors) | .592 (+.014, n.s.); uncompressed .841 | distractor pages are not what hurts |
| rate-4 finetune (twice the `<MEM>`s, job 174693) | .612 (+.034, CI +.015 to +.055); photos .670 vs .665 (n.s.) | more vectors everywhere buy a little, at 1.8× the tokens |
| rate-8 model read at rate 4 / rate 16 | +.011 / −.070 | the rate is learned: changing it needs training |
| `<MEM>`s stored in int8 / int4 | +.002 / −.018 | int8 is free and halves storage (0.31 MB per page) |

### 4. Serving and backbone

**Full-attention backbone** (vLLM simulation, same prompt lengths, jobs 174844-174845). Speed-up of PISCO ×8 over
raw:

| pages | requests | 1 token | 128 tokens | 512 tokens |
|---|---|---|---|---|
| 8 | 256 | 6.8× on Qwen3.5-4B / 7.6× on Qwen3-4B | 4.9× / 6.8× | 3.0× / 5.0× |
| 16 | 256 | 7.4× / 8.9× | 6.2× / 8.4× | 4.5× / 6.9× |
| 16 | 1 | 5.9× / 7.9× | 1.3× / 1.4× | 1.1× / 1.2× |

Qwen3.5 keeps a fixed float32 state of about 51.5 MB per request in its 24 linear-attention layers (as much memory as
1,570 tokens of KV cache); compression cannot shrink it, so once the prompt is short it is pure overhead. PISCO prompts
even run faster in absolute terms on Qwen3-4B (30.0 vs 25.5 requests/s, 8 pages, 256 requests, 128 tokens).

**Serving options on the PISCO decoder** (vLLM 0.19.1, A100, real `<MEM>` prompt embeddings, job 174846):

| option | result |
|---|---|
| linear-attention state in float16 | +9-14% requests/s for 256-token answers under load, +0-5% for short answers; 481/500 identical greedy answers (a backend swap alone: 485/500) |
| fp8 KV cache | not possible on A100: this path accepts only fp8 e4m3, which Triton cannot compile for sm80 (needs H100) |
| MTP speculative decoding (Qwen3.5-4B ships an MTP head) | refused with prompt embeddings: the draft head needs the prompt's token ids, `<MEM>` positions have none (needs a vLLM patch). On token prompts (job 174906): one request 7.1 → 4.9 ms per token (1.45×), 16 requests 1.3×, 64 requests none (0.95×); 69% of drafted tokens accepted. It would help exactly where compression does not: one request, long answers |
| prefix caching | prompt embeddings are hashed per block, but Qwen3.5 caches whole 528-token blocks only, so a PISCO prompt reuses at most one block (from the vLLM source) |

**Compression cost per page** (HF, A100, 1M-pixel page, batches of 8, job 174862): Qwen3.5-4B 65.6 ms (vision tower
25.5 + language model 40.2), 2B 43.2 ms (same vision tower), 0.8B 21.0 ms. A 0.8B compressor would index 3× faster.

## What the literature says

**Where the vectors enter.** Input embeddings, as in PISCO: OSCAR (2504.07109), ARC-Encoder (2510.20535), CLaRa
(2511.18659), REFRAG (2509.01092), LCLM (2606.09659), DEX-Comp (2609.05152), AdaMem (2609.22100). Per-layer KV
injection (500xCompressor 2408.03094) wins only with a frozen decoder, and would cover 8 of Qwen3.5's 32 layers.
Cross-attention readers (CEPE 2402.16617) are slower than REFRAG. DeepSeek-OCR 2 (2601.20552) moved to an LM encoder
with causal query tokens, i.e. PISCO's `<MEM>` design.

**Compressor size.** Small is enough for text: 0.6B about equals 4B (LCLM); a 1B compressor beats the decoder's own
layers (OSCAR).

**Budget.** Uneven budgets beat uniform ones: AdaMem (+1.9 at 16×, +7 at 64×), per-image budgets (ERASE 2605.09982,
VisionThink 2507.13348). Expanding a few chunks back to raw input is REFRAG's idea (an RL policy picks them); here the
model's own `find` skill does it with no training. One-rate models do not transfer to other rates (ARC; TokenFLEX
2504.03154), matching probe 3.

**Training: the largest lever in the literature.** RLComp (ACL 2026) names a "memorization-utilization gap":
reconstruction from the vectors is near perfect while supervised finetuning stays 7.5 points under the full context;
RL with long answers, which re-copy the relevant text before answering, closes most of it (+6.5). DEX-Comp keeps
PISCO's architecture: distillation on rows the teacher got right, then RL on the rows it got wrong, beats the
uncompressed model (72.0 vs 69.5; PISCO 64.8). Probe 2 shows this gap is real on multi-page documents.

**Images.** Retention at about 8×: photos 90%+, scene text 88-93%, infographics 35-45% without the question
(VisPruner 2412.01818, DART 2502.11494) and 70-94% only for methods that read the question with the full image at query
time (ERASE, GlimpsePrune 2508.01548), which cannot be precomputed. Rendering text as images to compress it
(DeepSeek-OCR 2510.18234, Glyph 2510.17800) does not beat text compression on plain text (2512.03643).

**KV cache.** Eviction and quantization (SnapKV 2404.14469, PyramidKV 2406.02069, KIVI 2402.02750) shrink the cache
after a full prefill; PISCO also removes the prefill. Per-document KV reuse (TurboRAG 2410.07590, Block-Attention
2409.15355, CacheBlend 2405.16444) removes the prefill of documents; on a hybrid model it needs the linear-attention
state too, exact with a stored transition matrix (Hypic 2607.01299) but about 50 MB per page on Qwen3.5-4B
(8 MB on Qwen3-4B).

## What to change, in order

| # | change | cost | expected effect |
|---|---|---|---|
| 1 | **Make route-then-expand the document-QA mode**; next test it on hard negatives (same-document and retrieved pages) and on more than 10 pages (score the full "Document j" answer) | inference only, a few GPU-hours | uncompressed accuracy at 2-3× the raw speed; keep all-`<MEM>` for photos and for speed-first use |
| 2 | **RL with an evidence-then-answer format** (RLComp, DEX-Comp), for the use failures | a GRPO run, about 1-2 GPU-days | the prompt alone gives +.025; RL targets the 54 of 93 failures where the answer is in the vectors |
| 3 | **Train the decoder on mixed raw + `<MEM>` prompts**, and the router on questions | one finetune | removes the zero-shot caveats of #1 |
| 4 | Store `<MEM>`s in int8; serve with a float16 linear-attention state | none | half the storage; +9-14% throughput on long answers |
| 5 | If speed matters more than reuse of Qwen3.5: a **full-attention VLM decoder** (e.g. Qwen3-VL-4B); it must stay a VLM, since #1 shows it raw pages | stage-2 + finetune, about 2-3 GPU-days | 5.0× instead of 3.0× at 512-token answers under load |
| 6 | A **0.8B compressor** | stage-2 + finetune | 3× cheaper indexing; risk on dense pages (smaller vision tower) |

Not worth it now: per-layer KV injection, cross-attention readers, rendering text as images, per-document decoder-state
caching on Qwen3.5 (50 MB per page), more `<MEM>`s everywhere (rate 4: +.034 for 1.8× the tokens).

## Reproduce

- Evals: `eval_mm_4b.sh` with env `RAW=gold|route1|route2|all`, `STYLE=expand`, `GOLD=1`, `MQ=int8|int4`,
  `XD=<n> TOPK=<4+n>`, `DML=<max prompt>` (`scripts/eval_mm_docqa.py` `--raw_policy`, `--answer_style`, `--gold_only`,
  `--mem_quant`, `--extra_distractors`). Launch lists: `expQ/archq/launch_wave1.sh`, `expQ/archq/launch_wave2.sh`.
- Tables: `python3 scripts/show_archq.py --ref <best eval json> <eval jsons…>`; `python3 scripts/show_backbone_sim.py`.
- Transcription probe: `probe_recall_4b.sh ref|ae_s2|ae_ftcomp` then `scripts/analyze_mem_recall.py`.
- Serving: `bench_combo_4b.sh ref,tri,ssm16,fp8,mtp,mtptext`; compression cost: `bench_compress_4b.sh`.
