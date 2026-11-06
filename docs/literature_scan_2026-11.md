# Literature scan, November 2026: what to try next after the position-layout win

Scope: papers from 2025-2026 first, then older ones the earlier reviews missed. Each entry says what the paper
does, what it measured, which of our eight weaknesses it touches (W1 use gap on documents, W2 rate / graceful
degradation, W3 long documents and long memories, W4 routing, W5 hierarchy, W6 objectives and RL, W7
query-dependent vs generic, W8 hour-long video), and the cheapest test in our system. Papers already used in
[mem_tokens_review.md](mem_tokens_review.md), [training_review.md](training_review.md),
[data_review.md](data_review.md), [architecture_review.md](architecture_review.md) and
[hierarchical_video.md](hierarchical_video.md) are listed at the end and only re-cited when a new angle appears.
Numbers are the papers' own; none is at our budget or on our questions unless said.

## 1. Summary

Five ideas stand out. (1) **Multi-position slots** ([PPE](https://arxiv.org/abs/2510.22936), ICLR 2026): a merged
token keeps several position ids by splitting its RoPE channels among them. It is the direct generalisation of the
layout that just gave us +.08 and it lifted DocVQA from 52 to 77 for a merging method at 55% token reduction; it is
an inference-only probe here. (2) **Per-head attention scales for reading the slots**
([SEAL](https://arxiv.org/abs/2501.15225), ACL 2025): one scalar per head, learned on 50 synthetic samples in under
an hour, moved long-context retrieval from .32 to .88; it targets the "answer is in the slots, decoder does not use
it" gap with ~1k parameters. (3) **Answer-aligned slots** ([CMC](https://arxiv.org/abs/2609.25537)): a contrastive
term pulls the mean memory embedding toward the teacher's hidden state at the answer span, and the decoder picks the
top-K memory embeddings at inference; it is a one-term change to our stage-2 KD and a built-in router.
(4) **Routing without training**: the decoder's own attention mass on each page's slots as a page picker
([AttentionRetriever](https://arxiv.org/abs/2602.12278)), plus a tiny probe on (query, slot) features that predicts
whether the slots still hold the answer (AUC .72 in [overflow detection](https://arxiv.org/abs/2602.12235)).
(5) **Nested budgets by prefix training**: causal slots are already prefix-nested
([No Mean Feat](https://arxiv.org/abs/2510.20797)); the Matryoshka speech work
([Llama-MTSK](https://arxiv.org/abs/2503.06362), [MoME](https://arxiv.org/abs/2510.04136)) shows that scale-specific
LoRA pieces keep the harsh rates from degrading, which is the training signal our coarse level lacks.
Two negative findings matter too: content-driven budgets did no better than uniform ones
([consequence-sensitive compression](https://arxiv.org/abs/2608.09176)), matching our closed density-adaptive probe,
and query-aware cache eviction broke in multi-query settings while reconstruction-based, query-agnostic scoring held
([KVzip](https://arxiv.org/abs/2505.23416)), an argument for generic slots plus query-time selection over the OSCAR path.

## 2. Ranked shortlist

| rank | paper | idea | W | expected effect, confidence | cheapest test here | cost |
|---|---|---|---|---|---|---|
| 1 | [PPE](https://arxiv.org/abs/2510.22936) 2510.22936, ICLR 2026 | each slot carries K position ids over its segment / tile, one per RoPE channel group | 1, 2, 3a | +.02 to +.05 documents; medium (their training-free gain was +1.6 DocVQA on a good merger, +46 on a bad one) | inference probe on the spread finetune: K ∈ {2, 4, 8} ids per slot, text first (1-D RoPE), then 2-D tiles; paired judge on 1,500 Q | half a day of code (custom rotary for per-group ids), 2 eval jobs per K |
| 2 | [SEAL](https://arxiv.org/abs/2501.15225) 2501.15225, ACL 2025 | learn one scalar per attention head (or per channel) that emphasises retrieval heads; 50 synthetic samples, 1 epoch | 1, 4 | +.01 to +.03 documents; medium | freeze everything, learn L×H scales on 50-200 rows with the teacher label, merge into `o_proj`; paired judge | ~1 GPU-hour training + 1 eval |
| 3 | [CMC](https://arxiv.org/abs/2609.25537) 2609.25537 | contrastive term: mean of a chunk's memory embeddings → teacher hidden state at the answer span; top-K memory selection at inference | 1, 4, 6 | +.01 to +.03; medium-low (their Mistral arm regressed) | probe first: cosine between gold-page slot mean and the teacher's answer-span state vs distractor pages (no training); then one stage-2 arm adding the term | probe hours; arm = one stage-2 run |
| 4 | [AttentionRetriever](https://arxiv.org/abs/2602.12278) + [overflow detection](https://arxiv.org/abs/2602.12235) 2602.12278 / 2602.12235 | route by decoder attention mass per page; gate by a probe on (query, slot) features | 4, 2 | pick@1 +.03 to +.08 over the digit-routing prompt; medium | compute attention mass on each page's slots at the first answer token over 1,500 Q; logistic probe on [query embedding, slot mean] with label "pisco right / raw right" | one forward pass per Q, CPU probe |
| 5 | [MoME](https://arxiv.org/abs/2510.04136) / [Llama-MTSK](https://arxiv.org/abs/2503.06362) 2510.04136 / 2503.06362 | Matryoshka training over token scales with scale-specific LoRA; mixture of scale experts | 5, 2 | 50% prefix cost from −.13 to about −.05; medium | first-k curve on the spread model (already planned); then a stage-1 arm with random prefix keep {25, 50, 100%} and a level token | arm = one stage-1 run (days) |
| 6 | [CTRL-RAG](https://arxiv.org/abs/2603.04406) 2603.04406 | reward = log p(answer with evidence) − log p(answer without) | 6 | denser signal for round 2; medium | add to the planned GRPO loop as a second reward: log p(label \| slots) − log p(label \| no slots) | zero extra compute beyond round 2 |
| 7 | [Rethinking efficient attention in hybrids](https://arxiv.org/abs/2606.15378) 2606.15378 | full-attention layers do the long-range retrieval in hybrid models | 1, backbone | small; medium-high | set `compressor_extra_layers` to the full-attention layer indices of Qwen3.5 (not evenly spaced ones) and rerun the archq2 read-out probe | existing code, 1 eval |
| 8 | [Causal2Vec](https://arxiv.org/abs/2507.23386) / [echo embeddings](https://arxiv.org/abs/2402.15449) 2507.23386 / 2402.15449 | prepend a contextual token from a tiny bidirectional encoder, or repeat the chunk, so causal slots see "the future" | backbone, 1 | +.01 to +.02; low-medium | echo variant needs no new module: compressor input = chunk, chunk, `<MEM>`s (2× compressor cost); stage-1 arm | one stage-1 run |
| 9 | [single-dimension scaling](https://arxiv.org/abs/2406.02536) / [Ms-PoE](https://arxiv.org/abs/2403.04797) / [CALIOPE](https://aclanthology.org/2026.findings-eacl.120) | training-free positional calibrations of the decoder | 1, 3b | ≤ +.02; low | run the released code on the decoder with `<MEM>` prompts; paired NLL on held-out rows first | hours |
| 10 | [TimeSuite TAPE](https://arxiv.org/abs/2410.19702) / [XComp](https://arxiv.org/abs/2604.14149) | temporal position added to compressed video tokens; local-attention segments against start/end bias | 8, 3b | grounding +1.6 to +3.7 in their setting; low-medium here | video only: give frame slots time-axis M-RoPE ids (already have the 2-D rule) and compare order probes | 1 eval on OpenEQA / hour-long set |

## 3. Per-domain notes

### 3.1 Text compression with memory tokens; position and attention layout

[PPE: Positional Preservation Embedding](https://arxiv.org/abs/2510.22936) (ICLR 2026, code). When several tokens
are merged into one, PPE does not average their positions: it splits the merged token's RoPE channels into K groups and
rotates each group with the position of one source token (K = 8 for video, 32 for images under M-RoPE). Applied to
Chat-UniVi merging at 55% reduction on Qwen2.5-VL-3B, DocVQA went 52.5 → 76.8, ChartQA 49.6 → 74.5, TextVQA 57.7 →
77.1; training-free on VisionZip it still added +1.6 DocVQA and +1.0 ChartQA. This is EPL taken one step further: a
slot can own a span, not a point. For us: our spread layout gives one id (the segment centre); PPE would give each
`<MEM>` 2-8 ids across its 8-token segment (text) or across its tile (image). It is an inference-only probe, and it
speaks to W3a (a long chunk's slots could cover wide spans) as well as W1.

[CMC: answer-aligned memory embeddings](https://arxiv.org/abs/2609.25537) (code). A small encoder (GPT2-L to
OPT-2.7B) writes memory embeddings for 256-token chunks at rate 2-8 for a frozen decoder (Llama-3-8B, Mistral-7B,
Gemma-2-9B). The loss has three terms: answer cross-entropy, KL to the decoder reading the raw window, and a
contrastive term that pulls the mean memory embedding toward the teacher's hidden state at the answer span. At inference
the top-30/40 memory embeddings are selected. SQuAD F1 78.3 vs ICAE 45.7 and xRAG 18.2 in their full-data run, but
Mistral regressed (69.3 → 55.2), so the recipe is fragile. Relevance: the contrastive term is a slot-level loss we do
not have (W6), and top-K memory selection is routing over slots (W4). Probe first, train second.

[Detecting overflow in compressed token representations](https://arxiv.org/abs/2602.12235) (EACL 2026 SRW, code).
"Overflow" = the raw text answers the question but the compressed token does not. A logistic probe on the query
embedding plus the compressed xRAG token reaches AUC .72 (TriviaQA .725, SQuADv2 .703, HotpotQA .720); saturation
statistics of the token alone are near chance (.53-.58). For us this is a cheap gate for route-then-expand: predict
from (question, page slots) whether to expand the page raw (W4, W2).

[Memory as a cache (SMem)](https://arxiv.org/abs/2609.32395). Each block is encoded alone; the reader cross-attends
over the union of all blocks' memory rows, so blocks compose and delete exactly. At 4× the training context it finds
planted needles 31-63 blocks away with EM .14-.28 where RoPE / Block-Attention transformers score ≤ .02. Trained from
scratch (160M-1.5B), so not usable directly, but it is the cleanest evidence that block-local encoding plus a reader
trained on many blocks does extrapolate (W3b).

[CoMeT](https://arxiv.org/abs/2602.01766) (ACL 2026, code). A plug-in module keeps a FIFO memory of recent chunks and
a gated global memory, both used as a soft prompt for the next chunk; finetuned at 32k it retrieves a passkey anywhere
in 1M tokens. The "memory as soft prompt, read by the same LLM" design is ours; the gated global memory is the
consolidation step we do not have for long videos (W3b, W8).

[ASTC](https://arxiv.org/abs/2610.41291) (code, Qwen3.5-9B). Training-free in-model compression for agents: on
Gated DeltaNet layers it measures how much of each value was already predicted by the recurrent state (the
"innovation"), pools absorbed spans into soft tokens, keeps identifiers as hard tokens. Matches the uncompressed agent
(AppWorld 33.3 vs 32.1) at half the attention pairs. Relevance: it is the only work that uses the hybrid backbone's
linear-attention state as a signal of what is already absorbed. A probe: the GDN innovation along a chunk as a
budget or position signal for `<MEM>`s (W2, backbone).

[KVzip](https://arxiv.org/abs/2505.23416) (NeurIPS 2025 oral, code). Query-agnostic KV eviction scored by how much
each pair helps reconstruct the context: 3-4× smaller cache with little loss on LLaMA-3.1 / Qwen2.5 / Gemma-3 up to
170k tokens, while query-aware eviction degrades even at a 90% budget once several queries hit the same cache. Two
lessons for W6/W7: reconstruction is a sound query-agnostic importance signal (our AE objective is not the enemy), and
query-dependent compression pays a hidden price when the index is reused.

One line each on EPL follow-ups: [SAC](https://arxiv.org/abs/2510.08907) adopts EPL's position ids for its anchors;
[SeCo](https://arxiv.org/abs/2605.09463) groups by semantics rather than position; the interleaved-gist family
([H-GSA](https://arxiv.org/abs/2604.20920)) adds a meta-gist level. [You don't need long
context](https://arxiv.org/abs/2611.05306) (Nov 2026) argues from cost (late tokens up to 7× dearer) for frequent
compaction; it is context for W8, not a method.

### 3.2 Visual token compression for documents and dense pages

[ET-Prune](https://arxiv.org/abs/2608.01979). Training-free pruning for text-rich images: question-to-visual
attention scores tokens, a text-like spatial prior protects small glyphs, and the keep ratio rises with the entropy of
the attention and the text coverage. On Qwen3-VL-8B at ~50% tokens it keeps 93% of the OCRBench-v2 score at 69% of
the FLOPs. Its diagnosis is ours: a fixed 50% budget left 66 tokens for an 86-token evidence span on the smallest
images. But the budget is set from the question, so this is the OSCAR path (W7), not a generic slot budget.

[Spatial provenance audit](https://arxiv.org/abs/2608.00077) (code). Accuracy hides whether the retained tokens
trace back to the OCR region that supports the answer: at 30% retention on Qwen3-VL-8B a targeted selector and the
full model tie (.786 vs .783) while positive-support coverage is .620 vs .270 for random. This is the visual analogue
of our transcription probe; the method (geometric provenance plus interventions) could score which tiles our slots
actually encode (W1).

[Consequence-sensitive visual token compression](https://arxiv.org/abs/2608.09176). Budgets are allocated by what an
error would cost, read from the question, not the image; a content-driven allocator was no better than uniform on
documents sharing the same images. High-stakes errors fell .300 → .133 at equal total tokens. It confirms our closed
density-adaptive result and says any non-uniform budget must come from the question (W2, W7).

[DocVLM](https://arxiv.org/abs/2412.08746) (CVPR 2025). An OCR encoder compressed into 64 learned queries is added to
a frozen VLM: at 448 px DocVQA went 56.0 → 86.6 (InternVL2) and 84.4 → 91.2 (Qwen2-VL). Dense pages are where our
slots lose most; a text side-channel (OCR of the page, compressed as text chunks by the same compressor) is a
one-collator-change arm that needs no new module (W1). The cost is an OCR pass per page at index time.

[DocSLM](https://arxiv.org/abs/2511.11313) (code) encodes each page's visual, textual and layout features into a
fixed-length sequence and streams pages with an entropy-based abstention; 82% fewer visual tokens at claimed parity.
[PVC](https://arxiv.org/abs/2412.09613) (CVPR 2025) compresses progressively across repeated "frames" of an image
and reports no loss on DocVQA / InfoVQA (DocVQA 92.5, InfoVQA 75.0 on InternVL2-8B). Both are fixed-budget per page;
neither reports a coarse level. [Token-level correlation-guided compression](https://arxiv.org/abs/2407.14439) (2024)
estimates each sub-image's information density from patch correlations and samples tokens accordingly, on DocOwl1.5,
without numbers in the abstract.

### 3.3 Compressed memory for long and streaming video

[FluxMem](https://arxiv.org/abs/2603.02096) (CVPR 2026, training-free): drop tokens repeated across adjacent frames,
then merge repeated regions within a frame, both rates set from scene statistics; MLVU 73.1 with 65% fewer tokens,
StreamingBench 76.4. [LatentStream](https://arxiv.org/abs/2609.04131): tiers of short / mid / long memory consolidated
by Jenks breaks; at query time groups of latent memory tokens with expanding receptive fields pull evidence into a
fixed-length memory, refined with a confidence reward from predictive entropy (no numbers in the abstract). The
"expanding receptive field" tokens are a hierarchy built at read time, the thing our flat slots lack (W5, W8).

[XComp: one token per highly selective frame](https://arxiv.org/abs/2604.14149) (NeurIPS 2025). LLM layers are trained
as progressive compressors down to one token per frame at the last layer; frames are picked by the LLM's attention;
videos are split into short segments with local attention to counter start / end bias. LVBench 42.9 → 46.2 on
VideoChat-Flash with 2.5% of the SFT data. Two ideas for W8: compress inside the decoder rather than before it, and
local attention to stop the long memory collapsing to its ends.

[TimeSuite](https://arxiv.org/abs/2410.19702) (ICLR 2025). Token shuffle (4 tokens merged along channels) plus TAPE, a
learned temporal embedding added to the compressed tokens: without TAPE Charades-STA R@1 48.7 → 47.1 and QVHighlights
54.1 → 50.4; shuffle beats pooling for grounding 48.7 vs 40.3. Position information must be re-attached to compressed
video tokens, which our 2-D tile rule already does for space and could do for time (W8).
[TimeLens](https://arxiv.org/abs/2512.14698) (CVPR 2026) finds interleaved textual timestamps the best way to tell the
model when a frame is; [PAS](https://arxiv.org/abs/2511.10979) smooths M-RoPE's temporal kernel with opposed phase
offsets across heads, training-free (no numbers in the abstract). [STC](https://github.com/lern-to-write/STC)
(CVPR 2026) caches unchanged ViT tokens and prunes to 64 per frame: −45% prefill for 64.4 → 62.5 on OVO real-time;
engineering, noted only. [LiveStarPro](https://arxiv.org/abs/2606.17798) (code) moves evicted history into a
tree of event chains for hour-scale streams (+28.9% semantic correctness on its own benchmark).

### 3.4 Nested / Matryoshka budgets

[Llama-MTSK](https://arxiv.org/abs/2503.06362) trains one audio-visual model at several token rates (pooling or
stacking) with three LoRA layouts: global, scale-specific, or both; it matches models trained at a fixed rate.
[MoME](https://arxiv.org/abs/2510.04136) replaces them with a mixture of Matryoshka experts and reports "notably lower
WER degradation at harsher compression rates" than MTSK. For W5 the transferable part is the training recipe: sample
the kept prefix per step, and give the harsh rates their own small adapter so the fine rate is not dragged down.
[No Mean Feat](https://arxiv.org/abs/2510.20797) already observed that causal compression tokens are prefix-nested by
construction; we only need the decoder to read prefixes, which is the "first-k curve" probe already planned.

[PARCEL](https://arxiv.org/abs/2605.30126): pooled anchor tokens for layout plus elastic query tokens conditioned on
them, one model for several budgets, "consistently outperforms matryoshka baselines" on 27 benchmarks (no numbers in
the abstract). [MM-Matryoshka](https://arxiv.org/abs/2606.07654): ColPali-style retrieval elastic in both vector
width and encoder depth. Relevant if our slots ever double as a retrieval index (W4). [FlexComp](https://arxiv.org/abs/2609.11192)
and TokenFLEX remain the text / image references (covered).

### 3.5 Making the decoder use compressed context

[SEAL](https://arxiv.org/abs/2501.15225) (ACL 2025). Pruning heads one at a time shows some heads help and some hurt
long-context retrieval by ±20%; SEAL learns one scalar per head (1,024 parameters on a 7B) or per channel, on 50
synthetic samples, one epoch, 10-40 minutes, then merges the scales into `o_proj`. LongChat-7B line retrieval .32 →
.88; RULER CWE on Llama-3.1-8B at 64k 0.1 → 95.7; LongBench multi-doc QA 23.3 → 31.3; MMLU unchanged. Scales transfer
between tasks of similar format only. Our use problem is exactly "the information is there, the heads do not look":
a SEAL arm on our decoder is the cheapest training we can do (W1).

Training-free positional calibrations, as a family to probe with released code:
[Scaling a single dimension](https://arxiv.org/abs/2406.02536) (ACL 2025 Findings) finds the hidden channel that
grows with position and scales it in layers 10-25 for the last token: Qwen1.5-7B NQ 56.8 → 58.3, KV retrieval 79.6 →
92.5; [Ms-PoE](https://arxiv.org/abs/2403.04797) rescales RoPE indices per head, up to +3.8 on ZeroSCROLLS;
[CALIOPE](https://aclanthology.org/2026.findings-eacl.120) (EACL 2026 Findings, code) applies a monotone transform
to RoPE inputs; [layer-specific scaling](https://arxiv.org/abs/2503.04355) searches a per-layer RoPE factor (+20% on
KV retrieval); [Found in the Middle](https://arxiv.org/abs/2406.16008) calibrates attention against the U-shaped
bias (up to +15 points). A caution: [one-pass attention sorting](https://arxiv.org/abs/2606.27793) found position-bias
correction closes only 37% of its gap, so expect small effects (W1, W3b).

Reward design for round 2: [CTRL-RAG](https://arxiv.org/abs/2603.04406) rewards the log-likelihood gap between
answering with and without the evidence, which is dense and label-free; [CORE-RAG](https://arxiv.org/abs/2508.19282)
(ICML 2026, code) trains a 1.5B hard-text compressor by GRPO against a frozen 14B reader with EM + F1 reward and beats
full top-5 context (NQ 41.0 vs 38.0) at 6% of the tokens, i.e. the RL can sit on the compressor side with the reader
frozen, the mirror of our decoder-only plan; [ZipRL](https://arxiv.org/abs/2605.28069) densifies GRPO with hindsight
replay (no plain-GRPO baseline reported) (W6).

[Locating failure in multi-page VRDU](https://arxiv.org/abs/2608.07943). Controlled interventions on
MMLongBench-Doc with Qwen3-VL-8B: images added to parser text +13.7 points; removing one gold page −20.7 to −23.3;
adding three top-ranked non-gold pages breaks 13.3% of answers and repairs 7.0%, and accuracy falls 54.4 → 43.2 from
0 to 18 distractor pages; abstain-with-CoT raises refusals on unanswerable questions 45.5 → 85.5% at no accuracy cost.
This quantifies W4 for raw pages; our compressed setting should be measured the same way (break / repair counts, not
only accuracy).

### 3.6 Routing and selection over many compressed units

[HierDoc](https://arxiv.org/abs/2607.29638): a page policy then a region policy, both trained with stage-wise GRPO
and set-valued rewards; +16.9% relative on LongDocURL, and regions on top of pages +5.5 accuracy. [Trident](https://arxiv.org/abs/2608.14841):
pages turned into structured records (caption, section path, entities) before one rerank call; an LLM rerank without
the records "barely changes" the first hit. Both say the router needs a page description, which our `desc` task
already produces from slots (W4).

[Retrieving memories from token-level evidence](https://arxiv.org/abs/2610.13277): late interaction (each query token
matches a few memory tokens, scores summed per memory) over frozen token states, training-free, plus an optional
low-rank query adapter: +13.3 Recall@10 on Qwen3-Embedding-4B. Our slots are token states; a MaxSim over `<MEM>`
vectors is a router that needs no new index (W4). [AttentionRetriever](https://arxiv.org/abs/2602.12278) uses
attention layers themselves as long-document retrievers (abstract only). [CANOPY](https://arxiv.org/abs/2610.00923)
keeps regions at the granularity where a parent / child score comparison says, and a critic triggers follow-up
retrieval; its ablation says the extra retrieval, not the compression, drives the multi-hop gains.

### 3.7 Query-dependent vs generic compression

[SUMER](https://arxiv.org/abs/2511.21726) (code) trains a search agent over raw memory with RL and beats all
goal-agnostic compression baselines on LoCoMo (Qwen2.5-7B, +43% over the prior best): when the raw data is kept and
the question is known, search wins. [RAGOCR](https://arxiv.org/abs/2608.00765) renders retrieved passages as images and
gives each a resolution by estimated relevance: one eighth of the tokens, +15% over naive RAG on medical QA. On the
other side, [KVzip](https://arxiv.org/abs/2505.23416) shows query-aware selection failing under reuse, and
[CMC](https://arxiv.org/abs/2609.25537) gets its gain from generic memories plus top-K selection at query time. Read
together with AdaMem (covered: +3.4% relative at 16×, +14.6% at 64×), the pattern is: generic slots for the index,
query-dependent decisions (budget, expansion, selection) at read time. That is route-then-expand (W7).

### 3.8 Compression on a hybrid / linear-attention backbone

[Rethinking the role of efficient attention in hybrid architectures](https://arxiv.org/abs/2606.15378): full-attention
layers do the long-range retrieval; the linear / sliding layers mainly shape training dynamics, and large windows delay
retrieval-head formation ("large-window laziness"). For us: the `<MEM>` read-out (last layer, or
`compressor_extra_layers`) and any SEAL-style scaling should target Qwen3.5's full-attention layers, which are a
quarter of the stack. [ASTC](https://arxiv.org/abs/2610.41291) (above) is the one method that reads the GDN state.
[Echo embeddings](https://arxiv.org/abs/2402.15449) (ICLR 2025) repeat the input so the second copy attends to the
whole first copy, +5% on MTEB over last-token pooling, at 2× cost; [Causal2Vec](https://arxiv.org/abs/2507.23386)
(ACL 2026) prepends one contextual token from a small BERT-style encoder and beats echo at lower cost. Either is a
causal substitute for the bidirectional slot attention we cannot have. [Raven](https://arxiv.org/abs/2607.25357)
(linear-time model with input-dependent routing into fixed memory slots) and [Boa](https://arxiv.org/abs/2610.33006) (covered) are architectures, not
retrofits. [RoPE distinguishes neither positions nor tokens in long contexts](https://arxiv.org/abs/2605.15514)
proves attention rankings become unreliable as context grows and that raising the RoPE base trades position for token
discrimination; a reason to keep slot position ranges compact, as the spread layout does.

## 4. Blocked by the backbone, and already covered

**Blocked (need bidirectional attention in the compressor, no workaround in the paper):** SAC's bidirectional
anchors (−2.6 without), No Mean Feat's bidirectional slots, AdmTree's bidirectional aggregator, MambaMia's
bidirectional SSM, DeepSeek-OCR 2's bidirectional visual tokens, LLM2Vec-style encoders. Partial workarounds: echo
repetition, Causal2Vec's contextual token, a chunk-local mixer in the connector (parked).

**Already covered in our reviews (not re-summarised):** Position IDs Matter, SAC, SeCo, Activation Beacon, UniGist,
H-GSA, ICAE, AutoCompressors, gist tokens, xRAG, COCOM, PISCO, OSCAR, ARC-Encoder, LCLM, CLaRa, DEX-Comp,
Memorization-Utilization Gap, LycheeMemory, Latent Memory, LatentPress, FlexComp, AdaMem, SeleCom, density-aware
semi-dynamic ratio, GMSA, GistPool, ComprExIT, Cognitive Chunking, MemoSight, M3, MQT-LLaVA, TokenFLEX, DeCo,
VisPruner, DART, ERASE, GlimpsePrune, DeepSeek-OCR, Glyph, DocOwl 1.5 / 2, CoSyn, PixMo-Docs, Docmatix, REFRAG,
VisionThink, DocR1, Doc-V*, MM-Doc-R1, LensVLM, CounterCredit, Boa, Tempo, VideoChat-Flash, Video-XL / XL-2,
MambaMia, ReVisionLLM, VideoTreeSearch, T*, CARVE, ExtremeWhenBench, StreamMem / CacheFlow / StreamKV / StreamMeCo,
KV-Distill, SnapKV / PyramidKV / KIVI, TurboRAG / Block-Attention / CacheBlend, Hypic.

## 5. Open questions the literature does not answer

1. **Does a slot with several position ids help when the slots were trained with one?** PPE's training-free gain
   was large only where the baseline was broken; nobody has tested multi-position ids on soft slots. Our probe decides
   whether it is an inference trick or a stage-1 variable.
2. **Where in the decoder is the use gap?** SEAL and the calibration papers act on raw-text contexts. No paper
   measures per-head attention to learned memory tokens, or whether retrieval heads treat them as tokens at all.
3. **A coarse level trained for picking.** Matryoshka work (M3, MQT, MTSK, PARCEL) optimises answer accuracy at each
   budget; none trains the coarse prefix to be a good router over many units, which is what W4 + W5 need.
4. **Routing over slots at scale.** Late interaction over token states (2610.13277) and attention-mass retrieval
   (2602.12278) are evaluated on raw tokens; we would be the first to report pick@1 over 10-100 pages from `<MEM>`s.
5. **Query-agnostic index, query-dependent read.** KVzip and CMC point there; no paper compares OSCAR-style
   compression against generic slots plus selection and expansion at equal total compute on documents or video.
6. **Linear-attention state as a compression signal.** ASTC uses GDN innovation for agents; whether it predicts which
   chunks our slots fail on (overflow) is untested.
7. **Hour-long video with a trained coarse level.** Consolidation (CoMeT, LatentStream) and temporal position for
   compressed tokens (TAPE) exist separately; no work trains both with an order probe, which our earlier review
   already flagged as the gap.
