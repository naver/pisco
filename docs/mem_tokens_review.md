# Improving the memory tokens themselves: literature and design review (2026-10-30)

What this covers: the `<MEM>` tokens the compressor writes into, and only them. Not the connector (settled by
[connector_probe.md](connector_probe.md)), not the decoder's use of the vectors (RL, prompting: see
[architecture_review.md](architecture_review.md), "What to change"). Numbers from papers carry the model, size and
rate; anything marked **(ours)** is our reading, not a measured result. About 35 papers, 2023 to October 2026,
arXiv ids checked.

## 1. What we have, and what we measured

**Today.** One special token `<MEM>`, the same embedding in every slot (the compressor's row is trainable through
PEFT `trainable_token_indices`; it starts from the resize default). `n = len(chunk) // 8 + 1` copies are appended
**after** the chunk (123 for a 980-token page image, 56 for a 440-token photo). The compressor is Qwen3.5-4B,
**causal**, hybrid (24 linear-attention layers that ignore any 4D mask + 8 full-attention layers), LoRA-trained.
The last-layer state at each `<MEM>` goes through the connector (a linear map in practice, plus a norm) and replaces
the `<MEM>` placeholder in the decoder prompt. Stage 1 = text autoencoding + continuation; stage 2 = image AE,
image continuation, text rehearsal, `desc`, `find` (optional KD to the uncompressed backbone); then a QA finetune.
Campaign archq3 (running) retrains stage 1 with each `<MEM>` placed, in position-id space, at the centre of the
segment it compresses (text) or of its tile (images).

**What the probes say** (paired answer-NLL on held-out rows, [connector_probe.md](connector_probe.md); failure
counts from [architecture_review.md](architecture_review.md) probe 2):

| finding | number | what it means for the memory tokens |
|---|---|---|
| each vector encodes its own position | linear read-out R² .92–.97; shuffling a chunk costs 0 | identity is not missing; order is not read |
| half the vectors carry almost everything | keep every second: docs +.17 ± .13, text +.07 ± .07, photos −.03 ± .03 | the 123 slots hold about 60 slots' worth of distinct content |
| early slots > late slots | keep first half: docs +.21, photos +.09; keep second half: docs +.53, photos +.34 | late `<MEM>`s mostly repeat what earlier ones wrote (causal compressor) |
| per-vector content matters | chunk mean: docs +.92, photos +.65 | the vectors are not interchangeable; any pooling must be trained in |
| one big shared direction | mean cosine .75–.79 between `<MEM>`s; only .02 higher within a chunk than across | a sink-like component; the linear connector can subtract it |
| dense infographics lose content at compression | 25 lost vs 17 "use" failures | a capacity-at-write problem, not a slot-count problem (rate 4: null on docs over 2 seeds) |
| multi-page documents keep content, decoder does not use it | 6 lost vs 27 use failures | not a memory-token problem; route-then-expand already closes it |

Two constraints on anything we design: (1) a custom attention mask inside the compressor reaches **8 of 32 layers**
on Qwen3.5 (the linear-attention layers ignore it), so a design that relies on a mask must be expressed through
**sequence order** or through a **separate module**; (2) an image span is atomic for the HF model, so `<MEM>`s cannot
be physically interleaved inside an image; the archq3 position layout is the feasible substitute.

## 2. The design space, with what the papers found

### 2.1 Shared vs distinct token embeddings, and their initialisation

| design | who uses it | evidence | our problem addressed? | cost here |
|---|---|---|---|---|
| one token repeated (ours) | Gist 2304.08467 (LLaMA-7B, 26×: "the gist token is the same from g1 to gk"), Activation Beacon 2401.03462 (one shared embedding, separate beacon Q/K/V projections), COCOM 2407.09252 (`<CTX>` placeholders), UniGist 2509.15763 | Gist: k ∈ {1,2,5,10} "generally insensitive"; too many sometimes hurt | — | — |
| k distinct learned embeddings | ICAE 2307.06945 (128 slots for 512 tokens, Llama-7B), AutoCompressors 2305.14788 (50 `<Sum>_i`), PISCO 2501.16075 (8 per 128-token doc at 16×), 500xCompressor 2408.03094 | no paper ablates shared vs distinct. AutoCompressors: κ = 50 beats 20/70/100, "performance does not increase with longer soft prompts" | none: our vectors already know their position (R² .92–.97) and the decoder ignores order. In a causal compressor slot k sees k−1 slots before it, so it knows its index anyway **(ours)** | stage 1 |
| initialisation | Hewitt-style mean-of-vocabulary sampling (Sentence-anchored gists 2511.08128), EOS-token init (AutoCompressors, OPT), mean of the chunk's tokens (Shopify gisting), average init (K-token merging 2604.15153: one epoch faster) | only convergence speed is reported, never final quality | none | — |
| per-sample optimised vectors (the ceiling) | Cramming 2502.13063: Llama-3.1-8B holds 1,568 tokens in **one** vector, 3B 1,024, 1B 512; near-linear in the number of vectors; measured ≈ 4,900 bits per vector | trained encoders stay at 4–10×: the slot is not the limit, the **writer** is. Progressive Cramming 2607.21231: such vectors are "brittle steering", they hurt even with the text present | explains why rate 4 did not help docs: dense pages fail at write time **(ours)** | — |

Verdict: distinct embeddings and initialisation are not levers for us. Capacity per slot is two orders of magnitude
above what we use; what limits us is how the compressor fills the slots.

### 2.2 How many slots: fixed, proportional, content-adaptive

| rule | evidence (model, rate, effect) | addresses | cost |
|---|---|---|---|
| fixed count | ICAE, PISCO, AutoCompressors, 500x (Llama-3-8B: F1 retention 72.9 / 67.1 / 62.3% with 16 / 4 / 1 tokens). ARC-Encoder 2510.20535: fixed memory tokens "perform poorly as sequence length increases" (38.4 vs 41.7 pooled, Mistral-7B 8×); AdmTree 2512.04550 blames ICAE's 21.8 LongBench (vs Beacon 40.1) on the fixed budget | — | — |
| proportional (ours) | COCOM, Beacon, UniGist, GMSA, ARC, LCLM 2606.09659: the norm since 2024 | — | — |
| **budget by content density** | AdaMem 2609.22100 (query relevance → budget across passages; vs OSCAR uniform: up to +3.2 pts at 16×, +14.6% relative at 64×); Density-aware 2603.25926 (Qwen3-0.6B predicts log₂(len / summary len) → ratio ∈ {2…32}; beats fixed at equal average ratio except the lowest); Z-tokens 2603.25340 (Qwen3-0.6B emits a variable count, count correlates with surprisal r = .54; QuALITY 39.3 vs ICAE 34.7 at 4×); AdmTree (perplexity × entropy ranks segments: top quarter gets 4× the slots of the bottom half; uniform allocation −2.4 single-doc); Data Distribution Matters 2602.01778 (encoder-side entropy predicts compression quality); images: ERASE 2605.09982 (Qwen2.5-VL-7B keeps 95.7% accuracy at 25% tokens, budget from raw-image statistics), M3 2405.17430 (≈ 9 of 576 tokens suffice on COCO-style data; "large gap" to the per-sample oracle) | **dense pages lost at compression**: more slots only where they are needed, instead of rate 4 everywhere (1.8× tokens, null on docs) | predictor + a compressor that writes at two rates → stage 2 with mixed rates |
| **nested budgets (Matryoshka)** | FlexComp 2609.11192 (Llama-3.2-1B; budget K ∈ {1, 10, 34} sampled per instance; one model within 1.3 F1 of three specialists; a K-predictor loses ≤ 0.7 F1 at 158–236×); MQT-LLaVA 2405.19315 (random m ≤ 256; 256 tokens match LLaVA-1.5's 576; 16 tokens −2.4 MMBench); TokenFLEX 2504.03154 (+1.6 / +1.0 / +0.4 at 64 / 144 / 256 tokens vs fixed); Beacon: ratio sampled per chunk 40.5 vs per instance 37.7 | the **redundancy** becomes a feature: the gist goes first, detail later; the budget is chosen per page at serving time | **finetune only** (decoder sees the first k slots; collator change) |

Our rate-8 model is already half-way nested (keep-first-half costs .09–.21, keep-second-half .34–.53). Re-compressing
at rate 16 with the rate-8 model costs −.07 accuracy, dropping every second vector costs nothing: pruning beats
re-compression, which is exactly what nested training formalises **(ours)**.

### 2.3 Where the slots sit: tail, interleaved, prefix, "soft" interleaving

| placement | evidence | addresses | cost |
|---|---|---|---|
| tail (ours) | ICAE, PISCO, COCOM, Gist, 500x, AutoCompressors | — | — |
| interleaved, one slot per unit | Beacon (Qwen2-7B single-doc QA 40.5 interleaved vs 35.2 all-at-end); UniGist (one gist per r tokens, HELMET 60.4 vs Beacon 51.8 on Llama-3.1-8B); gist study 2412.17483 (Llama-3.1-8B, 4×: fine-grained 46.2 vs coarse 30.9 long-context average; synthetic recall 40.6 vs 14.2; RAG 60.6 vs 51.7); GistPool 2504.08934 (Gemma2-9B, each gist attends only its window: DROP 2× error 60.6% → 16.6%); sentence-anchored gists 2511.08128 (Llama-3.2-3B, gists after each sentence) | redundancy (each slot sees a different past) and coverage | stage 1; impossible inside an image span |
| position-id interleaving (archq3) | Position IDs Matter 2409.14364 (ICAE-1B 5×: BLEU 31.8 → 72.4 encoder-side → 96.0 with decoder side; OOD F1 40.0 → 43.7 → 43.9; 500x-1B F1 45.8 → 48.0; **worse than default at 31×**); SAC 2510.08907 gets most of its gain from this layout (EPL 62.9 of SAC's 63.6 F1 vs ICAE 47.5); MemoSight 2604.14889 (−3.1 average without it) | same, works for images | running (archq3) |
| content-selected slots | Dodo/Nugget 2310.02409 (LLaMA-7B: scorer picks clausal delimiters; SQuAD 49.8 at 10× vs 64.5 full); SAC: anchors = middle token of each segment + a learned anchor embedding, informativeness-based selection "on par", random worse | nothing beyond archq3 | stage 1 |
| prefix | Gist (prompt then gists); rare otherwise | — | — |

The three failure modes of the gist study are worth keeping as names: **lost by the boundary** (the start of a
segment is encoded worst), **lost if surprise** (an unexpected detail is dropped first under a tight budget: PopQA
needle 50.7 → 35.8 when off-topic), **lost along the way** (long items spread over several slots drift at read-out).
Dense infographics are "surprise" content by construction **(ours)**.

### 2.4 Attention among the slots: causal, bidirectional, blocked, two-pass, registers

| pattern | evidence | addresses | cost |
|---|---|---|---|
| causal (ours) | everyone before 2025 | — | — |
| **bidirectional among memory tokens, causal to the text** | No Mean Feat 2510.20797 (Qwen3-8B F1: 4× 67.0 → 69.2, 16× 56.2 → 60.3, 128× 47.5 → 46.9; Qwen3-1.7B 4× 50.9 → 62.0). Their argument: under a causal mask the first k slots must be a valid compression for every k, so **late slots can only add residue** — our early > late result is this constraint showing | **redundancy of late slots**; more distinct content per 123 vectors | stage 1 and only 8 of 32 layers; or a separate mixer (below) |
| fully bidirectional encoder | SAC (Llama-3.2-1B 5×: removing it −2.6 F1 in-domain, −3.3 OOD); ARC-Encoder (3B encoder, Mistral-7B, 8×: causal 41.3 vs 41.7, "slightly worse"); LCLM (pooling, 0.6B encoder: causal gave *lower* loss); DeepSeek-OCR 2 2601.20552 (bidirectional visual tokens, causal queries: +3.7% OmniDocBench) | mixed; our early text checkpoints were bidirectional, the hybrid backbone refuses it | stage 1 |
| **blocked: slot t sees only block t and earlier slots** | Cognitive Chunking 2602.13980 (Qwen2.5-0.5B compressor, frozen Llama-3-8B: 16× F1 52.0 vs 48.8 full-causal; 64× 39.7 vs 30.6; cosine between slots goes from a heavy high-similarity tail to centred near 0) | redundancy and the shared direction, by construction | mask → 8 layers only; the sequence-order version is interleaving |
| mixer module after read-out | GMSA 2505.12215 (Qwen3-4B 4×: one block **initialised from the decoder's first layer** 60.4 EM, random init 19.5, MLP only 20.2); Interlat-style self-attention adapter (named in architecture_review) | lets the slots coordinate without touching the backbone | finetune-level if the new module gets its own lr (arm 1 lesson), else stage 2 |
| two-pass / recurrent | AutoCompressors (summaries accumulate), SeDeM 2608.00311 | not our bottleneck | stage 1, 2× compressor cost |
| registers / sinks | ViT registers 2309.16588 (high-norm background patches act as global storage; 1–4 non-read-out tokens fix it: DINOv2 ADE 46.6 → 47.9, LOST VOC07 35.3 → 55.4); MemTok 2602.21202 sees "information collapse" and a bias to the first slots in appended memory tokens; ComprExIT 2602.03784 finds ICAE gist attentions "highly correlated" | the .75 shared direction, **if** it is a sink artefact | stage 1; probe first |

### 2.5 Pooling instead of memory tokens

| method | evidence | note |
|---|---|---|
| mean of groups of hidden states | ARC-Encoder (pooled 41.7 vs memory tokens 38.4, 8×; the baseline is *fixed-count*, so count and mechanism are confounded); No Mean Feat (mean pooling of a bidirectional encoder is their strongest baseline up to 16×: LongBench-E single-doc 39.7 vs PCC 17.2, Qwen3-1.7B 4×); LCLM (0.6B encoder, 350B tokens); GMSA (group merging 60.4 vs learnable tokens 21.9 EM, same 4B, but the token arm has no PISCO/COCOM baseline) | each vector is local by construction: no causal redundancy, every token contributes |
| all-layer anchors transported to slots | ComprExIT (Llama-3.2-1B 4×: 66.6 F1 vs ICAE ≈ 56; last layer alone 49.4, layer 9 63.9) | read-out layer matters more than the carrier type |
| learned queries (Q-Former family) | DeCo 2405.20985 (pooling > Q-Former: +7.1% localisation), LLaVA-UHD v4, MemTok (memory tokens 54.4 vs clustering 56.7 nDCG@5 on ViDoRe at 64 vectors; full index 60.0) | settled against resamplers already |

Pooling is the strongest-supported alternative to memory tokens in 2025–26. It is a stage-1 retrain (the compressor
must learn to make *token* states decoder-readable) and `chunk_mean` (+.92) shows it cannot be bolted on at
inference. [architecture_review.md](architecture_review.md) already schedules it behind archq3.

### 2.6 Hierarchical / multi-resolution memories

AdmTree (LLaMA-2-7B LongBench 44.1 vs Beacon 40.1; a bidirectional aggregator over gist tokens; without the tree
36.5 → 28.5 single-doc), H-GSA 2604.20920 (meta-gist over 4 gists: Qwen2-7B 16× 45.3 → 46.5 after finetuning but
**worse** under continued pretraining alone), M3 / MQT nested tokens, SeDeM (select then decompress). Our version is
route-then-expand plus the `find` task, which already reaches the uncompressed score on documents (.801 vs .809).
A coarse per-page vector would be a training device for the same routing signal **(ours)**; low priority.

### 2.7 Objectives that act on the slots directly

| objective | evidence | addresses | cost |
|---|---|---|---|
| reconstruction share | ICAE (λ_AE .4–.6 best), COCOM (AE + continuation), ARC (0% reconstruction 39.8, 20% 41.6, 100% 37.5), GMSA (without AE 60.4 → 48.8 EM), SAC (no AE at all, LM only, still best) | we are in the good range | — |
| **per-slot ownership (fine-grained AE)**: slot k reconstructs **its own** segment | gist study 2412.17483 (Llama-3.1-8B, 4×: long-context average 46.1 → 49.8; synthetic recall 40.6 → 62.0; boundary MMLU-Pro 20.3 → 23.4); the mask version is Cognitive Chunking above | redundancy (each slot must carry something unique) and coverage of dense pages | stage 1 (text), stage 2 (image head) |
| token-importance weighting (TIE) | same study: loss weights from how much a token depends on the context: 46.1 → 48.3 | "lost if surprise" | stage 1 |
| contrastive / which-segment on the vectors | Latent Memory 2606.10572 (one `[MEM]` per evidence, InfoNCE query↔evidence + token-level KL + reconstruction; without reconstruction −1.7 EM at k = 5, Llama-3.2-1B); Bottleneck tokens 2604.11095 (generative condensation mask); our `find` task works on the decoder side | retrieval use of the slots, not QA quality | stage 2 |
| de-correlation / diversity loss | **none found** in soft compression; the orthogonality losses are on visual bases (2511.19515); all three papers that measured slot redundancy (Cognitive Chunking, ComprExIT, MemTok) removed it by **structure** (mask, transport, clustering), not by a loss | — | our extrapolation only |
| nested-budget / slot dropout | FlexComp, MQT, TokenFLEX, Beacon ratio mixing (section 2.2) | serving budget; mild regulariser | finetune only |
| KD into the vectors (hidden-state targets) | no winning recipe: CLaRa 2511.18659 and PISCO distil sequences, No Mean Feat and LCLM distil logits, xRAG 2405.13792 self-distils logits | — | — |
| RL on the decoder (RLComp, DEX-Comp 2609.05152, LycheeMemory 2602.08382) | acts on use, not on the slots | the 27 "use" failures | already item 2 of the architecture plan |

## 3. Ranked shortlist

| # | candidate | problem it targets | evidence strength | cost | expected effect on docs (1,500 Q, judge) | confidence |
|---|---|---|---|---|---|---|
| 1 | **`<MEM>` mixer**: one chunk-local bidirectional block over a chunk's vectors, decoder-initialised, zero-init output | late-slot redundancy → more distinct content per page | No Mean Feat (+2.2 to +4 F1 at 4–16×, 8B), SAC (−2.6 without), GMSA (decoder init 60.4 vs random 19.5); mechanism matches our probe | finetune with a separate lr for the module (or one stage-2 pass) | +.01 to +.03 | medium |
| 2 | **per-slot ownership loss**: slot k reconstructs its own segment (text) / its tile region (images, via the archq3 layout) | redundancy + dense-page coverage | 2412.17483 (+3.7 avg, +21 recall at 4×), Cognitive Chunking (+3.2 F1 at 16×) | stage 1 (+ stage 2 for the image head) | +.02 to +.04 on dense pages, ≈ 0 elsewhere | medium-low |
| 3 | **nested budgets**: train the decoder on the first k slots, k random; pick k per page at serving | tokens and latency; lets dense pages keep all 123 slots and plain pages 60 | FlexComp (≤ 1.3 F1 from specialists), MQT, TokenFLEX (+0.4–1.6), Beacon (+2.8); our keep-first-half probe | finetune only | ≈ 0 at full budget; −.00 to −.02 at half the tokens (vs −.07 for re-compressing at 16) | high for the budget feature, low for accuracy |
| 4 | **group pooling in place of `<MEM>`s** | redundancy, coverage, length robustness | ARC, No Mean Feat, LCLM, GMSA, ComprExIT | stage 1 | +.01 to +.03; or a loss on images (no literature on page images) | medium-low; wait for archq3 |
| 5 | **density-adaptive rate** (rate 4 on the densest pages only) | dense pages | AdaMem, Density-aware, Z-tokens, AdmTree, ERASE | stage 2 with mixed rates + a density predictor | +.01 to +.03 at ~1.2× tokens | low (the rate-4 gain itself did not replicate at seed 43) |

Not shortlisted, probe first: registers (one or two non-read-out tokens before the slots). Cheap to test whether the
shared direction carries content (section 4); add it to the next stage-1 chain only if the probe says the first slot
behaves like a sink.

### 3.1 Proposal 1: the `<MEM>` mixer

What changes.

- **Model**: a new module `mem_mixer` between the compressor read-out and the connector: one transformer block
  (self-attention + MLP), width 2560, initialised from the **decoder's layer 0** (GMSA: random init gives nothing),
  with the output projections of attention and MLP **zero-initialised** so the retrofitted model is byte-identical to
  the checkpoint (same trick as arm 1's extra columns; reuse `check_extra_cols.sh`-style equality check). Attention
  is block-diagonal over the chunk ids already carried by `compress()` (one block per chunk, so a page's 123 slots
  mix among themselves only). No position encoding: each vector already encodes its position and the decoder
  ignores order. Config: `PISCOConfig.mem_mixer = {none, decoder_layer0}`, `mem_mixer_lr`.
- **Training**: the module is new, so it needs its own learning rate (arm 1 showed a 2e-5 finetune leaves new
  parameters 50× too small). Two chains: (a) finetune only, module lr 1e-4 to 2e-4 (the `shake_lin_clr*` runs give
  the knob), seeds 42 and 43; (b) if (a) moves the redundancy signature but not accuracy, one stage-2 pass with the
  v4kd recipe at 100k samples, as in the archq3 reduced chain.
- **Why not inside the compressor**: bidirectional slot attention as a mask would reach 8 of 32 layers; if archq3
  needs a second stage-1 run anyway, add "slot-to-slot bidirectional in the full-attention layers" as a rider, but the
  mixer is the clean test.

Shake-out: equality at zero init on 4 rows; gradient norm of the mixer > 0 after one step; memory probe at bs 24;
`mm_shakeout.sh … "--config_override mem_mixer=decoder_layer0"`.

Comparison: paired judge accuracy vs 174908 / 176242 (norm, same seeds) on `mm_docqa_1500_1m` and
`mm_naturalqa_1500`; a gain counts above the seed spread (.03) at both seeds. **Success signature in the probes**
(run `probe_connector.sh` on the arm): keep-second-half cost falls towards keep-first-half, keep-even cost rises
(the vectors became complementary), within-chunk cosine drops below the across-chunk one. If accuracy is flat but
the signature appears, the slots now hold more content and the decoder is the next bottleneck.

### 3.2 Proposal 2: per-slot ownership loss

What changes.

- **Collator**: for a text chunk of L tokens with n slots, slot k owns tokens `[k·L/n, (k+1)·L/n)` (the segmentation
  the archq3 layout already uses). For an image, slot k owns the tile group of its spread position. The collator
  emits `owner_ids` (segment per slot) and the target: text = the bag of words of the segment (multi-hot over the
  vocabulary); image = the merger-output features of the tile group (unit-RMS).
- **Model**: a light head on the connector output: text = one Linear to the vocabulary with BCE (no decoding step,
  unlike the 1-layer decoder of 2412.17483); image = one Linear to the merger width with MSE. Loss
  `total = lm + λ·ownership`, λ ≈ 0.1–0.3 (sweep at shake-out so the auxiliary loss is 10–30% of the LM loss).
  Config: `PISCOConfig.mem_ownership_loss`, `ownership_weight`.
- **Chain**: stage 1 text (1M samples, 4 GPUs, ~16 h as in archq3) → stage 2 with the image head → finetune two
  seeds. It composes with archq3's layout and should be run **on top of it** if archq3 wins, as a separate arm if not.

Shake-out: the auxiliary loss falls in the first 200 steps; the **ownership probe** (section 4) run on a 2k-step
checkpoint shows slot k predicting its own segment better than its neighbours'.

Comparison: as in 3.1, plus the transcription probe (`probe_mem_recall.py`): the share of page words rebuilt from
the slots on infographics (58% today) is the direct read-out of coverage.

Risks: the gist study's gains shrink with the rate (4× → 32×); we are at 8×, inside the range. A coverage loss pulls
against "summarise": watch photos, where the vectors are already fully redundant and nothing is lost.

## 4. Cheap probes to run first (inference only, trained model, paired NLL as in connector_probe)

| probe | variant to add to `probe_connector.py` | premise tested | decides |
|---|---|---|---|
| late-from-early | for each late slot, the best linear reconstruction from its chunk's early slots (ridge, held-out R²); and the reverse | are late slots predictable from early ones? | #1, #2: high R² = the redundancy is removable |
| ownership (text) | ridge from slot k to the bag of words of segment k vs segment j ≠ k | do slots already specialise by segment? | #2: if they do, the loss is redundant |
| first-k curve | `keep_first_k` for k/n ∈ {.25, .5, .75} (generalises `keep_first_half`) | how nested is the model already? | #3: the curve before training is the baseline |
| sink check | per-position norm of the slots; `drop_position_p` for p = 1, n/2, n; `remove_shared_direction` (project out the top principal component) | is the shared direction a sink with no content? | registers: only if dropping slot 1 or the shared direction costs more than any other slot |
| density vs gain (no GPU) | per-question rate-4 minus rate-8 (jobs 174694 / 176425) against page density (OCR tokens, or compressor AE loss) | does the rate-4 gain live on the densest pages? | #5: a 2-rate model is worth it only if the gain concentrates |

## 5. Ruled out by our probes

| option | finding that argues against it |
|---|---|
| distinct `<MEM_1>…<MEM_n>` embeddings to give slots an identity | each vector already encodes its position (R² .92–.97); the decoder ignores order (shuffle 0 ± .03); a causal slot knows its index from the slots before it |
| decoder-side position layouts of any kind | arm 2 flat to negative at two seeds; the position is in the vectors |
| deeper / wider / nonlinear connectors, resamplers, Q-Former | a refit linear map equals the trained MLP on every backbone pairing (within .006) |
| shrinking or constraining the vectors' norm | norm probe: equalising lengths is fine, shrinking to token scale erases content; SeleCom 2602.15856 agrees |
| more slots everywhere (rate 4) | docs +.034 at seed 42, −.022 at seed 43; photos +.017 for 1.8× tokens |
| changing the rate at inference by re-compressing | rate-8 model read at rate 16: −.070; the write pattern is rate-specific |
| averaging or pooling the trained slots after the fact | `chunk_mean` +.92 docs / +.65 photos: pooling has to be trained in (section 2.5), it is not a post-processing step |
| query-dependent compression at finetune (OSCAR) | null here; AdaMem's gains need the query at compression time, which we cannot precompute |
| a coarse per-page summary vector for routing | the routing signal is already there: route-then-expand reaches .801 vs .809 uncompressed with no training |
| per-layer KV injection, cross-attention readers | not probed; settled by the first review (frozen-decoder methods, slower than input embeddings) |

## Appendix: papers used, one line each

| id | paper | setting | number used |
|---|---|---|---|
| 2304.08467 | Gist tokens (Mu et al.) | LLaMA-7B, FLAN-T5-XXL, up to 26× | one token repeated; k ∈ {1,2,5,10} insensitive |
| 2305.14788 | AutoCompressors | OPT-2.7B, Llama-2-7B | κ = 50 distinct `<Sum>_i` best of 20/50/70/100; EOS init helps |
| 2307.06945 | ICAE | Llama-7B, 4× | 128 distinct slots for 512 tokens; λ_AE .4–.6 |
| 2309.16588 | Vision Transformers Need Registers | DINOv2-g, OpenCLIP, DeiT-III | ADE 46.6 → 47.9; LOST VOC07 35.3 → 55.4; 1 register removes artefacts |
| 2310.02409 | Dodo (nuggets) | LLaMA-7B, 5–20× | scorer picks clausal delimiters; SQuAD 49.8 at 10× vs 64.5 |
| 2401.03462 | Activation Beacon | Llama-2-7B, Qwen2-7B, 2–32× | interleaved 40.5 vs all-at-end 35.2; ratio per chunk 40.5 vs per instance 37.7; one shared embedding |
| 2405.13792 | xRAG | Mistral-7B, one token per document | retriever embedding + bridge; logit self-distillation |
| 2405.17430 | Matryoshka Multimodal Models | LLaVA-1.5 | ≈ 9 of 576 tokens suffice on COCO-style; large oracle gap |
| 2405.19315 | MQT-LLaVA | LLaVA, m ≤ 256 random | 256 matches 576; 16 tokens −2.4 MMBench |
| 2405.20985 | DeCo | LLaVA-style | pooling beats Q-Former: +7.1% localisation |
| 2407.09252 | COCOM | Mistral-7B, 4–128× | `<CTX>` placeholders ∝ length; EM .585 / .577 / .540 vs .623 |
| 2408.03094 | 500xCompressor | Llama-3-8B, 6–480× | F1 retention 72.9 / 67.1 / 62.3% with 16 / 4 / 1 tokens |
| 2409.14364 | Position IDs Matter | ICAE-1B, 500x-1B, 5–51× | BLEU 31.8 → 72.4 → 96.0; OOD F1 40.0 → 43.9; worse at 31× |
| 2412.17483 | Gist study (Silver Bullet or Compromise) | Llama-3.1-8B, Qwen2-7B, 4–32× | fine 46.2 vs coarse 30.9; per-slot AE 46.1 → 49.8, recall 40.6 → 62.0; TIE 48.3 |
| 2501.16075 | PISCO | Mistral-7B, 16× | 8 distinct memory embeddings per 128-token doc; .68 vs .71 |
| 2502.13063 | Cramming 1568 tokens into one vector | Pythia to Llama-3.1-8B | 1,568 / 1,024 / 512 tokens per vector (8B / 3B / 1B); ≈ linear in vectors |
| 2504.03154 | TokenFLEX | VLM, 64–256 tokens | +1.6 / +1.0 / +0.4 vs fixed |
| 2504.08934 | GistPool | Gemma2-2B/9B, 1–50× | windowed gists: DROP 2× error 60.6% → 16.6% |
| 2505.12215 | GMSA | Qwen3-4B, 4× | merging 60.4 vs tokens 21.9 EM; decoder-init block 60.4 vs random 19.5; no AE 48.8 |
| 2509.15763 | UniGist | Llama-3.1-8B, 4× | one gist per r tokens; HELMET 60.4 vs Beacon 51.8 vs full 63.9 |
| 2510.08907 | SAC (semantic anchors) | Llama-3.2-1B, 5–51× | 63.6 F1 vs ICAE 47.5; −2.6 without bidirectional; EPL alone 62.9 |
| 2510.20535 | ARC-Encoder | 3B encoder → Mistral-7B, 4–8× | pooled 41.7 vs fixed memory tokens 38.4; causal 41.3 |
| 2510.20797 | No Mean Feat | Qwen3-1.7B / 8B, 4–128× | bidirectional slots: 8B 4× 67.0 → 69.2, 16× 56.2 → 60.3; 1.7B 4× 50.9 → 62.0 |
| 2511.08128 | Sentence-anchored gists | Llama-3.2-3B | gists after sentences; Hewitt mean-of-vocabulary init |
| 2512.04550 | AdmTree | LLaMA-2-7B, Qwen2-7B, 2–8× | 44.1 vs Beacon 40.1 vs ICAE 21.8; uniform allocation −2.4; no tree −8.0 |
| 2601.20552 | DeepSeek-OCR 2 | 80M compressor, 16× | bidirectional visual tokens + causal queries: +3.7% OmniDocBench |
| 2602.01778 | Data Distribution Matters | AE framework | encoder-side input entropy predicts compression quality |
| 2602.03784 | ComprExIT | Llama-3.2-1B/3B, 4–16× | last layer 49.4, layer 9 63.9, all-layer 66.6 F1; ICAE slots correlated |
| 2602.13980 | Cognitive Chunking | Qwen2.5-0.5B → Llama-3-8B, 4–64× | blocked mask: 16× 52.0 vs 48.8; 64× 39.7 vs 30.6; slot cosines centred at 0 |
| 2602.21202 | Multi-vector index compression (MemTok vs AGC) | Qwen2.5-VL-3B, 64 vectors | memory tokens 54.4 vs clustering 56.7 vs full 60.0 (ViDoRe); "information collapse" |
| 2603.25340 | LLM as token compressor / decompressor | Qwen3-0.6B, 4–8× | variable Z-count, r = .54 with surprisal; QuALITY 39.3 vs ICAE 34.7 |
| 2603.25926 | Density-aware semi-dynamic ratio | Qwen3-0.6B, 2–32× | ratio predicted from summary length; beats fixed at equal mean ratio |
| 2604.14889 | MemoSight | Qwen2.5-7B, 2–16× | uniform position layout −3.1 when removed |
| 2604.20920 | Gist sparse attention (H-GSA) | Qwen2-7B, 8–32× | meta-gists 45.3 → 46.5 after finetune, worse under CPT only |
| 2605.09982 | ERASE | Qwen2.5-VL-7B | 95.7% accuracy at 25% tokens; budget from raw-image statistics |
| 2606.09659 | LCLM (end-to-end compression at scale) | 0.6B encoder → Qwen3-4B, 4–16× | mean pooling; causal encoder gave lower loss |
| 2606.10572 | Latent Memory (one token per evidence) | Llama-3.2-1B compressor | InfoNCE + KL + reconstruction; without reconstruction −1.7 EM |
| 2607.21231 | Progressive Cramming | SmolLM2-1.7B | crammed vectors are brittle steering, hurt even with the text present |
| 2609.11192 | FlexComp | Llama-3.2-1B, 15–510× | one Matryoshka model within 1.3 F1 of specialists; K-predictor ≤ 0.7 F1 loss |
| 2609.22100 | AdaMem | OSCAR-style, 16–64× | relevance-weighted budget: ≤ +3.2 pts at 16×, +14.6% relative at 64× |

Also consulted, not load-bearing: 2602.15856 SeleCom (norm constraints), 2609.05152 DEX-Comp and 2602.08382
LycheeMemory (RL on the decoder), 2608.00311 SeDeM, 2604.11095 Bottleneck tokens, 2604.15153 K-token merging,
2511.18659 CLaRa, 2605.22337 Meta-Soft, 2503.19114 information preservation in prompt compression (soft
compression drops entities; granularity control preserved 2.7× more), 2606.21807 (compressed-evidence evaluations
shrink reader differences 31.8 → 7.8 points; an argument for our paired, same-questions protocol).
