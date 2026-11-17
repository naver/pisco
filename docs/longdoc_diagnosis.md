# Long inputs: where is the bottleneck? (campaign longdoc)

Rewritten by each campaign wake when an arm decides. Last rewrite 2026-11-17 15:10 (arm T closed: the long-chunk loss is image-specific and enters in stage 2; on text it shows only on one long chunk. Arm L started: documents at 10 and 20 pages, seed 42).
Question (owner): every attempt at long inputs loses accuracy — is the cause the evaluation, the training data,
training itself, or how the decoder reads many `<MEM>` slots (one `<MEM>` = one compressed slot, 8 tokens each)?

## 1. Bottleneck table
Accuracy = Qwen3-8B rubric judge, paired on the same questions. pt = 0.01. A cause needs ≥ 3 pt with the 95% CI
on the far side of 0 at both seeds; "not this cause" needs the CI inside ±3 pt.

| candidate | probe | n | number | gate | verdict | questions repaired |
|---|---|---|---|---|---|---|
| evaluation feeds less than it claims | input audit, CPU (job 191147) | 1,500 Q per variant × 14 variants (one set of 500) | truncation cut 0 / 1,500 and pisco-vs-uncompressed parity 1,500 / 1,500 on every gating variant; the cut check does fail when it should (CML 256 controls: 924 / 1,500 and 275 / 1,500). Limits: on documents the cut check can only fail on prompt length (pages are images, never cut); the audit's parity used one row builder for both modes, so it was re-checked on recorded runs: pages per question equal in both modes on 1,500 / 1,500 (4 pages), 1,500 / 1,500 (10 pages), 500 / 500 (whole documents) — count, not page identity | cut ≤ 1%, parity 100% | provisional: not this cause (gate passed; parity only partly independent) | 0 |
| training never shows long inputs | input audit (count only) | as above | 10- and 20-page documents: 100% of questions beyond the finetune range; 40 text passages: 99.9%; 20 passages: 2.3% | names a cause only with a paired effect from the reading or position probes | open (count, not yet a cause) | — |
| long-chunk training (stage-1 chunk 1,024 vs 256) | gold page only / picked page raw on the C256 and L1024 finetunes; page transcription and text autoencoding on their earlier stages; ten short vs one long text chunk | 1,500 Q × 2 seeds (QA); 500 pages / 500 documents (probes) | gold page only (distractor pages removed, 1,003,520 px): L1024 − C256 −6.7 [−8.9, −4.5] McNemar 106 / 207 (s42), −4.9 [−7.1, −2.5] 113 / 186 (s43). Picked page raw: +0.1 [−0.8, +1.1] (s42), −0.1 [−1.1, +0.9] (s43); both models pick the answer page on 98.5%. Page transcription (stage 2): −8.0 pt [−10.6, −5.5] of the page's words, p 4e-16. Text autoencoding at 256 tokens: stage 1 −.0018 nats, stage 2 +.0006 [−.0008, +.0019] p .82 (L1024 not worse). Text QA: −1.1 / −3.3 on one ~1.3k-token chunk, +2.3 / −0.6 on ten short chunks | deficit ≤ −3 pt with the gold page alone AND inside ±2 pt with the picked page raw → in the compressed page; stage-2 text autoencoding worse (CI above 0) → general stage-2 loss, else image-specific | provisional: CLOSED. IN THE COMPRESSED IMAGE PAGE, entering in stage 2: L1024's page slots hold fewer words, its text slots do not. Not multi-page reading, not the page pick, not the decoder. Motivates (owner): keep the 256-token stage 1; a long-chunk stage 1 would need its stage 2 changed, checked by page transcription before any finetune | the whole 4-page deficit: ~93 (s42) / ~65 (s43) of 1,500 questions |
| length on a fixed model | 4 / 10 / 20 pages, 10 / 20 / 40 passages | 1,500 Q per size | running (documents, seed 42: 10 and 20 pages, uncompressed 20 pages) | Δ_len ≤ −3 pt | open | — |
| reading many `<MEM>`s | keep the most-attended eighth at the long size | — | not started | dilution D ≥ +3 pt | open | — |
| positions past the trained range | shift `<MEM>` position ids | — | not started | offset cost ≤ −3 pt | open | — |

### Arm T detail: long-chunk finetunes on documents (`mm_docqa_1500_1m`, 1,500 Q, 500 per source, 1,003,520 px)
Same questions in every row. Rate 8 tokens per slot. Uncompressed = Qwen3.5-4B reading the same pages raw.

| input | `<MEM>` slots per question p50 / max | C256 s42 judge ↑ | L1024 s42 judge ↑ | L1024 − C256 s42 (95% CI) | C256 s43 judge ↑ | L1024 s43 judge ↑ | L1024 − C256 s43 (95% CI) | uncompressed judge ↑ |
|---|---|---|---|---|---|---|---|---|
| 4 pages (gold + 3 distractors) | 455 / 485 | .615 | .553 | −6.2 [−8.6, −3.9] | .603 | .561 | −4.3 [−6.7, −1.8] | .809 |
| gold page only | 119 / 238 | .635 | .568 | −6.7 [−8.9, −4.5] | .628 | .579 | −4.9 [−7.1, −2.5] | .841 |
| picked page raw, 3 others compressed | 348 / 367 + 1 raw page | .825 | .827 | +0.1 [−0.8, +1.1] | .824 | .823 | −0.1 [−1.1, +0.9] | .809 (4 pages raw) |

Reading: L1024 answers worse than C256 even when the decoder sees only the one page that holds the answer
(gap unchanged from 455 to 119 slots per question). When the picked page goes in raw, the gap vanishes at both seeds,
and both models pick the answer page equally often (98.5%). So L1024's page `<MEM>`s still say WHERE the answer is,
but hold the answer itself worse, and the decoder is not the problem.
Off-regime check (451,584 px, not gating): gold page only −6.3 (s42) / −4.7 (s43), the same gap at a smaller page budget.

### Arm T detail: what the compressed slots hold (no decoder question asked)
| probe | model stage | n | metric (direction) | C256 | L1024 | L1024 − C256 (95% CI) | sign test (L1024 better / worse) | gate | verdict |
|---|---|---|---|---|---|---|---|---|---|
| transcription: rebuild the gold page from its slots, compare with the uncompressed model's transcript | stage 2 (image pages), 1,003,520 px, ~97 slots per page | 500 pages (167 per source) | share of the page's words rebuilt ↑ | .647 | .566 | −8.0 pt [−10.6, −5.5]; infovqa −8.9, MP-DocVQA −11.0, SlideVQA −4.3 [−9.0, +0.2] | 147 / 322, p 4e-16 | ≥ 3 pt fewer, p < .05 | provisional: MET — L1024's page slots hold fewer words. Answer string found in the rebuild: 228 vs 253 of 500 |
| autoencoding: rebuild one 256-token chunk (the length later stages use) | stage 1 (text) | 500 held-out MS MARCO documents | nats per token ↓ | .0144 | .0126 | −.0018 [−.0031, −.0006] | 271 / 229, p .067 | L1024 worse at 256 | provisional: NOT MET — both near-perfect, L1024 not worse |
| autoencoding: rebuild one 1,024-token chunk | stage 1 (text) | 500 documents | nats per token ↓ | .1027 | .0093 | −.0934 [−.0988, −.0884] | 500 / 0 | L1024 better at 1,024 | MET (expected: C256 never trained on 1,024-token targets) |
| autoencoding: rebuild 1,024 tokens from 4 chunks of 256 | stage 1 (text) | 500 documents | nats per token ↓ | .0442 | .0161 | −.0281 [−.0301, −.0263] | 486 / 14 | — | reported |
| control: another document's slots | stage 1 (text) | 500 documents | nats per token ↓ | 2.575 | 2.715 | — | — | far worse than own slots | PASS (self-pair 0 on every row) |
| autoencoding: rebuild one 256-token chunk | stage 2 (after page training) | 500 documents | nats per token ↓ | .0284 | .0290 | +.0006 [−.0008, +.0019] | 247 / 253, p .82 | L1024 worse at 256 (CI above 0) → stage 2 erodes text too | provisional: NOT MET — the stage-2 loss is specific to image pages |
| autoencoding: rebuild one 1,024-token chunk | stage 2 | 500 documents | nats per token ↓ | .0404 | .0255 | −.0149 [−.0167, −.0133] | 432 / 68 | — | reported (L1024 still better on long text) |
| control: another document's slots | stage 2 | 500 documents | nats per token ↓ | 2.445 | 2.495 | — | — | far worse than own slots | PASS (self-pair 0) |

Reading: the long-chunk model rebuilds short TEXT chunks as well as C256 after stage 1 and still after stage 2 (both
near-perfect, .013 → .029 nats per token for both). Its IMAGE-page slots hold 8 pt fewer words after stage 2. So the
damage is specific to image pages and enters during stage 2 (page training on top of the long-chunk stage 1).
Training arm this motivates (owner's call): keep stage-1 chunks at the length later stages compress; a long-chunk stage
1 needs its stage 2 changed, and a page-transcription check before any finetune.

### Arm T detail: text, ten short chunks vs one long chunk (`text_hotpotqa_1500`, 1,500 Q, CML 4,096)
Rate 8 tokens per slot; both layouts hold the same 10 passages (slots per question p50 / max: 173 / 418 as ten chunks,
169 / 414 as one chunk). Uncompressed Qwen3.5-4B on the same passages: .752.

| layout | C256 s42 judge ↑ | L1024 s42 judge ↑ | L1024 − C256 s42 (95% CI, McNemar) | C256 s43 judge ↑ | L1024 s43 judge ↑ | L1024 − C256 s43 (95% CI, McNemar) |
|---|---|---|---|---|---|---|
| one ~1.3k-token chunk (g10, recorded) | .611 | .600 | −1.1 [−3.2, +1.2], 138 / 154 | .607 | .574 | −3.3 [−5.4, −1.1], 126 / 175 |
| ten short chunks (this campaign) | .579 | .603 | +2.3 [+0.3, +4.4], 136 / 101 | .595 | .589 | −0.6 [−2.7, +1.3], 121 / 130 |
| difference of the two gaps (paired per question) | | | +3.4 [+0.5, +6.1], sign 228 / 179, p .017 | | | +2.7 [−0.3, +5.5], sign 223 / 178, p .028 |

Reading: on text the long-chunk model is no worse when passages stay short; its small text loss shows only on one long
chunk (both seeds, sign p < .05, but below the 3-pt bar at s43: consistent with, not decided). Odd but noted: it
autoencodes long text better (above), yet answers worse from it.
Probe validity: the transcription port reproduced the legacy probe exactly (360 / 360 pages identical, word recall
70.5% = 70.5%, job 191169); a model paired with itself gives 0 in both probes.

## 2. Length curves (accuracy against input size)
Running (arm L, documents, seed 42: jobs for 10 and 20 pages and uncompressed 20 pages). Columns will be: input size | `<MEM>` slots per question | compression rate |
compressed judge accuracy ↑ | uncompressed judge accuracy ↑ | gap ↑ (pt) | Δ_len vs the shortest size (95% CI).

## 3. Input audit: what the decoder sees in training vs in evaluation
Job 191147, model R (expQ/ft_mm_4b_184778), images at 1,003,520 px, rate 8 tokens per slot, position stride 8.
Training rows: 2,000 sampled per stage. Eval rows: every question of the set. "Span" = the decoder position span of
the prompt (each `<MEM>` takes 8 positions). "Out of range" = share of questions above the finetune sample's maximum
`<MEM>` count (488) or position span (4,017).

| what | n | `<MEM>` slots per prompt p50 / p99 / max ↓ | decoder prompt tokens p50 / max | position span p50 / p99 / max | uncompressed prompt max (tokens) | out of range |
|---|---|---|---|---|---|---|
| training: stage 1, chunk 256 (C256) | 2,000 | 33 / 66 / 66 | 512 / 512 (at its cap) | 743 / 974 / 974 | — | — |
| training: stage 1, chunk 1,024 (L1024) | 2,000 | 76 / 258 / 258 | 700 / 2,048 | 1,235 / 3,854 / 3,854 | — | — |
| training: stage 2 (C256 chain) | 2,000 | 55 / 130 / 132 | 380 / 1,024 | 781 / 1,934 / 1,948 | — | — |
| training: finetune (C256 chain) | 2,000 | 476 / 488 / 488 | 560 / 630 | 3,904 / 3,994 / 4,017 | — | (reference) |
| eval: documents, 4 pages | 1,500 | 455 / 484 / 485 | 536 / 583 | 3,720 / 3,952 / 3,962 | 3,949 | 0% |
| eval: documents, 10 pages | 1,500 | 1,125 / 1,203 / 1,208 | 1,228 / 1,322 | 9,105 / 9,726 / 9,767 | 9,727 | 100% |
| eval: documents, 20 pages | 1,500 | 2,234 / 2,398 / 2,405 | 2,387 / 2,567 | 18,027 / 19,338 / 19,393 | 19,308 | 100% |
| eval: documents, gold page only | 1,500 | 119 / 194 / 238 | 183 / 321 | 1,015 / 1,654 / 1,987 | 1,977 | 0% |
| eval: text, 10 passages (chunk ≤ 4,096) | 1,500 | 173 / 299 / 418 | 282 / 525 | 1,493 / 2,505 / 3,451 | 3,397 | 0% |
| eval: text, 20 passages | 1,500 | 352 / 521 / 616 | 511 / 782 | 2,979 / 4,319 / 5,094 | 4,982 | 2.3% |
| eval: text, 40 passages | 1,500 | 713 / 919 / 1,035 | 972 / 1,289 | 5,965 / 7,601 / 8,534 | 8,315 | 99.9% |
| eval: text, 10 passages joined (g10, chunk ≤ 4,096) | 1,500 | 169 / 296 / 414 | 242 / 485 | 1,422 / 2,441 / 3,383 | 3,381 | 0% |
| eval: whole documents, up to 20 pages (500 Q) | 500 | 357 / 2,398 / 2,420 | 432 / 2,567 | 2,938 / 19,331 / 19,507 | 19,427 | 39.6% |

Reading: the 4-page document eval sits exactly at the finetune maximum; every longer document setting, and 40
text passages, puts the decoder on slot counts and positions it never saw in finetuning. This is a count. It names
a cause only if the reading or position probes show a paired loss.

Side finding (audit only, no accuracy effect measured): stage-1 C256 decoder rows sit at their 512-token cap on the
median row.

## 4. Figure
`expQ/longdoc/figures/length_curve.png` — not drawn yet (needs arm L).
