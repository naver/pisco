# Long inputs: where is the bottleneck? (campaign longdoc)

Rewritten by each campaign wake when an arm decides. Last rewrite 2026-11-17 07:45 (arm T: the long-chunk loss sits in the compressed page; L1024's stage-2 page slots hold 8 pt fewer words; its stage-1 text model is not worse on short chunks).
Question (owner): every attempt at long inputs loses accuracy — is the cause the evaluation, the training data,
training itself, or how the decoder reads many `<MEM>` slots (one `<MEM>` = one compressed slot, 8 tokens each)?

## 1. Bottleneck table
Accuracy = Qwen3-8B rubric judge, paired on the same questions. pt = 0.01. A cause needs ≥ 3 pt with the 95% CI
on the far side of 0 at both seeds; "not this cause" needs the CI inside ±3 pt.

| candidate | probe | n | number | gate | verdict | questions repaired |
|---|---|---|---|---|---|---|
| evaluation feeds less than it claims | input audit, CPU (job 191147) | 1,500 Q per variant × 14 variants (one set of 500) | truncation cut 0 / 1,500 and pisco-vs-uncompressed parity 1,500 / 1,500 on every gating variant; the cut check does fail when it should (CML 256 controls: 924 / 1,500 and 275 / 1,500). Limits: on documents the cut check can only fail on prompt length (pages are images, never cut); the audit's parity used one row builder for both modes, so it was re-checked on recorded runs: pages per question equal in both modes on 1,500 / 1,500 (4 pages), 1,500 / 1,500 (10 pages), 500 / 500 (whole documents) — count, not page identity | cut ≤ 1%, parity 100% | provisional: not this cause (gate passed; parity only partly independent) | 0 |
| training never shows long inputs | input audit (count only) | as above | 10- and 20-page documents: 100% of questions beyond the finetune range; 40 text passages: 99.9%; 20 passages: 2.3% | names a cause only with a paired effect from the reading or position probes | open (count, not yet a cause) | — |
| long-chunk training (stage-1 chunk 1,024 vs 256) | gold page only / picked page raw on the C256 and L1024 finetunes | 1,500 Q × 2 seeds | gold page only (distractor pages removed, 1,003,520 px): L1024 − C256 −6.7 [−8.9, −4.5] McNemar 106 / 207 (s42), −4.9 [−7.1, −2.5] 113 / 186 (s43), every source negative; the all-4-pages deficit was −6.2 / −4.3, so removing distractors does not shrink it. Picked page raw (the model picks the answer page from its `<MEM>`s, that page goes raw, the other 3 stay compressed): +0.1 [−0.8, +1.1] McNemar 29 / 27 (s42), −0.1 [−1.1, +0.9] 29 / 31 (s43), every source within ±0.6 pt; both models pick the answer page on 98.5% of questions | deficit stays ≤ −3 pt with the gold page alone AND shrinks inside ±2 pt with the picked page raw → it sits in the compressed page; shrinks inside ±2 pt with the gold page alone → multi-page reading | provisional: IN THE COMPRESSED PAGE (both halves met at both seeds). Not multi-page reading, not the page pick, not how the decoder reads a raw page. The page slots of L1024's stage-2 model hold 8.0 pt fewer of the page's words (500 pages, sign p 4e-16), while its stage-1 text model rebuilds 256-token chunks as well as C256's: the loss enters at stage 2 or later | the whole 4-page deficit: ~93 (s42) / ~65 (s43) of 1,500 questions (−6.2 / −4.3 pt × 1,500) |
| length on a fixed model | 4 / 10 / 20 pages, 10 / 20 / 40 passages | — | not started | Δ_len ≤ −3 pt | open | — |
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

Reading: at stage 1 the long-chunk model lost nothing on short text chunks (if anything it is slightly better). After
stage 2 its image-page slots hold 8 pt fewer words. So the damage enters during stage 2 (training on pages on top of the
long-chunk stage 1), not in what stage 1 learned about short chunks. Running (2026-11-17 07:45): the same autoencoding
probe on the two stage-2 models (is text detail lost too, or only image detail?) and ten short text chunks on the four
finetunes. Training arm this motivates (owner's call): keep stage-1 chunks at the length later stages compress; a
long-chunk stage 1 needs a stage-2 check (page transcription) before any finetune.
Probe validity: the transcription port reproduced the legacy probe exactly (360 / 360 pages identical, word recall
70.5% = 70.5%, job 191169); a model paired with itself gives 0 in both probes.

## 2. Length curves (accuracy against input size)
Not measured yet (arm L). Columns will be: input size | `<MEM>` slots per question | compression rate |
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
