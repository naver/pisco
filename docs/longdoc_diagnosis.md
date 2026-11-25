# Long inputs: where is the bottleneck? (campaign longdoc)

Rewritten by each campaign wake when an arm decides. Last rewrite 2026-11-26 03:10: every funded cell is in (96 of 96 runs, 98.3 GPU-h of 114). Arm U (slot reading) is positive at both seeds but meets the 3-pt cause bar at one seed only (D +2.8 / +3.3 pt); arm P (positions) is not a cause at either seed, also under the second judge prompt. The ranked answer below goes to the owner as the final checkpoint.
Question (owner): every attempt at long inputs loses accuracy — is the cause the evaluation, the training data,
training itself, or how the decoder reads many `<MEM>` slots (one `<MEM>` = one compressed slot, 8 tokens each)?

## 0. The answer, ranked by questions lost (provisional; LongDocURL, 1,407 questions, both seeds)
| rank | cause | questions it costs (of 1,407) | status | training arm it motivates |
|---|---|---|---|---|
| 1 | the compression itself, at every length (not a length effect) | ~233 (s42) / ~258 (s43) already at 10 pages | measured at both seeds, every set (−16.6 / −18.3 pt at 10 pages) | better slots (stage 2: campaigns slotcheck / rebuild) |
| 2 | extra loss with length, page images, 40 vs 10 pages | ~91 / ~75 (Δ_len −6.5 / −5.3 pt) | DECIDED (provisional) at both seeds, three of four scorers | — (split below) |
| 2a | of which: the decoder reads thousands of `<MEM>`s worse than a few hundred | ~40 / ~47 (D +2.8 / +3.3 pt) | positive at both seeds and under every scorer; meets the 3-pt bar at s43 only: consistent with, not confirmed | train the decoder on long inputs: finetune / stage-2 rows with 10-40 pages (1,000-5,000 slots; today ≤ 488) |
| 2b | of which: position ids past the trained range | 0 (−0.6 / −0.1 pt) | NOT a cause at both seeds | none |
| 2c | of which: unexplained | ~2-4 pt (Δ_len minus D; approximate: D is measured on the compressed reader alone) | — | — |
| 3 | long-chunk stage-1 training (1,024-token chunks) | ~93 / ~65 of 1,500 on 4-page documents | CLOSED (arm T): hurts image pages in stage 2; long text: seeds disagree | keep the 256-token stage 1 |
| — | the evaluation | — | sound for arm E (nothing cut, parity, needle, closed-book); the shared 32-token answer cap underrates the uncompressed reader (handed off) | — |

## 1. Bottleneck table
Accuracy = Qwen3-8B rubric judge, paired on the same questions. pt = 0.01. A cause needs ≥ 3 pt with the 95% CI
on the far side of 0 at both seeds; "not this cause" needs the CI inside ±3 pt.

| candidate | probe | n | number | gate | verdict | questions repaired |
|---|---|---|---|---|---|---|
| evaluation feeds less than it claims | input audit, CPU (job 191147) | 1,500 Q per variant × 14 variants (one set of 500) | truncation cut 0 / 1,500 and pisco-vs-uncompressed parity 1,500 / 1,500 on every gating variant; the cut check does fail when it should (CML 256 controls: 924 / 1,500 and 275 / 1,500). Limits: on documents the cut check can only fail on prompt length (pages are images, never cut); the audit's parity used one row builder for both modes, so it was re-checked on recorded runs: pages per question equal in both modes on 1,500 / 1,500 (4 pages), 1,500 / 1,500 (10 pages), 500 / 500 (whole documents) — count, not page identity | cut ≤ 1%, parity 100% | provisional: not this cause (gate passed; parity only partly independent) | 0 |
| the judge decides the number (arm J) | every length number re-scored with the project (Yes/No) prompt on copies (job 191629) and with exact match | 1,500 pairs per number | Δ_len rubric / project / EM: documents 10 vs 4 pages +0.5 / +0.6 / +1.6 (s42), +0.1 / −0.3 / +0.9 (s43); 20 vs 4 pages +4.3 / +5.1 / +7.7 (s42), +3.1 / +3.1 / +6.1 (s43, project re-judge 192142); text 40 vs 10 passages −2.6 / −1.5 / +0.8 (s42), +0.1 / +0.9 / +2.7 (s43). All these files carry the 32-token cap confound (next row). Calculator control: recovers the known −.169 within ±.005, self-pair 0 | a number keeps its sign and stays within 2 pt under the three scorers | provisional: by the law's rule (same sign AND within 2 pt under all three) all three are judge-sensitive: 10 pages s43 flips sign (+0.1 / −0.3, every CI inside ±3, so the no-loss reading survives); 20 pages keeps its sign but EM is 3.4 pt larger; text 40 passages flips sign with EM | — |
| the uncompressed reference is capped (arm E) | count uncompressed answers that reach the answer cap (token count with the Qwen3.5 tokenizer, padding off), same 300 trial rows per set at 32 / 256 / 512 tokens (jobs 191634 / 191635, 191669 / 191670, 191777 / 191778) | 300 rows per set | 32 tokens: 61-74% of page answers and 47-64% of text answers cut mid-sentence (character heuristic). 256 tokens: pages 2.7% at the cap, judge .340 → .717 (+37.7 [+32.0, +43.0], 114 / 1); text 8.0% at the cap, .423 → .580 (+15.7 [+10.7, +20.7], 58 / 11). 512 tokens: text 2.3% at the cap, judge .583; mixed 2.3%, .653; closed-book 5 / 1,407; R mixed 1.3%. R barely moves (pages .517 → .547 from 32 to 256). Gating documents set at 32 tokens: 4% (SlideVQA 11%). **Arm L files (1,500 Q each, 32 tokens, answers ≥ 30 tokens):** uncompressed documents 4 / 10 / 20 pages 5.7% / 6.9% / 16.3%, text 10 / 40 passages 32.9% / 46.9%; R 0.6-1.7% on documents, 2.7-5.9% on text (`expQ/longdoc/length/cap32_sensitivity.json`) | share at the cap < 5% | provisional: YES on LongDocURL — the shared 32-token cap made the uncompressed reader look worse than the compressed one. Fixed: every arm E cell runs at 512 tokens; the full uncompressed cells at 512 have 2.1% of text answers at the cap (47 / 2,246, job 192266), pages 0.6 / 1.0 / 1.0%, mixed 1.1 / 1.8 / 2.2% (1,407 rows each), all under 5%. YES on arm L too: every uncompressed arm L file fails the bar, and more at the long size, so every arm L Δ_len is pushed upward (next row) | — |
| training never shows long inputs | input audit (count only) | as above | 10- and 20-page documents: 100% of questions beyond the finetune range; 40 text passages: 99.9%; 20 passages: 2.3%; every arm E bucket 99-100% | names a cause only with a paired effect from the reading or position probes | provisional: position ids out of range are NOT a cause (arm P). The slot count out of range has a paired effect at both seeds (arm U, +2.8 / +3.3 pt), below the cause bar at s42: consistent with, not confirmed. Note: the kept eighth at 40 pages (~580 slots per question) sits near the finetune maximum (488) | ~40 / ~47 if arm U holds |
| long-chunk training (stage-1 chunk 1,024 vs 256) | gold page only / picked page raw on the C256 and L1024 finetunes; page transcription and text autoencoding on their earlier stages; ten short vs one long text chunk | 1,500 Q × 2 seeds (QA); 500 pages / 500 documents (probes) | gold page only (distractor pages removed, 1,003,520 px): L1024 − C256 −6.7 [−8.9, −4.5] McNemar 106 / 207 (s42), −4.9 [−7.1, −2.5] 113 / 186 (s43). Picked page raw: +0.1 [−0.8, +1.1] (s42), −0.1 [−1.1, +0.9] (s43); both models pick the answer page on 98.5%. Page transcription (stage 2): −8.0 pt [−10.6, −5.5] of the page's words, p 4e-16. Text autoencoding at 256 tokens: stage 1 −.0018 nats, stage 2 +.0006 [−.0008, +.0019] p .82 (L1024 not worse). Text QA: −1.1 / −3.3 on one ~1.3k-token chunk, +2.3 / −0.6 on ten short chunks. Arm E long text (one PDF, ~4k-32k tokens, CML 1,024, n 2,246): +4.6 [+3.1, +6.2] 231 / 127 (s42; C256 s42 is 5.0 pt below R), −0.0 [−1.7, +1.7] 184 / 185 (s43) | deficit ≤ −3 pt with the gold page alone AND inside ±2 pt with the picked page raw → in the compressed page; stage-2 text autoencoding worse (CI above 0) → general stage-2 loss, else image-specific | provisional: CLOSED. IN THE COMPRESSED IMAGE PAGE, entering in stage 2: L1024's page slots hold fewer words, its text slots do not. Not multi-page reading, not the page pick, not the decoder. Motivates (owner): keep the 256-token stage 1; a long-chunk stage 1 would need its stage 2 changed, checked by page transcription before any finetune. On one long text document (arm E) the seeds disagree (+4.6 / 0.0 pt): long-chunk stage 1 is not shown to help long text either | the whole 4-page deficit: ~93 (s42) / ~65 (s43) of 1,500 questions |
| length on a fixed model | 4 / 10 / 20 pages, 10 / 20 / 40 passages (arm L); 10 / 20 / 40 pages, 10 / 20 / 40 mixed items, ~4k-32k text tokens of one PDF (arm E) | 1,500 Q per size (L); 1,407 per size, text 411-637 per bucket (E) | all 1,500 Q, 32-token cap: documents 10 vs 4 pages +0.5 [−1.8, +2.7] (s42), +0.1 [−2.2, +2.5] (s43); 20 vs 4 pages +4.3 [+1.8, +6.9] (s42), +3.1 [+0.6, +5.7] sign 195 / 153 p .028 (s43); text 40 vs 10 passages −2.6 [−5.6, +0.3] (s42), +0.1 [−3.0, +3.1] (s43). Only questions where no file's answer reaches the cap (a sensitivity read: the subset is chosen on the answers): 10 pages −0.1 / −0.4 (n 1,357 / 1,354); 20 pages −0.6 [−3.3, +2.0] / −2.0 [−4.5, +0.4] (n 1,221 / 1,220); text 40 passages −4.8 [−8.3, −1.5] / −3.4 [−6.8, 0.0] (n 731 / 733). These sets add distractor pages or passages from other documents: not one long document (arm E builds that). Arm E (one long PDF per question, cap 512, both seeds, paired per question on itself at the shortest size, n 1,407): pages 40 vs 10 −6.5 [−9.5, −3.5] sign 165 / 249 p 4e-5 (s42), −5.3 [−8.4, −2.3] 177 / 245 p .001 (s43); pages 20 vs 10 −2.8 / −1.9; mixed 20 / 40 vs 10 −0.3 / −1.4 (s42), −1.2 / −3.0 [−5.9, 0.0] (s43); text ~8k / ~16k / ~32k vs ~4k +0.7 / +3.6 / +3.9 (s42), −2.5 / −0.7 / −1.7 (s43), every CI wider than ±3 (n 610 / 588 / 411); gap R − uncompressed −16 to −24 pt at every size, the seeds within 2 pt in every cell | Δ_len ≤ −3 pt (CI below 0) at both seeds = length loss; CI inside ±3 pt at both = none; every file < 5% of answers at the cap | provisional: DECIDED on arm E for page images — a length loss at 40 pages (both seeds, three of four scorers; by arm J's letter the EM flip makes it judge-sensitive, sourced to EM's fault on the reference); not decided for 20 pages, mixed, text. Cause (slots vs positions) needs arms U / P: owner question. Arm L cannot decide (its uncompressed files fail the cap bar; the "gap shrinks at 20 pages" is withdrawn: it vanishes without the capped questions; the 512 re-run was not approved). Arm E: page images lose with length at 40 pages, past the bar at BOTH seeds (rubric judge −6.5 / −5.3; project prompt −5.7 [−8.7, −2.6] / −6.1 [−9.2, −3.0], job 192405; match −4.6 / −3.6: all within 2 pt; EM flips the sign, a fault of EM on the sentence-long uncompressed answers). 20 pages, mixed and text meet no bar. The length share is small: −5 to −6 pt of a −23 pt gap at 40 pages; −17 to −20 pt is there at the shortest size | — |
| reading many `<MEM>`s | keep only the most-attended eighth of the slots (`KEEPF=0.125 KEEPM=attn_global`), at 10 and 40 pages of one LongDocURL PDF, R, cap 512 (arm U) | 1,407 Q per cell, 4 cells per seed | D = (eighth − plain at 40 pages) − (eighth − plain at 10 pages). Seed 42: eighth − plain at 40 pages +2.5 [+0.8, +4.3], eighth only right / plain only right 105 / 70 (jobs 193536 vs 192324); at 10 pages −0.4 [−2.1, +1.4] 77 / 82 (193606 vs 192326); D +2.8 [+0.4, +5.3] sign 152 / 113 p .019. Seed 43: at 40 pages +3.8 [+1.8, +5.9] 136 / 82 (193608 vs 192364); at 10 pages +0.5 [−1.4, +2.3] 91 / 84 (193607 vs 192360); D +3.3 [+0.8, +6.0] sign 173 / 135 p .035. Other scorers, D s42 / s43: project prompt +1.8 [−0.9, +4.5] / +3.8 [+0.9, +6.6] (job 193645), match +1.9 / +1.9, EM +1.8 / +1.9: same sign, within 2 pt of the rubric judge at each seed | D ≥ +3 pt, CI above 0, both seeds = the decoder reads many slots worse than few; CI inside ±3 at both = not this cause | provisional: BETWEEN THE BARS — positive at both seeds under every scorer, CI above 0 under the rubric at both, ≥ 3 pt at s43 only (s42 +2.8). Not confirmed, not ruled out. A repeat on new items needs runs beyond the cap (owner) | ~40 / ~47 of 1,407 at 40 pages |
| positions past the trained range | each 10-page question's `<MEM>`s and question moved to the position ids its own 40-page version uses (offset p50 28,934, min 27,990, max 29,670); slot count unchanged; `MPOS=0` first reproduced the plain run exactly (300 / 300, job 193603) (arm P) | 1,407 pairs per seed | shifted − plain: −0.6 [−1.4, +0.3] McNemar 16 / 24 p .27 (s42, job 193604 vs 192326), −0.1 [−0.8, +0.6] 11 / 13 p .84 (s43, 193605 vs 192360); project prompt −0.4 [−1.4, +0.6] / −0.1 [−0.8, +0.6] (job 193645); match +0.2 / +0.1, EM −0.6 / −0.4. The shift is real: 207 / 194 of 1,407 answers change wording, almost none change correctness | offset cost ≤ −3 pt with the CI below 0 at both seeds = cause; CI inside ±3 at both = not this cause | provisional: NOT THIS CAUSE at both seeds, under both judge prompts. The prediction written before the cells ("P costs ≥ 3 pt at both seeds") was WRONG | 0 |

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

### Arm U detail: keep only the most-attended eighth of the slots (LongDocURL pages, 1,407 Q per cell, cap 512)
Rate 8 tokens per slot, ~120 slots per page. "Eighth" = per question, one decoder pass scores every `<MEM>` slot by the attention
the last prompt position gives it; the top 12.5% over the whole document are kept, the others leave the prompt, and the
answer is generated from the shorter prompt (the kept slots' position ids close up; arm P shows positions alone cost
nothing, so the gain is the slot count) (`scripts/eval_mm_docqa.py:290-312`). Uncompressed on the same
pages: .672 (10 pages), .659 (40 pages).

| pages | slots per question (mean) plain / eighth | R plain s42 judge ↑ | R eighth s42 judge ↑ | eighth − plain s42 (95% CI) | R plain s43 judge ↑ | R eighth s43 judge ↑ | eighth − plain s43 (95% CI) |
|---|---|---|---|---|---|---|---|
| 10 | 1,200 / 145 | .507 (192326) | .503 (193606) | −0.4 [−2.1, +1.4] | .489 (192360) | .494 (193607) | +0.5 [−1.4, +2.3] |
| 40 | 4,803 / 580 | .429 (192324) | .453 (193536) | +2.5 [+0.8, +4.3] | .422 (192364) | .461 (193608) | +3.8 [+1.8, +5.9] |
| D (40-page gain − 10-page gain) | | | | +2.8 [+0.4, +5.3], sign p .019 | | | +3.3 [+0.8, +6.0], sign p .035 |

Reading (provisional): dropping 7 of every 8 slots costs nothing at 10 pages and helps at 40 pages, at both seeds:
the decoder answers better from ~580 well-chosen slots than from all ~4,800. It recovers about a third to a half of
R's own 10 → 40 page drop (−7.8 / −6.7 pt). The kept 580 slots sit near the most the decoder saw in finetuning (488),
which fits "the decoder was never trained to read thousands of slots", but this is not tested directly. Health: every
row judged, 0 unparseable, answers at the cap 0.6-2.0%, slots used .121 of the plain cells by design, code
model.py 2a1cc007. Tables `expQ/longdoc/length/U_D_pages_s4{2,3}_{judge,match,em,project}.json`.

## 2. Length curves (accuracy against input size)
Model R = reference finetune (expQ/ft_mm_4b_184778, seed 42; 184780 = seed 43). Same 1,500 questions at every size
(500 per source); longer inputs add distractor pages from other documents, shuffled, the same draw in both modes.
Rate 8 tokens per slot. Uncompressed = Qwen3.5-4B reading the same pages raw. Gap = compressed − uncompressed.
Δ_len = (R − uncompressed at this size) − (R − uncompressed at 4 pages), paired per question.

**Documents (`mm_docqa_1500_1m`, 1,003,520 px)** — every row ran at the 32-token answer cap; "at cap" = share of the
uncompressed answers that reach it (bar 5%). Δ_len no-cap = the same contrast on the questions where no file reaches it.

| input size | `<MEM>` slots per question p50 / max | compression (slots per page) | R s42 judge ↑ | R s43 judge ↑ | uncompressed judge ↑ | uncompressed at cap ↓ | gap s42 / s43 (pt) ↑ | Δ_len s42 (95% CI, sign test) | Δ_len s43 | Δ_len no-cap s42 / s43 |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 pages | 455 / 485 | ~114 | .610 | .615 | .809 | 5.7% | −19.9 / −19.4 | 0 (reference) | 0 (reference) | 0 |
| 10 pages | 1,125 / 1,208 | ~113 | .607 | .608 | .801 | 6.9% | −19.4 / −19.3 | +0.5 [−1.8, +2.7], 144 / 138, p .77 | +0.1 [−2.2, +2.5], 146 / 142, p .86 | −0.1 / −0.4 |
| 20 pages | 2,234 / 2,405 | ~112 | .599 | .591 | .754 | 16.3% | −15.5 / −16.3 | +4.3 [+1.8, +6.9], 197 / 146, p .007 | +3.1 [+0.6, +5.7], 195 / 153, p .028 | −0.6 [−3.3, +2.0] / −2.0 [−4.5, +0.4] |

Reading (provisional, decides nothing): at 20 pages the uncompressed reader's drop (.809 → .754) is mostly answers cut
by the cap: 245 of its 1,500 answers reach it at 20 pages vs 86 at 4 pages. Leaving out the capped questions, the gap
no longer shrinks (−0.6 / −2.0). The compressed reader is nearly flat from 4 to 20 pages at both seeds (−1.1 / −2.4 pt).
The added pages come from OTHER documents; arm E measures one long coherent document instead.

**Text (`text_hotpotqa_1500`, HotpotQA, ~140-token passages, chunk ≤ 4,096 tokens)** — 32-token cap, as above.

| input size | `<MEM>` slots per question p50 / max | compression (slots per passage) | R s42 judge ↑ | R s43 judge ↑ | uncompressed judge ↑ | uncompressed at cap ↓ | gap s42 / s43 (pt) ↑ | Δ_len s42 (95% CI, sign test) | Δ_len s43 | Δ_len no-cap s42 / s43 |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 passages | 173 / 418 | ~17 | .605 | .593 | .752 | 32.9% | −14.7 / −15.9 | 0 (reference) | 0 (reference) | 0 |
| 40 passages | 713 / 1,035 | ~18 | .534 | .549 | .707 | 46.9% | −17.3 / −15.8 | −2.6 [−5.6, +0.3], 199 / 236, p .084 | +0.1 [−3.0, +3.1], 220 / 224, p .89 | −4.8 [−8.3, −1.5] / −3.4 [−6.8, 0.0] |

Reading (provisional, decides nothing): a third to a half of the uncompressed text answers are cut by the cap, so the
uncompressed column is not a clean reference here. On the questions no cap touches (about 730 of 1,500), the gap widens
by 3-5 pt from 10 to 40 passages at both seeds: a length loss is plausible on text, but the subset is chosen on the
answers, so only a re-run at 512 tokens can decide it. R at 10 passages is borrowed from campaign spread (jobs 191518 /
191630, the same models 184778 / 184780, no committed eval-code change between the runs).

**Arm E trial runs (LongDocURL, one coherent PDF per question; 300 rows per set = 100 questions × 3 page sizes, or 75 × 4 text sizes; trial runs prove the path, they decide nothing)**
Rate 8 tokens per slot. Model R s42. Answer cap 256 tokens, both readers, same rows (jobs 191669 / 191670 / 191673 / 191674).
"Cut" = answers that stop at the cap (≥ 600 characters, no final punctuation).

| input size | `<MEM>` slots per question p50 | compression | R s42 judge ↑ | uncompressed judge ↑ | gap (pt) ↑ | uncompressed answers cut ↓ | R answers cut ↓ |
|---|---|---|---|---|---|---|---|
| 10 pages | 1,198 | ~120 slots per page | .58 | .71 | −13 | 2% | 1% |
| 20 pages | 2,392 | ~120 slots per page | .56 | .71 | −15 | 1% | 1% |
| 40 pages | 4,795 | ~120 slots per page | .50 | .73 | −23 | 4% | 2% |
| all pages (300 rows) | — | — | .547 | .717 | −17.0 [−23.0, −11.0], 24 / 75 | 2.3% | 1.3% |
| text ~4k tokens | 540 | 8 tokens per slot | .427 | .587 | −16 | 5% | 3% |
| text ~8k | 1,046 | 8 tokens per slot | .400 | .587 | −19 | 5% | 4% |
| text ~16k | 2,055 | 8 tokens per slot | .453 | .573 | −12 | 8% | 3% |
| text ~32k | 4,079 | 8 tokens per slot | .387 | .573 | −19 | 15% | 5% |
| all text (300 rows) | — | — | .417 | .580 | −16.3 [−21.7, −11.0], 15 / 64 | 8.3% | 3.7% |

Reading: with room to finish its answers, the uncompressed reader scores .72 on pages and .58 on text, well above R
(the 32-token cap had it at .34 / .42, below R). The pages gap widens at 40 pages (−13 → −23 pt) while the uncompressed
reader stays flat; 100 questions per size cannot decide it (a full cell can). At 512 tokens the uncompressed text score
is .583 (.587 / .587 / .587 / .573 from ~4k to ~32k tokens: flat with length) and 2.3% of answers reach the cap, so 512
is the arm E cap. The "cut" columns above use a character heuristic (≥ 600 characters, no final punctuation); the token
count at 256 tokens (padding off) is 2.7% on pages and 8.0% on text.

**Arm E trial, mixed (R s42 vs uncompressed, cap 512, 300 rows = 100 questions × 3 sizes; jobs 191887 / 191778; decides nothing)**

| input size (items: pages as images + pages as their text layer) | `<MEM>` slots per question p50 | R s42 judge ↑ | uncompressed judge ↑ | gap (pt) ↑, 95% CI | R only right / uncompressed only right |
|---|---|---|---|---|---|
| 10 items | 977 | .51 | .63 | −12 [−23, −1] | 11 / 23 |
| 20 items | 1,922 | .52 | .66 | −14 [−26, −2] | 12 / 26 |
| 40 items | 3,821 | .46 | .67 | −21 [−32, −10] | 8 / 29 |
| all (300 rows) | — | .497 | .653 | −15.7 [−22.0, −9.3], McNemar p < .001 | 31 / 78 |

Reading: like pages, the mixed gap grows from 10 to 40 items while the uncompressed reader stays flat; 100 questions per
size cannot decide it. R used exactly the `<MEM>` count the audit predicted (715,709 on 300 / 300 rows).

**Arm E soundness check 1 — is the document needed? (DECIDED on the full uncompressed cells, cap 512 for both files)**
Closed-book = the uncompressed decoder with the question only (job 191776, cap 512, every one of the 1,407 questions once:
judge .164, EM .090). Each row is paired with its question's closed-book answer (`scripts/longdoc/closedbook_pair.py`,
table `expQ/longdoc/soundness/closedbook_full.json`). Bar: with the document − closed-book ≥ 20 pt per bucket.
Uncompressed with the document: jobs 192266 (text, 2,246 rows), 192287 / 192288 / 192267 (pages 10 / 20 / 40),
192289 / 192290 / 192268 (mixed 10 / 20 / 40), 1,407 rows each; every row judged, 0 unparseable, 0.6-2.2% of the
answers at the cap (bar 5%; `expQ/longdoc/soundness/uncompressed_full_health.json`).

| set, bucket | questions | uncompressed with the document, judge ↑ | closed-book judge ↑ | difference (pt) ↑, 95% CI | with doc only right / closed-book only right | check |
|---|---|---|---|---|---|---|
| text ~4k tokens | 637 | .612 | .203 | +41.0 [+36.3, +45.5] | 299 / 38 | pass |
| text ~8k | 610 | .615 | .202 | +41.3 [+36.4, +46.1] | 290 / 38 | pass |
| text ~16k | 588 | .587 | .197 | +38.9 [+34.2, +43.9] | 263 / 34 | pass |
| text ~32k | 411 | .547 | .219 | +32.8 [+27.0, +38.9] | 170 / 35 | pass |
| pages 10 | 1,407 | .672 | .164 | +50.8 [+47.8, +53.8] | 772 / 57 | pass |
| pages 20 | 1,407 | .665 | .164 | +50.0 [+47.1, +53.0] | 763 / 59 | pass |
| pages 40 | 1,407 | .659 | .164 | +49.5 [+46.4, +52.5] | 758 / 62 | pass |
| mixed 10 items | 1,407 | .654 | .164 | +49.0 [+45.9, +52.0] | 751 / 62 | pass |
| mixed 20 items | 1,407 | .643 | .164 | +47.8 [+44.8, +50.7] | 732 / 59 | pass |
| mixed 40 items | 1,407 | .631 | .164 | +46.7 [+43.7, +49.7] | 723 / 66 | pass |

Reading: answers come from the document, not from the model's memory, in every bucket (lowest CI bound +27 pt). The
uncompressed reader loses little with length on one long document: pages .672 → .659 from 10 to 40 pages, mixed .654 →
.631, text .612 → .547 from ~4k to ~32k tokens. These are the uncompressed columns the R cells (running) pair with.

**Arm E soundness check 3 — can the uncompressed decoder read that far? (needle: a 6-digit access code planted at 10 / 50 / 90% depth of a real LongDocURL document; 25 documents per length × 3 depths; cap 512; jobs 191877 / 191878 / 191879)**
Scored exactly: the code string appears in the answer (`match`; no answer lists more than one code). EM is low only
because the base decoder answers in a sentence. Bar: ≥ 90% per length, else that length is dropped.

| set | length | rows | code found ↑ | judge ↑ | check |
|---|---|---|---|---|---|
| text | ~4k / 8k / 16k / 32k tokens | 75 each | 100% / 100% / 100% / 100% | 1.00 each | pass |
| pages | 10 / 20 / 40 pages | 75 each | 100% / 97.3% / 98.7% | 1.00 / .95 / .97 | pass |
| mixed | 10 / 20 / 40 items | 75 each | 100% / 100% / 98.7% | 1.00 / 1.00 / .99 | pass |

Reading: every length bucket stays in the yardstick; the uncompressed decoder reads up to 33.5k-token text, 40-page
and 68.7k-token mixed prompts. The judge gives the same verdict (≥ .88 at every depth, ≥ .95 per length), so two
scorings agree. EM fails everywhere (text ~4k .52 where the code is found 100%): it scores the whole sentence, a fault
of the check, not of the reader.

**Arm E full cells — compressed R vs uncompressed on one long document (cap 512 both, same code md5, paired per question)**
Rate 8 tokens per slot. Gap = R − uncompressed, paired bootstrap 95% CI (`scripts/paired_judge.py`); Δ_len = gap at
this size − gap at the shortest size of the set (decided only when both seeds are in).

| set, size | questions | `<MEM>` slots per question p50 | compression | R s42 judge ↑ | uncompressed judge ↑ | gap (pt) ↑, 95% CI | R only right / uncompressed only right | Δ_len vs shortest | R s43 judge ↑, gap (pt) ↑, 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| pages 10 | 1,407 | 1,198 | ~120 slots per page | .507 (192326) | .672 (192287) | −16.6 [−19.5, −13.9], McNemar p < .001 | 120 / 353 | reference | .489 (192360), −18.3 [−21.3, −15.4], 119 / 377 |
| pages 20 | 1,407 | 2,392 | ~120 slots per page | .471 (192356) | .665 (192288) | −19.4 [−22.4, −16.4], p < .001 | 121 / 394 | −2.8 [−5.6, −0.1], sign 157 / 193, p .06 (s42) | .462 (192362), −20.3 [−23.2, −17.5], 109 / 394; Δ_len s43 −1.9 [−4.8, +0.9], sign 174 / 197, p .25 |
| pages 40 | 1,407 | 4,795 | ~120 slots per page | .429 (192324) | .659 (192267) | −23.0 [−26.1, −19.9], p < .001 | 123 / 447 | −6.5 [−9.5, −3.5], sign 165 / 249, p 4e-5 (s42) | .422 (192364), −23.7 [−26.8, −20.5], 114 / 447; Δ_len s43 −5.3 [−8.4, −2.3], sign 177 / 245, p .001 |
| mixed 10 | 1,407 | 977 | ~8 tokens per slot | .449 (192357) | .654 (192289) | −20.5 [−23.3, −17.6], p < .001 | 100 / 388 | reference | .459 (192361), −19.5 [−22.3, −16.5], 110 / 384 |
| mixed 20 | 1,407 | 1,922 | ~8 tokens per slot | .435 (192358) | .643 (192290) | −20.8 [−23.8, −17.8], p < .001 | 109 / 401 | −0.3 [−3.1, +2.5], sign 174 / 176, p .96 (s42) | .436 (192363), −20.7 [−23.7, −17.7], 102 / 393; Δ_len s43 −1.2 [−4.0, +1.6], sign 164 / 184, p .31 |
| mixed 40 | 1,407 | 3,821 | ~8 tokens per slot | .412 (192325) | .631 (192268) | −21.9 [−24.7, −18.9], p < .001 | 105 / 413 | −1.4 [−4.5, +1.6], sign 206 / 223, p .44 (s42) | .407 (192365), −22.5 [−25.4, −19.6], 110 / 426; Δ_len s43 −3.0 [−5.9, 0.0], sign 186 / 235, p .02 |
| text ~4k | 637 | 540 | 8 tokens per slot | .422 (192323) | .612 (192266) | −19.0 [−22.9, −15.1] | 37 / 158 | reference | .455 (192359), −15.7 [−19.6, −11.8], 44 / 144 |
| text ~8k | 610 | 1,046 | 8 tokens per slot | .423 | .615 | −19.2 [−23.6, −14.9] | 38 / 155 | s42 +0.7 [−3.1, +4.4]; s43 −2.5 [−6.2, +1.3], n 610 | .425, −19.0 [−23.1, −15.1], 33 / 149 |
| text ~16k | 588 | 2,055 | 8 tokens per slot | .427 | .587 | −16.0 [−20.1, −11.7] | 40 / 134 | s42 +3.6 [−1.0, +8.0]; s43 −0.7 [−5.3, +3.9], n 588 | .418, −16.8 [−21.3, −12.8], 39 / 138 |
| text ~32k | 411 | 4,079 | 8 tokens per slot | .377 | .547 | −17.0 [−22.6, −11.9] | 32 / 102 | s42 +3.9 [−1.7, +9.7]; s43 −1.7 [−7.5, +4.1], n 411 | .370, −17.8 [−23.4, −12.9], 33 / 106 |
| text, whole set | 2,246 | — | 8 tokens per slot | .415 (192323) | .594 (192266) | −17.9 [−20.0, −15.8], p < .001 | 147 / 549 | — | .422 (192359), −17.3 [−19.4, −15.1], 149 / 537 |

Health of every R cell (s42 7 of 7, s43 7 of 7): every row judged, 0 unparseable, answers at the cap
1.1-3.8% (bar 5%), `<MEM>`s used = the audit's prediction exactly (pages 1,689,350 / 3,378,666 / 6,757,206; mixed
1,406,369 / 2,815,326 / 5,635,667; text 3,887,462), one attempt each, code md5 equal to the uncompressed cells'. Δ_len
pairs each question with itself at the shortest size (`scripts/longdoc/length_effect.py --strip_bucket`; text questions
exist at the longer sizes only when their document is long enough, hence n < 637). Tables:
`expQ/longdoc/length/E_dlen_*_s4{2,3}_judge.json`.
Reading (provisional): the compressed reader loses
16-24 pt to the uncompressed one at every size, on every set, and the two seeds agree within 2 pt in every cell (R s43
− R s42: pages −1.8 / −0.9 / −0.6, mixed +1.0 / +0.1 / −0.6, text +0.7 pt). Most of the gap is already there at the
shortest size (−17 to −20 pt at 10 pages, 10 items or ~4k tokens). Only page images lose more with length, and only
at 40 pages: Δ_len 40 vs 10 pages −6.5 [−9.5, −3.5] (s42) and −5.3 [−8.4, −2.3] (s43, sign 177 / 245, p .001), past
the length-loss bar at both seeds; R drops .507 / .489 → .429 / .422 while the uncompressed reader holds (.672 →
.659). 20 vs 10 pages −2.8 / −1.9 meets neither bar. Mixed documents (the same PDFs, each page as image or text
layer): 20 vs 10 items −0.3 / −1.2, 40 vs 10 −1.4 / −3.0 [−5.9, 0.0]: no bar met. Text: no bar met at either seed
(CIs wider than ±3, n 411-610 per bucket). Scorer check on pages 40 vs 10: match keeps the sign at both seeds (−4.6
[−7.1, −2.1] / −3.6 [−6.0, −1.1], within 2 pt of the judge), EM flips it (+6.0 / +7.2) because the uncompressed decoder
answers in longer sentences at 40 pages (its EM .190 → .080 while its judge .672 → .659): EM fails on the accepted
reference here, as on the needle check. By arm J's letter (sign kept under judge, project prompt AND EM) the EM flip
makes the 40-page number judge-sensitive; the owner weighs it at the final checkpoint. Project prompt (job 192405, copies
under `expQ/longdoc/rejudge/E_pages_p10_p40/`): −5.7 [−8.7, −2.6] sign 178 / 257 (s42), −6.1 [−9.2, −3.0] 183 / 271
(s43), within 1 pt of the rubric judge.

**Arm E long-chunk text — C256 vs L1024 finetunes of the same seed on `longdoc_text_1500`, CML 1,024 both, cap 512**
| bucket | questions | C256 s42 judge ↑ | L1024 s42 judge ↑ | L1024 − C256 s42 (95% CI), McNemar | C256 s43 judge ↑ | L1024 s43 judge ↑ | L1024 − C256 s43 (95% CI), McNemar | R s42 / s43 judge ↑ | uncompressed judge ↑ |
|---|---|---|---|---|---|---|---|---|---|
| ~4k tokens | 637 | .419 | .433 | +1.4 [−1.6, +4.6], 56 / 47 | .443 | .441 | −0.2 [−3.1, +2.8], 46 / 47 | .422 / .455 | .612 |
| ~8k | 610 | .385 | .418 | +3.3 [+0.3, +6.6], 57 / 37 | .426 | .413 | −1.3 [−4.6, +2.0], 48 / 56 | .423 / .425 | .615 |
| ~16k | 588 | .337 | .401 | +6.5 [+3.2, +9.7], 67 / 29 | .400 | .410 | +1.0 [−2.0, +4.4], 52 / 46 | .427 / .418 | .587 |
| ~32k | 411 | .294 | .384 | +9.0 [+5.1, +12.9], 51 / 14 | .358 | .363 | +0.5 [−3.6, +4.6], 38 / 36 | .377 / .370 | .547 |
| whole set | 2,246 | .366 (192366) | .412 (192367) | +4.6 [+3.1, +6.2], 231 / 127 | .411 (192374) | .411 (192375) | −0.0 [−1.7, +1.7], 184 / 185 | .415 / .422 | .594 |

Rate 8 tokens per slot, 540-4,079 slots per question (p50 by bucket). Health: every row judged, 0 unparseable,
`<MEM>`s used 3,887,462 = the audit's prediction in all four files; answers at the cap 4.1 / 5.08 / 3.0 / 4.6% (L1024
s42 is 2 answers over the 5% bar; on the 2,079 questions neither s42 file caps the difference is the same, +4.9).
Reading (provisional): the seeds disagree. At s42 L1024 beats C256 and the margin grows with length (+1.4 → +9.0),
but L1024 s42 equals R s42 (−0.4 [−2.0, +1.4]) and C256 s42 sits 5.0 pt below R s42 [−6.5, −3.3]: the s42 "gain" is
one weak C256 model on long text. At s43 C256 = L1024 = R within 1.1 pt, CI inside ±3. Neither bar is met at both
seeds, so long-chunk stage 1 is not shown to help or hurt on long text; with the arm T result (it hurts image pages)
the recommendation stays: keep the 256-token stage 1. The CML 256 second round opens only on a tie at both seeds: not
opened. (`text_hotpotqa_1500` showed the same seed pattern: +2.3 at s42, −0.6 at s43.)

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
| eval, arm E (job 191516): LongDocURL pages, 10 / 20 / 40 consecutive pages of one PDF | 1,407 per size | 1,198 / 2,392 / 4,795 (p50); max 4,920 | p99 5,217, max 5,356 | max 39,696 | 10,018 / 19,822 / 39,562 | 100% |
| eval, arm E: LongDocURL text layer, ~4k / 8k / 16k / 32k tokens (chunk ≤ 1,024) | 637 / 610 / 588 / 411 | 540 / 1,046 / 2,055 / 4,079 (p50); max 4,611 | p99 4,687, max 4,976 | max 37,253 | 8,010 / 10,126 / 18,911 / 36,965 | 100% |
| eval, arm E: LongDocURL mixed (each page text or image), 10 / 20 / 40 pages | 1,407 per size | 977 / 1,922 / 3,821 (p50); max 14,688 | p99 6,502, max 15,416 | max 118,232 | 36,912 / 64,536 / 117,451 | 99.4% / 100% / 100% |

Reading: the 4-page document eval sits exactly at the finetune maximum; every longer document setting, 40
text passages and every arm E bucket, puts the decoder on slot counts and positions it never saw in finetuning. This is a count. It names
a cause only if the reading or position probes show a paired loss.

Arm E input check (job 191516, every row, CML 1,024): nothing cut (0 rows) and both modes see the same pages in the
same order (parity 1.0000) on the text, pages, mixed and the three needle sets; the CML 256 control cuts 235 / 2,246
text rows (10.5%), as it should. The answer string appears verbatim in the text layer for about half the text rows
(326 / 637 at 4k up to 231 / 411 at 32k); the judge, not string match, scores these sets.

Side finding (audit only, no accuracy effect measured): stage-1 C256 decoder rows sit at their 512-token cap on the
median row.

## 4. Figure
`expQ/longdoc/figures/length_curve.svg` (`python3 scripts/longdoc/plot_length.py`; stdlib SVG, the pixi env has no
plotting library): arm E judge accuracy ↑ against `<MEM>` slots per question (median, log scale, 8 tokens per slot),
compressed R (mean of seeds 42 / 43, each seed a dot) solid, uncompressed on the same questions dashed, one colour
per modality. It shows the gap is mostly a constant offset: the uncompressed lines are flat (.67 → .66 pages, .65 →
.63 mixed, .61 → .55 text), the compressed lines sit 17-20 pt below them from the shortest size on and fall a further
5-6 pt only on page images at 40 pages.
