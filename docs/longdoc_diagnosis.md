# Long inputs: where is the bottleneck? (campaign longdoc)

Rewritten by each campaign wake when an arm decides. Last rewrite 2026-11-20 00:10 (arm E: the full uncompressed reference is in, 7 cells, 1,407-2,246 rows each; soundness check 1 passes in every bucket, +33 to +51 pt over closed-book; R seed 42 full cells started). Before that, 2026-11-19 08:30: every arm L uncompressed file ran at the 32-token answer cap and fails the 5% bar, so arm L decides nothing until re-run at 512 tokens.
Question (owner): every attempt at long inputs loses accuracy — is the cause the evaluation, the training data,
training itself, or how the decoder reads many `<MEM>` slots (one `<MEM>` = one compressed slot, 8 tokens each)?

## 1. Bottleneck table
Accuracy = Qwen3-8B rubric judge, paired on the same questions. pt = 0.01. A cause needs ≥ 3 pt with the 95% CI
on the far side of 0 at both seeds; "not this cause" needs the CI inside ±3 pt.

| candidate | probe | n | number | gate | verdict | questions repaired |
|---|---|---|---|---|---|---|
| evaluation feeds less than it claims | input audit, CPU (job 191147) | 1,500 Q per variant × 14 variants (one set of 500) | truncation cut 0 / 1,500 and pisco-vs-uncompressed parity 1,500 / 1,500 on every gating variant; the cut check does fail when it should (CML 256 controls: 924 / 1,500 and 275 / 1,500). Limits: on documents the cut check can only fail on prompt length (pages are images, never cut); the audit's parity used one row builder for both modes, so it was re-checked on recorded runs: pages per question equal in both modes on 1,500 / 1,500 (4 pages), 1,500 / 1,500 (10 pages), 500 / 500 (whole documents) — count, not page identity | cut ≤ 1%, parity 100% | provisional: not this cause (gate passed; parity only partly independent) | 0 |
| the judge decides the number (arm J) | every length number re-scored with the project (Yes/No) prompt on copies (job 191629) and with exact match | 1,500 pairs per number | Δ_len rubric / project / EM: documents 10 vs 4 pages +0.5 / +0.6 / +1.6 (s42), +0.1 / −0.3 / +0.9 (s43); 20 vs 4 pages +4.3 / +5.1 / +7.7 (s42), +3.1 / +3.1 / +6.1 (s43, project re-judge 192142); text 40 vs 10 passages −2.6 / −1.5 / +0.8 (s42), +0.1 / +0.9 / +2.7 (s43). All these files carry the 32-token cap confound (next row). Calculator control: recovers the known −.169 within ±.005, self-pair 0 | a number keeps its sign and stays within 2 pt under the three scorers | provisional: by the law's rule (same sign AND within 2 pt under all three) all three are judge-sensitive: 10 pages s43 flips sign (+0.1 / −0.3, every CI inside ±3, so the no-loss reading survives); 20 pages keeps its sign but EM is 3.4 pt larger; text 40 passages flips sign with EM | — |
| the uncompressed reference is capped (arm E) | count uncompressed answers that reach the answer cap (token count with the Qwen3.5 tokenizer, padding off), same 300 trial rows per set at 32 / 256 / 512 tokens (jobs 191634 / 191635, 191669 / 191670, 191777 / 191778) | 300 rows per set | 32 tokens: 61-74% of page answers and 47-64% of text answers cut mid-sentence (character heuristic). 256 tokens: pages 2.7% at the cap, judge .340 → .717 (+37.7 [+32.0, +43.0], 114 / 1); text 8.0% at the cap, .423 → .580 (+15.7 [+10.7, +20.7], 58 / 11). 512 tokens: text 2.3% at the cap, judge .583; mixed 2.3%, .653; closed-book 5 / 1,407; R mixed 1.3%. R barely moves (pages .517 → .547 from 32 to 256). Gating documents set at 32 tokens: 4% (SlideVQA 11%). **Arm L files (1,500 Q each, 32 tokens, answers ≥ 30 tokens):** uncompressed documents 4 / 10 / 20 pages 5.7% / 6.9% / 16.3%, text 10 / 40 passages 32.9% / 46.9%; R 0.6-1.7% on documents, 2.7-5.9% on text (`expQ/longdoc/length/cap32_sensitivity.json`) | share at the cap < 5% | provisional: YES on LongDocURL — the shared 32-token cap made the uncompressed reader look worse than the compressed one. Fixed: every arm E cell runs at 512 tokens; the full uncompressed cells at 512 have 2.1% of text answers at the cap (47 / 2,246, job 192266), pages 0.6 / 1.0 / 1.0%, mixed 1.1 / 1.8 / 2.2% (1,407 rows each), all under 5%. YES on arm L too: every uncompressed arm L file fails the bar, and more at the long size, so every arm L Δ_len is pushed upward (next row) | — |
| training never shows long inputs | input audit (count only) | as above | 10- and 20-page documents: 100% of questions beyond the finetune range; 40 text passages: 99.9%; 20 passages: 2.3% | names a cause only with a paired effect from the reading or position probes | open (count, not yet a cause) | — |
| long-chunk training (stage-1 chunk 1,024 vs 256) | gold page only / picked page raw on the C256 and L1024 finetunes; page transcription and text autoencoding on their earlier stages; ten short vs one long text chunk | 1,500 Q × 2 seeds (QA); 500 pages / 500 documents (probes) | gold page only (distractor pages removed, 1,003,520 px): L1024 − C256 −6.7 [−8.9, −4.5] McNemar 106 / 207 (s42), −4.9 [−7.1, −2.5] 113 / 186 (s43). Picked page raw: +0.1 [−0.8, +1.1] (s42), −0.1 [−1.1, +0.9] (s43); both models pick the answer page on 98.5%. Page transcription (stage 2): −8.0 pt [−10.6, −5.5] of the page's words, p 4e-16. Text autoencoding at 256 tokens: stage 1 −.0018 nats, stage 2 +.0006 [−.0008, +.0019] p .82 (L1024 not worse). Text QA: −1.1 / −3.3 on one ~1.3k-token chunk, +2.3 / −0.6 on ten short chunks | deficit ≤ −3 pt with the gold page alone AND inside ±2 pt with the picked page raw → in the compressed page; stage-2 text autoencoding worse (CI above 0) → general stage-2 loss, else image-specific | provisional: CLOSED. IN THE COMPRESSED IMAGE PAGE, entering in stage 2: L1024's page slots hold fewer words, its text slots do not. Not multi-page reading, not the page pick, not the decoder. Motivates (owner): keep the 256-token stage 1; a long-chunk stage 1 would need its stage 2 changed, checked by page transcription before any finetune | the whole 4-page deficit: ~93 (s42) / ~65 (s43) of 1,500 questions |
| length on a fixed model | 4 / 10 / 20 pages, 10 / 20 / 40 passages | 1,500 Q per size | all 1,500 Q, 32-token cap: documents 10 vs 4 pages +0.5 [−1.8, +2.7] (s42), +0.1 [−2.2, +2.5] (s43); 20 vs 4 pages +4.3 [+1.8, +6.9] (s42), +3.1 [+0.6, +5.7] sign 195 / 153 p .028 (s43); text 40 vs 10 passages −2.6 [−5.6, +0.3] (s42), +0.1 [−3.0, +3.1] (s43). Only questions where no file's answer reaches the cap (a sensitivity read: the subset is chosen on the answers): 10 pages −0.1 / −0.4 (n 1,357 / 1,354); 20 pages −0.6 [−3.3, +2.0] / −2.0 [−4.5, +0.4] (n 1,221 / 1,220); text 40 passages −4.8 [−8.3, −1.5] / −3.4 [−6.8, 0.0] (n 731 / 733). These sets add distractor pages or passages from other documents: not one long document (arm E builds that) | Δ_len ≤ −3 pt (CI below 0) at both seeds = length loss; CI inside ±3 pt at both = none; every file < 5% of answers at the cap | provisional: NOT DECIDED — the uncompressed files fail the cap bar (previous row). The earlier "gap shrinks at 20 pages" is withdrawn: it vanishes without the capped questions. Without them, text at 40 passages leans toward a length loss (−4.8 / −3.4). Deciding needs the arm L cells re-run at 512 tokens (owner question) | — |
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

| set, size | questions | `<MEM>` slots per question p50 | compression | R s42 judge ↑ | uncompressed judge ↑ | gap (pt) ↑, 95% CI | R only right / uncompressed only right | Δ_len vs shortest | R s43 |
|---|---|---|---|---|---|---|---|---|---|
| pages 10 | 1,407 | 1,198 | ~120 slots per page | .507 (192326) | .672 (192287) | −16.6 [−19.5, −13.9], McNemar p < .001 | 120 / 353 | reference | not run |
| pages 20 | 1,407 | 2,392 | ~120 slots per page | running (192356) | .665 (192288) | — | — | — | not run |
| pages 40 | 1,407 | 4,795 | ~120 slots per page | running (192324) | .659 (192267) | — | — | — | not run |
| mixed 10 / 20 / 40 | 1,407 each | 977 / 1,922 / 3,821 | ~8 tokens per slot | running (192357 / 192358 / 192325) | .654 / .643 / .631 | — | — | — | not run |
| text ~4k-32k | 2,246 | 540-4,079 | 8 tokens per slot | running (192323) | .612 / .615 / .587 / .547 (192266) | — | — | — | running (192359) |

Health of the R 10-page cell: 1,407 / 1,407 judged, 0 unparseable, 1.4% of answers at the cap, `<MEM>`s used
1,689,350 = the audit's prediction exactly, one attempt on homer-014. Reading so far: at 10 pages the compressed reader
loses 16.6 pt to the uncompressed one (300 trial rows said −13); whether the gap grows with length needs the 20- and
40-page cells.

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
`expQ/longdoc/figures/length_curve.png` — not drawn yet: drawn when arm L or arm E first decides (data so far: documents 4 / 10 / 20 pages, text 10 / 40 passages).
