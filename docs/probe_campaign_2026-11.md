# Probe campaign 2026-11 — why the compressed model loses, and what recovers it without training

Law: `campaign_probes/CAMPAIGN.md`. Live state: `campaign_probes/STATE.md`. This page is rewritten whenever an arm decides.
Metric everywhere: judge accuracy ↑ (share of questions the Qwen3-8B rubric judge accepts), paired against a baseline
cell on the same questions (95% bootstrap CI, exact McNemar p). Compression: 1 slot per 8 visual or text tokens
(`compr_rate: 8` in every model here). Document set (`mm_docqa_1500_1m`, PX=1003520, 4 pages per question): the plain
cell uses 430.7 slots per question (646,042 / 1,500). HotpotQA (10 passages per question): about 177 slots per question.

## Status (2026-11-11 07:50 KST)
- **Arm A decided at both seeds.** Showing the one page the model picks as raw pixels, the other three as slots,
  recovers the whole document gap: .820 / .822 vs .610 / .615 compressed, above the all-raw reader (.809).
  Removing the three distractor pages gains only about 2 pt; asking the decoder to copy the evidence before answering
  gains nothing (+.001, seed 42). Slots stored in int8 lose nothing at both seeds: adopted as a serving option.
- **Arm E decided at both seeds: given the question, half the slots carry everything the decoder uses.** Keeping the
  half the decoder attends to most gives the plain score (.610 = .610, .615 = .615) and beats a blind half by +.105 /
  +.107. provisional (seed 42 only): keeping a QUARTER still gives the plain score (.611 vs .610, slots .246×).
  The half also holds on photos (seed 42).
- **Arm B decided at both seeds: the long-chunk gain is context between the two answer passages.** Golds joined beats
  ten chunks by +.073 / +.067 and one chunk of all ten by +.038 / +.053. One chunk of all ten gains +.035 (seed 42,
  p = .004) but only +.014 (n.s.) at seed 43. Ranking passages by attention and joining the top 3 is no better than
  joining all ten at either seed: not adopted. Passage order inside a chunk is unused at both seeds.
- **Arm D decided:** attention picks image pages better than the digit prompt (+2.5 pt), the digit prompt picks text
  passages better (+4.9 pt); the end-to-end judge effect of either is too small to measure.
- **Arm C (evidence-then-answer on text) did not close the text gap** at seed 42.
- Running: arm G re-check of the patched "picked page at twice the slots" flag (job 188580). Arm F (TextVQA as text):
  set built, no cell (cell cap).

## Findings (ranked)
1. **The document loss is reading the compressed answer page — route-then-expand removes it.** Picked page raw:
   +.210 [+.187, +.233] (seed 42) and +.207 [+.183, +.232] (seed 43), n = 1,500 each, McNemar p < 1e-60; the pick is
   right on 98.6% / 98.9% of questions. Gold page alone, compressed: +.022 / +.025. Evidence-then-answer: +.001.
   Motivates: serve with route-then-expand (decoder prompt 1,264 tokens vs 511 compressed), and put training effort
   into how well one page's slots can be read (stage 1/2), not into distractor robustness or answer-format labels.
   Arm G tests the cheaper version (the picked page at twice the slots instead of raw).
2. **Given the question, the decoder uses at most half of the slots — maybe a quarter.** Keep the most-attended half
   over all pages, documents, n = 1,500 each, slots .496×: seed 42 .610 vs .610, +.000 [−.009, +.010], 27 / 27; seed 43
   .615 vs .615, +.000 [−.009, +.009], 25 / 25. Blind halves at the same slot count: every 2nd slot .505 / .508, so the
   attention half is +.105 [+.085, +.126] (p = 1e-24) and +.107 [+.087, +.127] (p = 5e-26) better. provisional, one
   seed: the most-attended QUARTER (slots .246×, 106 per question instead of 431) gives .611 vs .610, +.001 [−.009,
   +.012], 37 / 35, 1,197 / 1,500 answers identical. Photos (one seed): the half gives .677 = .677. The selection
   needs one full pass over every slot with the question, so it saves decoder context, not storage. Motivates: a
   query-aware slot budget, or nested-budget training so the needed quarter can be found without the full pass.
3. **The long-chunk gain is context between the passages that answer together.** Golds joined (9 chunks) vs ten
   chunks: +.073 [+.051, +.095] (seed 42, 183814) and +.067 [+.046, +.088] (seed 43, 183816), n = 1,500 each, same
   slots (.998×), all on bridge questions. Against one chunk of all ten: +.038 [+.015, +.061] (seed 42) and +.053
   [+.029, +.076] (seed 43). The tail-layout model (183538) shows the same gain, +.058 [+.035, +.082] (one model,
   provisional), so the layout is not what makes it. Joining all ten: +.035 [+.012, +.059] (seed 42), +.014 [−.008,
   +.035] (seed 43, n.s.). Without an oracle, joining the 3 passages the model's attention ranks first equals joining
   all ten (−.003 seed 42, −.001 seed 43): +.032 over ten chunks at seed 42, +.013 (n.s.) at seed 43, so the "rank, then
   re-compress the top passages together" serving mode is NOT adopted. Motivates: a long-chunk stage 1 that packs
   RELATED passages together (the sibling `longchunk` campaign); a ranker that finds both answer passages (both in the
   attention top 3 on .667 of questions) is the missing piece.
4. **Storing slots in int8 costs nothing, at both seeds.** Seed 42 +.001 [−.003, +.006], 1,445 / 1,500 answers
   identical; seed 43 −.001 [−.005, +.003], 1,444 / 1,500 identical. A serving option (half the memory of bf16 slots).
5. **Giving the decoder room to reason or copy does not close the gap.** Evidence-then-answer on MRQA text: compressed
   .699 vs .718 plain; raw .807 vs .807. The gap on DROP shrinks from −.204 to −.172 (below the 5 pt gate) and grows on
   DuoRC (−.084 → −.156) and RACE (−.152 → −.188); BioASQ drops .912 → .812 (the slots do not hold exact strings well).
   On documents (seed 42): .611 vs .610, +.001 [−.017, +.021], 108 / 106. Motivates: stage-1/2 objectives for exact
   content (the transcription side), not reasoning labels. Closes evidence-then-answer labels.
6. **Which page picker to use depends on the modality.** Attention mass: +.025 pick@1 on 10-page documents, −.049 on
   HotpotQA. A per-source switch passes the arm D gate, but it was chosen on the same questions it is scored on.
   Product fusion never loses (+.019 / +.006). No judge cell: the expected judge gain is +.025 × the hit-minus-miss
   judge gap of the picked-page-raw cells (.252 at seed 42: 1,479 hits .824 vs 21 misses .571; .325 at seed 43 on
   16 misses) ≈ +.006 to +.008, below the .02 noise band.

## Arm A — use-recovery on documents (reference models, `mm_docqa_1500_1m`, PX=1003520, 1,500 questions)
Baselines: plain compressed cells of the same model (seed 42: job 184781, homer-011; seed 43: job 184784, homer-016).
Compression column: slots per question (1 slot per 8 visual tokens) and the decoder prompt length.
| cell | model | n | judge accuracy ↑ | baseline judge accuracy ↑ | difference (95% CI) | McNemar p (up / down) | health (unparseable, rows, slots vs baseline) | compression: slots / question, decoder prompt tokens |
|---|---|---|---|---|---|---|---|---|
| picked page raw (job 188441, homer-016) | 184778 s42 | 1,500 | **.820** | .610 | **+.210 [+.187, +.233]** | <1e-60 (350 / 35) | 0, 1,500, 0.749 + 1 raw page | 322.4 slots + 1 raw page, 1,264 |
| picked page raw (job 188460, homer-020) | 184780 s43 | 1,500 | **.822** | .615 | **+.207 [+.183, +.232]** | <1e-60 (349 / 38) | 0, 1,500, 0.748 + 1 raw page | 322.4 + 1 raw page, 1,264 |
| gold page only (job 188440, homer-011) | 184778 s42 | 1,500 | .632 | .610 | +.022 [+.005, +.040] | .020 (111 / 78) | 0, 1,500, 0.270 | 116.4, 185 |
| gold page only (job 188459, homer-013) | 184780 s43 | 1,500 | .639 | .615 | +.025 [+.006, +.043] | .010 (116 / 79) | 0, 1,500, 0.270 | 116.4, 185 |
| evidence, then answer (job 188439, homer-011) | 184778 s42 | 1,500 | .611 | .610 | +.001 [−.017, +.021] | .95 (108 / 106) | 0, 1,500, 1.000 | 430.7, 544 (+ a longer answer) |
| slots in int8 (job 188478, homer-016) | 184778 s42 | 1,500 | .611 | .610 | +.001 [−.003, +.006] | .77 (7 / 5) | 0, 1,500, 1.000 | 430.7, 511 |
| slots in int8 (job 188536, homer-011) | 184780 s43 | 1,500 | .613 | .615 | −.001 [−.005, +.003] | .75 (4 / 6) | 0, 1,500, 1.000 | 430.7, 511 |

By source, picked page raw vs plain (seed 42): infographics .716 vs .480, multi-page .922 vs .666, slides .822 vs .684.
Evidence-then-answer by source (seed 42): infographics .480 = .480, multi-page .656 vs .666, slides .698 vs .684.
Context (older cells, not paired on node): uncompressed reader, all 4 pages raw .809; gold page only raw .841. The
distractor pages cost about 2-3 pt whether they are raw or compressed. Evidence-then-answer was not repeated at seed
43 (cell cap); a tie at one seed is not a result under the law, so it reads "no gain seen", not "proven useless".

## Arm B — long-chunk mechanism (HotpotQA, 10 passages, 1,500 questions; 183814 = seed 42, 183816 = seed 43)
Baselines, all on today's code: g1 = each passage its own chunk (seed 42 job 188486 .582; seed 43 job 188509 .601);
g10 = all 10 passages in one chunk (seed 42 job 188518 .617; seed 43 job 188510 .615). Tail-layout model 183538: ten
chunks job 188519 .569. Compression column: chunks per question, slots per question (1 per 8 text tokens), decoder
prompt tokens.
| cell | seed | n | judge accuracy ↑ | baseline judge accuracy ↑ | difference (95% CI) | McNemar p (up / down) | health (unparseable, rows, slots vs baseline) | compression: chunks, slots / question, decoder prompt tokens |
|---|---|---|---|---|---|---|---|---|
| golds joined, distractors alone (job 188471, homer-016) | 42 | 1,500 | **.655** | g1 .582 | **+.073 [+.051, +.095]** | 1.5e-10 (200 / 91) | 0, 1,500, 0.998 | 9, 176.8 |
| golds joined, distractors alone (job 188511, homer-016) | 43 | 1,500 | **.669** | g1 .601 | **+.067 [+.046, +.088]** | 8e-10 (186 / 85) | 0, 1,500, 0.998 | 9, 176.8, 282 |
| same cell | 42 | 1,500 | .655 | g10 .617 | **+.038 [+.015, +.061]** | .002 (185 / 128) | 0, 1,500, 1.020 | |
| same cell | 43 | 1,500 | .669 | g10 .615 | **+.053 [+.029, +.076]** | 9e-6 (200 / 120) | 0, 1,500, 1.020 | |
| golds joined, TAIL-layout model 183538 (job 188514, homer-012) | 42 | 1,500 | **.627** | g1 tail .569 | **+.058 [+.035, +.082]** | <.001 (202 / 115) | 0, 1,500, 0.998 | 9, 176.8 |
| one chunk of ten (job 188518, homer-010) | 42 | 1,500 | .617 | g1 .582 | +.035 [+.012, +.059] | .004 (184 / 132) | 0, 1,500, 0.978 | 1, 173.3, 246 |
| one chunk of ten (job 188510, homer-016) | 43 | 1,500 | .615 | g1 .601 | +.014 [−.008, +.035] | .23 (152 / 131) | 0, 1,500, 0.978 | 1, 173.3, 246 |
| no oracle: top 3 by attention joined (job 188521, homer-020) | 42 | 1,500 | .614 | g1 .582 | +.032 [+.009, +.056] | .007 (175 / 127) | 0, 1,500, 0.995 | 8, 176.4 |
| no oracle: top 3 by attention joined (job 188546, homer-012) | 43 | 1,500 | .615 | g1 .601 | +.013 [−.007, +.034] | .24 (139 / 119) | 0, 1,500, 0.995 | 8, 176.4, 277 |
| same cell | 42 | 1,500 | .614 | g10 .617 | −.003 [−.026, +.021] | .87 (156 / 160) | | |
| same cell | 43 | 1,500 | .615 | g10 .615 | −.001 [−.022, +.023] | 1.0 (148 / 149) | | |
| no oracle: top 2 by attention joined (job 188520, homer-010) | 42 | 1,500 | .612 | g1 .582 | +.030 [+.008, +.051] | .007 (154 / 109) | 0, 1,500, 0.998 | 9, 176.8 |
| distractors joined, golds alone (job 188472, homer-020) | 42 | 1,500 | .604 | g1 .582 | +.022 [+.003, +.041] | .027 (122 / 89) | 0, 1,500, 0.983 | 3, 174.2 |
| same cell | 42 | 1,500 | .604 | g10 .617 | −.013 [−.037, +.011] | .33 (158 / 177) | 0, 1,500, 1.005 | |
| distractors joined, golds alone (job 188512, homer-016) | 43 | 1,500 | .617 | g1 .601 | +.016 [−.003, +.035] | .11 (113 / 89) | 0, 1,500, 0.983 | 3, 174.2 |
| all 10 joined, shuffled order (job 188473, homer-011) | 42 | 1,500 | .615 | g10 .617 | −.002 [−.023, +.020] | .91 (141 / 144) | 0, 1,500, 1.000 | 1, 173.3 |
| all 10 joined, shuffled order (job 188513, homer-016) | 43 | 1,500 | .618 | g10 .615 | +.003 [−.021, +.026] | .87 (160 / 156) | 0, 1,500, 1.000 | 1, 173.3 |

By type, golds joined vs g1: bridge (n = 1,197) .647 vs .553 (seed 42), .668 vs .572 (seed 43); comparison (n = 303)
.686 vs .696 (seed 42), .673 vs .716 (seed 43, −.043 [−.079, −.007]). One chunk of ten vs g1, seed 42: bridge +.043,
comparison .000. Reading: the gain needs the two gold passages in the same chunk, at both seeds, and the tail-layout
model shows it too (one model). Joining all ten helps at seed 42 only. The no-oracle ranking (model 183814's attention
over the ten passages; both golds in its top 2 on .495 of questions, top 3 on .667) adds nothing over joining all ten;
the seed-43 cell reuses 183814's ranking on 183816, so it tests a fixed ranker. Passage order: no effect at either seed.

## Arm C — reasoning vs reading on text (MRQA, 6 sources × 250, one passage each, nothing cut)
Baseline note: the law's ".695" is model 183814 with 256-token chunks (job 185775); this arm uses the reference model
184778 with one chunk per passage (`CML=4096`, job 188445: .718).
| source | n | compressed plain ↑ (184778, job 188445) | uncompressed plain ↑ (185542) | gap, plain | compressed evidence-then-answer ↑ (job 188446) | uncompressed evidence-then-answer ↑ (job 188457) | gap, evidence-then-answer |
|---|---|---|---|---|---|---|---|
| all | 1,500 | .718 | .807 | −.089 | .699 | .807 | −.107 |
| DROP (reasoning) | 250 | .580 | .784 | −.204 | .668 | .840 | −.172 |
| DuoRC (narrative) | 250 | .600 | .684 | −.084 | .544 | .700 | −.156 |
| RACE (narrative) | 250 | .528 | .680 | −.152 | .500 | .688 | −.188 |
| BioASQ | 250 | .912 | .924 | −.012 | .812 | .904 | −.092 |
| RelationExtraction | 250 | .840 | .896 | −.056 | .852 | .876 | −.024 |
| TextbookQA | 250 | .848 | .872 | −.024 | .820 | .832 | −.012 |
Paired, evidence-then-answer minus plain: compressed −.019 [−.041, +.003], McNemar p = .11 (130 up / 158 down);
uncompressed .000 [−.017, +.016] (89 / 89). Health: unparseable 0, 1,500 rows each, compressed slots 80,003 both
(53.3 per question). The compressed plain cell re-ran on the same node with 1,500 / 1,500 identical answers. Gate: the
compressed − raw gap on DROP / DuoRC / RACE shrinks by ≥ 5 pt with the same treatment on both sides. NOT met (DROP
+3 pt, DuoRC −7 pt, RACE −4 pt). Thinking mode (the arm's second treatment) not run (cell cap).

## Arm D — page picking without training (model 183814, saved attention masses, CPU fusion, job 188515)
| picker | 10-page documents pick@1 ↑ (n = 1,500) | vs digit (95% CI), McNemar p | HotpotQA pick@1 ↑ (n = 1,500) | vs digit (95% CI), McNemar p |
|---|---|---|---|---|
| digit route prompt (baseline) | .861 | — | .922 | — |
| attention mass | .885 | +.025 [+.010, +.039], .001 | .873 | −.049 [−.066, −.032], 6e-9 |
| product of the two | .879 | +.019 [+.011, +.027], 8e-7 (31 / 3) | .928 | +.006 [−.003, +.015], .23 |
| reciprocal-rank fusion | .881 | +.020 [+.009, +.031], 6e-4 | .892 | −.030 [−.044, −.017], 2e-5 |
| per-source switch (chosen on these same questions) | .885 | +.025 [+.010, +.039], .001 | .922 | +.000 |
Gate (≥ +2 pt on documents AND no loss on HotpotQA) met by the per-source switch only, which was chosen on these same
questions (no held-out split). Closed without a judge cell: see finding 6.

## Arm E — attention-guided slot pruning (documents; photos for one cell)
How a cell works: one extra pass over the full prompt (all slots + the question); the attention of the last prompt
position on each slot, averaged over the heads of every full-attention layer; keep the top fraction (per chunk, or
over all pages), drop the rest and their placeholders, then answer. Controls passed: keep-all through the attention
pass gives 300 / 300 plain answers (job 188480); the patched eval's plain path gives 300 / 300 plain answers, EM .477 =
recorded (job 188481). All cells on el9_8 nodes, as the baselines (184781 homer-011, 184784 homer-016, 184783 homer-011).
| cell | n | judge accuracy ↑ | baseline judge accuracy ↑ (plain) | difference (95% CI) | McNemar p (up / down) | health (unparseable, rows, slots vs baseline) | compression: slots / question, decoder prompt tokens |
|---|---|---|---|---|---|---|---|
| seed 42 plain (baseline, job 184781) | 1,500 | .610 | — | — | — | 0, 1,500, 1.000 | 430.7, 511 |
| s42 keep the most-attended half overall (job 188485, homer-016) | 1,500 | **.610** | .610 | **+.000 [−.009, +.010]** | 1.0 (27 / 27) | 0, 1,500, 0.496 | 213.8, 294 |
| s42 keep the most-attended QUARTER overall (job 188538, homer-013) | 1,500 | **.611** | .610 | **+.001 [−.009, +.012]** | .91 (37 / 35) | 0, 1,500, 0.246 | 106.0, 186 |
| s42 keep the most-attended half of each chunk (job 188484, homer-016) | 1,500 | .606 | .610 | −.004 [−.015, +.007] | .57 (35 / 41) | 0, 1,500, 0.496 | 213.8, 294 |
| s42 keep every 2nd slot (job 188483, homer-016) | 1,500 | .505 | .610 | −.105 [−.127, −.085] | 3e-24 (49 / 207) | 0, 1,500, 0.504 | 216.9, 297 |
| s42 keep the first half of each chunk (job 188482, homer-013) | 1,500 | .495 | .610 | −.115 [−.136, −.095] | 4e-28 (46 / 219) | 0, 1,500, 0.496 | 213.8, 294 |
| s43 keep the most-attended half overall (job 188522, homer-012) | 1,500 | **.615** | .615 | **+.000 [−.009, +.009]** | 1.0 (25 / 25) | 0, 1,500, 0.496 | 213.8, 294 |
| s43 keep every 2nd slot (job 188537, homer-012) | 1,500 | .508 | .615 | −.107 [−.127, −.087] | 8e-26 (44 / 204) | 0, 1,500, 0.504 | 216.9, 297 |
| photos s42, keep the most-attended half overall (job 188539, homer-016) | 1,500 | .677 | .677 | +.000 [−.013, +.013] | 1.0 (49 / 49) | 0, 1,500, 0.492 | 26.0 (plain 52.9), 97 (plain 124) |
Answers identical to plain: s42 half overall 1,289 / 1,500, quarter 1,197, per chunk 1,230, every 2nd 917, first half
924; s43 half overall 1,306, every 2nd 903; photos half 1,186. By source, quarter vs plain (s42): infographics .474 vs
.480, multi-page .680 vs .666, slides .680 vs .684.
Gate (attention-picked minus the better blind half ≥ +4 pt, CI above 0, both seeds): seed 42 +.105 [+.085, +.126],
205 / 47, p = 1e-24 (188485 − 188483); seed 43 +.107 [+.087, +.127], 203 / 43, p = 5e-26 (188522 − 188537). **Met:
adopted.** Quarter (gate written before the cell: within .02 of plain → three quarters unused per question): met at
seed 42; the seed-43 cell is not run (cell cap), so it stays provisional.

## Arm G — route then densify (accepted by the owner 2026-11-11)
The page the model picks (same pick as "picked page raw") is re-compressed at rate 4 (twice its slots) and stays
compressed; the other pages keep rate 8. Code: `RAW=route1 DENSE=<rate>` in `eval_mm_4b.sh` (`--dense_rate` in
`scripts/eval_mm_docqa.py`, default path unchanged). Invariant tests PASS (job 188523: positions 12 / 12, collators
33 / 33, freeze 5 / 5); plain 300-question regression PASS (job 188524: 300 / 300 answers identical, EM .477 =
recorded). The flag at rate 8 on 300 questions FAILED its check (job 188525): 233 / 300 answers identical to plain
(needed ≥ 97%), same slot total (118,147), EM .487 vs .477, pick right on .983. Suspected cause: the flag re-compressed
the picked page alone, the plain path compresses the four pages in one padded batch. The two rate-4 cells were
cancelled before they started. Patch (9 lines): the page is re-compressed inside its example's batch, so at rate 8
its slots should be the plain slots exactly. Re-check running (job 188580, `scripts/probes/g_checks.sbatch`: tests,
the 300-question regression, the rate-8 control, same criteria); the rate-4 cells follow only if it passes. Gate:
gain over plain ≥ +.14 (2/3 of picked-page-raw's +.21) with CI above 0, at ≤ 1/3 of its extra decoder tokens, both seeds.

## Arm F — photo text vs pixels (TextVQA, 500 questions)
`eval_data/probes_textvqa_ocr` is built (job 188545, `scripts/probes/build_textvqa_ocr.py`): the 500 TextVQA rows of
the photo set, unchanged, with the OCR tokens of the same TextVQA validation row as text; 500 / 500 questions matched.
OCR text: 13 tokens per row on average (median 7, 14 rows empty); a gold answer appears word for word in it on .596 of
rows. The text-only cell (seed 42, paired with the TextVQA rows of the pixel cell, n = 500) is not run: cell cap. The
seed-43 photo baseline (job 184785) ran on homer-008, the node that changes image answers, so only seed 42 pairs cleanly.

## The instrument: same answers on re-run?
| check | model | set | n | node of the re-run (image) | node of the recorded run | answers identical | judge accuracy ↑, re-run vs recorded |
|---|---|---|---|---|---|---|---|
| gate re-run (job 188419) | 184778 compressed | documents | 1,500 | homer-011 (el9_8) | homer-011 | 1,500 / 1,500 | .610 vs .610 |
| same image (job 188476) | 184778 compressed | documents, first 300 | 300 | homer-016 (el9_8) | homer-011 | 300 / 300 | .610 vs .610 |
| older image (job 188477) | 184778 compressed | documents, first 300 | 300 | homer-007 (el9_7) | homer-011 | 300 / 300 | .610 vs .610 |
| homer-008 (job 188458) | 184778 compressed | documents, first 300 | 300 | homer-008 (el9_7) | homer-011 | 243 / 300 | .623 vs .610 (9 up / 5 down, p ≈ .42) |
| compressed text, code date (job 188518) | 183814 | HotpotQA one chunk of ten | 1,500 | homer-010 (el9_8), code of 11-11 | homer-017 (el9_8), code of 11-07 | 1,500 / 1,500 | .617 vs .617 |
| compressed text (job 188486) | 183814 | HotpotQA ten chunks | 1,500 | homer-010 (el9_8), code of 11-11 | homer-006 (el9_7), code of 11-07 | 1,445 / 1,500 | .582 vs .579 (10 / 6) |
| compressed text, same node (job 188445) | 184778 | MRQA | 1,500 | homer-016 | homer-016 | 1,500 / 1,500 | .718 vs .718 |
| compressed text, older image (job 188471 rerun) | 183814 | HotpotQA golds joined | 1,500 | homer-007 (el9_7) | homer-016 (el9_8) | 1,500 / 1,500 | .655 vs .655 |
All nodes carry A100-SXM4-80GB GPUs. Reading: homer-008 alone moves image answers (homer-007, same older image, agrees
on images and on text). The code date does not move compressed text answers (188518 vs 185601: 1,500 / 1,500), so the
55 changed answers of 185505 point to homer-006 (untested, excluded from launches).

## New eval sets and options
- `eval_data/probes_hotpot_{gold_joined,distr_joined,g10_shuf}`, 1,500 rows each (job 188465); the builder first
  rebuilds `text_hotpotqa_1500_g10` and matches it on 1,500 / 1,500 rows.
- `eval_data/probes_hotpot_pred{2_digit,2_attn,2_product,2_rrf,3_attn,3_product}`, 1,500 rows each (job 188516): no
  oracle, the K passages model 183814 ranks first joined, others alone. Both golds in the top 2: digit .404,
  attention .495, product .487; top 3: attention .667.
- `eval_data/probes_textvqa_ocr`, 500 rows (job 188545), see arm F.
- `scripts/eval_mm_docqa.py --mem_keep_mode attn | attn_global` (env `KEEPF=0.5 KEEPM=attn`): keeps the slots the
  decoder attends to most, same counts as `first`. Invariant tests 45 / 45 (job 188479).
- `scripts/eval_mm_docqa.py --dense_rate R` (env `RAW=route1 DENSE=R`): the picked page re-compressed at rate R (arm G),
  inside its example's batch since 2026-11-11 06:48 (before: alone, which moved answers; job 188525).

## Closed
- Gold page only (both seeds): +.022 / +.025, below the +3 pt gate. Distractor pages are not the main loss.
- Evidence-then-answer: documents +.001 (seed 42), text −.019 (seed 42); no gain seen (finding 5).
- Passage order inside a long chunk: −.002 (seed 42), +.003 (seed 43). Unused.
- Joining the distractor passages: +.022 (p = .027) at seed 42, +.016 (n.s.) at seed 43 over ten chunks; not adopted.
- "Rank by attention, then join the top 3" as a serving mode: equals joining all ten at both seeds; +.013 (n.s.) over
  ten chunks at seed 43.
- A judge cell for a fused page picker (expected gain below the noise band).
- Blind slot halving on the reference models: −.105 / −.115 (seed 42), −.107 (seed 43).
