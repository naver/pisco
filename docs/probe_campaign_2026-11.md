# Probe campaign 2026-11 — why the compressed model loses, and what recovers it without training

Law: `campaign_probes/CAMPAIGN.md`. Live state: `campaign_probes/STATE.md`. This page is rewritten whenever an arm decides.
Metric everywhere: judge accuracy ↑ (share of questions the Qwen3-8B rubric judge accepts), paired against a baseline
cell on the same questions (95% bootstrap CI, exact McNemar p). Compression: 1 slot per 8 visual or text tokens
(`compr_rate: 8` in every model here). Document set (`mm_docqa_1500_1m`, PX=1003520, 4 pages per question): the plain
cell uses 430.7 slots per question (646,042 / 1,500). HotpotQA (10 passages per question): about 177 slots per question.

## Status (2026-11-13 18:40 KST)
- **The campaign is finished: 66 of 66 cells used, every arm decided or closed, nothing queued.** The owner gave the
  last two cells (2026-11-13) to finish the slot-pruning question; both ran and passed health (arm E below).
- **Arm A decided at both seeds, every probe.** Showing the one page the model picks as raw pixels, the other three as
  slots, recovers the whole document gap: .820 / .822 vs .610 / .615 compressed, above the all-raw reader (.809).
  Removing the three distractor pages gains only about 2 pt; asking the decoder to copy the evidence before answering
  gains nothing at either seed (+.001 / −.009). Slots stored in int8 lose nothing at both seeds: adopted as a serving
  option.
- **Arm G decided at both seeds: twice the slots on the picked page is NOT enough.** Re-compressing the picked page at
  rate 4 (other pages rate 8) gains +.063 / +.049, below the +.14 gate (route1, the page raw, gains +.210 / +.207). It
  costs only 1/7 of route1's extra decoder tokens. The models were trained at rate 8 only.
- **Arm E decided at both seeds: given the question, an eighth of the slots carry what the decoder uses.**
  Keeping the half the decoder attends to most gives the plain score (.610 = .610, .615 = .615) and beats a blind half
  by +.105 / +.107. The most-attended QUARTER gives the plain score at both seeds (.611 vs .610, .619 vs .615, slots
  .246×), and so does the most-attended EIGHTH (.613 vs .610, .605 vs .615, slots .121×, 52 slots per question
  instead of 431). Provisional (seed 42 only): a SIXTEENTH (25 slots per question) gives .596 vs .610, −.014, still
  inside the .02 noise band. Losses grow as the share shrinks, but no level tested costs ≥ .05. The half also holds
  on photos (seed 42).
- **Arm B decided at both seeds: the long-chunk gain is context between the two answer passages.** Golds joined beats
  ten chunks by +.073 / +.067 and one chunk of all ten by +.038 / +.053. One chunk of all ten gains +.035 (seed 42,
  p = .004) but only +.014 (n.s.) at seed 43. Ranking passages by attention and joining the top 3 is no better than
  joining all ten at either seed: not adopted. Passage order inside a chunk is unused at both seeds.
- **Arm D decided:** attention picks image pages better than the digit prompt (+2.5 pt), the digit prompt picks text
  passages better (+4.9 pt); the end-to-end judge effect of either is too small to measure.
- **Arm C (evidence-then-answer on text) did not close the text gap** at seed 42.
- **Arm F decided (seed 42, reported only): reading a photo's text as OCR words is far worse than reading its
  pixels.** TextVQA, 500 questions: OCR text .346 vs pixels .780, −.434. No mixed-input arm is motivated.

## Findings (ranked)
1. **The document loss is reading the compressed answer page — route-then-expand removes it.** Picked page raw:
   +.210 [+.187, +.233] (seed 42) and +.207 [+.183, +.232] (seed 43), n = 1,500 each, McNemar p < 1e-60; the pick is
   right on 98.6% / 98.9% of questions. Gold page alone, compressed: +.022 / +.025. Evidence-then-answer: +.001 / −.009.
   Motivates: serve with route-then-expand (decoder prompt 1,264 tokens vs 511 compressed), and put training effort
   into how well one page's slots can be read (stage 1/2), not into distractor robustness or answer-format labels.
   The cheaper version, the picked page at twice the slots instead of raw, recovers only a quarter to a third of it
   (finding 2).
2. **Twice the slots on the picked page recovers only a quarter to a third of the raw-page gain (untrained rate).**
   Picked page re-compressed at rate 4, other pages rate 8, documents, n = 1,500 each: seed 42 .673 vs plain .610,
   +.063 [+.043, +.083], 158 / 64, p < .001; seed 43 .663 vs .615, +.049 [+.030, +.067], 147 / 74, p < .001. Gate
   (≥ +.14, 2/3 of the raw page's gain) failed at both seeds. Cost: +107.5 decoder prompt tokens per question vs
   +753 for the raw page (.143×), so per extra token it recovers about twice as much as the raw page (.06 vs .028
   per 100 tokens; consistent with, not a gate). The models never saw rate 4 in training. Motivates (for the
   `longchunk` campaign): a stage-2 arm that trains a mixed rate (picked page at rate 4); closes "densify instead of
   expand" as a training-free serving mode.
3. **Given the question, the decoder uses at most an eighth of the slots, at both seeds (a sixteenth at seed 42).**
   Keep the most-attended slots over all pages, documents, n = 1,500 per cell. Half (slots .496×): seed 42 .610 vs
   .610, +.000 [−.009, +.010], 27 / 27; seed 43 .615 vs .615, +.000 [−.009, +.009], 25 / 25. Blind halves at the same
   slot count: every 2nd slot .505 / .508, so the attention half is +.105 [+.085, +.126] (p = 1e-24) and +.107
   [+.087, +.127] (p = 5e-26) better. QUARTER (slots .246×, 106 per question instead of 431): seed 42 +.001 [−.009,
   +.012], 37 / 35; seed 43 +.005 [−.007, +.017], 45 / 38, p = .51. EIGHTH (slots .121×, 52 per question, decoder
   prompt 133 tokens vs 511): seed 42 .613 vs .610, +.003 [−.011, +.017], 59 / 55, p = .78; seed 43 .605 vs .615,
   −.010 [−.025, +.005], 54 / 69, p = .21. Provisional, seed 42 only: SIXTEENTH (slots .059×, 25 per question, decoder
   prompt 106 tokens) .596 vs .610, −.014 [−.030, +.002], 70 / 91, p = .115. Each halving below a quarter costs a
   little more (eighth vs quarter at seed 43 −.015, p = .03; sixteenth vs eighth at seed 42 −.017, p = .03), but
   never more than the .02 band; where the loss reaches .05 is not found. Photos (one seed): the half gives .677 =
   .677. The selection needs one full pass over every slot with the question, so it saves decoder context, not
   storage. Motivates: a query-aware slot budget, or nested-budget training so the needed eighth can be found
   without the full pass.
4. **The long-chunk gain is context between the passages that answer together.** Golds joined (9 chunks) vs ten
   chunks: +.073 [+.051, +.095] (seed 42, 183814) and +.067 [+.046, +.088] (seed 43, 183816), n = 1,500 each, same
   slots (.998×), all on bridge questions. Against one chunk of all ten: +.038 [+.015, +.061] (seed 42) and +.053
   [+.029, +.076] (seed 43). The tail-layout model (183538) shows the same gain, +.058 [+.035, +.082] (one model,
   provisional), so the layout is not what makes it. Joining all ten: +.035 [+.012, +.059] (seed 42), +.014 [−.008,
   +.035] (seed 43, n.s.). Without an oracle, joining the 3 passages the model's attention ranks first equals joining
   all ten (−.003 seed 42, −.001 seed 43): +.032 over ten chunks at seed 42, +.013 (n.s.) at seed 43, so the "rank, then
   re-compress the top passages together" serving mode is NOT adopted. Motivates: a long-chunk stage 1 that packs
   RELATED passages together (the sibling `longchunk` campaign); a ranker that finds both answer passages (both in the
   attention top 3 on .667 of questions) is the missing piece.
5. **Storing slots in int8 costs nothing, at both seeds.** Seed 42 +.001 [−.003, +.006], 1,445 / 1,500 answers
   identical; seed 43 −.001 [−.005, +.003], 1,444 / 1,500 identical. A serving option (half the memory of bf16 slots).
6. **Giving the decoder room to reason or copy does not close the gap.** Evidence-then-answer on MRQA text: compressed
   .699 vs .718 plain; raw .807 vs .807. The gap on DROP shrinks from −.204 to −.172 (below the 5 pt gate) and grows on
   DuoRC (−.084 → −.156) and RACE (−.152 → −.188); BioASQ drops .912 → .812 (the slots do not hold exact strings well).
   On documents: seed 42 .611 vs .610, +.001 [−.017, +.021], 108 / 106; seed 43 .606 vs .615, −.009 [−.027, +.011],
   98 / 111, p = .41 (infographics −.036 [−.070, −.002]). Motivates: stage-1/2 objectives for exact
   content (the transcription side), not reasoning labels. Closes evidence-then-answer labels.
7. **Which page picker to use depends on the modality.** Attention mass: +.025 pick@1 on 10-page documents, −.049 on
   HotpotQA. A per-source switch passes the arm D gate, but it was chosen on the same questions it is scored on.
   Product fusion never loses (+.019 / +.006). No judge cell: the expected judge gain is +.025 × the hit-minus-miss
   judge gap of the picked-page-raw cells (.252 at seed 42: 1,479 hits .824 vs 21 misses .571; .325 at seed 43 on
   16 misses) ≈ +.006 to +.008, below the .02 noise band.
8. **The photo-text loss is not an OCR-reading loss: giving the decoder the OCR words instead of the pixels is far
   worse.** TextVQA, model 184778 (seed 42), 500 questions paired on question id with the pixel cell: OCR text .346 vs
   pixels .780, −.434 [−.480, −.384], 17 up / 234 down, McNemar p = 6e-50; slots .081× (4.4 vs 54.2 per question).
   A gold answer appears word for word in the OCR on only .596 of rows, below what the pixels already score. Closes
   "text instead of pixels" for photo text; no mixed-input arm is motivated (gate: a gain ≥ 5 pt).

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
| evidence, then answer (job 189058, homer-011) | 184780 s43 | 1,500 | .606 | .615 | −.009 [−.027, +.011] | .41 (98 / 111) | 0, 1,500, 1.000 | 430.7, 544 (+ a longer answer) |
| slots in int8 (job 188478, homer-016) | 184778 s42 | 1,500 | .611 | .610 | +.001 [−.003, +.006] | .77 (7 / 5) | 0, 1,500, 1.000 | 430.7, 511 |
| slots in int8 (job 188536, homer-011) | 184780 s43 | 1,500 | .613 | .615 | −.001 [−.005, +.003] | .75 (4 / 6) | 0, 1,500, 1.000 | 430.7, 511 |

By source, picked page raw vs plain (seed 42): infographics .716 vs .480, multi-page .922 vs .666, slides .822 vs .684.
Evidence-then-answer by source: seed 42 infographics .480 = .480, multi-page .656 vs .666, slides .698 vs .684; seed 43
infographics .444 vs .480 (−.036 [−.070, −.002], p = .05), multi-page .668 vs .678, slides .706 vs .686. Answers
identical to plain at seed 43: 889 / 1,500 (the answer format changes). Gate (≥ +3 pt, CI above 0, both seeds): failed
at both seeds, closed. Context (older cells, not paired on node): uncompressed reader, all 4 pages raw .809; gold page
only raw .841. The distractor pages cost about 2-3 pt whether they are raw or compressed.

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
questions (no held-out split). Closed without a judge cell: see finding 7.

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
| s42 keep the most-attended EIGHTH overall (job 189057, homer-019) | 1,500 | **.613** | .610 | **+.003 [−.011, +.017]** | .78 (59 / 55) | 0, 1,500, 0.121 | 52.2, 133 |
| s42 keep the most-attended SIXTEENTH overall (job 189906, homer-013) | 1,500 | .596 | .610 | −.014 [−.030, +.002] | .115 (70 / 91) | 0, 1,500, 0.059 | 25.4, 106 |
| s42 keep the most-attended half of each chunk (job 188484, homer-016) | 1,500 | .606 | .610 | −.004 [−.015, +.007] | .57 (35 / 41) | 0, 1,500, 0.496 | 213.8, 294 |
| s42 keep every 2nd slot (job 188483, homer-016) | 1,500 | .505 | .610 | −.105 [−.127, −.085] | 3e-24 (49 / 207) | 0, 1,500, 0.504 | 216.9, 297 |
| s42 keep the first half of each chunk (job 188482, homer-013) | 1,500 | .495 | .610 | −.115 [−.136, −.095] | 4e-28 (46 / 219) | 0, 1,500, 0.496 | 213.8, 294 |
| s43 keep the most-attended half overall (job 188522, homer-012) | 1,500 | **.615** | .615 | **+.000 [−.009, +.009]** | 1.0 (25 / 25) | 0, 1,500, 0.496 | 213.8, 294 |
| s43 keep the most-attended QUARTER overall (job 189056, homer-019) | 1,500 | **.619** | .615 | **+.005 [−.007, +.017]** | .51 (45 / 38) | 0, 1,500, 0.246 | 106.0, 186 |
| s43 keep the most-attended EIGHTH overall (job 189905, homer-020) | 1,500 | **.605** | .615 | **−.010 [−.025, +.005]** | .21 (54 / 69) | 0, 1,500, 0.121 | 52.2, 133 |
| s43 keep every 2nd slot (job 188537, homer-012) | 1,500 | .508 | .615 | −.107 [−.127, −.087] | 8e-26 (44 / 204) | 0, 1,500, 0.504 | 216.9, 297 |
| photos s42, keep the most-attended half overall (job 188539, homer-016) | 1,500 | .677 | .677 | +.000 [−.013, +.013] | 1.0 (49 / 49) | 0, 1,500, 0.492 | 26.0 (plain 52.9), 97 (plain 124) |
Answers identical to plain: s42 half overall 1,289 / 1,500, quarter 1,197, eighth 1,107, sixteenth 1,001, per chunk
1,230, every 2nd 917, first half 924; s43 half overall 1,306, quarter 1,200, eighth 1,076, every 2nd 903; photos half
1,186. By source vs plain (infographics / multi-page / slides): s42 quarter .474 / .680 / .680 vs .480 / .666 / .684;
s43 quarter .486 / .686 / .686 vs .480 / .678 / .686; s42 eighth .478 / .666 / .694; s43 eighth .480 / .664 / .670;
s42 sixteenth .472 / .640 / .676 (multi-page −.026 [−.058, +.006]). Between levels: s42 eighth vs quarter +.001
[−.011, +.014], 48 / 46; s43 eighth vs quarter −.015 [−.027, −.003], 36 / 58, p = .03; s42 sixteenth vs eighth −.017
[−.031, −.003], 48 / 73, p = .03 (both below the .02 band).
Gate (attention-picked minus the better blind half ≥ +4 pt, CI above 0, both seeds): seed 42 +.105 [+.085, +.126],
205 / 47, p = 1e-24 (188485 − 188483); seed 43 +.107 [+.087, +.127], 203 / 43, p = 5e-26 (188522 − 188537). **Met:
adopted.** Quarter (gate written before the cell: within .02 of plain → three quarters unused per question): met at
both seeds (+.001, +.005). **Adopted.** Eighth (gate written before any cell: within .02 → seven eighths unused; a
loss ≥ .05 → the free share lies between 3/4 and 7/8): met at both seeds (+.003, −.010). **Adopted**; the seed-43 CI
reaches −.025. Sixteenth (gate written before the cell: within .02 → fifteen sixteenths unused; a loss ≥ .05 → the
free share lies between 7/8 and 15/16; .02-.05 decides nothing): met at seed 42 (−.014). **Provisional**: one seed,
no cell left to repeat it. Every cell of this arm ran on an el9_8 node (homer-012, 013, 016, 019, 020), as its
baseline (homer-011 / homer-016); that image agrees with itself (jobs 188476, 188580). ARM CLOSED (cap used up).

## Arm G — route then densify (accepted by the owner 2026-11-11)
The page the model picks (same pick as "picked page raw", 1,500 / 1,500 picks identical) is re-compressed at rate 4
(twice its slots) and stays compressed; the other pages keep rate 8. Code: `RAW=route1 DENSE=<rate>` in
`eval_mm_4b.sh` (`--dense_rate` in `scripts/eval_mm_docqa.py`, default path unchanged). Baselines: plain (seed 42 job
184781, homer-011; seed 43 job 184784, homer-016) and picked page raw (jobs 188441 / 188460).
| cell | model | n | judge accuracy ↑ | baseline judge accuracy ↑ | difference (95% CI) | McNemar p (up / down) | health (unparseable, rows, slots vs plain) | compression: slots / question, decoder prompt tokens |
|---|---|---|---|---|---|---|---|---|
| picked page at rate 4 (job 188590, homer-011) | 184778 s42 | 1,500 | .673 | plain .610 | +.063 [+.043, +.083] | <.001 (158 / 64) | 0, 1,500, 1.250 | 538.2 (picked page 215.8), 619 |
| same cell | 184778 s42 | 1,500 | .673 | picked page raw .820 | −.147 [−.169, −.126] | <.001 (44 / 265) | | raw: 322.4 + 1 raw page, 1,264 |
| picked page at rate 4 (job 188591, homer-012) | 184780 s43 | 1,500 | .663 | plain .615 | +.049 [+.030, +.067] | <.001 (147 / 74) | 0, 1,500, 1.250 | 538.2 (picked page 215.9), 619 |
| same cell | 184780 s43 | 1,500 | .663 | picked page raw .822 | −.159 [−.180, −.137] | <.001 (38 / 276) | | raw: 322.4 + 1 raw page, 1,264 |
Plain: 430.7 slots, 511 decoder prompt tokens per question. By source, rate 4 vs plain: seed 42 infographics .562 vs
.480 (+.082), multi-page .744 vs .666 (+.078), slides .712 vs .684 (+.028); seed 43 .528 vs .480, .758 vs .678, .704
vs .686. Answers identical to plain: 984 / 1,500 (seed 42), 993 / 1,500 (seed 43). On the 21 / 16 questions where the
pick is wrong, rate 4 scores .524 / .500 vs plain .571 / .562 (too few to read).
**Gate** (gain over plain ≥ 2/3 of the raw page's, CI above 0, at ≤ 1/3 of its extra decoder tokens, both seeds):
tokens met (+107.5 vs +753, .143×); gain FAILED at both seeds (+.063 < +.140; +.049 < +.138). **Closed.** Caveat: the
models were trained at rate 8 only, so the fail may be the untrained rate, not the idea.
How the flag was checked: invariant tests PASS (job 188523: positions 12 / 12, collators 33 / 33, freeze 5 / 5); plain
300-question regression PASS (job 188524). First version FAILED its rate-8 control (job 188525: 233 / 300 answers
identical to plain, needed ≥ 97%): it re-compressed the picked page alone, the plain path compresses the four pages in
one padded batch. Patched (9 lines): the page is re-compressed inside its example's batch. Re-check PASSED (job
188580, `scripts/probes/g_checks.sbatch`, homer-013): tests, plain 300 / 300, rate-8 control 300 / 300 identical to
plain (judge .610 = .610, slots 118,147 both).

## Arm F — photo text vs pixels (TextVQA, 500 questions)
`eval_data/probes_textvqa_ocr` is built (job 188545, `scripts/probes/build_textvqa_ocr.py`): the 500 TextVQA rows of
the photo set, unchanged, with the OCR tokens of the same TextVQA validation row as text; 500 / 500 questions matched.
OCR text: 13 tokens per row on average (median 7, 14 rows empty); a gold answer appears word for word in it on .596 of
rows. Baseline: the 500 TextVQA rows of the photo cell of 184778 (job 184783, homer-011, pixels at PX=1003520).
| cell | model | n | judge accuracy ↑ | baseline judge accuracy ↑ (pixels) | difference (95% CI) | McNemar p (up / down) | health (unparseable, rows, slots vs baseline) | compression: slots / question, decoder prompt tokens |
|---|---|---|---|---|---|---|---|---|
| OCR text instead of pixels, `pisco_text` (job 189059, homer-012) | 184778 s42 | 500 | .346 | .780 | −.434 [−.480, −.384] | 6e-50 (17 / 234) | 0, 500, 0.081 | 4.4 (pixels 54.2), 65 (pixels 115) |
EM: .292 vs .714, −.422 [−.470, −.372]. The input changes by design (text of a few words instead of the image), so
the comparison is "same questions, this input instead of that one", not "same budget". Reading: the OCR words lose
too much (layout, which word is the brand, context in the picture) for the decoder to answer from them; the pixels
already score above the share of rows whose answer is in the OCR at all (.780 vs .596). Gate (reported only; a gain
≥ 5 pt motivates a mixed-input arm): no gain. Seed 43 is not run: its photo baseline (job 184785) ran on homer-008,
the node that changes image answers.

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
- Evidence-then-answer: documents +.001 (seed 42) and −.009 (seed 43), text −.019 (seed 42); no gain (finding 6).
- Passage order inside a long chunk: −.002 (seed 42), +.003 (seed 43). Unused.
- Joining the distractor passages: +.022 (p = .027) at seed 42, +.016 (n.s.) at seed 43 over ten chunks; not adopted.
- "Rank by attention, then join the top 3" as a serving mode: equals joining all ten at both seeds; +.013 (n.s.) over
  ten chunks at seed 43.
- A judge cell for a fused page picker (expected gain below the noise band).
- Blind slot halving on the reference models: −.105 / −.115 (seed 42), −.107 (seed 43).
- The picked page at twice the slots (rate 4), untrained: +.063 / +.049 over plain, below the +.14 gate at both seeds.
- Photo text read as OCR words instead of pixels (TextVQA, seed 42): −.434 (finding 8). No mixed-input arm.
