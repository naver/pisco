# Probe campaign 2026-11 — why the compressed model loses, and what recovers it without training

Law: `campaign_probes/CAMPAIGN.md`. Live state: `campaign_probes/STATE.md`. This page is rewritten whenever an arm decides.
Metric everywhere: judge accuracy ↑ (share of questions the Qwen3-8B rubric judge accepts), paired against a baseline
cell on the same questions (95% bootstrap CI, exact McNemar p). Compression: 1 slot per 8 visual or text tokens
(`compr_rate: 8` in every model here). Document set (`mm_docqa_1500_1m`, PX=1003520, 4 pages per question): the plain
cell uses 430.7 slots per question (646,042 / 1,500). HotpotQA (10 passages per question): about 177 slots per question.

## Status (2026-11-11 06:45 KST)
- **Arm A decided at both seeds.** Showing the one page the model picks as raw pixels, the other three as slots,
  recovers the whole document gap: .820 / .822 vs .610 / .615 compressed, above the all-raw reader (.809).
  Removing the three distractor pages instead gains only about 2 pt. The loss is in reading the compressed answer page.
- **Arm B decided at both seeds: the long-chunk gain is context between the two answer passages.** Joining the two
  gold passages into one chunk beats ten separate chunks by +.073 (seed 42) and +.067 (seed 43). One chunk of all ten
  gains only +.014 (n.s.) at seed 43, so the "+.037 for one long chunk" the campaign set out to explain is half noise.
- **Arm E, seed 42 (provisional): given the question, half the slots carry everything the decoder uses.** Keeping
  the half of the slots the decoder attends to most gives the plain score (.610 = .610); a blind half loses .105-.115.
- **Arm D decided:** attention picks image pages better than the digit prompt (+2.5 pt), the digit prompt picks text
  passages better (+4.9 pt); the end-to-end judge effect of either is too small to measure.
- **Arm C (evidence-then-answer on text) did not close the text gap**: the compressed − raw gap shrinks 3 pt on DROP
  and grows on DuoRC and RACE. The text loss is mostly missing content, not missing room to reason.
- Running: arm A evidence-then-answer (seed 42); arm B no-oracle grouping (2 cells), one-chunk-of-ten re-run on
  today's code, golds joined on the tail-layout model and its ten-chunk re-run; arm E seed-43 repeat; arm G checks (tests, plain regression, rate-8 control).

## Findings (ranked)
1. **The document loss is reading the compressed answer page — route-then-expand removes it.** Picked page raw:
   +.210 [+.187, +.233] (seed 42) and +.207 [+.183, +.232] (seed 43), n = 1,500 each, McNemar p < 1e-60; the pick is
   right on 98.6% / 98.9% of questions. Gold page alone, compressed: +.022 / +.025. Motivates: serve with
   route-then-expand (decoder prompt 1,264 tokens vs 511 compressed), and put training effort into how well one
   page's slots can be read (stage 1/2), not into distractor robustness. Arm G tests the cheaper version (the picked
   page at twice the slots instead of raw).
2. **The long-chunk gain is context between the passages that answer together.** Golds joined (9 chunks) vs ten
   chunks: +.073 [+.051, +.095] (seed 42, 183814) and +.067 [+.046, +.088] (seed 43, 183816), n = 1,500 each, same
   slots (.998×). All on bridge questions (+.094 / +.095); comparison questions lose a little (−.010 / −.043). Against
   one chunk of all ten: +.038 (seed 42, against the 11-07 file; re-run on today's code queued) and +.053 [+.029, +.076]
   (seed 43). Motivates: a long-chunk stage 1 that packs RELATED passages together (the sibling `longchunk` campaign),
   and "rank, then re-compress the top passages together" at inference (two no-oracle cells running).
3. **provisional (seed 42 only): given the question, the decoder uses about half the slots.** Keep the most-attended
   half over all pages: .610 vs .610, +.000 [−.009, +.010], 27 / 27, 1,289 / 1,500 answers identical; per chunk:
   .606 (−.004 [−.015, +.007]). Blind halves: every 2nd slot .505 (−.105), first half of each chunk .495 (−.115).
   The selection needs one full pass over every slot with the question, so it saves decoder context, not storage.
   Motivates: a query-aware slot budget, or nested-budget training so the needed half can be found without the full
   pass. Seed-43 repeat running; a 75% drop is next.
4. **Storing slots in int8 costs nothing (seed 42).** +.001 [−.003, +.006], 1,445 / 1,500 answers identical. A
   serving option (half the memory of bf16 slots), not a recovery. Seed-43 repeat not run yet.
5. **On text, giving the decoder room to reason does not close the gap.** Evidence-then-answer on MRQA: compressed
   .699 vs .718 plain; raw .807 vs .807. The gap on DROP shrinks from −.204 to −.172 (below the 5 pt gate) and grows on
   DuoRC (−.084 → −.156) and RACE (−.152 → −.188). When the compressed model copies evidence from its slots, narrative and
   biomedical answers get worse (BioASQ .912 → .812): the slots do not hold exact strings well. Motivates: stage-1/2
   objectives for exact content (the transcription side), not reasoning labels. Closes evidence-then-answer labels for text.
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
| slots in int8 (job 188478, homer-016) | 184778 s42 | 1,500 | .611 | .610 | +.001 [−.003, +.006] | .77 (7 / 5) | 0, 1,500, 1.000 | 430.7, 511 |
| evidence, then answer (job 188439) | 184778 s42 | — | running | .610 | — | — | — | — |

By source, picked page raw vs plain (seed 42): infographics .716 vs .480, multi-page .922 vs .666, slides .822 vs .684.
Context (older cells, not paired on node): uncompressed reader, all 4 pages raw .809; gold page only raw .841. The
distractor pages cost about 2-3 pt whether they are raw or compressed.

## Arm B — long-chunk mechanism (HotpotQA, 10 passages, 1,500 questions; 183814 = seed 42, 183816 = seed 43)
Baselines: g1 = each passage its own chunk; g10 = all 10 passages in one chunk. Seed 42: g1 re-run on today's code
(job 188486, homer-010: .582; the 11-07 file 185505 gave .579, 1,445 / 1,500 answers identical); g10 .617 is the
11-07 file (job 185601), re-run on today's code queued (job 188518). Seed 43: both on today's code (jobs 188509 / 188510).
Compression column: chunks per question, slots per question (1 slot per 8 text tokens), decoder prompt tokens.
| cell | seed | n | judge accuracy ↑ | baseline judge accuracy ↑ | difference (95% CI) | McNemar p (up / down) | health (unparseable, rows, slots vs baseline) | compression: chunks, slots / question, decoder prompt tokens |
|---|---|---|---|---|---|---|---|---|
| golds joined, distractors alone (job 188471, homer-016) | 42 | 1,500 | **.655** | g1 .582 | **+.073 [+.051, +.095]** | 1.5e-10 (200 / 91) | 0, 1,500, 0.998 | 9, 176.8 |
| golds joined, distractors alone (job 188511, homer-016) | 43 | 1,500 | **.669** | g1 .601 | **+.067 [+.046, +.088]** | 8e-10 (186 / 85) | 0, 1,500, 0.998 | 9, 176.8, 282 |
| same cell, seed 42 | 42 | 1,500 | .655 | g10 .617 (11-07) | +.038 [+.015, +.061] | .002 (185 / 128) | 0, 1,500, 1.020 | |
| same cell, seed 43 | 43 | 1,500 | .669 | g10 .615 | **+.053 [+.029, +.076]** | 9e-6 (200 / 120) | 0, 1,500, 1.020 | |
| one chunk of ten (job 188510, homer-016) | 43 | 1,500 | .615 | g1 .601 | +.014 [−.008, +.035] | .23 (152 / 131) | 0, 1,500, 0.978 | 1, 173.3, 246 |
| distractors joined, golds alone (job 188472, homer-020) | 42 | 1,500 | .604 | g1 .582 | +.022 [+.003, +.041] | .027 (122 / 89) | 0, 1,500, 0.983 | 3, 174.2 |
| same cell | 42 | 1,500 | .604 | g10 .617 (11-07) | −.013 [−.037, +.011] | .33 | 0, 1,500, 1.005 | |
| all 10 joined, shuffled order (job 188473, homer-011) | 42 | 1,500 | .615 | g10 .617 (11-07) | −.002 [−.023, +.020] | .91 (141 / 144) | 0, 1,500, 1.000 | 1, 173.3 |
| distractors joined, golds alone (job 188512, homer-016) | 43 | 1,500 | .617 | g1 .601 | +.016 [−.003, +.035] | .11 (113 / 89) | 0, 1,500, 0.983 | 3, 174.2 |
| same cell | 43 | 1,500 | .617 | g10 .615 | +.002 [−.021, +.025] | .91 (159 / 156) | 0, 1,500, 1.005 | |
| all 10 joined, shuffled order (job 188513, homer-016) | 43 | 1,500 | .618 | g10 .615 | +.003 [−.021, +.026] | .87 (160 / 156) | 0, 1,500, 1.000 | 1, 173.3 |

By type, golds joined vs g1: bridge (n = 1,197) .647 vs .553 (seed 42), .668 vs .572 (seed 43); comparison (n = 303)
.686 vs .696 (seed 42), .673 vs .716 (seed 43, −.043 [−.079, −.007]).
Reading: the gain needs the two gold passages in the same chunk, at both seeds. Distractors joined: +.022 / +.016,
under the noise band. Passage order: closed at both seeds. Running: golds joined on the tail-layout model 183538 (job 188514) and its
ten-chunk re-run on today's code (job 188519); the no-oracle cells (jobs 188520 / 188521: the 2 or 3 passages the
model's attention ranks first, joined; both golds are in the top 2 on .495 of questions, top 3 on .667). Gate:
no-oracle minus g1 ≥ +.03 with CI above 0.

## Arm C — reasoning vs reading on text (MRQA, 6 sources × 250, one passage each, nothing cut)
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
uncompressed .000 [−.017, +.016] (89 / 89). Health: unparseable 0, 1,500 rows each, compressed slots 80,003 both.
Compressed: 80,003 slots (53.3 per question). The compressed plain cell re-ran on the same node with 1,500 / 1,500
identical answers. Gate: the compressed − raw gap on DROP / DuoRC / RACE shrinks by ≥ 5 pt with the same treatment on
both sides. NOT met (DROP +3 pt, DuoRC −7 pt, RACE −4 pt). Thinking mode (the arm's second treatment) is not run yet:
it needs a new eval flag and ~4 h per cell; given this result it is unlikely to change the reading.

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

## Arm E — attention-guided slot pruning (model 184778, documents, keep half the slots)
How a cell works: one extra pass over the full prompt (all slots + the question); the attention of the last prompt
position on each slot, averaged over the heads of every full-attention layer; keep the top half (per chunk, or over all
pages), drop the rest and their placeholders, then answer. Controls passed: keep-all through the attention pass gives
300 / 300 plain answers (job 188480); the patched eval's plain path gives 300 / 300 plain answers, EM .477 = recorded
(job 188481). All cells on el9_8 nodes, as the baseline (184781, homer-011).
| cell | n | judge accuracy ↑ | baseline judge accuracy ↑ (plain, 184781) | difference (95% CI) | McNemar p (up / down) | health (unparseable, rows, slots vs baseline) | compression: slots / question, decoder prompt tokens |
|---|---|---|---|---|---|---|---|
| plain (baseline, job 184781) | 1,500 | .610 | — | — | — | 0, 1,500, 1.000 | 430.7, 511 |
| keep the most-attended half overall (job 188485, homer-016) | 1,500 | **.610** | .610 | **+.000 [−.009, +.010]** | 1.0 (27 / 27) | 0, 1,500, 0.496 | 213.8, 294 |
| keep the most-attended half of each chunk (job 188484, homer-016) | 1,500 | .606 | .610 | −.004 [−.015, +.007] | .57 (35 / 41) | 0, 1,500, 0.496 | 213.8, 294 |
| keep every 2nd slot (job 188483, homer-016) | 1,500 | .505 | .610 | −.105 [−.127, −.085] | 3e-24 (49 / 207) | 0, 1,500, 0.504 | 216.9, 297 |
| keep the first half of each chunk (job 188482, homer-013) | 1,500 | .495 | .610 | −.115 [−.136, −.095] | 4e-28 (46 / 219) | 0, 1,500, 0.496 | 213.8, 294 |
Answers identical to plain: most-attended overall 1,289 / 1,500, per chunk 1,230, every 2nd 917, first half 924.
By source, most-attended overall vs plain: infographics .474 vs .480, multi-page .672 vs .666, slides .684 vs .684.
Gate: attention-picked minus the better blind half (every 2nd, .505) ≥ +4 pt with CI above 0: +.105 at seed 42, met.
Running: the seed-43 repeat (job 188522, 184780, vs .615). Next: photos, and a 75% drop (keep a quarter; gate: within
.02 of plain → three quarters unused per question).

## Arm G — route then densify (accepted by the owner 2026-11-11)
The page the model picks (same pick as "picked page raw") is re-compressed at rate 4 (twice its slots) and stays
compressed; the other pages keep rate 8. Code: `RAW=route1 DENSE=<rate>` in `eval_mm_4b.sh` (`--dense_rate` in
`scripts/eval_mm_docqa.py`, 25 lines, default path unchanged). Checks running before the two full cells: invariant
tests (job 188523), a 300-question plain regression (job 188524), and the flag at rate 8 on 300 questions (job 188525),
which must give the plain answers. Gate: gain over plain ≥ +.14 (2/3 of picked-page-raw's +.21) with CI above 0, at
≤ 1/3 of its extra decoder tokens, both seeds.

## The instrument: same answers on re-run?
| check | model | set | n | node of the re-run (image) | node of the recorded run | answers identical | judge accuracy ↑, re-run vs recorded |
|---|---|---|---|---|---|---|---|
| gate re-run (job 188419) | 184778 compressed | documents | 1,500 | homer-011 (el9_8) | homer-011 | 1,500 / 1,500 | .610 vs .610 |
| same image (job 188476) | 184778 compressed | documents, first 300 | 300 | homer-016 (el9_8) | homer-011 | 300 / 300 | .610 vs .610 |
| older image (job 188477) | 184778 compressed | documents, first 300 | 300 | homer-007 (el9_7) | homer-011 | 300 / 300 | .610 vs .610 |
| homer-008 (job 188458) | 184778 compressed | documents, first 300 | 300 | homer-008 (el9_7) | homer-011 | 243 / 300 | .623 vs .610 (9 up / 5 down, p ≈ .42) |
| compressed text (job 188486) | 183814 | HotpotQA ten chunks | 1,500 | homer-010 (el9_8), code of 11-11 | homer-006 (el9_7), code of 11-07 | 1,445 / 1,500 | .582 vs .579 (10 / 6) |
| compressed text, same node (job 188445) | 184778 | MRQA | 1,500 | homer-016 | homer-016 | 1,500 / 1,500 | .718 vs .718 |
| compressed text, older image (job 188471 rerun) | 183814 | HotpotQA golds joined | 1,500 | homer-007 (el9_7) | homer-016 (el9_8) | 1,500 / 1,500 | .655 vs .655 |
All nodes carry A100-SXM4-80GB GPUs. Reading: homer-008 alone moves answers (homer-007, same older image, agrees on
images and on text). The 55 changed text answers of 185505 come from homer-006 (untested) or the code date (11-07);
arm B pairs with the 11-11 re-run.

## New eval sets and options
- `eval_data/probes_hotpot_{gold_joined,distr_joined,g10_shuf}`, 1,500 rows each (job 188465); the builder first
  rebuilds `text_hotpotqa_1500_g10` and matches it on 1,500 / 1,500 rows.
- `eval_data/probes_hotpot_pred{2_digit,2_attn,2_product,2_rrf,3_attn,3_product}`, 1,500 rows each (job 188516): no
  oracle, the K passages model 183814 ranks first joined, others alone. Both golds in the top 2: digit .404,
  attention .495, product .487; top 3: attention .667.
- `scripts/eval_mm_docqa.py --mem_keep_mode attn | attn_global` (env `KEEPF=0.5 KEEPM=attn`): keeps the slots the
  decoder attends to most, same counts as `first`. Invariant tests 45 / 45 (job 188479).
- `scripts/eval_mm_docqa.py --dense_rate R` (env `RAW=route1 DENSE=R`): the picked page re-compressed at rate R (arm G).

## Waiting on a new eval set
- Photo text vs pixels (arm F): the photo set's rows carry no OCR text (`docs=[""]`); needs a set with TextVQA's OCR
  tokens (500 questions).

## Closed
- Gold page only (both seeds): +.022 / +.025, below the +3 pt gate. Distractor pages are not the main loss.
- int8 slots: no change at seed 42 (finding 4).
- Passage order inside a long chunk: −.002 / +.003 at both seeds.
- Joining the distractor passages: +.022 / +.016 over ten chunks, under the noise band.
- A judge cell for a fused page picker (expected gain below the noise band).
- Evidence-then-answer on text (arm C): the gap does not shrink (finding 5).
- Blind slot halving on the reference model: −.105 / −.115 (arm E baselines).
