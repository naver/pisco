# Probe campaign 2026-11 — why the compressed model loses, and what recovers it without training

Law: `campaign_probes/CAMPAIGN.md`. Live state: `campaign_probes/STATE.md`. This page is rewritten whenever an arm decides.
Metric everywhere: judge accuracy ↑ (share of questions the Qwen3-8B rubric judge accepts), paired against a baseline
cell on the same questions (95% bootstrap CI, exact McNemar p). Compression: 1 slot per 8 visual or text tokens
(`compr_rate: 8` in every model here). Document set (`mm_docqa_1500_1m`, PX=1003520, 4 pages per question): the plain
cell uses 430.7 slots per question (646,042 / 1,500). HotpotQA (10 passages per question): about 177 slots per question.

## Status (2026-11-11 05:45 KST)
- **Arm A decided at both seeds.** Showing the one page the model picks as raw pixels, the other three as slots,
  recovers the whole document gap: .820 / .822 vs .610 / .615 compressed, above the all-raw reader (.809).
  Removing the three distractor pages instead gains only about 2 pt. The loss is in reading the compressed answer page.
- **Arm B: joining the two gold passages into one chunk gains +.073 over ten separate chunks** and beats one chunk of
  all ten by +.038 (one model, provisional; repeat on the seed-43 twin running). Passage order does not matter.
- **Arm D decided:** attention picks image pages better than the digit prompt (+2.5 pt), the digit prompt picks text
  passages better (+4.9 pt); the end-to-end judge effect of either is too small to measure.
- **Arm C (evidence-then-answer on text) did not close the text gap**: the compressed − raw gap shrinks 3 pt on DROP
  and grows on DuoRC and RACE. The text loss is mostly missing content, not missing room to reason.
- Running: arm A evidence-then-answer (seed 42), arm E three slot-pruning cells, arm B six repeat / control cells.

## Findings (ranked)
1. **The document loss is reading the compressed answer page — route-then-expand removes it.** Picked page raw:
   +.210 [+.187, +.233] (seed 42) and +.207 [+.183, +.232] (seed 43), n = 1,500 each, McNemar p < 1e-60; the pick is
   right on 98.6% / 98.9% of questions. Gold page alone, compressed: +.022 / +.025. Motivates: serve with
   route-then-expand (decoder prompt 1,264 tokens vs 511 compressed; all-raw not logged, at most 4 × 980 visual
   tokens), and put training effort into how
   well one page's slots can be read (stage 1/2), not into distractor robustness.
2. **provisional: The long-chunk gain is context between related passages, and unrelated passages in the same chunk
   dilute it.** Golds joined (9 chunks) +.073 vs ten chunks, +.038 vs one chunk of ten; all on bridge questions (.647 vs
   .553 / .596), none on comparison questions. Distractors joined: +.022 (noise edge). Motivates: a long-chunk stage 1
   that packs RELATED passages together (the sibling `longchunk` campaign), and a two-pass "rank, then re-compress the
   top passages together" inference probe (sets built, see below). Needs the seed-43 repeat.
3. **Storing slots in int8 costs nothing.** +.001 [−.003, +.006], 1,445 / 1,500 answers identical. A serving option
   (half the memory of bf16 slots), not a recovery.
4. **On text, giving the decoder room to reason does not close the gap.** Evidence-then-answer on MRQA: compressed
   .699 vs .718 plain; raw .807 vs .807. The gap on DROP shrinks from −.204 to −.172 (below the 5 pt gate) and grows on
   DuoRC (−.084 → −.156) and RACE (−.152 → −.188). When the compressed model copies evidence from its slots, narrative and
   biomedical answers get worse (BioASQ .912 → .812): the slots do not hold exact strings well. Motivates: stage-1/2
   objectives for exact content (the transcription side), not reasoning labels. Closes evidence-then-answer labels for text.
5. **Which page picker to use depends on the modality.** Attention mass: +.025 pick@1 on 10-page documents, −.049 on
   HotpotQA. A per-source switch passes the arm D gate; product fusion never loses (+.019 / +.006). No judge cell: the
   expected judge gain (+.025 × a .25 hit-vs-miss judge gap ≈ +.006) is below the .02 noise band.

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

## Arm B — long-chunk mechanism (model 183814, HotpotQA, 10 passages, 1,500 questions)
Baselines: g1 = each passage its own chunk (re-run on el9_8, job 188486: .582; recorded 185505: .579, 1,445 / 1,500
answers identical); g10 = all 10 passages in one chunk (.617, job 185601).
| cell | n | judge accuracy ↑ | baseline judge accuracy ↑ | difference (95% CI) | McNemar p (up / down) | health (unparseable, rows, slots vs baseline) | compression: chunks, slots / question |
|---|---|---|---|---|---|---|---|
| golds joined, distractors alone (job 188471, homer-016) | 1,500 | **.655** | g1 .582 | **+.073 [+.051, +.095]** | 1.5e-10 (200 / 91) | 0, 1,500, 0.998 | 9, 176.8 |
| same cell | 1,500 | .655 | g10 .617 | **+.038 [+.015, +.061]** | .002 (185 / 128) | 0, 1,500, 1.020 | |
| distractors joined, golds alone (job 188472, homer-020) | 1,500 | .604 | g1 .582 | +.022 [+.003, +.041] | .027 (122 / 89) | 0, 1,500, 0.983 | 3, 174.2 |
| same cell | 1,500 | .604 | g10 .617 | −.013 [−.037, +.011] | .33 | 0, 1,500, 1.005 | |
| all 10 joined, shuffled order (job 188473, homer-011) | 1,500 | .615 | g10 .617 | −.002 [−.023, +.020] | .91 (141 / 144) | 0, 1,500, 1.000 | 1, 173.3 |

By type, golds joined vs g1: bridge (n = 1,197) .647 vs .553; comparison (n = 303) .686 vs .696.
Reading (provisional, one model): the gain needs the two gold passages in the same chunk; eight distractors sharing
that chunk (g10) halve it. Order is closed on this model (tie; repeat queued). Running: the same five cells on the
seed-43 twin 183816 (jobs 188509-188513) and golds joined on the tail-layout model 183538 (job 188514, vs its g1 .568).

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
| per-source switch (chosen on the other half) | .885 | +.025 [+.010, +.039], .001 | .922 | +.000 |
Gate (≥ +2 pt on documents AND no loss on HotpotQA) met by the per-source switch only. Closed without a judge cell:
see finding 5.

## Arm E — attention-guided slot pruning (model 184778, documents, keep half the slots)
Controls passed: keep-all through the attention pass gives 300 / 300 plain answers (job 188480); the patched eval's
plain path gives 300 / 300 plain answers, EM .477 = recorded (job 188481).
| cell | n | judge accuracy ↑ | baseline judge accuracy ↑ (plain, 184781) | difference (95% CI) | McNemar p (up / down) | health (unparseable, rows, slots vs baseline) | compression: slots / question |
|---|---|---|---|---|---|---|---|
| keep every 2nd slot (job 188483, homer-016) | 1,500 | .505 | .610 | −.105 [−.127, −.085] | 3e-24 (49 / 207) | 0, 1,500, 0.504 | 216.9 |
| keep the first half of each chunk (job 188482) | — | running | .610 | — | — | — | — |
| keep the most-attended half of each chunk (job 188484) | — | running | .610 | — | — | — | — |
| keep the most-attended half overall (job 188485) | — | running | .610 | — | — | — | — |
Gate: attention-picked minus the better untrained pick ≥ +4 pt with CI above 0.

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

## Waiting on a new eval set
- Photo text vs pixels (arm F): the photo set's rows carry no OCR text (`docs=[""]`); needs a set with TextVQA's OCR
  tokens (500 questions).

## Closed
- Gold page only (both seeds): +.022 / +.025, below the +3 pt gate. Distractor pages are not the main loss.
- int8 slots: no change (finding 3).
- Passage order inside a long chunk (seed 42 only, tie; repeat queued).
- A judge cell for a fused page picker (expected gain below the noise band).
- Evidence-then-answer on text (arm C): the gap does not shrink (finding 4).
