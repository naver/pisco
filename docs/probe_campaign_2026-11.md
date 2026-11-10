# Probe campaign 2026-11 — why the compressed model loses, and what recovers it without training

Law: `campaign_probes/CAMPAIGN.md`. Live state: `campaign_probes/STATE.md`. This page is rewritten whenever an arm decides.
Metric everywhere: judge accuracy ↑ (share of questions the Qwen3-8B rubric judge accepts), paired against a baseline
cell on the same questions (95% bootstrap CI, McNemar p). Compression: 1 slot per 8 visual or text tokens
(`compr_rate: 8` in every model used here); on the document set at PX=1003520 the plain cell uses 430.7 slots per
question (646,042 slots / 1,500 questions, 4 pages each, job 188419).

## Status (2026-11-11 04:41 KST)
- **The judge is reproducible on nodes with the same system image.** Two re-runs on el9_8 nodes (homer-011, homer-016)
  give the recorded answers exactly. One run on an older-image node (homer-008, el9_7) changed 57 of 300 answers on page
  images. Image comparisons pair cells from one image only.
- **First arm result (arm B, provisional):** joining only the 8 distractor passages into one chunk, with each gold
  passage alone, already gains +.025 over ten separate chunks. Most of the long-chunk gain does not need the gold
  passages to share a chunk. The gold-passages-joined cell is running.
- **Arm D controls recovered** on both sets (re-runs equal the 2026-11-06 numbers to the last digit); the fusion step
  runs next.
- **Arm E started:** a new eval option keeps the slots the decoder attends to most (code + tests done, 4 cells queued).
- 19 probe cells are open (8 running, 11 queued at 04:41). The `gpu-be` partition keeps taking GPUs back (preemption: SLURM stops the job and requeues
  it from zero): 18 preemptions of campaign cells since 03:22, 14 of them on homer-008 (sacct).

## The instrument: same answers on re-run?
| check | model | set | n | node of the re-run (image) | node of the recorded run | answers identical | judge accuracy ↑, re-run vs recorded | judge flips (up / down) |
|---|---|---|---|---|---|---|---|---|
| gate re-run (job 188419) | 184778 (seed 42) compressed | documents, 1M px, 4 pages | 1,500 | homer-011 (el9_8) | homer-011 (el9_8) | 1,500 / 1,500 | .610 vs .610 | 0 / 0 |
| same-image check (job 188476) | 184778 compressed | documents, first 300 questions | 300 | homer-016 (el9_8) | homer-011 (el9_8) | 300 / 300 | .610 vs .610 | 0 / 0 |
| other-image check (job 188458) | 184778 compressed | documents, first 300 questions | 300 | homer-008 (el9_7) | homer-011 (el9_8) | 243 / 300 | .623 vs .610 | 9 / 5 (McNemar p ≈ .42) |
| uncompressed check (job 188456) | uncompressed Qwen3.5-4B | MRQA text, first 300 questions | 300 | homer-011 (el9_8) | homer-008 (el9_7) | 300 / 300 | (EM .367 vs .367) | — |
| attention-route re-runs (jobs 188442, 188443) | 183814 | 10-page documents / HotpotQA | 1,500 each | homer-016 / homer-013 (el9_8) | homer-014 (el9_8) | every summary number equal | digit route .861 / .922, attention .885 / .873 | — |

Still to measure: homer-007 (older image, job 188477, waiting for that node) tells whether the older image or homer-008
alone changes the answers. Compressed text across images: job 188486 re-runs HotpotQA's ten-chunk baseline (recorded
on homer-006, el9_7) on an el9_8 node.

## Arm B — long-chunk mechanism (model 183814, HotpotQA, 10 passages, 1,500 questions)
Baselines: g1 = each passage its own chunk (judge .579, job 185505); g10 = all 10 passages in one chunk (.617, job 185601).
| cell | model | set | n | judge accuracy ↑ | baseline judge accuracy ↑ | difference (95% CI) | McNemar p | health (unparseable, rows, slots vs baseline) |
|---|---|---|---|---|---|---|---|---|
| 2 golds alone + 8 distractors joined, 3 chunks (job 188472) | 183814 | `probes_hotpot_distr_joined` | 1,500 | .604 | g1 .579 | **+.025 [+.005, +.044]** | .012 | 0, 1,500, 261,308 slots = 0.983 × g1 |
| same cell | 183814 | same | 1,500 | .604 | g10 .617 | −.013 [−.037, +.011] | .33 | 0, 1,500, 1.005 × g10 |
| by type: bridge questions | | | 1,197 | .587 | g1 .551 | +.037 [+.016, +.058] | .001 | |
| by type: comparison questions | | | 303 | .670 | g1 .693 | −.023 [−.063, +.013] | .32 | |

Reading (provisional, one cell): about two-thirds of g10's +.037 comes back when the gold passages share no chunk.
So the gain is consistent with fewer chunks / fewer distractor "documents" in the prompt, not with context between
the gold passages. Still running: the golds joined with distractors alone (job 188471), all 10 joined in a shuffled
order (188473), and the g1 re-run on el9_8 (188486). g1 was recorded on an older-image node; this cell is text, where
no node effect has been seen, but 188486 checks it.

## Arms in flight (no result yet)
| arm | cells | what it answers | training decision it feeds |
|---|---|---|---|
| A. use-recovery on documents | seed 42: evidence-then-answer (running, 800/1,500), gold page only, picked page raw, slots in int8 (queued). Seed 43: gold page only, picked page raw (running) | how much of the −.20 vs raw comes from reading vs picking vs using | evidence-then-answer labels in stage 2; route-then-expand at serving |
| C. reasoning vs reading on text | compressed plain (done once: .718, below), compressed evidence-then-answer (running, 450/1,500), uncompressed evidence-then-answer (running, 1,080/1,500) | is the text loss (DROP, DuoRC, RACE) room-to-reason or missing content | evidence-then-answer teacher labels for text |
| D. page picking without training | both attention re-runs done (controls exact); fusion of digit + attention scores next | can the decoder's attention beat the digit route prompt with no training | whether routing rows need training |
| E. attention-guided slot pruning | model 184778 documents, keep half the slots: first half of each chunk, every 2nd slot, most-attended half of each chunk, most-attended half overall. Starts after a keep-all control (300 questions, must give the plain answers) | can the decoder's own attention choose which half of the slots to keep (a coarse level with no training) | whether a coarse memory level needs training |

First number of arm C (provisional: one complete run, file `expQ/probes/c_plain_s42/eval/attempt2_0357_*.json`): the
reference model, compressed, MRQA (6 sources × 250 questions, one passage each, nothing cut), 80,003 slots in total:

| source | n | compressed judge accuracy ↑ (184778) | uncompressed judge accuracy ↑ (185542) | gap |
|---|---|---|---|---|
| all | 1,500 | .718 | .807 | −.089 |
| DROP (reasoning) | 250 | .580 | .784 | −.204 |
| DuoRC (narrative) | 250 | .600 | .684 | −.084 |
| RACE (narrative) | 250 | .528 | .680 | −.152 |
| BioASQ | 250 | .912 | .924 | −.012 |
| RelationExtraction | 250 | .840 | .896 | −.056 |
| TextbookQA | 250 | .848 | .872 | −.024 |

## New eval sets and options
- `eval_data/probes_hotpot_{gold_joined,distr_joined,g10_shuf}`, 1,500 rows each, built by `scripts/probes/build_groups.py`
  (job 188465). The builder first rebuilds the old one-chunk set `text_hotpotqa_1500_g10` and matches it on 1,500/1,500 rows.
- `scripts/eval_mm_docqa.py --mem_keep_mode attn | attn_global` (launcher env `KEEPF=0.5 KEEPM=attn`): one extra
  forward pass reads the decoder's attention from the last prompt position to every slot (mean over all heads of the
  8 full-attention layers), then keeps the most-attended slots, with the same counts as `first`. Invariant tests 45/45
  (job 188479).

## Findings
(none adopted yet; arm B's first cell is provisional, above)

## Waiting on a new eval set
- Photo text vs pixels (arm F) cannot run on the existing photo set: its rows carry no OCR text (`docs=[""]`). It needs
  a new set with TextVQA's OCR tokens (500 questions) before it can run.

## Closed
(none yet)
