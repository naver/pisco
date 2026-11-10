# STATE — campaign probes   (rewritten WHOLESALE by every LLM wake; never append)
last wake: 2026-11-11 04:30-04:45 KST (woken by "problems": 188443 finished, 188441 preempted)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- The instrument reproduces on nodes with the same system image (el9_8). Reference model 184778, documents,
  PX=1003520: re-run on homer-011 (node of the recorded run 184781) gives judge .610 = .610, 1,500/1,500 answers
  identical, 0 discordant (job 188419, closed). Re-run on homer-016: first 300 questions, 300/300 answers identical,
  judge .610 / EM .477 both, mems_used 118,147 both (job 188476, closed).
- Arm D's controls are recovered exactly. Model 183814, 10-page documents (retr10), n=1,500: digit route pick@1 .861,
  attention .885, every summary number equal to 185577 (job 188442, closed). HotpotQA, n=1,500: digit .922,
  attention .873, equal to 185580 (job 188443, closed). Fusion inputs (digit log-probs + masses) written for both.
- Invariant tests pass with this wake's eval patch (the attention-ranked slot keep modes): 45/45 tests OK
  (test_positions 12, test_mm_collators 33), exit 0 (job 188479, closed).
- Data for arm B exists and is checked: `scripts/probes/build_groups.py` rebuilds `text_hotpotqa_1500_g10` on
  1,500/1,500 rows identical, then writes three grouped sets of 1,500 rows (job 188465, closed).
- plumbing only: fusion selftest reproduces route .8607 / attention .8853 at n=1,500 (188427); unit tests 45/45 for
  the expand-in-base patch (188455); the uncompressed reader's default path gives 300/300 answers identical to 185542
  (188456). All closed.
- provisional: arm B, 2 gold passages each alone + the 8 distractors joined (3 chunks), model 183814, HotpotQA
  1,500 questions: judge .604 vs g1 (10 chunks) .579 = +.025 [+.005, +.044], McNemar p=.012; vs g10 (1 chunk) .617 =
  −.013 [−.037, +.011], p=.33. unparseable 0, 1,500 rows, mems_used 261,308 (0.983 × g1, 1.005 × g10). Job 188472,
  grid `probe judge=0.604`. One cell; g1 ran on an el9_7 node (re-run 188486 on el9_8 queued).
- provisional: on another image the instrument moves for image inputs: homer-008 (el9_7), first 300 document
  questions: 57/300 answers differ, judge .623 vs .610, 9 up / 5 down (McNemar p≈.42: the judge gap itself is noise,
  the 57 changed answers are the signal), EM .490 vs .477 (job 188458, closed). One 300-question cell; 188477
  (homer-007) separates "the image" from "homer-008 alone".
- provisional: the compressed reference model on MRQA text, one chunk per passage: judge .718, n=1,500,
  unparseable 0, mems_used 80,003 (read from the result file), DROP .580, DuoRC .600, RACE .528 (250 each) vs the
  uncompressed reader .807 (185542; DROP .784, DuoRC .684, RACE .680). Job 188445 attempt 2 (homer-016), copy
  `expQ/probes/c_plain_s42/eval/attempt2_0357_*.json`; grid `probe judge=0.718`.

**Open, and running** (19 cells at 04:41: 8 running, 11 queued)
- arm A (use-recovery on documents), 1,500 questions each. Seed 42, model 184778, baseline .610 (184781, homer-011):
  evidence-then-answer (188439, running on homer-016), gold page only (188440, queued), picked page raw (188441,
  queued after a preemption at 04:28), slots in int8 (188478, queued off el9_7). Seed 43, model 184780, baseline
  .615 (184784, homer-016): gold page only (188459, running on homer-013), picked page raw (188460, running on
  homer-020). All these nodes are el9_8, so they pair with their baselines.
- arm B (long-chunk mechanism), model 183814: golds joined (188471, running on homer-016), all 10 joined in a
  shuffled order (188473, running on homer-016), g1 re-run on el9_8 (188486, queued).
- arm C (reasoning vs reading on text), MRQA: compressed plain (188445, re-running on homer-016; attempt-2 file is
  the result), compressed evidence-then-answer (188446, running on homer-016), uncompressed evidence-then-answer
  (188457, running on homer-011, near the end).
- arm D: fusion (CPU) still to launch — the chained job 188444 was killed with the duplicate 188442 re-run.
- arm E (attention-guided slot pruning), model 184778, documents, PX=1003520, off el9_7: keep-all identity control
  (188480, 300 questions) and post-patch plain regression (188481, 300 questions), then four 1,500-question cells
  afterok on 188479+188480+188481: keep half by first (188482), every 2nd (188483), attn per chunk (188484),
  attn over all pages (188485).
- node check homer-007 (188477, pinned, waiting for that node).

## What happened this wake
- Diagnosed the two problems. 188443: completed, control exact (above). 188441: PREEMPTED at 04:28 after 32 min on
  homer-016, requeued, no file written → benign (known issue), no action.
- 188442 (retr10 attention route) was PREEMPTED at 04:38:47 seconds after writing all three output files
  (known preemption race). Files copied to `attempt1_0438_*`; the requeued re-run (41 min, could land on el9_7 and
  overwrite image-input masses) and its afterok fusion job 188444 killed with `expmon kill`, both verdict closed.
- 188476 decided: el9_8 nodes agree (300/300). Arm A cells on homer-013/016/020 pair with their baselines.
- Arm B first cell analysed (provisional above). The gate in the ledger ("distr_joined − g1 ≥ +.02 with CI above 0 →
  fewer chunks helps by itself") is met once.
- Arm E started. Patch to `scripts/eval_mm_docqa.py` (patchable path, 38 added lines, one flag):
  `--mem_keep_mode` gains `attn` / `attn_global`. New `mem_attention()`: one eager forward of the full prompt, hook
  on every full-attention module, attention of the last prompt position on each `<MEM>` (mean over heads × layers);
  attention kernel set via `_attn_implementation_internal` on every module config and restored, `rope_deltas`
  saved and restored. Keep counts = `first`'s (`max(1, int(n·frac))` per chunk; attn_global keeps the same total over
  all chunks). Default path unchanged: the branch runs only when `--mem_keep_frac` is set; `even` now an explicit
  `elif`. Launcher `eval_mm_4b.sh` already passes `KEEPF`/`KEEPM` (file suffix `_keep05_attn`).
  Tests (188479, run_tests_cpu.sh on gpu-be gres:0):
  ```
  === test_positions.py ===  Ran 12 tests in 0.210s  OK
  === test_mm_collators.py ===  Ran 33 tests in 22.350s  OK
  === EXIT CODE 0 ===
  ```
  The law's LIMIT=300 plain regression is 188481 (queued); the identity control 188480 must give 300/300 the plain
  answers (pass 1 runs, all slots kept).
- Arm E gate (in the ledger before launch): attn or attn_global minus the better of first/even ≥ +4 pt with CI above
  0 → attention picks the slots that matter (a coarse level with no training; then seed 43 + photos); within .02 →
  closes attention-guided pruning.
- CAMPAIGN.md tightened: Same node rule now records 188476 PASS and requires `--exclude=homer-[005-009]` on EVERY
  cell, text too (homer-008 took 14 of the 18 campaign preemptions, sacct). known_issues: the preemption-race row
  now says to kill the requeued duplicate and its afterok dependents.
- Deliverable `docs/probe_campaign_2026-11.md` rewritten: instrument table + homer-016 + arm D re-runs, arm B table
  with the metric columns, arm E description.
- 8 submissions (cap 8): 188479 tests, 188480 identity control, 188481 plain regression, 188482-188485 arm E,
  188486 g1 re-run. Gripes: preempted-after-result requeue (`expmon settle` idea); a node-image filter + node column.

## Owned jobs (jobid | arm | state at 04:41 | node | note)
| 188439 | A expand s42 | RUNNING | homer-016 | `expQ/probes/a_expand_s42/eval/`; --time 5:12:00 |
| 188440 | A gold s42 | PENDING | — | `a_gold_s42`; check its node (sacct -D) before pairing |
| 188441 | A route1 s42 | PENDING (preempted 04:28) | — | `a_route1_s42`; DML 6144 |
| 188478 | A int8 s42 | PENDING | excl. 005-009 | `a_mq8_s42`; mems_used must equal 646,042 |
| 188459 | A gold s43 | RUNNING | homer-013 | `a_gold_s43`; baseline 184784 (homer-016) |
| 188460 | A route1 s43 | RUNNING | homer-020 | `a_route1_s43` |
| 188471 | B golds joined | RUNNING | homer-016 | `b_gold_joined`; vs g1 / g10 |
| 188473 | B g10 shuffled | RUNNING | homer-016 | `b_g10_shuf`; vs g10 185601 |
| 188486 | B g1 re-run | PENDING | excl. 005-009 | `b_g1_rep`; 1,500/1,500 identical to 185505 → text is node-proof |
| 188445 | C plain s42 | RUNNING (4th start) | homer-016 | attempt-2 file is the result; re-run should give the same bytes |
| 188446 | C expand s42 | RUNNING | homer-016 | `c_expand_s42` (suffix `_cml4096_expand`) |
| 188457 | C base expand | RUNNING | homer-011 | `expQ/base_qwen35_4b/eval/mm_docqa_base_px1003520_expand_text_mrqa_ood_1500.json` |
| 188480 | E identity control | PENDING | excl. 005-009 | `e_attn_ctrl`, KEEPF=1.0 KEEPM=attn LIMIT=300; must equal 184781's first 300 |
| 188481 | E plain regression | PENDING | excl. 005-009 | `reg_e_plain`, LIMIT=300; EM within .01 of .477 |
| 188482 | E keep half, first | PENDING (afterok) | excl. 005-009 | `e_keep05_first` |
| 188483 | E keep half, every 2nd | PENDING (afterok) | excl. 005-009 | `e_keep05_even` |
| 188484 | E keep half, attn per chunk | PENDING (afterok) | excl. 005-009 | `e_keep05_attn` |
| 188485 | E keep half, attn overall | PENDING (afterok) | excl. 005-009 | `e_keep05_attng` |
| 188477 | node check | PENDING (homer-007 busy) | homer-007 pinned | `reg_h007`: vs 184781 AND vs 188458 |

## Adopted jobs
(none. 188421/188422 and 188466-188470 belong to the sibling `longchunk` campaign.)

## Open arms and the next decision for each
| arm | status | next action |
|---|---|---|
| node checks | homer-016 PASS; homer-007 queued | 188477 identical to 188458 → the older image is the cause (keep the exclude); identical to 184781 → homer-008 alone. Rewrite the Same node rule with the result |
| A. use-recovery | s42 ×4, s43 ×2 open | per cell: node (sacct -D), health, `scripts/paired_judge.py` vs the same seed's plain file. s43 expand only if s42 expand is a tie or a gain |
| B. long-chunk mechanism | 1 of 3 cells done (+.025 vs g1) | after 188471 / 188473 / 188486: pair each vs g1 and g10 (use 188486 as g1 if it differs from 185505); then the tail control 183538 only for the grouping that decides |
| C. reasoning vs reading | plain done once, 2 open | per-source gap compressed − raw on DROP / DuoRC / RACE, plain vs expand; gate ≥ 5 pt shrink. Compare 188445's re-run file to the attempt-2 copy. Thinking mode: no flag yet; decide after expand |
| D. page picking | controls exact; fusion not run | relaunch `scripts/probes/route_fusion.sbatch expQ/probes/attn_route_188442 expQ/probes/attn_route_188443` (no dependency, `--time 0:20:00`); gate pick@1 ≥ +2 pt on retr10 AND no loss on hotpot; then `RAW=route1 RANK=fused_<picker>.json` |
| E. attention-guided pruning | 6 cells queued | when 188480 ends: 300/300 identical to 184781's first 300? If not, `expmon kill` 188484 + 188485 (the eager pass perturbs pass 2) and debug. Then pair attn / attn_global vs the better of first / even |
| F. photo text vs pixels | blocked on data | build `eval_data/probes_textvqa_ocr` (500 TextVQA rows + `ocr_tokens` from lmms-lab/textvqa) via a SLURM job |

## Health and budget
- budget: 33 of 60 cells committed (25 before this wake + 8). GPU-hours of 40: the tick's sensor reports 1.4 GPU-h
  in the last 24 h; the open cells add ≈ 15 from their measured walls.
- gpu-be preemptions: 18 of campaign cells since 03:22, 14 on homer-008. Every new cell now excludes homer-005..009.

## Next wake
`wake-when` on the first group leaving the queue: arm C (188445, 188446, 188457); arm E controls (188480, 188481);
arm B (188471, 188473, 188486); arm A seed 42 (188439-188441, 188478); arm A seed 43 (188459, 188460); heartbeat 2 h
(fusion relaunch is pending).

## Blockers / needs the owner
- none blocking. FYI: `eval_mm_4b.sh`, `probe_attn_route.sh`, `run_tests_cpu.sh` are not under git (`.gitignore` has
  `*.sh`); force-add them when convenient. The ALERT lines about the 01:00 / 01:30 failed wakes are yours to clear
  (cause: the account's 5-hour usage limit, known_issues row).
