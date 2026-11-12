# STATE — campaign probes   (rewritten WHOLESALE by every LLM wake; never append)
last wake: 2026-11-13 06:30-06:50 KST (woken by the 4-hour cap of the last wait; arm F decided; two queued cells
widened to `gpu,gpu-be`, one of them started at once)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Showing the routed page raw recovers the whole document gap, at both seeds. Reference models, documents
  (`mm_docqa_1500_1m`, 4 pages, PX=1003520), n=1,500 each: seed 42 judge .820 vs plain .610 = +.210 [+.187, +.233],
  350 up / 35 down, McNemar p<1e-60, slots .749× plain + one raw page (job 188441, verdict best); seed 43 .822 vs .615
  = +.207 [+.183, +.232], 349 / 38, slots .748× + one raw page (job 188460, verdict best). The pick is right on 98.6% /
  98.9% of questions; the all-raw reader scores .809.
- Twice the slots on the picked page is NOT enough (arm G closed). Picked page re-compressed at rate 4, other pages
  rate 8, documents, n=1,500 each, slots 1.250× plain: seed 42 .673 vs plain .610 = +.063 [+.043, +.083], 158 / 64,
  p<.001 (job 188590, homer-011 = baseline's node, verdict closed); seed 43 .663 vs .615 = +.049 [+.030, +.067],
  147 / 74, p<.001 (job 188591, homer-012 el9_8 vs baseline homer-016 el9_8, verdict closed). Gate ≥ +.14 failed at
  both seeds. Vs the raw page: −.147 [−.169, −.126] (s42), −.159 [−.180, −.137] (s43). Extra decoder prompt tokens
  per question vs plain: +107.5 (rate 4) vs +752.9 (raw page) = .143×. The models were trained at rate 8 only.
- Removing the distractor pages instead gains only about 2 pt, below the +3 pt gate at both seeds: gold page only,
  compressed, slots .270×, seed 42 .632 vs .610 = +.022 [+.005, +.040], p=.020 (job 188440); seed 43 .639 vs .615 =
  +.025 [+.006, +.043], p=.010 (job 188459).
- Slots stored in int8 change nothing, at both seeds: seed 42 .611 vs .610 = +.001 [−.003, +.006], 1,445/1,500
  identical (job 188478); seed 43 .613 vs .615 = −.001 [−.005, +.003], 4 / 6, 1,444/1,500 identical, n=1,500 (job
  188536, verdict best; homer-011 vs baseline homer-016, both el9_8). Adopted as a serving option.
- Given the question, the most-attended half of the slots carries what the decoder uses, at both seeds (arm E
  adopted). Documents, n=1,500 each, slots .496×: seed 42 .610 vs .610, +.000 [−.009, +.010] (job 188485, homer-016
  vs baseline homer-011, both el9_8); seed 43 .615 vs .615, +.000 [−.009, +.009], 25 / 25 (job 188522, verdict best).
  Blind every-2nd half at the same slot count: seed 42 .505 (job 188483), seed 43 .508 (job 188537, verdict closed),
  so attention minus blind = +.105 [+.085, +.126], 205 / 47, p=1e-24 and +.107 [+.087, +.127], 203 / 43, p=5e-26.
  Gate (≥ +4 pt, CI above 0, both seeds) met.
- The long-chunk gain is context between the two answer passages, at both seeds. HotpotQA, 1,500 Q each, golds
  joined in one chunk (others alone) vs ten separate chunks on today's code: seed 42 (183814) .655 vs .582 = +.073
  [+.051, +.095], p=1.5e-10 (jobs 188471 vs 188486); seed 43 (183816) .669 vs .601 = +.067 [+.046, +.088], 186 / 85,
  p=8e-10 (jobs 188511 vs 188509, verdict best). Slots .998× vs ten chunks. Vs one chunk of all ten (slots 1.020×):
  seed 42 +.038 [+.015, +.061], 185 / 128, p=.002 (vs 188518); seed 43 +.053 [+.029, +.076], p=9e-6 (vs 188510).
- One chunk of all ten vs ten chunks is above the band at seed 42 only: seed 42 .617 vs .582 = +.035 [+.012, +.059],
  184 / 132, p=.004, slots .978× (job 188518, verdict closed; 1,500/1,500 answers identical to the 11-07 file 185601);
  seed 43 .615 vs .601 = +.014 [−.008, +.035], p=.23 (job 188510).
- The no-oracle grouping is not adopted. Top 3 passages by 183814's attention joined, others alone: seed 42 .614 vs
  .582 = +.032 [+.009, +.056], p=.007 (job 188521, verdict closed); seed 43 .615 vs .601 = +.013 [−.007, +.034],
  139 / 119, p=.24, n=1,500 (job 188546: verdict closed in verdicts.jsonl, no grid row — known issue; its file
  `expQ/probes/b_pred3_attn_s43/eval/` has 1,500 rows, judge .6147). Against one chunk of ten: −.003 (s42), −.001 (s43).
- Passage order inside a long chunk is unused, both seeds on today's code: seed 42 .615 vs .617 = −.002 [−.023,
  +.020], 141 / 144, p=.91 (188473 vs 188518); seed 43 .618 vs .615 = +.003 [−.021, +.026], p=.87 (188513 vs 188510).
- Joining the eight distractor passages is not adopted: seed 42 .604 vs .582 = +.022 [+.003, +.041], p=.027 (job
  188472 vs 188486); seed 43 .617 vs .601 = +.016 [−.003, +.035], p=.11 (job 188512 vs 188509).
- Blind slot halving on 184778 (documents, 1,500 Q): every 2nd slot (slots .504×) −.105 [−.127, −.085] (job 188483);
  first half of each chunk (slots .496×) .495, −.115 [−.136, −.095] (job 188482).
- Arm F decided (reported only; the law allows seed 42 only, its seed-43 baseline ran on homer-008): reading photo
  text as OCR words instead of pixels is far worse. Model 184778, TextVQA, 500 questions paired on query_id with the
  TextVQA rows of the pixel cell 184783 (homer-011): judge .346 vs .780 = −.434 [−.480, −.384], 17 up / 234 down,
  McNemar p=6e-50; EM .292 vs .714; slots .081× (4.4 vs 54.2 per question); health unparseable 0, 500/500 rows,
  homer-012 (job 189059, verdict closed). The gold answer is in the OCR words on only .596 of rows (job 188545).
  Gate (gain ≥ 5 pt motivates a mixed-input arm) not met → no mixed-input arm.
- Arm G flag checks: first version's rate-8 control FAILED, 233/300 identical (job 188525, verdict closed); patched
  version (page re-compressed inside its example's padded batch) PASSED on homer-013 (job 188580, verdict closed):
  tests 12/12 + 33/33 + 5/5, plain 300/300, rate-8 control 300/300 identical to plain (judge .610 = .610, slots 118,147).
- Data: three grouped HotpotQA sets (job 188465), six no-oracle grouped sets (job 188516), 1,500 rows each;
  `eval_data/probes_textvqa_ocr` (job 188545): 500 TextVQA rows, 500/500 questions matched.

**Provisional (one seed, or a tie the law wants repeated)**
- provisional (a tie at one seed; its seed-43 repeat 189058 is RUNNING): evidence-then-answer on documents gains
  nothing at seed 42: .611 vs .610 = +.001 [−.017, +.021], 108 / 106, p=.95, n=1,500, slots 1.000× (job 188439,
  verdict closed; file from the homer-011 attempt, same node as 184781).
- provisional (one seed; its seed-43 repeat 189056 is queued): keeping only the most-attended QUARTER still ties
  plain. 184778, documents, n=1,500, slots .246×: .611 vs .610 = +.001 [−.009, +.012], 37 / 35, p=.91, 1,197/1,500
  identical (job 188538, verdict best, homer-013).
- provisional (one seed, no blind control on photos): the attention half ties plain on photos. 184778,
  `mm_naturalqa_1500`, n=1,500, slots .492×: .677 vs .677, +.000 [−.013, +.013], 49 / 49 (job 188539, verdict closed).
- provisional: the instrument gives the same answers on re-run on every node tested except homer-008: plain document
  cell 1,500/1,500 on homer-011 (job 188419); first 300 questions 300/300 on homer-016 (188476, 188524), homer-007
  (188477) and homer-013 (188580); compressed text 1,500/1,500 between homer-016 and homer-007 (188471 rerun) and
  between 11-07 and 11-11 code (188518 vs 185601); homer-008 changed 57/300 image answers (188458). homer-006 is
  suspected (by elimination) of the 55/1,500 compressed-text drift of 185505 vs 188486; homer-005 and 009 are untested;
  homer-012 (el9_8) carries 188522, 188537, 188546, 188591, 189059 without a check of its own. All el9_7 nodes are
  excluded.
- provisional: page picking (arm D, model 183814, n=1,500 per set, job 188515): 10-page documents digit .861,
  attention .885 (+.025 [+.010, +.039]); HotpotQA digit .922, attention .873. The per-source switch that passes the
  gate was chosen on the same questions. Expected judge gain ≈ +.006 to +.008 (under the .02 band), no judge cell.
- provisional, one model: the cross-passage gain needs no spread layout. Tail-layout model 183538, golds joined
  .627 vs ten chunks .569 = +.058 [+.035, +.082], 202 / 115, p<.001, slots .998× (jobs 188514 vs 188519, verdict
  best). The grid lists 188514's parent as 185505 (a label from its launch), its baseline is 188519.
- provisional: evidence-then-answer does not close the text gap at seed 42 (MRQA, n=1,500): compressed .699 vs plain
  .718 (−.019 [−.041, +.003], p=.11, job 188446); uncompressed .807 = .807 (job 188457). The law's ".695" baseline is
  model 183814 with 256-token chunks (job 185775); the arm uses the reference model 184778 with one chunk per passage
  (job 188445, .718).
- provisional (not a gate, a wake-designed ratio from the two arm G cells, no CI, no control): per extra decoder token,
  rate 4 on the picked page recovers more than the raw page does: seed 42 +.063 / 107.5 tokens ≈ .059 per 100 tokens
  vs raw +.210 / 753 ≈ .028 (about 2×, jobs 188590 / 188441); seed 43 .049 / 107.5 ≈ .046 vs .207 / 753 ≈ .027
  (about 1.7×, jobs 188591 / 188460).

**Open, and running** (3 cells of the raised cap, submitted 2026-11-12 23:33; each preempted on `gpu-be` 1-2 times
before writing anything)
- arm A, evidence-then-answer at seed 43 (job 189058) — RUNNING since 06:34 on partition `gpu` (not preemptible),
  homer-011; the law's repeat after the seed-42 tie. Slowest cell of this kind 104 min → ends by about 08:20.
- arm E, keep a quarter of the slots at seed 43 (job 189056) — PENDING on `gpu,gpu-be`, start estimate 11-13 14:49;
  decides whether "three quarters of the slots go unused per question" holds at both seeds.
- arm E, keep an eighth of the slots at seed 42 (job 189057) — PENDING on `gpu-be`, start estimate 11-13 16:30;
  finds where slot dropping starts to cost.

## What happened this wake
- Arm F's cell 189059 COMPLETED (attempt 2, homer-012, 02:40-02:47; attempt 1 on the same node also completed at
  02:20 before its preemption, so the file was rewritten by the same node). Paired with `scripts/paired_judge.py`
  against the 500 TextVQA rows of 184783; recorded `expmon probe 189059 judge=0.346 em=0.292`, verdict closed, ledger
  note; arm F row of CAMPAIGN.md marked DECIDED (it first read the pixel baseline as .677, the whole photo set; the
  TextVQA rows score .780).
- Queue: the three other cells had pended 6.5-7 h (more than the law's 6 h) with start estimates up to 11-18.
  `gpucap land a100 1`: owner share on `gpu` 6 of 8, 1 free non-excluded node there. Widened the two cells that each
  decide a claim at both seeds (189056, 189058) with `scontrol update … Partition=gpu,gpu-be` (the law's limit is 2;
  the last STATE named the two arm E cells, changed because 189057 is one seed only and had the earliest estimate).
  189058 started on `gpu` within a minute; 189056's estimate moved from 11-18 01:00 to 11-13 14:49. Owner `gpu` share
  now 7 of 8. New known_issues row.
- Deliverable: `docs/probe_campaign_2026-11.md` rewritten (status of 11-13, finding 8, arm F table, closed list);
  `expmon display --exp probes` refreshed (still 0 table rows: known issue, gripe already filed).
- Efficiency lines of the brief: "waiting for the owner 72 %" was the cap wait of 11-12 (answered 23:14), nothing open
  for the owner now; "25 % queued with none running" — acted on: `gpucap land` before the widening, as the line asks.
- Verifier: 0 unsupported; its 8 provisional items stay provisional (unchanged).
- Gripe filed: the brief's "new result files" list shows old non-campaign LongBench files, not the campaign's own.

## Owned jobs (jobid | cell | arm | state at 11-13 06:45 | intent — also in the ledger)
| job | cell dir (`expQ/probes/…`) → model | arm | state | intent |
|---|---|---|---|---|
| 189058 | `a_expand_s43` → 184780 | A | RUNNING on `gpu`, homer-011 (el9_8), since 06:34 | cell 64. `STYLE=expand PX=1003520`, documents, 1,500 Q. Baseline plain 184784 .615 (homer-016, el9_8; el9_8 nodes agree, 188476). Gate ≥ +3 pt with CI above 0 at both seeds (s42 +.001). `--time 5:12:00` = slowest expand docs cell (174852, 104 min) × 3 |
| 189056 | `e_keep025_attng_s43` → 184780 | E | PENDING `gpu,gpu-be` (preempted twice, homer-013), est. 14:49 | cell 61. `KEEPF=0.25 KEEPM=attn_global PX=1003520`, `mm_docqa_1500_1m`, 1,500 Q, judge chained. Baseline plain 184784 .615. Gate (written 2026-11-11 before the s42 cell): within .02 of plain at both seeds → three quarters of the slots unused per question; a loss ≥ .05 → the free share lies between 1/2 and 3/4. `--time 4:54:00` |
| 189057 | `e_keep0125_attng` → 184778 | E | PENDING `gpu-be` (preempted once, homer-013), est. 16:30 | cell 62. `KEEPF=0.125 KEEPM=attn_global PX=1003520`, documents, 1,500 Q. Baseline plain 184781 .610 (homer-011). Gate (written 2026-11-11 before any cell): within .02 → seven eighths unused; a loss ≥ .05 → the free share lies between 3/4 and 7/8. One seed = provisional. `--time 4:54:00` |
| 189059 | `f_text_s42` → 184778 | F | COMPLETED 02:47, decided | see Settled |

## Adopted jobs
(none. 188466, 188707-188712 belong to the sibling `longchunk` campaign; 189094, 189095-189099, 189434, 189445-189450
to the sibling `freeze` campaign.)

## Open arms and the next decision for each
| arm | status | next action |
|---|---|---|
| A. use-recovery | route1 adopted, int8 adopted, gold-only closed; evidence-then-answer a tie at s42 | when 189058 ends: health, node of the attempt that wrote the file, pair with 184784, decide the arm at both seeds |
| B. long-chunk mechanism | closed | none |
| C. reasoning vs reading | closed for budget | none |
| D. page picking | decided, caveat: rule chosen in-sample | none |
| E. attention-guided pruning | half adopted (both seeds); quarter met at s42 | when 189056 ends: pair with 184784 → quarter at both seeds; when 189057 ends: pair with 184781 → eighth gate (one seed, provisional) |
| F. photo text vs pixels | decided 11-13 (−.434), closed | none |
| G. route then densify | closed (both seeds below the gate) | none |

## Health and budget
- cells: 64 of 65 used (61-64 = 189056-189059); the 65th is the ONE health relaunch (a cell failing `unparseable: 0`,
  its row count, or its slot ratio). A preemption requeue is free. Nothing else may launch without the owner.
- GPU-hours of 40: tick sensor 0.1 GPU-h in the last 24 h; the three open cells add about 3-5 GPU-h.
- Widened cells: 189056, 189058 (the law's limit of 2 is used). When 189058 ends, 189057 may be widened if it still
  pends, provided the owner's `gpu` share has room (sibling jobs counted, never more than 2 campaign cells on `gpu`).
- Pairing nodes: every image cell needs its node from the attempt that wrote its file (same node rule; compare the
  `eval/` file mtime with `sacct -j <id> -D -o NodeList,End`, preemption often races the end).

## Next wake
- NEXT = `wake-when job:189058 job:189056,189057 max 12h`: wakes when the running arm A cell leaves the queue (about
  08:20), or when both arm E cells have, or at 18:45 at the latest.
- On wake: for each finished cell, `unparseable: 0`, row count 1,500, node of the attempt that wrote the file,
  `scripts/paired_judge.py` against its baseline (CI, McNemar), slot ratio from `mems_used`; then `expmon verdict`,
  `expmon probe <id> judge=…`, one ledger note per decided arm, rewrite `docs/probe_campaign_2026-11.md`, refresh
  `expmon display --exp probes`. If 189057 still pends then, widen it (see Health). Then `wake-when` the remaining
  cells, max 12 h.
- When all three are decided and no health relaunch is needed: every arm is decided or closed for the cap → write
  `done` and a close-out summary for the owner (the window runs to 2027-02-09; close-out steps in CAMPAIGN.md).

## Blockers / needs the owner
- none.

## Proposals for the sibling `longchunk` campaign (training is not this campaign's job)
- Stage-1/2 data that packs related passages in one chunk (both gold passages of a multi-hop question) rather than
  random long chunks: +.067-.073 with the golds together (both seeds), +.058 on the tail layout. Joining all ten:
  +.035 (seed 42) / +.014 n.s. (seed 43). A ranker's top 3 joined does no better than all ten.
- A query-aware slot budget (nested-budget training): half the slots, picked by the decoder's attention with the
  question, lose nothing at both seeds; a quarter loses nothing at seed 42 (seed 43 queued).
- A mixed-rate stage 2 (the picked page at rate 4, others at rate 8): untrained, rate 4 on the picked page gains
  +.063 / +.049 at 1/7 of the raw page's extra decoder tokens; whether training at rate 4 closes more of the raw
  page's +.21 is open.
- Not proposed: OCR text as an extra input for photos (arm F: OCR words instead of pixels −.434 on TextVQA).
