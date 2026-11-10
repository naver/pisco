# STATE — campaign probes   (rewritten WHOLESALE by every LLM wake; never append)
last wake: 2026-11-11 07:30-07:55 KST (scheduled; the ALERT was the verifier's 06:00 note, fixed below)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Showing the routed page raw recovers the whole document gap, at both seeds. Reference models, documents
  (`mm_docqa_1500_1m`, 4 pages, PX=1003520), n=1,500 each: seed 42 judge .820 vs plain .610 = +.210 [+.187, +.233],
  350 up / 35 down, McNemar p<1e-60 (job 188441, verdict best); seed 43 .822 vs .615 = +.207 [+.183, +.232], 349 / 38
  (job 188460, verdict best). The pick is right on 98.6% / 98.9% of questions; the all-raw reader scores .809.
- Removing the distractor pages instead gains only about 2 pt, below the +3 pt gate at both seeds: gold page only,
  compressed, seed 42 .632 vs .610 = +.022 [+.005, +.040], p=.020 (job 188440); seed 43 .639 vs .615 = +.025
  [+.006, +.043], p=.010 (job 188459).
- Evidence-then-answer on documents gains nothing at seed 42: .611 vs .610 = +.001 [−.017, +.021], 108 / 106, p=.95,
  n=1,500, slots 1.000× (job 188439, verdict closed; file written by the homer-011 attempt, same node as 184781).
- Slots stored in int8 change nothing, at both seeds: seed 42 .611 vs .610 = +.001 [−.003, +.006], 1,445/1,500
  identical (job 188478); seed 43 .613 vs .615 = −.001 [−.005, +.003], 4 / 6, 1,444/1,500 identical, n=1,500 (job
  188536, verdict best; homer-011 vs baseline homer-016, both el9_8). Adopted as a serving option.
- Given the question, the most-attended half of the slots carries what the decoder uses, at both seeds (arm E
  adopted). Documents, n=1,500 each, slots .496×: seed 42 .610 vs .610, +.000 [−.009, +.010] (job 188485); seed 43
  .615 vs .615, +.000 [−.009, +.009], 25 / 25 (job 188522, verdict best). Blind every-2nd half at the same slot count:
  seed 42 .505 (job 188483), seed 43 .508 (job 188537, verdict closed), so attention minus blind = +.105 [+.085,
  +.126], 205 / 47, p=1e-24 and +.107 [+.087, +.127], 203 / 43, p=5e-26. Gate (≥ +4 pt, CI above 0, both seeds) met.
- provisional (one seed): keeping only the most-attended QUARTER still ties plain. 184778, documents, n=1,500, slots
  .246×: .611 vs .610 = +.001 [−.009, +.012], 37 / 35, p=.91, 1,197/1,500 identical (job 188538, verdict best,
  homer-013). Seed 43 not run (cell cap: BLOCKED.md asks the owner).
- provisional (one seed): the attention half holds on photos. 184778, `mm_naturalqa_1500`, n=1,500, slots .492×:
  .677 vs .677, +.000 [−.013, +.013], 49 / 49 (job 188539, verdict closed).
- The long-chunk gain is context between the two answer passages, at both seeds. HotpotQA, 1,500 Q each, golds
  joined in one chunk (others alone) vs ten separate chunks on today's code: seed 42 (183814) .655 vs .582 = +.073
  [+.051, +.095], p=1.5e-10 (jobs 188471 vs 188486); seed 43 (183816) .669 vs .601 = +.067 [+.046, +.088], 186 / 85,
  p=8e-10 (jobs 188511 vs 188509, verdict best). Slots .998×. Vs one chunk of all ten: seed 42 +.038 [+.015, +.061],
  185 / 128, p=.002 (vs 188518); seed 43 +.053 [+.029, +.076], p=9e-6 (vs 188510).
- One chunk of all ten vs ten chunks is above the band at seed 42 only: seed 42 .617 vs .582 = +.035 [+.012, +.059],
  184 / 132, p=.004, slots .978× (job 188518, verdict closed; 1,500/1,500 answers identical to the 11-07 file 185601);
  seed 43 .615 vs .601 = +.014 [−.008, +.035], p=.23 (job 188510).
- The no-oracle grouping is not adopted. Top 3 passages by 183814's attention joined, others alone: seed 42 .614 vs
  .582 = +.032 [+.009, +.056], p=.007 (job 188521, verdict closed); seed 43 .615 vs .601 = +.013 [−.007, +.034],
  139 / 119, p=.24, n=1,500 (job 188546, verdict closed). Against one chunk of ten it is −.003 (s42) and −.001 (s43).
- Passage order inside a long chunk is unused, both seeds on today's code: seed 42 .615 vs .617 = −.002 [−.023,
  +.020], 141 / 144, p=.91 (188473 vs 188518); seed 43 .618 vs .615 = +.003 [−.021, +.026], p=.87 (188513 vs 188510).
- Joining the eight distractor passages is not adopted: seed 42 .604 vs .582 = +.022 [+.003, +.041], p=.027 (job
  188472 vs 188486); seed 43 .617 vs .601 = +.016 [−.003, +.035], p=.11 (job 188512 vs 188509).
- Blind slot halving on 184778 (documents, 1,500 Q, slots .50×): every 2nd slot −.105 [−.127, −.085] (job 188483);
  first half of each chunk .495, −.115 [−.136, −.095] (job 188482).
- The instrument is reproducible on every node tested except homer-008: plain document cell 1,500/1,500 identical
  on homer-011 (job 188419); first 300 questions 300/300 on homer-016 (188476, 188524) and homer-007 (188477);
  compressed text 1,500/1,500 between homer-016 and homer-007 (188471 rerun) and between 11-07 and 11-11 code
  (188518 vs 185601); homer-008 changed 57/300 image answers (188458).
- Arm G, first `--dense_rate` code: tests 12/12 + 33/33 + 5/5 (job 188523); plain 300-Q regression 300/300
  identical, EM .477 = recorded (job 188524); its rate-8 control FAILED: 233/300 identical (law: ≥ 97%), slots
  118,147 = plain, EM .487 vs .477 (job 188525, verdict closed).
- provisional: the 67 moved answers come from re-compressing the picked page alone (batch of one) instead of inside
  the four-page padded batch of the plain path. From reading the code (2026-11-11 07:40): the decoder prompt is built
  the same way ("Document j:" + slots, `_build_chat_prompt`, no system prompt) in both paths; only the slot count is
  measured. The re-check (job 188580) tests the patched code.
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
- Data: three grouped HotpotQA sets (job 188465), six no-oracle grouped sets (job 188516), 1,500 rows each;
  `eval_data/probes_textvqa_ocr` (job 188545): 500 TextVQA rows, 500/500 questions matched.

**Open, and running** (1 campaign job at 07:55)
- arm G: re-check of the patched flag (job 188580, homer-013, RUNNING since 07:43): tests, 300-Q plain regression,
  300-Q rate-8 control, exit 1 on any failure.
- owner: BLOCKED.md asks to raise the cell cap 60 → 65 (keep a quarter at seed 43 is the main reason).

## What happened this wake
- ✗ lines in the brief: none of this campaign's (only self-cancelled jobs). Preemptions raced the end of three cells
  (188439 file 06:54:19 vs end 06:54:21; 188536 07:22:48 vs 07:22:51; 188537 07:37 vs 07:37:06): files copied aside
  as `attempt_<node>_<hhmm>_*`, the requeued reruns killed with `expmon kill` (nothing waited on them).
- Verifier findings: arm G hypothesis kept provisional, the code-reading check written out; g10 seed 42 now measured
  (188518); passage order re-paired at seed 42 with today's g10; tail parent label explained; no-oracle s43 decided;
  arm E gate decided with paired CIs at both seeds; int8 decided at both seeds; arm C baseline explained; 188546
  paired by jobid (it has no grid row: its launch left only an adopted manifest row; gripe + known_issues row).
- Paired and decided 7 cells (188518, 188546, 188538, 188539, 188439, 188536, 188537): all 1,500 rows, unparseable 0,
  nodes el9_8 like their baselines. 188521 re-verdicted closed (superseded by 188546).
- Submitted 1 cell: the arm G re-check 188580 (`expmon launch --force --exp probes --stage G --parent 184781
  --time 1:30:00 --exclude=homer-[005-009] scripts/probes/g_checks.sbatch`).
- Law (CAMPAIGN.md): arm A, B, E rows record their decisions (arm B closed; the no-oracle prediction marked wrong);
  arm E gets the keep-an-eighth gate written before any cell; budget line 58 used; same-node rule notes the code
  date is ruled out for text.
- Doc `docs/probe_campaign_2026-11.md` rewritten (status, findings re-ranked, arms A / B / E tables, instrument table,
  Closed). `expmon display --exp probes` refreshed (still 0 rows: known expmon defect).
- Gripes: launch without a manifest row (188546); preemption racing the end of eval cells.

## Owned jobs (jobid | arm | state at 07:55 | node | note)
| 188580 | G re-check | RUNNING | homer-013 | log `g_checks_188580.out`; last line `G CHECKS PASS|FAIL`, `=== EXIT CODE n ===` |

## Adopted jobs
(none. 188422, 188466-188470 belong to the sibling `longchunk` campaign; 188528-188535 to `freeze`.)

## Open arms and the next decision for each
| arm | status | next action |
|---|---|---|
| A. use-recovery | decided: route1 adopted, int8 adopted, gold-only and evidence-then-answer closed | none (expand s43 only if the cap is raised) |
| B. long-chunk mechanism | closed | none |
| C. reasoning vs reading | closed for budget | none |
| D. page picking | decided, caveat: rule chosen in-sample | none |
| E. attention-guided pruning | half adopted (both seeds); quarter met at s42 | if the cap is raised: `e_keep025_attng_s43` (→ 184780, parent 184784, `KEEPF=0.25 KEEPM=attn_global PX=1003520`, `--time 4:54:00`) and `e_keep0125_attng` (→ 184778, parent 184781, `KEEPF=0.125`); dirs to create with the symlink rule |
| F. photo text vs pixels | set built (188545) | cell only if the cap is raised or arm G fails: `f_text_s42` → 184778 symlink, `pisco_text "" eval_data/probes_textvqa_ocr`, PX=1003520, `--time 2:12:00`, paired on query_id with the TextVQA rows of 184783's photo file (n=500) |
| G. route then densify | re-check 188580 running | read `g_checks_188580.out`. PASS → `g_dense4_s42` (→ 184778, parent 184781) and `g_dense4_s43` (→ 184780, parent 184784), `RAW=route1 DENSE=4 PX=1003520 eval_mm_4b.sh expQ/probes/g_dense4_s4X/model pisco "" eval_data/mm_docqa_1500_1m`, `--time 4:54:00 --exclude=homer-[005-009]`; dirs exist. FAIL → arm G closed (the patch was the one allowed for this symptom: BLOCKED.md section), the 2 cells go to arm F + one health relaunch. Preempted → wait for the requeue |

## Health and budget
- cells: 58 of 60 used (every submission counts, killed ones too; 188546 counted by hand). Last 2: the rate-4 cells.
- GPU-hours of 40: tick sensor 13.0 GPU-h in the last 24 h at 07:30 (campaign started 00:00).
- gpu-be preempted 8 campaign attempts between 06:50 and 07:40; check each file's mtime against `sacct -D` attempt ends
  before pairing.

## Next wake
Read the arm G re-check result. On PASS submit the two rate-4 cells; on FAIL close arm G. Then check BLOCKED.md for
the owner's answer on the cap.

## Blockers / needs the owner
- BLOCKED.md: raise the cell cap 60 → 65 (recommended) so the keep-a-quarter result gets its second seed. Nothing else
  waits on the owner.

## Proposals for the sibling `longchunk` campaign (training is not this campaign's job)
- Stage-1/2 data that packs related passages in one chunk (both gold passages of a multi-hop question) rather than
  random long chunks: +.067-.073 with the golds together (both seeds), +.058 on the tail layout. Joining all ten:
  +.035 (seed 42) / +.014 n.s. (seed 43). A ranker's top 3 joined does no better than all ten.
- A query-aware slot budget (nested-budget training): half the slots, picked by the decoder's attention with the
  question, lose nothing at both seeds; a quarter loses nothing at seed 42.
