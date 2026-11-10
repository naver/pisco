# STATE — campaign probes   (rewritten WHOLESALE by every LLM wake; never append)
last wake: 2026-11-11 06:30-06:55 KST (woken by "problems": the ✗ was 188518's preemption + requeue, a known issue)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Showing the routed page raw recovers the whole document gap, at both seeds. Reference models, documents
  (`mm_docqa_1500_1m`, 4 pages, PX=1003520), n=1,500 each: seed 42 judge .820 vs plain .610 = +.210 [+.187, +.233],
  350 up / 35 down, McNemar p<1e-60 (job 188441, verdict best); seed 43 .822 vs .615 = +.207 [+.183, +.232], 349 / 38
  (job 188460, verdict best). The pick is right on 98.6% / 98.9% of questions; the all-raw reader scores .809.
- Removing the distractor pages instead gains only about 2 pt, below the +3 pt gate at both seeds: gold page only,
  compressed, seed 42 .632 vs .610 = +.022 [+.005, +.040], p=.020 (job 188440); seed 43 .639 vs .615 = +.025
  [+.006, +.043], p=.010 (job 188459). So the document loss is reading the compressed answer page, not distraction.
- The long-chunk gain is context between the two answer passages, at both seeds. HotpotQA, 1,500 Q each, golds
  joined in one chunk (others alone) vs ten separate chunks on today's code: seed 42 (183814) .655 vs .582 = +.073
  [+.051, +.095], p=1.5e-10 (jobs 188471 vs 188486); seed 43 (183816) .669 vs .601 = +.067 [+.046, +.088], 186 / 85,
  p=8e-10 (jobs 188511 vs 188509, verdict best). Slots .998×. Vs one chunk of all ten at seed 43: +.053 [+.029, +.076],
  p=9e-6 (vs 188510).
- Joining the eight distractor passages is not adopted: above the noise band once at seed 42, .604 vs .582 = +.022
  [+.003, +.041], p=.027 (job 188472 vs 188486), not repeated at seed 43, .617 vs .601 = +.016 [−.003, +.035],
  p=.11 (job 188512 vs 188509).
- Blind slot halving on the reference model 184778 (documents, 1,500 Q, slots .50×): every 2nd slot .505 vs .610 =
  −.105 [−.127, −.085], p=3e-24 (job 188483); first half of each chunk .495, −.115 [−.136, −.095], p=4e-28 (job 188482).
- The instrument is reproducible on every node tested except homer-008: plain document cell 1,500/1,500 identical
  on homer-011 (job 188419); first 300 questions 300/300 identical on homer-016 (188476, and 188524 on the patched
  code) and homer-007 (188477); compressed HotpotQA text 1,500/1,500 identical between homer-016 and homer-007
  (job 188471 and its rerun); homer-008 changed 57/300 image answers (188458).
- The first `--dense_rate` code (arm G) left the default path unchanged: invariant tests 12/12 + 33/33 + 5/5, exit 0
  (job 188523); plain 300-Q regression 300/300 answers identical to 184781, EM .477 = recorded (job 188524).
- That code's rate-8 control FAILED its identity check: `RAW=route1 DENSE=8`, 184778, first 300 documents: 233/300
  answers identical to 184781 (law: ≥ 97%), slots 118,147 = plain, EM .487 vs .477, pick right on .983 (job 188525,
  homer-016, verdict closed; answers saved as `attempt_h016_0644_*`, not judged: preempted while judging). The two
  rate-4 cells 188541 / 188542 were cancelled before they started.
- provisional: the 67 moved answers come from re-compressing the picked page alone (batch of one) instead of inside
  the four-page padded batch of the plain path; the prompt text and slot count are the same. Untested until the
  patched re-check runs (the patched code makes the rate-8 slots bit-identical, so the check must then pass).
- Data: three grouped HotpotQA sets (job 188465) and six no-oracle grouped sets (job 188516; both gold passages in
  the model's top 2 on .404 digit / .495 attention / .487 product of the 1,500 questions), 1,500 rows each;
  `eval_data/probes_textvqa_ocr` (job 188545): 500 TextVQA rows, 500/500 questions matched, OCR 13 tokens per row on
  average, gold answer verbatim in the OCR on .596 of rows.
- provisional: one chunk of all ten vs ten chunks at seed 43: .615 vs .601 = +.014 [−.008, +.035], p=.23 (job
  188510), smaller than the +.037 of seed 42 on the 11-07 files. The seed-42 value on today's code is unmeasured
  until the re-run 188518 returns; the difference between seeds may be seed variation.
- provisional: passage order inside a long chunk: seed 43 .618 vs .615 = +.003 [−.021, +.026], p=.87 (job 188513
  vs 188510, both today's code). The seed-42 leg (−.002, job 188473) is against the retired 11-07 file 185601; it
  waits on 188518.
- provisional, one model: the cross-passage gain needs no spread layout. Tail-layout model 183538, golds joined
  .627 vs ten chunks (today's code) .569 = +.058 [+.035, +.082], 202 / 115, p<.001, slots .998× (jobs 188514 vs
  188519, verdict best).
- provisional (seed 42): the gain survives without an oracle. Top 3 passages by 183814's attention joined, others
  alone: .614 vs ten chunks .582 = +.032 [+.009, +.056], 175 / 127, p=.007, slots .995× (job 188521 vs 188486,
  verdict best); top 2: .612, +.030 [+.008, +.051], p=.007 (job 188520). Gate +.03 met. Seed-43 repeat = job 188546.
- provisional: given the question, half the slots carry what the decoder uses. Keep the most-attended half over all
  pages, documents, 1,500 Q each, slots .496×: seed 42 (184778) .610 vs .610, +.000 [−.009, +.010], 27 / 27, 1,289
  identical (job 188485); seed 43 (184780) .615 vs .615, +.000 [−.009, +.009], 25 / 25, 1,306 identical (job 188522
  vs 184784, verdict best). Per chunk, seed 42: .606, −.004 [−.015, +.007] (job 188484). +.105 over the better blind
  half at seed 42 (188483); the seed-43 blind half (188537) is the last piece of the gate.
- provisional: page picking (arm D, model 183814, n=1,500 per set, job 188515): 10-page documents digit .861,
  attention .885 (+.025 [+.010, +.039]); HotpotQA digit .922, attention .873. The per-source switch that passes the
  gate was chosen on the same questions (no held-out split). No judge cell: expected judge gain = +.025 × the route1
  hit-minus-miss judge gap (.252 at seed 42: 1,479 hits .824 vs 21 misses .571, file a_route1_s42; .325 at seed 43 on
  16 misses) ≈ +.006 to +.008, under the .02 band.
- provisional: slots stored in int8 change nothing at seed 42: .611 vs .610 = +.001 [−.003, +.006], 1,445/1,500
  answers identical, n=1,500 (job 188478). Seed-43 repeat = job 188536.
- provisional: evidence-then-answer does not close the text gap at seed 42 (MRQA, n=1,500, 250 per source):
  compressed .699 vs plain .718 (−.019, p=.11, job 188446); uncompressed .807 = .807 (job 188457; its plain side
  185542 ran on homer-008 with 11-06 code, 300/300 re-checked by 188456). Per-source gaps rest on 250 Q, no CI.

**Open, and running** (7 campaign jobs at 06:50)
- arm A: evidence-then-answer on documents, seed 42 (188439, homer-011, 1,400/1,500 at 06:47); int8 seed 43 (188536).
- arm B: one chunk of ten re-run on today's code, 183814 (188518, 1,130/1,500 at 06:47, 2nd attempt); no-oracle
  top 3 at seed 43 (188546).
- arm E: blind every-2nd, seed 43 (188537); keep a quarter, seed 42 (188538); photos, seed 42 (188539).
- arm F: set built (188545), no cell (budget).
- arm G: patched, re-check not submitted yet (this wake's 8 submissions were used).

## What happened this wake
- Diagnosed the ✗: 188518 PREEMPTED on homer-010 at 06:10 (4 min in, empty `eval/`), requeued — known issue, no action.
- Verifier findings fixed: "half noise" removed (the seed-42 value on today's code is unmeasured); the seed-42
  passage-order leg marked as paired with a retired file; distractors joined rewritten as "above the band once, not
  repeated, not adopted"; the missing seed-43 blind cell for arm E is now running (188537).
- Paired 5 ended cells: arm B tail control + its baseline, no-oracle top 2 and top 3; arm E seed 43. All 1,500 rows,
  unparseable 0, 0 non-binary judges, nodes el9_8 like their baselines.
- Arm G: tests (188523) and plain regression (188524) PASS; the rate-8 control (188525) FAILED identity (233/300).
  Killed the two rate-4 cells before start and the control's requeued rerun (same answers, nothing waits on it;
  its answers were saved first). Patch, `scripts/eval_mm_docqa.py` dense branch (9 lines, default path untouched):
  re-compress the picked page inside its example's batch. `py_compile` passes; invariant tests NOT yet run on it
  (they run inside the re-check job).
- New code (`scripts/probes/`, free): `build_textvqa_ocr.py` + `.sbatch` (arm F set, ran OK), `g_checks.sbatch` +
  `g_checks_compare.py` (arm G re-check in one job: tests, 300-Q regression, 300-Q `DENSE=8` control, the law's
  criteria, exit 1 on failure; also judges the saved pre-patch control answers).
- 8 submissions (cap 8): 188536-188539, 188541-188542 (killed), 188545, 188546.
- Law (CAMPAIGN.md): arm B row (tail and no-oracle results); arm G row (control failure kept and marked, the patch,
  the one-job re-check, full cells only after a pass); budget line defines a cell and gives the last 3 to arm G.
- Doc `docs/probe_campaign_2026-11.md` rewritten in place (status, findings 2-3, arms B / E / F / G, Closed).
- The guard refused one `expmon note` whose text contained the word for a raw submit (nothing was submitted);
  reworded; known_issues row + gripe filed.

## Owned jobs (jobid | arm | state at 06:50 | node | note)
| 188439 | A evidence-then-answer s42 | RUNNING | homer-011 | `a_expand_s42`; vs 184781 .610 |
| 188518 | B one chunk of ten, 183814, today's code | RUNNING (2nd attempt) | homer-020 | `b_g10_rep`; re-pairs 188471 / 188473 at s42 |
| 188536 | A int8 s43 | RUNNING | homer-016 | `a_mq8_s43`; vs 184784 .615 (homer-016) |
| 188537 | E blind every-2nd, s43 | RUNNING | homer-016 | `e_keep05_even_s43`; 188522 − it ≥ +.04, CI above 0 → arm E adopted |
| 188538 | E keep a quarter, s42 | RUNNING | homer-013 | `e_keep025_attng`; vs 184781 .610 |
| 188539 | E photos, attention half, s42 | RUNNING | homer-016 | `e_keep05_attng_photo`; vs 184783 .677 (homer-011) |
| 188546 | B no-oracle top 3, s43 | RUNNING | homer-012 | `b_pred3_attn_s43`; vs 188509 .601 |

## Adopted jobs
(none. 188422, 188466-188470 belong to the sibling `longchunk` campaign; 188528-188535 to `freeze`.)

## Open arms and the next decision for each
| arm | status | next action |
|---|---|---|
| A. use-recovery | route1 adopted; gold-only closed; expand s42 + int8 s43 running | 188439: pair vs 184781 (no seed-43 cell: budget). 188536: within .02 at both seeds → adopt int8 as a serving option |
| B. long-chunk mechanism | adopted: context between answer passages (both seeds); tail control met (one model); no-oracle met at s42 | 188546 ≥ +.03 over 188509 with CI above 0 → adopt "rank, then re-compress the top passages together" (serving mode; proposal to `longchunk` below). 188518: re-pair 188471 and 188473 at s42 against today's g10 |
| C. reasoning vs reading | expand at s42: gap does not shrink | closed for budget |
| D. page picking | decided (job 188515), caveat: rule chosen in-sample | none |
| E. attention-guided pruning | ties plain at both seeds (188485, 188522) | 188522 − 188537 ≥ +.04 with CI above 0 → adopt. 188538: within .02 → three quarters unused; loss ≥ .05 → free share between 1/2 and 3/4. 188539 within .02 of 184783 → holds on photos |
| F. photo text vs pixels | set built (188545) | no cell unless arm G fails its re-check or the owner raises the cap. Cell when allowed: `f_text_s42` → 184778 symlink, `pisco_text "" eval_data/probes_textvqa_ocr`, PX=1003520, `--time 2:12:00`, paired on query_id with the TextVQA rows of 184783's photo file (n=500) |
| G. route then densify | control failed, flag patched | FIRST ACT next wake: `expmon launch --force --exp probes --stage G --parent 184781 --time 1:30:00 '--exclude=homer-[005-009]' scripts/probes/g_checks.sbatch` (no env vars; `--time` = (tests 3.5 + 2 × 11 min slowest 300-Q cell 188481 + 3 judge) × 3). On PASS: `g_dense4_s42` (→ 184778, parent 184781) and `g_dense4_s43` (→ 184780, parent 184784), `RAW=route1 DENSE=4 PX=1003520`, `--time 4:54:00`; their dirs exist. On FAIL: close arm G (the patch was the one allowed for this symptom → BLOCKED.md), give the 2 cells to arm F + one relaunch |

## Health and budget
- cells: 57 of 60 used (every submission counts, killed ones too; law budget line). Last 3: arm G re-check, then the
  two rate-4 cells only if it passes. Dropped: arm F cell, arm A expand s43, arm C s43, arm F s43, health relaunches.
- GPU-hours of 40: the tick's sensor counted 10.2 GPU-h in the last 24 h at 06:30 (campaign started 00:00); the
  running cells add about 5, arm G about 4.
- Photo seed-43 baseline 184785 ran on homer-008 (the node that moves image answers): no seed-43 photo pairing.
- gpu-be preemptions this wake: 188525 at 06:43 (while judging; file saved), 188518 at 06:10 (no file). Before each
  pairing, compare file mtime with the attempt ends.

## Next wake
Submit the arm G re-check first. Then pair whatever ended: 188439 (expand s42), 188537 (arm E gate), 188518,
188536, 188538, 188539, 188546.

## Blockers / needs the owner
- none blocking. The 60-cell cap leaves 3 cells, all for arm G. Raising it by 3 is the owner's call: it would run
  the text-only TextVQA cell (arm F, set built), evidence-then-answer at seed 43, and keep one health relaunch.

## Proposals for the sibling `longchunk` campaign (training is not this campaign's job)
- Stage-1/2 data that packs related passages in one chunk (both gold passages of a multi-hop question, or the top
  passages of a ranker) rather than random long chunks: +.067-.073 with the golds together (both seeds), +.058 on the
  tail layout, +.014 n.s. for all ten together at seed 43, +.032 for the top 3 by attention at seed 42.
