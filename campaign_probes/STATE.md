# STATE — campaign probes   (rewritten WHOLESALE by every LLM wake; never append)
last wake: 2026-11-11 05:30-06:00 KST (woken by the condition "arm E controls 188480 + 188481 ended")

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Showing the routed page raw recovers the whole document gap, at both seeds. Reference models, documents
  (`mm_docqa_1500_1m`, 4 pages, PX=1003520), n=1,500 each: seed 42 judge .820 vs plain .610 = +.210 [+.187, +.233],
  350 up / 35 down, McNemar p<1e-60 (job 188441, verdict best); seed 43 .822 vs .615 = +.207 [+.183, +.232], 349 / 38
  (job 188460, verdict best). The pick is right on 98.6% / 98.9% of questions; the all-raw reader scores .809.
- Removing the distractor pages instead gains only about 2 pt, below the +3 pt gate at both seeds: gold page only,
  compressed, seed 42 .632 vs .610 = +.022 [+.005, +.040], p=.020 (job 188440); seed 43 .639 vs .615 = +.025
  [+.006, +.043], p=.010 (job 188459). So the document loss is reading the compressed answer page, not distraction.
- Page picking without training (arm D, model 183814, n=1,500 per set, job 188515): 10-page documents digit .861,
  attention .885 (+.025 [+.010, +.039]), product .879; HotpotQA digit .922, attention .873 (−.049), product .928.
  The gate passes only with a per-source switch (attention on images, digit on text). No judge cell: the expected
  judge gain is ≈ +.006, under the .02 noise band.
- The instrument is reproducible on every node tested except homer-008: plain document cell 1,500/1,500 identical
  on homer-011 (job 188419); first 300 questions 300/300 identical on homer-016 (188476) and homer-007 (188477);
  compressed HotpotQA text 1,500/1,500 identical between homer-016 and homer-007 (job 188471 and its rerun);
  homer-008 changed 57/300 image answers (188458).
- Arm E plumbing passes: keeping all slots through the attention pass gives the plain answers 300/300 (job 188480);
  the patched eval's plain path gives 300/300, EM .477 = recorded (job 188481); invariant tests 45/45 (job 188479).
- Data: three grouped HotpotQA sets (job 188465) and six no-oracle grouped sets (job 188516; both gold passages in
  the model's top 2 on .404 digit / .495 attention / .487 product of the 1,500 questions), 1,500 rows each.
- provisional: arm B, model 183814, HotpotQA 1,500 Q, against the 11-11 ten-chunk re-run .582 (job 188486):
  golds joined .655 = +.073 [+.051, +.095], p=1.5e-10, and +.038 [+.015, +.061] vs one chunk of ten (.617), all
  on bridge questions (job 188471); distractors joined .604 = +.022 [+.003, +.041], p=.027 (job 188472); all ten
  joined in shuffled order .615 vs .617 = −.002 [−.023, +.020], p=.91 (job 188473). One model; repeats running.
- provisional: slots stored in int8 change nothing at seed 42: .611 vs .610 = +.001 [−.003, +.006], 1,445/1,500
  answers identical, n=1,500 (job 188478). A tie: the seed-43 repeat is due.
- provisional: evidence-then-answer does not close the text gap at seed 42 (MRQA, n=1,500, 250 per source):
  compressed .699 vs plain .718 (−.019, p=.11, job 188446); uncompressed .807 = .807 (job 188457). Gap
  compressed − raw: DROP −.204 → −.172, DuoRC −.084 → −.156, RACE −.152 → −.188 (gate ≥ 5 pt shrink: not met).
- provisional: keeping every 2nd slot on the reference model costs −.105 [−.127, −.085]: .505 vs .610, n=1,500,
  slots 0.504 × (job 188483). This is arm E's untrained baseline; three cells still running.

**Open, and running** (10 cells at 05:55: 8 running, 2 queued)
- arm A: evidence-then-answer on documents, seed 42 (job 188439, running on homer-011 after a preemption at 05:06).
- arm B repeat on the seed-43 twin 183816 (183814's recipe + seed 43): ten chunks (188509), one chunk (188510),
  golds joined (188511), distractors joined (188512), shuffled one chunk (188513, queued); golds joined on the
  tail-layout model 183538 (188514, queued).
- arm E, reference model 184778, documents, keep half the slots: first half per chunk (188482), most-attended half
  per chunk (188484), most-attended half over all pages (188485).

## What happened this wake
- Diagnosed every ended job with sacct -D + the result files. Five cells had been preempted seconds AFTER writing
  their judged result (known race): 188459 (gold s43, 05:01), 188460 (route1 s43, 05:31), 188471 (golds joined,
  04:58), 188446 (text expand, 05:35), plus 188440 / 188441 (verifier finding, both later re-completed). Copied each
  file aside as `attempt_<node>_<hhmm>_*`; killed the reruns of 188459, 188460, 188446 with `expmon kill` (same
  image, nothing waits on them); kept 188471's rerun on homer-007 as the text node check (1,500/1,500 identical).
- Verifier findings fixed: 188440 / 188441 are complete results (now verdicts), not "queued"; the 188441 "no file
  written" line was wrong (it was the race).
- Recorded 13 probe values and 18 verdicts; arm A, arm C (expand), arm D decided at the levels above.
- Law (CAMPAIGN.md) tightened: layout results repeat on 183816 before adoption; arm D's judge cell only when its
  expected size beats the noise band; node rule rewritten with 188477 / 188471 (homer-008 alone is off; homer-005,
  006, 009 untested; keep `--exclude=homer-[005-009]`); the "text did not move" line marked partly wrong.
- Code (free path): `scripts/probes/build_groups.py` gains `pred<K>_<picker>` kinds (+40 lines): the K passages a
  model ranks first joined, others alone, ranks from an attention-route dir; checks the route dir matches the base
  set row for row (gold, passage count) and prints top-K recall of both golds. Built by job 188516, exit 0.
- 8 submissions (cap 8): 188509-188514 arm B, 188515 fusion (= 188444 without its dead dependency), 188516 build.
- `docs/probe_campaign_2026-11.md` rewritten (findings ranked, arm tables A/B/C/D/E, instrument table).
  `expmon display --exp probes` refreshed (still 0 rows: known expmon defect, gripe filed earlier).
- Gripes: `expmon probe` with one quoted argv records nothing silently; eval results not picked up into the grid.

## Owned jobs (jobid | arm | state at 05:55 | node | note)
| 188439 | A evidence-then-answer s42 | RUNNING | homer-011 | `a_expand_s42`; vs 184781 .610; ~90 min + judge |
| 188482 | E keep first half | RUNNING | homer-013 | `e_keep05_first` |
| 188484 | E keep most-attended half per chunk | RUNNING | homer-016 | `e_keep05_attn` |
| 188485 | E keep most-attended half overall | RUNNING | homer-016 | `e_keep05_attng` |
| 188509 | B ten chunks, 183816 | RUNNING | homer-010 | `b_g1_s43` |
| 188510 | B one chunk, 183816 | RUNNING | homer-016 | `b_g10_s43` |
| 188511 | B golds joined, 183816 | RUNNING | homer-016 | `b_gold_joined_s43` |
| 188512 | B distractors joined, 183816 | RUNNING | homer-016 | `b_distr_joined_s43` |
| 188513 | B shuffled one chunk, 183816 | PENDING | excl. 005-009 | `b_g10_shuf_s43` |
| 188514 | B golds joined, tail 183538 | PENDING | excl. 005-009 | `b_gold_joined_tail`; vs tail g1 .568 (11-07 file) |

## Adopted jobs
(none. 188422 and 188466-188470 belong to the sibling `longchunk` campaign.)

## Open arms and the next decision for each
| arm | status | next action |
|---|---|---|
| A. use-recovery | route1 adopted (both seeds); gold-only closed; int8 tie at s42; expand s42 running | when 188439 ends: pair vs 184781; tie or gain → launch expand s43. Launch int8 s43 (`MQ=int8`, `a_mq8_s43` → 184780, `--time 4:54:00`) to close the int8 tie |
| B. long-chunk mechanism | 3 cells provisional on 183814; 6 cells running | pair 183816 cells: golds joined vs its g1 and g10, distractors joined vs g1, shuffled vs g10. Adopt "cross-gold context, diluted by distractors" if golds joined beats g1 AND g10 by > .02 with CI above 0 again. Tail: re-run the tail's g1 on 11-11 code (`b_g1_tail` → 183538) before pairing 188514 (the .568 file is from 11-07 code, the 55-answer drift). If adopted: launch the no-oracle cells `probes_hotpot_pred2_attn` and `pred3_attn` on 183814 (CML=4096 TOPK=10, `--time 1:14:00`); gate (written now): pred − g1 re-run ≥ +.03 with CI above 0 → "rank, then re-compress the top passages together" recovers ≥ 40% of the gold-joined gain without an oracle |
| C. reasoning vs reading | expand at s42: gap does not shrink | tie → seed-43 repeat: compressed plain + compressed expand on 184780 (`c_plain_s43`, `c_expand_s43`, CML=4096; `--time` plain 0:51:00 = 17 min × 3, expand 4:54:00 = 98 min × 3). Thinking mode: deprioritised (new flag, ~4 h per cell, expand already says "content, not room to reason") |
| D. page picking | decided (job 188515) | none |
| E. attention-guided pruning | controls pass; every-2nd −.105; 3 cells running | pair attn and attn_global vs the better of first / every-2nd; gate ≥ +4 pt with CI above 0 → seed 43 + photos; within .02 → close attention-guided pruning |
| F. photo text vs pixels | blocked on data | build `eval_data/probes_textvqa_ocr` (500 TextVQA rows + `ocr_tokens` from lmms-lab/textvqa) with a `scripts/probes/` builder via SLURM; check the compute node reaches the HF hub first |
| G. (proposal) route then densify | not in the law yet | route1 shows the answer page needs more than its slots. One knob: re-compress only the picked page at rate 4 (2× slots) instead of raw. Needs one eval flag. Gate to write before: ≥ 2/3 of route1's +.21 at ≤ 1/3 of its extra decoder tokens → a serving mode and a stage-2 target. Owner's call whether it enters the law |

## Health and budget
- budget: 41 of 60 cells committed (33 before this wake + 8). GPU-hours of 40: the tick's sensor said 6.8 GPU-h in
  the last 24 h at 05:30; the 10 open cells add ≈ 8 (6 × 25 min, 3 × 30-98 min, 1 × 90 min).
- next wake needs about 7 cells (int8 s43, expand s43 if due, arm C s43 ×2, tail g1, no-oracle ×2): 48 of 60.
- gpu-be preemptions keep racing job ends: 7 cells this morning wrote their result seconds before a preemption.

## Next wake
`wake-when` on the first group to leave the queue: arm B (188509-188514), arm E (188482, 188484, 188485), arm A
expand (188439); heartbeat 2 h.

## Blockers / needs the owner
- none blocking. FYI: `eval_mm_4b.sh`, `probe_attn_route.sh`, `run_tests_cpu.sh` are not under git (`.gitignore` has
  `*.sh`); force-add them when convenient. The ALERT lines about the 01:00 / 01:30 failed wakes are yours to clear
  (cause: the account's 5-hour usage limit, known_issues row). Arm G above is a proposal for you to accept or drop.
