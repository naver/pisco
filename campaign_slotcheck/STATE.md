# STATE — campaign slotcheck   (rewritten WHOLESALE by every LLM wake; never append)

## TL;DR — where this leaves the question (readable as an email)
All four checks are decided. Nothing is running. The one end-of-campaign owner question is in BLOCKED.md (open a
stage-2 rebuild that teaches the image slots to keep the page's words: recommended, about 90 GPU-hours). Deliverable:
`docs/slot_checks.md` (a ranked "campaign answer" table, then one table per check).

**Settled**
- Check 1 (arm C, feasibility gate PASS): the finetune collator's 4-document cap never removes ALL answer pages of a
  trained row — 0 / 20,000 document rows, 0 / 8,000 photo rows → not a cause. Beside it: 1,910 / 20,000 document rows
  (9.55%; all Docmatix) lose one of their two source pages. Job 191663 (closed).
- Check 2a (arm P, padding probe), n = 120 pages, job 191672 (closed): not inert (median cosine .934 at 63 pads, bar
  .9999). Split, n = 40 pages, job 191781 (closed): a ONE-row padded batch leaks the pads (median cosine .935 bf16 /
  .953 fp32 at 63 pads; .995 / 1.0000 with the pad zeroing forced on) — latent, no recorded run takes that path
  (hand-off 2de131). A two-row padded batch: median cosine .9948 bf16 / 1.0000 fp32 = the no-pad two-row floor
  .9943 / 1.0000, but in fp32 its max abs difference is 2.3-2.8× that floor (median .078 at 63 pads / .095 at 17 vs
  .034): a small residual pad effect on the trained path, read as such (interpretation). Its cost on answers: check 2b.
- Check 2b (arm P) decided: NOT A CAUSE. Padded to 64 − not padded, judge accuracy, n = 1,500 paired questions per
  seed: seed 42 .615 − .610 = +0.5 pt [−1.1, +2.1], McNemar p .62 (job 191783, closed); seed 43 .624 − .615 =
  +0.9 pt [−0.7, +2.5], p .29 (job 191784, closed). Both CIs inside ±3 pt.
- provisional (one training, no repeat): Check 4 (arm O): slot capacity is NOT the limit (gate ≥ 90%). Word recall of
  the bf16 overfit model from its slots on the 100 trained pages .921, bootstrap 95% CI of the mean [.902, .938],
  vs the start model's .702: +21.9 pt [+18.5, +25.3], 98 up / 2 down, sign p 8e-27, n = 100 pages (job 192209,
  closed; start 191893). Held-out 100 pages .667 vs .670: −0.3 pt [−4.2, +3.6]. Text-layer ceiling .974. 116-123
  slots per page. Loss not flat at step 210 (job 192002) → .921 is a lower bound. A repeat needs 2 cells (training +
  recall); 1 is left → offered to the owner as option C.
- provisional (one seed; pooled read at the 5-pt bar; 1.52× the slots): Check 3 (arm X) DECIDED, n = 1,000
  paired questions (two disjoint reads of `mm_docqa_1500_1m`, R seed 42), judge accuracy:
  - image + transcription slots − image slots: read 1 +4.6 pt [+1.0, +8.0], McNemar 51 / 28, p .013 (job 192223);
    repeat on 500 new questions +5.4 pt [+1.8, +8.8], 56 / 29, p .005, no source CI wholly below 0 (job 192239)
    → PASSES the repeat (gate fixed before launch). Pooled 1,000: .667 vs .617, **+5.0 pt [+2.5, +7.6]**,
    107 / 57, p 1.2e-4 = at the law's 5-pt bar with the CI above 0 → "the image slots lose content the page's own
    words keep" (consistent with; 1.52× the slots, see caveat).
  - transcription slots alone − image slots: −12.0 [−16.6, −7.6] (read 1), −15.4 [−20.2, −10.6] (read 2), pooled
    **−13.7 pt [−17.1, −10.3]**, 84 / 221, p 2.3e-15 → worse.
  - Uncompressed on the same questions: .790 / .832, pooled .811. Image slots .602 / .632 (pooled .617).
  - Per source pooled (image + text − image): InfoVQA +6.6 [+1.5, +11.4], MP-DocVQA +7.5 [+2.7, +12.0], SlideVQA
    +0.9 [−2.4, +4.2]. The whole net gain (+50 questions) sits on the 878 / 1,000 questions whose answer string is in
    the transcription; 0 on the other 122.
  - Slots per question image 430.4 / 431.9, text 222.8 / 222.6, image + text 653.2 / 654.5 (8 tokens per slot).
  - Caveat: image + text spends 1.52× the slots; no slot-matched arm in the law. Seed 43 not run (the law runs it only
    if the FIRST read clears 5 pt; it gave +4.6).
- Arm X plumbing: transcriptions round 1 (192010 / 192210 / 192211 / 192212) 1,268 / 1,268 pages, empty 0, 124 at
  the cap; round 2 (192237) 1,252 / 1,252, empty 0, 121 at the cap; builds 192221 (round 1; its one "cut" row is a
  chunk-boundary split, decoded by 192238's round-1 re-audit: cut_boundary 1 = cut 1, ok=True) and 192238 (round 2:
  cut 0, ok=True); all closed.
- Arm O plumbing: build 191891, smoke 191892, bf16 overfit 192002 (210 steps, loss .602 → .028), all closed. fp32
  second training not run: arm O ≈ 7.9 GPU-h vs its ≈ 5 budget line.

**Open, and running**
- (none) — waiting for the owner's answer in BLOCKED.md.

## Owned jobs
| job | arm | state | note |
|---|---|---|---|
| 191663 | C (check 1) | COMPLETED | closed |
| 191671 / 191672 / 191781 | P (2a) | COMPLETED | closed |
| 191782 / 191783 / 191784 | P (2b) | COMPLETED | closed: not a cause |
| 191890 / 191891 / 191892 / 191893 | O build smoke / build / training smoke / recall baseline | COMPLETED | closed |
| 192002 | O bf16 overfit | FAILED exit 3 on a 3rd attempt (attempt 2 COMPLETED, model complete) | closed; `expQ/slotcheck/o_bf16/{model,checkpoint-210}` |
| 192003 | O recall (dead dependency) | CANCELLED, never ran | closed; relaunched as 192209 |
| 192209 | O recall of the overfit | COMPLETED | closed: check 4 decided |
| 192009 / 192010 / 192210 / 192211 / 192212 | X transcription round 1: smoke / shards 0-3 | COMPLETED | closed; `expQ/slotcheck/transcribe/` |
| 192214 | X build smoke (30 Q) | FAILED exit 4 (health-rule bug) | closed |
| 192215 / 192216 / 192217 / 192222 | X children of failed parents | CANCELLED, never ran | closed |
| 192220 | X eval plumbing, 30 Q × 3 sets | COMPLETED | closed (plumbing passed) |
| 192221 | X build round 1 | FAILED exit 4 (cut-rule misfire, sets complete) | closed; `eval_data/slotcheck_docqa_{img,text,imgtext}_500` |
| 192223 | X judged eval round 1 | COMPLETED 0:0 36:49 homer-014 | closed: first read |
| 192236 | X transcription smoke round 2 | COMPLETED 0:0 5:14 homer-017 | closed (plumbing) |
| 192237 | X transcription round 2, 1,252 pages | COMPLETED 0:0 4:13:18 homer-017 | closed; `expQ/slotcheck/transcribe_r2/shard0of1.json` |
| 192238 | X build round 2 + round-1 re-audit (CPU) | COMPLETED 0:0 4:39 homer-017 | closed; `eval_data/slotcheck_docqa_{img,text,imgtext}_500_r2`, audits `expQ/slotcheck/textsets_192238{,_r1}/` |
| 192239 | X judged eval round 2 (repeat) | COMPLETED 0:0 35:56 homer-016 | closed: check 3 decided; `expQ/slotcheck/x_s42/eval/*_500_r2.json` |

## Adopted jobs (found running in this project, not launched by the campaign — never killed, never duplicated)
(none)

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| C (check 1) | decided: not a cause | none |
| P (check 2a / 2b) | decided: not a cause | none |
| O (check 4) | decided, one training: capacity is not the limit | a repeat only if the owner picks option C (2 cells, ≈ 5 GPU-h, cap 32 → 34) |
| X (check 3) | decided: image + text +5.0 pt pooled, passes its repeat; text alone −13.7 | none |
| campaign | all four decided | the owner's answer to BLOCKED.md (A / B / C / D). A, B or D: nothing more to launch here → close-out. C: launch the second overfit seed (`pt_overfit.sbatch` with another seed, then `recall.sbatch`, chained) |

## Health evidence this wake (2026-11-24 19:00; numbers, not adjectives)
- No campaign job ran since 2026-11-20 (brief: running 0, pending 0, problems 0, new result files 0, gpuh24 0.0).
- The last cells (192237 / 192238 / 192239) closed on 2026-11-20 with health OK; their evidence is in their
  `expmon verdict` lines (grid VERDICT `closed`), not repeated here.
- Budget unchanged: 31 of 32 cells; GPU-hours, all attempts from `sacct -D`: 20.9 of 30. Arm X 10.1 of its ≈ 12 line.

## Verifier (pass of 2026-11-20 00:00)
16 claims: 7 supported, 9 provisional, 0 unsupported. Nothing to fix. The provisional ones are already marked
provisional above (check 4 = one training; check 3 = one seed, pooled read at the 5-pt bar, 1.52× the slots).

## This wake
- Woken by alert + daily heartbeat. The alert was the account's weekly usage limit (wake 2026-11-21 00:00 failed
  in 10 s; held until 2026-11-24 19:00). The hold is over; the ALERT file is deleted (known_issues row 23, now
  also covers the weekly limit). +1 on the open gripe 69929754 (the tick spends a wake on a self-resolving alert).
- No REQUEST, no GATE, no answer to owner question d336f3 in `expmon questions`.

## Efficiency line (SPEED: the login's usage limit held this loop 23.5 h, 100 % of the window)
Moving the loop to another login is an owner matter (`cc-team`, files under ~/.claude, which the loop may not
touch). It costs nothing here: the campaign has no launch left until the owner answers BLOCKED.md.

## Law notes this wake
- None changed.

## Blockers / needs the owner
- BLOCKED.md: the ONE end-of-campaign question (the four verdicts + the stage-2 rebuild they motivate), queue id
  d336f3. NEXT `waiting-owner`.
- Hand-off 2de131 (pad leak guard) still open: the guard sits uncommitted in `pisco/model.py` (f8ae6efe); no-op on
  the eval path (192239: 500 / 500 same answers), still untested on the one-row padded path it fixes.
