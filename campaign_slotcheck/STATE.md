# STATE — campaign slotcheck   (rewritten WHOLESALE by every LLM wake; never append)

## TL;DR — where this leaves the question (readable as an email)
The campaign is closed (stand-down 2026-11-25). Checks 1 and 2 are decided by their gates; checks 3 and 4 are
decided provisionally (check 3: one model seed; check 4: one training). The owner answered the end question on
2026-11-25: **option A, in a NEW campaign** — rebuild stage 2 on the spread `<MEM>` layout, one run with more page
transcription on document pages and one control run (a second seed of today's recipe). This wake may not create or
arm a campaign, so it wrote the brief `campaign_slotcheck/SUCCESSOR.md` and handed the set-up to the pisco chats
(hand-off 4c0b26). Nothing is running. Deliverable: `docs/slot_checks.md`.

**Settled**
- Check 1 (arm C, feasibility gate PASS): the finetune collator's 4-document cap never removes ALL answer pages of a
  trained row — 0 / 20,000 document rows, 0 / 8,000 photo rows → not a cause. Beside it: 1,910 / 20,000 document rows
  (9.55%; all Docmatix) lose one of their two source pages. Job 191663 (closed).
- Check 2a (arm P, padding probe), n = 120 pages, job 191672 (closed): not inert (median cosine .934 at 63 pads, bar
  .9999).
- provisional (rests on reasoning, no row count per batch): Check 2a split, n = 40 pages, job 191781 (closed): a
  ONE-row padded batch leaks the pads (median cosine .935 bf16 / .953 fp32 at 63 pads; .995 / 1.0000 with the pad
  zeroing forced on) — latent, no recorded run takes that path (hand-off 2de131).
- provisional (interpretation): a two-row padded batch gives median cosine .9948 bf16 / 1.0000 fp32 = the no-pad
  two-row floor .9943 / 1.0000; in fp32 its max abs difference is 2.3-2.8× that floor (median .078 at 63 pads /
  .095 at 17 vs .034), but the one-row path with zeroing forced on also reads .0759, so the gap may be rounding that
  depends on sequence shape (job 191781). Its cost on answers: check 2b, none.
- Check 2b (arm P) decided: NOT A CAUSE. Padded to 64 − not padded, judge accuracy, n = 1,500 paired questions per
  seed: seed 42 .615 − .610 = +0.5 pt [−1.1, +2.1], McNemar p .62 (job 191783, closed); seed 43 .624 − .615 =
  +0.9 pt [−0.7, +2.5], p .29 (job 191784, closed). Both CIs inside ±3 pt.
- provisional (one training, no repeat): Check 4 (arm O): slot capacity is NOT the limit (gate ≥ 90%). Word recall of
  the bf16 overfit model from its slots on the 100 trained pages .921, bootstrap 95% CI of the mean [.902, .938],
  vs the start model's .702: +21.9 pt [+18.5, +25.3], 98 up / 2 down, sign p 8e-27, n = 100 pages (job 192209,
  closed; start 191893). Held-out 100 pages .667 vs .670: −0.3 pt [−4.2, +3.6]. Text-layer ceiling .974. 116-123
  slots per page. Loss not flat at step 210 (job 192002) → .921 is a lower bound. The owner declined a repeat
  (option C, 2026-11-25).
- provisional (one seed; pooled read at the 5-pt bar; 1.52× the slots): Check 3 (arm X), n = 1,000 paired questions
  (two disjoint reads of `mm_docqa_1500_1m`, R seed 42), judge accuracy:
  - image + transcription slots − image slots: read 1 +4.6 pt [+1.0, +8.0], McNemar 51 / 28, p .013 (job 192223);
    repeat on 500 new questions +5.4 pt [+1.8, +8.8], 56 / 29, p .005 (job 192239) → passes the repeat. Pooled
    1,000: .667 vs .617, **+5.0 pt [+2.5, +7.6]**, 107 / 57, p 1.2e-4.
  - transcription slots alone − image slots: −12.0 [−16.6, −7.6] (read 1), −15.4 [−20.2, −10.6] (read 2), pooled
    **−13.7 pt [−17.1, −10.3]**, 84 / 221, p 2.3e-15 → worse.
  - Uncompressed on the same questions: .790 / .832, pooled .811. Image slots .602 / .632 (pooled .617).
  - provisional (post hoc, ad-hoc pooling script, not in the grid): per source pooled (image + text − image):
    InfoVQA +6.6 [+1.5, +11.4], MP-DocVQA +7.5 [+2.7, +12.0], SlideVQA +0.9 [−2.4, +4.2]; the net gain (+50
    questions) sits on the 878 / 1,000 questions whose answer string is in the transcription (`docs/slot_checks.md`).
  - Slots per question image 430.4 / 431.9, text 222.8 / 222.6, image + text 653.2 / 654.5 (8 tokens per slot).
    No slot-matched arm. Seed 43 not run (the law runs it only if the FIRST read clears 5 pt; it gave +4.6).
- Arm X plumbing: transcriptions round 1 (192010 / 192210 / 192211 / 192212) 1,268 / 1,268 pages, empty 0; round 2
  (192237) 1,252 / 1,252, empty 0; builds 192221 and 192238; all closed.
- Arm O plumbing: build 191891, smoke 191892, bf16 overfit 192002 (210 steps, loss .602 → .028), all closed.

**Open, and running**
- (none). Successor work sits with the pisco chats: hand-off 4c0b26 (draft + arm the stage-2 rebuild campaign from
  `SUCCESSOR.md`) and hand-off 2de131 (commit or drop the one-row pad guard in `pisco/model.py`).

## Owner REQUEST executed this wake (2026-11-25 19:00)
- REQUEST 18:40: owner's answer "A" to the end question; 18:54: both stage-2 runs on the spread layout
  (`expQ/archq3/pt_mm_5M_spread` recipe). Executed as far as this wake may:
  - `expmon campaign init` REFUSED by the guard (hard rule 3: the loop's own campaigns are the owner's). Not retried.
  - Wrote `campaign_slotcheck/SUCCESSOR.md`: the owner's words verbatim, the two runs, five design points with one
    recommendation each (main one: the collator has no per-source task mix, `pisco/mm_collator.py:313-318`, so
    "on document pages" needs a small `pisco/` change or a global share that also moves photos), gates, budget,
    reusable sets.
  - Hand-off 4c0b26 to pisco (its chats), with `--owner-said` carrying the owner's words.
  - `docs/slot_checks.md`: the owner's decision added under the campaign's answer.
  - Ledger note; REQUEST deleted; BLOCKED.md deleted (its one question is answered).
- Close-out: `expmon campaign retro` → RETRO.md (Lessons rewritten by hand), retro gripe, `expmon display --exp
  slotcheck` refreshed (32 table rows in the HTML; the "0 rows" message is gripe 43db294f), then
  `expmon campaign standby`. The owner runs `disarm` and pastes `campaign_slotcheck/SYNC.md` when convenient.

## Owned jobs (all ended, all with a verdict)
| job | arm | verdict |
|---|---|---|
| 191663 | C (check 1) | closed: not a cause |
| 191671 / 191672 / 191781 | P (2a) | closed |
| 191782 / 191783 / 191784 | P (2b) | closed: not a cause |
| 191890 / 191891 / 191892 / 191893 / 192002 / 192003 / 192209 | O (check 4) | closed |
| 192009 / 192010 / 192210-192212 / 192214-192217 / 192220-192223 / 192236-192239 | X (check 3) | closed |

## Budget at close
31 of 32 cells; 20.9 of 30 GPU-hours (all attempts, `sacct -D`, as measured on 2026-11-20; the retro's SLURM
counter says 17.3). LLM wakes 17 of 30.

## Efficiency line
SPEED (waiting for the owner 100% of the last 24 h): resolved — the owner answered; the campaign stands down, so no
further wake spends quota.

## Law notes this wake
- None changed. The law sends any stage-2 training to the owner (hard rule 9); the owner put it in a new campaign.

## Not done, and why
- The project memory file for this campaign (`~/.claude/projects/.../memory/pisco-slotcheck-campaign.md`) should say
  "closed 2026-11-25, owner chose A → successor campaign"; the wake may not write under `~/.claude`. For the chat.
