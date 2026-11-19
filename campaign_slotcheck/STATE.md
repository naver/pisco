# STATE — campaign slotcheck   (rewritten WHOLESALE by every LLM wake; never append)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Check 1 (arm C, feasibility gate PASS): the finetune collator's 4-document cap never removes ALL answer pages of a
  trained row — 0 / 20,000 document rows, 0 / 8,000 photo rows → not a cause of the document gap. Beside it: 1,910 /
  20,000 document rows (9.55%; all Docmatix) lose one of their two source pages. Job 191663 (verdict closed).
- Check 2a (arm P, padding probe), n = 120 pages, job 191672 (closed): not inert (median cosine .934 at 63 pads, bar
  .9999). Split, n = 40 pages, job 191781 (closed): a ONE-row padded batch leaks the pads (median cosine .935 bf16 /
  .953 fp32 at 63 pads; .995 / 1.0000 with the pad zeroing forced on) — latent, no training or eval run takes that path
  (hand-off 2de131). A TWO-row padded batch: median cosine .9948 bf16 / 1.0000 fp32 = the no-pad two-row floor .9943 /
  1.0000, BUT in fp32 its max abs difference is 2.3-2.8× that floor (median .078 at 63 pads / .095 at 17 vs .034) → a
  small residual pad effect on the trained path, not bf16 rounding; it fails the law's max-abs inert bar. Its cost on
  answers is check 2b (none).
- **Check 2b (arm P) decided: NOT A CAUSE.** Padded to 64 − not padded, judge accuracy, n = 1,500 paired questions per
  seed, same node and code per pair: seed 42 .615 − .610 = **+0.5 pt [−1.1, +2.1]**, McNemar p .62 (job 191783,
  closed); seed 43 .624 − .615 = **+0.9 pt [−0.7, +2.5]**, p .29 (job 191784, closed). Both CIs inside ±3 pt.
- **Check 4 (arm O) decided: slot capacity is NOT the limit** (gate ≥ 90%). Word recall of the bf16 overfit model from
  its slots on the 100 trained pages **.921, bootstrap 95% CI of the mean [.902, .938]** (1.4% of 2,000 resamples under
  .90; recomputed this wake with the tool's own normalisation, probe on job 192209) vs the start model's **.702**:
  **+21.9 pt [+18.5, +25.3]**, 98 pages up / 2 down, sign p 8e-27, n = 100 pages (job 192209, closed; start 191893).
  Held-out 100 pages .667 vs .670: −0.3 pt [−4.2, +3.6] → no general gain, the cell measured memorisation. Text-layer
  ceiling .974. 116-123 slots per page (8 visual tokens per slot). The loss had not flattened (−21% over the last 50
  steps, rule ≤ 2%), so .921 is a LOWER bound. One training, one seed: "consistent with: the limit is training, not the
  number of slots".
- **Check 3 (arm X) first read**, n = 500 paired questions, R seed 42, all three sets in ONE job on ONE node (job
  192223, homer-014, COMPLETED 0:0 in 36:49; closed): judge accuracy image slots **.602**, transcription slots **.482**,
  image + transcription slots **.648**; uncompressed **.790** on the same 500.
  - text − image **−12.0 pt [−16.6, −7.6]**, McNemar 39 vs 99, p 3.4e-7 → by the law "the transcription path is
    worse" (≤ −5 pt, CI below 0). One seed; per source −7.2 / −16.2 / −12.7 pt (InfoVQA / MP-DocVQA / SlideVQA).
  - image+text − image **+4.6 pt [+1.0, +8.0]**, McNemar 51 vs 28, p .013; per source +2.4 / +9.6 [+3.6, +16.2] /
    +1.8 pt, none below 0 → under the 5-pt bar, CI above 0, CI not inside ±5 → **PROMISING** by the owner rule
    (CAMPAIGN.md "Promising, then ONE repeat") → one repeat on new questions (running). Seed 43 not run: the law runs
    it only if the first read clears the 5-pt bar.
  - Reproduction: the image cell equals the recorded R s42 file on the same 500 questions (0 discordant, .602 = .602).
  - Slots per question image 430.4 / text 222.8 / image+text 653.2 (= the sum; 1.52× image). Split by answer string
    in the transcription (424 / 500 questions): image .630 / text .514 / image+text .684 / uncompressed .830; on the
    other 76: .447 / .303 / .447 / .566.
- Arm X transcription, round 1 (4 shards 192010 / 192210 / 192211 / 192212, closed): 1,268 / 1,268 pages, empty 0,
  124 / 1,268 at the 1,024-token cap. Build 192221 (closed): 500 rows × 3 sets; its exit 4 = the cut-rule misfire on
  1 / 500 questions (fixed this wake in `build_text_sets.py` 6b438662; the round-1 re-audit in job 192238 decodes it).
- Arm O plumbing / training: build 191891, smoke 191892, bf16 overfit 192002 (210 steps, loss .602 → .028), all
  closed. fp32 second training not run: arm O ≈ 7.9 GPU-h vs its ≈ 5 budget line.

**Open, and running**
- Check 3 repeat (arm X, round 2): the next 167 / 167 / 166 questions per source of `mm_docqa_1500_1m` (disjoint
  query_ids; 2,000 page slots, 1,252 unique pages), same model R s42, same code. Chain, each `afterok` the previous:
  transcription smoke 192236 (passed, closed) → full transcription 192237 (1,252 pages, RUNNING homer-017 since
  16:52; 6-9 h projected from the smoke, `--time 15:00`) → build
  192238 (round-2 sets + round-1 re-audit with the fixed cut rule) → judged eval 192239 (image / text / image+text).
  Gate fixed before launch: passes if image+text − image > 0 with the 95% CI above 0 and no source's CI wholly below
  0; else check 3's image+text read is closed with both reads. Pooled 1,000-question difference reported beside it vs
  the 5-pt bar, never gating.

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
| 192215 / 192216 / 192217 | X eval smoke / full build / full eval | CANCELLED, never ran | closed |
| 192220 | X eval plumbing, 30 Q × 3 sets, JUDGE=0 | COMPLETED | closed (plumbing passed) |
| 192221 | X full build round 1 | FAILED exit 4 (cut-rule misfire, sets complete) | closed; `eval_data/slotcheck_docqa_{img,text,imgtext}_500` |
| 192222 | X full eval (first try) | CANCELLED, never ran | closed |
| 192223 | X judged eval round 1, R s42, 3 × 500 Q | COMPLETED 0:0 36:49 homer-014 | closed: first read; `expQ/slotcheck/x_s42/eval/mm_docqa_pisco_*_slotcheck_docqa_{img,text,imgtext}_500.json` |
| 192236 | X transcription smoke round 2 (10 pages) | COMPLETED 0:0 5:14 homer-017 | closed (plumbing passed): 10 / 10 pages, empty 0, 3 at the cap |
| 192237 | X transcription round 2, 1 shard, 1,252 pages | RUNNING homer-017 | `--time 15:00`; out `expQ/slotcheck/transcribe_r2/shard0of1.json` |
| 192238 | X build round 2 + round-1 re-audit (CPU) | PENDING (afterok 192237) | `--time 0:15`; sets `eval_data/slotcheck_docqa_*_500_r2`, audits `expQ/slotcheck/textsets_192238{,_r1}/` |
| 192239 | X judged eval round 2, R s42, 3 × 500 Q | PENDING (afterok 192238) | `--time 1:51`; argv `x_s42 img,text,imgtext all _r2` |

## Adopted jobs (found running in this project, not launched by the campaign — never killed, never duplicated)
(none)

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| C (check 1) | decided: not a cause | none |
| P (check 2a / 2b) | decided: not a cause | none |
| O (check 4) | decided: capacity is not the limit | none |
| X (check 3) | first read: text worse (−12.0), image+text promising (+4.6); repeat chain 192236-192239 | when 192239 ends: health (500 rows per set, judge on all, 0 unparseable, 0 empty, no "exceeds"; 192238's `round-1 re-audit` RESULT shows `text_cut_boundary=1` for the round-1 misfire, round-2 `cut - cut_boundary = 0`, `round 2: … questions of round 1` absent); `python3 scripts/paired_judge.py <imgtext_r2> <img_r2>` and `<text_r2> <img_r2>`; uncompressed = the same 500 query_ids of `expQ/base_qwen35_4b/eval/mm_docqa_base_px1003520_mm_docqa_1500_1m.json`; reproduction = img_r2 vs the recorded R s42 file on those 500; pooled 1,000-Q difference (concatenate both reads' pairs). Then the check 3 verdict, `docs/slot_checks.md`, and the ONE end-of-campaign owner question (BLOCKED.md) |

Arm X repeat files: transcriptions `expQ/slotcheck/transcribe_r2/`; sets `eval_data/slotcheck_docqa_{img,text,imgtext}_500_r2`;
results `expQ/slotcheck/x_s42/eval/mm_docqa_pisco_{,chunk_cml256_,k8_…}…_500_r2.json` (names follow eval_mm_4b.sh).

## Health evidence this wake (numbers, not adjectives)
- 192223: `sacct` COMPLETED 0:0 36:49 homer-014, one attempt; each set 500 / 500 rows, 500 unique query_ids, judge on
  500, `judge_unparseable` 0, empty predictions 0, `grep -c exceeds` 0, three `EXIT CODE 0`. `topk_docs` 4 / 4 / 8,
  `image_max_pixels` 1,003,520, greedy (`num_samples` 0).
- Round-2 smoke 192236: `sacct` COMPLETED 0:0 5:14 homer-017; FACTORS line shows `round=2` (the env reached the job);
  "500 questions, 2000 page slots, 1252 unique pages"; 10 / 10 pages, empty 0, 3 at the 1,024-token cap, median
  25.2 s/page (round-1 smoke 13.6, round-1 shards 8.6-9.7): the full run projects to 6-9 h, inside `--time 15:00`. Question picking tested on a fake 1,500-row dataset (stdlib, frontend, seconds): round 1 = the first
  block per source (unchanged), round 2 = the second block, overlap 0; the build also fails if any round-2 query_id is
  in round 1's `questions.json`.
- Patches this wake (all `scripts/slotcheck/`, freely patchable; `bash -n` passes on the 3 sbatch files,
  `py_compile` on the 2 .py): `transcribe_pages.py` 03867f94 (`--round`), `transcribe_pages.sbatch` 185a70e2 (env
  ROUND → `--round`, own out dir), `build_text_sets.py` 6b438662 (`--round`, overlap check, `cut_boundary` re-check:
  a cut row whose answer is found in the chunks joined as they are, whitespace ignored, does not fail),
  `build_text_sets.sbatch` a84224d6 (env ROUND; round-1 re-audit first, does not gate), `eval_x.sbatch` 364eadd3
  (`all` = no limit). Shared files unchanged: eval_mm_4b.sh b3d1cda1 / eval_mm_docqa.py dfe32aa0 / judge_mm_docqa.py
  02bc8748 / paired_judge.py a3cc13b2 / pisco/model.py 994cff3d.
- `--time`: smoke 0:15 (192009 222 s × 3); transcription 15:00 (measured 12.56 s/page × 1,252 = 4.4 h × 3 = 13.1 h;
  set before the page count was known, from ≤ 1,400 pages); build 0:15 (192221 2:19 × 2 builds × 3); eval 1:51
  (192223 36:49 × 3). argv checked in `.expmon/runs.jsonl`: 192236 `0 1 10` env ROUND=2; 192237 `0 1` ROUND=2;
  192238 none, ROUND=2; 192239 4 args `x_s42 img,text,imgtext all _r2`.
- Budget: 31 of 32 cells submitted (this wake 192236-192239). GPU-hours, all attempts from `sacct -D`: 16.05 so far
  (`expmon campaign report` says 12.4: it counts final attempts only); the repeat adds ≈ 5.0 → ≈ 21 of 30. Arm X ≈ 10.3
  of its ≈ 12 line. One cell of slack left for a relaunch.

## Verifier findings of the 15:30 wake, fixed
- Check 4 "no CI on the mean .921": the bootstrap 95% CI of the mean over the 100 pages is [.902, .938] (probe on
  192209, `docs/slot_checks.md` check 4 table and health paragraph). Still one training, one seed, lower bound.
- Check 3 "rests on a build that failed its own cut rule, inferred not decoded": the build's cut rule is fixed
  (`cut_boundary`), and job 192238 re-audits the round-1 sets with it (its `round-1 re-audit` RESULT line decodes the
  37710.jpeg-9 row). Worst case it changes 1 / 500 = 0.2 pt.

## Law notes this wake
- Tightened CAMPAIGN.md (Promising rule): for check 3 the repeat's new items are the next 167 / 167 / 166 questions per
  source of `mm_docqa_1500_1m`, because the named document sets reuse the gating questions (docs/spread_cases.md:63-65).

## Blockers / needs the owner
(none). The end-of-campaign owner question (four verdicts + the stage-2 rebuild each motivates) comes after the check 3
repeat. To raise there: image+text spends 1.52× the slots of image alone, so its gain cannot be split between "the
words" and "more slots" — no slot-matched arm exists in the law.
