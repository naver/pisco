# STATE — campaign slotcheck   (rewritten WHOLESALE by every LLM wake; never append)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Check 1 (arm C, feasibility gate PASS): the finetune collator's 4-document cap never removes ALL answer pages of a
  trained row — 0 / 20,000 document rows, 0 / 8,000 photo rows → not a cause of the document gap. Beside it: 1,910 /
  20,000 document rows (9.55%; all Docmatix) lose one of their two source pages. Job 191663 (verdict closed).
- Check 2a (arm P, padding probe), n = 120 pages, job 191672 (closed): not inert. Split, n = 40 pages, job 191781
  (closed): a ONE-row padded batch leaks the pads (median cosine .935 bf16 / .953 fp32 at 63 pads; .995 / 1.0000 with
  the pad zeroing forced on) — latent, no training or eval run takes that path (hand-off 2de131). A two-row padded
  batch: median .9948 bf16 / 1.0000 fp32 = the no-pad two-row floor .9943 / 1.0000 → bf16 rounding, not a pad effect.
- **Check 2b (arm P) decided: NOT A CAUSE.** Padded to 64 − not padded, judge accuracy, n = 1,500 paired questions per
  seed, same node and code per pair: seed 42 .615 − .610 = **+0.5 pt [−1.1, +2.1]**, McNemar p .62 (job 191783,
  closed); seed 43 .624 − .615 = **+0.9 pt [−0.7, +2.5]**, p .29 (job 191784, closed). Both CIs inside ±3 pt.
- Check 4 (arm O) baseline, job 191893 (closed): word recall of the START model from its slots, n = 100 pages each:
  **.702** on the 100 pages to overfit, **.670** on the 100 held-out; text-layer ceiling .974 / .983; reference vs
  itself 1.000 / 1.000. Pages: 119-121 slots each (8 visual tokens per slot).
- Arm O plumbing: build 191891 (closed: 100 + 100 pages, overlap 0, 47,498 trained docids excluded); 20-step training
  smoke 191892 (closed: 32.8 s/step, loss 0.600 → 0.488 at steps 10 → 20, host MaxRSS 8.35 GB). Peak GPU memory was
  NOT recorded (the smoke log has none; the law's health item is missing — see Open arms, fp32 cell).
- Arm O bf16 overfit, job 192002 (closed): trained to the end, 210 steps = 14 epochs of the 936 trained rows ≈ 134
  presentations per page; image-AE loss .602 (step 10) → .028 (step 210), no NaN. **Plateau NOT reached**: mean loss
  steps 161-210 .0280 vs 111-160 .0355 = −21% (rule: within 2%) → its recall will be a LOWER bound of what training
  can store; it can say "capacity is fine" (≥ 90%) but never "capacity limit" (≤ 75%).
- Arm X transcription: plumbing 192009 (closed) and shard 0 of 4, job 192010 (closed): 317 / 1,268 unique pages,
  empty 0 / 317, 28 / 317 end within 2 tokens of the 1,024-token cap (of the 20 at ≥ 1,024: about 8 greedy
  repetition loops, 12 dense pages cut), 189 words per page on average, 1.07 GPU-h per shard. Owner checkpoint read:
  projected arm X ≤ 6.9 GPU-h < 12 → no owner question; shards 1-3 launched.

**Open, and running**
- Check 4: recall of the bf16 overfit model on the 100 overfit + 100 held-out pages, job 192209 RUNNING on homer-011
  since 10:35 (relaunch of 192003, whose dependency died; ≈ 1:25 expected, limit 4:10).
- Check 3: transcription shards 1, 2, 3 of 4, jobs 192210 / 192211 / 192212 PENDING (Priority), ≈ 1:05 each.

## Owned jobs
| job | arm | state | note |
|---|---|---|---|
| 191663 | C (check 1) | COMPLETED | closed |
| 191671 / 191672 / 191781 | P (2a) | COMPLETED | closed |
| 191782 / 191783 / 191784 | P (2b) | COMPLETED | closed: not a cause |
| 191890 / 191891 / 191892 / 191893 | O build smoke / build / training smoke / recall baseline | COMPLETED | closed |
| 192002 | O bf16 overfit | FAILED exit 3 on a 3rd attempt (attempt 2 batch COMPLETED 0:0, model complete) | closed; `expQ/slotcheck/o_bf16/{model,checkpoint-210}`; loss in `checkpoint-210/trainer_state.json` (the good attempt's log was overwritten) |
| 192003 | O recall of the overfit | CANCELLED (DependencyNeverSatisfied, never ran) | closed; relaunched as 192209 |
| 192209 | O recall of the overfit | RUNNING homer-011, `--time 4:10 --mem 8G -c 1` | → `expQ/slotcheck/recall/o_bf16_{dense100,heldout}.json`; log `expQ/slotcheck/recall_192209.out` |
| 192009 / 192010 | X transcription plumbing / shard 0 | COMPLETED | closed; `expQ/slotcheck/transcribe/shard0of4.json` + `questions.json` |
| 192210 / 192211 / 192212 | X transcription shards 1 / 2 / 3 | PENDING (Priority), `--time 3:15 --mem 14G -c 1` | → `expQ/slotcheck/transcribe/shard{1,2,3}of4.json` |

## Adopted jobs (found running in this project, not launched by the campaign — never killed, never duplicated)
(none)

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| C (check 1) | decided: not a cause | none |
| P (check 2a / 2b) | decided: not a cause | none |
| O (check 4) | recall 192209 running | when it ends: `python3 -I scripts/longdoc/recall_pair.py --ref expQ/slotcheck/recall/ref_dense100.json --a …/start_dense100.json --b …/o_bf16_dense100.json` (and the held-out triple). Read: overfit recall vs 90% (lower bound: plateau not reached), gain vs start .702 (sign test), held-out must not rise above .670 beyond its CI. Then the fp32 cell: worth it only if bf16 < 90% (a lower bound below 90% leaves "training or capacity" open; fp32 masters are the lost-update lever). If launched: `pt_overfit.sbatch o_fp32 <steps> fp32` needs a 20-step fp32 smoke first for s/step AND peak GPU memory (add a GPU-memory sampler to the sbatch as logging plumbing) — `max_steps` from the 2 GPU-h rule at the fp32 s/step |
| X (check 3) | shards 1-3 queued | when 192210-192212 end (health: empty < 2%, at-cap count): ONE chain of 4 = `build_text_sets.sbatch 30 _smoke` (CPU, `-c 4 --mem 16G --time 0:30`) → `eval_x.sbatch x_s42 img,text,imgtext 30 _smoke` with `--env JUDGE=0` (plumbing; `--time 0:45`) → `build_text_sets.sbatch` (full, `--time 1:00`) → `eval_x.sbatch x_s42` (full, `--time` = 191783's 27.6 min per 1,500 Q → 3 × 500 Q ≈ 35 min incl. loads, × 3 ≈ 1:45; `--mem 8G`), each `--dependency afterok:<previous>`. Then `scripts/paired_judge.py` text − img and imgtext − img (n = 500), uncompressed = the 500 rows of `expQ/base_qwen35_4b/eval/mm_docqa_base_px1003520_mm_docqa_1500_1m.json` (subset by query_id), and the img cell's predictions vs the recorded R s42 file on the same 500 (reproduction check) |

Arm X plan, fixed before any result (unchanged): sets `eval_data/slotcheck_docqa_{img,text,imgtext}_500` over the
same 500 questions (167 / 167 / 166 per source). `img` = the rows unchanged (image bytes copied as is); `text` = the 4
transcriptions as text documents; `imgtext` = per page its image then its transcription (8 documents, TOPK 8). Text
documents are cut into 256-token chunks (`CHUNK=1`, `CML=256`), one "Document j:" per page; DML = the build's measured
max prompt + 256 (`<set>/dml.txt`). Caveats reported beside the result: R's finetune set has no text QA rows; 28 / 317
transcriptions in shard 0 end at the 1,024-token cap (≈ 8 of them repetition loops).

## Health evidence this wake (numbers, not adjectives)
- 192002: `sacct -D`: attempt 1 PREEMPTED 02:04-03:37 (1:32:46), attempt 2 batch COMPLETED 0:0 03:39-05:45
  (2:05:46, MaxRSS 8.3 GB), attempt 3 FAILED 3:0 in 0 s, log = "refusing to overwrite …/o_bf16/model" (known_issues
  rows "FAILED exit=3 … refusing to overwrite" and "afterok child … DependencyNeverSatisfied"). `checkpoint-210/trainer_state.json`: global_step 210 = max_steps, epoch 14.0; loss every 10 steps .602 .473
  .374 .276 .220 .165 .117 .091 .072 .056 .047 .043 .038 .032 .033 .031 .029 .028 .029 .026 .028; grad norm 1.8 → 0.3.
  `training_config.yaml` differs from the smoke's only in `max_steps: 210` (output paths filtered).
- 192010: COMPLETED 0:0, 1:04:14, MaxRSS 9.0 GiB, RESULT `pages=317 empty=0 at_cap=28 mean_words=189 max_tokens=1026
  s_per_page_median=8.6 wall_s=3806`. Shards 1-3 sizing: `--time` 3:15 = 1:04:14 × 3; `--mem` 14G = 9.0 GiB × 1.5.
- Arm X cost (law owner checkpoint, 12 GPU-h): 4 × 1.07 + 0.07 + ≤ 2.5 (5 judged 500-Q cells; 191783 ran 2 × 1,500 Q
  in 55 min) = ≤ 6.9 GPU-h → below the bar.
- New code this wake (freely patchable, `scripts/slotcheck/`): `build_text_sets.py` 604ffd1b (`py_compile` OK; re-derives
  the 500 questions with `transcribe_pages.pick_questions`, checks every page key against `questions.json` AND that the
  copied image bytes decode to the transcribed pixels; audit through `scripts/longdoc/audit_inputs.py` `eval_row`,
  imported as a tool with the cell's `chunk_docs=True` collator; exit 4 on cut > 0, empty > 2%, prompt + 256 > 2048, a
  chunk > CML), `build_text_sets.sbatch` 2e1d483f and `eval_x.sbatch` f9412ad1 (`bash -n` OK; one node runs the three
  sets through the unchanged `eval_mm_4b.sh`). None has run yet: the smoke chain above is their test.
- Patch (sbatch plumbing, invariant `bash -n` OK; the training command is untouched): `pt_overfit.sbatch` 44e3fe86 →
  875feaab — a requeue whose model is complete at `max_steps` (trainer_state global_step) exits 0 instead of 3, and
  `--open-mode=append` keeps every attempt's log. Symptom: `pt_overfit.sbatch` — "refusing to overwrite …/model" (exit
  3) killed the afterok recall chain. Frozen for 192002 (already ran); used by a future fp32 cell.
- Cell dir `expQ/slotcheck/x_s42/model` → `expQ/ft_mm_4b_184778/model` (R seed 42) created for arm X.
- Budget: 19 of 32 cells (this wake: 192209-192212); ≈ 10.4 of 30 GPU-h used (192002 3.6 of it, 1.5 lost to the
  preemption); in flight ≈ 1.4 + 3.2 expected → ≈ 15; arm X evals ≈ 2 more.

## Verifier findings of the 02:00 wake, fixed
- "≈ 14 passes over each page" was wrong: 14 EPOCHS of the 936 trained rows ≈ 134 presentations per page (10 copies
  per page, 64 copies held in the eval split). Corrected above.
- Smoke peak GPU memory: not recorded; stated as missing above, planned for the fp32 smoke.
- `transcribe_pages.{py,sbatch}` were untracked: committed by the tick in 783c8f8 (`git log -- scripts/slotcheck/`).

## Law notes this wake
- No change to the law.
- The 06:00 usage-limit ALERT is stale (the quota reset; this wake ran). Deleting it was refused by the guard: it is
  the owner's file, left for the owner.

## Blockers / needs the owner
(none)
