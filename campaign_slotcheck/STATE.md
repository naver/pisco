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
  1.0000, BUT in fp32 its max abs difference is 2.3-2.8× that floor (median .078 at 63 pads / .095 at 17 vs .034;
  relative L2 .006 vs .003) → a small residual pad effect remains on the trained path, not bf16 rounding; it fails the
  law's max-abs inert bar. Its cost on answers is check 2b (none).
- **Check 2b (arm P) decided: NOT A CAUSE.** Padded to 64 − not padded, judge accuracy, n = 1,500 paired questions per
  seed, same node and code per pair: seed 42 .615 − .610 = **+0.5 pt [−1.1, +2.1]**, McNemar p .62 (job 191783,
  closed); seed 43 .624 − .615 = **+0.9 pt [−0.7, +2.5]**, p .29 (job 191784, closed). Both CIs inside ±3 pt.
- **Check 4 (arm O) decided: slot capacity is NOT the limit** (gate ≥ 90%). Word recall of the bf16 overfit model from
  its slots on the 100 trained pages **.921** vs the start model's **.702**: **+21.9 pt [+18.5, +25.3]**, 98 pages up /
  2 down, sign p 8e-27, n = 100 pages; median .951, 76 / 100 pages ≥ .90, 7 / 100 ≤ .75 (job 192209, closed; start
  191893). Held-out 100 pages .667 vs .670: −0.3 pt [−4.2, +3.6] → no general gain, the cell measured memorisation.
  Text-layer ceiling .974 (.921 = 94.6% of it); a file paired with itself 0.000. 116-123 slots per page (8 visual
  tokens per slot, ≈ 4 words per slot), identical before and after on every page. The loss had not flattened (−21%
  over the last 50 steps, rule ≤ 2%), so .921 is a LOWER bound — which is the direction that decides "capacity is
  fine". One training, one seed: "consistent with: the limit is training, not the number of slots".
- Arm O plumbing: build 191891 (closed: 100 + 100 pages, overlap 0, 47,498 trained docids excluded); 20-step training
  smoke 191892 (closed: 32.8 s/step, loss 0.600 → 0.488 at steps 10 → 20, host MaxRSS 8.35 GB, peak GPU memory
  52.0 GB of 80 from SLURM accounting).
- Arm O bf16 overfit, job 192002 (closed): trained to the end, 210 steps = 14 epochs of the 936 trained rows ≈ 131
  presentations per page; image-AE loss .602 (step 10) → .028 (step 210), no NaN.
- Arm O fp32 second training: NOT run. The law (CAMPAIGN.md "Check 4") runs it only if the budget line holds; arm O's
  line is ≈ 5 GPU-h and arm O has used ≈ 7.9 (smoke 191892 0.27 + recall baseline 191893 2.63 + training 192002 3.6 +
  recall 192209 1.45, from `sacct` / the 192002 probe). bf16 already clears the 90% bar.
- Arm X transcription, all 4 shards (jobs 192010 / 192210 / 192211 / 192212, closed): 1,268 / 1,268 unique pages of the
  500 questions, empty 0 (bar < 2%), 124 / 1,268 = 9.8% end within 2 tokens of the 1,024-token cap (28 / 31 / 23 / 42
  per shard), 187-207 words per page on average, ≈ 4.4 GPU-h.
- Arm X build plumbing, job 192214 (closed, decides nothing): the 30-question smoke sets pass on data (empty 0, cut 0 in
  all 3 sets, `<MEM>`s per question mean img 380.1 / text 80.4 / imgtext 460.5) but the job exited 4 on a bug in its
  own health rule (fixed in `build_text_sets.py` 1c110e22). Its chain children 192215-192217 never ran (closed).

**Open, and running**
- Check 3 (arm X): the judged eval of R seed 42 on the three 500-question sets (image slots / transcription slots /
  both), one job on one node — job 192223, RUNNING on homer-014 since 15:43, `--time 1:45`. Its plumbing passed this
  wake: eval smoke 192220 (30 Q × 3 sets, closed) and the full build 192221 (500 Q × 3 sets complete, closed; its
  exit 4 is a misfire of the cut rule on 1 / 500 questions, see Health).

## Owned jobs
| job | arm | state | note |
|---|---|---|---|
| 191663 | C (check 1) | COMPLETED | closed |
| 191671 / 191672 / 191781 | P (2a) | COMPLETED | closed |
| 191782 / 191783 / 191784 | P (2b) | COMPLETED | closed: not a cause |
| 191890 / 191891 / 191892 / 191893 | O build smoke / build / training smoke / recall baseline | COMPLETED | closed |
| 192002 | O bf16 overfit | FAILED exit 3 on a 3rd attempt (attempt 2 COMPLETED 0:0, model complete) | closed; `expQ/slotcheck/o_bf16/{model,checkpoint-210}` |
| 192003 | O recall (dead dependency) | CANCELLED, never ran | closed; relaunched as 192209 |
| 192209 | O recall of the overfit | COMPLETED | closed: check 4 decided |
| 192009 / 192010 / 192210 / 192211 / 192212 | X transcription plumbing / shards 0-3 | COMPLETED | closed; `expQ/slotcheck/transcribe/` |
| 192214 | X build smoke (30 Q) | FAILED exit 4 (health-rule bug) | closed |
| 192215 / 192216 / 192217 | X eval smoke / full build / full eval | CANCELLED, never ran | closed; relaunched as 192220 / 192221 / 192223 |
| 192220 | X eval plumbing, 30 Q × 3 sets, JUDGE=0 | COMPLETED 0:0 8:54 homer-014 | closed (plumbing passed); `expQ/slotcheck/x_s42/eval/*_smoke_lim30.json` |
| 192221 | X full build, 500 Q × 3 sets | FAILED exit 4 in 2:19 homer-014 (cut-rule misfire) | closed: sets complete; `eval_data/slotcheck_docqa_{img,text,imgtext}_500`, audit `expQ/slotcheck/textsets_192221/audit.json` |
| 192222 | X full eval (first try) | CANCELLED, never ran (DependencyNeverSatisfied after 192221's exit 4) | closed; relaunched as 192223 |
| 192223 | X full judged eval, R s42, 3 × 500 Q | RUNNING homer-014 | `expQ/slotcheck/x_s42/eval/mm_docqa_pisco_*_slotcheck_docqa_{img,text,imgtext}_500.json` |

## Adopted jobs (found running in this project, not launched by the campaign — never killed, never duplicated)
(none)

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| C (check 1) | decided: not a cause | none |
| P (check 2a / 2b) | decided: not a cause | none |
| O (check 4) | decided: capacity is not the limit | none (fp32 not run: arm O over its budget line) |
| X (check 3) | full judged eval 192223 running | when 192223 ends: health (500 rows per set, judge on all, 0 unparseable, no "exceeds"); `python3 scripts/paired_judge.py <text>.json <img>.json` and `<imgtext> <img>` (n = 500, bar 5 pt with CI on the far side of 0); uncompressed = the same 500 query_ids of `expQ/base_qwen35_4b/eval/mm_docqa_base_px1003520_mm_docqa_1500_1m.json`; reproduction check = the img cell vs the recorded R s42 file `expQ/ft_mm_4b_184778/eval/mm_docqa_pisco_px1003520_mm_docqa_1500_1m.json` on the same 500. Seed 43 only if a read clears the bar. Then the check 3 table in `docs/slot_checks.md` and the ONE end-of-campaign owner question (BLOCKED.md) |

Arm X plan, fixed before any result (unchanged): sets `eval_data/slotcheck_docqa_{img,text,imgtext}_500` over the
same 500 questions (167 / 167 / 166 per source). `img` = the rows unchanged; `text` = the 4 transcriptions as text
documents; `imgtext` = per page its image then its transcription (8 documents, TOPK 8). Text documents are cut into
256-token chunks (`CHUNK=1`, `CML=256`), one "Document j:" per page; DML = the build's measured max prompt + 256
(`<set>/dml.txt`: img 839 / text 862 / imgtext 1,361). Caveats reported beside the result: R's finetune set has no
text QA rows; 124 / 1,268 transcriptions end at the 1,024-token cap (some are greedy repetition loops); the answer
string appears in the transcription for 424 / 500 questions. Budgets differ by design (500-Q build): `<MEM>`s per
question img 430.4 / text 222.8 / imgtext 653.2 = img + text, so text slots are ≈ 0.52 of image slots.

## Health evidence this wake (numbers, not adjectives)
- 192220 eval smoke: `sacct` COMPLETED 0:0 8:54 homer-014, one attempt; each set 30 / 30 rows, empty predictions 0,
  no "exceeds" error, every set `EXIT CODE 0`; `<MEM>`s used img 11,404 / text 2,412 / imgtext 13,816 (= img + text
  exactly); prompt max 494 / 269 / 673 tokens. JUDGE=0 by design; decides nothing.
- 192221 full build: `sacct` FAILED 4:0 2:19 homer-014. Sets written, 500 rows each, `stats.json` + `dml.txt` present;
  empty transcriptions 0; img `<MEM>`s 215,202 = 430.4 per question (the 1,500-Q set: 430.7); text chunks ≤ 289 =
  CML + CML//8 + 1 (the patched rule passed); gold dropped 0. Exit 4 = `FAIL text / imgtext: cut 1 rows`, the same
  question 37710.jpeg-9 (answer "2016"). Not truncation: `n_max_chunks` is None and that question's 4 pages are
  chunked into 4×256 | 256+207 | 4×256 | 256+256+103 text tokens = the transcriptions' own token counts 1,024 / 463 /
  1,024 / 615, so every token reaches the compressor. "2016" sits at character 1,068 of 1,416 on page 4, near the
  512-token chunk boundary; the audit joins decoded chunks with a space and Qwen splits digits one per token, so the
  string reads as missing. Inferred, not decoded (needs the tokenizer on a compute node). Worst case 1 / 500 = 0.2 pt
  against a 5-pt bar. New known_issues.md row.
- 192223 launched without a dependency after both: `--time 1:45` = 3 × the projected 35 min (191783: 27.6 min per
  1,500 judged questions; the three sets hold 653k `<MEM>`s, the size of one 1,500-Q pass at 646k).
- Launch: 192220 needed `--force`: `expmon launch` refused it as "closed as a dead arm by run 192215"; 192215 is the
  identical cell that never ran (killed when its parent failed), no factor difference (gripe +1 of bcedcbaf). argv
  checked in `.expmon/runs.jsonl`: 192220 4 args (`x_s42 img,text,imgtext 30 _smoke`, env JUDGE=0), 192221 0 args,
  192222 / 192223 2 args (`x_s42 img,text,imgtext`).
- md5 this wake: eval_mm_4b.sh b3d1cda1 / eval_mm_docqa.py dfe32aa0 / judge_mm_docqa.py 02bc8748 / paired_judge.py
  a3cc13b2 / pisco/model.py 994cff3d / build_text_sets.py 1c110e22 / build_text_sets.sbatch 2e1d483f / eval_x.sbatch
  f9412ad1 — unchanged since the 13:30 wake.
- Budget: 27 of 32 cells (this wake 192220-192223; 192222 never ran); ≈ 15.2 of 30 GPU-h before this wake, + 0.15
  (smoke) + ≈ 0.6-1.7 for 192223. Left after check 3: 5 cells, ≈ 13 GPU-h; seed 43 (1 cell) only if a read clears
  the bar. Submission cap 4 of 4 used this wake.

## Verifier findings of the 13:30 wake, fixed
- Check 2a "two-row padding = bf16 rounding": wrong — it fails the law's max-abs bar in fp32 (2.3-2.8× the floor).
  Rewritten above and in `docs/slot_checks.md` (2a status line, table row, "What it means" (2), the 2b read lines).
- Arm O fp32 "only if bf16 stayed under 90%": no such rule in the law; the law's condition is the budget line. The
  reason is now the budget line (arm O ≈ 7.9 of ≈ 5 GPU-h), here and in `docs/slot_checks.md`.

## Law notes this wake
- No change to the law.

## Blockers / needs the owner
(none). The end-of-campaign owner question (four verdicts + the stage-2 rebuild each motivates) comes after check 3.
