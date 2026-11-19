# STATE — campaign slotcheck   (rewritten WHOLESALE by every LLM wake; never append)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Check 1 (arm C, feasibility gate PASS): the finetune collator's 4-document cap never removes ALL answer pages of a
  trained row — 0 / 20,000 document rows, 0 / 8,000 photo rows → not a cause of the document gap. Beside it: 1,910 /
  20,000 document rows (9.55%; all Docmatix) lose one of their two source pages. Job 191663 (verdict closed).
- Check 2a (arm P, padding probe), n = 120 pages, job 191672 (closed): not inert. Split, n = 40 pages, job 191781
  (closed): a ONE-row padded batch leaks the pads (median cosine .935 bf16 / .953 fp32 at 63 pads; .995 / 1.0000 with
  the pad zeroing forced on) — latent, no training or eval run takes that path (hand-off 2de131). A two-row padded
  batch: median cosine .9948 bf16 / 1.0000 fp32 = the no-pad two-row floor .9943 / 1.0000 (median cosine only: in fp32
  the max abs median .078 is 2.3× the .034 floor) → bf16 rounding, not a pad effect.
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
  smoke 191892 (closed: 32.8 s/step, loss 0.600 → 0.488 at steps 10 → 20, host MaxRSS 8.35 GB, **peak GPU memory
  52.0 GB** of 80 — SLURM accounting `TRESUsageInMax gres/gpumem=52020M`, recorded as a probe this wake).
- Arm O bf16 overfit, job 192002 (closed): trained to the end, 210 steps = 14 epochs of the 936 trained rows ≈ 131
  presentations per page; image-AE loss .602 (step 10) → .028 (step 210), no NaN.
- Arm O fp32 second training: NOT run — the plan fixed before the result ran it only if bf16 stayed under 90%.
- Arm X transcription, all 4 shards (jobs 192010 / 192210 / 192211 / 192212, closed): 1,268 / 1,268 unique pages of the
  500 questions, empty 0 (bar < 2%), 124 / 1,268 = 9.8% end within 2 tokens of the 1,024-token cap (28 / 31 / 23 / 42
  per shard), 187-207 words per page on average, 4.35 GPU-h.
- Arm X build plumbing, job 192214 (closed, decides nothing): the 30-question smoke sets were written and pass on data
  (empty 0, cut 0 in all 3 sets, `<MEM>`s per question mean img 380.1 / text 80.4 / imgtext 460.5, prompt max 494 /
  269 / 673 tokens, the answer string appears in the transcription for 26 / 30 questions) but the job exited 4 on a
  BUG in its own health rule (see Health evidence). Its chain children 192215-192217 never ran (closed, killed).

**Open, and running**
- Check 3 (arm X): nothing running. Next wake launches the eval chain (below). The submission cap of 4 was spent on the
  chain whose head failed.

## Owned jobs
| job | arm | state | note |
|---|---|---|---|
| 191663 | C (check 1) | COMPLETED | closed |
| 191671 / 191672 / 191781 | P (2a) | COMPLETED | closed |
| 191782 / 191783 / 191784 | P (2b) | COMPLETED | closed: not a cause |
| 191890 / 191891 / 191892 / 191893 | O build smoke / build / training smoke / recall baseline | COMPLETED | closed |
| 192002 | O bf16 overfit | FAILED exit 3 on a 3rd attempt (attempt 2 COMPLETED 0:0, model complete) | closed; `expQ/slotcheck/o_bf16/{model,checkpoint-210}` |
| 192003 | O recall (dead dependency) | CANCELLED, never ran | closed; relaunched as 192209 |
| 192209 | O recall of the overfit | COMPLETED 0:0 1:26:45 homer-011 | closed: check 4 decided; `expQ/slotcheck/recall/o_bf16_{dense100,heldout}.json` |
| 192009 / 192010 / 192210 / 192211 / 192212 | X transcription plumbing / shards 0-3 | COMPLETED | closed; `expQ/slotcheck/transcribe/shard{0,1,2,3}of4.json` + `questions.json` |
| 192214 | X build smoke (30 Q) | FAILED exit 4 in 1:11 (health-rule bug) | closed; sets `eval_data/slotcheck_docqa_{img,text,imgtext}_500_smoke` (complete: stats.json + dml.txt); audit `expQ/slotcheck/textsets_192214/audit.json` |
| 192215 / 192216 / 192217 | X eval smoke / full build / full eval | CANCELLED (never ran; parent failed) | closed |

## Adopted jobs (found running in this project, not launched by the campaign — never killed, never duplicated)
(none)

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| C (check 1) | decided: not a cause | none |
| P (check 2a / 2b) | decided: not a cause | none |
| O (check 4) | decided: capacity is not the limit | none (fp32 not run, by the pre-fixed plan) |
| X (check 3) | transcriptions done; build health rule patched | next wake, 3 submissions: (a) `eval_x.sbatch x_s42 img,text,imgtext 30 _smoke` `--env JUDGE=0` (`--time 0:45:00 --mem 8G -c 4 --gres=gpu:1`, plumbing on the existing smoke sets) and, in parallel, (b) `build_text_sets.sbatch` full (`--gres=gpu:0 -c 4 --mem 16G --time 1:00:00`; this is the first run of patched 1c110e22, CPU, the cheapest cell); then (c) `eval_x.sbatch x_s42 img,text,imgtext` full, `--dependency=afterok:<a>:<b>` (`--time 1:45:00 --mem 8G -c 4`). All `-p gpu-be --exclude='homer-[005-009]'`. Then `scripts/paired_judge.py` text − img and imgtext − img (n = 500), uncompressed = the 500 rows of `expQ/base_qwen35_4b/eval/mm_docqa_base_px1003520_mm_docqa_1500_1m.json` by query_id, and the img cell vs the recorded R s42 file on the same 500 (reproduction check). Bar 5 pt; seed 43 only if the first read clears it |

Arm X plan, fixed before any result (unchanged): sets `eval_data/slotcheck_docqa_{img,text,imgtext}_500` over the
same 500 questions (167 / 167 / 166 per source). `img` = the rows unchanged (image bytes copied as is); `text` = the 4
transcriptions as text documents; `imgtext` = per page its image then its transcription (8 documents, TOPK 8). Text
documents are cut into 256-token chunks (`CHUNK=1`, `CML=256`), one "Document j:" per page; DML = the build's measured
max prompt + 256 (`<set>/dml.txt`). Caveats reported beside the result: R's finetune set has no text QA rows; 124 /
1,268 transcriptions end at the 1,024-token cap (some are greedy repetition loops). The smoke shows the budgets differ
by design: text slots are ≈ 1/5 of image slots per question (80 vs 380).

## Health evidence this wake (numbers, not adjectives)
- 192209: `sacct` COMPLETED 0:0 01:26:45 homer-011 (10:34-12:01), one attempt; log prints `mem_position_stride=8` (3×)
  and `EXIT CODE 0`; `recall_pair.py` (md5 ea4a6bcd, `analyze_mem_recall.py` cdcded44) on ref / start / overfit:
  dense100 +0.219 [+0.185, +0.253] 98/2 p 8e-27; held-out −0.003 [−0.042, +0.036] 32/66; self-pair 0.000.
- 192210 / 192211 / 192212: COMPLETED 0:0, 1:05:50 / 1:05:10 / 1:07:10 (homer-011 / 018 / 018), RESULT lines:
  pages 317 each, empty 0, at_cap 31 / 23 / 42, mean_words 201 / 187 / 207 (probes recorded for 192210 / 192211).
- 192214 symptom → fix (freely patchable, `scripts/slotcheck/build_text_sets.py` 604ffd1b → 1c110e22, +10 / −2 lines):
  `build_text_sets.py — FAIL text: a text chunk of 289 tokens > CML 256 (+1 <MEM>)` and `FAIL imgtext: a text chunk
  of 903 tokens …`. Cause: `scripts/longdoc/audit_inputs.py:173` `chunk_toks = [len(c) for c in batch.input_ids]`
  counts the chunk's `<MEM>`s (256 + 256 // 8 + 1 = 289) and imgtext's image chunks. New rule: text set chunk ≤ CML +
  CML // 8 + 1; imgtext `<MEM>`s per question = img + text on every question. `py_compile` OK; the new rule replayed
  with stdlib on 192214's `audit.json`: 0 fails on the 30 real questions (max text chunk 289; additivity 30 / 30),
  and it flags both planted faults (a 290-token text chunk; one imgtext question off by one `<MEM>`).
- md5 this wake: eval_mm_4b.sh b3d1cda1 / eval_mm_docqa.py dfe32aa0 / judge_mm_docqa.py 02bc8748 / paired_judge.py
  a3cc13b2 / pisco/model.py 994cff3d / build_text_sets.sbatch 2e1d483f / eval_x.sbatch f9412ad1 — unchanged.
- Budget: 23 of 32 cells (this wake 192214-192217; 3 of them never ran); ≈ 15.2 of 30 GPU-h (192209 1.45 + shards
  3.31 this wake). Left for arm X: 3 cells next wake ≈ 1.2 GPU-h; seed 43 (1-2 cells) only if the bar clears.

## Verifier findings of the 10:30 wake, fixed
- Slots per page are 116-123 (not 119-121); the overfit is ≈ 131 presentations per page (not 134): corrected above and
  in `docs/slot_checks.md`.
- Arm O smoke peak GPU memory: was missing; now 52.0 GB from SLURM accounting (probe on 191892).
- Check 2a "fp32 two-row = floor": now qualified as median cosine only (max abs 2.3× the floor).

## Law notes this wake
- No change to the law. Deliverable `docs/slot_checks.md`: check 4 table and its stage-2 line written; status table
  updated. `expmon display --exp slotcheck` refreshed (24 `<tr>`; its "0 table rows" message is the known gripe).

## Blockers / needs the owner
(none). The end-of-campaign owner question (four verdicts + the stage-2 rebuild each motivates) comes after check 3.
