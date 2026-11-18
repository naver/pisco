# STATE — campaign slotcheck   (rewritten WHOLESALE by every LLM wake; never append)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Check 1 (arm C, feasibility gate PASS): the finetune collator's 4-document cap never removes ALL answer pages of a
  trained row — 0 / 20,000 document rows, 0 / 8,000 photo rows → not a cause of the document gap. Beside it: 1,910 /
  20,000 document rows (9.55%; all Docmatix) lose one of their two source pages. Job 191663 (verdict closed).
- Check 2a (arm P, padding probe), n = 120 pages, job 191672 (closed): not inert. Split, n = 40 pages, job 191781
  (closed): a ONE-row padded batch leaks the pads (median cosine .935 bf16 / .953 fp32 at 63 pads; .995 / 1.0000 with
  the pad zeroing forced on) — latent, no training or eval run takes that path (hand-off 2de131). A two-row padded
  batch: median .9948 bf16 / 1.0000 fp32 = the no-pad two-row floor .9943 / 1.0000 → bf16 rounding, not a pad effect
  (the worst single slot in fp32 is .985 padded vs .994 at the floor, so the medians match but the tails do not quite).
- **Check 2b (arm P) decided: NOT A CAUSE.** Padded to 64 − not padded, judge accuracy, n = 1,500 paired questions per
  seed, same node and code per pair: seed 42 .615 − .610 = **+0.5 pt [−1.1, +2.1]**, McNemar p .62 (job 191783,
  homer-020, closed); seed 43 .624 − .615 = **+0.9 pt [−0.7, +2.5]**, p .29 (job 191784, homer-019, closed). Both CIs
  inside ±3 pt. Health: 1,500 rows × 2 files each, judge on every row, 0 unparseable, no "exceeds", slot count ratio
  1.0000 (646,042 both); the unpadded arm reproduces the recorded .610 / .615. Padding changes 368 / 432 answers and
  flips the judge on 145 / 148 questions, both directions equally (bf16 rounding).
- Check 4 (arm O) baseline, job 191893 (closed): word recall of the START model from its slots, n = 100 pages each:
  **.702** on the 100 pages to overfit, **.670** on the 100 held-out; text-layer ceiling .974 / .983; reference vs
  itself 1.000 / 1.000. Pages: 120 slots each (8 visual tokens per slot, ≈ 4 words per slot).
- Arm O plumbing: build 191891 (closed: 100 + 100 pages, overlap 0, 47,498 trained docids excluded); 20-step training
  smoke 191892 (closed: 32.8 s/step, loss 0.600 → 0.488 at steps 10 → 20, MaxRSS 8.35 GB).

**Open, and running**
- Check 4: the bf16 overfit, 210 steps (≈ 14 passes over each page), job 192002 RUNNING on homer-011 since 02:03 →
  its recall on the 100 overfit + 100 held-out pages, job 192003 (afterok).
- Check 3: page transcription, 500 questions × 4 pages: 10-page plumbing job 192009 PENDING (no gpu-be node has a free
  GPU and an idle CPU at 02:09) → shard 1 of 4, job 192010 (afterok), which measures s/page for the owner checkpoint.

## Owned jobs
| job | arm | state | note |
|---|---|---|---|
| 191663 | C (check 1) | COMPLETED, 56 s | closed |
| 191671 / 191672 | P (2a) plumbing / full | COMPLETED | closed |
| 191781 | P (2a split) | COMPLETED, 9:22 | closed |
| 191782 | P (2b) plumbing | COMPLETED, 19:42 | closed (wrapper invariant PASS) |
| 191783 / 191784 | P (2b) seed 42 / 43 | COMPLETED 0:0, 55:10 / 1:17:41 | closed: not a cause; files `expQ/slotcheck/p2b_s4{2,3}/eval/*_pad{none,64}.json` |
| 191890 / 191891 | O build smoke / full | COMPLETED, CPU | closed |
| 191892 | O training smoke | COMPLETED 0:0, 16:07, homer-016 | closed |
| 191893 | O recall baseline | COMPLETED 0:0, 2:37:33, homer-016 | closed; `expQ/slotcheck/recall/{start,ref}_{dense100,heldout}.json` |
| 192002 | O bf16 overfit | RUNNING, homer-011, `--time 2:30 --mem 13G -c 1` | `pt_overfit.sbatch o_bf16 210 bf16` → `expQ/slotcheck/o_bf16/`; log `expQ/slotcheck/pt_overfit_192002.out`; ends ≈ 04:05 |
| 192003 | O recall of the overfit | PENDING (afterok 192002), `--time 4:10 --mem 8G -c 1` | → `expQ/slotcheck/recall/o_bf16_{dense100,heldout}.json`; ≈ 1:25 |
| 192009 | X transcription plumbing | PENDING (Resources), `--time 0:20 --mem 14G -c 1` | 10 pages of shard 0 → `expQ/slotcheck/transcribe/shard0of4_lim10.json` |
| 192010 | X transcription shard 0 / 4 | PENDING (afterok 192009), `--time 5:30 --mem 14G -c 1` | → `expQ/slotcheck/transcribe/shard0of4.json` + `questions.json` |

## Adopted jobs (found running in this project, not launched by the campaign — never killed, never duplicated)
(none)

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| C (check 1) | decided: not a cause | none |
| P (check 2a / 2b) | **decided: not a cause** | none |
| O (check 4) | overfit training running | when 192003 ends: `python3 -I scripts/longdoc/recall_pair.py --ref expQ/slotcheck/recall/ref_dense100.json --a …/start_dense100.json --b …/o_bf16_dense100.json` (and the held-out triple). Read: overfit recall vs 90% / 75%, gain vs start .702 (sign test), held-out must not rise above .670 beyond its CI. Plateau rule from the logged loss (`=== LOSS` block of the training log): mean of steps 161-210 within 2% of steps 111-160 → reached; else the recall is a LOWER bound and cannot read ≤ 75% as a capacity verdict. Then decide whether the fp32 cell is worth it (one variable; fp32 masters on a full-decoder 4B training may not fit 80 GB — check the smoke's memory first) |
| X (check 3) | transcription plumbing + shard 0 queued | when 192010 ends: health (< 2% empty, at-cap count, mean words) + `s_per_page_median` → projected arm X cost = 4 shards × (unique pages / 4) × s/page + 5 judged 500-Q cells; > 12 GPU-h → ONE BLOCKED question (law), else launch shards 1-3 and write `scripts/slotcheck/build_text_sets.py` |

Plan fixed now for arm X (before any result exists): sets `eval_data/slotcheck_docqa_{img,text,imgtext}_500` over the
same 500 questions (167 / 167 / 166 per source). `img` = the rows unchanged (baseline, re-run on this code and node);
`text` = the 4 transcriptions as text documents; `imgtext` = per page its image then its transcription (8 documents,
TOPK 8). Text documents are cut into 256-token chunks (`CHUNK=1`, `CML=256` = R's finetune `compressor_max_length`),
so no word of a transcription is cut and every chunk has the length R was trained on; DML = the build's measured max
prompt + 256. Caveat to report beside the result: R's finetune set has no text QA rows (image pages + photos only).

## Health evidence this wake (numbers, not adjectives)
- 191783 / 191784: sacct COMPLETED 0:0; md5 at run eval_mm_4b.sh b3d1cda1, eval_mm_docqa.py dfe32aa0, judge 02bc8748,
  judge_prompts bae44fa7, paired_judge a3cc13b2, model.py 994cff3d; `grep -c exceeds` = 0 in both logs.
- 191892 config diff vs `expQ/archq3/pt_mm_5M_spread/training_config.yaml` (the law's invariant):
  ```
  >   max_steps: 20
  <   training_dataset: …mm_pretrain_pdfa_1m_desc:80000,…wiki_1m_desc:30000,…natural_desc:50000
  >   training_dataset: …/data/slotcheck_dense100_train
  <   p_image_ae: 0.2 / p_text: 0.5 / p_image_cont: 0.05 / p_desc: 0.15 / p_find: 0.1
  >   p_image_ae: 1.0 / p_text: 0.0 / p_image_cont: 0.0 / p_desc: 0.0 / p_find: 0.0
  < model_name_or_path: …/expQ/archq3/pt_text_5M_spread/model
  > model_name_or_path: …/expQ/archq3/pt_mm_5M_spread/model
  ```
  The `model_name_or_path` line is the start by design (the law's start; the start's own config names its parent).
  `image_max_pixels` 1,003,520, bf16, save_steps 250, eval_steps 250, logging 10: identical to the start.
- 192002 sizing: max_steps 210 = (7,200 s − 311 s smoke overhead) / 32.8 s/step; `--time` 2:30 = 32.8 × 210 × 1.25 +
  311 s; `--mem` 13G = 8.35 GB × 1.5; `-c 1` because the smoke used 1.03 CPU on average (TotalCPU 16:39 over 16:07) and
  the nodes with a free GPU had ≤ 1 idle CPU. Risk: save_steps 250 > 210, so a preemption restarts from step 0.
  Live at 02:10: steps 2-4 at 35 s/step (smoke 32.8, +7% with 1 CPU) → ≈ 2:10 projected vs the 2:30 limit. If it
  TIMEOUTs, that is a wall-time misestimate (pre-authorised relaunch, `--time` from this run's s/step).
- 192003 sizing: two AE passes measured 2,360 + 2,302 s (+ ≈ 300 s plumbing and loads) × 3 = 4:10; MaxRSS 4.8 GB × 1.5.
- New code this wake (freely patchable): `scripts/slotcheck/transcribe_pages.py` d57525f3 (imports TRANSCRIBE + run_ref
  of `scripts/probe_mem_recall.py` 14730fed; identical pages once by pixel md5; lazy image load; resumes; exit 4 if > 2%
  empty) and `transcribe_pages.sbatch` 02f479e7. `py_compile` and `bash -n` pass. The plumbing job is its test.
- Budget: 15 of 32 cells; ≈ 5.7 of 30 GPU-h used; booked at limits 2.5 + 4.2 + 0.3 + 5.5, expected ≈ 5.3 → ≈ 11.

## Brief's efficiency line
- "the login's usage limit held this loop 3.5 h": the quota is shared with the owner's other chats; moving logins is the
  owner's choice (`cc-team`) and does not block this plan, so no BLOCKED question. This loop already sleeps on job ends
  (one wake per batch of results).

## Law notes this wake
- No change to the law. Arm X's set design and chunking (above) are fixed before any transcription exists.

## Blockers / needs the owner
(none)
