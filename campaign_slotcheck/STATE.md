# STATE — campaign slotcheck   (rewritten WHOLESALE by every LLM wake; never append)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Check 1 (arm C, feasibility gate PASS): the finetune collator's 4-document cap never removes ALL answer pages of a
  trained row — 0 / 20,000 document rows, 0 / 8,000 photo rows → not a cause of the document gap. Beside it: 1,910 /
  20,000 document rows (9.55%; all Docmatix, 19.1% of its 10,000) lose one of their two source pages; the record does
  not say which page holds the answer. Job 191663 (CPU audit, 56 s, verdict closed). Table: `docs/slot_checks.md`.
- Check 2a (arm P, padding probe) is NOT inert, n = 120 pages (job 191672, verdict closed; controls pass: same input
  twice cos 1.0000 / max abs 0 on 120 / 120, another page median cos .739; `<MEM>` count identical at every width).
  One page alone + 17 / 63 left pads: median cos .934 / .934, max abs 2.65 / 2.80 (floor 0). A question's 4 pages
  padded to 64 as in training (30 questions): median cos .9966. The law's bar (median ≥ .9999) is missed on both paths.
- Check 2a cause split, n = 40 pages × widths 17 / 63 × bf16 + fp32 (job 191781, verdict closed). (1) One-row padded
  batch: median cos .935 bf16 / .953 fp32 at 63 pads; with the pad zeroing forced on, .995 / 1.0000 → the leak is the
  installed transformers rule that zeroes pads in Qwen3.5's linear-attention layers only for batches of more than one
  row (`modeling_qwen3_5.py:198`). Latent: every training batch has ≥ 2 compressor rows (stage 2 bs 8, finetune bs 2,
  all chunks padded together in `collate_compressor`), and the eval pads nothing. Hand-off 2de131 to the pisco chat.
  (2) Two-row padded batch (masked, the path training and the eval use): .9948 bf16 / 1.0000 fp32 = the no-pad
  two-row floor .9943 / 1.0000 → bf16 batch-shape rounding, not a pad effect.
- Check 2b wrapper invariant PASS (job 191782, plumbing, verdict closed): on R s42, 300 questions, homer-020, the
  plain `eval_mm_4b.sh` and the wrapper with pad=None give 300 / 300 identical predictions and `mems_used` (118,147
  both); with pad=64, 233 / 300 predictions are identical (67 change). No judge (plumbing decides nothing).
- Arm O builder plumbing OK (job 191890, verdict closed): 3,000 scanned rows → 100 + 100 pages of 674-760 tokens
  (mean 714 / 716), overlap 0, training set 1,000 rows (10 copies), text-layer files written.

**Open, and running**
- Check 2b full pairs: 191783 (seed 42, homer-020, pad=None generation 1,220 / 1,500 at 22:07) and 191784 (seed 43,
  homer-019, 740 / 1,500), each pad=None then pad=64 on its own node, judged (rubric), then `paired_judge` 64 − None.
- Check 4 (arm O) chain: full page-set build 191891 RUNNING → 20-step training smoke 191892 and the start model's
  recall baseline 191893 (start + uncompressed reference on the 100 overfit and the 100 held-out pages), both afterok.
- Check 3 (text slots): not started.

## Owned jobs
| job | arm | state | note |
|---|---|---|---|
| 191663 | C (check 1) | COMPLETED 0:0, 56 s, homer-017 | verdict closed |
| 191671 | P (2a) plumbing | COMPLETED 0:0, homer-011 | verdict closed |
| 191672 | P (2a) full | COMPLETED 0:0, 4:04, homer-011 | verdict closed (not inert) |
| 191781 | P (2a split) | COMPLETED 0:0, 9:22, homer-011 | verdict closed; probe recorded; output `expQ/slotcheck/pad_split_191781/` |
| 191782 | P (2b) plumbing | COMPLETED 0:0, 19:42, homer-020 | verdict closed (INVARIANT PASS) |
| 191783 | P (2b) full s42 | RUNNING since 21:47, homer-020, `--time 2:45` | cell `expQ/slotcheck/p2b_s42`; ≈ 24 min per generation + judge → ends ≈ 23:00 |
| 191784 | P (2b) full s43 | RUNNING since 21:49, homer-019, `--time 2:45` | cell `expQ/slotcheck/p2b_s43`; ≈ 37 min per generation (busier node) → ends ≈ 23:20 |
| 191890 | O build smoke | COMPLETED 0:0, 1:01, homer-010, MaxRSS 1.2 GB | verdict closed (plumbing) |
| 191891 | O build full | RUNNING (started 22:10), `--time 0:30`, `--mem 16G`, CPU | scan 60,000 → `data/slotcheck_dense100_train`, `eval_data/slotcheck_dense100{,_heldout}`, `expQ/slotcheck/dense100_191891/` |
| 191892 | O training smoke | PENDING (afterok 191891), `--time 0:55`, `--mem 64G`, `-c 4` | `pt_overfit.sbatch o_smoke 20` → `expQ/slotcheck/o_smoke/`; log `expQ/slotcheck/pt_overfit_191892.out` |
| 191893 | O recall baseline | PENDING (afterok 191891), `--time 4:30`, `--mem 14G` | 4 passes → `expQ/slotcheck/recall/{start,ref}_{dense100,heldout}.json` (+ `start_dense100_lim5.json` plumbing) |

## Adopted jobs (found running in this project, not launched by the campaign — never killed, never duplicated)
(none — 191785 mech_probes and the eval_mnt jobs belong to the siblings spread and longdoc)

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| C (check 1) | **decided: not a cause** | none |
| P (check 2a / 2b) | 2a decided and split; 2b full pairs running (6 + 1 split + 1 plumbing cells used) | when 191783 / 191784 end: health (1,500 rows × 2 files each, judge on every row, 0 unparseable, no "exceeds", `mems_used` ratio within ±1%), then `paired_judge` 64 − None per seed vs the gate (≥ +3 pt CI > 0 both seeds → eval fault hand-off; CI inside ±3 both → not a cause; 1.2-3 CI > 0 → PROMISING, photo repeat with the one-row rule now in the law). Keep `.kept_` copies if a requeue threatens a complete file. Write the check 2b table in `docs/slot_checks.md` |
| O (check 4) | build running; smoke + baseline queued | smoke 191892: read s/step, peak GPU memory, MaxRSS, the loss at steps 10 / 20, and the CONFIG DIFF block (must show only data path, the five task-mix fields, `max_steps`, output paths — `hydra.run.dir`, `PT_OUT` — paste it here). Then launch the bf16 overfit: `pt_overfit.sbatch o_bf16 <STEPS>` with STEPS = min(1,000, (7,200 s − smoke load s) / smoke s/step), `--time` = s/step × STEPS × 1.25 + load, `--mem` = smoke MaxRSS × 1.5; chain its recall `recall.sbatch o_bf16_dense100:<model dir>:slotcheck_dense100 o_bf16_heldout:<model dir>:slotcheck_dense100_heldout` afterok. The model dir is `expQ/slotcheck/o_bf16/model`. Risk: `save_steps 250` > STEPS, so a gpu-be preemption restarts the training from step 0 (save_steps is not in the law's allowed config diff) |
| X (check 3) | not started | last (biggest cost); first transcription shard measures the 12 GPU-h checkpoint |

## Health evidence this wake (numbers, not adjectives)
- 191781: sacct COMPLETED 0:0, 9:22, homer-011; 40 pages; bf16 pass 103 s, fp32 pass 362 s; repeat exact (max abs 0)
  in bf16, max abs 0 in fp32. RESULT keys re-sent with `expmon probe` (expmon reads no RESULT line, known issue).
- 191782: COMPLETED 0:0, 19:42, homer-020; `INVARIANT PASS: plain vs wrapper_none` printed; md5 at run: eval_mm_4b.sh
  b3d1cda1, scripts/eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py 02bc8748, judge_prompts.py bae44fa7,
  paired_judge.py a3cc13b2, pisco/model.py 994cff3d, eval_pad.py cb7187b2, compare_preds.py 0c227be5.
- 191890: COMPLETED 0:0, 1:01; RESULT ok=True; no scanned docid among the start model's 80,000 trained pdfa rows.
- New / changed code this wake (freely patchable, `scripts/slotcheck/`): `build_dense100.py` b7170cf5 (before its
  first run: pages over 760 WORDS dropped unread, every other candidate tokenized in one batch — a top-4,000-by-words
  cut would mostly exceed the 760-token cap; `--copies 10`; writes `textlayer_{train,heldout}.json`);
  `pt_overfit.sbatch` 44e3fe86 (calls the shared `pt_mm_4b.sh` 3b63ee99 with the start's recipe as `++` overrides:
  lr 3e-5, KD .5 with `teacher_inputs`, 1,003,520 px, task mix image-AE only, `max_steps`; prints the config diff
  and the loss log); `recall.sbatch` b61a7b4c (runs `scripts/longdoc/mem_recall.py` 9f241620 per NAME:ARM:SET,
  a 5-page plumbing pass before the first AE pass must show stride 8 and non-empty text). `bash -n` passes on all
  three sbatch files; `py_compile` passes on the builder.
- Why the copies: `pisco/train.py:197` moves `min(64, rows // 10)` random rows (no seed) to the eval split.
- Checked before writing the launcher: an AE-only batch has no KD rows, and `build_teacher_inputs` returns empty
  (0, 1) tensors for that case (`pisco/mm_utils.py:275`); start model's stage 2 (job 184777) ran 43.5 s/step at
  bs 8 × accum 8, MaxRSS 42.8 GB, 2,499 steps in 30 h.
- Budget: 11 of 32 cells, ≈ 0.8 of 30 GPU-h used (+ booked: 2b pairs ≤ 5.5, smoke ≤ 0.9, baseline ≤ 4.5 at limits;
  expected ≈ 3.5 actual).

## Law notes this wake
- Tightened check 4 (before any arm O cell): `max_steps` from the smoke within 2 GPU-h; plateau read afterwards (not
  reached → the recall is a lower bound, never a ≤ 75% capacity verdict); added the text-layer ceiling control; the
  training set repeats each page 10 times.
- Tightened check 2b's photo repeat: one-row questions take the one-row pad leak when padded → counted and reported
  apart, not gating.
- Corrected in `docs/slot_checks.md`: the earlier guess "a photo question in training is a one-row batch" was wrong
  (training batches always hold ≥ 2 compressor rows).
- The 2a inert test's "max abs ≤ 2 × the repeat floor" needs bit-identical vectors (floor 0); left as written; the
  split reports the batch-shape floor beside it.

## Blockers / needs the owner
(none)
