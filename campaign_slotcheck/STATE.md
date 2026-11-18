# STATE — campaign slotcheck   (rewritten WHOLESALE by every LLM wake; never append)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Check 1 (arm C, feasibility gate PASS): the finetune collator's 4-document cap never removes ALL answer pages of a
  trained row — 0 / 20,000 document rows, 0 / 8,000 photo rows → not a cause of the document gap. Beside it: 1,910 /
  20,000 document rows (9.55%; all Docmatix, 19.1% of its 10,000) lose one of their two source pages; the record does
  not say which page holds the answer. Job 191663 (CPU audit, 56 s, verdict closed). Table: `docs/slot_checks.md`.
- Check 2a (arm P, padding probe) is NOT inert, n = 120 pages (job 191672, verdict closed; controls pass: same input
  twice cos 1.0000 / max abs 0 on 120 / 120, another page median cos .739; `<MEM>` count identical at every width).
  One page alone + 17 / 63 left pads: median cos .934 / .934, worst slot .632, max abs 2.65 / 2.80 (floor 0). A
  question's 4 pages padded to 64 as in training (30 questions): median cos .9966, worst slot .637. The law's bar
  (median ≥ .9999) is missed on both paths, so check 2b runs. The 6-page plumbing run 191671 (closed) agreed.

**Open, and running**
- Why 2a moves the slots (job 191781, 40 pages, PENDING): splits the 1-row path (pads not zeroed in Qwen3.5's
  linear-attention layers: installed transformers zeroes them only for batches of > 1 row, `modeling_qwen3_5.py:198`
  — a code reading, untested) from the ≥ 2-row path (pads zeroed), forces the zeroing on for 1 row as a cause test, and
  repeats in fp32 (rounding or real). Reading fixed before the run is in its ledger intent note.
- Check 2b (does training-style padding change judge accuracy?): wrapper check 191782 (300 Q, no judge: plain
  `eval_mm_4b.sh` vs wrapper pad=None must give identical predictions, then pad=64) → full pairs 191783 (seed 42) and
  191784 (seed 43), 1,500 Q each, pad None then pad 64 on one node, judged, paired; both `afterok:191782`.
- Checks 3 (text slots) and 4 (overfit capacity): not started.

## Owned jobs (jobid | arm | state | elapsed | epoch or step/total | eta | note)
| job | arm | state | note |
|---|---|---|---|
| 191663 | C (check 1) | COMPLETED 0:0, 56 s, homer-017 | verdict closed |
| 191671 | P (2a) plumbing | COMPLETED 0:0, ≈ 4 min, homer-011 | verdict closed (plumbing) |
| 191672 | P (2a) full | COMPLETED 0:0, 4:04, homer-011, MaxRSS 3.6 GB | verdict closed (not inert); probe recorded |
| 191781 | P (2a split) | PENDING (Priority), `--time 0:20`, `--mem 8G` | 40 pages × (repeat, twin0, A/Afix/B × widths 17/63) × bf16 + fp32 |
| 191782 | P (2b) plumbing | PENDING (Priority), `--time 0:50`, `--mem 8G`, `JUDGE=0` | exits 4 if plain ≠ wrapper-None |
| 191783 | P (2b) full s42 | PENDING (Dependency afterok 191782), `--time 2:45`, `--mem 8G` | cell `expQ/slotcheck/p2b_s42` |
| 191784 | P (2b) full s43 | PENDING (Dependency afterok 191782), `--time 2:45`, `--mem 8G` | cell `expQ/slotcheck/p2b_s43` |

## Adopted jobs (found running in this project, not launched by the campaign — never killed, never duplicated)
(none — 191780 eval_mm_4b and the other eval jobs belong to the siblings)

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| C. finetune-set count (check 1) | **decided: not a cause** (0 / 20,000), 1 of 2 cells used | none |
| P. padding (check 2a / 2b) | 2a decided not inert; split 191781 + 2b chain 191782 → 191783 / 191784 queued (6 of 6 + 1 split + 1 plumbing cells) | 191782: INVARIANT PASS line → let the pairs run; FAIL (exit 4) → `expmon kill 191783 191784`, diagnose the wrapper. 191781: write its reading into `docs/slot_checks.md`; if Afix ≈ B and A ≪ B → the 1-row leak is a hand-off candidate (training-only, photo rows; `pisco/**` is never patched here). Pairs: `paired_judge` 64 − None per seed vs the 2b gate |
| O. overfit capacity (check 4) | builder written, not launched (`scripts/slotcheck/build_dense100.{py,sbatch}`, CPU) | launch the builder smoke (`build_dense100.sbatch 3000 _smoke`, `-p gpu-be --gres=gpu:0 -c 4 --mem 16G --time 0:30`) then the full build `afterok`; write `pt_overfit.sbatch` (the start model's launch row is NOT in `.expmon/runs.jsonl` — rebuild the overrides from `expQ/archq3/pt_mm_5M_spread/training_config.yaml` + `train.log` head; task mix `p_image_ae=1, p_text=0, p_image_cont=0, p_desc=0, p_find=0`); 20-step smoke; config diff. **Budget conflict to settle before the training:** the recipe's 8 × 8 = 64 samples per step × `max_steps` 1,000 = 64,000 samples ≈ 12 h at the stage-2 rate (5,300 samples / h), far over hard rule 9's 2 GPU-h per cell; at 64 samples / step, 2 GPU-h ≈ 165 steps ≈ 106 passes over the 100 pages. The smoke's s/step decides; the stop rule becomes plateau OR 1,000 steps OR 2 GPU-h, whichever first (a tightening) |
| X. text slots (check 3) | not started | last (biggest cost); first shard measures the 12 GPU-h checkpoint |

## Health evidence this wake (numbers, not adjectives)
- 191672: sacct COMPLETED 0:0, 4:04, homer-011, MaxRSS 3,603,040 KB; 120 pages; `<MEM>` count identical at 0 / 17 / 63
  on every page (97 at 867 tokens, 73 at 651); repeat max abs 0 on all 120; output `expQ/slotcheck/pad_probe_191672/`.
- New code (freely patchable, `scripts/slotcheck/`): `pad_split.py` e2143c71 + `.sbatch`; `eval_pad.py` cb7187b2 (imports
  `scripts/eval_mm_docqa.py`, sets `pad_to_multiple_of` on the collator it returns, asserts the shared value is still
  None); `eval_pad.sbatch` 67241721; `compare_preds.py` 0c227be5. `bash -n` and `py_compile` pass. Checked in
  `scripts/eval_mm_docqa.py` that the pisco path builds `coll = make_collator(...)` (line 676) and pads through
  `collate_compressor([batch], coll, ...)` (line 285), which reads `pad_to_multiple_of` (`pisco/mm_utils.py:229`).
- Code md5 at launch: eval_mm_4b.sh b3d1cda1, scripts/eval_mm_docqa.py dfe32aa0, scripts/judge_mm_docqa.py 02bc8748,
  scripts/paired_judge.py a3cc13b2, pisco/model.py 994cff3d (unchanged since the campaign started).
- `--time` sources: 2b full = 191495 (R s42, same set, PX 1,003,520, today's launcher) 27:13 × 2 evals × 3 = 2:45;
  split = 191672's 4:04 for 540 compress calls, 720 here, × 3 ≈ 0:20.
- Arm O builder (not run yet): pages from the TAIL of `data/mm_pretrain_pdfa_1m` (60,000 rows scanned), never a docid
  among the first 80,000 rows of `data/mm_pretrain_pdfa_1m_desc` (what the start model trained on); dense = most
  compressor tokens in [300, 760], so the AE target (`text[:768]`) is the whole page text; one page per docid; seeded
  split 100 overfit / 100 held-out; the eval copies keep the original JPEG bytes (no re-encode). Writes
  `data/slotcheck_dense100_train`, `eval_data/slotcheck_dense100{,_heldout}` (format of `scripts/longdoc/mem_recall.py`:
  `doc_images`, `gold_positions` [0], `source` pdfa_dense, `query`). `py_compile` and `bash -n` pass.
- Budget: 7 of 32 cells, ≈ 0.3 of 30 GPU-h used (+≈ 6.5 GPU-h booked by the 4 queued jobs at their limits).

## Law notes this wake
- The 2a inert test uses "max abs ≤ 2 × the repeat floor"; the repeat floor is exactly 0 (a deterministic forward),
  so that half of the test asks for bit-identical vectors (verifier, 17:44). Left as written (changing the floor would
  relax the rule); the split probe reports a batch-size rounding floor (twin0) beside it.

## Blockers / needs the owner
(none)
