# STATE — campaign rebuild   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-27 01:30 (woken: ALERT "expmon daemon not running"). The daemon job had reached its 15-day wall
(TIMEOUT 01:00) and the tick had already resubmitted it (RUNNING 01:30, homer-020): ALERT deleted, known_issues row
added, gripe 7e8d2f8c +1. Nothing launched, no code edited. Previous science wake: 2026-11-25 23:00 (HEAD ca348f9).

Efficiency line ("99 % waiting for the owner"): does not apply as a fix here. The law makes this wait: a feasibility-gate
FAIL stops the campaign before training until the owner answers. The recommended answer (A) relaxes the law's R2 bar, so the
standing "≤ 5 GPU-h, weakens nothing" rule cannot accept it; option B (≈ 2 GPU-h) would fit that rule but only measures the
noise and leaves the same question open, so the recommendation is kept unchanged.

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Feasibility gate, known positive PASSES: the overfit model `expQ/slotcheck/o_bf16/model` gives page word recall .921
  on `slotcheck_dense100` (100 pages) = recorded .921 (job 192209), 100 / 100 rebuilt texts identical — job 193566.
- Feasibility gate, reference PASSES: the fresh uncompressed transcription paired with itself 1.000, with the recorded
  reference 1.000 (9 of 100 pages differ by < 0.1 pt) — job 193567.
- Feasibility gate, R2 MISSES its bar: R2 `expQ/archq3/pt_mm_5M_spread/model` .715 vs bar .702 ± .01 (job 191893),
  today − recorded +1.3 pt [−2.4, +5.2], 52 / 45 pages, sign p .54, n 100 — job 193565. 99 / 100 texts differ; per
  page sd .20, 28 pages move > 10 pt. Not a code change: the `pisco/model.py` diff since 994cff3d is the opt-in pad
  guard (off); the 5-page pass of the same job on homer-019 wrote the 5 recorded texts exactly, the 100-page pass on
  homer-017 did not. By the law (FAIL → stop before training) the campaign waits for the owner (BLOCKED.md).
- Pre-arm proof, stage-2 half: job 193564 COMPLETED 0:0, `RESULT preflight_cpu_ok=1`, 40 / 40 unit tests, stage-2
  K / S / W / P configs differ from the recorded one only in the law's keys. It ran on uncommitted files (git
  622ae40+dirty), but every file it used for stage 2 (`pisco/model.py`, `pisco/mm_collator.py`, `test_mm_collators.py`,
  `scripts/rebuild/config_check.py`, `pt_stage2.sbatch`) was last written before the job started (20:07:17) and has
  the md5 of commit 52cc9e9.
- provisional: pre-arm proof, finetune half. `scripts/rebuild/ft.sbatch` was edited at 20:17:50, after the job ended
  (20:17:26), so the K / L finetune config checks did not test the committed file (8f9eb8d6). Re-run
  `ft.sbatch <ARM> 42 cfg` (and P) before any finetune.

**Open, and running**
- Nothing runs. Owner question in BLOCKED.md: count the feasibility gate as passed (recommended, 0 GPU-h now) or stop.

## Owned jobs (jobid | arm | state | elapsed | epoch or step/total | eta | note)
- 193554 | unit tests | FAILED rc 1, verdict closed | 3:10 | — | — | superseded by 193564
- 193564 | preflight (CPU) | COMPLETED 0:0, verdict closed | 10:09 | — | — | tests + 8 configs OK
- 193565 | feasibility: R2 recall, dense100 | COMPLETED 0:0 (3 attempts, 2 preempted), verdict closed | 49:35 | 100 / 100 pages | — | `expQ/rebuild/recall/r2_dense100.json`
- 193566 | feasibility: o_bf16 recall, dense100 | COMPLETED 0:0 (3 attempts, 2 preempted), verdict closed | 55:54 | 100 / 100 pages | — | `expQ/rebuild/recall/o_bf16_dense100.json`
- 193567 | feasibility: reference, dense100 | COMPLETED 0:0, verdict closed | 37:08 | 100 / 100 pages | — | `expQ/rebuild/recall/ref_dense100.json`

`expmon check`'s `✗ COMPLETED but crash signature: preempted` on 193566 (and 193565) is benign: attempt 2 wrote the
complete file and was preempted a minute later; attempt 3 found it (`exists, skipped`) and exited 0; the appended log
still holds attempt 1-2's preemption line (known_issues row added, gripe filed).

## Adopted jobs (found running in this project, not launched by the campaign — never killed, never duplicated)
(none; 193534-193536 are the live sibling longdoc's, never touched)

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| feasibility gate | decided: 2 pass, R2 misses by .003 | owner answer (BLOCKED.md). If "A": add to the law's gate 1 "every model of one page set is read in ONE recall job (one node), with a fresh R2 pass beside K; report R2's read-to-read difference beside the seed floor" (`recall.sbatch` already takes several `NAME:MODEL:SET` passes) |
| K / S / W / P (stage 2) | waiting on the owner | after "A": four 20-step smokes, `expmon launch --exp rebuild --stage <ARM>_smoke --constraint=gpu_a100 --time 1:30:00 --mem 65G scripts/rebuild/pt_stage2.sbatch <ARM> 20` (`--mem` = R2 run 184777 MaxRSS 42.8 GB × 1.5; `gpucap land a100 1` first). Then full runs, `--time` = smoke s/step × 2,499 × 1.25 |
| L (cleaned labels) | waiting on the owner | after the smokes (4-per-wake cap): `ft.sbatch` config re-check, dump + judge the 28k teacher labels, build `data/rebuild_ft_clean_{docpages,natural}`, the R re-run check, then L s42 / s43 |
| gate-1 references | not started | one `--arm ref` pass on `slotcheck_dense100_heldout` and on `mm_docqa_1500_1m` gold pages (`:120`), shared by every model |
| gate-2 finetunes | `ft.sbatch <ARM> 42` | config re-check first (see the provisional line) |

## Health evidence this wake (numbers, not adjectives)
- Each recall file: 100 records, stride 8 on every page, 116-123 slots per page (mean 119.9, identical for R2 and o_bf16),
  5-page plumbing words R2 [479, 412, 461, 474, 260], o_bf16 [480, 449, 484, 488, 366]; md5 at run: eval_mm_4b.sh
  b3d1cda1, eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py 02bc8748, paired_judge.py a3cc13b2, model.py 2a1cc007,
  mm_collator.py 12d80001, mem_recall.py 9f241620, probe_mem_recall.py 14730fed, recall.sbatch 7a5109fc (= the committed
  file in 503368a). All nodes A100-SXM4-80GB (homer-019, -017, -014), outside homer-[005-009].
- Pairing: `python3 -I scripts/longdoc/recall_pair.py --ref <ref> --a <A> --b <B>` (stdlib) on the fresh and recorded
  files; o_bf16 − R2 on today's files +20.6 pt [+17.1, +24.3], 97 / 3 pages (recorded +21.9 pt).
- Budget: ≈ 2.4 of 190 GPU-h (193565 0.83 + 193566 0.93 + 193567 0.62, sacct elapsed, 1 GPU each); 5 of 60 cells.
  Recall measured 23 s per page (law assumed 14 s): gate-1 recall projects ≈ 21 GPU-h vs 12 budgeted (≈ 200 of 190 in
  total) — told to the owner as a heads-up; the wake asks before going over.

## Blockers / needs the owner
- BLOCKED.md: the feasibility gate's R2 bar missed by .003 (run-to-run readout noise, not code); recommendation A =
  count it passed and read gate-1 models in one job with a fresh R2 (0 GPU-h now, +1.3 at gate 1). It relaxes a bar,
  so it needs the owner's word.
