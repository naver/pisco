# STATE — campaign rebuild   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-25 20:30 (first wake after arming). HEAD 6440bd1, `git status pisco/ scripts/rebuild/` clean.

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Pre-arm proof holds: CPU job 193564 COMPLETED 0:0 (10:09), `RESULT preflight_cpu_ok=1`: 40 / 40 unit tests
  (options off = the old collator's RNG stream draw for draw vs commit 69869340) and the resolved configs of stage 2
  K / S / W / P and finetunes K s42 / K s43 / L s42 / L s43 differ from the recorded ones only in the law's keys.
  The earlier unit-test job 193554 (rc 1, 39 / 40, a test-side miscount) is superseded and closed.

**Open, and running**
- Feasibility gate (can page word recall be read on today's code, `pisco/model.py` 2a1cc007 vs 994cff3d when recorded):
  three cells on `slotcheck_dense100` (100 pages), queued on gpu-be: R2 (193565, bar .702 ± .01), overfit o_bf16
  (193566, bar .921 ± .01), a fresh uncompressed reference (193567, paired with itself = 1.000, and with the recorded ref).
- Nothing trains yet: the law starts training only after this gate passes.

## Owned jobs (jobid | arm | state | elapsed | epoch or step/total | eta | note)
- 193554 | unit tests | FAILED rc 1, verdict closed | 3:10 | — | — | superseded by 193564
- 193564 | preflight (CPU) | COMPLETED 0:0, verdict closed | 10:09 | — | — | tests + 8 configs OK
- 193565 | feasibility: R2 recall, dense100 | PENDING (Priority) | 0 | 100 pages | ~45 min after start | `--time 2:20 --mem 10G`
- 193566 | feasibility: o_bf16 recall, dense100 | PENDING (Priority) | 0 | 100 pages | ~45 min after start | `--time 2:20 --mem 10G`
- 193567 | feasibility: ref recall, dense100 | PENDING (Priority) | 0 | 100 pages | ~40 min after start | `--time 1:55 --mem 10G`

All three: `scripts/rebuild/recall.sbatch` (7a5109fc, written this wake: slotcheck's b61a7b4c with outputs under
`expQ/rebuild/recall/`, any model dir as ARM, optional `:PER_SOURCE`, every law md5 printed), `-p gpu-be
--exclude=homer-[005-009]`. `--time` = measured pass wall × 3 (40-41 min per 100-page pass + 5 min plumbing, jobs
191893 / 192209); `--mem` = 192209 MaxRSS 6.5 GB × 1.5. md5 at launch: eval_mm_4b.sh b3d1cda1, eval_mm_docqa.py
dfe32aa0, judge_mm_docqa.py 02bc8748, paired_judge.py a3cc13b2, model.py 2a1cc007, mm_collator.py 12d80001,
mem_recall.py 9f241620, probe_mem_recall.py 14730fed. Logs: `expQ/rebuild/recall_<jid>.out`.

## Adopted jobs (found running in this project, not launched by the campaign — never killed, never duplicated)
(none; 193533-193536 are the live sibling longdoc's, never touched)

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| feasibility gate | 3 cells queued | read: `python3 -I scripts/longdoc/recall_pair.py --ref expQ/rebuild/recall/ref_dense100.json --a expQ/rebuild/recall/r2_dense100.json --b expQ/rebuild/recall/o_bf16_dense100.json` → A .702 ± .01, B .921 ± .01; self-pair `--a ref --b ref` → 1.000; fresh vs recorded ref (`--b expQ/slotcheck/recall/ref_dense100.json`). Also check each log: `mem_position_stride=8`, plumbing words > 0. FAIL → BLOCKED.md "gate 1 cannot be read on today's code", NEXT `waiting-owner` |
| K / S / W / P (stage 2) | launchers committed | after the gate: four 20-step smokes, `expmon launch --exp rebuild --stage <ARM>_smoke --constraint=gpu_a100 --time 1:30:00 --mem 65G scripts/rebuild/pt_stage2.sbatch <ARM> 20` (`--mem` = R2 run 184777 MaxRSS 42.8 GB × 1.5; `gpucap land a100 1` first; 6 A100 nodes free at 20:30). Then full runs, `--time` = smoke s/step × 2,499 × 1.25 |
| L (cleaned labels) | launcher committed | after the gate (the wake after the smokes, 4-per-wake cap): dump + judge the 28k teacher labels, build `data/rebuild_ft_clean_{docpages,natural}`, the R re-run check, then L s42 / s43 |
| gate-1 references | not started | after the gate: one `--arm ref` pass on `slotcheck_dense100_heldout` and on `mm_docqa_1500_1m` gold pages (`:120`), shared by every model |
| gate-2 finetunes | `ft.sbatch <ARM> 42` | run `ft.sbatch P 42 cfg` before any P finetune (its config resolution was added after 193564) |

## Health evidence this wake (numbers, not adjectives)
- Recorded gate-1 numbers re-derived on the frontend (stdlib `recall_pair.py`): R2 .702, o_bf16 .921, o_bf16 − R2
  +.219 [+.185, +.253], 98 / 2 pages, n 100.
- Budget: 0 of 190 GPU-h spent on GPUs (only CPU jobs ran); 5 of 60 cells. The feasibility gate will cost ≈ 2.3 GPU-h
  (3 cells × ~45 min), above the law's 1.5 estimate (it assumed 14 s/page; measured walls are ~24 s/page with loads).

## Blockers / needs the owner
(none)
