# STATE — campaign freeze   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-11 12:00 tick (wake 5)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- **Feasibility gate PASS** (frozen 20-step smoke, job 188528, 1,280 rows = .046 epoch, against the unfrozen 20-step smoke, job 188534, same rows and lr schedule):
  - the freeze holds: every compressor and connector weight, the norm gain included, changed 0.0% (job 188578); the decoder LoRA moved (lora_B 76.3% of entries changed);
  - the check can see movement: in the unfrozen smoke the compressor LoRA (25.4% / 76.4%) and the connector MLP (25.4%) moved (jobs 188535, 188579);
  - train loss: step 10 .594 frozen vs .639 unfrozen (−.045), step 20 .584 vs .580 (+.004): both within the .05 the law allows;
  - config diff vs the reference finetune 184778: only `max_steps` and `freeze_compressor`.
- First smoke attempt died on a recipe bug (`max_steps` needs the `+` form), fixed (job 188526, closed).
- provisional: the `freeze_compressor` unit tests pass (test_freeze.py 5/5, job 188517, exit 0). They ran on an UNCOMMITTED copy (git 9dd687d+dirty); the flag and the test were committed afterwards as d167725. No run on a committed sha yet: a re-run on HEAD is planned (one cheap CPU job, next wake: this wake used its 6 submissions).
- provisional: in this bf16 recipe (lr 2e-5) the connector norm gain does not move even when trainable (one 20-step unfrozen smoke, job 188579: changed 0.0%, value .46094). One short run only.
- provisional: early previews (step-100 checkpoint = .23 epoch, 6,400 rows seen; first 300 documents; judge accuracy ↑) sit 2-3 pt below the FINISHED reference, inside noise: seed 42 .583 vs .610 (−.027, 17 frozen-only right / 25 reference-only right, exact McNemar p = .28, job 188594); seed 43 .593 vs .610 (−.017, 14 / 19, p = .49, job 188595). Two confounds: training stage (.23 vs 1.0 epoch; the reference kept no checkpoint-100), and eval code (the reference side is the OLD judged file `expQ/ft_mm_4b_18477x/eval/…mm_docqa_1500_1m.json`, written before the 11-11 change of `scripts/eval_mm_docqa.py`). A preview never gates.

**Open, and running**
- Arm F, frozen finetune, seed 42 (job 188582) and seed 43 (job 188583): healthy. Both at .46 epoch (12,800 of 28,032 rows) by 11:51. Eval loss ↓ stays below the reference at the same epoch: .444 vs .455 (seed 42), .431 vs .439 (seed 43). Expected end ~16:20 (seed 42), ~16:50 (seed 43).
- Chained to start when each finetune ends: 2 module-move checks and 16 eval cells = 8 frozen / reference pairs, on the same eval code and in the same window: documents, photos, MRQA text, HotpotQA, each at both seeds. Every gating cell is now queued.

## Owned jobs (jobid | arm | state | rows seen | eta | note)
| jobid | arm | state | rows seen | eta | note |
|---|---|---|---|---|---|
| 188582 | F frozen, seed 42, full finetune | RUNNING homer-010 | 12,800 of 28,032 (.46 epoch) at 11:37 | ~16:20 | 71.4 s/step (checkpoint-100 → 200 mtimes); out `expQ/freeze/ft_frozen_s42`, `--time 13:00:00` |
| 188583 | F frozen, seed 43, full finetune | RUNNING homer-006 | 12,800 (.46 epoch) at 11:51 | ~16:50 | 75.6 s/step; out `expQ/freeze/ft_frozen_s43` |
| 188584 / 188585 | module-move check after each finetune | PENDING afterok 188582 / 188583 | - | after the run | frozen rows must still read 0.0% |
| 188596 / 188598 | F documents, s42 / s43, 1,500 Q | PENDING afterok 188582 / 188583 | - | ~17:00 / ~17:30 | `--time 0:40:00` = 31:47 × 1.25 |
| 188597 / 188599 | R documents re-eval, s42 / s43 | PENDING afterok 188582 / 188583 | - | same | model via symlink `expQ/freeze/ref_s42` / `ref_s43` |
| 188601 / 188603 | F photos `mm_naturalqa_1500`, s42 / s43 | PENDING afterok 188582 / 188583 | - | same | `--time 0:27:00` = 20:53 × 1.25 |
| 188602 / 188604 | R photos re-eval, s42 / s43 | PENDING | - | same | via `ref_s42` / `ref_s43` |
| 188605 / 188606 | F / R MRQA text `text_mrqa_ood_1500`, s42, CML=4096 | PENDING afterok 188582 | - | same | `--time 0:27:00` |
| 188612 / 188613 | F / R MRQA text, s43, CML=4096 | PENDING afterok 188583 | - | same | submitted this wake, `--time 0:27:00` |
| 188614 / 188615 | F / R HotpotQA `text_hotpotqa_1500`, s42, CML=4096 TOPK=10 | PENDING afterok 188582 | - | same | submitted this wake; `--time 0:29:00` = 22:38 (job 188509, same set and settings) × 1.25 |
| 188616 / 188617 | F / R HotpotQA, s43 | PENDING afterok 188583 | - | same | submitted this wake |

All eval cells: `PX=1003520 eval_mm_4b.sh <model> pisco "" <set>`, partition `gpu-be`, `--exclude=homer-[005-009]`, judge chained. Eval code md5 (checked again this wake, unchanged): `scripts/eval_mm_docqa.py` dfe32aa0 (mtime 11-11 06:45; differs from git HEAD but is the copy every cell uses), `eval_mm_4b.sh` 50abc167, `scripts/judge_mm_docqa.py` 02bc8748. The env (PX / CML / TOPK) is not stored in the run manifest: confirm it from each log's `FACTORS` line and the output file name (`…_px1003520_cml4096_…`) when the cells end.

## Adopted jobs
(none — every other job in the queue belongs to the sibling campaigns `longchunk` and `probes`)

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| F frozen, s42 + s43 | full runs in flight, healthy at .46 epoch | after the run: read the eval_loss curve at .69 / .92 / 1.0 epoch from trainer_state, module-move check (chained) |
| F vs R, all 4 sets × 2 seeds | 16 cells chained | when all have ended: `scripts/paired_judge.py` per pair (through SLURM, CPU on `gpu-be`), apply the gate |
| R3 third seed | only on a tie | — |

Next wake (all 18 chained jobs have left the queue, ~17:30-18:00): confirm exit codes + the FACTORS line of each cell; submit the paired-judge job (8 pairs) and the unit-test re-run on HEAD (`run_tests_cpu.sh`); fill the deliverable and module-movement tables; refresh `expmon display --exp freeze`.

## Mid-run health — eval_loss ↓ (64 eval rows, same seed = same eval rows)
| epoch | .23 | .46 | .69 | .92 | 1.0 |
|---|---|---|---|---|---|
| R seed 42 (184778) | .513 | .455 | .393 | .394 | .414 |
| R seed 43 (184780) | .471 | .439 | .411 | .441 | .375 |
| F seed 42 (188582) | .455 | .444 | | | |
| F seed 43 (188583) | .442 | .431 | | | |
Kill rule (law): NaN, or 2 probes above the reference's first value (.513 seed 42, .471 seed 43). Train loss at .46 epoch: .447 (s42), .432 (s43); grad_norm .04-.05.

## Deliverable (filled as evals land) — judge accuracy ↑ (Qwen3-8B rubric judge), n = 1,500 per set unless noted
| model | what trains | rate (tokens per slot; slots per page) | documents s42 / s43 | Δ vs R [95% CI] | photos s42 / s43 | Δ vs R | MRQA text s42 / s43 | Δ vs R | HotpotQA s42 / s43 | Δ vs R |
|---|---|---|---|---|---|---|---|---|---|---|
| R reference (old files, 11-09/11-10; MRQA 11-11 05:23) | compressor LoRA + merger + connector + decoder LoRA | 8; 123 per 1M-px page (980 visual tokens) | .610 / .615 | — | .677 / .685 | — | .718 / — | — | — / — | — |
| provisional: F frozen, .23-epoch preview, first 300 docs | decoder LoRA only | same | .583 / .593 (R final on the same 300: .610 / .610) | −.027 (p .28) / −.017 (p .49), n = 300 | — | | — | | — | |
| F frozen, final | decoder LoRA only | same | pending | | pending | | pending | | pending | |
| uncompressed decoder | — | 1 (raw tokens) | .809 | | .717 | | .807 | | — | |
The R row will be replaced by the re-evals on the current code. Preview split by source (n = 100 infovqa / 200 slidevqa): s42 .51 / .62 vs R .53 / .65; s43 .46 / .66 vs R .50 / .665.

## Module movement (entries changed %, per module, vs the stage-2 model 184777)
| module | F smoke s42 (188578) | unfrozen smoke s42 (188579) | F full s42 | F full s43 |
|---|---|---|---|---|
| decoder lora_A / lora_B | 30.6 / 76.3 | 26.7 / 75.8 | pending | pending |
| compressor lora_A / lora_B | 0.0 / 0.0 | 25.4 / 76.4 | | |
| compressor `<MEM>` token row | 0.0 | 0.0 | | |
| connector MLP | 0.0 | 25.4 | | |
| connector norm gain | 0.0 | 0.0 (bf16 lost update) | | |

## Training cost (A100-80GB)
| run | s/step steady | peak GPU memory (sacct gpumem) | trainable params |
|---|---|---|---|
| F frozen smoke (188528) | 72-76 (85.1 overall) | 49.7 GB | 155,978,240 (decoder only) |
| F full s42 / s43 (188582 / 188583), steps 100-200 | 71.4 / 75.6 | pending | same |
| unfrozen smoke (188534) | 88-90 (100.7 overall) | 31.9 GB | 307,771,392 |
| R full runs 184778 / 184780 | 90.7 / 113.6 overall incl. evals | — | 307,771,392 |
Open question: the frozen smoke uses more GPU memory than the unfrozen one (49.7 vs 31.9 GB). A guess, not checked: gradient checkpointing only runs in train mode, and the frozen side is in eval mode. It fits on 80 GB; it does not affect the gate.

## Notes this wake
- No intermediate wake for the .69 / .92 epoch probes: eval_loss sits .06-.07 under the kill threshold at both seeds, so the next wake reads the whole curve from trainer_state at the end.
- `expmon display --exp freeze` wrote an empty page last wake (gripe filed); not re-run: no arm decided.
- gpucap not needed: no training launched (evals go to `gpu-be`).

## Patches this wake
(none)

## Blockers / needs the owner
(none)
