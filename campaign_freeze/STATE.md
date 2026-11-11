# STATE — campaign freeze   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-11 10:30 tick (wake 4)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- **Feasibility gate PASS** (frozen 20-step smoke, job 188528, 1,280 rows = .046 epoch, against the unfrozen 20-step smoke, job 188534, same rows and lr schedule):
  - the freeze holds: every compressor and connector weight, the norm gain included, changed 0.0% (job 188578); the decoder LoRA moved (lora_B 76.3% of entries changed);
  - the check can see movement: in the unfrozen smoke the compressor LoRA (25.4% / 76.4%) and the connector MLP (25.4%) moved (jobs 188535, 188579);
  - train loss: step 10 .594 frozen vs .639 unfrozen (−.045), step 20 .584 vs .580 (+.004): both within the .05 the law allows;
  - config diff vs the reference finetune 184778: only `max_steps` and `freeze_compressor`.
- First smoke attempt died on a recipe bug (`max_steps` needs the `+` form), fixed (job 188526, closed).
- provisional: the `freeze_compressor` unit tests pass (test_freeze.py 5/5, job 188517, exit 0). They ran on an UNCOMMITTED copy (git 9dd687d+dirty); the flag and the test were committed afterwards as d167725. No run on a committed sha yet: a re-run on HEAD is planned (one cheap CPU job).
- provisional: in this bf16 recipe (lr 2e-5) the connector norm gain does not move even when trainable (one 20-step unfrozen smoke, job 188579: changed 0.0%, value .46094). One short run only.
- provisional: early previews (step-100 checkpoint = .23 epoch, 6,400 rows seen; first 300 documents; judge accuracy ↑) sit 2-3 pt below the FINISHED reference, inside noise: seed 42 .583 vs .610 (−.027, 17 frozen-only right / 25 reference-only right, exact McNemar p = .28, job 188594); seed 43 .593 vs .610 (−.017, 14 / 19, p = .49, job 188595). Confounded by training stage (.23 vs 1.0 epoch; the reference kept no checkpoint-100). A preview never gates.

**Open, and running**
- Arm F, frozen finetune, seed 42 (job 188582) and seed 43 (job 188583): healthy. At 10:33, seed 42 has seen ~9,200 of 28,032 rows (.33 epoch) and seed 43 ~8,800 (.31 epoch). Eval loss ↓ at .23 epoch is below the reference: .455 vs .513 (seed 42), .442 vs .471 (seed 43). Expected end ~16:35 (seed 42), ~16:55 (seed 43).
- Chained to start when each finetune ends (10 eval cells + 2 module-move checks): documents, photos (both seeds) and MRQA text (seed 42), each as a frozen / reference pair on the same eval code and in the same window.

## Owned jobs (jobid | arm | state | rows seen | eta | note)
| jobid | arm | state | rows seen | eta | note |
|---|---|---|---|---|---|
| 188582 | F frozen, seed 42, full finetune | RUNNING homer-010 | ~9,200 of 28,032 (.33 epoch) at 10:33 | ~16:35 | 72 s/step steady; out `expQ/freeze/ft_frozen_s42`, `--time 13:00:00` |
| 188583 | F frozen, seed 43, full finetune | RUNNING homer-006 | ~8,800 (.31 epoch) | ~16:55 | 75 s/step steady; out `expQ/freeze/ft_frozen_s43` |
| 188584 / 188585 | module-move check after each finetune | PENDING afterok 188582 / 188583 | - | after the run | frozen rows must still read 0.0% |
| 188596 / 188598 | F documents, s42 / s43, 1,500 Q | PENDING afterok 188582 / 188583 | - | ~17:15 / ~17:35 | `--time 0:40:00` = 31:47 × 1.25 |
| 188597 / 188599 | R documents re-eval, s42 / s43 | PENDING afterok 188582 / 188583 | - | same | model via symlink `expQ/freeze/ref_s42` / `ref_s43` |
| 188601 / 188603 | F photos `mm_naturalqa_1500`, s42 / s43 | PENDING afterok 188582 / 188583 | - | same | `--time 0:27:00` = 20:53 × 1.25 (slowest measured photos/MRQA cell) |
| 188602 / 188604 | R photos re-eval, s42 / s43 | PENDING afterok 188582 / 188583 | - | same | via `ref_s42` / `ref_s43` |
| 188605 / 188606 | F / R MRQA text `text_mrqa_ood_1500`, s42, CML=4096 | PENDING afterok 188582 | - | same | `--time 0:27:00` |

All eval cells: `PX=1003520 eval_mm_4b.sh <model> pisco "" <set>`, partition `gpu-be`, `--exclude=homer-[005-009]`, judge chained. Eval code md5 (checked again this wake, unchanged): `scripts/eval_mm_docqa.py` dfe32aa0 (mtime 11-11 06:45), `eval_mm_4b.sh` 50abc167, `scripts/judge_mm_docqa.py` 02bc8748. Every later cell must find the same hashes, or both sides re-run.

## Adopted jobs
(none — every other job in the queue belongs to the sibling campaigns `longchunk` and `probes`)

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| F frozen, s42 + s43 | full runs in flight, healthy | probe eval_loss at .46 epoch (checkpoint-200, ~12:00); after the run: module-move check + evals (all chained) |
| R reference re-evals (184778 / 184780) | documents + photos at both seeds, MRQA s42 chained | MRQA s43 pair and HotpotQA × 4 still to submit |
| R3 third seed | only on a tie | — |

Next wake (checkpoint-200 of seed 43 appears, ~12:00): probe eval_loss of both runs at .46 epoch (reference .455 / .439). Then submit, chained `afterok` on the finetune of the same seed, 6 cells:
- MRQA s43 pair: `PX=1003520 CML=4096 … eval_data/text_mrqa_ood_1500` on `ft_frozen_s43/model` and `ref_s43/model`, `--time 0:27:00`;
- HotpotQA × 4: `PX=1003520 CML=4096 TOPK=10 … eval_data/text_hotpotqa_1500`, `--time 0:29:00` (22:38 × 1.25, job 188509), frozen + reference × 2 seeds.
The wake after: the unit-test re-run on HEAD (one CPU job, `run_tests_cpu.sh`).

## Mid-run health — eval_loss ↓ (64 eval rows, same seed = same eval rows)
| epoch | .23 | .46 | .69 | .92 | 1.0 |
|---|---|---|---|---|---|
| R seed 42 (184778) | .513 | .455 | .393 | .394 | .414 |
| R seed 43 (184780) | .471 | .439 | .411 | .441 | .375 |
| F seed 42 (188582) | .455 | | | | |
| F seed 43 (188583) | .442 | | | | |
Kill rule (law): NaN, or 2 probes above the reference's first value (.513 seed 42, .471 seed 43). Train loss at .32 epoch: .472 (s42); at .30 epoch .462 (s43); grad_norm .05-.09.

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
| F full s42 / s43 (188582 / 188583), first 144 / 137 steps | 72 / 75 | pending | same |
| unfrozen smoke (188534) | 88-90 (100.7 overall) | 31.9 GB | 307,771,392 |
| R full runs 184778 / 184780 | 90.7 / 113.6 overall incl. evals | — | 307,771,392 |
Open question: the frozen smoke uses more GPU memory than the unfrozen one (49.7 vs 31.9 GB). A guess, not checked: gradient checkpointing only runs in train mode, and the frozen side is in eval mode. It fits on 80 GB; it does not affect the gate.

## Notes this wake
- Preview pairing done with `jq` on the login node (join by `query_id`, no python); exact McNemar p by `awk`. `scripts/paired_judge.py` (bootstrap CI) is used for the 1,500-Q cells, through SLURM.
- `expmon display --exp freeze` writes an empty page (0 rows from 0 runs) although the grid lists 20 runs: gripe filed. The tables above are the deliverable until it is fixed.
- gpucap not needed: no training launched (evals and checks go to `gpu-be`).

## Patches this wake
(none)

## Blockers / needs the owner
(none)
