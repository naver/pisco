# STATE — campaign freeze   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-11 10:00 tick (wake 3)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- **Feasibility gate PASS** (frozen 20-step smoke, job 188528, 1,280 rows = .046 epoch, against the unfrozen 20-step smoke, job 188534, same rows and lr schedule):
  - the freeze holds: every compressor and connector weight, the norm gain included, changed 0.0% (job 188578); the decoder LoRA moved (lora_B 76.3% of entries changed);
  - the check can see movement: in the unfrozen smoke the compressor LoRA (25.4% / 76.4%) and the connector MLP (25.4%) moved (jobs 188535, 188579);
  - train loss: step 10 .594 frozen vs .639 unfrozen (−.045), step 20 .584 vs .580 (+.004): both within the .05 the law allows;
  - config diff vs the reference finetune 184778: only `max_steps` and `freeze_compressor`.
- First smoke attempt died on a recipe bug (`max_steps` needs the `+` form), fixed (job 188526, closed).
- provisional: the `freeze_compressor` unit tests pass (test_freeze.py 5/5, job 188517, exit 0). They ran on an UNCOMMITTED copy (git 9dd687d+dirty); the flag and the test were committed afterwards as d167725, and `pisco/model.py` / `test_freeze.py` are unchanged since. No run on a committed sha yet: a re-run on HEAD is planned (one cheap CPU job).
- provisional: in this bf16 recipe (lr 2e-5) the connector norm gain does not move even when trainable (one 20-step unfrozen smoke, job 188579: changed 0.0%, value .46094). The bf16 step near .461 is 2^-9 = .002 and one update is ~1e-5, so the update is lost. One short run only.

**Open, and running**
- Arm F, frozen finetune, seed 42 (job 188582) and seed 43 (job 188583): healthy at .23 epoch (6,400 of 28,032 rows). Eval loss ↓ is BELOW the reference at the same point: .455 vs .513 (seed 42), .442 vs .471 (seed 43). Expected end ~16:20 (seed 42) and ~16:50 (seed 43).
- Previews of the step-100 checkpoints on the first 300 documents, judge chained: jobs 188594 (seed 42) and 188595 (seed 43), running since 10:05.
- Documents evals, 1,500 Q, chained to start when each finetune ends: frozen s42 (188596) + reference s42 re-eval (188597); frozen s43 (188598) + reference s43 re-eval (188599).

## Owned jobs (jobid | arm | state | rows seen | eta | note)
| jobid | arm | state | rows seen | eta | note |
|---|---|---|---|---|---|
| 188582 | F frozen, seed 42, full finetune | RUNNING homer-010 | ~7,600 of 28,032 (step ~119 at 10:00) | ~16:20 | 71 s/step steady; out `expQ/freeze/ft_frozen_s42`, `--time 13:00:00` |
| 188583 | F frozen, seed 43, full finetune | RUNNING homer-006 | ~7,200 (step ~112) | ~16:50 | 75 s/step steady; out `expQ/freeze/ft_frozen_s43` |
| 188584 / 188585 | module-move check after each finetune | PENDING afterok 188582 / 188583 | - | after the run | frozen rows must still read 0.0% |
| 188594 / 188595 | preview, checkpoint-100, 300 documents | RUNNING homer-011 / homer-012 | - | ~10:30 | PX=1003520, `--time 0:40:00` |
| 188596 / 188598 | F documents eval s42 / s43, 1,500 Q | PENDING afterok 188582 / 188583 | - | ~17:00 / ~17:30 | PX=1003520, `--time 0:40:00` = 31:47 × 1.25 |
| 188597 / 188599 | R documents re-eval s42 / s43, 1,500 Q | PENDING afterok 188582 / 188583 | - | same | model via symlink `expQ/freeze/ref_s42` / `ref_s43` (the 11-09/11-10 files stay untouched) |

All eval cells: `eval_mm_4b.sh <model> pisco "" <set>`, partition `gpu-be`, `--exclude=homer-[005-009]`. Eval code md5 at submit (the working copy, uncommitted edits by the probes chats): `scripts/eval_mm_docqa.py` dfe32aa0 (mtime 11-11 06:45), `eval_mm_4b.sh` 50abc167, `scripts/judge_mm_docqa.py` 02bc8748. Every later cell must find the same hashes, or both sides re-run.

## Adopted jobs
(none — every other job in the queue belongs to the sibling campaigns `longchunk` and `probes`)

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| F frozen, s42 + s43 | full runs in flight, healthy | probe eval_loss at .46 / .69 / .92 epoch; after the run: module-move check (chained), then 4 eval cells per seed |
| R reference re-evals (184778 / 184780) | documents chained | photos, MRQA text, HotpotQA cells still to submit, chained the same way |
| R3 third seed | only on a tie | — |

Next wake (when both previews end): record the preview judge (read against the reference final on the same 300 Q, s42 .610 / s43 .610 — no reference checkpoint-100 exists, 184778/184780 kept only 400 and 437). Then submit, chained `afterok` on the finetune of the same seed, within 6 per wake:
- photos `eval_data/mm_naturalqa_1500`, PX=1003520, `--time 0:27:00` (20:53 × 1.25, job 188539): frozen + reference × 2 seeds = 4 cells;
- MRQA `eval_data/text_mrqa_ood_1500`, PX=1003520 CML=4096, `--time 0:22:00` (17 min × 1.25, job 188445): seed 42 pair = 2 cells.
The wake after: MRQA seed 43 pair, HotpotQA `eval_data/text_hotpotqa_1500` PX=1003520 CML=4096 TOPK=10 `--time 0:29:00` (22:38 × 1.25, job 188509) × 4, and the unit-test re-run on HEAD.

## Mid-run health — eval_loss ↓ (64 eval rows, same seed = same eval rows)
| epoch | .23 | .46 | .69 | .92 | 1.0 |
|---|---|---|---|---|---|
| R seed 42 (184778) | .513 | .455 | .393 | .394 | .414 |
| R seed 43 (184780) | .471 | .439 | .411 | .441 | .375 |
| F seed 42 (188582) | .455 | | | | |
| F seed 43 (188583) | .442 | | | | |
Kill rule (law): NaN, or 2 probes above the reference's first value (.513 seed 42, .471 seed 43). Train loss at .25 epoch: .459 (s42), .470 (s43); grad_norm .065 both.

## Deliverable (filled as evals land) — judge accuracy ↑ (Qwen3-8B rubric judge), n = 1,500 per set
| model | what trains | rate (tokens per slot; slots per page) | documents s42 / s43 | Δ vs R [95% CI] | photos s42 / s43 | Δ vs R | MRQA text s42 / s43 | Δ vs R | HotpotQA s42 / s43 | Δ vs R |
|---|---|---|---|---|---|---|---|---|---|---|
| R reference (old files, 11-09/11-10; MRQA 11-11 05:23) | compressor LoRA + merger + connector + decoder LoRA | 8; 123 per 1M-px page (980 visual tokens) | .610 / .615 | — | .677 / .685 | — | .718 / — | — | — / — | — |
| F frozen | decoder LoRA only | same | pending | | pending | | pending | | pending | |
| uncompressed decoder | — | 1 (raw tokens) | .809 | | .717 | | .807 | | — | |
The R row will be replaced by the re-evals on the current code. The old MRQA reference file is `expQ/probes/c_plain_s42/eval/mm_docqa_pisco_px1003520_cml4096_text_mrqa_ood_1500.json` (written before the 06:45 eval-code change, so it is re-run too).

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
| F full s42 / s43 (188582 / 188583), first 120 steps | 71 / 75 (~2:01 / 2:06 for 100 steps incl. one eval) | pending | same |
| unfrozen smoke (188534) | 88-90 (100.7 overall) | 31.9 GB | 307,771,392 |
| R full runs 184778 / 184780 | 90.7 / 113.6 overall incl. evals | — | 307,771,392 |
Open question: the frozen smoke uses more GPU memory than the unfrozen one (49.7 vs 31.9 GB). A guess, not checked: gradient checkpointing only runs in train mode, and the frozen side is in eval mode. It fits on 80 GB; it does not affect the gate.

## Notes this wake
- One rule-8 slip: I printed 6 lines of `.expmon/runs.jsonl` with a one-line stdlib `python3` on the login node (under a second). Later reads use `jq`. No dataset, no pixi.
- gpucap not needed: no training launched (evals and checks go to `gpu-be`).
- Gripes filed: `--dependency` and launch-line env vars (PX, CML, TOPK) are not recorded in the manifest.

## Patches this wake
(none)

## Blockers / needs the owner
(none)
