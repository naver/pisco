# STATE — campaign freeze   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-11 07:30 tick (wake 2)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- The `freeze_compressor` flag's unit tests pass: only the decoder trains, no gradient reaches the frozen side, the frozen side stays in eval mode, flag off = old behaviour (test_freeze.py 5/5, job 188517, ran on git 9dd687d+dirty; verdict closed).
- **Feasibility gate PASS** (frozen 20-step smoke, job 188528, 1,280 rows = .046 epoch, against the unfrozen 20-step smoke, job 188534, same rows and lr schedule):
  - the freeze holds: every compressor and connector weight, the norm gain included, changed 0.0% (job 188578); the decoder LoRA moved (lora_B 76.3% of entries changed);
  - the check can see movement: in the unfrozen smoke the compressor LoRA (25.4% / 76.4%) and the connector MLP (25.4%) moved (jobs 188535, 188579);
  - train loss: step 10 .594 frozen vs .639 unfrozen (−.045), step 20 .584 vs .580 (+.004): both within the .05 the law allows;
  - config diff vs the reference finetune 184778: only `max_steps` and `freeze_compressor`.
- Side finding: in this bf16 recipe (lr 2e-5) the connector norm gain does not move even when it is trainable (unfrozen smoke: changed 0.0%, job 188579). The bf16 step near .461 is .002, one update is ~1e-5, so the update is lost. Freezing it changes nothing here.
- First smoke attempt died on a recipe bug (`max_steps` needs the `+` form), fixed (job 188526, closed).

**Open, and running**
- Arm F, frozen finetune, seed 42 (job 188582) and seed 43 (job 188583): full runs, 28k rows = 1 epoch each, started 07:37 on A100-80GB; each has a module-move check chained (jobs 188584, 188585).
- Next: evals of both frozen models on documents, photos, MRQA text, HotpotQA, and re-evals of the two reference models on the same code (their judged files date from 11-09/11-10; the eval script changed 11-11 06:45).

## Owned jobs (jobid | arm | state | elapsed | rows seen | eta | note)
| jobid | arm | state | elapsed | rows seen | eta | note |
|---|---|---|---|---|---|---|
| 188582 | F frozen, seed 42, full | RUNNING homer-010 | 2 min | 0 of 28,032 | ~17:00-18:30 (438 steps × 75-85 s) | out `expQ/freeze/ft_frozen_s42`, `--time 13:00:00` = 85.1 s/step × 438 × 1.25 |
| 188583 | F frozen, seed 43, full | RUNNING homer-006 | 1 min | 0 of 28,032 | same | out `expQ/freeze/ft_frozen_s43`, same `--time` |
| 188584 | module-move check of 188582 | PENDING afterok | - | - | after 188582 | frozen rows must still read 0.0% |
| 188585 | module-move check of 188583 | PENDING afterok | - | - | after 188583 | same |

## Adopted jobs
(none — every other job in the queue belongs to the sibling campaigns `longchunk` and `probes`)

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| R reference (184778 s42 / 184780 s43) | exists, not retrained | re-run its 8 eval cells (`--stage eval-ref`, 4 sets × 2 seeds) so both sides of every pair run the same eval code; submit them chained `afterok` on the frozen finetune of the same seed, so they run when the frozen evals run |
| F frozen, s42 + s43 | full runs in flight | health at each 100 steps (eval_loss vs the reference); after the run: module-move check, then 4 eval cells per seed (judge chained), then `paired_judge.py` per set |
| R3 third seed | only on a tie | — |

Next wake (at the seed-42 checkpoint of step 100, ~2.2 h in): probe eval_loss at .23 epoch for both seeds; submit the one pre-authorised preview (seed 42, checkpoint-100, first 300 documents) and the reference re-eval cells chained on the finetunes, within the 6 submissions per wake.

## Mid-run health — the reference curves to compare with (eval_loss ↓, same seed = same eval rows)
| epoch | .23 | .46 | .69 | .92 | 1.0 |
|---|---|---|---|---|---|
| R seed 42 (184778) | .513 | .455 | .393 | .394 | .414 |
| R seed 43 (184780) | .471 | .439 | .411 | .441 | .375 |
| F seed 42 (188582) | | | | | |
| F seed 43 (188583) | | | | | |
Kill rule (law): NaN, or 2 probes above the reference's first value (.513 seed 42, .471 seed 43).

## Deliverable (filled as evals land) — judge accuracy ↑ (Qwen3-8B rubric judge), n = 1,500 per set
| model | what trains | rate (tokens per slot; slots per page) | documents s42 / s43 | Δ vs R [95% CI] | photos s42 / s43 | Δ vs R | MRQA text s42 | Δ vs R | HotpotQA s42 | Δ vs R |
|---|---|---|---|---|---|---|---|---|---|---|
| R reference (11-09/11-10 files, old eval code) | compressor LoRA + merger + connector + decoder LoRA | 8; 123 per 1M-px page (980 visual tokens) | .610 / .615 | — | .677 / .685 | — | .718 | — | — | — |
| F frozen | decoder LoRA only | same | pending | | pending | | pending | | pending | |
| uncompressed decoder | — | 1 (raw tokens) | .809 | | .717 | | .807 | | — | |
The R row will be replaced by the re-evals on the current code.

## Module movement (entries changed %, per module, vs the stage-2 model 184777)
| module | F smoke s42 (188578) | unfrozen smoke s42 (188579) | F full s42 | F full s43 |
|---|---|---|---|---|
| decoder lora_A / lora_B | 30.6 / 76.3 | 26.7 / 75.8 | pending | pending |
| compressor lora_A / lora_B | 0.0 / 0.0 | 25.4 / 76.4 | | |
| compressor `<MEM>` token row | 0.0 | 0.0 | | |
| connector MLP | 0.0 | 25.4 | | |
| connector norm gain | 0.0 | 0.0 (bf16 lost update) | | |

## Training cost (20-step smokes, A100-80GB; full runs pending)
| run | s/step overall (steady) | peak GPU memory (sacct gpumem) | trainable params |
|---|---|---|---|
| F frozen smoke (188528) | 85.1 (72-76) | 49.7 GB | 155,978,240 (decoder only) |
| unfrozen smoke (188534) | 100.7 (88-90) | 31.9 GB | 307,771,392 |
| R full runs 184778 / 184780 | 90.7 / 113.6 incl. evals | — | 307,771,392 |
Open question: the frozen smoke uses more GPU memory than the unfrozen one (49.7 vs 31.9 GB). A guess, not checked: gradient checkpointing only runs in train mode, and the frozen side is in eval mode. It fits on 80 GB; it does not affect the gate.

## Health evidence this wake (numbers, not adjectives)
- Eval loss at step 20 (64 eval rows): frozen .527 vs unfrozen .488. The law does not gate on it; reported only.
- grad_norm frozen .079 / .059 vs unfrozen 1.20 / 1.56 (steps 10 / 20): almost all the gradient norm of the reference recipe sits on the compressor side.
- GPU cap on `gpu`: 6 of 8 in use (longchunk 4, arm F 2); 2 left. `gpucap land a100 1` said yes before the launch.

## Patches this wake
- `expQ/freeze/check_module_move.sh` — no traceback: the norm gain row was missing because the gain is absent from the stage-2 init — 6 lines added (compare a retrofitted gain with `connector_out_norm_init` from the trained config.json). Re-run on both smokes: jobs 188578 / 188579.

## Blockers / needs the owner
(none)
