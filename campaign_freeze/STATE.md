# STATE — campaign freeze   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-11 06:30 tick (wake 1)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- The `freeze_compressor` flag's unit tests pass: only the decoder trains, no gradient reaches the frozen side, the frozen side stays in eval mode, flag off = old behaviour (job 188517, commit d167725; verdict closed).
- First smoke attempt died on a recipe bug (`max_steps` needs the `+` form), fixed in the recipe (job 188526, closed).

**Open, and running**
- Feasibility gate, frozen 20-step smoke (job 188528, running): the freeze is live in the log (`froze 506 compressor + connector tensors`, `Compressor trainable parameters: 0`); config diff vs reference 184778 is clean; provisional: train loss .594 after 640 rows (.023 epoch), ~75 s/step, 49.7 GB GPU.
- Its module-move check (job 188529) waits on it: every compressor / connector row must read changed 0.0%.
- New control this wake: the same 20-step smoke with nothing frozen (job 188534, running) + its module-move check (job 188535). Reason: a 20-step run warms up its learning rate in 1 step, the full run in 22, so the smoke's loss can't be compared with the full run's (the law is amended).

## Owned jobs (jobid | arm | state | elapsed | rows seen | eta | note)
| jobid | arm | state | elapsed | rows seen | eta | note |
|---|---|---|---|---|---|---|
| 188528 | F frozen, seed 42, 20-step smoke | RUNNING homer-006 (A100-80GB) | 23 min | 832 of 1,280 (step 13 of 20) | ~06:52 | out `expQ/freeze/ft_frozen_s42_smoke` |
| 188529 | module-move check of 188528 | PENDING afterok 188528 | - | - | ~07:00 | log `check_move_freeze_188529.out` |
| 188534 | unfrozen control, seed 42, 20-step smoke | RUNNING homer-010 (A100) | 1 min | 0 of 1,280 | ~07:15 | `ft_freeze.sh unfrozen 42 20`, out `expQ/freeze/ft_unfrozen_s42_smoke`, `--time 0:50:00` = 20 × 90.7 s × 1.25 + ~5 min load/save |
| 188535 | module-move check of 188534 | PENDING afterok 188534 | - | - | ~07:20 | positive control: compressor / connector must read changed > 0% |

## Adopted jobs
(none — every other job in the queue belongs to the sibling campaigns `longchunk` and `probes`)

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| R reference (184778 s42 / 184780 s43) | exists, not retrained | reference eval files must come from the 11-11 eval code: re-run them as `--stage eval-ref` cells if missing or older than the last change of `scripts/eval_mm_docqa.py` |
| F frozen, s42 + s43 | feasibility gate in flight (smoke 188528 + control 188534) | if gate PASS: launch `ft_freeze.sh frozen 42` and `… frozen 43`, `--time` = measured frozen s/step × 438 × 1.25 (provisional ~75 s → ~11.4 h), each with its module-move check and the 4 eval cells chained afterok |
| R3 third seed | only on a tie | — |

## Feasibility gate — evidence so far
| criterion | reading | status |
|---|---|---|
| log `freeze_compressor=True: froze N` | `froze 506 compressor + connector tensors (eval mode kept)` | PASS |
| log `Compressor trainable parameters: 0` | `0`; decoder trainable 155,978,240 | PASS |
| config diff vs `expQ/ft_mm_4b_184778/training_config.yaml` | only `max_steps: 20` and `freeze_compressor: true` | PASS (clean) |
| compressor / connector changed 0.0%, decoder lora_B > 0% | job 188529 pending | open |
| loss within .05 of the unfrozen smoke at steps 10 and 20 | frozen step 10 = .594; unfrozen 188534 not yet logged | open |
| check script detects movement (positive control) | job 188535 pending | open |

Config diff (frozen smoke vs 184778):
```
23a24
>   max_steps: 20
36a38
>   freeze_compressor: true
```

## Health evidence this wake (numbers, not adjectives)
- Frozen smoke: GPU 49.7 / 80 GB, util 98% (`expmon occupancy`), ~75 s/step steady (first step 248 s incl. warm-up), grad_norm .079 at step 10 (reference full run: .72 — it includes the compressor's gradients).
- Reference cost for comparison: 184778 11.0 h, 184780 13.8 h for 438 steps on A100-80GB = 90.7 / 113.6 s/step incl. evals.
- GPU cap on `gpu`: 6 of 8 in use after this wake's launch (longchunk 4, frozen smoke 1, control 1); 2 left.

## Blockers / needs the owner
(none)
