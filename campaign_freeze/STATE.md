# STATE — campaign freeze   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-11 19:30 tick (wake 7)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- **Gate 0 at two seeds = TIE under the law → third-seed pair running.** Freezing the compressor and connector (only the decoder LoRA trains) costs 1-3.5 pt of judge accuracy on every set, at both seeds (8 of 8 pairs negative). Paired per question, n = 1,500 each, same eval code, job 188650 (re-derived wake 7 from `paired_judge_freeze_188650.out` and `expmon grid freeze`):
  - documents: seed 42 −.028 [95% CI −.049, −.007], seed 43 −.035 [−.057, −.014] → CI wholly below 0 at both seeds;
  - photos: −.009 [−.026, +.008], −.013 [−.030, +.004] → not shown to differ;
  - MRQA text: −.023 [−.040, −.008], −.027 [−.043, −.009] → CI wholly below 0 at both seeds;
  - HotpotQA (reported, not gating): −.028 [−.045, −.013], −.023 [−.041, −.006].
  - Why a tie and not a fail: "Validated" needs ≥ −2 pt everywhere (documents and MRQA miss at both seeds); "Failed" needs ≤ −3 pt with the CI below 0 at 2 seeds on one set — documents seed 42 is −2.8 pt. The law sends this to a third seed and decides on 2 of 3.
- **The freeze held over the full runs** (1.0 epoch, 28,032 rows): every compressor and connector row, norm gain included, changed 0.0%; decoder lora_B 66.7% / 66.3% of entries changed (jobs 188584 / 188585). Caveat: the norm gain and the `<MEM>` token row also read 0.0% in the unfrozen 20-step control, so for those two rows 0.0% does not prove the freeze.
- **Feasibility gate PASS** (20-step smokes, 1,280 rows = .046 epoch): frozen side 0.0% changed (188578); the check sees movement in the unfrozen control (compressor LoRA 25.4% / 76.4%, connector 25.4%, 188579); train loss within .05 of the unfrozen smoke at steps 10 and 20 (.594 vs .639, .584 vs .580, jobs 188528 / 188534); config diff vs 184778 = `max_steps` + `freeze_compressor` only.
- Unit tests pass on the committed code (HEAD be0d842; `pisco/` and `test_*.py` clean): `test_positions.py` 12/12, `test_mm_collators.py` 33/33, `test_freeze.py` 5/5, exit 0 (job 188655).
- provisional: the loss is spread over the data sources, not one type: 28 of 28 per-source differences are negative (job 188650), but each source has only n = 250-500, only three have a CI below 0 (mpdocvqa seed 43 −.050, DROP seed 43 −.060, hotpot_bridge), and several sit inside the 2-pt noise band.
- provisional: frozen training is cheaper: 71.5 / 76.1 s/step (train_runtime 31,310 / 33,320 s, 188582 / 188583) vs the reference 90.3 / 113.2 s/step (39,550 / 49,600 s, 184778 / 184780), peak GPU memory 49.9 GB (frozen; no reference figure). Crosses nodes and dates, and the reference's own two seeds differ by 25%. The seed-44 pair runs at the same time on two A100 nodes (homer-010 / homer-009): 71 vs 89 s/step at .07-.10 epoch — the clean comparison when both end.
- provisional: the frozen runs end at a HIGHER eval loss than the reference (64 eval rows): .435 vs .414 (seed 42), .413 vs .375 (seed 43); the frozen curve is lower early (.23 epoch: .455 vs .513) and flatter. Consistent with the judge gap; 64 rows, no paired test.
- provisional: in this bf16 recipe the connector norm gain does not move even when trainable (one 20-step unfrozen smoke, job 188579: changed 0.0%, value .46094). The module-move check of the full seed-44 reference run (next wake) tests it over a whole epoch.
- provisional: previews at .23 epoch on the first 300 documents (closed, never gate): seed 42 .583 vs .610 (−.027, p .28), seed 43 .593 vs .610 (−.017, p .49) (jobs 188594 / 188595). Seed 42's preview gap equals its final gap (−.028); seed 43's is half its final gap (−.035). Neither preview is significant.

**Open, and running**
- Arm R3, third seed (seed 44), both sides from the same stage-2 model: frozen finetune (job 188651, homer-010, .10 epoch at 19:30, 71 s/step, ends ~03:20 on 11-12) and reference finetune that trains everything (job 188652, homer-009, .07 epoch, 89 s/step, train loss .497, ends ~06:00-08:30).
- All 8 seed-44 eval cells are queued, each `afterok` on its finetune: documents 188653 / 188654, photos 188672 / 188673, MRQA 188674 / 188675, HotpotQA 188676 / 188677.

## Owned jobs (jobid | arm | state | rows seen | eta | note)
| jobid | arm | state | rows seen | eta | note |
|---|---|---|---|---|---|
| 188651 | R3 frozen, seed 44 | RUNNING homer-010 | 2,816 of 28,032 (.10 epoch) at 19:30 | ~03:20 | `ft_freeze.sh frozen 44`, `--time 11:40:00`; out `expQ/freeze/ft_frozen_s44`; config vs frozen s43: only `seed: 44` |
| 188652 | R3 reference, seed 44 | RUNNING homer-009 | 1,920 (.07 epoch) | ~06:00-08:30 | `ft_freeze.sh unfrozen 44`, `--time 17:20:00`; out `expQ/freeze/ft_unfrozen_s44`; config vs 184780: only `seed: 44` |
| 188653 / 188654 | R3 documents eval, frozen / reference | PENDING afterok 188651 / 188652 | - | after each run | `PX=1003520`, `--time 0:40:00` |
| 188672 / 188673 | R3 photos eval, frozen / reference | PENDING afterok | - | after each run | `PX=1003520`, `--time 0:24:00` = 19:00 (188603) × 1.25 |
| 188674 / 188675 | R3 MRQA text eval, frozen / reference | PENDING afterok | - | after each run | `PX=1003520 CML=4096`, `--time 0:22:00` = 17:21 (188612) × 1.25 |
| 188676 / 188677 | R3 HotpotQA eval, frozen / reference | PENDING afterok | - | after each run | `PX=1003520 CML=4096 TOPK=10`, `--time 0:29:00` = 22:44 (188617) × 1.25 |
All evals: `eval_mm_4b.sh <model> pisco "" <set>`, gpu-be, `--exclude=homer-[005-009]`; eval code md5 `eval_mm_docqa.py` dfe32aa0, `eval_mm_4b.sh` 50abc167, `judge_mm_docqa.py` 02bc8748 (re-checked 19:30, same as every seed-42/43 cell).

GPU budget: ~27 GPU-h used of 192; R3 adds ~25 (training) + ~3 (evals).

## Adopted jobs
(none — the other jobs in the queue belong to the sibling campaigns `longchunk` and `probes`)

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| F frozen vs R reference, seeds 42 + 43 | evaluated: tie | — (cells closed with their numbers) |
| R3 seed-44 pair | training; 8 evals queued | next wake: chain the last 3 jobs (below); then the 3-seed decision (2 of 3) |

Next wake (in 30 min: nothing to wait on; this wake used its 6-submission cap) submits 3 jobs, no more:
1. `expmon launch --exp freeze --stage check --parent 188651 --dependency afterok:188651 --partition gpu-be --time 0:05:00 expQ/freeze/check_module_move.sh /beegfs/scratch/user/hdejean/pisco/expQ/archq3/pt_mm_5M_spread/model /beegfs/scratch/user/hdejean/pisco/expQ/freeze/ft_frozen_s44/model` (frozen rows must read 0.0%);
2. the same for 188652 / `ft_unfrozen_s44` (positive control; also tests the norm-gain claim over a full epoch);
3. `expmon launch --exp freeze --stage judge --dependency afterok:188653:188654:188672:188673:188674:188675:188676:188677 --partition gpu-be --time 0:15:00 expQ/freeze/paired_judge_cpu.sh ft_frozen_s44:ft_unfrozen_s44` (files land in `expQ/freeze/ft_{frozen,unfrozen}_s44/eval/` with the same names as seeds 42/43).
Then NEXT = `wake-when job:<paired test id> max 16h`.

What seed 44 can decide (written before its result; the law's "2 of 3" reading in CAMPAIGN.md "Tie"):
- arm F cannot validate any more: documents and MRQA already miss ≥ −2 pt at 2 of 2 seeds;
- FAILED if seed 44 documents is ≤ −3 pt with its CI wholly below 0 (seed 43 already is → 2 of 3); MRQA cannot reach 2 of 3 (both seeds above −3);
- otherwise still TIED after three seeds → BLOCKED.md for the owner (a consistent ~2-3 pt loss that the gate's bands do not classify).

## Mid-run health — eval_loss ↓ (64 eval rows, same seed = same eval rows)
| epoch | .23 | .46 | .69 | .92 | 1.0 |
|---|---|---|---|---|---|
| R seed 42 (184778) | .513 | .455 | .393 | .394 | .414 |
| R seed 43 (184780) | .471 | .439 | .411 | .441 | .375 |
| F seed 42 (188582) | .455 | .444 | .438 | .434 | .435 |
| F seed 43 (188583) | .442 | .431 | .416 | .415 | .413 |
| F seed 44 (188651) | pending | | | | |
| R seed 44 (188652) | pending | | | | |
Kill rule for seed 44 (law): NaN, or 2 probes above the reference's first value; use the seed-44 reference run's own .23-epoch value once it exists.

## Deliverable — judge accuracy ↑ (Qwen3-8B rubric judge), n = 1,500 per set, paired diff to R of the same seed [95% CI] (job 188650)
| model | what trains | rate (tokens per slot; slots per page) | documents s42 / s43 | Δ docs | photos s42 / s43 | Δ photos | MRQA text s42 / s43 | Δ MRQA | HotpotQA s42 / s43 | Δ HotpotQA |
|---|---|---|---|---|---|---|---|---|---|---|
| R reference (184778 / 184780, re-evaluated 11-11 on the current code) | compressor LoRA + merger + connector + decoder LoRA | 8; 123 per 1M-px page (980 visual tokens) | .610 / .615 | — | .673 / .685 | — | .718 / .724 | — | .603 / .593 | — |
| F frozen (188582 / 188583) | decoder LoRA only | same | .582 / .580 | −.028 [−.049, −.007] / −.035 [−.057, −.014] | .664 / .672 | −.009 [−.026, +.008] / −.013 [−.030, +.004] | .695 / .697 | −.023 [−.040, −.008] / −.027 [−.043, −.009] | .575 / .571 | −.028 [−.045, −.013] / −.023 [−.041, −.006] |
| F frozen, seed 44 | decoder LoRA only | same | pending | | pending | | pending | | pending | |
| uncompressed decoder (earlier runs, same questions) | — | 1 (raw tokens) | .809 | | .717 | | .807 | | — | |
Discordant pairs (frozen-only right / reference-only right), McNemar p: documents 111/153 p .011, 109/161 p .002; photos 78/92 p .32, 78/97 p .17; MRQA 59/94 p .006, 67/107 p .003; HotpotQA 64/106 p .002, 68/102 p .011.

## Module movement (entries changed %, per module, vs the stage-2 model 184777)
| module | F smoke s42 (188578) | unfrozen smoke s42 (188579) | F full s42 (188584) | F full s43 (188585) |
|---|---|---|---|---|
| decoder lora_A / lora_B | 30.6 / 76.3 | 26.7 / 75.8 | 14.7 / 66.7 | 14.7 / 66.3 |
| compressor lora_A / lora_B | 0.0 / 0.0 | 25.4 / 76.4 | 0.0 / 0.0 | 0.0 / 0.0 |
| compressor `<MEM>` token row | 0.0 | 0.0 | 0.0 | 0.0 |
| connector MLP | 0.0 | 25.4 | 0.0 | 0.0 |
| connector norm gain | 0.0 | 0.0 (bf16 lost update) | 0.0 | 0.0 |
Odd, not explained: the full runs changed FEWER decoder entries than the 20-step smoke (14.7% vs 30.6% of lora_A) although they moved further in norm (lora_B .070 vs .025 of its norm). A guess, not checked: the smoke hit peak lr at step 1 (1 warm-up step), when Adam's first update is ±lr for every entry; the full run warms up over 22 steps and most small updates are lost in bf16.

## Training cost (A100-80GB) — provisional: different nodes and dates; the seed-44 pair (same time, homer-010 / homer-009) is the clean comparison
| run | s/step (train_runtime / 437 steps) | wall | peak GPU memory (sacct gpumem) | trainable params |
|---|---|---|---|---|
| F frozen s42 / s43 (188582 / 188583) | 71.5 / 76.1 | 8.7 / 9.3 h | 49.9 / 49.9 GB | 155,978,240 (decoder only) |
| R reference s42 / s43 (184778 / 184780) | 90.3 / 113.2 | 11.0 / 13.8 h | — | 307,771,392 |
| unfrozen smoke (188534) | 88-90 steady | — | 31.9 GB | 307,771,392 |
Open question: why the frozen run uses more GPU memory (49.9 vs 31.9 GB). A guess, not checked: gradient checkpointing only runs in train mode and the frozen compressor is in eval mode. It fits on 80 GB.

## Proposed text for the owner's chats (provisional until seed 44)
- docs/training_plan.md, gate 0 of "Standing proposal": "Decoder-only finetune on frozen `<MEM>`s (campaign freeze, 2 seeds, n = 1,500 per set): −2.8 / −3.5 pt on documents and −2.3 / −2.7 pt on MRQA text (paired CI below 0 at both seeds), photos −.9 / −1.3 (not significant), HotpotQA −2.8 / −2.3. Roughly 20% faster training (to confirm on the same-time seed-44 pair). The finetune does use the compressor; a third seed decides between 'tie' and 'failed'."
- HANDOFF.md thread 7: "Frozen compressor costs ~2-3 pt; every source is lower (per-source n 250-500, mostly inside noise); stage B would need either the split test (connector-only / compressor-only frozen, a new flag in `pisco/`) or accepting ~2-3 pt."

## Notes this wake
- Wake 7: re-derived the two-seed headline from the paired-test output (no change in any number); applied the verifier's five marks (per-source spread, speed, eval loss, norm gain, previews now `provisional:` with their n and caveats).
- Wake 7: submitted the 6 remaining seed-44 eval cells; `--time` re-derived from the slowest seed-42/43 cell of each set × 1.25 (shorter than wake 6's 0:27, which predated those measurements). Env per set read back from the seed-42 output file names (`_px1003520`, `_cml4096`, `_k10`): the manifest does not store env (gripe filed).
- Wake 7: one probe of 188651 was first recorded under a wrong name (`loss_s_per_step=71.0`); re-recorded as `s_per_step=71.0`. Ignore the first.

## Patches this wake
(none)

## Blockers / needs the owner
(none)
