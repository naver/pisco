# STATE — campaign freeze   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-11 20:30 tick (wake 8)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- **Gate 0 at two seeds = TIE under the law → third-seed pair running.** Freezing the compressor and connector (only the decoder LoRA trains) costs 2.3-3.5 pt of judge accuracy on documents, MRQA text and HotpotQA at both seeds; photos are lower by .9 / 1.3 pt, inside the 2-pt noise band with a CI spanning 0. Paired per question, n = 1,500 each, same eval code, job 188650 (`paired_judge_freeze_188650.out`, `expmon grid freeze`):
  - documents: seed 42 −.028 [95% CI −.049, −.007], seed 43 −.035 [−.057, −.014] → CI wholly below 0 at both seeds;
  - photos: −.009 [−.026, +.008], −.013 [−.030, +.004] → not shown to differ;
  - MRQA text: −.023 [−.040, −.008], −.027 [−.043, −.009] → CI wholly below 0 at both seeds;
  - HotpotQA (reported, not gating): −.028 [−.045, −.013], −.023 [−.041, −.006].
  - Why a tie and not a fail: "Validated" needs ≥ −2 pt everywhere (documents and MRQA miss at both seeds); "Failed" needs ≤ −3 pt with the CI below 0 at 2 seeds on one set — documents seed 42 is −2.8 pt. The law sends this to a third seed and decides on 2 of 3.
- **The freeze held over the full runs** (1.0 epoch, 28,032 rows): every compressor and connector row, norm gain included, changed 0.0%; decoder lora_B 66.7% / 66.3% of entries changed (jobs 188584 / 188585). Caveat: the norm gain and the `<MEM>` token row also read 0.0% in the unfrozen 20-step control (188579), so for those two rows 0.0% does not prove the freeze.
- **Feasibility gate PASS** (20-step smokes, 1,280 rows = .046 epoch): frozen side 0.0% changed (188578); the check sees movement in the unfrozen control (compressor LoRA 25.4% / 76.4%, connector 25.4%, 188579); train loss within .05 of the unfrozen smoke at steps 10 and 20 (.594 vs .639, .584 vs .580, jobs 188528 / 188534); config diff vs 184778 = `max_steps` + `freeze_compressor` only.
- Unit tests pass on the committed code (HEAD be0d842; `pisco/` and `test_*.py` clean): `test_positions.py` 12/12, `test_mm_collators.py` 33/33, `test_freeze.py` 5/5, exit 0 (job 188655).
- provisional: the loss is spread over the data sources, not one type: 28 of 28 per-source differences are negative (job 188650); most sources have n = 250-500 (hotpot_bridge n = 1,197); only mpdocvqa seed 43 (−.050), DROP seed 43 (−.060) and hotpot_bridge (both seeds) have a CI below 0; several sit inside the 2-pt noise band.
- provisional: frozen training is cheaper. Same-time seed-44 pair (two A100 nodes, homer-010 / homer-009), at .16-.22 epoch: frozen 71.2 s/step vs reference 90.8 s/step (−22%). Earlier seeds, different nodes and dates: 71.5 / 76.1 s/step (train_runtime 31,310 / 33,320 s over 437 steps, 188582 / 188583) vs 90.3 / 113.2 (184778 / 184780). Peak GPU memory 49.9 GB frozen (no reference figure; unfrozen smoke 31.9 GB). The final numbers come when the seed-44 pair ends.
- provisional: the frozen runs end at a HIGHER eval loss than the reference (64 eval rows): .435 vs .414 (seed 42), .413 vs .375 (seed 43); the frozen curve is lower early (.23 epoch: .455 vs .513) and flatter. Consistent with the judge gap; 64 rows, no paired test.
- provisional: in this bf16 recipe the connector norm gain does not move even when trainable (one 20-step unfrozen smoke, job 188579: changed 0.0%, value .46094). The module-move check of the full seed-44 reference run (job 188700) tests it over a whole epoch.
- provisional: previews at .23 epoch on the first 300 documents (closed, never gate): seed 42 .583 vs .610 (−.027, p .28), seed 43 .593 vs .610 (−.017, p .49) (jobs 188594 / 188595). Seed 42's preview gap equals its final gap (−.028); seed 43's is half its final gap (−.035). Neither preview is significant.

**Open, and running**
- Arm R3, third seed (seed 44), both sides from the same stage-2 model, healthy at 20:32 (no NaN, train loss .45-.52):
  - frozen finetune (job 188651, homer-010): 94 of 437 steps = 6,016 of 28,032 rows (.22 epoch), 71 s/step, ends ~03:20 on 11-12;
  - reference finetune that trains everything (job 188652, homer-009): 68 of 437 = 4,352 rows (.16 epoch), 91 s/step, ends ~05:55 on 11-12.
- The whole rest of the chain is queued, nothing left to submit for seed 44:
  - 8 evals, each `afterok` on its finetune: documents 188653 / 188654, photos 188672 / 188673, MRQA 188674 / 188675, HotpotQA 188676 / 188677;
  - module-move checks: frozen 188699 (afterok 188651), reference 188700 (afterok 188652);
  - paired test 188701 (afterok all 8 evals): `paired_judge_cpu.sh ft_frozen_s44:ft_unfrozen_s44`. Expected ~06:30-07:00 on 11-12.

## Owned jobs (jobid | arm | state | rows seen | eta | note)
| jobid | arm | state | rows seen | eta | note |
|---|---|---|---|---|---|
| 188651 | R3 frozen, seed 44 | RUNNING homer-010 | 6,016 of 28,032 (.22 epoch) at 20:32 | ~03:20 | `ft_freeze.sh frozen 44`, `--time 11:40:00`; out `expQ/freeze/ft_frozen_s44`; config vs frozen s43: only `seed: 44` |
| 188652 | R3 reference, seed 44 | RUNNING homer-009 | 4,352 (.16 epoch) at 20:32 | ~05:55 | `ft_freeze.sh unfrozen 44`, `--time 17:20:00`; out `expQ/freeze/ft_unfrozen_s44`; config vs 184780: only `seed: 44` |
| 188653 / 188654 | R3 documents eval, frozen / reference | PENDING afterok 188651 / 188652 | - | after each run | `PX=1003520`, `--time 0:40:00` |
| 188672 / 188673 | R3 photos eval, frozen / reference | PENDING afterok | - | after each run | `PX=1003520`, `--time 0:24:00` = 19:00 (188603) × 1.25 |
| 188674 / 188675 | R3 MRQA text eval, frozen / reference | PENDING afterok | - | after each run | `PX=1003520 CML=4096`, `--time 0:22:00` = 17:21 (188612) × 1.25 |
| 188676 / 188677 | R3 HotpotQA eval, frozen / reference | PENDING afterok | - | after each run | `PX=1003520 CML=4096 TOPK=10`, `--time 0:29:00` = 22:44 (188617) × 1.25 |
| 188699 | R3 module-move check, frozen s44 | PENDING afterok 188651 | - | ~03:25 | `check_module_move.sh <184777 model> expQ/freeze/ft_frozen_s44/model`, gpu-be CPU, `--time 0:05:00` (measured 47 s, 188584) |
| 188700 | R3 module-move check, reference s44 | PENDING afterok 188652 | - | ~06:00 | same, `ft_unfrozen_s44/model` |
| 188701 | R3 paired test | PENDING afterok 8 evals | - | ~06:30-07:00 | `paired_judge_cpu.sh ft_frozen_s44:ft_unfrozen_s44`, `--time 0:05:00` (measured 19 s, 188650) |
All evals: `eval_mm_4b.sh <model> pisco "" <set>`, gpu-be, `--exclude=homer-[005-009]`; eval code md5 `eval_mm_docqa.py` dfe32aa0, `eval_mm_4b.sh` 50abc167, `judge_mm_docqa.py` 02bc8748 (same as every seed-42/43 cell). Watch: `eval_mm_4b.sh` exits 0 when only its judge step fails; the paired test then exits 1 on the missing scores.

GPU budget: ~28.5 GPU-h used of 192; R3 adds ~25 (training) + ~3 (evals).

## Adopted jobs
(none — the other jobs in the queue belong to the sibling campaigns `longchunk` and `probes`)

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| F frozen vs R reference, seeds 42 + 43 | evaluated: tie | — (cells closed with their numbers) |
| R3 seed-44 pair | training; full chain queued | when 188701 ends: verdicts on the 8 evals + 2 checks + the paired test; the 3-seed decision (2 of 3) |

What seed 44 can decide (written before its result; the law's "2 of 3" reading in CAMPAIGN.md "Tie"):
- arm F cannot validate any more: documents and MRQA already miss ≥ −2 pt at 2 of 2 seeds;
- FAILED if seed 44 documents is ≤ −3 pt with its CI wholly below 0 (seed 43 already is → 2 of 3); MRQA cannot reach 2 of 3 (both seeds above −3);
- otherwise still TIED after three seeds → BLOCKED.md for the owner (a consistent ~2-3 pt loss that the gate's bands do not classify).
- If FAILED: the law forbids variants of freezing; BLOCKED.md proposes the split test (freeze the connector only / the compressor only; a new flag in `pisco/`, written by the owner's chats).

## Mid-run health — eval_loss ↓ (64 eval rows, same seed = same eval rows)
| epoch | .23 | .46 | .69 | .92 | 1.0 |
|---|---|---|---|---|---|
| R seed 42 (184778) | .513 | .455 | .393 | .394 | .414 |
| R seed 43 (184780) | .471 | .439 | .411 | .441 | .375 |
| F seed 42 (188582) | .455 | .444 | .438 | .434 | .435 |
| F seed 43 (188583) | .442 | .431 | .416 | .415 | .413 |
| F seed 44 (188651) | pending (step 100 ≈ 20:40) | | | | |
| R seed 44 (188652) | pending (step 100 ≈ 21:20) | | | | |
Kill rule for seed 44 (law): NaN, or 2 probes above the reference's first value; use the seed-44 reference run's own .23-epoch value once it exists.

## Deliverable — judge accuracy ↑ (Qwen3-8B rubric judge), n = 1,500 per set, paired diff to R of the same seed [95% CI] (job 188650)
| model | what trains | rate (tokens per slot; slots per page) | documents s42 / s43 | Δ docs | photos s42 / s43 | Δ photos | MRQA text s42 / s43 | Δ MRQA | HotpotQA s42 / s43 | Δ HotpotQA |
|---|---|---|---|---|---|---|---|---|---|---|
| R reference (184778 / 184780, re-evaluated 11-11 on the current code) | compressor LoRA + merger + connector + decoder LoRA | 8; 123 per 1M-px page (980 visual tokens) | .610 / .615 | — | .673 / .685 | — | .718 / .724 | — | .603 / .593 | — |
| F frozen (188582 / 188583) | decoder LoRA only | same | .582 / .580 | −.028 [−.049, −.007] / −.035 [−.057, −.014] | .664 / .672 | −.009 [−.026, +.008] / −.013 [−.030, +.004] | .695 / .697 | −.023 [−.040, −.008] / −.027 [−.043, −.009] | .575 / .571 | −.028 [−.045, −.013] / −.023 [−.041, −.006] |
| R / F seed 44 (188652 / 188651) | as above | same | pending | | pending | | pending | | pending | |
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
Seed-44 columns (frozen 188699, reference 188700) come with the runs. Odd, not explained: the full runs changed FEWER decoder entries than the 20-step smoke (14.7% vs 30.6% of lora_A) although they moved further in norm (lora_B .070 vs .025 of its norm). A guess, not checked: the smoke hit peak lr at step 1 (1 warm-up step), when Adam's first update is ±lr for every entry; the full run warms up over 22 steps and most small updates are lost in bf16.

## Training cost (A100-80GB) — provisional until the seed-44 pair ends
| run | s/step | wall | peak GPU memory (sacct gpumem) | trainable params |
|---|---|---|---|---|
| F frozen s42 / s43 (188582 / 188583) | 71.5 / 76.1 (train_runtime / 437) | 8.7 / 9.3 h | 49.9 / 49.9 GB | 155,978,240 (decoder only) |
| R reference s42 / s43 (184778 / 184780) | 90.3 / 113.2 | 11.0 / 13.8 h | — | 307,771,392 |
| F / R seed 44, same time (188651 / 188652), at .22 / .16 epoch | 71.2 / 90.8 (tqdm) | running | — | as above |
| unfrozen smoke (188534) | 88-90 steady | — | 31.9 GB | 307,771,392 |
Open question: why the frozen run uses more GPU memory (49.9 vs 31.9 GB). A guess, not checked: gradient checkpointing only runs in train mode and the frozen compressor is in eval mode. It fits on 80 GB.

## Proposed text for the owner's chats (provisional until seed 44)
- docs/training_plan.md, gate 0 of "Standing proposal": "Decoder-only finetune on frozen `<MEM>`s (campaign freeze, 2 seeds, n = 1,500 per set): −2.8 / −3.5 pt on documents and −2.3 / −2.7 pt on MRQA text (paired CI below 0 at both seeds), photos −.9 / −1.3 (not significant), HotpotQA −2.8 / −2.3. About 20% faster training (same-time seed-44 pair: 71 vs 91 s/step). The finetune does use the compressor; a third seed decides between 'tie' and 'failed'."
- HANDOFF.md thread 7: "Frozen compressor costs ~2-3 pt; every source is lower (most per-source n 250-500, mostly inside noise); stage B would need either the split test (connector-only / compressor-only frozen, a new flag in `pisco/`) or accepting ~2-3 pt."

## Notes this wake
- Wake 8: submitted the last 3 jobs of the seed-44 chain (2 module-move checks, 1 paired test), all `afterok`; `--time` 0:05:00 from measured 47 s / 19 s.
- Wake 8: applied the verifier's marks: photos no longer folded into "costs 1-3.5 pt on every set"; hotpot_bridge n = 1,197; seed-44 reference s/step updated (89 at .07 epoch → 91 at .16 epoch).
- Wake 8: probes recorded: 188651 s_per_step 71.2 @ .215; 188652 s_per_step 90.8, loss .5235 @ .156.

## Patches this wake
(none)

## Blockers / needs the owner
(none)
