# STATE — campaign freeze   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-12 13:00 tick (wake 9)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- **Gate 0 at two seeds = TIE under the law → third-seed pair trained, its evals are running.** Freezing the compressor and connector (only the decoder LoRA trains) costs 2.3-3.5 pt of judge accuracy on documents, MRQA text and HotpotQA at both seeds; photos are lower by .9 / 1.3 pt, inside the 2-pt noise band with a CI spanning 0. Paired per question, n = 1,500 each, same eval code, job 188650 (`paired_judge_freeze_188650.out`, `expmon grid freeze`):
  - documents: seed 42 −.028 [95% CI −.049, −.007], seed 43 −.035 [−.057, −.014] → CI wholly below 0 at both seeds;
  - photos: −.009 [−.026, +.008], −.013 [−.030, +.004] → not shown to differ;
  - MRQA text: −.023 [−.040, −.008], −.027 [−.043, −.009] → CI wholly below 0 at both seeds;
  - HotpotQA (reported, not gating): −.028 [−.045, −.013], −.023 [−.041, −.006].
  - Why a tie and not a fail: "Validated" needs ≥ −2 pt everywhere (documents and MRQA miss at both seeds); "Failed" needs ≤ −3 pt with the CI below 0 at 2 seeds on one set — documents seed 42 is −2.8 pt. The law sends this to a third seed and decides on 2 of 3.
- **The freeze held over all three full runs** (1.0 epoch, 28,032 rows each): every compressor and connector row, norm gain included, changed 0.0%; decoder lora_B 66.7% / 66.3% / 66.7% of entries changed (seeds 42 / 43 / 44, jobs 188584 / 188585 / 188699). The seed-44 reference run, checked the same way, moved compressor lora_B 74.1% and the connector MLP 20.7% (job 188700): the check sees movement when there is some.
- **In this bf16 recipe the connector norm gain never moves, even when trainable**: 0.0% changed after 20 steps (job 188579) and after a whole epoch (seed-44 reference, job 188700), value .46094 both times. So for the norm gain, 0.0% in a frozen run does not by itself prove the freeze; the frozen-vs-reference difference is compressor LoRA + merger + connector MLP only.
- **Feasibility gate PASS** (20-step smokes, 1,280 rows = .046 epoch): frozen side 0.0% changed (188578); the check sees movement in the unfrozen control (compressor LoRA 25.4% / 76.4%, connector 25.4%, 188579); train loss within .05 of the unfrozen smoke at steps 10 and 20 (.594 vs .639, .584 vs .580, jobs 188528 / 188534); config diff vs 184778 = `max_steps` + `freeze_compressor` only.
- Unit tests pass on the committed code (HEAD be0d842; `pisco/` and `test_*.py` clean): `test_positions.py` 12/12, `test_mm_collators.py` 33/33, `test_freeze.py` 5/5, exit 0 (job 188655).
- provisional: the loss is spread over the data sources, not one type: 28 of 28 per-source differences are negative (job 188650); most sources have n = 250-500 (hotpot_bridge n = 1,197); only mpdocvqa seed 43 (−.050), DROP seed 43 (−.060) and hotpot_bridge (both seeds) have a CI below 0; several sit inside the 2-pt noise band.
- provisional: frozen training is about 20-25% cheaper in time and needs more GPU memory. Seed-44 pair started the same minute, on two A100 nodes (homer-010 / homer-009): frozen 71.5 s/step vs reference 95.1 s/step over the full epoch (train_runtime 31,250 vs 41,540 s / 437 steps, −25%; jobs 188651 / 188652). Peak GPU memory (sacct gpumem) 50.0 GB frozen vs 34.0 GB reference. Earlier seeds, different nodes and dates: frozen 71.5 / 76.1 s/step (188582 / 188583) vs reference 90.3 / 113.2 (184778 / 184780).
- provisional: the frozen runs end at a HIGHER eval loss than the reference of the same seed (64 eval rows): .435 vs .414 (seed 42), .413 vs .375 (seed 43), .427 vs .401 (seed 44); the frozen curve is lower or equal early (.23 epoch: .455 vs .513, .442 vs .471, .456 vs .468) and flatter. Consistent with the judge gap; 64 rows, no paired test.
- provisional: previews at .23 epoch on the first 300 documents (closed, never gate): seed 42 .583 vs .610 (−.027, p .28), seed 43 .593 vs .610 (−.017, p .49) (jobs 188594 / 188595). Neither is significant.

**Open, and running**
- Arm R3, third seed (seed 44): both finetunes COMPLETED (frozen 8.7 h, reference 11.6 h, exit 0, no NaN). The 8 judge-scored evals are in the queue on `gpu-be`:
  - frozen side running since ~12:45 on homer-016 (documents 188653, photos 188672, MRQA 188674, HotpotQA 188676); photos and MRQA answers already written, judge step next;
  - reference side PENDING for a free `gpu-be` GPU outside homer-[005-009] (documents 188654, photos 188673, MRQA 188675, HotpotQA 188677). `gpu-be` was full for ~9 h after the finetunes ended: that, not a fault, is why the evals started late.
  - paired test 188701 (`paired_judge_cpu.sh ft_frozen_s44:ft_unfrozen_s44`) runs `afterok` all 8 evals. It decides the 3-seed reading.

## Owned jobs (jobid | arm | state | rows seen | eta | note)
| jobid | arm | state | rows seen | eta | note |
|---|---|---|---|---|---|
| 188653 / 188654 | R3 documents eval, frozen / reference | RUNNING homer-016 / PENDING (no free GPU) | - | ~13:30 / after a GPU frees + ~32 min | `PX=1003520`, `--time 0:40:00` |
| 188672 / 188673 | R3 photos eval, frozen / reference | RUNNING (judging) / PENDING | - | ~13:15 / +~19 min | `PX=1003520`, `--time 0:24:00` = 19:00 (188603) × 1.25 |
| 188674 / 188675 | R3 MRQA text eval, frozen / reference | RUNNING (judging) / PENDING | - | ~13:15 / +~17 min | `PX=1003520 CML=4096`, `--time 0:22:00` = 17:21 (188612) × 1.25 |
| 188676 / 188677 | R3 HotpotQA eval, frozen / reference | RUNNING / PENDING | - | ~13:25 / +~23 min | `PX=1003520 CML=4096 TOPK=10`, `--time 0:29:00` = 22:44 (188617) × 1.25 |
| 188701 | R3 paired test | PENDING afterok 8 evals | - | after the last reference eval | `paired_judge_cpu.sh ft_frozen_s44:ft_unfrozen_s44`, `--time 0:05:00` (measured 19 s, 188650) |
All evals: `eval_mm_4b.sh <model> pisco "" <set>`, gpu-be, `--exclude=homer-[005-009]`; eval code md5 `eval_mm_docqa.py` dfe32aa0, `eval_mm_4b.sh` 50abc167, `judge_mm_docqa.py` 02bc8748 (same as every seed-42/43 cell). Watch: `eval_mm_4b.sh` exits 0 when only its judge step fails; the paired test then exits 1 on the missing scores.
Finished this wake: finetunes 188651 / 188652 (COMPLETED, out `expQ/freeze/ft_frozen_s44` / `ft_unfrozen_s44`), module-move checks 188699 / 188700 (verdict closed).

GPU budget: ~49 GPU-h used of 192 (seeds 42/43 ~28.5 + seed-44 training 20.3 + evals so far); the 8 seed-44 evals add ~3.

## Adopted jobs
(none — the other jobs in the queue belong to the sibling campaigns `longchunk` and `probes`)

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| F frozen vs R reference, seeds 42 + 43 | evaluated: tie | — (cells closed with their numbers) |
| R3 seed-44 pair | trained; 4 evals running, 4 waiting for a GPU, paired test chained | when 188701 ends: verdicts on the 8 evals + the paired test; the 3-seed decision (2 of 3) |

What seed 44 can decide (written before its result; the law's "2 of 3" reading in CAMPAIGN.md "Tie"):
- arm F cannot validate any more: documents and MRQA already miss ≥ −2 pt at 2 of 2 seeds;
- FAILED if seed 44 documents is ≤ −3 pt with its CI wholly below 0 (seed 43 already is → 2 of 3); MRQA cannot reach 2 of 3 (both seeds above −3);
- otherwise still TIED after three seeds → BLOCKED.md for the owner (a consistent ~2-3 pt loss that the gate's bands do not classify).
- If FAILED: the law forbids variants of freezing; BLOCKED.md proposes the split test (freeze the connector only / the compressor only; a new flag in `pisco/`, written by the owner's chats).

## Training curves — eval_loss ↓ (64 eval rows, same seed = same eval rows)
| epoch | .23 | .46 | .69 | .92 | 1.0 |
|---|---|---|---|---|---|
| R seed 42 (184778) | .513 | .455 | .393 | .394 | .414 |
| R seed 43 (184780) | .471 | .439 | .411 | .441 | .375 |
| R seed 44 (188652) | .468 | .414 | .421 | .417 | .401 |
| F seed 42 (188582) | .455 | .444 | .438 | .434 | .435 |
| F seed 43 (188583) | .442 | .431 | .416 | .415 | .413 |
| F seed 44 (188651) | .456 | .444 | .427 | .428 | .427 |

## Deliverable — judge accuracy ↑ (Qwen3-8B rubric judge), n = 1,500 per set, paired diff to R of the same seed [95% CI] (job 188650)
| model | what trains | rate (tokens per slot; slots per page) | documents s42 / s43 | Δ docs | photos s42 / s43 | Δ photos | MRQA text s42 / s43 | Δ MRQA | HotpotQA s42 / s43 | Δ HotpotQA |
|---|---|---|---|---|---|---|---|---|---|---|
| R reference (184778 / 184780, re-evaluated 11-11 on the current code) | compressor LoRA + merger + connector + decoder LoRA | 8; 123 per 1M-px page (980 visual tokens) | .610 / .615 | — | .673 / .685 | — | .718 / .724 | — | .603 / .593 | — |
| F frozen (188582 / 188583) | decoder LoRA only | same | .582 / .580 | −.028 [−.049, −.007] / −.035 [−.057, −.014] | .664 / .672 | −.009 [−.026, +.008] / −.013 [−.030, +.004] | .695 / .697 | −.023 [−.040, −.008] / −.027 [−.043, −.009] | .575 / .571 | −.028 [−.045, −.013] / −.023 [−.041, −.006] |
| R / F seed 44 (188652 / 188651) | as above | same | pending | | pending | | pending | | pending | |
| uncompressed decoder (earlier runs, same questions) | — | 1 (raw tokens) | .809 | | .717 | | .807 | | — | |
Discordant pairs (frozen-only right / reference-only right), McNemar p: documents 111/153 p .011, 109/161 p .002; photos 78/92 p .32, 78/97 p .17; MRQA 59/94 p .006, 67/107 p .003; HotpotQA 64/106 p .002, 68/102 p .011.

## Module movement (entries changed %, per module, vs the stage-2 model 184777)
| module | F smoke s42 (188578) | unfrozen smoke s42 (188579) | F full s42 (188584) | F full s43 (188585) | F full s44 (188699) | R full s44 (188700) |
|---|---|---|---|---|---|---|
| decoder lora_A / lora_B | 30.6 / 76.3 | 26.7 / 75.8 | 14.7 / 66.7 | 14.7 / 66.3 | 14.8 / 66.7 | 15.1 / 67.8 |
| compressor lora_A / lora_B | 0.0 / 0.0 | 25.4 / 76.4 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 16.9 / 74.1 |
| compressor `<MEM>` token row | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| connector MLP | 0.0 | 25.4 | 0.0 | 0.0 | 0.0 | 20.7 (moved .0073 of its norm) |
| connector norm gain | 0.0 | 0.0 (bf16 lost update) | 0.0 | 0.0 | 0.0 | 0.0 (bf16 lost update) |
Odd, not explained: the full runs changed FEWER decoder entries than the 20-step smokes (≈15% vs ≈27-31% of lora_A) although they moved further in norm (lora_B .070 vs .025 of its norm). The seed-44 reference shows the same pattern, so it is the schedule, not the freeze. A guess, not checked: the smoke hit peak lr at step 1 (1 warm-up step), when Adam's first update is ±lr for every entry; the full run warms up over 22 steps and most small updates are lost in bf16.

## Training cost (A100-80GB) — provisional (one same-time pair, two nodes)
| run | s/step | wall | peak GPU memory (sacct gpumem) | trainable params |
|---|---|---|---|---|
| F frozen s42 / s43 / s44 (188582 / 188583 / 188651) | 71.5 / 76.1 / 71.5 (train_runtime / 437) | 8.7 / 9.3 / 8.7 h | 49.9 / 49.9 / 50.0 GB | 155,978,240 (decoder only) |
| R reference s42 / s43 / s44 (184778 / 184780 / 188652) | 90.3 / 113.2 / 95.1 | 11.0 / 13.8 / 11.6 h | — / — / 34.0 GB | 307,771,392 |
| unfrozen smoke (188534) | 88-90 steady | — | 31.9 GB | 307,771,392 |
Open question: why the frozen run uses MORE GPU memory (50.0 vs 34.0 GB, seed-44 pair). A guess, not checked: gradient checkpointing only runs in train mode and the frozen compressor is in eval mode, so its activations are kept. It fits on 80 GB.

## Proposed text for the owner's chats (provisional until seed 44's judge result)
- docs/training_plan.md, gate 0 of "Standing proposal": "Decoder-only finetune on frozen `<MEM>`s (campaign freeze, 2 seeds, n = 1,500 per set): −2.8 / −3.5 pt on documents and −2.3 / −2.7 pt on MRQA text (paired CI below 0 at both seeds), photos −.9 / −1.3 (not significant), HotpotQA −2.8 / −2.3. About 25% faster training (same-time seed-44 pair: 71.5 vs 95.1 s/step) but 50 vs 34 GB peak GPU memory. The finetune does use the compressor; a third seed decides between 'tie' and 'failed'."
- HANDOFF.md thread 7: "Frozen compressor costs ~2-3 pt; every source is lower (most per-source n 250-500, mostly inside noise); stage B would need either the split test (connector-only / compressor-only frozen, a new flag in `pisco/`) or accepting ~2-3 pt."

## Notes this wake
- Wake 9 (scheduled, the 16 h max of the previous wait expired): both seed-44 finetunes COMPLETED (frozen 03:17, reference 06:11), module-move checks 188699 / 188700 done → verdicts closed. Probes recorded: 188651 eval_loss .427, 71.5 s/step @ 1.0; 188652 eval_loss .401, 95.1 s/step @ 1.0.
- The 8 evals sat ~9.5 h waiting for `gpu-be` (all GPUs busy; no reservation; the only free A100s now are on homer-007/009, excluded, and homer-013). Frozen-side evals started ~12:45. Nothing to do: they are chained, the paired test follows.
- Verifier: all five provisional bullets confirmed by VERIFY.md; nothing to fix. The norm-gain bullet is now settled by the full-epoch check 188700.

## Patches this wake
(none)

## Blockers / needs the owner
(none)
