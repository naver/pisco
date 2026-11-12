# STATE — campaign freeze   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-12 14:00 tick (wake 10)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- **Gate 0 FAILED on three seeds: the question-answering finetune needs the compressor side to train.** With the compressor and connector frozen (only the decoder LoRA trains), documents lose 2.8 / 3.5 / 4.1 pt of judge accuracy at seeds 42 / 43 / 44, each with a 95% CI wholly below 0; the law's "Failed" rule (≤ −3 pt with the CI below 0 at 2 of 3 seeds on one gating set) is met by seeds 43 and 44. Paired per question, n = 1,500 per set, same eval code (md5 `eval_mm_docqa.py` dfe32aa0, `eval_mm_4b.sh` 50abc167, `judge_mm_docqa.py` 02bc8748, unchanged since 11-11 06:45), jobs 188650 (seeds 42 / 43) and 188701 (seed 44):
  - documents: −.028 [−.049, −.007], −.035 [−.057, −.014], −.041 [−.062, −.021] → FAILED;
  - MRQA text: −.023 [−.040, −.008], −.027 [−.043, −.009], −.025 [−.041, −.009] → below the −2 pt bar at all three seeds, CI below 0, not ≤ −3 pt;
  - photos: −.009 [−.026, +.008], −.013 [−.030, +.004], +.005 [−.013, +.021] → no difference shown at any seed;
  - HotpotQA (reported, not gating): −.028 [−.045, −.013], −.023 [−.041, −.006], −.026 [−.043, −.010].
  - The prediction in CAMPAIGN.md "Mission" (the reference finetune barely moves the connector, so freezing may tie) is refuted and marked there.
- **The seed-44 pair is clean:** both finetunes ran from the same stage-2 model with the same recipe file (`ft_freeze.sh frozen|unfrozen 44`); all 8 evals ran on the same node (homer-016) with the eval-code md5 above; the paired test exited 0 (job 188701).
- **The freeze held over all three full runs** (1.0 epoch, 28,032 rows each): every compressor and connector row, norm gain included, changed 0.0%; decoder lora_B 66.7% / 66.3% / 66.7% of entries changed (seeds 42 / 43 / 44, jobs 188584 / 188585 / 188699). The seed-44 reference run, checked the same way, moved compressor lora_B 74.1% and the connector MLP 20.7% (job 188700): the check sees movement when there is some.
- **In this bf16 recipe the connector norm gain never moves, even when trainable**: 0.0% changed after 20 steps (job 188579) and after a whole epoch (seed-44 reference, job 188700), value .46094 both times. So the frozen-vs-reference difference is compressor LoRA + merger + connector MLP only.
- **Feasibility gate PASS** (20-step smokes, 1,280 rows = .046 epoch): frozen side 0.0% changed (188578); the check sees movement in the unfrozen control (compressor LoRA 25.4% / 76.4%, connector 25.4%, 188579); train loss within .05 of the unfrozen smoke at steps 10 and 20 (.594 vs .639, .584 vs .580, jobs 188528 / 188534); config diff vs 184778 = `max_steps` + `freeze_compressor` only.
- Unit tests pass on the committed code (HEAD be0d842; `pisco/` and `test_*.py` clean): `test_positions.py` 12/12, `test_mm_collators.py` 33/33, `test_freeze.py` 5/5, exit 0 (job 188655).
- provisional: the loss is spread over the data sources, not one type: 39 of 42 per-source differences are negative over the three seeds (28 of 28 at seeds 42 / 43, job 188650; 11 of 14 at seed 44, job 188701 — the 3 positive ones are photos realworldqa +.010, vizwiz +.020 and hotpot_comparison +.013). Most per-source n = 250-500 (hotpot_bridge n = 1,197). Per-source CIs below 0: mpdocvqa s43, DROP s43, infovqa s44, RelationExtraction s44, hotpot_bridge at all three seeds.
- provisional: frozen training is about 25% cheaper in time and needs more GPU memory. Seed-44 pair started the same minute, on two A100 nodes (homer-010 / homer-009): frozen 71.5 s/step vs reference 95.1 s/step over the full epoch (train_runtime 31,250 vs 41,540 s / 437 steps, −25%; jobs 188651 / 188652). Peak GPU memory 50,020 MB frozen vs 34,044 MB reference (`sacct -o TRESUsageInMax`, gres/gpumem, batch step). Earlier seeds, different nodes and dates: frozen 71.5 / 76.1 s/step (188582 / 188583) vs reference 90.3 / 113.2 (184778 / 184780).
- provisional: the frozen runs end at a HIGHER eval loss than the reference of the same seed (64 eval rows): .435 vs .414 (seed 42), .413 vs .375 (seed 43), .427 vs .401 (seed 44); the frozen curve is lower or equal early (.23 epoch) and flatter. Consistent with the judge gap; 64 rows, no paired test.
- provisional: previews at .23 epoch on the first 300 documents (closed, never gate): seed 42 .583 vs .610 (−.027, p .28), seed 43 .593 vs .610 (−.017, p .49) (jobs 188594 / 188595). Neither is significant.

**Open, and running**
- Nothing is running. The law's next step after a FAILED gate is the owner's: BLOCKED.md asks whether to run the split test (freeze only the compressor, or only the connector; needs a new flag in `pisco/`), accept the ~3 pt cost, or close the campaign.

## Owned jobs (jobid | arm | state | rows seen | eta | note)
(none in the queue — every campaign cell has ended and has a verdict)
Finished this wake: 8 seed-44 evals (frozen 188653 / 188672 / 188674 / 188676, reference 188654 / 188673 / 188675 / 188677, COMPLETED on homer-016, 12:44-13:35) and the paired test 188701 (COMPLETED 13:36, exit 0).
GPU budget: ~52 GPU-h used of 192 (estimate from wall times: training 26.7 h frozen + 11.6 h seed-44 reference + ~1.2 h smokes; ~12 h of evals).

## Adopted jobs
(none — the other jobs in the queue belong to the sibling campaigns `longchunk` and `probes`)

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| F frozen vs R reference, seeds 42 / 43 / 44 | decided: FAILED (documents, 2 of 3 seeds) | — (all cells closed with their numbers) |
| split test (freeze the compressor only / the connector only) | proposed by the law after a FAIL; not in the arms table; needs a new flag in `pisco/` (not patchable here) | the owner, BLOCKED.md |
| after-gate arms for a VALIDATED gate (higher decoder lr, fp32 masters, second pass) | void: the gate failed; the law forbids variants of freezing | — |

## Training curves — eval_loss ↓ (64 eval rows, same seed = same eval rows)
| epoch | .23 | .46 | .69 | .92 | 1.0 |
|---|---|---|---|---|---|
| R seed 42 (184778) | .513 | .455 | .393 | .394 | .414 |
| R seed 43 (184780) | .471 | .439 | .411 | .441 | .375 |
| R seed 44 (188652) | .468 | .414 | .421 | .417 | .401 |
| F seed 42 (188582) | .455 | .444 | .438 | .434 | .435 |
| F seed 43 (188583) | .442 | .431 | .416 | .415 | .413 |
| F seed 44 (188651) | .456 | .444 | .427 | .428 | .427 |

## Deliverable — judge accuracy ↑ (Qwen3-8B rubric judge), n = 1,500 per set; Δ = paired difference frozen − reference of the same seed [95% CI] (jobs 188650, 188701)
| model | what trains | rate (tokens per slot; slots per page) | documents ↑ s42 / s43 / s44 | photos ↑ s42 / s43 / s44 | MRQA text ↑ s42 / s43 / s44 | HotpotQA ↑ s42 / s43 / s44 |
|---|---|---|---|---|---|---|
| R reference (184778 / 184780 re-evaluated 11-11 on the current code; 188652) | compressor LoRA + merger + connector + decoder LoRA | 8; 123 per 1M-px page (980 visual tokens) | .610 / .615 / .620 | .673 / .685 / .671 | .718 / .724 / .721 | .603 / .593 / .603 |
| F frozen (188582 / 188583 / 188651) | decoder LoRA only | same | .582 / .580 / .579 | .664 / .672 / .675 | .695 / .697 / .697 | .575 / .571 / .577 |
| uncompressed decoder (earlier runs, same questions) | — | 1 (raw tokens) | .809 | .717 | .807 | — |

| paired Δ frozen − reference ↑ (CI) | seed 42 | seed 43 | seed 44 | mean of the 3 Δ |
|---|---|---|---|---|
| documents | −.028 [−.049, −.007] | −.035 [−.057, −.014] | −.041 [−.062, −.021] | −.035 |
| photos | −.009 [−.026, +.008] | −.013 [−.030, +.004] | +.005 [−.013, +.021] | −.006 |
| MRQA text | −.023 [−.040, −.008] | −.027 [−.043, −.009] | −.025 [−.041, −.009] | −.025 |
| HotpotQA | −.028 [−.045, −.013] | −.023 [−.041, −.006] | −.026 [−.043, −.010] | −.026 |
Discordant pairs (frozen-only right / reference-only right), McNemar p — seeds 42 / 43 / 44: documents 111/153 p .011, 109/161 p .002, 103/164 p < .001; photos 78/92 p .32, 78/97 p .17, 90/83 p .65; MRQA 59/94 p .006, 67/107 p .003, 58/95 p .003; HotpotQA 64/106 p .002, 68/102 p .011, 64/103 p .003.

## Module movement (entries changed %, per module, vs the stage-2 model 184777)
| module | F smoke s42 (188578) | unfrozen smoke s42 (188579) | F full s42 (188584) | F full s43 (188585) | F full s44 (188699) | R full s44 (188700) |
|---|---|---|---|---|---|---|
| decoder lora_A / lora_B | 30.6 / 76.3 | 26.7 / 75.8 | 14.7 / 66.7 | 14.7 / 66.3 | 14.8 / 66.7 | 15.1 / 67.8 |
| compressor lora_A / lora_B | 0.0 / 0.0 | 25.4 / 76.4 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 16.9 / 74.1 |
| compressor `<MEM>` token row | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| connector MLP | 0.0 | 25.4 | 0.0 | 0.0 | 0.0 | 20.7 (moved .0073 of its norm) |
| connector norm gain | 0.0 | 0.0 (bf16 lost update) | 0.0 | 0.0 | 0.0 | 0.0 (bf16 lost update) |
Odd, not explained: the full runs changed FEWER decoder entries than the 20-step smokes (≈15% vs ≈27-31% of lora_A) although they moved further in norm. The seed-44 reference shows the same pattern, so it is the schedule, not the freeze. A guess, not checked: the smoke hit peak lr at step 1; the full run warms up over 22 steps and most small updates are lost in bf16.

## Training cost (A100-80GB) — provisional (one same-time pair, two nodes)
| run | s/step | wall | peak GPU memory (sacct gres/gpumem) | trainable params |
|---|---|---|---|---|
| F frozen s42 / s43 / s44 (188582 / 188583 / 188651) | 71.5 / 76.1 / 71.5 (train_runtime / 437) | 8.7 / 9.3 / 8.7 h | 49.9 / 49.9 / 50.0 GB | 155,978,240 (decoder only) |
| R reference s42 / s43 / s44 (184778 / 184780 / 188652) | 90.3 / 113.2 / 95.1 | 11.0 / 13.8 / 11.6 h | — / — / 34.0 GB | 307,771,392 |
| unfrozen smoke (188534) | 88-90 steady | — | 31.9 GB | 307,771,392 |
Open question: why the frozen run uses MORE GPU memory. A guess, not checked: gradient checkpointing only runs in train mode and the frozen compressor is in eval mode, so its activations are kept.

## Proposed text for the owner's chats (the gate is decided)
- docs/training_plan.md, gate 0 of "Standing proposal": "FAILED (campaign freeze, 3 seeds, n = 1,500 per set, paired judge). A decoder-only finetune on frozen `<MEM>`s loses 2.8 / 3.5 / 4.1 pt on documents and 2.3 / 2.7 / 2.5 pt on MRQA text (CI below 0 at every seed), 2.3-2.8 pt on HotpotQA; photos tie (−.9 / −1.3 / +.5, not significant). Training is ~25% faster (71.5 vs 95.1 s/step, same-time pair) but peaks at 50 vs 34 GB GPU memory. Stage B as proposed (compress once, train the decoder only) costs ~3 pt on documents and text. Open: does freezing only the compressor (connector trains, `<MEM>`s cacheable before the connector) close the gap?"
- HANDOFF.md thread 7: "Freezing compressor + connector at finetune fails gate 0 (documents −3.5 pt mean of 3 seeds, MRQA −2.5, photos ~0). The loss is spread over sources (39 of 42 per-source differences negative). Next, if pursued: the split test, which needs a `pisco/` flag to freeze the compressor and the connector separately."

## Notes this wake
- Wake 10 (condition: paired test 188701 left the queue). Seed-44 evals and the paired test COMPLETED; verdicts recorded on all 8 evals, the paired test and the 4 finetune cells of arms F / R3; probes recorded (judge accuracy, n = 1,500) on the 8 evals.
- Verifier asked to re-check the eval-code md5 before reading 188701: done — the three files are unchanged since 11-11 06:45 (mtime) and match the ledger md5; `git status` shows them modified relative to the commit, but the evals of both sides of every pair ran on the same working copy.
- The 34.0 GB reference peak now has a source: `sacct -j 188652 -o TRESUsageInMax` gres/gpumem = 34,044 MB (188651: 50,020 MB).
- CAMPAIGN.md: the Mission's "why it may tie" prediction marked refuted (kept, not deleted).
- `expmon display --exp freeze` still writes 0 table rows (already filed twice); the table above is the deliverable.

## Patches this wake
(none)

## Blockers / needs the owner
- BLOCKED.md: the gate failed; the law hands the next step (split test / accept the cost / close) to the owner.
