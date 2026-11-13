# STATE — campaign freeze   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-13 17:00 tick (wake 20; woken because A1 seed 43 left the queue — the wait condition written at 15:30)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- **Gate 0 FAILED on three seeds: the question-answering finetune needs the compressor side to train.** With the compressor and connector frozen (only the decoder LoRA trains), documents lose 2.8 / 3.5 / 4.1 pt of judge accuracy at seeds 42 / 43 / 44, each with a 95% CI wholly below 0; the law's "Failed" rule (≤ −3 pt with the CI below 0 at 2 of 3 seeds on one gating set) is met by seeds 43 and 44. Paired per question, n = 1,500 per set, same eval code (md5 `eval_mm_docqa.py` dfe32aa0, `eval_mm_4b.sh` 50abc167, `judge_mm_docqa.py` 02bc8748), jobs 188650 (seeds 42 / 43) and 188701 (seed 44), `ALL` rows:
  - documents: −.028 [−.049, −.007], −.035 [−.057, −.014], −.041 [−.062, −.021] → FAILED;
  - MRQA text: −.023 [−.040, −.008], −.027 [−.043, −.009], −.025 [−.041, −.009] → below the −2 pt bar at all three seeds, CI below 0, not ≤ −3 pt;
  - photos: −.009 [−.026, +.008], −.013 [−.030, +.004], +.005 [−.013, +.021] → no difference shown at any seed;
  - HotpotQA (reported, not gating): −.028 [−.045, −.013], −.023 [−.041, −.006], −.026 [−.043, −.010].
  - The prediction in CAMPAIGN.md "Mission" (the reference finetune barely moves the connector, so freezing may tie) is refuted and marked there.
- **The seed-44 pair is clean:** both finetunes ran from the same stage-2 model with the same recipe file (`ft_freeze.sh frozen|unfrozen 44`); all 8 evals used the eval-code md5 above; the paired test exited 0 (job 188701).
- **The freeze held over all three full F runs** (1.0 epoch, 27,936 rows each: log `num_rows: 27936`, 437 steps): every compressor and connector row, norm gain included, changed 0.0%; decoder lora_B 66.7% / 66.3% / 66.7% of entries changed (seeds 42 / 43 / 44, jobs 188584 / 188585 / 188699). The seed-44 reference run, checked the same way, moved compressor lora_B 74.1% and the connector MLP 20.7% (job 188700): the check sees movement when there is some. It cannot see the `<MEM>` row and has no row for the merger or the vision tower.
- **In this bf16 recipe the connector norm gain never moves, even when trainable**: 0.0% changed after 20 steps (jobs 188579, 189066) and after a whole epoch (seed-44 reference 188700, A1 seed 42 189099, A1 seed 43 189450), value .46094 each time. (The A2 smoke check 189740 also reads 0.0%, but there the gain is FROZEN: it shows the freeze working, not a lost update.)
  - provisional: so the frozen-vs-reference difference is compressor LoRA + connector MLP, plus possibly the merger and the compressor `<MEM>` row — neither is measured: the merger has no row in the check, and the `<MEM>` row sits inside a 1,270M-entry token-adapter group where one changed 2,560-entry row rounds to 0.0% (no positive control for either). A second, non-weight difference: the frozen side runs in eval mode (no LoRA dropout, and possibly no gradient checkpointing, not checked in the code).
- **Split-freeze setting installed and tested**: `PISCOConfig.freeze_part = compressor | connector` (default unset) in `pisco/model.py` (md5 994cff3d, unchanged at 17:05), 5 new `test_freeze.py` cases; staged copy (job 189060) and installed tree (job 189063) both pass `test_positions.py` 12/12, `test_mm_collators.py` 33/33, `test_freeze.py` 10/10, exit 0.
- **Split arm A1 (compressor frozen) feasibility PASS** (20-step smoke, seed 42, 1,280 rows = .046 epoch; jobs 189064 + 189066): log `froze 501 compressor tensors` + `Compressor trainable parameters: 0`; compressor rows 0.0%, connector MLP 28.7%, decoder lora_B 76.2%; train loss .6033 / .5952 at steps 10 / 20 vs the unfrozen smoke's .639 / .580 (188534), gaps .036 / .015; config diff one line (`freeze_compressor: true` → `freeze_part: compressor`); 81.2 s/step, 49.8 GB peak.
- **A1 seed 42 full run COMPLETED and its freeze held** (job 189094, homer-013, exit 0, 437 of 437 steps = 1.0 epoch, 27,936 rows, 8.7 h, 71.5 s/step, peak GPU memory 50.2 GB). After-run module check (job 189099, exit 0): compressor lora_A / lora_B / token-adapter 0.0%; connector MLP 19.1% changed (.0061 of its norm); norm gain 0.0%; decoder lora_B 66.0%.
- **A1 seed 43 full run COMPLETED and its freeze held** (job 189445, homer-020, exit 0 at 16:39, 437 of 437 steps = 1.0 epoch, 27,936 rows, 11.2 h, 91.7 s/step overall, peak GPU memory 50.1 GB, train loss .481). After-run module check (job 189450, exit 0): compressor lora_A / lora_B / token-adapter 0.0%; connector MLP 19.4% changed (.0061 of its norm); norm gain 0.0%; decoder lora_B 66.4%.
- **Split arm A2 (connector frozen) feasibility PASS** (20-step smoke, seed 42, 1,280 rows = .046 epoch, homer-019; jobs 189739 + 189740, both exit 0): log `freeze_part=connector: froze 5 connector tensors (eval mode kept)` + `Compressor trainable parameters: 130812416` (> 0); module check: connector MLP 0.0%, norm gain 0.0%, compressor lora_A / lora_B 25.4 / 76.2%, decoder lora_A / lora_B 26.9 / 75.8%; train loss .6165 / .561 at steps 10 / 20 vs the unfrozen smoke's .639 / .580 (188534), gaps .023 / .019 (< .05); `training_config.yaml` diff vs `ft_frozen_s42_smoke` = one line (`freeze_compressor: true` → `freeze_part: connector`; output paths are not in that file); 98.9 s/step overall (~88 steady), peak GPU memory 31.4 GB.
- provisional: the gate-0 loss is spread over the data sources: 39 of 42 per-source differences are negative over the three seeds (28 of 28 at seeds 42 / 43, job 188650; 11 of 14 at seed 44, job 188701 — the 3 positive ones are photos realworldqa +.010, vizwiz +.020 and hotpot_comparison +.013). A sign count, not a test: most per-source CIs span 0 (per-source n = 250-500, hotpot_bridge 1,197).
- provisional: a frozen compressor costs ~25% less time but ~16 GB MORE GPU memory. Seed-44 pair: frozen 71.5 s/step vs reference 95.1 (grid probes of 188651 / 188652, two different A100 nodes, so node speed is mixed in); peak GPU memory (sacct gres/gpumem) 50,020 MB (188651) vs 34,044 MB (188652). The smokes split it by part: compressor frozen 49.7 GB (F, 188528) and 49.8 GB (A1, 189064); compressor training 31.9 GB (unfrozen, 188534) and 31.4 GB (A2, 189739); the two A1 full runs peak at 50.2 / 50.1 GB (189094 / 189445). Consistent with the guess that the frozen compressor runs in eval mode, where gradient checkpointing is off, so its activations are kept; not checked in the code. (Time is not consistent across nodes: A1 s43 took 91.7 s/step on homer-020 vs 71.5 for A1 s42 on homer-013.)
- provisional: the fully frozen runs end at a HIGHER eval loss than the reference of the same seed (64 eval rows, no paired test): .435 vs .414, .413 vs .375, .427 vs .401 (seeds 42 / 43 / 44; the .414 / .375 of 184778 / 184780 are not campaign cells).

**Open, and running**
- **Split test (owner's choice 2026-11-12): A1 first, then A2, seeds 42 and 43, same gates as gate 0.**
  - A1 = compressor frozen (adapter, `<MEM>` row, merger); connector + decoder LoRA train. Both seeds trained, both freezes held (above).
    - provisional: held-out loss (64 rows) at .23 / .46 / .69 / .92 / 1.0 epoch — seed 42 (189094) .459 / .449 / .437 / .435 / .435, the same as F s42 (188582) .455 / .444 / .438 / .434 / .435, not the reference 184778 .513 / .455 / .393 / .394 / .414; seed 43 (189445) .464 / .449 / .431 / .429 / .428, above F s43 (188583) .442 / .431 / .416 / .415 / .413 at every probe, above the reference 184780 .471 / .439 / .411 / .441 / .375 at .46, .69 and 1.0 epoch and below it at .23 and .92. Both seeds end above the reference. Consistent with the prediction "A1 fails like F"; a 64-row loss decides nothing.
    - provisional: the seed-42 step-100 preview (.23 epoch, job 189434) scores judge .593 on the first 300 documents; on the same 300 questions (query-id order checked identical): F s42 step-100 preview .583, F s42 final .590, reference 184778 final .610 (no reference checkpoint-100 exists). Discordant pairs A1-preview-only / reference-final-only right 17 / 22; A1-preview / F-preview 6 / 3. n = 300, never gates. The job finished (exit 0, 16:31, homer-013) and was then marked PREEMPTED and requeued; the requeued run only repeats it.
    - Judged evals (documents / photos / MRQA / HotpotQA): seed 42 jobs 189095-189098, seed 43 jobs 189446-189449, all PENDING on `gpu-be` (scheduler estimates at 17:05: 11-14 23:45 to 11-15 01:00).
  - A2 = connector frozen; compressor LoRA + merger + decoder LoRA train.
    - Seed 42 RUNNING on homer-019 since ~15:05 (job 189842, `gpu` A100): step 75 of 437 at ~17:01 (4,800 rows, .17 epoch), ~89 s/step, no NaN, no held-out probe yet (first at step 100, ~17:40). End ~01:55 on 11-14 (limit ~06:08). Chained `afterok`: 4 judged evals 189844-189847 and module check 189843.
    - Seed 43 SUBMITTED this wake (job 189899, `ft_freeze.sh connfrozen 43`, `gpu` A100, --time 15:03:00 = 98.9 s/step × 438 × 1.25): PENDING Priority, scheduler estimate 11-14 04:45. Chained `afterok`: evals 189900 documents, 189901 photos, 189902 MRQA (CML=4096), 189903 HotpotQA (CML=4096 TOPK=10), all PX=1003520 on `gpu-be` excluding homer-[005-009]; module check 189904.
  - Prediction, not a result (written before any A1 / A2 judge number): A1 fails like F, A2 ties R.

## Owned jobs (jobid | arm | state | rows seen | eta | note)
| job | what | state at 17:05 | rows | eta | note |
|---|---|---|---|---|---|
| 189434 | A1 s42 step-100 PREVIEW, first 300 documents | result written (judge .593), then requeued: PENDING Priority | 300 Q | est. 11-15 00:45 (a repeat) | leave it; it rewrites the same file |
| 189095-189098 | A1 s42 evals: documents / photos / MRQA (CML=4096) / HotpotQA (CML=4096 TOPK=10, not gating) | PENDING (Priority / node availability) | 1,500 Q each | est. 11-14 23:45 to 11-15 01:00 | --time 0:40 / 0:25 / 0:22 / 0:29 |
| 189446-189449 | A1 s43 evals, same env and --time | PENDING Priority | 1,500 Q each | est. 11-15 00:00 | pair with ref_s43 files 188599 / 188604 / 188613 / 188617 |
| 189842 | A2 full `ft_freeze.sh connfrozen 42` | RUNNING (homer-019) | 4,800 of 27,936 (.17 epoch) | ~01:55 on 11-14; checkpoint-100 ~17:40 | kill on NaN or 2 held-out probes above .513 (184778 at step 100); model → `expQ/freeze/ft_connfrozen_s42/model` |
| 189844-189847 | A2 s42 evals, same env and --time as A1 | PENDING afterok 189842 | 1,500 Q each | after the run | pair with ref_s42 files 188597 / 188602 / 188606 / 188615 |
| 189843 | module move, A2 s42 full vs 184777 | PENDING afterok 189842 | — | | --time 0:02:00 |
| 189899 | A2 full `ft_freeze.sh connfrozen 43` | PENDING Priority (`gpu` A100) | 0 | start est. 11-14 04:45, end ~11 h later | kill on NaN or 2 held-out probes above .471 (184780 at step 100); model → `expQ/freeze/ft_connfrozen_s43/model` |
| 189900-189903 | A2 s43 evals, same env and --time | PENDING afterok 189899 | 1,500 Q each | after the run | pair with ref_s43 files 188599 / 188604 / 188613 / 188617 |
| 189904 | module move, A2 s43 full vs 184777 | PENDING afterok 189899 | — | | --time 0:02:00 |
All evals: `gpu-be`, `--exclude=homer-[005-009]`, PX=1003520. Eval code md5 re-checked at 17:05: `eval_mm_docqa.py` dfe32aa0, `eval_mm_4b.sh` 50abc167, `judge_mm_docqa.py` 02bc8748 = gate 0, so the judged reference files of 184778 / 184780 still pair as they are. Re-check before reading the A1 / A2 evals.
Trainings in flight: 189842 + 189899 = 2 = CAMP_MAX_TRAIN. Owner's `gpu` cap: 5 of 8 GPUs in use at 17:03 (longchunk stage 1 4 + A2 s42 1); 189899 will make 6.
GPU budget: ~82 GPU-h used of 192 (gate 0 ~52 + A1 smoke .5 + A1 s42 8.7 + A1 s43 11.2 + A2 smoke .6 + A2 s42 2.0 so far + preview .2 + checks); left in the plan: A2 s42 ~9 h, A2 s43 ~11 h, 16 evals ≈ 8 h → ~28 GPU-h (+ ~22 per arm on a tie).

## Adopted jobs
(none — the other jobs in the queue belong to the sibling campaigns `longchunk` and `probes`)

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| F frozen vs R reference, seeds 42 / 43 / 44 | decided: FAILED (documents, 2 of 3 seeds) | — |
| A1 compressor frozen | both seeds trained + freezes checked; preview done; 8 evals queued (est. 11-14 23:45 to 11-15 01:00) | when both seeds are judged: re-check eval md5, then `paired_judge_cpu.sh` A1 vs ref_s42 / ref_s43 (and A1 vs F, reported), then the three gate rules |
| A2 connector frozen | s42 RUNNING (.17 epoch) + chain; s43 submitted + chain (PENDING, est. 11-14 04:45) | (1) when 189842 writes checkpoint-100 (~17:40): copy it without optimizer files (cmp each file) to `expQ/freeze/ft_connfrozen_s42_step100/checkpoint-100` before step 300 (~22:35) rotates it away, then launch the pre-authorised preview (`eval_mm_4b.sh <copy> pisco 300 eval_data/mm_docqa_1500_1m`, PX=1003520, `gpu-be` excl homer-[005-009], --time 0:40:00); (2) held-out probes every 100 steps vs 184778 / 184780; (3) same preview for 189899 at its checkpoint-100; (4) after each run: module check must show connector rows 0.0% |
| after-gate arms for a VALIDATED gate | void: gate 0 failed | — |

## Next
- The tick wakes this loop when A2 seed 42's checkpoint-100 appears (~17:40), or at 4 h. A new failure in any chained job also wakes it.
- At that wake: copy checkpoint-100, launch the A2 seed-42 preview, probe its first held-out loss. This wake used all 6 submissions (A2 s43 + its 5 chained jobs), so the preview could not go in now.

## Deliverable — judge accuracy ↑ (Qwen3-8B rubric judge), n = 1,500 per set; Δ = paired difference arm − reference of the same seed [95% CI] (jobs 188650, 188701)
| model | what trains | rate (tokens per slot; slots per page) | documents ↑ s42 / s43 / s44 | photos ↑ s42 / s43 / s44 | MRQA text ↑ s42 / s43 / s44 | HotpotQA ↑ s42 / s43 / s44 |
|---|---|---|---|---|---|---|
| R reference (184778 / 184780 re-evaluated 11-11; 188652) | compressor LoRA + merger + connector + decoder LoRA | 8; 123 per 1M-px page (980 visual tokens) | .610 / .615 / .620 | .673 / .685 / .671 | .718 / .724 / .721 | .603 / .593 / .603 |
| F frozen (188582 / 188583 / 188651) | decoder LoRA only | same | .582 / .580 / .579 | .664 / .672 / .675 | .695 / .697 / .697 | .575 / .571 / .577 |
| A1 compressor frozen (189094 / 189445) | connector + decoder LoRA | same | both trained, evals queued (est. 11-15); provisional: s42 step-100 preview .593 on 300 Q | | | |
| A2 connector frozen (189842 / 189899) | compressor LoRA + merger + decoder LoRA | same | s42 training (.17 epoch) / s43 queued | | | |
| uncompressed decoder (earlier runs, same questions) | — | 1 (raw tokens) | .809 | .717 | .807 | — |

| paired Δ frozen F − reference ↑ (CI) | seed 42 | seed 43 | seed 44 | mean of the 3 Δ |
|---|---|---|---|---|
| documents | −.028 [−.049, −.007] | −.035 [−.057, −.014] | −.041 [−.062, −.021] | −.035 |
| photos | −.009 [−.026, +.008] | −.013 [−.030, +.004] | +.005 [−.013, +.021] | −.006 |
| MRQA text | −.023 [−.040, −.008] | −.027 [−.043, −.009] | −.025 [−.041, −.009] | −.025 |
| HotpotQA | −.028 [−.045, −.013] | −.023 [−.041, −.006] | −.026 [−.043, −.010] | −.026 |
Discordant pairs (frozen-only right / reference-only right), McNemar p — seeds 42 / 43 / 44: documents 111/153 p .011, 109/161 p .002, 103/164 p < .001; photos 78/92 p .32, 78/97 p .17, 90/83 p .65; MRQA 59/94 p .006, 67/107 p .003, 58/95 p .003; HotpotQA 64/106 p .002, 68/102 p .011, 64/103 p .003.

Step-100 previews (judge accuracy ↑ on the first 300 documents of `mm_docqa_1500_1m`, provisional, never gates):
| model | at .23 epoch (checkpoint-100) | final (1.0 epoch), same 300 Q |
|---|---|---|
| R reference s42 (184778) | — (no checkpoint-100 kept) | .610 |
| F frozen s42 (preview 188594; final 188596) | .583 | .590 |
| A1 compressor frozen s42 (preview 189434) | .593 | evals queued |

Held-out loss ↓ (64 rows, no paired test; a health signal, not a gate), at .23 / .46 / .69 / .92 / 1.0 epoch:
| run | seed 42 | seed 43 |
|---|---|---|
| R reference (184778 / 184780 trainer_state) | .513 / .455 / .393 / .394 / .414 | .471 / .439 / .411 / .441 / .375 |
| F frozen (188582 / 188583 logs) | .455 / .444 / .438 / .434 / .435 | .442 / .431 / .416 / .415 / .413 |
| A1 compressor frozen (189094 / 189445 logs) | .459 / .449 / .437 / .435 / .435 | .464 / .449 / .431 / .429 / .428 |
| A2 connector frozen (189842 / 189899) | running (first probe at .23 epoch, ~17:40) | queued |

## Module movement (entries changed %, per module, vs the stage-2 model 184777)
| module | F smoke s42 (188578) | unfrozen smoke s42 (188579) | A1 smoke s42 (189066) | A2 smoke s42 (189740) | F full s42 (188584) | F full s43 (188585) | F full s44 (188699) | R full s44 (188700) | A1 full s42 (189099) | A1 full s43 (189450) |
|---|---|---|---|---|---|---|---|---|---|---|
| decoder lora_A / lora_B | 30.6 / 76.3 | 26.7 / 75.8 | 30.6 / 76.2 | 26.9 / 75.8 | 14.7 / 66.7 | 14.7 / 66.3 | 14.8 / 66.7 | 15.1 / 67.8 | 14.6 / 66.0 | 14.7 / 66.4 |
| compressor lora_A / lora_B | 0.0 / 0.0 | 25.4 / 76.4 | 0.0 / 0.0 | 25.4 / 76.2 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 16.9 / 74.1 | 0.0 / 0.0 | 0.0 / 0.0 |
| compressor token-adapter group (holds the `<MEM>` row; 1,270M entries, so one row is invisible) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| connector MLP | 0.0 | 25.4 | 28.7 (.0027 of its norm) | 0.0 | 0.0 | 0.0 | 0.0 | 20.7 (.0073 of its norm) | 19.1 (.0061 of its norm) | 19.4 (.0061 of its norm) |
| connector norm gain | 0.0 | 0.0 (bf16 lost update) | 0.0 (bf16 lost update) | 0.0 (frozen) | 0.0 | 0.0 | 0.0 | 0.0 (bf16 lost update) | 0.0 (bf16 lost update) | 0.0 (bf16 lost update) |
| merger / vision tower | not measured (no row in the check) | | | | | | | | | |

## Training cost (A100-80GB) — provisional (one same-time pair, two nodes)
| run | s/step | wall | peak GPU memory (sacct gres/gpumem) | trainable params |
|---|---|---|---|---|
| F frozen s42 / s43 / s44 (188582 / 188583 / 188651) | 71.5 / 76.1 / 71.5 | 8.7 / 9.3 / 8.7 h | 49.9 / 49.9 / 50.0 GB | 155,978,240 (decoder only) |
| R reference s42 / s43 / s44 (184778 / 184780 / 188652) | 90.3 / 113.2 / 95.1 | 11.0 / 13.8 / 11.6 h | — / — / 34.0 GB | 307,771,392 |
| A1 smoke s42 (189064), 20 steps | 81.2 overall (~70.5 steady) | 29 min | 49.8 GB | 176,958,976 (decoder + connector) |
| A1 full s42 / s43 (189094 homer-013 / 189445 homer-020) | 71.5 / 91.7 | 8.7 / 11.2 h | 50.2 / 50.1 GB | 176,958,976 |
| A2 smoke s42 (189739, homer-019), 20 steps | 98.9 overall (~88 steady) | 33 min | 31.4 GB | 286,790,656 (decoder + compressor LoRA + merger) |
| A2 full s42 (189842, homer-019), steps 1-75 | ~89 | ~10.8 h projected | — (running) | 286,790,656 |

## Proposed text for the owner's chats (gate 0 is decided)
- docs/training_plan.md, gate 0 of "Standing proposal": "FAILED (campaign freeze, 3 seeds, n = 1,500 per set, paired judge). A decoder-only finetune on frozen `<MEM>`s loses 2.8 / 3.5 / 4.1 pt on documents and 2.3 / 2.7 / 2.5 pt on MRQA text (CI below 0 at every seed), 2.3-2.8 pt on HotpotQA; photos tie. Training is ~25% faster but peaks at 50 vs 34 GB GPU memory. Split test running (compressor frozen only / connector frozen only)."
- HANDOFF.md thread 7: "Freezing compressor + connector at finetune fails gate 0 (documents −3.5 pt mean of 3 seeds, MRQA −2.5, photos ~0). `PISCOConfig.freeze_part` (compressor | connector) added by the freeze campaign for the split test; diff in campaign_freeze/patches/split_freeze.diff."

## Notes this wake
- Woken by the wait condition: A1 seed 43 (189445) left the queue — COMPLETED, exit 0, 1.0 epoch; its module check 189450 ran and passed (verdict closed). No ✗ of this campaign in the brief.
- Verifier findings of the 15:30 STATE.md: all six were `provisional` items it confirmed as provisional (no unsupported claim). Kept provisional; the A1 s43 loss line is rewritten with the two new probes.
- Launched 6 (= cap): A2 seed 43 full run 189899 + its 4 evals 189900-189903 + module check 189904, all `--force` (the empty-factor dead-arm false match, known_issues). gpucap at 17:03: "yes — 2 free node(s)", owner 5 of 8 on `gpu`. The run pends Priority (est. 11-14 04:45).
- Found that the A1 s42 preview had finished before its requeue; recorded its number (probe on 189434, ledger note). New known_issues row for this case.
- Gripes filed: launch env vars (PX/CML/TOPK) are not in the run record; a finished-then-requeued eval still shows PENDING with no result.
- Code md5 unchanged at 17:05: `eval_mm_docqa.py` dfe32aa0, `eval_mm_4b.sh` 50abc167, `judge_mm_docqa.py` 02bc8748, `pisco/model.py` 994cff3d, `ft_freeze.sh` 668268e3.

## Patches this wake
- None. `pisco/model.py` + `test_freeze.py` from wake 11 are still uncommitted (this loop cannot commit `pisco/`); their diff is `campaign_freeze/patches/split_freeze.diff`, md5 d4412015df7ecd2cfd8408693ec3d5ef.

## Blockers / needs the owner
- None blocking. An owner's chat should commit `pisco/model.py` and `test_freeze.py` (the split-freeze setting): the running and queued A2 jobs use them from the working tree.
- Optional, owner only: the 8 A1 evals wait until ~11-15 on `gpu-be`; letting evals also use `gpu` (owner's cap, ~30 min each) would bring the A1 verdict ~1-2 days earlier. The law currently says evals go to `gpu-be`.
