# STATE — campaign freeze   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-13 10:30 tick (wake 17; woken because A1 seed 42 (189094) left the queue, the `wake-when` of wake 16)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- **Gate 0 FAILED on three seeds: the question-answering finetune needs the compressor side to train.** With the compressor and connector frozen (only the decoder LoRA trains), documents lose 2.8 / 3.5 / 4.1 pt of judge accuracy at seeds 42 / 43 / 44, each with a 95% CI wholly below 0; the law's "Failed" rule (≤ −3 pt with the CI below 0 at 2 of 3 seeds on one gating set) is met by seeds 43 and 44. Paired per question, n = 1,500 per set, same eval code (md5 `eval_mm_docqa.py` dfe32aa0, `eval_mm_4b.sh` 50abc167, `judge_mm_docqa.py` 02bc8748), jobs 188650 (seeds 42 / 43) and 188701 (seed 44), `ALL` rows:
  - documents: −.028 [−.049, −.007], −.035 [−.057, −.014], −.041 [−.062, −.021] → FAILED;
  - MRQA text: −.023 [−.040, −.008], −.027 [−.043, −.009], −.025 [−.041, −.009] → below the −2 pt bar at all three seeds, CI below 0, not ≤ −3 pt;
  - photos: −.009 [−.026, +.008], −.013 [−.030, +.004], +.005 [−.013, +.021] → no difference shown at any seed;
  - HotpotQA (reported, not gating): −.028 [−.045, −.013], −.023 [−.041, −.006], −.026 [−.043, −.010].
  - The prediction in CAMPAIGN.md "Mission" (the reference finetune barely moves the connector, so freezing may tie) is refuted and marked there.
- **The seed-44 pair is clean:** both finetunes ran from the same stage-2 model with the same recipe file (`ft_freeze.sh frozen|unfrozen 44`); all 8 evals used the eval-code md5 above; the paired test exited 0 (job 188701).
- **The freeze held over all three full F runs** (1.0 epoch, 28,032 rows each): every compressor and connector row, norm gain included, changed 0.0%; decoder lora_B 66.7% / 66.3% / 66.7% of entries changed (seeds 42 / 43 / 44, jobs 188584 / 188585 / 188699). The seed-44 reference run, checked the same way, moved compressor lora_B 74.1% and the connector MLP 20.7% (job 188700): the check sees movement when there is some. It cannot see the `<MEM>` row and has no row for the merger or the vision tower.
- **In this bf16 recipe the connector norm gain never moves, even when trainable**: 0.0% changed after 20 steps (jobs 188579, 189066) and after a whole epoch (seed-44 reference 188700, A1 seed 42 189099), value .46094 each time.
  - provisional: so the frozen-vs-reference difference is compressor LoRA + connector MLP, plus possibly the merger and the compressor `<MEM>` row — neither is measured: the merger has no row in the check, and the `<MEM>` row sits inside a 1,270M-entry token-adapter group where one changed 2,560-entry row rounds to 0.0% (no positive control for either).
- **Split-freeze setting installed and tested**: `PISCOConfig.freeze_part = compressor | connector` (default unset) in `pisco/model.py` (md5 994cff3d, unchanged at 10:30), 5 new `test_freeze.py` cases; staged copy (job 189060) and installed tree (job 189063) both pass `test_positions.py` 12/12, `test_mm_collators.py` 33/33, `test_freeze.py` 10/10, exit 0.
- **Split arm A1 (compressor frozen) feasibility PASS** (20-step smoke, seed 42, 1,280 rows = .046 epoch; jobs 189064 + 189066): log `froze 501 compressor tensors` + `Compressor trainable parameters: 0`; compressor rows 0.0%, connector MLP 28.7%, decoder lora_B 76.2%; train loss .6033 / .5952 at steps 10 / 20 vs the unfrozen smoke's .639 / .580 (188534), gaps .036 / .015; config diff one line (`freeze_compressor: true` → `freeze_part: compressor`); 81.2 s/step, 49.8 GB peak.
- **A1 seed 42 full run COMPLETED and its freeze held** (job 189094, homer-013, exit 0, 437 of 437 steps = 1.0 epoch, 28,032 rows, 8.7 h, 71.5 s/step). After-run module check (job 189099, exit 0): compressor lora_A / lora_B / token-adapter 0.0%; connector MLP 19.1% changed (.0061 of its norm); norm gain 0.0%; decoder lora_B 66.0%.
- provisional: the gate-0 loss is spread over the data sources: 39 of 42 per-source differences are negative over the three seeds (28 of 28 at seeds 42 / 43, job 188650; 11 of 14 at seed 44, job 188701 — the 3 positive ones are photos realworldqa +.010, vizwiz +.020 and hotpot_comparison +.013). A sign count, not a test: most per-source CIs span 0 (per-source n = 250-500, hotpot_bridge 1,197).
- provisional: fully frozen training is about 25% cheaper in time and needs more GPU memory. Seed-44 pair: frozen 71.5 s/step vs reference 95.1 (grid probes of 188651 / 188652, two different A100 nodes, so node speed is mixed in); peak GPU memory 50,020 vs 34,044 MB (from `sacct` gres/gpumem, not in the grid).
- provisional: the fully frozen runs end at a HIGHER eval loss than the reference of the same seed (64 eval rows, no paired test): .435 vs .414, .413 vs .375, .427 vs .401 (seeds 42 / 43 / 44; the .414 / .375 of 184778 / 184780 are not campaign cells).

**Open, and running**
- **Split test (owner's choice 2026-11-12): A1 first, then A2, seeds 42 and 43, same gates as gate 0.**
  - A1 = compressor frozen (adapter, `<MEM>` row, merger); connector + decoder LoRA train.
    - Seed 42 trained (above). provisional: its held-out loss (64 rows) ends where the fully frozen arm F ends, not where the reference ends: .459 / .449 / .437 / .435 / .435 at .23 / .46 / .69 / .92 / 1.0 epoch; F s42 (188582) .455 / .444 / .438 / .434 / .435; reference 184778 .513 / .455 / .393 / .394 / .414. Consistent with the prediction "A1 fails like F"; a 64-row loss, not a judge number, decides nothing.
    - Seed 42 judged evals (documents / photos / MRQA / HotpotQA, jobs 189095-189098) and the step-100 preview (first 300 documents, 189434): PENDING on `gpu-be` for Priority, scheduler start estimates 11-16 03:45-05:15. Another user's 1-GPU `gpu-be` jobs (higher priority) sit ahead; the 4 free A100s are on nodes with no free CPU (homer-006 / 007 / 019: 27 of 28 CPUs allocated).
    - Seed 43 RUNNING since 05:28 on homer-020 (`gpu`, job 189445): at 10:28 step 206 of 437 = .47 epoch (13,184 rows), 93.4 s/step, no NaN; held-out loss .464 / .449 at .23 / .46 epoch vs F s43 (188583) .442 / .431 and reference 184780 .471 / .439. 231 steps left × 93.4 s ≈ 6.0 h → ends about 16:30, time limit 17:49 (margin ~1.3 h). Its 4 judged evals (189446-189449) and module check (189450) are chained `afterok`.
  - A2 = connector frozen; compressor LoRA + merger + decoder LoRA train.
    - 20-step smoke re-launched at 10:35 (job 189739, same command as the cancelled 189065, `gpu` A100, --time 0:45:00) with its module check chained `afterok` (189740, `gpu-be` CPU, 0:02:00). PENDING Resources, scheduler estimate 14:49 (same CPU shortage as above). The old dir of 189065 (log, no checkpoint) was renamed `expQ/freeze/VOID_ft_connfrozen_s42_smoke_189065` so the re-run starts clean.
    - provisional: the preempted smoke 189065's log already showed `[pisco] freeze_part=connector: froze 5 connector tensors (eval mode kept)` and `Compressor trainable parameters: 130812416` (> 0), but no logged loss step; the PASS rule needs the full 20 steps + module check.
  - Prediction, not a result (written before any A1 / A2 judge number): A1 fails like F, A2 ties R.

## Owned jobs (jobid | arm | state | rows seen | eta | note)
| job | what | state at 10:35 | rows | eta | note |
|---|---|---|---|---|---|
| 189094 | A1 full `ft_freeze.sh compfrozen 42` | COMPLETED 10:17, exit 0 | 28,032 (1.0 epoch) | — | model `expQ/freeze/ft_compfrozen_s42/model` |
| 189099 | module move, A1 s42 full vs 184777 | COMPLETED, exit 0, verdict closed | — | — | freeze held (numbers above) |
| 189434 | A1 s42 step-100 PREVIEW, first 300 documents | PENDING Priority (Restarts 1) | 300 Q | est. 11-16 03:45 | on `expQ/freeze/ft_compfrozen_s42_step100/checkpoint-100`; --time 0:40:00 |
| 189095-189098 | A1 s42 evals: documents / photos / MRQA (CML=4096) / HotpotQA (CML=4096 TOPK=10, not gating) | PENDING Priority | 1,500 Q each | est. 11-16 04:15-05:15 | --time 0:40 / 0:25 / 0:22 / 0:29 |
| 189445 | A1 full `ft_freeze.sh compfrozen 43` | RUNNING (homer-020, `gpu`) | 13,184 of 28,032 (.47 epoch) | ~16:30 (limit 17:49) | kill on NaN or 2 held-out probes above .471 (184780 at step 100) |
| 189446-189449 | A1 s43 evals, same env and --time as seed 42 | PENDING afterok 189445 | 1,500 Q each | after the run | pair with ref_s43 files 188599 / 188604 / 188613 / 188617 |
| 189450 | module move, A1 s43 full vs 184777 | PENDING afterok 189445 | — | | --time 0:02:00 |
| 189739 | A2 smoke `ft_freeze.sh connfrozen 42 20` | PENDING Resources | 1,280 rows (.046 epoch) when done | est. 14:49 start, ~30-40 min | `gpu` A100, --time 0:45:00; gpucap before launch: owner 5 of 8 `gpu` GPUs, 6 after |
| 189740 | module move, A2 smoke vs 184777 | PENDING afterok 189739 | — | | `gpu-be` CPU, --time 0:02:00 |
All evals: `gpu-be`, `--exclude=homer-[005-009]`, PX=1003520. Eval code md5 re-checked at 10:30: `eval_mm_docqa.py` dfe32aa0, `eval_mm_4b.sh` 50abc167, `judge_mm_docqa.py` 02bc8748 = gate 0, so the judged reference files of 184778 / 184780 still pair as they are. Re-check before reading the A1 evals.
Trainings in flight: 189445 + 189739 = 2 = CAMP_MAX_TRAIN.
GPU budget: ~73 GPU-h used of 192 (gate 0 ~52 + A1 smoke .5 + A1 s42 8.7 + A1 s43 5.0 so far + checks); left in the plan: A1 s43 ~6 h, A2 smoke ~.6 h, A2 2 × ~11 h, 16 evals ≈ 8 h → ~37 GPU-h (+ ~22 on a tie).

## Adopted jobs
(none — the other jobs in the queue belong to the sibling campaigns `longchunk` and `probes`)

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| F frozen vs R reference, seeds 42 / 43 / 44 | decided: FAILED (documents, 2 of 3 seeds) | — |
| A1 compressor frozen | s42 trained + freeze checked, evals + preview queued (est. 11-16); s43 running (.47 epoch) + chain queued | (1) s43 held-out probes at .69 / .92 epoch vs 184780 .411 / .441, F s43 .416 / .415; (2) preview 189434 vs arm F s42's preview (188594: .583) and 184778's final (.610) on the same 300 Q, `provisional:`, n = 300, never gates; (3) after the s43 run: module check 189450 must show compressor.* 0.0%; (4) when both seeds are judged: `paired_judge_cpu.sh` A1 vs ref_s42 / ref_s43, then the three gate rules |
| A2 connector frozen | smoke + check queued (189739 / 189740) | read against the A2 PASS rule (log lines; connector.linear + norm_gain 0.0%; compressor lora_B > 0%; decoder lora_B > 0%; loss at steps 10 / 20 within .05 of .639 / .580; config diff vs `ft_frozen_s42_smoke` only `freeze_compressor` → `freeze_part` + paths; record s/step + peak memory). PASS → full s42 `ft_freeze.sh connfrozen 42` on `gpu` A100, --time = smoke s/step × 438 × 1.25, + 4 evals + check `afterok` (as A1); s43 when A1 s43 has ended (training cap). FAIL → BLOCKED.md |
| after-gate arms for a VALIDATED gate | void: gate 0 failed | — |

## Next
- The tick wakes this loop when the A2 smoke and its check (189739, 189740) have left the queue, or at 8 h. A1 s43 (189445) ends ~16:30; if it or anything chained fails, the new failure wakes the loop.
- At that wake: read the A2 smoke against its PASS rule; on PASS launch A2 full s42 + its chain (A1 s43 still holds the other slot until ~16:30), and A2 s43 at the wake after A1 s43 ends.

## Deliverable — judge accuracy ↑ (Qwen3-8B rubric judge), n = 1,500 per set; Δ = paired difference arm − reference of the same seed [95% CI] (jobs 188650, 188701)
| model | what trains | rate (tokens per slot; slots per page) | documents ↑ s42 / s43 / s44 | photos ↑ s42 / s43 / s44 | MRQA text ↑ s42 / s43 / s44 | HotpotQA ↑ s42 / s43 / s44 |
|---|---|---|---|---|---|---|
| R reference (184778 / 184780 re-evaluated 11-11; 188652) | compressor LoRA + merger + connector + decoder LoRA | 8; 123 per 1M-px page (980 visual tokens) | .610 / .615 / .620 | .673 / .685 / .671 | .718 / .724 / .721 | .603 / .593 / .603 |
| F frozen (188582 / 188583 / 188651) | decoder LoRA only | same | .582 / .580 / .579 | .664 / .672 / .675 | .695 / .697 / .697 | .575 / .571 / .577 |
| A1 compressor frozen (189094 / 189445) | connector + decoder LoRA | same | s42 trained, evals queued (est. 11-16) / s43 training (.47 epoch) | | | |
| A2 connector frozen | compressor LoRA + merger + decoder LoRA | same | smoke queued (189739) | | | |
| uncompressed decoder (earlier runs, same questions) | — | 1 (raw tokens) | .809 | .717 | .807 | — |

| paired Δ frozen F − reference ↑ (CI) | seed 42 | seed 43 | seed 44 | mean of the 3 Δ |
|---|---|---|---|---|
| documents | −.028 [−.049, −.007] | −.035 [−.057, −.014] | −.041 [−.062, −.021] | −.035 |
| photos | −.009 [−.026, +.008] | −.013 [−.030, +.004] | +.005 [−.013, +.021] | −.006 |
| MRQA text | −.023 [−.040, −.008] | −.027 [−.043, −.009] | −.025 [−.041, −.009] | −.025 |
| HotpotQA | −.028 [−.045, −.013] | −.023 [−.041, −.006] | −.026 [−.043, −.010] | −.026 |
Discordant pairs (frozen-only right / reference-only right), McNemar p — seeds 42 / 43 / 44: documents 111/153 p .011, 109/161 p .002, 103/164 p < .001; photos 78/92 p .32, 78/97 p .17, 90/83 p .65; MRQA 59/94 p .006, 67/107 p .003, 58/95 p .003; HotpotQA 64/106 p .002, 68/102 p .011, 64/103 p .003.

Held-out loss ↓ (64 rows, no paired test; a health signal, not a gate), at .23 / .46 / .69 / .92 / 1.0 epoch:
| run | seed 42 | seed 43 |
|---|---|---|
| R reference (184778 / 184780 trainer_state) | .513 / .455 / .393 / .394 / .414 | .471 / .439 / .411 / .441 / .375 |
| F frozen (188582 / 188583 logs) | .455 / .444 / .438 / .434 / .435 | .442 / .431 / .416 / .415 / .413 |
| A1 compressor frozen (189094 log / 189445 probes) | .459 / .449 / .437 / .435 / .435 | .464 / .449 / — (running) |

## Module movement (entries changed %, per module, vs the stage-2 model 184777)
| module | F smoke s42 (188578) | unfrozen smoke s42 (188579) | A1 smoke s42 (189066) | F full s42 (188584) | F full s43 (188585) | F full s44 (188699) | R full s44 (188700) | A1 full s42 (189099) |
|---|---|---|---|---|---|---|---|---|
| decoder lora_A / lora_B | 30.6 / 76.3 | 26.7 / 75.8 | 30.6 / 76.2 | 14.7 / 66.7 | 14.7 / 66.3 | 14.8 / 66.7 | 15.1 / 67.8 | 14.6 / 66.0 |
| compressor lora_A / lora_B | 0.0 / 0.0 | 25.4 / 76.4 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 16.9 / 74.1 | 0.0 / 0.0 |
| compressor token-adapter group (holds the `<MEM>` row; 1,270M entries, so one row is invisible) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| connector MLP | 0.0 | 25.4 | 28.7 (.0027 of its norm) | 0.0 | 0.0 | 0.0 | 20.7 (.0073 of its norm) | 19.1 (.0061 of its norm) |
| connector norm gain | 0.0 | 0.0 (bf16 lost update) | 0.0 (bf16 lost update) | 0.0 | 0.0 | 0.0 | 0.0 (bf16 lost update) | 0.0 (bf16 lost update) |
| merger / vision tower | not measured (no row in the check) | | | | | | | |

## Training cost (A100-80GB) — provisional (one same-time pair, two nodes)
| run | s/step | wall | peak GPU memory (sacct gres/gpumem) | trainable params |
|---|---|---|---|---|
| F frozen s42 / s43 / s44 (188582 / 188583 / 188651) | 71.5 / 76.1 / 71.5 | 8.7 / 9.3 / 8.7 h | 49.9 / 49.9 / 50.0 GB | 155,978,240 (decoder only) |
| R reference s42 / s43 / s44 (184778 / 184780 / 188652) | 90.3 / 113.2 / 95.1 | 11.0 / 13.8 / 11.6 h | — / — / 34.0 GB | 307,771,392 |
| A1 smoke s42 (189064), 20 steps | 81.2 overall (~70.5 steady) | 29 min | 49.8 GB | 176,958,976 (decoder + connector) |
| A1 full s42 (189094, homer-013) | 71.5 | 8.7 h | — (not read yet) | 176,958,976 |
| A1 full s43 (189445, homer-020), steps 1-206 | 93.4 now (87.6 average) | ~11.0 h projected | — (running) | 176,958,976 |
Open question: why a frozen compressor uses MORE GPU memory. A guess, not checked: gradient checkpointing only runs in train mode and the frozen compressor is in eval mode, so its activations are kept. A1 (compressor frozen, connector training) peaks the same as F, consistent with that guess.

## Proposed text for the owner's chats (gate 0 is decided)
- docs/training_plan.md, gate 0 of "Standing proposal": "FAILED (campaign freeze, 3 seeds, n = 1,500 per set, paired judge). A decoder-only finetune on frozen `<MEM>`s loses 2.8 / 3.5 / 4.1 pt on documents and 2.3 / 2.7 / 2.5 pt on MRQA text (CI below 0 at every seed), 2.3-2.8 pt on HotpotQA; photos tie. Training is ~25% faster but peaks at 50 vs 34 GB GPU memory. Split test running (compressor frozen only / connector frozen only)."
- HANDOFF.md thread 7: "Freezing compressor + connector at finetune fails gate 0 (documents −3.5 pt mean of 3 seeds, MRQA −2.5, photos ~0). `PISCOConfig.freeze_part` (compressor | connector) added by the freeze campaign for the split test; diff in campaign_freeze/patches/split_freeze.diff."

## Notes this wake
- No ✗ of this campaign in the brief; the wake condition (189094 left the queue) was met: COMPLETED, exit 0, 1.0 epoch.
- Probes recorded: 189094 at .92 / 1.0 epoch (held-out .4346 / .4349, 71.5 s/step); 189445 at .46 epoch (.4485, 93.4 s/step).
- Verdicts: 189099 closed (A1 s42 freeze held); 189065 and 189067 closed (cancelled A2 smoke + its never-run check, replaced by 189739 / 189740).
- Launched 2 (cap 6): A2 smoke 189739 + check 189740. Dry-run first; gpucap: yes, 3 free A100 nodes, owner 5 of 8 `gpu` GPUs.
- The A1 evals wait ~3 days on `gpu-be` (Priority). The law sends evals to `gpu-be` only; this loop may not widen them to `gpu` (that would relax a rule). No question to the owner: the delay slows the A1 verdict but blocks nothing; A2 trains meanwhile.
- Eval and model code md5 unchanged at 10:30 (`eval_mm_docqa.py` dfe32aa0, `eval_mm_4b.sh` 50abc167, `judge_mm_docqa.py` 02bc8748, `pisco/model.py` 994cff3d, `ft_freeze.sh` 668268e3, patch diff d4412015).
- Verifier: every row it checked was already marked provisional; none needed fixing.

## Patches this wake
- None. `pisco/model.py` + `test_freeze.py` from wake 11 are still uncommitted (this loop cannot commit `pisco/`); their diff is `campaign_freeze/patches/split_freeze.diff`, md5 d4412015df7ecd2cfd8408693ec3d5ef.

## Blockers / needs the owner
- None blocking. An owner's chat should commit `pisco/model.py` and `test_freeze.py` (the split-freeze setting) — the running A1 seed-43 job and the A2 smoke use them from the working tree.
- Optional, owner only: the 4 A1 seed-42 evals + preview wait until ~11-16 on `gpu-be`; letting evals also use `gpu` (owner's cap, ~30 min each) would bring the A1 verdict ~3 days earlier. The law currently says evals go to `gpu-be`.
