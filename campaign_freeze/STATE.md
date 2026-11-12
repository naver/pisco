# STATE — campaign freeze   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-12 23:30 tick (wake 11, woken by the owner's REQUEST)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- **Gate 0 FAILED on three seeds: the question-answering finetune needs the compressor side to train.** With the compressor and connector frozen (only the decoder LoRA trains), documents lose 2.8 / 3.5 / 4.1 pt of judge accuracy at seeds 42 / 43 / 44, each with a 95% CI wholly below 0; the law's "Failed" rule (≤ −3 pt with the CI below 0 at 2 of 3 seeds on one gating set) is met by seeds 43 and 44. Paired per question, n = 1,500 per set, same eval code (md5 `eval_mm_docqa.py` dfe32aa0, `eval_mm_4b.sh` 50abc167, `judge_mm_docqa.py` 02bc8748), jobs 188650 (seeds 42 / 43) and 188701 (seed 44); re-derived this wake from the `ALL` rows of both outputs (exit 0):
  - documents: −.028 [−.049, −.007], −.035 [−.057, −.014], −.041 [−.062, −.021] → FAILED;
  - MRQA text: −.023 [−.040, −.008], −.027 [−.043, −.009], −.025 [−.041, −.009] → below the −2 pt bar at all three seeds, CI below 0, not ≤ −3 pt;
  - photos: −.009 [−.026, +.008], −.013 [−.030, +.004], +.005 [−.013, +.021] → no difference shown at any seed;
  - HotpotQA (reported, not gating): −.028 [−.045, −.013], −.023 [−.041, −.006], −.026 [−.043, −.010].
  - The prediction in CAMPAIGN.md "Mission" (the reference finetune barely moves the connector, so freezing may tie) is refuted and marked there.
- **The seed-44 pair is clean:** both finetunes ran from the same stage-2 model with the same recipe file (`ft_freeze.sh frozen|unfrozen 44`); all 8 evals ran on the same node (homer-016) with the eval-code md5 above; the paired test exited 0 (job 188701).
- **The freeze held over all three full runs** (1.0 epoch, 28,032 rows each): every compressor and connector row, norm gain included, changed 0.0%; decoder lora_B 66.7% / 66.3% / 66.7% of entries changed (seeds 42 / 43 / 44, jobs 188584 / 188585 / 188699). The seed-44 reference run, checked the same way, moved compressor lora_B 74.1% and the connector MLP 20.7% (job 188700): the check sees movement when there is some.
- **In this bf16 recipe the connector norm gain never moves, even when trainable**: 0.0% changed after 20 steps (job 188579) and after a whole epoch (seed-44 reference, job 188700), value .46094 both times. So the frozen-vs-reference difference is compressor LoRA + merger + connector MLP only.
- **Feasibility gate PASS** (20-step smokes, 1,280 rows = .046 epoch): frozen side 0.0% changed (188578); the check sees movement in the unfrozen control (compressor LoRA 25.4% / 76.4%, connector 25.4%, 188579); train loss within .05 of the unfrozen smoke at steps 10 and 20 (.594 vs .639, .584 vs .580, jobs 188528 / 188534); config diff vs 184778 = `max_steps` + `freeze_compressor` only.
- **The split-freeze setting passes its unit tests on a staged copy** (owner REQUEST 2026-11-12 23:14): `PISCOConfig.freeze_part = compressor | connector` (default unset) in `pisco/model.py`, 5 new `test_freeze.py` cases. Job 189060 imported `pisco.model` from `expQ/freeze/split_stage/` and passed `test_positions.py` 12/12, `test_mm_collators.py` 33/33, `test_freeze.py` 10/10 (the 5 old tests unchanged + 5 new), exit 0. The same two files are installed in `pisco/` (byte-identical to the staged ones; one space fixed in a new test after the staged run) and pass again in place: job 189063 imported `pisco/model.py` (md5 994cff3d) and passed 12/12, 33/33, 10/10, exit 0.
- provisional: the loss is spread over the data sources, not one type: 39 of 42 per-source differences are negative over the three seeds (28 of 28 at seeds 42 / 43, job 188650; 11 of 14 at seed 44, job 188701 — the 3 positive ones are photos realworldqa +.010, vizwiz +.020 and hotpot_comparison +.013). Most per-source n = 250-500 (hotpot_bridge n = 1,197).
- provisional: frozen training is about 25% cheaper in time and needs more GPU memory. Seed-44 pair, same minute, two A100 nodes: frozen 71.5 s/step vs reference 95.1 s/step (train_runtime 31,250 vs 41,540 s / 437 steps; jobs 188651 / 188652); peak GPU memory 50,020 vs 34,044 MB (`sacct -o TRESUsageInMax`, gres/gpumem).
- provisional: the frozen runs end at a HIGHER eval loss than the reference of the same seed (64 eval rows): .435 vs .414, .413 vs .375, .427 vs .401 (seeds 42 / 43 / 44).

**Open, and running**
- **Split test (owner's choice, option A of the gate-0 question): A1 first, then A2, 2 seeds each, same gates as gate 0** (written in CAMPAIGN.md "Split test gates" before any cell).
  - A1 = compressor frozen (adapter, `<MEM>` row, merger), connector + decoder LoRA train. If it ties the reference, stage B stays open with the compressor output cached before the connector.
  - A2 = connector frozen, compressor LoRA + merger + decoder LoRA train. Diagnostic: says where the 3 pt sit.
- Queued now: both 20-step feasibility smokes (A1 189064 on `gpu`, A2 189065 on `gpu-be`), each followed by its module-movement check (189066 / 189067). Both smokes pend for CPUs, not GPUs: the nodes with a free A100 have 1-2 free CPUs and the recipe asks 8 (scheduler estimate: A1 ~11-13 07:00; A2 on `gpu-be` later). Prediction, not a result: A1 fails like F, A2 ties R (the reference moved compressor lora_B 74% of entries, the connector 20.7%).

## Owned jobs (jobid | arm | state | rows seen | eta | note)
| job | what | state at 23:45 | rows | eta | note |
|---|---|---|---|---|---|
| 189063 | unit tests on the installed tree | COMPLETED 4:46, exit 0 (homer-009) | — | done | closed; 4:46 of a 5:00 limit: next test run --time 0:07:00 |
| 189064 | A1 smoke `ft_freeze.sh compfrozen 42 20` | PENDING (Resources: CPUs) | 1,280 (.046 epoch) | start est. ~07:00 | `gpu`, --time 0:45 |
| 189065 | A2 smoke `ft_freeze.sh connfrozen 42 20` | PENDING (Priority) | 1,280 (.046 epoch) | later than A1 | `gpu-be` (preemptible), --time 0:45 |
| 189066 | module move, A1 smoke vs 184777 | PENDING (afterok 189064) | — | — | --time 0:02 |
| 189067 | module move, A2 smoke vs 184777 | PENDING (afterok 189065) | — | — | --time 0:02 |
GPU budget: ~52 GPU-h used of 192 before this wake; the split test adds ~4 × 11 h finetunes + 16 evals ≈ 55 GPU-h (+ ~22 on a tie).

## Adopted jobs
(none — the other jobs in the queue belong to the sibling campaigns `longchunk` and `probes`)

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| F frozen vs R reference, seeds 42 / 43 / 44 | decided: FAILED (documents, 2 of 3 seeds) | — |
| A1 compressor frozen | feasibility smoke queued | next wake: read 189064 + 189066 against the A1 PASS rule (log line, `Compressor trainable parameters: 0`, compressor rows 0.0%, connector.linear > 0%, loss within .05 of .639 / .580, config diff = `freeze_compressor` → `freeze_part` only); PASS → full cells seeds 42 and 43 (`--time` = smoke s/step × 438 × 1.25), each with its 4 evals and module check chained |
| A2 connector frozen | feasibility smoke queued | same, with the A2 PASS rule; full cells after A1's full cells are submitted, within CAMP_MAX_TRAIN = 2 and the 8-GPU `gpu` cap |
| after-gate arms for a VALIDATED gate | void: gate 0 failed | — |

## Deliverable — judge accuracy ↑ (Qwen3-8B rubric judge), n = 1,500 per set; Δ = paired difference arm − reference of the same seed [95% CI] (jobs 188650, 188701)
| model | what trains | rate (tokens per slot; slots per page) | documents ↑ s42 / s43 / s44 | photos ↑ s42 / s43 / s44 | MRQA text ↑ s42 / s43 / s44 | HotpotQA ↑ s42 / s43 / s44 |
|---|---|---|---|---|---|---|
| R reference (184778 / 184780 re-evaluated 11-11; 188652) | compressor LoRA + merger + connector + decoder LoRA | 8; 123 per 1M-px page (980 visual tokens) | .610 / .615 / .620 | .673 / .685 / .671 | .718 / .724 / .721 | .603 / .593 / .603 |
| F frozen (188582 / 188583 / 188651) | decoder LoRA only | same | .582 / .580 / .579 | .664 / .672 / .675 | .695 / .697 / .697 | .575 / .571 / .577 |
| A1 compressor frozen | connector + decoder LoRA | same | queued | | | |
| A2 connector frozen | compressor LoRA + merger + decoder LoRA | same | queued | | | |
| uncompressed decoder (earlier runs, same questions) | — | 1 (raw tokens) | .809 | .717 | .807 | — |

| paired Δ frozen F − reference ↑ (CI) | seed 42 | seed 43 | seed 44 | mean of the 3 Δ |
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

## Training cost (A100-80GB) — provisional (one same-time pair, two nodes)
| run | s/step | wall | peak GPU memory (sacct gres/gpumem) | trainable params |
|---|---|---|---|---|
| F frozen s42 / s43 / s44 (188582 / 188583 / 188651) | 71.5 / 76.1 / 71.5 | 8.7 / 9.3 / 8.7 h | 49.9 / 49.9 / 50.0 GB | 155,978,240 (decoder only) |
| R reference s42 / s43 / s44 (184778 / 184780 / 188652) | 90.3 / 113.2 / 95.1 | 11.0 / 13.8 / 11.6 h | — / — / 34.0 GB | 307,771,392 |
Open question: why the frozen run uses MORE GPU memory. A guess, not checked: gradient checkpointing only runs in train mode and the frozen compressor is in eval mode, so its activations are kept.

## Proposed text for the owner's chats (gate 0 is decided)
- docs/training_plan.md, gate 0 of "Standing proposal": "FAILED (campaign freeze, 3 seeds, n = 1,500 per set, paired judge). A decoder-only finetune on frozen `<MEM>`s loses 2.8 / 3.5 / 4.1 pt on documents and 2.3 / 2.7 / 2.5 pt on MRQA text (CI below 0 at every seed), 2.3-2.8 pt on HotpotQA; photos tie. Training is ~25% faster but peaks at 50 vs 34 GB GPU memory. Split test running (compressor frozen only / connector frozen only)."
- HANDOFF.md thread 7: "Freezing compressor + connector at finetune fails gate 0 (documents −3.5 pt mean of 3 seeds, MRQA −2.5, photos ~0). `PISCOConfig.freeze_part` (compressor | connector) added by the freeze campaign for the split test; diff in campaign_freeze/patches/split_freeze.diff."

## Notes this wake
- **Owner REQUEST executed (2026-11-12 23:14, option A, split test):** (1) the one-time exception written into CAMPAIGN.md "## Patchable", dated, before any edit; (2) the patch written on a staged copy (`expQ/freeze/split_stage/`, gitignored), tested there (189060), then installed in `pisco/` by an atomic rename after checking the originals' md5 had not moved (model.py e48f9945, test_freeze.py 0198640e = HEAD); (3) both arms' gates written into CAMPAIGN.md "Split test gates" before the first launch; (4) two new `ARM)` cases in `expQ/freeze/ft_freeze.sh` (`compfrozen`, `connfrozen`; `bash -n` OK; the existing arms' lines unchanged). REQUEST and BLOCKED.md deleted (the question is answered).
- **Not weakened:** `freeze_compressor=True` takes the same branch with the same log line and freezes both parts; with `freeze_part` unset, `frozen_parts()` returns `()` so `PISCO.__init__` and `PISCO.train` do nothing new (old checkpoints load with the default `None`). Setting both raises `ValueError`, as does an unknown part. The other chats' uncommitted files (HANDOFF.md, eval_mm_4b.sh, scripts/eval_mm_docqa.py) were not touched.
- New campaign tool: `expQ/freeze/run_tests_freeze.sh [DIR]` runs the 3 CPU test files against a tree (default the repo); `--time 0:05:00` = longest earlier test run 3:36 × 1.25.
- Smokes `--time 0:45:00` = unfrozen smoke 188534 35:58 × 1.25. Module checks `--time 0:02:00` = 53 s (188699) × 1.25.
- A2 smoke on `gpu-be`: the owner's `gpu` cap is 8 GPUs and 6 are in use (longchunk 4 + 1 + 1); the A1 smoke takes one, the last stays free.
- Slip: one empty `python3` heredoc ran on the login node while backing up a file (did nothing). Not repeated.

## Patches this wake
- `pisco/model.py` — owner REQUEST (not a crash) — +33 / −7 lines: `PISCOConfig.freeze_part`, `FREEZE_PARTS`, `frozen_parts(config)`, `freeze_compressor_side(model, parts=FREEZE_PARTS)`, one `elif` in `PISCO.__init__`, `PISCO.train` loops over `frozen_parts`.
- `test_freeze.py` — owner REQUEST — +65 lines: class `TestFreezePart` (5 tests) + helpers; the 5 old tests unchanged.
- Saved as `campaign_freeze/patches/split_freeze.diff` (`git diff pisco/model.py test_freeze.py` vs HEAD c36daf1), **md5 d4412015df7ecd2cfd8408693ec3d5ef**. Installed files: `pisco/model.py` md5 994cff3d483cf85645f7f78c2eb85d4a, `test_freeze.py` md5 69b9c6fc2a54e50ee6ccef54cbb685ff. The wake cannot commit (CAMP_WAKE_GIT=0) and `pisco/` is outside CAMP_COMMIT_PATHS: an owner's chat should commit these two files.
- `expQ/freeze/ft_freeze.sh` — 2 new `ARM)` cases + 2 comment lines (Patchable: new arm, gate written first). `expQ/freeze/run_tests_freeze.sh` — new.

## Blockers / needs the owner
- None blocking. The patched `pisco/model.py` and `test_freeze.py` are uncommitted (this loop cannot commit them); `git diff` = `campaign_freeze/patches/split_freeze.diff`.
