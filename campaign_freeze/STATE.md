# STATE — campaign freeze   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-13 01:30 tick (wake 12, woken because the A1 smoke's module check left the queue)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- **Gate 0 FAILED on three seeds: the question-answering finetune needs the compressor side to train.** With the compressor and connector frozen (only the decoder LoRA trains), documents lose 2.8 / 3.5 / 4.1 pt of judge accuracy at seeds 42 / 43 / 44, each with a 95% CI wholly below 0; the law's "Failed" rule (≤ −3 pt with the CI below 0 at 2 of 3 seeds on one gating set) is met by seeds 43 and 44. Paired per question, n = 1,500 per set, same eval code (md5 `eval_mm_docqa.py` dfe32aa0, `eval_mm_4b.sh` 50abc167, `judge_mm_docqa.py` 02bc8748), jobs 188650 (seeds 42 / 43) and 188701 (seed 44), `ALL` rows:
  - documents: −.028 [−.049, −.007], −.035 [−.057, −.014], −.041 [−.062, −.021] → FAILED;
  - MRQA text: −.023 [−.040, −.008], −.027 [−.043, −.009], −.025 [−.041, −.009] → below the −2 pt bar at all three seeds, CI below 0, not ≤ −3 pt;
  - photos: −.009 [−.026, +.008], −.013 [−.030, +.004], +.005 [−.013, +.021] → no difference shown at any seed;
  - HotpotQA (reported, not gating): −.028 [−.045, −.013], −.023 [−.041, −.006], −.026 [−.043, −.010].
  - The prediction in CAMPAIGN.md "Mission" (the reference finetune barely moves the connector, so freezing may tie) is refuted and marked there.
- **The seed-44 pair is clean:** both finetunes ran from the same stage-2 model with the same recipe file (`ft_freeze.sh frozen|unfrozen 44`); all 8 evals ran on the same node (homer-016) with the eval-code md5 above; the paired test exited 0 (job 188701).
- **The freeze held over all three full runs** (1.0 epoch, 28,032 rows each): every compressor and connector row, norm gain included, changed 0.0%; decoder lora_B 66.7% / 66.3% / 66.7% of entries changed (seeds 42 / 43 / 44, jobs 188584 / 188585 / 188699). The seed-44 reference run, checked the same way, moved compressor lora_B 74.1% and the connector MLP 20.7% (job 188700): the check sees movement when there is some.
- **In this bf16 recipe the connector norm gain never moves, even when trainable**: 0.0% changed after 20 steps (jobs 188579, 189066) and after a whole epoch (seed-44 reference, job 188700), value .46094 each time. So the frozen-vs-reference difference is compressor LoRA + merger + connector MLP only.
- **Split-freeze setting installed and tested**: `PISCOConfig.freeze_part = compressor | connector` (default unset) in `pisco/model.py` (md5 994cff3d), 5 new `test_freeze.py` cases; staged copy (job 189060) and installed tree (job 189063) both pass `test_positions.py` 12/12, `test_mm_collators.py` 33/33, `test_freeze.py` 10/10, exit 0.
- **Split arm A1 (compressor frozen) feasibility PASS** (20-step smoke, seed 42, 1,280 rows = .046 epoch; jobs 189064 + 189066), every rule of CAMPAIGN.md "Split test gates":
  - log: `[pisco] freeze_part=compressor: froze 501 compressor tensors (eval mode kept)` and `Compressor trainable parameters: 0`; total trainable 176,958,976 = decoder 155,978,240 + connector 20,980,736;
  - module move vs the stage-2 model: compressor lora_A / lora_B / token row 0.0%; connector MLP 28.7% changed; norm gain 0.0% (bf16 lost update, as above); decoder lora_B 76.2%;
  - train loss .6033 at step 10 and .5952 at step 20 vs the unfrozen smoke's .639 / .580 (job 188534): gaps .036 and .015, both within .05;
  - `training_config.yaml` vs the frozen smoke's: one line, `freeze_compressor: true` → `freeze_part: compressor`;
  - cost: 81.2 s/step over the smoke (1,624 s / 20; first step 232 s, steady ~70.5), peak GPU memory 49,822 MB (sacct gres/gpumem) — same as the fully frozen smoke.
- provisional: the gate-0 loss is spread over the data sources: 39 of 42 per-source differences are negative over the three seeds (28 of 28 at seeds 42 / 43, job 188650; 11 of 14 at seed 44, job 188701 — the 3 positive ones are photos realworldqa +.010, vizwiz +.020 and hotpot_comparison +.013). A sign count, not a test: most per-source CIs span 0 (per-source n = 250-500, hotpot_bridge 1,197).
- provisional: fully frozen training is about 25% cheaper in time and needs more GPU memory. Seed-44 pair: frozen 71.5 s/step vs reference 95.1 (train_runtime 31,250 vs 41,540 s / 437 steps; grid probes of 188651 / 188652, two different A100 nodes); peak GPU memory 50,020 vs 34,044 MB (from `sacct` gres/gpumem, not in the grid).
- provisional: the fully frozen runs end at a HIGHER eval loss than the reference of the same seed (64 eval rows, no paired test): .435 vs .414, .413 vs .375, .427 vs .401 (seeds 42 / 43 / 44; the .414 / .375 of 184778 / 184780 are not campaign cells).

**Open, and running**
- **Split test (owner's choice 2026-11-12): A1 first, then A2, seeds 42 and 43, same gates as gate 0.**
  - A1 = compressor frozen (adapter, `<MEM>` row, merger), connector + decoder LoRA train. Seed-42 full run RUNNING since 01:33 (homer-013, `gpu`), its 4 judged evals + module check chained after it. Seed 43 waits for a training slot (see Next).
  - A2 = connector frozen, compressor LoRA + merger + decoder LoRA train. Its 20-step smoke still PENDING on `gpu-be` (reason Priority, scheduler start estimate 11-13 13:45), its module check chained.
  - Prediction, not a result (written before any A1 / A2 number): A1 fails like F, A2 ties R.

## Owned jobs (jobid | arm | state | rows seen | eta | note)
| job | what | state at 01:40 | rows | eta | note |
|---|---|---|---|---|---|
| 189094 | A1 full `ft_freeze.sh compfrozen 42` | RUNNING (homer-013, `gpu`) | 0 of 28,032 (1 epoch) | ~10-12 h (est. 11-13 12:00-13:30) | --time 12:21:00 = 81.2 s/step × 438 × 1.25; health: eval_loss every 100 steps vs 184778 .513 / .455 / .393 / .394 |
| 189095 | A1 s42 documents eval | PENDING afterok 189094 | 1,500 Q | after the run | --time 0:40:00 = 31:47 × 1.25 |
| 189096 | A1 s42 photos eval | PENDING afterok 189094 | 1,500 Q | | --time 0:25:00 = 19:36 × 1.25 |
| 189097 | A1 s42 MRQA text eval (CML=4096) | PENDING afterok 189094 | 1,500 Q | | --time 0:22:00 = 17:21 × 1.25 |
| 189098 | A1 s42 HotpotQA eval (CML=4096 TOPK=10, not gating) | PENDING afterok 189094 | 1,500 Q | | --time 0:29:00 = 22:44 × 1.25 |
| 189099 | module move, A1 s42 full vs 184777 | PENDING afterok 189094 | — | | --time 0:02:00 = 65 s × 1.25 |
| 189065 | A2 smoke `ft_freeze.sh connfrozen 42 20` | PENDING (Priority, `gpu-be`) | 1,280 (.046 epoch) | start est. 11-13 13:45 | --time 0:45:00 |
| 189067 | module move, A2 smoke vs 184777 | PENDING afterok 189065 | — | | --time 0:02:00 |
All evals: `gpu-be`, `--exclude=homer-[005-009]`, PX=1003520; eval code md5 checked this wake = gate 0 (above), so the judged reference files of 184778 / 184780 pair as they are.
GPU budget: ~54 GPU-h used of 192 (gate 0 ~52 + A1 smoke .5 + tests); the split test adds ~4 × 11 h finetunes + 16 evals ≈ 55 GPU-h (+ ~22 on a tie).

## Adopted jobs
(none — the other jobs in the queue belong to the sibling campaigns `longchunk` and `probes`)

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| F frozen vs R reference, seeds 42 / 43 / 44 | decided: FAILED (documents, 2 of 3 seeds) | — |
| A1 compressor frozen | feasibility PASS; seed 42 full RUNNING + chain | seed 43 full (`ft_freeze.sh compfrozen 43`, `--time 12:21:00`, same 4 evals + module check chained) as soon as BOTH hold: a training slot (CAMP_MAX_TRAIN = 2: now A1 s42 + the pending A2 smoke) and a `gpu` slot that is not the owner's last (now 7 of 8 used). Then the paired judge `paired_judge_cpu.sh` A1 vs ref_s42 / ref_s43 when both seeds' evals are judged |
| A2 connector frozen | smoke PENDING | when 189065 + 189067 end: read against the A2 PASS rule (log `freeze_part=connector: froze N connector tensors`, `Compressor trainable parameters` > 0, connector.linear + norm_gain 0.0%, compressor lora_B > 0%, decoder lora_B > 0%, loss within .05 of .639 / .580, config diff = `freeze_compressor` → `freeze_part` only). Full cells only after A1's two full cells are submitted |
| after-gate arms for a VALIDATED gate | void: gate 0 failed | — |

## Next
- The tick wakes this loop when the A2 smoke's check (189067) or the A1 s42 run (189094) leaves the queue, or at 6 h (about 07:40) for the A1 health probes at steps 100-300.
- At that wake, if the A2 smoke still pends and a `gpu` slot is free, A1 seed 43 still cannot start (training cap). The law gives no pre-authorised way to swap them; leaving it as is (A1 s43 starts after the A2 smoke, ~8 h later than a swap would allow).

## Deliverable — judge accuracy ↑ (Qwen3-8B rubric judge), n = 1,500 per set; Δ = paired difference arm − reference of the same seed [95% CI] (jobs 188650, 188701)
| model | what trains | rate (tokens per slot; slots per page) | documents ↑ s42 / s43 / s44 | photos ↑ s42 / s43 / s44 | MRQA text ↑ s42 / s43 / s44 | HotpotQA ↑ s42 / s43 / s44 |
|---|---|---|---|---|---|---|
| R reference (184778 / 184780 re-evaluated 11-11; 188652) | compressor LoRA + merger + connector + decoder LoRA | 8; 123 per 1M-px page (980 visual tokens) | .610 / .615 / .620 | .673 / .685 / .671 | .718 / .724 / .721 | .603 / .593 / .603 |
| F frozen (188582 / 188583 / 188651) | decoder LoRA only | same | .582 / .580 / .579 | .664 / .672 / .675 | .695 / .697 / .697 | .575 / .571 / .577 |
| A1 compressor frozen (189094 / —) | connector + decoder LoRA | same | s42 running / s43 not yet launched | | | |
| A2 connector frozen | compressor LoRA + merger + decoder LoRA | same | smoke queued | | | |
| uncompressed decoder (earlier runs, same questions) | — | 1 (raw tokens) | .809 | .717 | .807 | — |

| paired Δ frozen F − reference ↑ (CI) | seed 42 | seed 43 | seed 44 | mean of the 3 Δ |
|---|---|---|---|---|
| documents | −.028 [−.049, −.007] | −.035 [−.057, −.014] | −.041 [−.062, −.021] | −.035 |
| photos | −.009 [−.026, +.008] | −.013 [−.030, +.004] | +.005 [−.013, +.021] | −.006 |
| MRQA text | −.023 [−.040, −.008] | −.027 [−.043, −.009] | −.025 [−.041, −.009] | −.025 |
| HotpotQA | −.028 [−.045, −.013] | −.023 [−.041, −.006] | −.026 [−.043, −.010] | −.026 |
Discordant pairs (frozen-only right / reference-only right), McNemar p — seeds 42 / 43 / 44: documents 111/153 p .011, 109/161 p .002, 103/164 p < .001; photos 78/92 p .32, 78/97 p .17, 90/83 p .65; MRQA 59/94 p .006, 67/107 p .003, 58/95 p .003; HotpotQA 64/106 p .002, 68/102 p .011, 64/103 p .003.

## Module movement (entries changed %, per module, vs the stage-2 model 184777)
| module | F smoke s42 (188578) | unfrozen smoke s42 (188579) | A1 smoke s42 (189066) | F full s42 (188584) | F full s43 (188585) | F full s44 (188699) | R full s44 (188700) |
|---|---|---|---|---|---|---|---|
| decoder lora_A / lora_B | 30.6 / 76.3 | 26.7 / 75.8 | 30.6 / 76.2 | 14.7 / 66.7 | 14.7 / 66.3 | 14.8 / 66.7 | 15.1 / 67.8 |
| compressor lora_A / lora_B | 0.0 / 0.0 | 25.4 / 76.4 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 16.9 / 74.1 |
| compressor `<MEM>` token row | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| connector MLP | 0.0 | 25.4 | 28.7 (.0027 of its norm) | 0.0 | 0.0 | 0.0 | 20.7 (.0073 of its norm) |
| connector norm gain | 0.0 | 0.0 (bf16 lost update) | 0.0 (bf16 lost update) | 0.0 | 0.0 | 0.0 | 0.0 (bf16 lost update) |

## Training cost (A100-80GB) — provisional (one same-time pair, two nodes)
| run | s/step | wall | peak GPU memory (sacct gres/gpumem) | trainable params |
|---|---|---|---|---|
| F frozen s42 / s43 / s44 (188582 / 188583 / 188651) | 71.5 / 76.1 / 71.5 | 8.7 / 9.3 / 8.7 h | 49.9 / 49.9 / 50.0 GB | 155,978,240 (decoder only) |
| R reference s42 / s43 / s44 (184778 / 184780 / 188652) | 90.3 / 113.2 / 95.1 | 11.0 / 13.8 / 11.6 h | — / — / 34.0 GB | 307,771,392 |
| A1 smoke s42 (189064), 20 steps | 81.2 overall (~70.5 steady) | 29 min | 49.8 GB | 176,958,976 (decoder + connector) |
Open question: why a frozen compressor uses MORE GPU memory. A guess, not checked: gradient checkpointing only runs in train mode and the frozen compressor is in eval mode, so its activations are kept. A1 (compressor frozen, connector training) peaks the same as F, consistent with that guess.

## Proposed text for the owner's chats (gate 0 is decided)
- docs/training_plan.md, gate 0 of "Standing proposal": "FAILED (campaign freeze, 3 seeds, n = 1,500 per set, paired judge). A decoder-only finetune on frozen `<MEM>`s loses 2.8 / 3.5 / 4.1 pt on documents and 2.3 / 2.7 / 2.5 pt on MRQA text (CI below 0 at every seed), 2.3-2.8 pt on HotpotQA; photos tie. Training is ~25% faster but peaks at 50 vs 34 GB GPU memory. Split test running (compressor frozen only / connector frozen only)."
- HANDOFF.md thread 7: "Freezing compressor + connector at finetune fails gate 0 (documents −3.5 pt mean of 3 seeds, MRQA −2.5, photos ~0). `PISCOConfig.freeze_part` (compressor | connector) added by the freeze campaign for the split test; diff in campaign_freeze/patches/split_freeze.diff."

## Notes this wake
- Verifier fix: the last STATE said both smokes pended for CPUs. Only A1 did (Resources); the A2 smoke pends for Priority on `gpu-be`. Corrected above.
- A1 s42 launched on `gpu` after `gpucap land a100 1`: the owner's cap is 8, 6 were in use (longchunk: 4 + 1 + 1), 7 after. A1 s43 cannot also go to `gpu` now (it would take the last slot) and the training cap (2) is full with A1 s42 + the A2 smoke.
- `--time` of the full A1 cell follows the law literally (smoke s/step incl. the 232-s first step × 438 × 1.25 = 12:21). The F full runs took 8.7-9.3 h, so this is generous.
- Slip: one short `python3` string-slicing heredoc ran on the login node while reading the old STATE.md tables (seconds, no data). Hard rule 8 forbids it; not repeated.

## Patches this wake
- None. `pisco/model.py` + `test_freeze.py` from wake 11 are still uncommitted (this loop cannot commit `pisco/`); their diff is `campaign_freeze/patches/split_freeze.diff`, md5 d4412015df7ecd2cfd8408693ec3d5ef.

## Blockers / needs the owner
- None blocking. An owner's chat should commit `pisco/model.py` and `test_freeze.py` (the split-freeze setting) — the running A1 job uses them from the working tree.
