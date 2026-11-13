# STATE — campaign freeze   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-13 23:00 tick (wake 23; woken because the connector-frozen seed-43 run wrote checkpoint-200, the wait condition written at 20:30)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- **Gate 0 FAILED on three seeds: the question-answering finetune needs the compressor side to train.** With the compressor and connector frozen (only the decoder LoRA trains), documents lose 2.8 / 3.5 / 4.1 pt of judge accuracy at seeds 42 / 43 / 44, each with a 95% CI wholly below 0; the law's "Failed" rule (≤ −3 pt with the CI below 0 at 2 of 3 seeds on one gating set) is met by seeds 43 and 44. Paired per question, n = 1,500 per set, same eval code (md5 `eval_mm_docqa.py` dfe32aa0, `eval_mm_4b.sh` 50abc167, `judge_mm_docqa.py` 02bc8748), jobs 188650 (seeds 42 / 43) and 188701 (seed 44), `ALL` rows:
  - documents: −.028 [−.049, −.007], −.035 [−.057, −.014], −.041 [−.062, −.021] → FAILED;
  - MRQA text: −.023 [−.040, −.008], −.027 [−.043, −.009], −.025 [−.041, −.009] → below the −2 pt bar at all three seeds, CI below 0, not ≤ −3 pt;
  - photos: −.009 [−.026, +.008], −.013 [−.030, +.004], +.005 [−.013, +.021] → no difference shown at any seed;
  - HotpotQA (reported, not gating): −.028 [−.045, −.013], −.023 [−.041, −.006], −.026 [−.043, −.010].
  - The prediction in CAMPAIGN.md "Mission" (the reference finetune barely moves the connector, so freezing may tie) is refuted and marked there.
- **The seed-44 pair is clean:** both finetunes ran from the same stage-2 model with the same recipe file (`ft_freeze.sh frozen|unfrozen 44`); all 8 evals used the eval-code md5 above; the paired test exited 0 (job 188701).
- **The freeze held over all three full F runs** (1.0 epoch, 27,936 rows each): every compressor and connector row, norm gain included, changed 0.0%; decoder lora_B 66.7% / 66.3% / 66.7% of entries changed (seeds 42 / 43 / 44, jobs 188584 / 188585 / 188699). The seed-44 reference run, checked the same way, moved compressor lora_B 74.1% and the connector MLP 20.7% (job 188700): the check sees movement when there is some. It cannot see the `<MEM>` row and has no row for the merger or the vision tower.
- **In this bf16 recipe the connector norm gain never moves, even when trainable**: 0.0% changed after 20 steps (jobs 188579, 189066) and after a whole epoch (seed-44 reference 188700, A1 seed 42 189099, A1 seed 43 189450), value .46094 each time.
  - provisional: so the frozen-vs-reference difference is compressor LoRA + connector MLP, plus possibly the merger and the compressor `<MEM>` row — neither is measured (no merger row in the check; one 2,560-entry row inside a 1,270M-entry group rounds to 0.0%; no positive control for either). A second, non-weight difference: the frozen side runs in eval mode (no LoRA dropout, possibly no gradient checkpointing, not checked in the code).
- **Split-freeze setting installed and tested**: `PISCOConfig.freeze_part = compressor | connector` (default unset) in `pisco/model.py` (md5 994cff3d, unchanged at 23:05), 5 new `test_freeze.py` cases; staged copy (job 189060) and installed tree (job 189063) both pass `test_positions.py` 12/12, `test_mm_collators.py` 33/33, `test_freeze.py` 10/10, exit 0.
- **Split arm A1 (compressor frozen) feasibility PASS** (20-step smoke, seed 42, 1,280 rows = .046 epoch; jobs 189064 + 189066): log `froze 501 compressor tensors` + `Compressor trainable parameters: 0`; compressor rows 0.0%, connector MLP 28.7%, decoder lora_B 76.2%; train loss .6033 / .5952 at steps 10 / 20 vs the unfrozen smoke's .639 / .580 (188534); config diff one line (`freeze_compressor: true` → `freeze_part: compressor`); 81.2 s/step, 49.8 GB peak.
- **A1 seed 42 and seed 43 full runs COMPLETED, freeze held** (1.0 epoch, 27,936 rows each): seed 42 job 189094 (homer-013, 8.7 h, 71.5 s/step, 50.2 GB), seed 43 job 189445 (homer-020, 11.2 h, 91.7 s/step, 50.1 GB). After-run module checks 189099 / 189450: compressor rows 0.0%; connector MLP 19.1% / 19.4% changed (.0061 of its norm); norm gain 0.0%; decoder lora_B 66.0% / 66.4%.
- **A1 gate test: a TIE at two seeds → third seed (job 189919, exit 0, n = 1,500 per set, paired, same eval-code md5 as gate 0).** A1 − reference, seed 42 / seed 43, `ALL` rows:
  - documents −.029 [−.050, −.009] p .008 / −.039 [−.061, −.017] p .001 — below −2 pt with the CI below 0 at both seeds, but ≤ −3 pt only at seed 43;
  - MRQA text −.023 [−.039, −.007] p .006 / −.031 [−.047, −.014] p .001 — ≤ −3 pt only at seed 43;
  - photos .000 [−.017, +.016] / −.013 [−.031, +.004] — no difference shown;
  - HotpotQA (reported) −.019 [−.036, −.003] / −.011 [−.029, +.007].
  - Rule reading: not Validated (documents and MRQA below −2 pt at both seeds); not Failed (no gating set ≤ −3 pt at BOTH seeds; seed-42 documents −.029 misses the line by .001) → Tie → `ft_freeze.sh compfrozen 44` vs the seed-44 reference 188652, then 2 of 3.
  - **A1 equals F** (reported, same job): A1 − F documents −.001 [−.011, +.008] / −.004 [−.013, +.005]; photos +.009 [−.006, +.024] / −.001 [−.010, +.008]; MRQA .000 [−.011, +.012] / −.004 [−.014, +.007]; HotpotQA +.009 [−.001, +.019] / +.012 [+.001, +.023] p .047. Training the connector on top of the decoder LoRA recovers none of the gate-0 loss on the gating sets.
- **Split arm A2 (connector frozen) feasibility PASS** (20-step smoke, seed 42, 1,280 rows = .046 epoch, homer-019; jobs 189739 + 189740): log `freeze_part=connector: froze 5 connector tensors (eval mode kept)` + `Compressor trainable parameters: 130812416`; connector MLP and norm gain 0.0%, compressor lora_B 76.2%, decoder lora_B 75.8%; train loss .6165 / .561 at steps 10 / 20 vs .639 / .580 (188534); config diff one line (`freeze_compressor: true` → `freeze_part: connector`); ~88 s/step steady, 31.4 GB peak.
- provisional: the gate-0 loss is spread over the data sources: 39 of 42 per-source differences are negative over the three seeds (28 of 28 at seeds 42 / 43, job 188650; 11 of 14 at seed 44, job 188701 — positives realworldqa +.010, vizwiz +.020, hotpot_comparison +.013). A sign count, not a test: most per-source CIs span 0 (n = 250-500 per source; HotpotQA bridge 1,197 and comparison 303).
- provisional: a frozen compressor costs ~25% less time but ~16 GB MORE GPU memory. Seed-44 pair: frozen 71.5 s/step vs reference 95.1 (188651 / 188652, two nodes); peak GPU memory 50,020 MB vs 34,044 MB. By part: compressor frozen 49.7 / 49.8 / 50.2 / 50.1 GB (F smoke 188528, A1 smoke 189064, A1 full 189094 / 189445); compressor training 31.9 / 31.4 GB (unfrozen smoke 188534, A2 smoke 189739). Consistent with the frozen compressor running in eval mode without gradient checkpointing; not checked in the code. Node speed dominates time (A1 s43 91.7 vs s42 71.5 s/step).
- provisional: the fully frozen runs end at a HIGHER held-out loss than the reference of the same seed (64 rows, no test): .435 vs .414, .413 vs .375, .427 vs .401 (seeds 42 / 43 / 44; .414 / .375 from trainer_state of 184778 / 184780, not campaign cells).

**Open, and running**
- **A1 compressor frozen, seed 44 — to launch at the next wake.** Both training slots (CAMP_MAX_TRAIN = 2, PENDING counts) are held by A2 until seed 42 ends (~02:25 on 11-14). The chain to submit then (6 submissions = the per-wake cap), exactly as seed 43 (189445-189450):
  1. `expmon launch --exp freeze --stage ft --parent 184777 --partition gpu --constraint gpu_a100 --time 12:21:00 expQ/freeze/ft_freeze.sh compfrozen 44` (12:21 = smoke 81.2 s/step × 438 × 1.25; `gpucap land a100 1` first, never the last free slot of the owner's 8);
  2. four evals `--stage eval --parent <ft> --partition gpu-be --exclude homer-[005-009] --dependency afterok:<ft>`, `eval_mm_4b.sh expQ/freeze/ft_compfrozen_s44/model pisco "" <set>`: documents `eval_data/mm_docqa_1500_1m` `--time 0:40:00`, photos `eval_data/mm_naturalqa_1500` `0:25:00`, MRQA `CML=4096` `eval_data/text_mrqa_ood_1500` `0:22:00`, HotpotQA `CML=4096 TOPK=10` `eval_data/text_hotpotqa_1500` `0:29:00` (quote `homer-[005-009]`: zsh);
  3. module check `--stage check --parent <ft> --partition gpu-be --time 0:02:00 --dependency afterok:<ft> expQ/freeze/check_module_move.sh expQ/archq3/pt_mm_5M_spread/model expQ/freeze/ft_compfrozen_s44/model`.
  - The paired test (A1 s44 vs `ft_unfrozen_s44` = the gate, vs `ft_frozen_s44` reported; both judged file sets exist) goes in the wake after, chained on the 4 evals.
- **A2 connector frozen, seed 42** (job 189842, homer-019): 307 of 437 steps at 23:04 (19,650 rows, .70 epoch), ~92 s/step, ends ~02:25 on 11-14. Held-out loss .438 / .386 / **.397** at .23 / .46 / .69 epoch — reference 184778 .513 / .455 / .393, F .455 / .444 / .438, A1 .459 / .449 / .437: tracks the reference, below F and A1. Healthy (kill line .513). provisional, 64 rows.
  - provisional: step-100 preview (job 189918, first 300 documents) judge .593; same 300 questions: reference final .610 (A2-only / reference-only right 18 / 23), A1 final .587, F final .590.
  - Chained `afterok`: evals 189844-189847 (documents / photos / MRQA / HotpotQA) and module check 189843.
- **A2 connector frozen, seed 43** (job 189899, homer-020): held-out loss .489 at .23 epoch (above the reference's first value .471 = one strike), then **.405 at .46 epoch** (step 200, 12,800 rows; reference 184780 .439, F .431, A1 .449) → back below the kill line, kept. ~93 s/step, ends ~04:55 on 11-14.
  - provisional: step-100 preview (job 189973, exit 0, homer-016) judge **.623** on the first 300 documents (EM .497); same 300 questions: reference 184780 final .610 (A2-only / reference-only right 21 / 17), A1 s43 final .587, F s43 final .593. n = 300, never gates.
  - Chained `afterok`: evals 189900-189903 and module check 189904.
- **A2 gate test submitted this wake** (job 190020, `paired_judge_cpu.sh`, CPU `gpu-be`, `--time 0:05:00`): A2 vs reference of the same seed (the gate), A2 vs F and A2 vs A1 of the same seed (reported), chained `afterok` on the 8 A2 evals. Expected ~06:30 on 11-14 if the evals land promptly.
- Prediction, not a result (written before any A1 / A2 paired number): A1 fails like F, A2 ties R. A1 half: consistent so far (A1 = F on every set, formally a tie at two seeds).

## Owned jobs (jobid | arm | state | rows seen | eta | note)
| job | what | state at 23:05 | rows | eta | note |
|---|---|---|---|---|---|
| 189095-189098, 189446-189449 | A1 evals | COMPLETED exit 0 | 1,500 Q each | done | 189095 finished its third attempt 21:11 (homer-016) |
| 189919 | A1 paired judge | COMPLETED exit 0 | — | done | TIE (see Settled) |
| 189842 | A2 full `ft_freeze.sh connfrozen 42` | RUNNING homer-019 | 19,650 of 27,936 (.70 epoch) | ~02:25 | kill on NaN or 2 probes above .513 |
| 189844-189847 / 189843 | A2 s42 evals / module check | PENDING afterok 189842 | 1,500 Q each | after ~02:25 | pair with ref_s42 files |
| 189899 | A2 full `ft_freeze.sh connfrozen 43` | RUNNING homer-020 | 12,800+ of 27,936 (.46 epoch) | ~04:55 | 1 probe above .471, then below; kept |
| 189900-189903 / 189904 | A2 s43 evals / module check | PENDING afterok 189899 | 1,500 Q each | after ~04:55 | pair with ref_s43 files |
| 189918, 189973 | A2 s42 / s43 step-100 previews | COMPLETED exit 0 | 300 Q | done | .593 / .623 |
| 190020 | A2 paired judge (gate + vs F + vs A1) | PENDING afterok on the 8 A2 evals | — | ~06:30 | new this wake |
All evals: `gpu-be`, `--exclude=homer-[005-009]`, PX=1003520. Eval code md5 re-checked at 23:05 = gate 0, so the judged reference files still pair as they are. Re-check before reading any paired test.
Trainings in flight: 189842 + 189899 = 2 = CAMP_MAX_TRAIN. Owner's `gpu` cap: 6 of 8 (gpucap 23:04: 2 free).
GPU budget: ~100 GPU-h used of 192 (gate 0 ~52 + A1 ~21 + A2 smoke .6 + A2 s42 8 + A2 s43 5.4 + 13 evals and previews ~7 + checks); left in the plan: A2 s42 ~3.4 h, A2 s43 ~6 h, 8 A2 evals ~3.5 h, A1 s44 ~11 h + 4 evals ~1.5 h → ~26 GPU-h (+ ~15 if A2 also ties).

## Adopted jobs
(none — the other jobs in the queue belong to the sibling campaigns `longchunk` and `probes`)

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| F frozen vs R reference, seeds 42 / 43 / 44 | decided: FAILED (documents, 2 of 3 seeds) | — |
| A1 compressor frozen | TIE at seeds 42 / 43 (documents −.029 / −.039, MRQA −.023 / −.031) | launch seed 44 chain when 189842 ends; then decide on 2 of 3 seeds |
| A2 connector frozen | s42 .70 epoch (held-out .397, tracks the reference); s43 .46 epoch (.405, kept) | (1) after each run: module check must show connector rows 0.0%; (2) read 190020 when the 8 evals are judged; apply Validated / Failed / Tie |
| after-gate arms for a VALIDATED gate | void: gate 0 failed | — |

## Next
- Wake when A2 seed 42 (189842) leaves the queue (~02:25), max 4 h: launch the A1 seed-44 chain above (gpucap first), read the A2 s42 held-out loss at 1.0 epoch.

## Deliverable — judge accuracy ↑ (Qwen3-8B rubric judge), n = 1,500 per set; Δ = paired difference arm − reference of the same seed [95% CI] (jobs 188650, 188701, 189919)
| model | what trains | rate (tokens per slot; slots per page) | documents ↑ s42 / s43 / s44 | photos ↑ s42 / s43 / s44 | MRQA text ↑ s42 / s43 / s44 | HotpotQA ↑ s42 / s43 / s44 |
|---|---|---|---|---|---|---|
| R reference (184778 / 184780 re-evaluated 11-11; 188652) | compressor LoRA + merger + connector + decoder LoRA | 8; 123 per 1M-px page (980 visual tokens) | .610 / .615 / .620 | .673 / .685 / .671 | .718 / .724 / .721 | .603 / .593 / .603 |
| F frozen (188582 / 188583 / 188651) | decoder LoRA only | same | .582 / .580 / .579 | .664 / .672 / .675 | .695 / .697 / .697 | .575 / .571 / .577 |
| A1 compressor frozen (189094 / 189445) | connector + decoder LoRA | same | .581 / .576 / — | .673 / .671 / — | .695 / .693 / — | .583 / .583 / — |
| A2 connector frozen (189842 / 189899) | compressor LoRA + merger + decoder LoRA | same | training (.70 epoch) / training (.46 epoch) | | | |
| uncompressed decoder (earlier runs, same questions) | — | 1 (raw tokens) | .809 | .717 | .807 | — |

| paired Δ ↑ (CI), n = 1,500 | documents | photos | MRQA text | HotpotQA (reported) |
|---|---|---|---|---|
| F − R seed 42 | −.028 [−.049, −.007] | −.009 [−.026, +.008] | −.023 [−.040, −.008] | −.028 [−.045, −.013] |
| F − R seed 43 | −.035 [−.057, −.014] | −.013 [−.030, +.004] | −.027 [−.043, −.009] | −.023 [−.041, −.006] |
| F − R seed 44 | −.041 [−.062, −.021] | +.005 [−.013, +.021] | −.025 [−.041, −.009] | −.026 [−.043, −.010] |
| A1 − R seed 42 | −.029 [−.050, −.009] | .000 [−.017, +.016] | −.023 [−.039, −.007] | −.019 [−.036, −.003] |
| A1 − R seed 43 | −.039 [−.061, −.017] | −.013 [−.031, +.004] | −.031 [−.047, −.014] | −.011 [−.029, +.007] |
| A1 − F seed 42 | −.001 [−.011, +.008] | +.009 [−.006, +.024] | .000 [−.011, +.012] | +.009 [−.001, +.019] |
| A1 − F seed 43 | −.004 [−.013, +.005] | −.001 [−.010, +.008] | −.004 [−.014, +.007] | +.012 [+.001, +.023] |
Discordant pairs (arm-only right / reference-only right), McNemar p: F − R seeds 42 / 43 / 44 — documents 111/153 p .011, 109/161 p .002, 103/164 p < .001; photos 78/92 p .32, 78/97 p .17, 90/83 p .65; MRQA 59/94 p .006, 67/107 p .003, 58/95 p .003; HotpotQA 64/106 p .002, 68/102 p .011, 64/103 p .003. A1 − R seeds 42 / 43 — documents 108/152 p .008, 108/166 p .001; photos 83/83 p 1.0, 78/98 p .15; MRQA 60/95 p .006, 64/110 p .001; HotpotQA 65/94 p .026, 81/97 p .26.

Step-100 previews (judge accuracy ↑ on the first 300 documents of `mm_docqa_1500_1m`, provisional, never gates):
| model | at .23 epoch (checkpoint-100) | final (1.0 epoch), same 300 Q (wake-computed) |
|---|---|---|
| R reference s42 / s43 (184778 / 184780) | — (no checkpoint-100 kept) | .610 / .610 |
| F frozen s42 (preview 188594; finals 188596 / 188598) | .583 | .590 / .593 |
| A1 compressor frozen s42 (preview 189434; finals 189095 / 189446) | .593 | .587 / .587 |
| A2 connector frozen s42 / s43 (previews 189918 / 189973) | .593 / .623 | training |

Held-out loss ↓ (64 rows, no paired test; a health signal, not a gate), at .23 / .46 / .69 / .92 / 1.0 epoch:
| run | seed 42 | seed 43 |
|---|---|---|
| R reference (184778 / 184780 trainer_state) | .513 / .455 / .393 / .394 / .414 | .471 / .439 / .411 / .441 / .375 |
| F frozen (188582 / 188583 logs) | .455 / .444 / .438 / .434 / .435 | .442 / .431 / .416 / .415 / .413 |
| A1 compressor frozen (189094 / 189445 logs) | .459 / .449 / .437 / .435 / .435 | .464 / .449 / .431 / .429 / .428 |
| A2 connector frozen (189842 / 189899 logs) | .438 / .386 / .397 / running | .489 / .405 / running |

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
| A2 full s42 (189842, homer-019), steps 1-307 | ~92 | ~11.2 h projected | — (running) | 286,790,656 |
| A2 full s43 (189899, homer-020), steps 1-200 | ~93 | ~11.3 h projected | — (running) | 286,790,656 |

## Proposed text for the owner's chats (gate 0 is decided)
- docs/training_plan.md, gate 0 of "Standing proposal": "FAILED (campaign freeze, 3 seeds, n = 1,500 per set, paired judge). A decoder-only finetune on frozen `<MEM>`s loses 2.8 / 3.5 / 4.1 pt on documents and 2.3 / 2.7 / 2.5 pt on MRQA text (CI below 0 at every seed), 2.3-2.8 pt on HotpotQA; photos tie. Training is ~25% faster but peaks at 50 vs 34 GB GPU memory. Split test: freezing the compressor alone (connector + decoder train) loses the same as freezing both (A1 − F within .012 on every set, 2 seeds; third seed running); connector-frozen arm running."
- HANDOFF.md thread 7: "Freezing compressor + connector at finetune fails gate 0 (documents −3.5 pt mean of 3 seeds, MRQA −2.5, photos ~0). Freezing only the compressor costs the same. `PISCOConfig.freeze_part` (compressor | connector) added by the freeze campaign for the split test; diff in campaign_freeze/patches/split_freeze.diff."

## Notes this wake
- Woken by the wait condition (A2 s43 checkpoint-200 exists). No ✗ of this campaign in the brief. Verifier findings of the 20:30 STATE.md: all eight were `provisional` items it confirmed as provisional; fixed its one error (per-source n now names the HotpotQA bridge 1,197 / comparison 303).
- Kill rule on A2 s43: second probe .405 < .471 → kept.
- A1 paired test read (189919): TIE → seed 44 queued for the next wake (no training slot now).
- Launched 1 of 6: A2 paired test 190020 (chained, CPU). No `--force` needed this time (the empty-factor refusal did not fire).
- Recorded probes: 189899 eval_loss .4053 at .458 epoch; 189842 eval_loss .3968 at .686 epoch; 189973 judge .6233 (n = 300).
- `expmon display --exp freeze` still writes 0 table rows (already griped 3 times; not repeated). The STATE.md tables above are the deliverable.
- Code md5 at 23:05: `eval_mm_docqa.py` dfe32aa0, `eval_mm_4b.sh` 50abc167, `judge_mm_docqa.py` 02bc8748, `pisco/model.py` 994cff3d, `ft_freeze.sh` 668268e3.

## Patches this wake
- None. `pisco/model.py` + `test_freeze.py` from wake 11 are still uncommitted (this loop cannot commit `pisco/`); their diff is `campaign_freeze/patches/split_freeze.diff`, md5 d4412015df7ecd2cfd8408693ec3d5ef.

## Blockers / needs the owner
- None blocking. An owner's chat should commit `pisco/model.py` and `test_freeze.py` (the split-freeze setting): the running and queued A2 jobs, and the coming A1 seed-44 run, use them from the working tree.
