# STATE — campaign freeze   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-14 12:00 tick (wake 26; woken on schedule: the 6 h maximum of the wait on the A2 gate test 190020 ran out; 190020 still waits on its evals)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- **Gate 0 FAILED on three seeds: the question-answering finetune needs the compressor side to train.** With the compressor and connector frozen (only the decoder LoRA trains), documents lose 2.8 / 3.5 / 4.1 pt of judge accuracy at seeds 42 / 43 / 44, each with a 95% CI wholly below 0; the law's "Failed" rule (≤ −3 pt with the CI below 0 at 2 of 3 seeds on one gating set) is met by seeds 43 and 44. Paired per question, n = 1,500 per set, same eval code (md5 `eval_mm_docqa.py` dfe32aa0, `eval_mm_4b.sh` 50abc167, `judge_mm_docqa.py` 02bc8748), jobs 188650 (seeds 42 / 43) and 188701 (seed 44), `ALL` rows:
  - documents: −.028 [−.049, −.007], −.035 [−.057, −.014], −.041 [−.062, −.021] → FAILED;
  - MRQA text: −.023 [−.040, −.008], −.027 [−.043, −.009], −.025 [−.041, −.009] → below the −2 pt bar at all three seeds, CI below 0, not ≤ −3 pt;
  - photos: −.009 [−.026, +.008], −.013 [−.030, +.004], +.005 [−.013, +.021] → no difference shown at any seed;
  - HotpotQA (reported, not gating): −.028 [−.045, −.013], −.023 [−.041, −.006], −.026 [−.043, −.010].
  - The prediction in CAMPAIGN.md "Mission" (the reference finetune barely moves the connector, so freezing may tie) is refuted and marked there.
- **The seed-44 pair is clean:** both finetunes ran from the same stage-2 model with the same recipe file (`ft_freeze.sh frozen|unfrozen 44`); all 8 evals used the eval-code md5 above; the paired test exited 0 (job 188701).
- **The freeze held in every full run checked** (1.0 epoch, 27,936 rows each): F seeds 42 / 43 / 44 (188584 / 188585 / 188699) every compressor and connector row 0.0%; A1 seeds 42 / 43 / **44** (189099 / 189450 / **190129, new**) every compressor row 0.0%, connector MLP 19.1 / 19.4 / 19.5%; A2 seeds 42 / 43 (189843 / 189904) connector MLP and norm gain 0.0%, compressor lora_B 73.6% / 73.8%. The seed-44 reference run moved compressor lora_B 74.1% and the connector MLP 20.7% (188700): the check sees movement when there is some. It cannot see the `<MEM>` row and has no row for the merger or the vision tower.
- **In this bf16 recipe the connector norm gain never moves, even when trainable**: 0.0% changed after 20 steps (jobs 188579, 189066) and after a whole epoch (seed-44 reference 188700, A1 seeds 42 / 43 / 44 189099 / 189450 / 190129), value .46094 each time.
  - provisional: so the frozen-vs-reference difference is compressor LoRA + connector MLP, plus possibly the merger and the compressor `<MEM>` row — neither is measured (no merger row in the check; one 2,560-entry row inside a 1,270M-entry group rounds to 0.0%; no positive control for either). A second, non-weight difference: the frozen side runs in eval mode — no LoRA dropout, and no gradient checkpointing: `PISCO.train()` puts the frozen parts back in eval mode (`pisco/model.py:1294-1300`) and HF checkpoints a layer only when `self.training` is true (`transformers/modeling_layers.py:60`).
- **Split-freeze setting installed and tested**: `PISCOConfig.freeze_part = compressor | connector` (default unset) in `pisco/model.py` (md5 994cff3d, unchanged at 12:00), 5 new `test_freeze.py` cases; staged copy (job 189060) and installed tree (job 189063) both pass `test_positions.py` 12/12, `test_mm_collators.py` 33/33, `test_freeze.py` 10/10, exit 0.
- **Split arm A1 (compressor frozen) feasibility PASS** (20-step smoke, seed 42, 1,280 rows = .046 epoch; jobs 189064 + 189066): log `froze 501 compressor tensors` + `Compressor trainable parameters: 0`; compressor rows 0.0%, connector MLP 28.7%, decoder lora_B 76.2%; train loss .6033 / .5952 at steps 10 / 20 vs the unfrozen smoke's .639 / .580 (188534); config diff one line (`freeze_compressor: true` → `freeze_part: compressor`); 81.2 s/step, 49.8 GB peak.
- **Split arm A2 (connector frozen) feasibility PASS** (20-step smoke, seed 42, 1,280 rows = .046 epoch, homer-019; jobs 189739 + 189740): log `freeze_part=connector: froze 5 connector tensors (eval mode kept)` + `Compressor trainable parameters: 130812416`; connector MLP and norm gain 0.0%, compressor lora_B 76.2%, decoder lora_B 75.8%; train loss .6165 / .561 at steps 10 / 20 vs .639 / .580 (188534); config diff one line (`freeze_compressor: true` → `freeze_part: connector`); 98.9 s/step overall, 31.4 GB peak.
- **A1 gate test: a TIE at two seeds → third seed (job 189919, exit 0, n = 1,500 per set, paired, same eval-code md5 as gate 0; verdict closed, tie read).** A1 − reference, seed 42 / seed 43, `ALL` rows:
  - documents −.029 [−.050, −.009] p .008 / −.039 [−.061, −.017] p .001 — below −2 pt with the CI below 0 at both seeds, but ≤ −3 pt only at seed 43;
  - MRQA text −.023 [−.039, −.007] p .006 / −.031 [−.047, −.014] p .001 — ≤ −3 pt only at seed 43;
  - photos .000 [−.017, +.016] / −.013 [−.031, +.004] — no difference shown;
  - HotpotQA (reported) −.019 [−.036, −.003] / −.011 [−.029, +.007].
  - Rule reading: not Validated (documents and MRQA below −2 pt at both seeds); not Failed (no gating set ≤ −3 pt at BOTH seeds; seed-42 documents .5807 − .6100 = −.0293 misses the line by .001) → Tie → third seed (trained, see Open).
- provisional: **A1 − F (reported, job 189919, not a gate):** documents −.001 [−.011, +.008] / −.004 [−.013, +.005]; MRQA .000 [−.011, +.012] / −.004 [−.014, +.007]; photos +.009 [−.006, +.024] / −.001 [−.010, +.008]; HotpotQA +.009 [−.001, +.019] / +.012 [+.001, +.023] p .047. No recovery of the gate-0 loss is shown; the upper CI bounds still allow up to ~1 pt on documents (+.008 / +.005) and MRQA (+.012 / +.007), against a gate-0 loss of 2.3-4.1 pt.
- provisional: the gate-0 loss is spread over the data sources: 39 of 42 per-source differences are negative over the three seeds (28 of 28 at seeds 42 / 43, job 188650; 11 of 14 at seed 44, job 188701 — positives realworldqa +.010, vizwiz +.020, hotpot_comparison +.013). A sign count, not a test: most per-source CIs span 0 (n = 250-500 per source; HotpotQA bridge 1,197 and comparison 303).
- provisional: training the compressor costs GPU memory LESS than freezing it. Peak GPU memory (sacct gres/gpumem, grid probes `gpumem_gb`): compressor frozen 49.7 / 49.8 / 49.9 / 49.9 / 50.0 / 50.2 / 50.1 / **50.1** GB (F smoke 188528, A1 smoke 189064, F full 188582 / 188583 / 188651, A1 full 189094 / 189445 / **190124**); compressor training 31.9 / 31.4 / 33.8 / 33.7 / 34.0 GB (unfrozen smoke 188534, A2 smoke 189739, A2 full 189842 / 189899, reference s44 188652). Consistent with the eval-mode frozen compressor running without gradient checkpointing (code checked, see above); whether its activations are kept for backward (e.g. through an input-embedding `requires_grad` hook) is not checked. Time: frozen compressor 71.5-91.7 s/step, training compressor 92.2-95.1 s/step; node speed confounds it (A1 s43 91.7 vs s42 / s44 71.5 s/step).
- provisional: the fully frozen and compressor-frozen runs end at a HIGHER held-out loss than the reference of the same seed (64 rows, no test): F .435 / .413 / .427 and A1 .435 / .428 / **.423** vs reference .414 / .375 / .401 (seeds 42 / 43 / 44; .414 / .375 from trainer_state of 184778 / 184780, not campaign cells).

**Open, and running**
- **A1 compressor frozen, seed 44 — trained, 4 evals queued, gate test chained.**
  - 190124 COMPLETED exit 0 (homer-019, 02:32-11:15; 1.0 epoch, 27,936 rows; 8.7 h, 71.5 s/step, 50.1 GB). Held-out .451 / .436 / .421 / .422 / .423 at .23 / .46 / .69 / .92 / 1.0 epoch (reference 188652 .468 / .414 / .421 / .417 / .401; F s44 .427 at 1.0; 64 rows). Never above the kill line .468. Freeze held (190129).
  - Evals 190125-190128 (documents / photos / MRQA `CML=4096` / HotpotQA `CML=4096 TOPK=10`, `PX=1003520`): PENDING `Priority`, start estimates 16:15-17:15. Env plumbing confirmable from the output file names when they land (gripe 254abbe4).
  - Paired test 190218 (`ft_compfrozen_s44:ft_unfrozen_s44` = the gate's third seed; `ft_compfrozen_s44:ft_frozen_s44` reported), `afterok` 190125-190128: expected ~18:00 if no preemption.
  - provisional preview (step 100 = .23 epoch, 6,400 rows; first 300 documents, n = 300, job 190217): judge .597; same 300 questions (wake-computed with jq, query ids checked equal): reference s44 final .597, F s44 final .583. A preview never gates.
- **A2 connector frozen — both seeds trained, 3 of 8 evals done, gate test chained.**
  - seed 42 (189842): 1.0 epoch, held-out .434 (ref 184778 .414). Photos .667 (189845). MRQA 189846: its 05:22 attempt finished but the job was preempted and requeued; the 11:48 re-run rewrote the result file at 12:00 and is judging now. Documents 189844 and HotpotQA 189847 requeued after preemptions (189844 four times, 189847 four times), start estimates 17:30 / 18:45.
  - seed 43 (189899): 1.0 epoch, held-out .399 (ref 184780 .375). **New:** MRQA text judge .727 (189902, n = 1,500, exit 0) and HotpotQA .600 (189903, n = 1,500, exit 0). Documents 189900 and photos 189901 requeued after preemptions, start estimates 18:00 / 18:30.
  - provisional, unpaired: A2 seed 43 sits at the reference level on text — MRQA .727 vs reference .724 (F .697, A1 .693), HotpotQA .600 vs .593 (F .571, A1 .583). Consistent with the prediction "A2 ties R"; the paired test decides.
  - Gate test 190020 (A2 vs R same seed = the gate; vs F and vs A1 same seed, reported), `afterok` on the 8 evals: expected ~19:30 if no more preemptions.
- Prediction, not a result (CAMPAIGN.md "Split test gates", written 2026-11-12 23:40, before any A1 / A2 paired number): A1 fails like F, A2 ties R. A1 half partly consistent (A1 is a 2-seed tie; A1 − F shows no difference); A2 half: unpaired seed-43 text numbers consistent, no paired number yet.

## Owned jobs
| job | what | state at 12:20 | rows | eta | note |
|---|---|---|---|---|---|
| 189842 / 189843, 189899 / 189904 | A2 s42 / s43 full + module checks | COMPLETED exit 0 | 27,936 each (1.0 epoch) | done | freeze held |
| 189845 | A2 s42 photos eval | COMPLETED exit 0 | 1,500 Q | done | judge .667 |
| 189846 | A2 s42 MRQA eval | RUNNING homer-019 (re-run after a preemption that landed after the first attempt finished) | 1,500 Q | ~12:30 | |
| 189844 / 189847 | A2 s42 documents / HotpotQA | PENDING `Priority` (requeued) | 1,500 Q each | 17:30 / 18:45 start | |
| 189902 / 189903 | A2 s43 MRQA / HotpotQA | COMPLETED exit 0 | 1,500 Q each | done | .727 / .600 |
| 189900 / 189901 | A2 s43 documents / photos | PENDING `Priority` (requeued) | 1,500 Q each | 18:00 / 18:30 start | |
| 190020 | A2 paired judge (gate + vs F + vs A1) | PENDING afterok on the 8 A2 evals | — | ~19:30 | |
| 190124 / 190129 | A1 s44 full + module check | COMPLETED exit 0 | 27,936 (1.0 epoch) | done | freeze held |
| 190217 | A1 s44 step-100 preview | COMPLETED exit 0 | 300 Q | done | .597 |
| 190125-190128 | A1 s44 evals | PENDING `Priority` | 1,500 Q each | 16:15-17:15 start | pair with ft_unfrozen_s44 (188654 / 188673 / 188675 / 188677) |
| 190218 | A1 s44 paired judge | PENDING afterok 190125-190128 | — | ~18:00 | |
All evals: `gpu-be`, `--exclude=homer-[005-009]`, PX=1003520. Eval code md5 re-checked at 12:00 = gate 0 (`paired_judge.py` a3cc13b2), so the judged reference files still pair as they are. Re-check before reading any paired test.
Trainings in flight: 0 of CAMP_MAX_TRAIN 2. Nothing is launchable: an A2 third seed is owed only on a tie, and A1 decides on its third seed.
GPU budget: ~120 GPU-h used of 192 (gate 0 ~52 + A1 s42 / s43 / s44 ~29 + A2 smoke .6 + A2 s42 / s43 22.5 + 17 evals and previews ~9 + checks); left in the plan: 9 evals ~4 h → ~4 GPU-h (+ ~13 if A2 ties and needs seed 44).

## Adopted jobs
(none — the other jobs in the queue belong to the sibling campaigns `longchunk` and `probes`)

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| F frozen vs R reference, seeds 42 / 43 / 44 | decided: FAILED (documents, 2 of 3 seeds) | — |
| A1 compressor frozen | TIE at seeds 42 / 43; seed 44 trained, freeze held, evals queued | read 190218 (~18:00): decide on 2 of 3 seeds; still a tie → BLOCKED.md (no fourth seed) |
| A2 connector frozen | both seeds trained, freeze held; 3 of 8 evals done | read 190020 (~19:30): Validated / Failed / Tie; on a Tie launch `ft_freeze.sh connfrozen 44` (vs 188652) + its 4 evals + check (+ preview later) |
| after-gate arms for a VALIDATED gate | void: gate 0 failed | — |

## Next
- Wake when BOTH paired tests (A1 seed 44 190218 and A2 190020) have left the queue, max 12 h: read both, apply the split-test rules (A1 on 2 of 3 seeds; A2 on 2 seeds, third seed on a tie), record verdicts, refresh the page.

## Deliverable — judge accuracy ↑ (Qwen3-8B rubric judge), n = 1,500 per set; Δ = paired difference arm − reference of the same seed [95% CI] (jobs 188650, 188701, 189919)
| model | what trains | rate (tokens per slot; slots per page) | documents ↑ s42 / s43 / s44 | photos ↑ s42 / s43 / s44 | MRQA text ↑ s42 / s43 / s44 | HotpotQA ↑ s42 / s43 / s44 |
|---|---|---|---|---|---|---|
| R reference (184778 / 184780 re-evaluated 11-11; 188652) | compressor LoRA + merger + connector + decoder LoRA | 8; 123 per 1M-px page (980 visual tokens) | .610 / .615 / .620 | .673 / .685 / .671 | .718 / .724 / .721 | .603 / .593 / .603 |
| F frozen (188582 / 188583 / 188651) | decoder LoRA only | same | .582 / .580 / .579 | .664 / .672 / .675 | .695 / .697 / .697 | .575 / .571 / .577 |
| A1 compressor frozen (189094 / 189445 / 190124) | connector + decoder LoRA | same | .581 / .576 / queued | .673 / .671 / queued | .695 / .693 / queued | .583 / .583 / queued |
| A2 connector frozen (189842 / 189899) | compressor LoRA + merger + decoder LoRA | same | queued / queued | .667 / queued | judging / **.727** | queued / **.600** |
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

Step-100 previews (judge accuracy ↑ on the first 300 documents of `mm_docqa_1500_1m`, provisional, never gates; the "final, same 300 Q" column is wake-computed with jq, not a grid cell):
| model | at .23 epoch (checkpoint-100) s42 / s43 / s44 | final (1.0 epoch), same 300 Q, s42 / s43 / s44 |
|---|---|---|
| R reference (184778 / 184780 / 188652) | — (no checkpoint-100 kept) | .610 / .610 / .597 |
| F frozen (preview 188594; finals 188596 / 188598 / 188653) | .583 / — / — | .590 / .593 / .583 |
| A1 compressor frozen (previews 189434 / — / 190217; finals 189095 / 189446 / queued) | .593 / — / .597 | .587 / .587 / queued |
| A2 connector frozen (previews 189918 / 189973) | .593 / .623 / — | queued / queued / — |

Held-out loss ↓ (64 rows, no paired test; a health signal, not a gate), at .23 / .46 / .69 / .92 / 1.0 epoch:
| run | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| R reference (184778 / 184780 trainer_state; 188652 log) | .513 / .455 / .393 / .394 / .414 | .471 / .439 / .411 / .441 / .375 | .468 / .414 / .421 / .417 / .401 |
| F frozen (188582 / 188583 / 188651 logs) | .455 / .444 / .438 / .434 / .435 | .442 / .431 / .416 / .415 / .413 | .456 / .444 / .427 / .428 / .427 |
| A1 compressor frozen (189094 / 189445 / 190124) | .459 / .449 / .437 / .435 / .435 | .464 / .449 / .431 / .429 / .428 | .451 / .436 / .421 / .422 / .423 |
| A2 connector frozen (189842 / 189899 logs) | .438 / .386 / .397 / .407 / .434 | .489 / .405 / .461 / .411 / .399 | — |

## Module movement (entries changed %, per module, vs the stage-2 model 184777)
| module | F smoke s42 (188578) | unfrozen smoke s42 (188579) | A1 smoke s42 (189066) | A2 smoke s42 (189740) | F full s42 / s43 / s44 (188584 / 188585 / 188699) | R full s44 (188700) | A1 full s42 / s43 / s44 (189099 / 189450 / 190129) | A2 full s42 / s43 (189843 / 189904) |
|---|---|---|---|---|---|---|---|---|
| decoder lora_A / lora_B | 30.6 / 76.3 | 26.7 / 75.8 | 30.6 / 76.2 | 26.9 / 75.8 | 14.7 / 66.7, 14.7 / 66.3, 14.8 / 66.7 | 15.1 / 67.8 | 14.6 / 66.0, 14.7 / 66.4, 14.7 / 66.7 | 14.8 / 67.0, 14.8 / 66.9 |
| compressor lora_A / lora_B | 0.0 / 0.0 | 25.4 / 76.4 | 0.0 / 0.0 | 25.4 / 76.2 | 0.0 / 0.0 (all three) | 16.9 / 74.1 | 0.0 / 0.0 (all three) | 16.5 / 73.6 (.0783 of its norm), 16.6 / 73.8 (.0800) |
| compressor token-adapter group (holds the `<MEM>` row; 1,270M entries, so one row is invisible) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| connector MLP | 0.0 | 25.4 | 28.7 (.0027 of its norm) | 0.0 | 0.0 (all three) | 20.7 (.0073 of its norm) | 19.1 / 19.4 / 19.5 (.0061 / .0061 / .0063 of its norm) | 0.0 (frozen) |
| connector norm gain | 0.0 | 0.0 (bf16 lost update) | 0.0 (bf16 lost update) | 0.0 (frozen) | 0.0 | 0.0 (bf16 lost update) | 0.0 (bf16 lost update) | 0.0 (frozen) |
| merger / vision tower | not measured (no row in the check) | | | | | | | |

## Training cost (A100-80GB) — provisional (node speed confounds time); peak memory from sacct gres/gpumem, recorded as grid probes `gpumem_gb`
| run | s/step | wall | peak GPU memory | trainable params |
|---|---|---|---|---|
| F frozen s42 / s43 / s44 (188582 / 188583 / 188651) | 71.5 / 76.1 / 71.5 | 8.7 / 9.3 / 8.7 h | 49.9 / 49.9 / 50.0 GB | 155,978,240 (decoder only) |
| R reference s42 / s43 / s44 (184778 / 184780 / 188652) | 90.3 / 113.2 / 95.1 | 11.0 / 13.8 / 11.6 h | — / — / 34.0 GB | 307,771,392 |
| A1 smoke s42 (189064), 20 steps | 81.2 overall (~70.5 steady) | 29 min | 49.8 GB | 176,958,976 (decoder + connector) |
| A1 full s42 / s43 / s44 (189094 homer-013 / 189445 homer-020 / 190124 homer-019) | 71.5 / 91.7 / 71.5 | 8.7 / 11.2 / 8.7 h | 50.2 / 50.1 / 50.1 GB | 176,958,976 |
| A2 smoke s42 (189739, homer-019), 20 steps | 98.9 overall (~88 steady) | 33 min | 31.4 GB | 286,790,656 (decoder + compressor LoRA + merger) |
| A2 full s42 / s43 (189842 homer-019 / 189899 homer-020) | 92.2 / 93.3 | 11.2 / 11.3 h | 33.8 / 33.7 GB | 286,790,656 |

## Proposed text for the owner's chats (gate 0 is decided; split test still open)
- docs/training_plan.md, gate 0 of "Standing proposal": "FAILED (campaign freeze, 3 seeds, n = 1,500 per set, paired judge). A decoder-only finetune on frozen `<MEM>`s loses 2.8 / 3.5 / 4.1 pt on documents and 2.3 / 2.7 / 2.5 pt on MRQA text (CI below 0 at every seed), 2.3-2.8 pt on HotpotQA; photos tie. Provisional (node-confounded time, no code check of the memory cause): training is faster (71.5-91.7 vs 90-113 s/step) but peaks at ~50 vs ~34 GB GPU memory. Split test: freezing the compressor alone (connector + decoder train) shows no difference from freezing both (A1 − F CIs within ±1.4 pt on documents and MRQA, 2 seeds; third seed trained, evals queued); connector-frozen arm trained at 2 seeds, gate test pending."
- HANDOFF.md thread 7: "Freezing compressor + connector at finetune fails gate 0 (documents −3.5 pt mean of 3 seeds, MRQA −2.5, photos ~0). Freezing only the compressor shows no difference from freezing both (2 seeds). `PISCOConfig.freeze_part` (compressor | connector) added by the freeze campaign for the split test; diff in campaign_freeze/patches/split_freeze.diff."

## Notes this wake
- Woken on schedule (the 6 h maximum of the 05:30 wait). No ✗ of this campaign in the brief; no owner file. Verifier findings of the 05:30 STATE.md: all marked provisional there and kept so; nothing unsupported.
- Recorded: probes 190124 (.4229 at 1.0, 71.5 s/step, 50.1 GB), 190217 (judge .5967, n 300), 189902 (.7267, n 1,500), 189903 (.6000, n 1,500); verdict 190129 closed (freeze held); 5 ledger notes.
- Preemptions on `gpu-be` are the bottleneck: the 4 requeued A2 evals were preempted 3-4 times each since 02:00 (sacct -D). Known issue, no action within the law (moving evals to `gpu` would relax a rule).
- Launched nothing (0 of 6): no training is owed until a gate reads.
- Code md5 at 12:00: `eval_mm_docqa.py` dfe32aa0, `eval_mm_4b.sh` 50abc167, `judge_mm_docqa.py` 02bc8748, `paired_judge.py` a3cc13b2, `pisco/model.py` 994cff3d, `ft_freeze.sh` 668268e3.
- `expmon display --exp freeze` still reports "0 table rows" (gripe a95163a1, already +1'd).

## Patches this wake
- None. `pisco/model.py` + `test_freeze.py` from wake 11 are still uncommitted (this loop cannot commit `pisco/`); their diff is `campaign_freeze/patches/split_freeze.diff`, md5 d4412015df7ecd2cfd8408693ec3d5ef.

## Blockers / needs the owner
- None blocking. An owner's chat should commit `pisco/model.py` and `test_freeze.py` (the split-freeze setting): the queued evals of this campaign load checkpoints trained with them from the working tree.
