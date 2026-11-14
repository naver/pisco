# STATE — campaign freeze   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-14 16:00 tick (wake 27; woken because both paired tests it waited on — compressor-frozen seed 44 and the connector-frozen gate — left the queue)

## TL;DR — where this leaves the question (readable as an email)
**The campaign's question is answered. The question-answering finetune cannot be decoder-only: the compressor's own finetune carries ~3 pt on documents and text. The connector, however, can stay frozen at no cost.** Nothing is left to launch; NEXT = `done`.

**Settled**
- **Gate 0 FAILED on three seeds (compressor + connector frozen, only the decoder LoRA trains).** Paired judge accuracy, frozen − reference of the same seed, n = 1,500 per set, jobs 188650 (seeds 42 / 43) and 188701 (seed 44), `ALL` rows:
  - documents −.028 [−.049, −.007], −.035 [−.057, −.014], −.041 [−.062, −.021] → FAILED (≤ −3 pt with the CI below 0 at seeds 43 and 44);
  - MRQA text −.023 [−.040, −.008], −.027 [−.043, −.009], −.025 [−.041, −.009];
  - photos −.009 [−.026, +.008], −.013 [−.030, +.004], +.005 [−.013, +.021] → no difference shown;
  - HotpotQA (reported) −.028 [−.045, −.013], −.023 [−.041, −.006], −.026 [−.043, −.010].
- **Split arm A1 (compressor frozen; connector + decoder LoRA train) FAILED on 2 of 3 seeds** — jobs 189919 (seeds 42 / 43) and 190218 (seed 44, exit 0, new), n = 1,500 per set, A1 − reference:
  - documents −.029 [−.050, −.009], −.039 [−.061, −.017], **−.035 [−.055, −.014] p .002** → ≤ −3 pt with the CI below 0 at seeds 43 and 44 = the law's "Failed" (one gating set fails at ≥ 2 of 3 seeds);
  - MRQA text −.023 [−.039, −.007], −.031 [−.047, −.014], **−.007 [−.023, +.009]**;
  - photos .000 [−.017, +.016], −.013 [−.031, +.004], **+.007 [−.009, +.023]**;
  - HotpotQA (reported) −.019 [−.036, −.003], −.011 [−.029, +.007], **−.025 [−.041, −.009]**.
- **Split arm A2 (connector frozen; compressor LoRA + merger + decoder LoRA train) VALIDATED at seeds 42 and 43** — job 190020 (exit 0, new), n = 1,500 per set, A2 − reference: every gating set ≥ −2 pt and no CI wholly below 0 at both seeds:
  - documents +.023 [+.003, +.041] p .030 / −.006 [−.026, +.014];
  - photos −.006 [−.021, +.010] / −.013 [−.030, +.003];
  - MRQA text +.001 [−.015, +.016] / +.003 [−.013, +.018];
  - HotpotQA (reported) −.004 [−.020, +.013] / +.007 [−.009, +.023].
- **The prediction written before any split number (CAMPAIGN.md, 2026-11-12 23:40: "A1 fails like F, A2 ties R") holds.** Meaning, per the law: the compressor's own finetune carries the ~3 pt; a decoder-only stage B on cached `<MEM>`s costs them; the `<MEM>`-caching hand-off reserved for a validated gate is NOT triggered.
- **Where the gate-0 loss sits (reported pairs, same jobs):**
  - A2 − F: documents +.051 [+.029, +.071] / +.029 [+.008, +.051]; MRQA +.024 [+.009, +.040] / +.029 [+.014, +.047]; HotpotQA +.024 [+.009, +.039] / +.029 [+.013, +.047]; photos +.003 / −.001 (seeds 42 / 43, job 190020). Training the compressor recovers the loss.
  - A2 − A1: documents +.052 [+.031, +.073] / +.033 [+.012, +.054]; MRQA +.024 [+.008, +.040] / +.033 [+.017, +.050]; photos −.006 / .000; HotpotQA +.015 [−.001, +.031] / +.017 [+.000, +.034] (job 190020).
  - provisional: A1 − F (training the connector on top of a frozen compressor): documents −.001 / −.004 / +.006, photos +.009 / −.001 / +.003, MRQA .000 / −.004 / **+.017 [+.005, +.031] p .010**, HotpotQA +.009 / +.012 / +.001 (seeds 42 / 43 / 44; jobs 189919, 190218). No recovery on documents at any seed; the seed-44 MRQA gain is one seed out of three.
- **Same eval code on both sides of every pair:** md5 at 16:00 `eval_mm_docqa.py` dfe32aa0, `eval_mm_4b.sh` 50abc167, `judge_mm_docqa.py` 02bc8748, `paired_judge.py` a3cc13b2 = the gate-0 versions; every new eval file has 1,500 records, 0 unparseable judge verdicts, `image_max_pixels` 1,003,520 and its own checkpoint path; all ran on homer-017 / 019 (outside the excluded homer-[005-009]).
- **The freeze held in every full run checked** (1.0 epoch, 27,936 rows each): F seeds 42 / 43 / 44 (188584 / 188585 / 188699) every compressor and connector row 0.0%; A1 seeds 42 / 43 / 44 (189099 / 189450 / 190129) every compressor row 0.0%, connector MLP 19.1 / 19.4 / 19.5%; A2 seeds 42 / 43 (189843 / 189904) connector MLP and norm gain 0.0%, compressor lora_B 73.6 / 73.8%. The seed-44 reference moved compressor lora_B 74.1% and the connector MLP 20.7% (188700): the check sees movement when there is some. It has no row for the merger or the vision tower and cannot see the one `<MEM>` row.
- **In this bf16 recipe the connector norm gain never moves, even when trainable**: 0.0% after 20 steps (188579, 189066) and after a whole epoch (188700; A1 189099 / 189450 / 190129), value .46094 each time (bf16 lost update).
- **Split-freeze setting**: `PISCOConfig.freeze_part = compressor | connector` (default unset) in `pisco/model.py` (md5 994cff3d) + 5 new `test_freeze.py` cases; staged (189060) and installed (189063) test runs pass 12/12 + 33/33 + 10/10. Still uncommitted: the working-tree diff equals `campaign_freeze/patches/split_freeze.diff` (md5 d4412015df7ecd2cfd8408693ec3d5ef, re-checked 16:00). Hand-off 43ec9b asks the project chats to commit it.
- provisional: training the compressor costs GPU memory LESS than freezing it: compressor frozen 49.7-50.2 GB (8 runs: 188528, 189064, 188582 / 188583 / 188651, 189094 / 189445 / 190124), compressor training 31.4-34.0 GB (5 runs: 188534, 189739, 189842 / 189899, 188652). Consistent with the eval-mode frozen compressor running without gradient checkpointing (`pisco/model.py:1294-1300`; HF checkpoints a layer only when `self.training`); not verified. Time is node-confounded (see the cost table).
- provisional: F and A1 end at a HIGHER held-out loss than the reference of the same seed (F .435 / .413 / .427, A1 .435 / .428 / .423 vs R .414 / .375 / .401; 64 rows, no test); A2 ends at .434 / .399, between them.
- provisional: the gate-0 loss is spread over the data sources: 39 of 42 per-source differences negative over three seeds (188650, 188701). A sign count with no test.

**Open, and running**
- (none) — every arm in the law is decided. No campaign job is queued or running.

## Owned jobs
None in the queue. Everything this campaign submitted has ended and carries a verdict (`expmon campaign report --section leaves`: Open = none).
Trainings in flight: 0 of CAMP_MAX_TRAIN 2.
provisional (estimate, not summed from sacct): ~125 GPU-h used of 192 — trainings ~104 (gate 0 F × 3 + reference seed 44 ~52, A1 × 3 ~29, A2 smoke + × 2 ~23) and ~50 eval / preview cells at .3-.5 h each ~20. Nothing more is planned.

## Adopted jobs
(none — the other jobs in the queue belong to the sibling campaign `longchunk`)

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| F compressor + connector frozen | FAILED (documents, 2 of 3 seeds) | — |
| A1 compressor frozen | FAILED (documents, 2 of 3 seeds) | — |
| A2 connector frozen | VALIDATED (2 of 2 seeds) | — (adoption as the default recipe is the owner's call; it saves no time or memory: 92 s/step, 34 GB, like the reference) |
| after-gate arms for a validated gate 0 | void | — |

## Next
- Nothing to launch. NEXT = `done`.
- Hand-off 43ec9b (to the pisco project chats): commit the split-freeze code; write the result into `docs/training_plan.md` and HANDOFF.md thread 7 (text below).
- Ideas for a successor campaign (owner's call; each changes the data or the mission, which this law forbids):
  - Stage A / stage B proper: take the reference finetune's QA-trained compressor (184778), freeze it, and train only the decoder on NEW data or a new task, against a full finetune on that data. Gate 0 only showed that the compressor must see question-answering training once — not that it must see every later dataset.
  - Make the connector-frozen recipe the default: it ties the reference and removes one moving part, but it buys no speed or memory.

## Deliverable — judge accuracy ↑ (Qwen3-8B rubric judge), n = 1,500 per set; paired Δ = arm − reference of the same seed [95% CI]
| model | what trains | rate (tokens per slot; slots per page) | documents ↑ s42 / s43 / s44 | photos ↑ s42 / s43 / s44 | MRQA text ↑ s42 / s43 / s44 | HotpotQA ↑ s42 / s43 / s44 |
|---|---|---|---|---|---|---|
| R reference (184778 / 184780 re-evaluated 11-11; 188652) | compressor LoRA + merger + connector + decoder LoRA | 8; 123 per 1M-px page (980 visual tokens) | .610 / .615 / .620 | .673 / .685 / .671 | .718 / .724 / .721 | .603 / .593 / .603 |
| F frozen (188582 / 188583 / 188651) | decoder LoRA only | same | .582 / .580 / .579 | .664 / .672 / .675 | .695 / .697 / .697 | .575 / .571 / .577 |
| A1 compressor frozen (189094 / 189445 / 190124) | connector + decoder LoRA | same | .581 / .576 / .585 | .673 / .671 / .678 | .695 / .693 / .714 | .583 / .583 / .578 |
| A2 connector frozen (189842 / 189899) | compressor LoRA + merger + decoder LoRA | same | .633 / .609 / — | .667 / .671 / — | .719 / .727 / — | .599 / .600 / — |
| uncompressed decoder (earlier runs, same questions) | — | 1 (raw tokens) | .809 | .717 | .807 | — |

| paired Δ ↑ [95% CI], n = 1,500 | documents | photos | MRQA text | HotpotQA (reported) | gate read |
|---|---|---|---|---|---|
| F − R seed 42 | −.028 [−.049, −.007] | −.009 [−.026, +.008] | −.023 [−.040, −.008] | −.028 [−.045, −.013] | |
| F − R seed 43 | −.035 [−.057, −.014] | −.013 [−.030, +.004] | −.027 [−.043, −.009] | −.023 [−.041, −.006] | |
| F − R seed 44 | −.041 [−.062, −.021] | +.005 [−.013, +.021] | −.025 [−.041, −.009] | −.026 [−.043, −.010] | F FAILED |
| A1 − R seed 42 | −.029 [−.050, −.009] | .000 [−.017, +.016] | −.023 [−.039, −.007] | −.019 [−.036, −.003] | |
| A1 − R seed 43 | −.039 [−.061, −.017] | −.013 [−.031, +.004] | −.031 [−.047, −.014] | −.011 [−.029, +.007] | |
| A1 − R seed 44 | −.035 [−.055, −.014] | +.007 [−.009, +.023] | −.007 [−.023, +.009] | −.025 [−.041, −.009] | A1 FAILED |
| A2 − R seed 42 | +.023 [+.003, +.041] | −.006 [−.021, +.010] | +.001 [−.015, +.016] | −.004 [−.020, +.013] | |
| A2 − R seed 43 | −.006 [−.026, +.014] | −.013 [−.030, +.003] | +.003 [−.013, +.018] | +.007 [−.009, +.023] | A2 VALIDATED |
| A1 − F seed 42 / 43 / 44 | −.001 / −.004 / +.006 | +.009 / −.001 / +.003 | .000 / −.004 / +.017 | +.009 / +.012 / +.001 | reported |
| A2 − F seed 42 / 43 | +.051 / +.029 | +.003 / −.001 | +.024 / +.029 | +.024 / +.029 | reported |
| A2 − A1 seed 42 / 43 | +.052 / +.033 | −.006 / .000 | +.024 / +.033 | +.015 / +.017 | reported |
Discordant pairs (arm-only right / reference-only right), McNemar p — F − R seeds 42 / 43 / 44: documents 111/153 p .011, 109/161 p .002, 103/164 p < .001; photos 78/92, 78/97, 90/83; MRQA 59/94 p .006, 67/107 p .003, 58/95 p .003. A1 − R seeds 42 / 43 / 44: documents 108/152 p .008, 108/166 p .001, 104/156 p .002; photos 83/83, 78/98, 87/76; MRQA 60/95 p .006, 64/110 p .001, 74/85 p .43. A2 − R seeds 42 / 43: documents 133/99 p .030, 110/119 p .60; photos 66/75 p .50, 69/89 p .13; MRQA 68/67 p 1.0, 71/67 p .80.

Step-100 previews (judge accuracy ↑ on the first 300 documents, n = 300, provisional, never gate; "final, same 300" is wake-computed with jq, query ids checked equal):
| model | at .23 epoch (checkpoint-100) s42 / s43 / s44 | final (1.0 epoch), same 300, s42 / s43 / s44 |
|---|---|---|
| R reference (184778 / 184780 / 188652) | — (no checkpoint-100 kept) | .610 / .610 / .597 |
| F frozen (preview 188594) | .583 / — / — | .590 / .593 / .583 |
| A1 compressor frozen (previews 189434 / — / 190217) | .593 / — / .597 | .587 / .587 / .600 |
| A2 connector frozen (previews 189918 / 189973) | .593 / .623 / — | .637 / .617 / — |

Held-out loss ↓ (64 rows, no paired test; a health signal), at .23 / .46 / .69 / .92 / 1.0 epoch:
| run | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| R reference (184778 / 184780 trainer_state; 188652 log) | .513 / .455 / .393 / .394 / .414 | .471 / .439 / .411 / .441 / .375 | .468 / .414 / .421 / .417 / .401 |
| F frozen (188582 / 188583 / 188651) | .455 / .444 / .438 / .434 / .435 | .442 / .431 / .416 / .415 / .413 | .456 / .444 / .427 / .428 / .427 |
| A1 compressor frozen (189094 / 189445 / 190124) | .459 / .449 / .437 / .435 / .435 | .464 / .449 / .431 / .429 / .428 | .451 / .436 / .421 / .422 / .423 |
| A2 connector frozen (189842 / 189899) | .438 / .386 / .397 / .407 / .434 | .489 / .405 / .461 / .411 / .399 | — |

## Module movement (entries changed %, per module, vs the stage-2 model 184777)
| module | F smoke s42 (188578) | unfrozen smoke s42 (188579) | A1 smoke s42 (189066) | A2 smoke s42 (189740) | F full s42 / s43 / s44 (188584 / 188585 / 188699) | R full s44 (188700) | A1 full s42 / s43 / s44 (189099 / 189450 / 190129) | A2 full s42 / s43 (189843 / 189904) |
|---|---|---|---|---|---|---|---|---|
| decoder lora_A / lora_B | 30.6 / 76.3 | 26.7 / 75.8 | 30.6 / 76.2 | 26.9 / 75.8 | 14.7 / 66.7, 14.7 / 66.3, 14.8 / 66.7 | 15.1 / 67.8 | 14.6 / 66.0, 14.7 / 66.4, 14.7 / 66.7 | 14.8 / 67.0, 14.8 / 66.9 |
| compressor lora_A / lora_B | 0.0 / 0.0 | 25.4 / 76.4 | 0.0 / 0.0 | 25.4 / 76.2 | 0.0 / 0.0 (all three) | 16.9 / 74.1 | 0.0 / 0.0 (all three) | 16.5 / 73.6, 16.6 / 73.8 |
| compressor token-adapter group (holds the `<MEM>` row; one row is invisible in it) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| connector MLP | 0.0 | 25.4 | 28.7 | 0.0 | 0.0 (all three) | 20.7 | 19.1 / 19.4 / 19.5 | 0.0 (frozen) |
| connector norm gain | 0.0 | 0.0 (bf16 lost update) | 0.0 (bf16 lost update) | 0.0 (frozen) | 0.0 | 0.0 (bf16 lost update) | 0.0 (bf16 lost update) | 0.0 (frozen) |
| merger / vision tower | not measured (no row in the check) | | | | | | | |

## Training cost (A100-80GB) — provisional (node speed confounds time); peak memory = sacct gres/gpumem (grid probes `gpumem_gb`)
| run | s/step | wall | peak GPU memory | trainable params |
|---|---|---|---|---|
| R reference s42 / s43 / s44 (184778 / 184780 / 188652) | 90.3 / 113.2 / 95.1 | 11.0 / 13.8 / 11.6 h | — / — / 34.0 GB | 307,771,392 |
| F frozen s42 / s43 / s44 (188582 / 188583 / 188651) | 71.5 / 76.1 / 71.5 | 8.7 / 9.3 / 8.7 h | 49.9 / 49.9 / 50.0 GB | 155,978,240 (decoder only) |
| A1 full s42 / s43 / s44 (189094 homer-013 / 189445 homer-020 / 190124 homer-019) | 71.5 / 91.7 / 71.5 | 8.7 / 11.2 / 8.7 h | 50.2 / 50.1 / 50.1 GB | 176,958,976 (decoder + connector) |
| A2 full s42 / s43 (189842 homer-019 / 189899 homer-020) | 92.2 / 93.3 | 11.2 / 11.3 h | 33.8 / 33.7 GB | 286,790,656 (decoder + compressor LoRA + merger) |

## Proposed text for the owner's chats (hand-off 43ec9b)
- docs/training_plan.md, "Standing proposal", gate 0: "FAILED (campaign freeze, 2026-11-14; paired Qwen3-8B judge, n = 1,500 per set). A decoder-only finetune on frozen `<MEM>`s loses 2.8 / 3.5 / 4.1 pt on documents and 2.3-2.7 pt on MRQA text (3 seeds, CI below 0 at each); photos tie. Split test: freezing the compressor alone (connector + decoder train) fails the same way (documents −2.9 / −3.9 / −3.5 pt, 3 seeds); freezing the connector alone (compressor LoRA + merger + decoder train) ties the reference (documents +2.3 / −0.6, photos −0.6 / −1.3, MRQA +0.1 / +0.3 pt, 2 seeds). So the compressor's own question-answering finetune carries the ~3 pt; a stage B that trains only the decoder on cached `<MEM>`s costs them. Open: whether a compressor that has had one QA finetune can then stay frozen for later data."
- HANDOFF.md thread 7: "Freeze campaign closed 2026-11-14: decoder-only finetune fails (documents −3.5 pt mean of 3 seeds, MRQA −2.5, photos ~0); freezing only the compressor fails the same way; freezing only the connector ties the reference (2 seeds). `PISCOConfig.freeze_part` (compressor | connector) added by the campaign; diff in campaign_freeze/patches/split_freeze.diff."

## Notes this wake
- No ✗ of this campaign in the brief; no owner file. Verifier findings on the 12:00 STATE.md: all were already marked provisional; kept so (the A1 − F line now includes seed 44).
- Read 190218 and 190020 (both COMPLETED exit 0); all 10 evals feeding them COMPLETED exit 0 (the 4 requeued A2 evals finished 15:21-15:44).
- Recorded: probes for 9 eval cells (judge, n 1,500); verdicts A1 trainings closed (FAILED), A2 trainings `best` (VALIDATED), both paired tests closed, every eval / preview / module check closed; 3 ledger notes; point "connector-frozen-both-seeds-traine" closed; CAMPAIGN.md marks the split-test prediction as confirmed (tightening: any further arm needs a new gate first and may not be a variant of freezing the compressor side).
- Launched nothing (0 of 6).
- `expmon display --exp freeze` still prints "0 table rows" (gripes a95163a1 / be8057a1 already open; not re-filed).

## Patches this wake
- None.

## Blockers / needs the owner
- None blocking. Owner-side follow-ups are in hand-off 43ec9b (commit + docs) and in "Next" (successor-campaign ideas).
