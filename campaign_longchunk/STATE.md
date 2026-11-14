# STATE — campaign longchunk   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-15 07:00 (woken because the 1,024-token arm's seed-42 finetune wrote its step-100 checkpoint. Copied it, launched its 2 previews, read the finished ownership probe and stage 2, checked the 2,048-token stage 1. No kill, no failure.)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Stage-1 corpus = MS MARCO documents: 43.4% of documents ≥ 1,024 tokens, 20.4% ≥ 2,048 (107,120 of 3.2M sampled) — data job 188358, closed. SmolLM2 rejected (23.8% / 8.3% of 100,000 sampled) — data job 188339, closed.
- The 2,048-token chunk arm FITS on 4 × A100-80GB at batch 3 per GPU: 50/50 steps, ~72 GB per GPU, 58.6 s per step — job 188420, closed.
- The 1,024-token chunk arm FITS at batch 6 per GPU: 50/50 steps, ~71 GB per GPU, 42.1 s per step — job 188421, closed.
- Control stage 1 (256-token chunks) FINISHED: 1M documents (5,208 steps, 1.0 epoch) in 16:43 h, exit 0 — job 188422, closed. Held-out loss .801 / .877 / 1.007 at steps 2,000 / 4,000 / 5,208 (64 held-out rows).
- Control stage-1 slot ownership = .972 on text (3,053 test slots; gate ≥ .95; SmolLM2 reference .977) — job 188470, closed.
- Control stage 2 (images) FINISHED: 100k rows (1,562 steps, 1.0 epoch) in 19:12 h, exit 0; held-out loss .785 at the end — job 188467, closed.
- Control finetunes FINISHED: 28k rows each (437 steps, 1.0 epoch), exit 0; held-out loss at the end .367 (seed 42, job 188468) / .370 (seed 43, job 188469), closed.
- Control gating scores, all 6 from FINISHED eval jobs (exit 0), judge accuracy ↑ (Qwen3-8B rubric judge, n = 1,500 questions per cell, rate 8 tokens per slot, 0 unparseable verdicts each). These are the comparison points; no arm has a score yet, so nothing is decided:

  | set | seed 42 | seed 43 |
  |---|---|---|
  | grouped HotpotQA g10 (one long chunk per question) | .611 — job 188707, homer-019 | .607 — job 188710, homer-019 |
  | documents `mm_docqa_1500_1m` | .615 — job 188708, homer-013 | .603 — job 188711, homer-016 |
  | photos `mm_naturalqa_1500` | .667 — job 188709, homer-014 | .670 — job 188712, homer-014 |

- 1,024-token arm (L1024) stage 1 FINISHED: 1M documents (5,208 steps, 1.0 epoch), exit 0, 60.2 h on 4 A100 (241 GPU-h, 41.6 s per step), ≈ 1.04 B text tokens (estimate, 2.3 × the control). Held-out loss .611 / .612 / .850 at steps 2,000 / 4,000 / 5,208; train loss .79, no NaN → no collapse — job 188466, closed.
- L1024 stage-1 slot ownership = .969 on text (3,053 test slots) ≥ .95 gate (control .972) → health PASS — job 189691, closed.
- L1024 stage 2 (images) FINISHED: 100k rows (1,562 steps, 1.0 epoch) in 19:08 h, exit 0; held-out loss .818 at the end — job 189692, closed.

**Open, and running**
- provisional (64 held-out rows, not a gate): the L1024 model starts the image/QA stages a little behind the control on the SAME recipe and rows. Stage 2 held-out loss .818 vs .785 at the end, higher at all 7 eval points by .03–.06 (189692 vs 188467). Finetune seed 42 at step 100 (6,400 of 28k rows): held-out .647 vs .465 (189694 vs 188468); train loss .05–.08 higher over the first 12 logs. Both stages use 256-token text chunks for every arm, as the law says. Whether this costs judge accuracy is what the previews below answer.
- provisional (unpaired, other corpus — never a comparison under the law): the MS MARCO control scores like the SmolLM2 reference (documents .615 / .603 vs .599 / .618; photos .667 / .670 vs .671 / .671; g10 .611 / .607 vs reference seed 42 .617). All gaps are inside the 2-pt noise band.
- provisional (n = 300 per cell, pairing points only): the control's step-100 previews: g10 .603 / .570, documents .597 / .603 (seed 42 / 43) — jobs 188981 / 188983 / 188980 / 188982.
- provisional: all three finished stage-1 runs (SmolLM2 reference, control, L1024) show a held-out loss jump at the LAST step only; the control alone also rises earlier (+.077 by step 4,000), L1024 does not (.611 → .612). Cause untested; 64 held-out rows.
- L1024 seed-42 PREVIEWS (step 100, 300 Q each, judge chained) — documents 190574 RUNNING homer-017, g10 190575 RUNNING homer-020, since 07:05. Pair against 188980 / 188981.
- L1024 finetune seed 42 — 189694 RUNNING homer-012, step 119 of 437 at 07:00 (≈ 7,600 of 28k rows), 95.8 s per step → ends ≈ 15:30.
- L1024 finetune seed 43 — 189695 PENDING (afterany:189694) → ≈ 15:30 to ≈ 11-16 03:00; its checkpoint-100 ≈ 18:10, deleted at step 300 ≈ 23:30.
- L1024 gating evals chained: seed 42 g10 190270 / documents 190271 / photos 190272 (after 189694); seed 43 190273 / 190274 / 190275 (after 189695).
- 2,048-token arm (L2048) stage 1 — 189693 RUNNING homer-019 (4 A100), step 1,376 of 5,208 at 07:00 (≈ 264k of 1M documents), 55.6 s per step → ends ≈ 11-17 18:00 (limit 4 d 10 h = 11-18 18:40). Train loss 2.18 → 1.89 over 1,300 steps, then a sharp drop to 1.01 at step ~1,350 — the same drop L1024 made at step ~800 (1.89 → .93). No NaN, no collapse. Held-out checks: step 2,000 ≈ 16:40 today, step 4,000 ≈ 11-16 23:30.

## Owned jobs (jobid | arm / stage | state | progress | eta | note)
| job | arm / stage | state | progress | eta | note |
|---|---|---|---|---|---|
| 190574 | L1024 s42 preview, documents | RUNNING gpu-be homer-017 | — | ≈ 07:45 | = control preview 188980's command, only the checkpoint changes: `PX=1003520 eval_mm_4b.sh expQ/longchunk/ft_cml1024_s42_step100/checkpoint-100 pisco 300 eval_data/mm_docqa_1500_1m`, 0:40:00 |
| 190575 | L1024 s42 preview, g10 | RUNNING gpu-be homer-020 | — | ≈ 07:30 | = 188981's command: `CML=4096 TOPK=10 … eval_data/text_hotpotqa_1500_g10`, 0:26:00 |
| 189693 | L2048 stage 1 | RUNNING gpu homer-019, 4 GPUs | step 1,376 / 5,208 (≈ 264k docs) | ≈ 11-17 18:00 | `pt_text_4b_long.sh 1000000 2048 4096 3 16`; out `expQ/longchunk/pt_text_1M_cml2048_189693` |
| 189694 | L1024 finetune s42 | RUNNING gpu homer-012 | step 119 / 437 | ≈ 15:30 | control 188468's argv + env `FT_DATA`; out `expQ/longchunk/ft_cml1024_s42`; checkpoint-100 preserved |
| 189695 | L1024 finetune s43 | PENDING afterany:189694 | — | ≈ 15:30 → 11-16 03:00 | same + `+hf_training.seed=43`; out `expQ/longchunk/ft_cml1024_s43`. **Copy its checkpoint-100 before step 300 (≈ 18:10 → 23:30)** |
| 190270 / 190271 / 190272 | L1024 s42 evals g10 / documents / photos | PENDING afterok:189694 | — | ≈ 15:30+ | = control cells 188707 / 188708 / 188709, only the checkpoint changes |
| 190273 / 190274 / 190275 | L1024 s43 evals g10 / documents / photos | PENDING afterok:189695 | — | ≈ 11-16 03:00+ | = control cells 188710 / 188711 / 188712 |

All evals: `--partition gpu-be --exclude 'homer-[005-009]'` (same-node rule: image cells pair only between agreeing nodes; record the node of the attempt that wrote each file with `sacct -j <id> -D -X -o NodeList,Start,End`).
Code md5, re-checked 2026-11-15 07:00 (unchanged since the control evals): `eval_mm_4b.sh` 50abc167, `scripts/eval_mm_docqa.py` dfe32aa0, `scripts/judge_mm_docqa.py` 02bc8748, `pisco/model.py` 994cff3d, `pt_mm_4b.sh` 3b63ee99, `ft_mm_4b.sh` 4522a9fc. Sibling chats have uncommitted edits in these files: if an md5 differs when an L1024 eval RUNS, that cell is not paired until both sides ran the same code.
Preserved step-100 checkpoints (model files only, 3.1 GB each, `diff -rq` shows only optimizer.pt / scheduler.pt / training_args.bin): `expQ/longchunk/ft_cml256_s42_step100`, `ft_cml256_s43_step100`, `ft_cml1024_s42_step100` (2026-11-15 07:02).
Resolved-config diffs against the control: stage 1 L1024 188466 and L2048 189693 differ ONLY in `per_device_train_batch_size` (24 → 6 / 3), `gradient_accumulation_steps` (2 → 8 / 16), `compressor_max_length` (256 → 1024 / 2048), `decoder_max_length` (512 → 2048 / 4096); effective batch 192 documents held. Stage 2 189692 vs 188467: only `model_name_or_path`. Finetune 189694 vs 188468: only `model_name_or_path`, plus `freeze_part: null` in the model config — a no-op field (known_issues.md; trainable parameters identical, 307,771,392).

## Adopted jobs (found running in this project, not launched by the campaign — never killed, never duplicated)
(none. The freeze and probes campaigns are stood down; 190481 `fs_merge_retrieve.sh` is not this campaign's.)

## Hand-offs
- 5f4e2f → the pisco chats (2026-11-13 07:35): make `eval_mm_4b.sh` skip generation and only judge when the result file already holds every row. Not blocking.

## Open arms and the next decision for each
| arm | chunk / text kept per doc (tokens) | text tokens at 1M docs | stage-1 cost | status | next decision |
|---|---|---|---|---|---|
| C256 control | 256 / 512 | ≈ 0.45 B | 67 GPU-h (done) | all training and all 6 gating evals done | g1 + MRQA (4 cells) only if an arm adopts |
| L1024 | 1,024 / 2,048 | ≈ 1.04 B (estimate) | 241 GPU-h (done) | ownership PASS; stage 2 done; finetune s42 running, s43 chained; 2 previews running; 6 gating evals chained | read the previews; copy the s43 checkpoint-100 and preview it; gate after the 6 evals (≈ 11-16 06:00) |
| L2048 | 2,048 / 4,096 | ≈ 1.30 B | ≈ 326 GPU-h (projected from 55.6 s/step) | stage 1 running, healthy | held-out check at step 2,000; its downstream chain is submitted once stage 1 is near its end |
| SmolLM2 reference | 256 / 512 | ≈ 0.39 B | done | reported as a row only | — |

Known confound (report, do not fix): the long arms see 2.3× / 2.9× more text tokens than the control at the same 1M documents.

## Results table (deliverable; judge accuracy ↑, n = 1,500 Q per set unless marked; rate 8 tokens per slot)
| row | corpus | chunk (tokens) | slots per chunk | ownership ↑ | g10 s42 / s43 | documents s42 / s43 | photos s42 / s43 |
|---|---|---|---|---|---|---|---|
| SmolLM2 reference (183814 / 183816) | SmolLM2 | 256 | 33 | .977 | .617 / — | .599 / .618 | .671 / .671 |
| C256 control (188468 / 188469) | MS MARCO | 256 | 33 | .972 | .611 / .607 | .615 / .603 | .667 / .670 |
| L1024 | MS MARCO | 1,024 | 129 | .969 | — | — | — |
| L2048 | MS MARCO | 2,048 | 257 | — | — | — | — |
| uncompressed decoder | — | — | — | — | — | .809 | .717 |

Paired difference to the control (arm − C256, same seed, `scripts/paired_judge.py`): none yet.

## Budget (38 GPU-days in the law)
- Spent ≈ 476 GPU-h = 19.8 GPU-days: fit tests ≈ 7 + control stage 1 66.9 + ownership probes 0.1 + L1024 stage 1 240.8 + control stage 2 19.2 + control finetunes 23.9 + previews and control eval attempts ≈ 6.6 + L1024 stage 2 19.1 + L2048 stage 1 so far 89.2 + L1024 finetune s42 so far 3.2.
- Projected left: L2048 stage 1 ≈ 237 (59.2 h × 4) + L1024 downstream ≈ 42 (finetunes ≈ 20, evals and previews ≈ 22) + L2048 downstream ≈ 61 → total ≈ 816 GPU-h = 34.0 GPU-days ≤ 38. Slack ≈ 4 GPU-days: a tie's third seed (≈ 3) fits.
- Train cap (CAMP_MAX_TRAIN = 2 running at once): L2048 stage 1 + ONE L1024 finetune at a time (seed 43 waits for seed 42). Evals do not count.

## Blockers / needs the owner
- Nothing.
- Next wake: when `expQ/longchunk/ft_cml1024_s43/checkpoint-100/trainer_state.json` exists (≈ 18:10), max 14 h (21:00, still before step 300 if the seed-43 run started late). A failure of any chained job wakes earlier. At that wake: copy the s43 checkpoint-100 to `expQ/longchunk/ft_cml1024_s43_step100/checkpoint-100` (model files only, `diff -rq`), launch its 2 previews (same commands as 190574 / 190575 on the s43 copy; pair against 188982 / 188983); pair the s42 previews 190574 / 190575 against 188980 / 188981 (`scripts/paired_judge.py`, n = 300, provisional); check the s42 gating evals 190270–190272 started on the same code md5; `expmon probe` L2048 at step 2,000.
