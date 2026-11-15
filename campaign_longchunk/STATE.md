# STATE — campaign longchunk   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-15 18:30 (woken because the 1,024-token arm's seed-43 finetune wrote its step-100 checkpoint. Copied it, launched its 2 previews, paired the seed-42 finals and previews against the control, recorded the 2,048-token arm at step 2,000. No kill, no failure.)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Stage-1 corpus = MS MARCO documents: 43.4% of documents ≥ 1,024 tokens, 20.4% ≥ 2,048 (107,120 of 3.2M sampled) — data job 188358, closed. SmolLM2 rejected (23.8% / 8.3% of 100,000 sampled) — data job 188339, closed.
- The 2,048-token chunk arm FITS on 4 × A100-80GB at batch 3 per GPU: 50/50 steps, ~72 GB per GPU, 58.6 s per step — job 188420, closed.
- The 1,024-token chunk arm FITS at batch 6 per GPU: 50/50 steps, ~71 GB per GPU, 42.1 s per step — job 188421, closed.
- Control stage 1 (256-token chunks) FINISHED: 1M documents (5,208 steps, 1.0 epoch) in 16:43 h, exit 0 — job 188422, closed. Held-out loss .801 / .877 / 1.007 at steps 2,000 / 4,000 / 5,208 (64 held-out rows).
- Control stage-1 slot ownership = .972 on text (3,053 test slots; gate ≥ .95; SmolLM2 reference .977) — job 188470, closed.
- Control stage 2 (images) FINISHED: 100k rows (1,562 steps, 1.0 epoch) in 19:12 h, exit 0; held-out loss .785 at the end — job 188467, closed.
- Control finetunes FINISHED: 28k rows each (437 steps, 1.0 epoch), exit 0; held-out loss at the end .367 (seed 42, job 188468) / .370 (seed 43, job 188469), closed.
- Control gating scores (comparison points), judge accuracy ↑, Qwen3-8B rubric judge, n = 1,500 questions per cell, rate 8 tokens per slot, 0 unparseable: g10 .611 (188707) / .607 (188710); documents .615 (188708) / .603 (188711); photos .667 (188709) / .670 (188712), seed 42 / 43.
- 1,024-token arm (L1024) stage 1 FINISHED: 1M documents (5,208 steps), exit 0, 60.2 h on 4 A100 (241 GPU-h), ≈ 1.04 B text tokens (estimate, 2.3 × the control). Held-out loss .611 / .612 / .850 at steps 2,000 / 4,000 / 5,208; no collapse — job 188466, closed.
- L1024 stage-1 slot ownership = .969 on text (3,053 test slots) ≥ .95 gate (control .972) → health PASS — job 189691, closed.
- L1024 stage 2 (images) FINISHED: 100k rows (1,562 steps), 19:08 h, exit 0; held-out .818 at the end — job 189692, closed.
- L1024 finetune seed 42 FINISHED: 28k rows (437 steps, 1.0 epoch) in 11:42 h, exit 0; held-out .615 at the end — job 189694 (probe @437).

**Open, and running**
- provisional (ONE seed; the gate needs both): **L1024 seed 42 loses on documents and gains nothing on long text.** Paired against the control seed 42 (`scripts/paired_judge.py`, n = 1,500 Q per set, rate 8, same code md5, each cell one attempt):

  | set (judge accuracy ↑) | L1024 s42 | C256 s42 | arm − control | 95% CI | McNemar p | nodes (arm / control) |
  |---|---|---|---|---|---|---|
  | g10 grouped HotpotQA, one long chunk | .600 (190270) | .611 (188707) | −1.1 pt | [−3.2, +1.2] | .38 | homer-017 / homer-014 |
  | documents `mm_docqa_1500_1m` | .553 (190271) | .615 (188708) | **−6.2 pt** | [−8.6, −3.9] | < .001 | homer-020 / homer-013 |
  | photos `mm_naturalqa_1500` | .670 (190272) | .667 (188709) | +0.3 pt | [−1.5, +2.2] | .79 | homer-011 / homer-014 |

  Documents by source: MP-DocVQA −10.8 [−15.2, −6.4], SlideVQA −3.8 [−7.2, −0.4], InfoVQA −4.0 [−8.0, 0.0]. Seed 42 meets the law's CLOSE condition twice (g10 CI excludes +2 pt; documents ≤ −3 pt with CI below 0). The arm is decided only when seed 43 agrees.
- provisional (64 held-out rows, not a gate): L1024 is behind the control through every image/QA stage on the SAME recipe and rows. Stage 2 held-out .818 vs .785 at the end, higher at all 7 eval points by .03–.07 (189692 vs 188467). Finetune seed 42 held-out higher at all 5 eval points by .18–.25 (.647 / .633 / .581 / .580 / .615 vs .464 / .457 / .385 / .360 / .367, steps 100–437); train loss higher at every sampled log (every 40 steps) by .02–.12. Both stages use 256-token text chunks for every arm, as the law says.
- provisional (n = 300, step 100, previews never gate): L1024 s42 documents .540 (190574) vs control .597 (188980), −5.7 pt [−10.7, −0.7]; g10 .597 (190575) vs .603 (188981), −0.7 [−5.3, +4.0]. The step-100 preview foretold the final documents loss (−5.7 → −6.2).
- provisional (unpaired, other corpus — never a comparison under the law): the MS MARCO control scores like the SmolLM2 reference (documents .615 / .603 vs .599 / .618; photos .667 / .670 vs .671 / .671; g10 .611 / .607 vs .617). All gaps inside the 2-pt band.
- provisional: all three finished stage-1 runs (SmolLM2 reference, control, L1024) show a held-out loss jump at the LAST step only; the control alone also rises earlier (+.077 by step 4,000), L1024 does not (.611 → .612). Cause untested; 64 held-out rows.
- L1024 finetune seed 43 — 189695 RUNNING homer-012 since 15:36, 95.8 s per step → ends ≈ 11-16 03:15. Its gating evals g10 190273 / documents 190274 / photos 190275 are chained (afterok) → ≈ 03:50.
- L1024 seed-43 PREVIEWS (step 100 copy, 300 Q each, judge chained) — documents 190768 RUNNING homer-020, g10 190769 RUNNING homer-011, since 18:40. Pair against control previews 188982 / 188983 (.603 / .570).
- 2,048-token arm (L2048) stage 1 — 189693 RUNNING homer-019 (4 A100), step 2,054 of 5,208 at 18:35 (≈ 394k of 1M documents). Held-out .526 at step 2,000 (Rouge-L .784), train loss .84, grad norm .03 → healthy, no collapse. Pace slowed to ~61 s per step (55.6 earlier) → ends ≈ 11-18 01:00 (limit 11-18 18:42, margin ≈ 17 h). Next held-out check at step 4,000 ≈ 11-17 08:00.

## Owned jobs (jobid | arm / stage | state | progress | eta | note)
| job | arm / stage | state | progress | eta | note |
|---|---|---|---|---|---|
| 189693 | L2048 stage 1 | RUNNING gpu homer-019, 4 GPUs | step 2,054 / 5,208 (≈ 394k docs) | ≈ 11-18 01:00 | `pt_text_4b_long.sh 1000000 2048 4096 3 16`; out `expQ/longchunk/pt_text_1M_cml2048_189693` |
| 189695 | L1024 finetune s43 | RUNNING gpu homer-012 | ≈ step 110 / 437 | ≈ 11-16 03:15 | control 188469's argv + env `FT_DATA`, `+hf_training.seed=43`; out `expQ/longchunk/ft_cml1024_s43`; checkpoint-100 preserved |
| 190768 / 190769 | L1024 s43 previews, documents / g10 | RUNNING gpu-be homer-020 / homer-011 | — | ≈ 19:20 | = 190574 / 190575's commands on `expQ/longchunk/ft_cml1024_s43_step100/checkpoint-100`; 0:40:00 / 0:26:00 |
| 190273 / 190274 / 190275 | L1024 s43 evals g10 / documents / photos | PENDING afterok:189695 | — | ≈ 11-16 03:50 | = control cells 188710 / 188711 / 188712, only the checkpoint changes |

Finished this half-day: 189694 (finetune s42), 190270–190272 (s42 gating evals), 190574 / 190575 (s42 previews), all COMPLETED exit 0, one attempt each.
All evals: `--partition gpu-be --exclude 'homer-[005-009]'` (same-node rule: record the node of the attempt that wrote each file with `sacct -j <id> -D -X -o NodeList,Start,End`).
Code md5, re-checked 2026-11-15 18:30 (unchanged since the control evals): `eval_mm_4b.sh` 50abc167, `scripts/eval_mm_docqa.py` dfe32aa0, `scripts/judge_mm_docqa.py` 02bc8748, `pisco/model.py` 994cff3d, `pt_mm_4b.sh` 3b63ee99, `ft_mm_4b.sh` 4522a9fc. Sibling chats have uncommitted edits in these files: if an md5 differs when an L1024 eval RUNS, that cell is not paired until both sides ran the same code.
Preserved step-100 checkpoints (model files only, 3.1 GB each, `diff -rq` shows only optimizer.pt / scheduler.pt / training_args.bin): `expQ/longchunk/ft_cml256_s42_step100`, `ft_cml256_s43_step100`, `ft_cml1024_s42_step100`, `ft_cml1024_s43_step100` (2026-11-15 18:33).
Resolved-config diffs against the control: stage 1 L1024 188466 and L2048 189693 differ ONLY in `per_device_train_batch_size` (24 → 6 / 3), `gradient_accumulation_steps` (2 → 8 / 16), `compressor_max_length` (256 → 1024 / 2048), `decoder_max_length` (512 → 2048 / 4096); effective batch 192 documents held. Stage 2 189692 vs 188467: only `model_name_or_path`. Finetune 189694 vs 188468: only `model_name_or_path`, plus `freeze_part: null` in the model config — a no-op field (known_issues.md; trainable parameters identical, 307,771,392).

## Adopted jobs (found running in this project, not launched by the campaign — never killed, never duplicated)
(none)

## Hand-offs
- 5f4e2f → the pisco chats (2026-11-13 07:35): make `eval_mm_4b.sh` skip generation and only judge when the result file already holds every row. Not blocking.

## Open arms and the next decision for each
| arm | chunk / text kept per doc (tokens) | text tokens at 1M docs | stage-1 cost | status | next decision |
|---|---|---|---|---|---|
| C256 control | 256 / 512 | ≈ 0.45 B | 67 GPU-h (done) | all training and all 6 gating evals done | g1 + MRQA (4 cells) only if an arm adopts |
| L1024 | 1,024 / 2,048 | ≈ 1.04 B (estimate) | 241 GPU-h (done) | seed 42 meets CLOSE (documents −6.2, g10 −1.1); seed 43 finetune running, evals chained | seed 43 evals (≈ 03:50): if documents ≤ −3 pt with CI below 0 or the g10 CI excludes +2 pt again → `verdict closed` |
| L2048 | 2,048 / 4,096 | ≈ 1.30 B | ≈ 350 GPU-h (projected at 61 s/step) | stage 1 running, healthy, 39% done | if L1024 closes: ask the owner (BLOCKED.md) whether to finish L2048 (≈ 170 GPU-h of stage 1 left at that moment + 61 downstream) as a dose-response check, or stop it. The law gives a kill without collapse to the owner. Otherwise its downstream chain is submitted near the end of stage 1 |
| SmolLM2 reference | 256 / 512 | ≈ 0.39 B | done | reported as a row only | — |

Known confound (report, do not fix): the long arms see 2.3× / 2.9× more text tokens than the control at the same 1M documents.

## Results table (deliverable; judge accuracy ↑, n = 1,500 Q per set; rate 8 tokens per slot)
| row | corpus | chunk (tokens) | slots per chunk | ownership ↑ | g10 s42 / s43 | documents s42 / s43 | photos s42 / s43 |
|---|---|---|---|---|---|---|---|
| SmolLM2 reference (183814 / 183816) | SmolLM2 | 256 | 33 | .977 | .617 / — | .599 / .618 | .671 / .671 |
| C256 control (188468 / 188469) | MS MARCO | 256 | 33 | .972 | .611 / .607 | .615 / .603 | .667 / .670 |
| L1024 (189694 / 189695) | MS MARCO | 1,024 | 129 | .969 | .600 / — | .553 / — | .670 / — |
| L1024 − C256, paired, 95% CI | | | | | −1.1 [−3.2, +1.2] / — | **−6.2 [−8.6, −3.9]** / — | +0.3 [−1.5, +2.2] / — |
| L2048 | MS MARCO | 2,048 | 257 | — | — | — | — |
| uncompressed decoder | — | — | — | — | — | .809 | .717 |

## Budget (38 GPU-days in the law)
- Spent ≈ 535 GPU-h = 22.3 GPU-days: fit tests ≈ 7 + control stage 1 66.9 + ownership probes 0.1 + L1024 stage 1 240.8 + control stage 2 19.2 + control finetunes 23.9 + control previews and eval attempts ≈ 6.6 + L1024 stage 2 19.1 + L2048 stage 1 so far 135.6 + L1024 finetunes 11.7 + 3.0 + L1024 s42 evals and previews 1.4.
- Projected left: L2048 stage 1 ≈ 214 (53.5 h × 4) + L1024 seed 43 rest ≈ 10 + L2048 downstream ≈ 61 → total ≈ 820 GPU-h = 34.2 GPU-days ≤ 38. Slack ≈ 3.8 GPU-days.
- Train cap (CAMP_MAX_TRAIN = 2 running at once): L2048 stage 1 + the L1024 seed-43 finetune. Evals do not count.

## Blockers / needs the owner
- Nothing now. Likely question at the next wake (only if seed 43 confirms the documents loss): finish or stop the 2,048-token arm.
- Next wake: when 190273, 190274 and 190275 have all left the queue (≈ 03:50), max 12 h. A failure wakes earlier. At that wake: pair the s43 previews 190768 / 190769 against 188982 / 188983 (n = 300, provisional); check each s43 eval is a complete single attempt (1,500 rows, judge on every row, 0 unparseable), its node, the code md5; pair s43 finals against 188710 / 188711 / 188712; apply the gate at both seeds; `verdict` L1024; refresh the display; write the L2048 question if L1024 closes.
