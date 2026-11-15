# STATE — campaign longchunk   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-16 04:00 (woken because the 1,024-token arm's seed-43 gating evals ended. Checked them, paired them and its step-100 previews against the control, applied the gate at both seeds → the arm is CLOSED. Marked the refuted prediction in the law. Asked the owner whether the 2,048-token run should finish (BLOCKED.md). No submission, no kill, no failure.)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Stage-1 corpus = MS MARCO documents: 43.4% of documents ≥ 1,024 tokens, 20.4% ≥ 2,048 (107,120 of 3.2M sampled) — data job 188358, closed. SmolLM2 rejected (23.8% / 8.3% of 100,000 sampled) — data job 188339, closed.
- The 2,048-token chunk arm FITS on 4 × A100-80GB at batch 3 per GPU: 50/50 steps, ~72 GB per GPU, 58.6 s per step — job 188420, closed.
- The 1,024-token chunk arm FITS at batch 6 per GPU: 50/50 steps, ~71 GB per GPU, 42.1 s per step — job 188421, closed.
- Control stage 1 (256-token chunks) FINISHED: 1M documents (5,208 steps, 1.0 epoch) in 16:43 h, exit 0 — job 188422, closed. Held-out loss .801 / .877 / 1.007 at steps 2,000 / 4,000 / 5,208 (64 held-out rows).
- Control stage-1 slot ownership = .972 on text (3,053 test slots; gate ≥ .95; SmolLM2 reference .977) — job 188470, closed.
- Control stage 2 (images) FINISHED: 100k rows (1,562 steps, 1.0 epoch), exit 0; held-out loss .785 at the end — job 188467, closed.
- Control finetunes FINISHED: 28k rows each (437 steps, 1.0 epoch), exit 0 — jobs 188468 (seed 42) / 188469 (seed 43), closed.
- Control gating scores (comparison points), judge accuracy ↑, Qwen3-8B rubric judge, n = 1,500 questions per cell, rate 8 tokens per slot, 0 unparseable: g10 .611 (188707) / .607 (188710); documents .615 (188708) / .603 (188711); photos .667 (188709) / .670 (188712), seed 42 / 43 — closed.
- 1,024-token arm (L1024) stage 1 FINISHED: 1M documents (5,208 steps), exit 0, 60.2 h on 4 A100 (241 GPU-h), ≈ 1.04 B text tokens (estimate, 2.3 × the control) — job 188466, closed. Slot ownership .969 on text (3,053 test slots) ≥ .95 → health PASS — job 189691, closed. Stage 2 FINISHED 100k rows, exit 0, held-out .818 at the end — job 189692, closed.
- L1024 finetunes FINISHED: 28k rows each (437 steps), exit 0; held-out .615 (seed 42, job 189694) / .470 (seed 43, job 189695) at the end — closed.
- **L1024 is CLOSED by the law's gate at BOTH seeds** — long chunks in stage 1 did not help long text and hurt document pages. Paired per question against the control of the same seed (`scripts/paired_judge.py`, n = 1,500 Q per set, rate 8 tokens per slot, one complete attempt per cell, 0 unparseable, code md5 equal to the control's):

  | set (judge accuracy ↑) | seed 42: arm / control | arm − control, 95% CI, McNemar p | seed 43: arm / control | arm − control, 95% CI, McNemar p |
  |---|---|---|---|---|
  | g10 grouped HotpotQA, one long chunk | .600 (190270) / .611 (188707) | −1.1 pt [−3.2, +1.2], p .38 | .574 (190273) / .607 (188710) | **−3.3 pt [−5.4, −1.1], p .006** |
  | documents `mm_docqa_1500_1m` | .553 (190271) / .615 (188708) | **−6.2 pt [−8.6, −3.9], p < .001** | .561 (190274) / .603 (188711) | **−4.3 pt [−6.7, −1.8], p < .001** |
  | photos `mm_naturalqa_1500` | .670 (190272) / .667 (188709) | +0.3 pt [−1.5, +2.2], p .79 | .670 (190275) / .670 (188712) | 0.0 pt [−1.9, +1.8], p 1.0 |

  Gate met twice at each seed: the g10 CI excludes +2 pt, AND documents ≤ −3 pt with the CI below 0. Verdicts closed on 189694 / 189695 and the six evals.

**Open, and running**
- 2,048-token arm (L2048) stage 1 — 189693 RUNNING homer-019 (4 A100), step 2,678 of 5,208 at 04:15 (≈ 514k of 1M documents, ≈ 0.67 B text tokens, estimate). Held-out .526 at step 2,000 (Rouge-L .784), no collapse. ~56–61 s per step → ends ≈ 11-17 19:00 (time limit 11-18 18:44). **Owner question in BLOCKED.md: finish it or stop it.**
- provisional (documents pairs not on same-node-checked nodes): node pairs arm / control — seed 42 documents homer-020 / homer-013, seed 43 documents homer-020 / homer-016. All four are outside the excluded homer-005–009, but no node-agreement check exists for them. The g10 close condition (text, no image path) holds at both seeds on its own, so the closed verdict does not depend on the node question.
- provisional (64 held-out rows, not a gate): L1024 is behind the control in every image/QA stage on the same recipe and rows: stage 2 held-out .818 vs .785 at the end (189692 vs 188467); finetune held-out higher at all 5 eval points at both seeds — seed 42 .647 / .633 / .581 / .580 / .615 vs .464 / .457 / .385 / .360 / .367; seed 43 .547 / .508 / .498 / .476 / .470 vs .468 / .397 / .351 / .386 / .370 (steps 100 / 200 / 300 / 400 / 437).
- provisional (n = 300, step 100, previews never gate): documents seed 42 −5.7 pt [−10.7, −0.7] (190574 vs 188980), seed 43 −2.7 [−7.3, +2.0] (190768 vs 188982); g10 seed 42 −0.7 [−5.3, +4.0] (190575 vs 188981), seed 43 +2.7 [−2.3, +7.7] (190769 vs 188983). The seed-43 g10 preview pointed the wrong way (+2.7 at step 100, −3.3 at the end): previews show the code ran, nothing more.
- provisional (unpaired, other corpus — never a comparison under the law): the MS MARCO control scores within 2 pt of the SmolLM2 reference on every set (documents .615 / .603 vs .599 / .618; photos .667 / .670 vs .671 / .671; g10 seed 42 .611 vs .617).
- provisional: at the last stage-1 step, held-out loss jumps in all three finished runs (SmolLM2 reference .920 → 1.023, control .877 → 1.007, L1024 .612 → .850). The control also rose earlier (.801 → .877 between steps 2,000 and 4,000); L1024 did not (.611 → .612). Cause untested; 64 held-out rows.

## Owned jobs (jobid | arm / stage | state | progress | eta | note)
| job | arm / stage | state | progress | eta | note |
|---|---|---|---|---|---|
| 189693 | L2048 stage 1 | RUNNING gpu homer-019, 4 GPUs | step 2,678 / 5,208 (≈ 514k docs) | ≈ 11-17 19:00 | `pt_text_4b_long.sh 1000000 2048 4096 3 16`; out `expQ/longchunk/pt_text_1M_cml2048_189693`; next held-out check at step 4,000 |

Finished since the last wake, all COMPLETED exit 0, one attempt each: 189695 (L1024 finetune seed 43, homer-012, 11:28 h), 190273 / 190274 / 190275 (seed-43 evals, homer-012 / homer-020 / homer-012), 190768 / 190769 (seed-43 previews, homer-020 / homer-011).
Code md5 re-checked 2026-11-16 04:15, unchanged since the control evals: `eval_mm_4b.sh` 50abc167, `scripts/eval_mm_docqa.py` dfe32aa0, `scripts/judge_mm_docqa.py` 02bc8748, `pisco/model.py` 994cff3d.
Resolved-config diffs against the control: stage 1 L1024 188466 and L2048 189693 differ ONLY in `per_device_train_batch_size` (24 → 6 / 3), `gradient_accumulation_steps` (2 → 8 / 16), `compressor_max_length` (256 → 1024 / 2048), `decoder_max_length` (512 → 2048 / 4096); effective batch 192 documents held. Stage 2 189692 vs 188467: only `model_name_or_path`. Finetunes 189694 / 189695 vs 188468 / 188469: only `model_name_or_path`, plus `freeze_part: null` — a no-op field (known_issues.md).

## Adopted jobs (found running in this project, not launched by the campaign — never killed, never duplicated)
(none)

## Hand-offs
- 5f4e2f → the pisco chats (2026-11-13): make `eval_mm_4b.sh` skip generation and only judge when the result file already holds every row. Not blocking.

## Open arms and the next decision for each
| arm | chunk / text kept per doc (tokens) | text tokens at 1M docs | stage-1 cost | status | next decision |
|---|---|---|---|---|---|
| C256 control | 256 / 512 | ≈ 0.45 B | 67 GPU-h (done) | all training and 6 gating evals done | g1 + MRQA only if an arm adopts (none now) |
| L1024 | 1,024 / 2,048 | ≈ 1.04 B (estimate) | 241 GPU-h (done) | **CLOSED 2026-11-16** at both seeds | — |
| L2048 | 2,048 / 4,096 | ≈ 1.30 B | ≈ 340 GPU-h projected | stage 1 running, healthy, 51% done | owner: A stop now / B finish + full chain (law default if no answer) / C finish + one seed. If B or no answer: when stage 1 ends, check the final held-out loss, run the ownership probe and submit stage 2 → finetune seed 42 → seed 43 → evals in one chain |
| SmolLM2 reference | 256 / 512 | ≈ 0.39 B | done | reported as a row only | — |

Known confound (report, do not fix): the long arms see 2.3× / 2.9× more text tokens than the control at the same 1M documents. It cannot explain the L1024 loss away: more tokens did not help.

## Results table (deliverable; judge accuracy ↑, n = 1,500 Q per set; compression rate 8 tokens per slot)
| row | corpus | chunk (tokens) | slots per chunk | ownership ↑ | g10 s42 / s43 | documents s42 / s43 | photos s42 / s43 |
|---|---|---|---|---|---|---|---|
| SmolLM2 reference (183814 / 183816) | SmolLM2 | 256 | 33 | .977 | .617 / — | .599 / .618 | .671 / .671 |
| C256 control (188468 / 188469) | MS MARCO | 256 | 33 | .972 | .611 / .607 | .615 / .603 | .667 / .670 |
| L1024 (189694 / 189695) — CLOSED | MS MARCO | 1,024 | 129 | .969 | .600 / .574 | .553 / .561 | .670 / .670 |
| L1024 − C256, paired, 95% CI | | | | | −1.1 [−3.2, +1.2] / **−3.3 [−5.4, −1.1]** | **−6.2 [−8.6, −3.9]** / **−4.3 [−6.7, −1.8]** | +0.3 [−1.5, +2.2] / 0.0 [−1.9, +1.8] |
| L2048 | MS MARCO | 2,048 | 257 | — | — | — | — |
| uncompressed decoder (ledger) | — | — | — | — | — | .809 | .717 |

g1 and MRQA were not run: the law runs them only for an adopted arm.

Proposed text for `docs/multimodal_results.md` (the owner's chats write it): "Long compressor chunks in stage 1 (1,024 tokens instead of 256, MS MARCO documents, same 1M documents and recipe otherwise) do NOT help: paired against a 256-token control, long text −1.1 / −3.3 pt, document pages −6.2 / −4.3 pt, photos 0 (two finetune seeds, 1,500 Q per set). Chunk length stays a free inference-time choice under the spread layout; training on long chunks costs pages."

## Budget (38 GPU-days in the law)
- Spent ≈ 583 GPU-h = 24.3 GPU-days (estimate from elapsed times): fit tests ≈ 7 + control stage 1 66.9 + ownership probes 0.1 + L1024 stage 1 240.8 + control stage 2 19.2 + control finetunes 23.9 + control previews and eval attempts ≈ 6.6 + L1024 stage 2 19.1 + L1024 finetunes 11.7 + 11.5 + L1024 evals and previews ≈ 2.8 + L2048 stage 1 so far ≈ 174.
- Left if the plan runs (option B): L2048 stage 1 ≈ 163 (≈ 2,530 steps × 58 s × 4 GPUs) + downstream ≈ 61 → total ≈ 807 GPU-h = 33.6 GPU-days ≤ 38. If stopped now (option A): ≈ 24.3 GPU-days total.

## Blockers / needs the owner
- BLOCKED.md: finish or stop the 2,048-token run (recommendation: stop). Without an answer the loop follows the law and continues it.
- Next wake: when 189693 (L2048 stage 1) leaves the queue, max 48 h (the tick caps it); the owner's answer wakes earlier. At that wake: if the owner said A, `expmon kill` is then the owner's order (record it); if B / no answer: check the run ended at step 5,208, record the final held-out loss, launch the ownership probe and the stage 2 → finetunes → evals chain with the law's exact commands.
