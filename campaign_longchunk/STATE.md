# STATE — campaign longchunk   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-14 09:00 (woken because the 1,024-token arm's stage 2 and the 2,048-token arm's stage 1 both started, after the 1,024-token stage 1 finished. Checked both new configs, confirmed the control's last eval, chained the 1,024-token arm's 6 gating evals behind its finetunes. No kill.)

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

  g10 seed 42: the finished run (homer-019, 21:04–21:21) gives 1,500 / 1,500 predictions and judge verdicts identical to the earlier preempted attempt on homer-014.
- 1,024-token arm (L1024) stage 1 FINISHED: 1M documents (5,208 steps, 1.0 epoch), exit 0, 60.2 h on 4 A100 (241 GPU-h, 41.6 s per step), ≈ 1.04 B text tokens (estimate, 2.3 × the control). Held-out loss .611 / .612 / .850 at steps 2,000 / 4,000 / 5,208, Rouge-L .790 / .767 / .736; train loss .79, no NaN → no collapse — job 188466, closed.

**Open, and running**
- provisional (unpaired, other corpus — never a comparison under the law): the MS MARCO control scores like the SmolLM2 reference (documents .615 / .603 vs .599 / .618; photos .667 / .670 vs .671 / .671; g10 .611 / .607 vs reference seed 42 .617). All gaps are inside the 2-pt noise band; nothing more is claimed.
- provisional (n = 300 per cell, pairing points only): the control's step-100 previews (6,400 of 28k finetune rows): g10 .603 / .570, documents .597 / .603 (seed 42 / 43) — jobs 188981 / 188983 / 188980 / 188982. Each later arm's preview pairs against these.
- provisional: all three finished stage-1 runs (SmolLM2 reference, control, L1024) show a held-out loss jump at the LAST step only; the control alone also rises earlier (+.077 by step 4,000), L1024 does not (.611 → .612). Cause untested; 64 held-out rows; losses are not comparable across chunk lengths.
- L1024 ownership probe — job 189691 PENDING on gpu-be (Priority), 10 min. Gate ≥ .95 (control .972).
- L1024 stage 2 (images) — job 189692 RUNNING on homer-020 since 08:46, 1 A100, ~43 s per step at step 8 (control 44) → ends ≈ 2026-11-15 04:00. Resolved config differs from the control's stage 2 ONLY in the init model (`pt_text_1M_cml1024_188466/model`); `training_dataset` = pdfa_1m_desc:50000 + wiki_1m_desc:19000 + natural_desc:31000, as the control.
- 2,048-token arm (L2048) stage 1 — job 189693 RUNNING on homer-019 since 08:42, 4 A100, ~56 s per step at step 12 (fit test 58.6) → ends ≈ 2026-11-17 18:00–21:00 (limit 4 d 10 h). Resolved config differs from the control 188422 ONLY in the four documented fields (below). Held-out checks at steps 2,000 (≈ 11-15 16:00) and 4,000 (≈ 11-16 23:00).
- L1024 finetunes chained: seed 42 189694 (after stage 2, ≈ 11-15 04:00 → 17:00), seed 43 189695 (after seed 42, ≈ 11-15 17:00 → 11-16 06:00).
- L1024 gating evals chained (submitted this wake, start by themselves when each finetune exits 0): seed 42 g10 190270 / documents 190271 / photos 190272; seed 43 g10 190273 / documents 190274 / photos 190275.

## Owned jobs (jobid | arm / stage | state | progress | eta | note)
| job | arm / stage | state | progress | eta | note |
|---|---|---|---|---|---|
| 189691 | L1024 ownership probe | PENDING gpu-be (Priority) | — | when gpu-be frees an A100 | `SETS=text probe_mixer.sh …188466/model 40` = the control's 188470 command; 0:10:00 |
| 189692 | L1024 stage 2 | RUNNING (gpu, homer-020) | step ~8 of 1,562 at 09:00 (~600 of 100k rows) | ≈ 11-15 04:00 | the control's 188467 argv + env `PT_DATA`; out `expQ/longchunk/pt_mm_cml1024`; limit 24:00:00 |
| 189693 | L2048 stage 1 | RUNNING (gpu, homer-019, 4 GPUs) | step ~12 of 5,208 at 09:00 (~2,300 of 1M documents) | ≈ 11-17 18:00–21:00 | `pt_text_4b_long.sh 1000000 2048 4096 3 16`; out `expQ/longchunk/pt_text_1M_cml2048_189693`; limit 4-10:00:00 |
| 189694 | L1024 finetune s42 | PENDING afterok:189692 | — | ≈ 11-15 04:00 → 17:00 | the control's 188468 argv + env `FT_DATA`; out `expQ/longchunk/ft_cml1024_s42`; 16:06:00. **Copy checkpoint-100 before step 300 (≈ 6 h window, ≈ 07:00 → 13:00)** |
| 189695 | L1024 finetune s43 | PENDING afterok:189692, afterany:189694 | — | ≈ 11-15 17:00 → 11-16 06:00 | same + `+hf_training.seed=43`; out `…/ft_cml1024_s43`; copy its checkpoint-100 too |
| 190270 / 190271 / 190272 | L1024 s42 evals g10 / documents / photos | PENDING afterok:189694 | — | ≈ 11-15 17:00+ | = control cells 188707 / 188708 / 188709, only the checkpoint changes (`ft_cml1024_s42/model`); `CML=4096 TOPK=10` 0:26:00 / `PX=1003520` 0:40:00 / `PX=1003520` 0:24:00 |
| 190273 / 190274 / 190275 | L1024 s43 evals g10 / documents / photos | PENDING afterok:189695 | — | ≈ 11-16 06:00+ | = control cells 188710 / 188711 / 188712 (`ft_cml1024_s43/model`) |
| 188707 – 188712 | C256 gating evals | COMPLETED exit 0 | 1,500 Q each, judged | done | results in TL;DR |

All evals: `--partition gpu-be --exclude 'homer-[005-009]'` (same-node rule: image cells pair only between agreeing nodes; record the node of the attempt that wrote each file with `sacct -j <id> -D -X -o NodeList,Start,End`).
Code md5, re-checked 2026-11-14 09:10 (unchanged since the control evals were submitted): `eval_mm_4b.sh` 50abc167, `scripts/eval_mm_docqa.py` dfe32aa0, `scripts/judge_mm_docqa.py` 02bc8748, `pisco/model.py` 994cff3d, `pt_mm_4b.sh` 3b63ee99, `ft_mm_4b.sh` 4522a9fc, recipe `expQ/longchunk/pt_text_4b_long.sh` 8d3745e9 (not in git — `expQ/` is gitignored). The sibling chats have uncommitted edits in these files: if an md5 differs when an L1024 eval RUNS (python scripts are read at run time), that cell is not paired until both sides ran the same code.
Preserved step-100 checkpoints: `expQ/longchunk/ft_cml256_s42_step100/checkpoint-100` and `…_s43_step100/checkpoint-100` (3.1 GB each, model files only).
Resolved-config diffs against the control 188422 (stage 1): L1024 188466 and L2048 189693 each differ ONLY in `per_device_train_batch_size` (24 → 6 / 3), `gradient_accumulation_steps` (2 → 8 / 16), `compressor_max_length` (256 → 1024 / 2048), `decoder_max_length` (512 → 2048 / 4096); effective batch 192 documents held. Stage 2: L1024 189692 vs control 188467 differs only in `model_name_or_path`.

## Adopted jobs (found running in this project, not launched by the campaign — never killed, never duplicated)
(none. The freeze and probes campaigns' jobs belong to those sibling campaigns.)

## Hand-offs
- 5f4e2f → the pisco chats (2026-11-13 07:35): make `eval_mm_4b.sh` skip generation and only judge when the result file already holds every row. Not blocking. Preemptions cost ≈ 5 GPU-h of regenerated control evals.

## Open arms and the next decision for each
| arm | chunk / text kept per doc (tokens) | text tokens at 1M docs | stage-1 cost | status | next decision |
|---|---|---|---|---|---|
| C256 control | 256 / 512 | ≈ 0.45 B | 67 GPU-h (done) | all training and all 6 gating evals done | g1 + MRQA (4 cells) only if an arm adopts |
| L1024 | 1,024 / 2,048 | ≈ 1.04 B (estimate) | 241 GPU-h (done) | stage 2 running; ownership probe queued; 2 finetunes + 6 evals chained | ownership ≥ .95; when the seed-42 finetune writes checkpoint-100: copy it, 2 previews (documents + g10, 300 Q) paired against 188980 / 188981 |
| L2048 | 2,048 / 4,096 | ≈ 1.30 B | 339 GPU-h (projected) | stage 1 running, config clean | held-out check at step 2,000 (collapse only); its downstream chain (stage 2, finetunes, evals) is submitted once stage 1 is near its end |
| SmolLM2 reference | 256 / 512 | ≈ 0.39 B | done | reported as a row only | — |

Known confound (report, do not fix): the long arms see 2.3× / 2.9× more text tokens than the control at the same 1M documents.

## Results table (deliverable; judge accuracy ↑, n = 1,500 Q per set unless marked; rate 8 tokens per slot)
| row | corpus | chunk (tokens) | slots per chunk | ownership ↑ | g10 s42 / s43 | documents s42 / s43 | photos s42 / s43 |
|---|---|---|---|---|---|---|---|
| SmolLM2 reference (183814 / 183816) | SmolLM2 | 256 | 33 | .977 | .617 / — | .599 / .618 | .671 / .671 |
| C256 control (188468 / 188469) | MS MARCO | 256 | 33 | .972 | .611 / .607 | .615 / .603 | .667 / .670 |
| L1024 | MS MARCO | 1,024 | 129 | (189691 queued) | — | — | — |
| L2048 | MS MARCO | 2,048 | 257 | — | — | — | — |
| uncompressed decoder | — | — | — | — | — | .809 | .717 |

Paired difference to the control (arm − C256, same seed, `scripts/paired_judge.py`): none yet.

## Budget (38 GPU-days in the law)
- Spent: ≈ 7 GPU-h (fit tests) + 66.9 (control stage 1) + 0.05 (ownership probe) + 240.8 (L1024 stage 1) + 19.2 (control stage 2) + 23.9 (control finetunes) + ≈ 6.6 (previews and control eval attempts, preempted ones included) + ≈ 1.7 (L2048 and L1024 stage 2 so far) ≈ 366 GPU-h = 15.3 GPU-days.
- Projected left with measured speeds: L1024 downstream ≈ 61 + L2048 ≈ 400 (stage 1 339 + downstream 61) ≈ 461 → total ≈ 827 GPU-h = 34.5 GPU-days ≤ 38. Slack ≈ 3.5 GPU-days: a tie's third seed (≈ 3) fits, just.
- Train cap (CAMP_MAX_TRAIN = 2 running at once): now L2048 stage 1 + L1024 stage 2; then L2048 + ONE L1024 finetune at a time (seed 43 waits for seed 42). Evals do not count.

## Blockers / needs the owner
- Nothing now.
- Next wake: when `expQ/longchunk/ft_cml1024_s42/checkpoint-100/trainer_state.json` exists (≈ 11-15 07:00), max 30 h; a failure of any chained job wakes earlier. At that wake: copy checkpoint-100 to `expQ/longchunk/ft_cml1024_s42_step100/checkpoint-100` (model files only, `diff -rq` shows only optimizer/scheduler/training_args), launch the 2 previews (`eval_mm_4b.sh <copy> pisco 300 eval_data/mm_docqa_1500_1m` with `PX=1003520` 0:40:00, and `CML=4096 TOPK=10 … eval_data/text_hotpotqa_1500_g10` 0:26:00, gpu-be excl homer-[005-009]); read the ownership probe 189691 and stage 2's held-out loss; `expmon probe` L2048 at step 2,000 if reached.
