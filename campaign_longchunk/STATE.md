# STATE — campaign longchunk   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-13 08:00 (woken by a problem-count change: the control's seed-42 photos eval was preempted on gpu-be at 07:52 after 2 min and requeued — benign, known_issues row 1. This wake chained the rest of the 1,024-token arm and the 2,048-token arm's stage 1 behind the running stage 1: 5 jobs submitted, all waiting on it.)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Stage-1 corpus = MS MARCO documents: 43.4% of documents ≥ 1,024 tokens, 20.4% ≥ 2,048 (107,120 of 3.2M sampled) — data job 188358, closed. SmolLM2 rejected (23.8% / 8.3% of 100,000 sampled) — data job 188339, closed.
- The 2,048-token chunk arm FITS on 4 × A100-80GB at batch 3 per GPU: 50/50 steps, ~72 GB per GPU, 58.6 s per step → full stage 1 = 84.8 h — job 188420, closed.
- The 1,024-token chunk arm FITS at batch 6 per GPU: 50/50 steps, ~71 GB per GPU, 42.1 s per step → full stage 1 = 60.9 h — job 188421, closed.
- Control stage 1 (256-token chunks) FINISHED: 1M documents (5,208 steps, 1.0 epoch) in 16:43 h, exit 0 — job 188422, closed. Held-out loss .801 / .877 / 1.007 at steps 2,000 / 4,000 / 5,208 (64 held-out rows); train loss .91–.96 at the end, no NaN.
- Control stage-1 slot ownership = .972 on text (3,053 test slots; gate ≥ .95; SmolLM2 reference .977) — job 188470, closed.
- Control stage 2 (images) FINISHED: 100k rows (1,562 steps, 1.0 epoch) in 19:12 h, exit 0; held-out loss .785 at the end, no NaN — job 188467, closed.
- Control finetune seed 42 FINISHED: 28k rows (437 steps, 1.0 epoch) on homer-020, exit 0; held-out loss .464 / .457 / .385 / .360 / .367 at steps 100 / 200 / 300 / 400 / 437, no NaN — job 188468, closed.
- Control finetune seed 43 FINISHED: 28k rows (437 steps, 1.0 epoch) on homer-012, exit 0; held-out loss .468 / .397 / .351 / .386 / .370 at the same steps, no NaN — job 188469, closed.
- No gating accuracy yet: no arm has a judged number on 1,500 questions.

**Open, and running**
- provisional (n = 300 per cell, pairing points only, close nothing): the control's four step-100 previews (finetune at 6,400 of 28k rows, .23 epoch; rate 8 tokens per slot), judge accuracy ↑:

  | set (first 300 Q) | seed 42 | seed 43 | job s42 / s43 |
  |---|---|---|---|
  | grouped HotpotQA g10 (one long chunk per question) | .603 (bridge .560, comparison .789) | .570 (bridge .535, comparison .719) | 188981 / 188983 |
  | documents `mm_docqa_1500_1m` (first 300 = infovqa ~100 + slidevqa ~200 only) | .597 (infovqa .470, slidevqa .660) | .603 (infovqa .490, slidevqa .660) | 188980 / 188982 |

  At n = 300 the sampling error of an unpaired difference is about 4 pt, so no preview gap is a result. Only the law's paired CI on 1,500 questions decides. Each later arm's step-100 preview is paired against the same seed here, as a check that the code ran.
- provisional: why the control's stage-1 held-out loss rises. The SmolLM2 reference 180486 also rises at the end (.924 / .920 / 1.023), but the control starts rising earlier (+.077 by step 4,000; the reference is flat, −.003). Cause untested; 64 held-out rows only; the downstream judge scores are the real health check.
- 1,024-token arm stage 1 — job 188466 RUNNING on homer-019, no restart: step 3,071 of 5,208 = 589,632 of 1M documents (.59 epoch) at 08:10; train loss .78–.80 (last 5 logs), no NaN; ~41.9 s per step → ends ≈ 2026-11-14 09:00 (limit 2026-11-15 00:46). Last held-out check (step 2,000): .611. Next at step 4,000 ≈ 2026-11-13 19:00.
- The control's 6 gating evals (1,500 Q each, judge chained) — all 6 PENDING on gpu-be (Priority). Four have been preempted once and restart from scratch: g10 s43 (188710, 06:30), documents s43 (188711, 07:20, had 1,030 of 1,500 answered), photos s43 (188712, 07:20, generation done, judge at 816), photos s42 (188709, 07:52, 2 min in). Scheduler start estimates moved from today 15:15–17:00 to 2026-11-16 16:30 – 11-17 01:15. Not on the critical path: the 1,024-token arm's own evals cannot start before ≈ 2026-11-15 evening.
- NEW this wake — chained behind stage 1 188466 (all PENDING on Dependency; they start by themselves when it exits 0):
  - L1024 ownership probe 189691 (gpu-be, 10 min), then L1024 stage 2 189692 (gpu, 24 h limit) → L1024 finetune s42 189694 (16:06 limit) → L1024 finetune s43 189695 (after s42, see budget section).
  - L2048 stage 1 189693 (gpu, 4 GPUs, limit 106 h): starts when L1024's stage 1 ends — one stage 1 at a time.

## Owned jobs (jobid | arm / stage | state | progress | eta | note)
| job | arm / stage | state | progress | eta | note |
|---|---|---|---|---|---|
| 188466 | L1024 stage 1 | RUNNING (gpu, homer-019) | 3,071 / 5,208 steps = 589,632 / 1M documents at 08:10 | ≈ 2026-11-14 09:00 | `pt_text_4b_long.sh 1000000 1024 2048 6 8`; out `expQ/longchunk/pt_text_1M_cml1024_188466`; held-out eval + checkpoint every 2,000 steps |
| 189691 | L1024 ownership probe | PENDING afterok:188466 | — | ≈ 11-14 09:15 | `SETS=text probe_mixer.sh …188466/model 40` = control's 188470 command; gpu-be a100, 0:10:00 |
| 189692 | L1024 stage 2 | PENDING afterok:188466 | — | start ≈ 11-14 09:00, end ≈ 11-15 04:30 | the control's 188467 argv + env `PT_DATA` (pdfa_1m_desc:50000, wiki_1m_desc:19000, natural_desc:31000); `PT_OUT=expQ/longchunk/pt_mm_cml1024`; 24:00:00 = 19:12 h × 1.25. **Check `training_dataset` in its `training_config.yaml` once it starts.** |
| 189694 | L1024 finetune s42 | PENDING afterok:189692 | — | ≈ 11-15 04:30 → 17:30 | the control's 188468 argv + env `FT_DATA` (docpages_20k_1m_teacher:20000, natural_8k_teacher:8000); `FT_OUT=expQ/longchunk/ft_cml1024_s42`; 16:06:00 = 12:53 h × 1.25. Copy checkpoint-100 before step 300 (≈ 3 h → 8 h after start) |
| 189695 | L1024 finetune s43 | PENDING afterok:189692,afterany:189694 | — | ≈ 11-15 17:30 → 11-16 06:30 | same + `+hf_training.seed=43`; `FT_OUT=…/ft_cml1024_s43` |
| 189693 | L2048 stage 1 | PENDING afterok:188466 | — | ≈ 11-14 09:00 → 11-17 22:00 | `pt_text_4b_long.sh 1000000 2048 4096 3 16`; out `expQ/longchunk/pt_text_1M_cml2048_189693`; 4-10:00:00 = 58.6 s × 5,208 × 1.25 |
| 188707 / 188710 | C256 eval g10 s42 / s43 | PENDING (Priority; 188710 Restarts 1) | — | est. 11-16 16:30 / 17:15 | `CML=4096 TOPK=10`, `text_hotpotqa_1500_g10`, 1,500 Q; limit 0:26; model `expQ/longchunk/ft_cml256_s4X/model` |
| 188708 / 188711 | C256 eval documents s42 / s43 | PENDING (Priority; 188711 Restarts 1) | — | est. 11-16 16:45 / 11-17 00:45 | `PX=1003520`, `mm_docqa_1500_1m`, 1,500 Q; limit 0:40 |
| 188709 / 188712 | C256 eval photos s42 / s43 | PENDING (Priority; both Restarts 1) | — | est. 11-17 01:15 / 11-16 17:30 | `PX=1003520`, `mm_naturalqa_1500`, 1,500 Q; limit 0:24 |

All evals: `--partition gpu-be --exclude 'homer-[005-009]'` (same-node rule: image cells pair only between agreeing el9_8 nodes; record the node of the attempt that wrote each file with `sacct -j <id> -D -o NodeList,End`). The law has no row to widen an eval to `gpu`.
Leftover from the preempted photos s43 attempt: `expQ/longchunk/ft_cml256_s43/eval/mm_docqa_pisco_px1003520_mm_naturalqa_1500.json` (07:18, 1,500 rows, NO judge fields — not a result; the requeued 188712 overwrites it).
Code md5, re-checked 2026-11-13 08:10 (unchanged since the control evals were submitted): `eval_mm_4b.sh` 50abc167, `scripts/eval_mm_docqa.py` dfe32aa0, `scripts/judge_mm_docqa.py` 02bc8748, `pisco/model.py` 994cff3d, `pt_mm_4b.sh` 3b63ee99, `ft_mm_4b.sh` 4522a9fc, recipe `expQ/longchunk/pt_text_4b_long.sh` 8d3745e9 (not in git — `expQ/` is gitignored). If the judge-only-resume hand-off changes `eval_mm_4b.sh`, the arm's evals run a different md5 from the control's: record both and say so before pairing.
Preserved step-100 checkpoints: `expQ/longchunk/ft_cml256_s42_step100/checkpoint-100` and `…_s43_step100/checkpoint-100` (3.1 GB each, model files only).
Resolved-config diff, control 188422 vs the 1,024-token arm 188466: `compressor_max_length` 256→1024, `decoder_max_length` 512→2048, `per_device_train_batch_size` 24→6, `gradient_accumulation_steps` 2→8 (effective batch 192 held). Nothing else differs. Do the same diff for 189693 (L2048) once it starts.

## Adopted jobs (found running in this project, not launched by the campaign — never killed, never duplicated)
(none. The freeze and probes campaigns' jobs belong to those sibling campaigns.)

## Hand-offs
- 5f4e2f → the pisco chats (2026-11-13 07:35): make `eval_mm_4b.sh` skip generation and only judge when the result file already holds every row. Not blocking.

## Open arms and the next decision for each
| arm | chunk / text kept per doc (tokens) | text tokens at 1M docs | stage-1 cost (measured) | status | next decision |
|---|---|---|---|---|---|
| C256 control | 256 / 512 | ≈ 0.45 B | 67 GPU-h (done) | all training done; 4 of 4 previews done; 6 gating evals queued | read the 6 full evals when they end; then g1 + MRQA (4 cells) only if an arm adopts |
| L1024 | 1,024 / 2,048 | ≈ 1.04 B (≈ 0.61 B so far, estimate) | 244 GPU-h (projected) | stage 1 at 59%, healthy; probe + stage 2 + 2 finetunes chained | held-out check at step 4,000 (collapse → kill 188466 AND the 5 chained jobs); after the s42 finetune writes checkpoint-100: copy it, 2 previews, 3 s42 evals |
| L2048 | 2,048 / 4,096 | ≈ 1.30 B | 339 GPU-h (projected) | stage 1 chained (189693), budget rule passed at submit | when it starts: config diff vs 188422; held-out checks every 2,000 steps |
| SmolLM2 reference | 256 / 512 | ≈ 0.39 B | done | reported as a row only | — |

Known confound (report, do not fix): the long arms see 2.3× / 2.9× more text tokens than the control at the same 1M documents.

## Results table (deliverable; judge accuracy ↑, n = 1,500 Q per set unless marked; rate 8 tokens per slot)
| row | corpus | chunk (tokens) | slots per chunk | ownership ↑ | g10 s42 / s43 | documents s42 / s43 | photos s42 / s43 |
|---|---|---|---|---|---|---|---|
| SmolLM2 reference (183814 / 183816) | SmolLM2 | 256 | 33 | .977 | .617 / — | .599 / .618 | .671 / .671 |
| C256 control | MS MARCO | 256 | 33 | .972 | pending (step-100 previews .603 / .570, n = 300) | pending (step-100 previews .597 / .603, n = 300) | pending |
| L1024 | MS MARCO | 1,024 | 129 | — | — | — | — |
| L2048 | MS MARCO | 2,048 | 257 | — | — | — | — |
| uncompressed decoder | — | — | — | — | — | .809 | .717 |

## Budget (38 GPU-days in the law)
- Spent: ≈ 7 GPU-h (fit tests) + 66.9 (control stage 1) + 0.05 (ownership probe) + ≈ 142 (L1024 so far, 35.5 h × 4) + 19.2 (control stage 2) + 23.9 (control finetunes) + ≈ 1.4 (previews and preempted eval attempts) ≈ 262 GPU-h = 10.9 GPU-days.
- Projected left with measured speeds: L1024 stage 1 ≈ 100 + L1024 downstream ≈ 61 + L2048 ≈ 400 + control evals ≈ 20 ≈ 581 → total ≈ 843 GPU-h = 35.1 GPU-days ≤ 38. This is the L2048 budget rule, checked at its submit (2026-11-13 08:15). Slack ≈ 2.9 GPU-days: a tie's third seed (≈ 3) would need the owner.
- Train cap (CAMP_MAX_TRAIN = 2 running at once): L2048 stage 1 runs alongside L1024 stage 2, then alongside ONE L1024 finetune at a time (s43 waits for s42). Cost ≈ 13 h on L1024's second seed; the L1024 decision waits for the control evals anyway (est. 11-16/17).

## Blockers / needs the owner
- Nothing now.
- Next wake: when stage 1 188466 writes its step-4,000 checkpoint (≈ 19:00 today: health check, `expmon probe`), or when the 6 control evals have all left the queue, max 14 h; a failure wakes earlier. At the step-4,000 wake: collapse → `expmon kill` 188466 + 189691-189695 and verdict closed; healthy → probe it and wait for the s42 finetune's checkpoint-100 (file `expQ/longchunk/ft_cml1024_s42/checkpoint-100`) or the control evals.
