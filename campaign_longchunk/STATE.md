# STATE — campaign longchunk   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-13 07:00 (woken by a problem-count change: the control's seed-43 g10 eval was preempted on gpu-be at 06:30 and requeued — benign, known_issues row 1. Both control finetunes and the last 3 control previews had finished cleanly; recorded. Nothing launched.)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Stage-1 corpus = MS MARCO documents: 43.4% of documents ≥ 1,024 tokens, 20.4% ≥ 2,048 (107,120 of 3.2M sampled) — data job 188358, closed. SmolLM2 rejected (23.8% / 8.3% of 100,000 sampled) — data job 188339, closed.
- The 2,048-token chunk arm FITS on 4 × A100-80GB at batch 3 per GPU: 50/50 steps, ~72 GB per GPU, 58.6 s per step → full stage 1 = 84.8 h — job 188420, closed.
- The 1,024-token chunk arm FITS at batch 6 per GPU: 50/50 steps, ~71 GB per GPU, 42.1 s per step → full stage 1 = 60.9 h — job 188421, closed.
- Control stage 1 (256-token chunks) FINISHED: 1M documents (5,208 steps, 1.0 epoch) in 16:43 h, exit 0 — job 188422, closed. Held-out loss .801 / .877 / 1.007 at steps 2,000 / 4,000 / 5,208 (64 held-out rows); train loss .91–.96 at the end, no NaN.
- Control stage-1 slot ownership = .972 on text (3,053 test slots; gate ≥ .95; SmolLM2 reference .977) — job 188470, closed.
- Control stage 2 (images) FINISHED: 100k rows (1,562 steps, 1.0 epoch) in 19:12 h, exit 0; held-out loss .785 at the end, no NaN — job 188467, closed.
- Control finetune seed 42 FINISHED: 28k rows (437 steps, 1.0 epoch) in 12:53 h on homer-020, exit 0; held-out loss .464 / .457 / .385 / .360 / .367 at steps 100 / 200 / 300 / 400 / 437, no NaN — job 188468, closed.
- Control finetune seed 43 FINISHED: 28k rows (437 steps, 1.0 epoch) in 11:00 h on homer-012, exit 0; held-out loss .468 / .397 / .351 / .386 / .370 at the same steps, no NaN — job 188469, closed.
- No gating accuracy yet: no arm has a judged number on 1,500 questions.

**Open, and running**
- provisional (n = 300 per cell, pairing points only, close nothing): the control's four step-100 previews (finetune at 6,400 of 28k rows, .23 epoch; rate 8 tokens per slot), judge accuracy ↑:

  | set (first 300 Q) | seed 42 | seed 43 | job s42 / s43 |
  |---|---|---|---|
  | grouped HotpotQA g10 (one long chunk per question) | .603 (bridge .560, comparison .789) | .570 (bridge .535, comparison .719) | 188981 / 188983 |
  | documents `mm_docqa_1500_1m` | .597 (infovqa .470, slidevqa .660) | .603 (infovqa .490, slidevqa .660) | 188980 / 188982 |

  The two seeds differ by 3.3 pt on g10 at n = 300: a later arm's preview gap smaller than that is seed noise. Each later arm's step-100 preview pairs against the same seed here.
- provisional: why the control's stage-1 held-out loss rises. The SmolLM2 reference 180486 also rises at the end (.924 / .920 / 1.023), but the control starts rising earlier (+.077 by step 4,000; the reference is flat, −.004). Cause untested; 64 held-out rows only; the downstream judge scores are the real health check.
- 1,024-token arm stage 1 — job 188466 RUNNING on homer-019: step 2,981 of 5,208 = 572,000 of 1M documents (.57 epoch) at 07:00. 41 s per step → ends ≈ 2026-11-14 08:30 (limit 2026-11-15 00:46). Last held-out check (step 2,000): .611. Next held-out check at step 4,000 ≈ 2026-11-13 18:40.
- The control's 6 gating evals (1,500 Q each, judge chained) — dependencies met, now on gpu-be: documents s43 (188711) and photos s43 (188712) RUNNING on homer-019 since 07:01; g10 s42 (188707, est. start 07:25), documents s42 (188708, est. 15:15), photos s42 (188709, est. 15:45), g10 s43 (188710, preempted once at 06:30, est. 16:00) PENDING (Priority).

## Owned jobs (jobid | arm / stage | state | progress | eta | note)
| job | arm / stage | state | progress | eta | note |
|---|---|---|---|---|---|
| 188466 | L1024 stage 1 | RUNNING (gpu, homer-019) | 2,981 / 5,208 steps = 572k / 1M documents at 07:00 | ≈ 2026-11-14 08:30 | `pt_text_4b_long.sh 1000000 1024 2048 6 8`; out `expQ/longchunk/pt_text_1M_cml1024_188466`; held-out eval + checkpoint every 2,000 steps |
| 188707 / 188710 | C256 eval g10 s42 / s43 | PENDING (Priority; 188710 Restarts 1) | — | est. 07:25 / 16:00 | `CML=4096 TOPK=10`, `text_hotpotqa_1500_g10`, 1,500 Q; limit 0:26; model `expQ/longchunk/ft_cml256_s4X/model` |
| 188708 / 188711 | C256 eval documents s42 / s43 | PENDING / RUNNING (homer-019) | — | est. 15:15 / ≈ 07:40 | `PX=1003520`, `mm_docqa_1500_1m`, 1,500 Q; limit 0:40 |
| 188709 / 188712 | C256 eval photos s42 / s43 | PENDING / RUNNING (homer-019) | — | est. 15:45 / ≈ 07:25 | `PX=1003520`, `mm_naturalqa_1500`, 1,500 Q; limit 0:24 |

All evals: `--partition gpu-be --exclude 'homer-[005-009]'` (same-node rule: image cells pair only between agreeing el9_8 nodes; record the node of the attempt that wrote each file with `sacct -j <id> -D -o NodeList,End`). Eval code md5 at submit AND re-checked 2026-11-13 07:02 (unchanged): `eval_mm_4b.sh` 50abc167, `scripts/eval_mm_docqa.py` dfe32aa0, `scripts/judge_mm_docqa.py` 02bc8748; `pisco/model.py` 994cff3d (the freeze campaign's 2026-11-12 23:40 `freeze_part` patch; every control eval and every arm eval run on it, so pairs stay on the same code).
Preserved step-100 checkpoints: `expQ/longchunk/ft_cml256_s42_step100/checkpoint-100` and `…_s43_step100/checkpoint-100` (3.1 GB each, model files only); the 4 preview result files sit in their `eval/` dirs. All 4 previews' final attempts ran on el9_8 nodes (188981 homer-016; 188980 / 188982 / 188983 homer-012).
Recipe file `expQ/longchunk/pt_text_4b_long.sh` md5 8d3745e9cc343b6fb30d8d4a2bc2cd8c (unchanged; not in git — `expQ/` is gitignored).
Resolved-config diff, control 188422 vs the 1,024-token arm 188466: `compressor_max_length` 256→1024, `decoder_max_length` 512→2048, `per_device_train_batch_size` 24→6, `gradient_accumulation_steps` 2→8 (effective batch 192 held). Nothing else differs.

## Adopted jobs (found running in this project, not launched by the campaign — never killed, never duplicated)
(none. The freeze and probes campaigns' jobs belong to those sibling campaigns.)

## Open arms and the next decision for each
| arm | chunk / text kept per doc (tokens) | text tokens at 1M docs | stage-1 cost (measured) | status | next decision |
|---|---|---|---|---|---|
| C256 control | 256 / 512 | ≈ 0.45 B | 67 GPU-h (done) | stage 1, stage 2, both finetunes done; 4 of 4 previews done; 6 gating evals running / queued | read the 6 full evals when they end; then g1 + MRQA (4 cells) only if an arm adopts |
| L1024 | 1,024 / 2,048 | ≈ 1.04 B (≈ 0.60 B so far, estimate) | 244 GPU-h (projected) | stage 1 at 57%, healthy | held-out check at step 4,000 (collapse check only); after it ends: ownership probe, then the control's exact stage 2 → finetunes → evals |
| L2048 | 2,048 / 4,096 | ≈ 1.30 B | 339 GPU-h (projected) | fits; not launched (one stage 1 at a time) | launch after L1024's stage 1 only if spent + projected ≤ 38 GPU-days (law) |
| SmolLM2 reference | 256 / 512 | ≈ 0.39 B | done | reported as a row only | — |

Known confound (report, do not fix): the long arms see 2.3× / 2.9× more text tokens than the control at the same 1M documents.

## Results table (deliverable; judge accuracy ↑, n = 1,500 Q per set unless marked; rate 8 tokens per slot)
| row | corpus | chunk (tokens) | slots per chunk | ownership ↑ | g10 s42 / s43 | documents s42 / s43 | photos s42 / s43 |
|---|---|---|---|---|---|---|---|
| SmolLM2 reference (183814 / 183816) | SmolLM2 | 256 | 33 | .977 | .617 / — | .599 / .618 | .671 / .671 |
| C256 control | MS MARCO | 256 | 33 | .972 | pending (step-100 previews .603 / .570, n = 300) | pending (step-100 previews .597 / .603, n = 300) | pending |
| L1024 | MS MARCO | 1,024 | 129 | — | — | — | — |
| uncompressed decoder | — | — | — | — | — | .809 | .717 |

## Budget (38 GPU-days in the law)
- Spent: ≈ 7 GPU-h (fit tests) + 66.9 (control stage 1) + 0.05 (ownership probe) + ≈ 138 (L1024 so far, 34.5 h × 4) + 19.2 (control stage 2) + 23.9 (control finetunes, done) + ≈ 0.6 (previews, with preempted partial runs) ≈ 256 GPU-h = 10.7 GPU-days.
- Projected with measured speeds: ≈ 841 GPU-h = 35.0 GPU-days for the full plan → ≈ 3 GPU-days of slack under 38.

## Blockers / needs the owner
- Nothing now. Budget check for L2048 is due ≈ 2026-11-14 08:30 when L1024's stage 1 ends.
- Next wake: when the 6 gating control evals have left the queue, max 18 h (covers the L1024 held-out check at step 4,000 ≈ 18:40); a failure of any job wakes earlier. At that wake: read the 6 judge scores, check each image cell's node, record verdicts, fill the C256 row of the results table, refresh `expmon display --exp longchunk`.
