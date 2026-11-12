# STATE — campaign longchunk   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-13 02:00 (woken by a problem-count change; no failure. The control's g10 preview at seed 42 finished and was recorded; the other 3 previews keep being preempted on gpu-be and requeued — benign. Nothing launched.)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Stage-1 corpus = MS MARCO documents: 43.4% of documents ≥ 1,024 tokens, 20.4% ≥ 2,048 (107,120 of 3.2M sampled) — data job 188358, closed. SmolLM2 rejected (23.8% / 8.3% of 100,000 sampled) — data job 188339, closed.
- The 2,048-token chunk arm FITS on 4 × A100-80GB at batch 3 per GPU: 50/50 steps, ~72 GB per GPU, 58.6 s per step → full stage 1 = 84.8 h — job 188420, closed.
- The 1,024-token chunk arm FITS at batch 6 per GPU: 50/50 steps, ~71 GB per GPU, 42.1 s per step → full stage 1 = 60.9 h — job 188421, closed.
- Control stage 1 (256-token chunks) FINISHED: 1M documents (5,208 steps, 1.0 epoch) in 16:43 h, exit 0 — job 188422, closed. Not a collapse by the law's definition (train loss fell from 2.30 to .91–.96 over the last ~1,000 steps, no NaN). Held-out loss .801 / .877 / 1.007 at steps 2,000 / 4,000 / 5,208 (64 held-out rows).
- Control stage-1 slot ownership = .972 on text (3,053 test slots; gate ≥ .95; SmolLM2 reference .977) — job 188470, closed.
- Control stage 2 (images) FINISHED: 100k rows (1,562 steps, 1.0 epoch) in 19:12 h, exit 0; held-out loss .858 / .832 / .785 at steps 1,250 / 1,500 / 1,562, no NaN — job 188467, closed. (SmolLM2-chain stage 2 183537 ended .741; corpus differs, not compared.)
- No gating accuracy yet: no arm has a judged number on 1,500 questions.

**Open, and running**
- provisional (n = 300, a pairing point only, closes nothing): control finetune seed 42 at step 100 (6,400 of 28k rows, .23 epoch) scores judge accuracy **.603** on the first 300 questions of grouped HotpotQA (one long chunk per question; rate 8 tokens per slot); bridge .560, comparison .789 — job 188981, closed. Each later arm's seed-42 step-100 g10 preview pairs against it.
- provisional: why the control's stage-1 held-out loss rises. The SmolLM2 reference 180486 also rises at the end (.924 / .920 / 1.023), but the control starts rising earlier (+.077 by step 4,000; the reference is flat, −.004). Cause untested; 64 held-out rows only; the downstream judge scores are the real health check.
- 1,024-token arm stage 1 — job 188466 RUNNING on homer-019: step 2,546 of 5,208 = 489,000 of 1M documents (.49 epoch) at 02:01. Train loss .80, grad norm .044, no NaN. Last held-out check (step 2,000): .611. 40.5 s per step → ends ≈ 2026-11-14 08:00 (limit 2026-11-15 00:46). Next held-out check at step 4,000 ≈ 2026-11-13 18:30.
- Control finetunes RUNNING: seed 42 (job 188468, homer-020) step 329 of 437 = 21,000 of 28k rows (.75 epoch), train loss .41, 105.5 s per step → ends ≈ 05:15 (limit 06:22, ≈ 1.1 h margin); seed 43 (job 188469, homer-012) step 361 = 23,100 rows (.82 epoch), train loss .48, 88.7 s per step → ends ≈ 03:55 (limit 06:58). Held-out loss s42 .464 / .457 / .385 and s43 .468 / .397 / .351 at steps 100 / 200 / 300.
- Their 6 gating evals are chained on them (all with the judge): g10 (188707 s42 / 188710 s43), documents (188708 / 188711), photos (188709 / 188712).
- Control previews still queued on gpu-be (300 Q each, judge chained): documents s42 (188980), documents s43 (188982), g10 s43 (188983). gpu-be keeps preempting them; scontrol at 02:02 reads Restarts = 3 / 2 / 2; 188982 is starting now, the other two have SLURM start estimates of 11-15. Each restarts from scratch (no partial file); benign.

## Owned jobs (jobid | arm / stage | state | progress | eta | note)
| job | arm / stage | state | progress | eta | note |
|---|---|---|---|---|---|
| 188466 | L1024 stage 1 | RUNNING (gpu, homer-019) | 2,546 / 5,208 steps = 489k / 1M documents at 02:01 | ≈ 2026-11-14 08:00 | `pt_text_4b_long.sh 1000000 1024 2048 6 8`; out `expQ/longchunk/pt_text_1M_cml1024_188466`; held-out eval + checkpoint every 2,000 steps |
| 188468 | C256 finetune s42 | RUNNING (gpu, homer-020) | 329 / 437 steps = 21.0k / 28k rows | ≈ 2026-11-13 05:15 | 183814's argv + env `FT_DATA`; out `expQ/longchunk/ft_cml256_s42`; limit 06:22 |
| 188469 | C256 finetune s43 | RUNNING (gpu, homer-012) | 361 / 437 steps = 23.1k / 28k rows | ≈ 2026-11-13 03:55 | + `+hf_training.seed=43`; out `expQ/longchunk/ft_cml256_s43`; limit 06:58 |
| 188707 / 188710 | C256 eval g10 s42 / s43 | PENDING (afterok 188468 / 188469) | — | after its finetune, when gpu-be has a GPU | `CML=4096 TOPK=10`, `text_hotpotqa_1500_g10`, 1,500 Q; limit 0:26 |
| 188708 / 188711 | C256 eval documents s42 / s43 | PENDING (afterok) | — | same | `PX=1003520`, `mm_docqa_1500_1m`, 1,500 Q; limit 0:40 |
| 188709 / 188712 | C256 eval photos s42 / s43 | PENDING (afterok) | — | same | `PX=1003520`, `mm_naturalqa_1500`, 1,500 Q; limit 0:24 |
| 188980 / 188982 | C256 preview documents s42 / s43 | PENDING / starting (Restarts 3 / 2) | — | — | `PX=1003520`, first 300 Q of `mm_docqa_1500_1m`; limit 0:40 |
| 188983 | C256 preview g10 s43 | PENDING (Restarts 2) | — | — | `CML=4096 TOPK=10`, first 300 Q of `text_hotpotqa_1500_g10`; limit 0:26 |
| 188981 | C256 preview g10 s42 | done, closed | 300 / 300 Q, judge .603 | — | 5:21 on homer-016; result json under `expQ/longchunk/ft_cml256_s42_step100/eval/` |

All evals: `--partition gpu-be --exclude 'homer-[005-009]'` (same-node rule in CAMPAIGN.md). Eval code md5 at submit AND re-checked 2026-11-13 02:00 (unchanged): `eval_mm_4b.sh` 50abc167, `scripts/eval_mm_docqa.py` dfe32aa0, `scripts/judge_mm_docqa.py` 02bc8748. Note: `pisco/model.py` changed at 2026-11-12 23:40 (the freeze campaign's owner-requested `freeze_part` patch, md5 994cff3d; its tests say the unset path is unchanged). The preview 188981 and every control eval still to run use the patched file; arm evals will too, so pairs stay on the same code.
Preserved step-100 checkpoints: `expQ/longchunk/ft_cml256_s42_step100/checkpoint-100` and `…_s43_step100/checkpoint-100` (3.1 GB each, model files only).
Recipe file `expQ/longchunk/pt_text_4b_long.sh` md5 8d3745e9cc343b6fb30d8d4a2bc2cd8c (unchanged; not in git — `expQ/` is gitignored).
Resolved-config diff, control 188422 vs the 1,024-token arm 188466: `compressor_max_length` 256→1024, `decoder_max_length` 512→2048, `per_device_train_batch_size` 24→6, `gradient_accumulation_steps` 2→8 (effective batch 192 held). Nothing else differs.

## Adopted jobs (found running in this project, not launched by the campaign — never killed, never duplicated)
(none. The freeze and probes campaigns' jobs belong to those sibling campaigns.)

## Open arms and the next decision for each
| arm | chunk / text kept per doc (tokens) | text tokens at 1M docs | stage-1 cost (measured) | status | next decision |
|---|---|---|---|---|---|
| C256 control | 256 / 512 | ≈ 0.45 B | 67 GPU-h (done) | stage 1 + stage 2 done; finetunes at .75 / .82 epoch; 6 gating evals chained; 1 of 4 previews done (.603) | read the 6 full evals when they end; then g1 + MRQA (4 cells) only if an arm adopts |
| L1024 | 1,024 / 2,048 | ≈ 1.04 B (≈ 0.51 B so far, estimate) | 244 GPU-h (projected) | stage 1 at 49%, healthy | held-out check at step 4,000 (collapse check only); after it ends: ownership probe, then the control's exact stage 2 → finetunes → evals |
| L2048 | 2,048 / 4,096 | ≈ 1.30 B | 339 GPU-h (projected) | fits; not launched (one stage 1 at a time) | launch after L1024's stage 1 only if spent + projected ≤ 38 GPU-days (law) |
| SmolLM2 reference | 256 / 512 | ≈ 0.39 B | done | reported as a row only | — |

Known confound (report, do not fix): the long arms see 2.3× / 2.9× more text tokens than the control at the same 1M documents.

## Results table (deliverable; judge accuracy ↑, n = 1,500 Q per set unless marked; rate 8 tokens per slot)
| row | corpus | chunk (tokens) | slots per chunk | ownership ↑ | g10 s42 / s43 | documents s42 / s43 | photos s42 / s43 |
|---|---|---|---|---|---|---|---|
| SmolLM2 reference (183814 / 183816) | SmolLM2 | 256 | 33 | .977 | .617 / — | .599 / .618 | .671 / .671 |
| C256 control | MS MARCO | 256 | 33 | .972 | pending (step-100 preview s42: .603, n = 300) | pending | pending |
| L1024 | MS MARCO | 1,024 | 129 | — | — | — | — |
| uncompressed decoder | — | — | — | — | — | .809 | .717 |

## Budget (38 GPU-days in the law)
- Spent: ≈ 7 GPU-h (fit tests) + 66.9 (control stage 1) + 0.05 (ownership probe) + ≈ 118 (L1024 so far, 29.5 h × 4) + 19.2 (control stage 2) + ≈ 18.8 (control finetunes so far) + ≈ 0.3 (previews, with preempted partial runs) ≈ 230 GPU-h = 9.6 GPU-days.
- Projected with measured speeds: ≈ 841 GPU-h = 35.0 GPU-days for the full plan → ≈ 3 GPU-days of slack under 38.

## Blockers / needs the owner
- Nothing now. Budget check for L2048 is due ≈ 2026-11-14 08:00 when L1024's stage 1 ends.
- Watch: if the seed-42 control finetune TIMEOUTs (≈ 1.1 h margin now), it is a transient wall-time misestimate → relaunch the same command per the law (it resumes from its last checkpoint), `--time` = measured s/step × remaining steps × 1.25 + load.
- Next wake: when the 6 gating control evals have left the queue, max 18 h (covers the L1024 held-out check at step 4,000 ≈ 18:30); a failure of any job wakes earlier.
