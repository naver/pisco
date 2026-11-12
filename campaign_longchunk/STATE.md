# STATE — campaign longchunk   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-12 20:00 (woken because the 1,024-token arm wrote `checkpoint-2000/trainer_state.json`)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Stage-1 corpus = MS MARCO documents: 43.4% of documents ≥ 1,024 tokens, 20.4% ≥ 2,048 (107,120 of 3.2M sampled) — data job 188358, closed. SmolLM2 rejected (23.8% / 8.3% of 100,000 sampled) — data job 188339, closed.
- The 2,048-token chunk arm FITS on 4 × A100-80GB at batch 3 per GPU: 50/50 steps, ~72 GB per GPU, 58.6 s per step → full stage 1 = 84.8 h — job 188420, closed.
- The 1,024-token chunk arm FITS at batch 6 per GPU: 50/50 steps, ~71 GB per GPU, 42.1 s per step → full stage 1 = 60.9 h — job 188421, closed.
- Control stage 1 (256-token chunks) FINISHED: 1M documents (5,208 steps, 1.0 epoch) in 16:43 h, exit 0 — job 188422, closed. Not a collapse by the law's definition (train loss .93–.95 to the end, no NaN). Held-out loss .801 / .877 / 1.007 at steps 2,000 / 4,000 / 5,208 (64 held-out rows).
- Control stage-1 slot ownership = .972 on text (3,053 test slots; gate ≥ .95; SmolLM2 reference .977) — job 188470, closed.
- Control stage 2 (images) FINISHED: 100k rows (1,562 steps, 1.0 epoch) in 19:12 h, exit 0; held-out loss .858 / .832 / .785 at steps 1,250 / 1,500 / 1,562, no NaN — job 188467, closed. (SmolLM2-chain stage 2 183537 ended .741; corpus differs, not compared.)
- No accuracy result yet: no arm has a judged number.

**Open, and running**
- provisional: why the control's stage-1 held-out loss rises. The SmolLM2 reference 180486 also rises at the end (.924 / .920 / 1.023), but the control starts rising earlier (+.077 by step 4,000; the reference is flat, −.004). Cause untested; 64 held-out rows only; the downstream judge scores are the real health check.
- 1,024-token arm stage 1 — job 188466 RUNNING on homer-019: step 2,000 of 5,208 = 384,000 of 1M documents (.38 epoch) at 19:43. Health at step 2,000: train loss 2.17 (step 50) → .81, held-out loss .611, Rouge-L .790 (64 rows), no NaN → not a collapse. (Losses are not comparable to the control's: different chunk length.) 41.8 s per step → ends ≈ 2026-11-14 09:00 (limit 2026-11-15 00:46).
- Control finetunes (seed 42 job 188468 on homer-020, seed 43 job 188469 on homer-012) RUNNING: both past step 100 of 437 (6,400 of 28k rows); held-out loss at step 100 = .465 (s42) / .468 (s43); ≈ 102 / 92 s per step → end ≈ 2026-11-13 04:45 / 04:10. Their 6 gating evals are chained on them (all with the judge): g10 (188707 s42 / 188710 s43), documents (188708 / 188711), photos (188709 / 188712).
- Control PREVIEWS at step 100 (300 Q each, judge chained) — PENDING on gpu-be: documents 188980 (s42) / 188982 (s43), g10 188981 (s42) / 188983 (s43). Purpose: the pairing point for each later arm's step-100 preview (same step, same seed). They read a preserved copy of checkpoint-100 (below), so the Trainer's rotation cannot break them.

## Owned jobs (jobid | arm / stage | state | progress | eta | note)
| job | arm / stage | state | progress | eta | note |
|---|---|---|---|---|---|
| 188466 | L1024 stage 1 | RUNNING (gpu, homer-019) | 2,000 / 5,208 steps = 384k / 1M documents at 19:43 | ≈ 2026-11-14 09:00 | `pt_text_4b_long.sh 1000000 1024 2048 6 8`; out `expQ/longchunk/pt_text_1M_cml1024_188466`; held-out eval + checkpoint every 2,000 steps (next: step 4,000 ≈ 2026-11-13 19:00) |
| 188468 | C256 finetune s42 | RUNNING (gpu, homer-020) | > 100 / 437 steps | ≈ 2026-11-13 04:45 | 183814's argv + env `FT_DATA`; out `expQ/longchunk/ft_cml256_s42`; limit 14:03 |
| 188469 | C256 finetune s43 | RUNNING (gpu, homer-012) | > 100 / 437 steps | ≈ 2026-11-13 04:10 | + `+hf_training.seed=43`; out `expQ/longchunk/ft_cml256_s43` |
| 188707 / 188710 | C256 eval g10 s42 / s43 | PENDING (afterok 188468 / 188469) | — | ≈ 25 min after its finetune | `CML=4096 TOPK=10`, `text_hotpotqa_1500_g10`, 1,500 Q; limit 0:26 |
| 188708 / 188711 | C256 eval documents s42 / s43 | PENDING (afterok) | — | ≈ 35 min after | `PX=1003520`, `mm_docqa_1500_1m`, 1,500 Q; limit 0:40 |
| 188709 / 188712 | C256 eval photos s42 / s43 | PENDING (afterok) | — | ≈ 20 min after | `PX=1003520`, `mm_naturalqa_1500`, 1,500 Q; limit 0:24 |
| 188980 / 188982 | C256 preview documents s42 / s43 | PENDING (gpu-be, Priority) | — | when a gpu-be GPU frees | `PX=1003520`, first 300 Q of `mm_docqa_1500_1m`; limit 0:40 |
| 188981 / 188983 | C256 preview g10 s42 / s43 | PENDING (gpu-be, Priority) | — | same | `CML=4096 TOPK=10`, first 300 Q of `text_hotpotqa_1500_g10`; limit 0:26 |

All evals: `--partition gpu-be --exclude 'homer-[005-009]'` (same-node rule in CAMPAIGN.md). Eval code md5 at submit AND re-checked 2026-11-12 20:00 (unchanged): `eval_mm_4b.sh` 50abc167, `scripts/eval_mm_docqa.py` dfe32aa0 (uncommitted edits in the working tree, the owner's chats), `scripts/judge_mm_docqa.py` 02bc8748.
Preserved step-100 checkpoints: `expQ/longchunk/ft_cml256_s42_step100/checkpoint-100` and `…_s43_step100/checkpoint-100` (3.1 GB each; `diff -rq` vs the original shows only `optimizer.pt`, `scheduler.pt`, `training_args.bin` missing). The finetune keeps 2 checkpoints, so the original checkpoint-100 is deleted at step 300.
Recipe file `expQ/longchunk/pt_text_4b_long.sh` md5 8d3745e9cc343b6fb30d8d4a2bc2cd8c (unchanged; not in git — `expQ/` is gitignored).
Resolved-config diff, control 188422 vs the 1,024-token arm 188466: `compressor_max_length` 256→1024, `decoder_max_length` 512→2048, `per_device_train_batch_size` 24→6, `gradient_accumulation_steps` 2→8 (effective batch 192 held). Nothing else differs.

## Adopted jobs (found running in this project, not launched by the campaign — never killed, never duplicated)
(none. The freeze campaign's jobs belong to the sibling campaign `freeze`.)

## Open arms and the next decision for each
| arm | chunk / text kept per doc (tokens) | text tokens at 1M docs | stage-1 cost (measured) | status | next decision |
|---|---|---|---|---|---|
| C256 control | 256 / 512 | ≈ 0.45 B | 67 GPU-h (done) | stage 1 + stage 2 done; finetunes past step 100; 6 gating evals chained; 4 step-100 previews queued | read previews + the 6 full evals when they end (≈ 2026-11-13 06:00); then chain g1 + MRQA (4 cells, needed only if an arm adopts) |
| L1024 | 1,024 / 2,048 | ≈ 1.04 B (≈ 0.40 B so far, estimate) | 244 GPU-h (projected) | stage 1 at 38%, healthy at step 2,000 | probe step 4,000 (collapse check only); after it ends: ownership probe, then the control's exact stage 2 → finetunes → evals |
| L2048 | 2,048 / 4,096 | ≈ 1.30 B | 339 GPU-h (projected) | fits; not launched (one stage 1 at a time) | launch after L1024's stage 1 only if spent + projected ≤ 38 GPU-days (law) |
| SmolLM2 reference | 256 / 512 | ≈ 0.39 B | done | reported as a row only | — |

Known confound (report, do not fix): the long arms see 2.3× / 2.9× more text tokens than the control at the same 1M documents.

## Budget (38 GPU-days in the law)
- Spent: ≈ 7 GPU-h (fit tests) + 66.9 (control stage 1) + 0.05 (ownership probe) + ≈ 94 (L1024 so far, 23.5 h × 4) + 19.2 (control stage 2) + ≈ 6.9 (control finetunes so far) ≈ 194 GPU-h = 8.1 GPU-days.
- Projected with measured speeds: ≈ 841 GPU-h = 35.0 GPU-days for the full plan (+ ≈ 1 GPU-h of previews) → ≈ 3 GPU-days of slack under 38.

## Health evidence this wake (numbers, not adjectives)
- L1024 stage 1 at step 2,000: train loss .813 / .820 / .813 / .815 / .810 at steps 1,800–2,000; grad norm .047–.054; learning rate 3.24e-5 (cosine from 5e-5).
- Control stage 2 end: train loss .82 (ce .73, kd .18) at step 1,560.
- Control finetunes at step 100: train loss .48 (s42) / .51 (s43).

## Blockers / needs the owner
- Nothing now. Budget check for L2048 is due ≈ 2026-11-14 when L1024's stage 1 ends.
- Next wake: when all 6 control evals and the 4 previews have left the queue (≈ 2026-11-13 06:00), max 18 h; a failure of any job wakes earlier.
