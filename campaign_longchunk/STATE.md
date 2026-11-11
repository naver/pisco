# STATE — campaign longchunk   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-11 21:30 (woken because the control's stage 2 wrote its `training_config.yaml`)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Stage-1 corpus = MS MARCO documents: 43.4% of documents ≥ 1,024 tokens, 20.4% ≥ 2,048 (107,120 of 3.2M sampled) — data job 188358, closed. SmolLM2 rejected (23.8% / 8.3% of 100,000 sampled) — data job 188339, closed.
- The 2,048-token chunk arm FITS on 4 × A100-80GB at batch 3 per GPU: 50/50 steps, ~72 GB per GPU, 58.6 s per step → full stage 1 = 84.8 h — job 188420, closed.
- The 1,024-token chunk arm FITS at batch 6 per GPU: 50/50 steps, ~71 GB per GPU, 42.1 s per step → full stage 1 = 60.9 h — job 188421, closed.
- Control stage 1 (256-token chunks) FINISHED: 1M documents (5,208 steps, 1.0 epoch) in 16:43 h, exit 0 — job 188422, closed. Not a collapse by the law's definition (train loss .93–.95 to the end, no NaN). Held-out loss .801 / .877 / 1.007 at steps 2,000 / 4,000 / 5,208 (64 held-out rows).
- Control stage-1 slot ownership = .972 on text (3,053 test slots; gate ≥ .95; SmolLM2 reference .977) — job 188470, closed.
- No accuracy result yet: no arm has a judged number.

**Open, and running**
- provisional: why the control's held-out loss rises. The SmolLM2 reference 180486 also rises at the end (.924 / .920 / 1.023), but the control starts rising earlier (+.077 by step 4,000; the reference is flat, −.004). Cause untested; the downstream judge scores are the real health check.
- 1,024-token arm stage 1 — job 188466 RUNNING on homer-019 since 20:31: 77 / 5,208 steps ≈ 14,800 of 1M documents at 21:30, 41.5 s per step → ends ≈ 2026-11-14 08:30 (limit 3-04:15). Its `training_config.yaml` differs from the control's in exactly the 4 documented fields (below).
- Control stage 2 (images) — job 188467 RUNNING on homer-020 since 21:06: 21 / 1,562 steps at 21:30, 43 s per step → ends ≈ 2026-11-12 15:45 (limit 23:51). Data mix VERIFIED in its `training_config.yaml`: pdfa_1m_desc:50000, wiki_1m_desc:19000, natural_desc:31000. Full config diff vs the reference stage 2 (183537, `expQ/archq3/pt_mm_spread`) = `model_name_or_path` only (the init model). Loss 2.10 → 1.97 at steps 10 / 20, no NaN.
- Control finetunes (seed 42 job 188468, seed 43 job 188469) chained on stage 2; their 6 gating evals chained on them, all with the judge: deciding set g10 (188707 s42 / 188710 s43), documents (188708 / 188711), photos (188709 / 188712).

## Owned jobs (jobid | arm / stage | state | progress | eta | note)
| job | arm / stage | state | progress | eta | note |
|---|---|---|---|---|---|
| 188466 | L1024 stage 1 | RUNNING (gpu, homer-019) | 77 / 5,208 steps ≈ 14,800 / 1M documents at 21:30 | ≈ 2026-11-14 08:30 | `pt_text_4b_long.sh 1000000 1024 2048 6 8`; out `expQ/longchunk/pt_text_1M_cml1024_188466`; held-out eval + checkpoint every 2,000 steps |
| 188467 | C256 stage 2 | RUNNING (gpu, homer-020) | 21 / 1,562 steps (of 100k rows) at 21:30 | ≈ 2026-11-12 15:45 | 183537's argv + env `PT_DATA` (verified); out `expQ/longchunk/pt_mm_cml256`; limit 23:51 |
| 188468 | C256 finetune s42 | PENDING (afterok 188467) | 0 / 28k rows | ≈ 11 h after stage 2 | 183814's argv + env `FT_DATA`; out `expQ/longchunk/ft_cml256_s42`; limit 14:03 |
| 188469 | C256 finetune s43 | PENDING (afterok 188467) | 0 / 28k rows | same | + `+hf_training.seed=43`; out `expQ/longchunk/ft_cml256_s43` |
| 188707 / 188710 | C256 eval g10 s42 / s43 | PENDING (afterok 188468 / 188469) | — | ≈ 25 min after its finetune | `CML=4096 TOPK=10`, `text_hotpotqa_1500_g10`, 1,500 Q, judge chained; limit 0:26 |
| 188708 / 188711 | C256 eval documents s42 / s43 | PENDING (afterok 188468 / 188469) | — | ≈ 35 min after | `PX=1003520`, `mm_docqa_1500_1m`, 1,500 Q; limit 0:40 |
| 188709 / 188712 | C256 eval photos s42 / s43 | PENDING (afterok 188468 / 188469) | — | ≈ 20 min after | `PX=1003520`, `mm_naturalqa_1500`, 1,500 Q; limit 0:24 |

All evals: `--partition gpu-be --exclude 'homer-[005-009]'` (same-node rule in CAMPAIGN.md). Eval code md5 at submit: `eval_mm_4b.sh` 50abc167, `scripts/eval_mm_docqa.py` dfe32aa0 (uncommitted edits in the working tree, the owner's chats), `scripts/judge_mm_docqa.py` 02bc8748.
Recipe file `expQ/longchunk/pt_text_4b_long.sh` md5 8d3745e9cc343b6fb30d8d4a2bc2cd8c (unchanged; not in git — `expQ/` is gitignored).
Resolved-config diff, control 188422 vs the 1,024-token arm 188466: `compressor_max_length` 256→1024, `decoder_max_length` 512→2048, `per_device_train_batch_size` 24→6, `gradient_accumulation_steps` 2→8 (effective batch 192 held). Nothing else differs.

## Adopted jobs (found running in this project, not launched by the campaign — never killed, never duplicated)
(none. ft_freeze / eval_mm / check_module_move / paired_judge jobs 188651–188701 belong to the sibling campaign `freeze`.)

## Open arms and the next decision for each
| arm | chunk / text kept per doc (tokens) | text tokens at 1M docs | stage-1 cost (measured) | status | next decision |
|---|---|---|---|---|---|
| C256 control | 256 / 512 | ≈ 0.45 B | 67 GPU-h (done) | stage 1 done, ownership .972; stage 2 running (data verified); finetunes + 6 evals chained | when the finetunes end (≈ 2026-11-13 03:00): optionally chain g1 + MRQA evals (4 cells, needed only if an arm adopts) |
| L1024 | 1,024 / 2,048 | ≈ 1.04 B | 244 GPU-h (projected) | stage 1 running, 1.5% | probe held-out loss at steps 2,000 / 4,000 (collapse check only); after it: ownership probe, then the control's exact stage 2 → finetunes → evals |
| L2048 | 2,048 / 4,096 | ≈ 1.30 B | 339 GPU-h (projected) | fits; not launched (one stage 1 at a time) | launch after L1024's stage 1 only if spent + projected ≤ 38 GPU-days (law) |
| SmolLM2 reference | 256 / 512 | ≈ 0.39 B | done | reported as a row only | — |

Known confound (report, do not fix): the long arms see 2.3× / 2.9× more text tokens than the control at the same 1M documents.

## Budget (38 GPU-days in the law)
- Spent: ≈ 7 GPU-h (the two fit tests) + 66.9 GPU-h (control stage 1, 16.72 h × 4 GPUs) + 0.05 (ownership probe) + ≈ 4 GPU-h (L1024 so far, 1 h × 4) + ≈ 0.4 GPU-h (stage 2 so far) ≈ 78 GPU-h = 3.3 GPU-days.
- Projected with measured speeds: ≈ 841 GPU-h = 35.0 GPU-days for the full plan → 3 GPU-days of slack under 38 (a tie's third seed ≈ 3).

## Health evidence this wake (numbers, not adjectives)
- Control stage 2: 43 s per step at step 21 (reference 183537 took 19:04 h for the same 1,562 steps ≈ 44 s per step); loss 2.096 / 1.967, ce 1.665 / 1.596, kd .862 / .741 at steps 10 / 20.
- L1024 stage 1: 41.5 s per step at step 77 (smoke measured 42.1).

## Blockers / needs the owner
- Nothing now. Budget check for L2048 is due ≈ 2026-11-14 when L1024's stage 1 ends.
- Next wake: when L1024 writes `checkpoint-2000/trainer_state.json` (≈ 2026-11-12 19:40) to probe its held-out loss; a failure of any chained job wakes earlier.
