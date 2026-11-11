# STATE — campaign longchunk   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-11 21:00 (woken because the control's slot-ownership probe, job 188470, left the queue)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Stage-1 corpus = MS MARCO documents: 43.4% of documents ≥ 1,024 tokens, 20.4% ≥ 2,048 (107,120 of 3.2M sampled) — data job 188358, closed. SmolLM2 rejected (23.8% / 8.3% of 100,000 sampled) — data job 188339, closed.
- The 2,048-token chunk arm FITS on 4 × A100-80GB at batch 3 per GPU: 50/50 steps, ~72 GB per GPU, 58.6 s per step → full stage 1 = 84.8 h — job 188420, closed.
- The 1,024-token chunk arm FITS at batch 6 per GPU: 50/50 steps, ~71 GB per GPU, 42.1 s per step → full stage 1 = 60.9 h — job 188421, closed.
- Control stage 1 (256-token chunks) FINISHED: 1M documents (5,208 steps, 1.0 epoch) in 16:43 h, exit 0 — job 188422, closed. Held-out loss .801 / .877 / 1.007 at steps 2,000 / 4,000 / 5,208 (64 held-out rows); the SmolLM2 reference 180486 shows the same rise at the end (.924 / .920 / 1.023), so it is the eval protocol, not a collapse (known_issues.md).
- Control stage-1 slot ownership = .972 on text (3,053 test slots; gate ≥ .95; SmolLM2 reference .977) — job 188470, closed.
- No accuracy result yet: no arm has a judged number.

**Open, and running**
- 1,024-token arm stage 1 — job 188466 RUNNING since 20:31 on homer-019, 42.2 s per step → ends ≈ 2026-11-14 09:30 (limit 3-04:15). Its `training_config.yaml` differs from the control's in exactly the 4 documented fields (below).
- provisional: control stage 2 (images) — job 188467 PENDING (Resources on `gpu`) since 20:31. Its data env (`PT_DATA`) is confirmed only when it writes `expQ/longchunk/pt_mm_cml256/training_config.yaml` (`training_dataset` must list pdfa_1m_desc:50000, wiki_1m_desc:19000, natural_desc:31000).
- Control finetunes (seed 42 job 188468, seed 43 job 188469) chained on stage 2; their 6 gating evals chained on them (submitted this wake, all with the judge): deciding set g10 (188707 s42 / 188710 s43), documents (188708 / 188711), photos (188709 / 188712).

## Owned jobs (jobid | arm / stage | state | progress | eta | note)
| job | arm / stage | state | progress | eta | note |
|---|---|---|---|---|---|
| 188466 | L1024 stage 1 | RUNNING (gpu, homer-019) | 34 / 5,208 steps ≈ 6,500 / 1M documents at 20:58 | ≈ 2026-11-14 09:30 | `pt_text_4b_long.sh 1000000 1024 2048 6 8`; out `expQ/longchunk/pt_text_1M_cml1024_188466` |
| 188467 | C256 stage 2 | PENDING (Resources, gpu) | 0 / 100k rows | ≈ 19 h after start | 183537's argv + env `PT_DATA`; out `expQ/longchunk/pt_mm_cml256`; limit 23:51 |
| 188468 | C256 finetune s42 | PENDING (afterok 188467) | 0 / 28k rows | ≈ 11 h after stage 2 | 183814's argv + env `FT_DATA`; out `expQ/longchunk/ft_cml256_s42`; limit 14:03 |
| 188469 | C256 finetune s43 | PENDING (afterok 188467) | 0 / 28k rows | same | + `+hf_training.seed=43`; out `expQ/longchunk/ft_cml256_s43` |
| 188707 / 188710 | C256 eval g10 s42 / s43 | PENDING (afterok 188468 / 188469) | — | ≈ 25 min after its finetune | `CML=4096 TOPK=10`, `text_hotpotqa_1500_g10`, 1,500 Q, judge chained; limit 0:26 |
| 188708 / 188711 | C256 eval documents s42 / s43 | PENDING (afterok 188468 / 188469) | — | ≈ 35 min after | `PX=1003520`, `mm_docqa_1500_1m`, 1,500 Q; limit 0:40 |
| 188709 / 188712 | C256 eval photos s42 / s43 | PENDING (afterok 188468 / 188469) | — | ≈ 20 min after | `PX=1003520`, `mm_naturalqa_1500`, 1,500 Q; limit 0:24 |

All evals: `--partition gpu-be --exclude 'homer-[005-009]'` (same-node rule, now in CAMPAIGN.md). Eval code md5 at submit: `eval_mm_4b.sh` 50abc167, `scripts/eval_mm_docqa.py` dfe32aa0 (uncommitted edits in the working tree, the owner's chats), `scripts/judge_mm_docqa.py` 02bc8748.
Recipe file `expQ/longchunk/pt_text_4b_long.sh` md5 8d3745e9cc343b6fb30d8d4a2bc2cd8c (unchanged; not in git — `expQ/` is gitignored).
Resolved-config diff, control 188422 vs the 1,024-token arm 188466 (re-checked on 188466's own file this wake): `compressor_max_length` 256→1024, `decoder_max_length` 512→2048, `per_device_train_batch_size` 24→6, `gradient_accumulation_steps` 2→8 (effective batch 192 held). Nothing else differs.

## Adopted jobs (found running in this project, not launched by the campaign — never killed, never duplicated)
(none. ft_freeze / eval_mm / check_module_move / paired_judge jobs 188651–188701 belong to the sibling campaign `freeze`.)

## Open arms and the next decision for each
| arm | chunk / text kept per doc (tokens) | text tokens at 1M docs | stage-1 cost (measured) | status | next decision |
|---|---|---|---|---|---|
| C256 control | 256 / 512 | ≈ 0.45 B | 67 GPU-h (done) | stage 1 done, ownership .972; stage 2 queued; finetunes + 6 evals chained | when stage 2 starts: check `training_dataset`; when the finetunes end: chain g1 + MRQA evals (4 cells) if wanted for the report; preview at step ~100 is NOT needed for the control |
| L1024 | 1,024 / 2,048 | ≈ 1.04 B | 244 GPU-h (projected) | stage 1 running, 0.7% | probe held-out loss at steps 2,000 / 4,000 (collapse check only); after it: ownership probe, then the control's exact stage 2 → finetunes → evals |
| L2048 | 2,048 / 4,096 | ≈ 1.30 B | 339 GPU-h (projected) | fits; not launched (one stage 1 at a time) | launch after L1024's stage 1 only if spent + projected ≤ 38 GPU-days (law) |
| SmolLM2 reference | 256 / 512 | ≈ 0.39 B | done | reported as a row only | — |

Known confound (report, do not fix): the long arms see 2.3× / 2.9× more text tokens than the control at the same 1M documents.

## Budget (38 GPU-days in the law)
- Spent: ≈ 7 GPU-h (the two fit tests) + 66.9 GPU-h (control stage 1, 16.72 h × 4 GPUs) + 0.05 (ownership probe) + ≈ 2 GPU-h (L1024 so far) ≈ 76 GPU-h = 3.2 GPU-days.
- Projected with measured speeds: ≈ 841 GPU-h = 35.0 GPU-days for the full plan → 3 GPU-days of slack under 38 (a tie's third seed ≈ 3).

## Health evidence this wake (numbers, not adjectives)
- Control stage 1, final: train loss 1.076 averaged over the run (≈ .94 at the end), 16.68 documents per second; held-out Rouge-L .783 → .758 → .723 at steps 2,000 / 4,000 / 5,208 (64 rows; same pattern as the reference, see TL;DR).
- Control ownership .972 (own segment wins; cosine own .501 vs other .113 over 3,053 test slots); late-from-early R² .015.
- L1024: 42.2 s per step at step 34 (smoke measured 42.1).

## Blockers / needs the owner
- Nothing now. Budget check for L2048 is due ≈ 2026-11-14 when L1024's stage 1 ends.
- If stage 2 (188467) is still PENDING for Resources at 02:31 on 2026-11-12 (6 h), the law allows one move to `gpu-be`; the next wake does it.
