# STATE — campaign longchunk   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-11 10:30 (woken because the control's stage-1 checkpoint at step 2,000 appeared)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Stage-1 corpus = MS MARCO documents: 43.4% of documents ≥ 1,024 tokens, 20.4% ≥ 2,048 (107,120 of 3.2M sampled) — data job 188358, closed. SmolLM2 rejected (23.8% / 8.3% of 100,000 sampled) — data job 188339, closed.
- The 2,048-token chunk arm FITS on 4 × A100-80GB at batch 3 per GPU: 50/50 steps, exit 0, ~72 GB per GPU (sacct 288 GB over 4), 58.6 s per step (steps 10–49 of `pt_text_long_188420.out`) → full stage 1 (5,208 steps, 1M documents) = 84.8 h — job 188420, closed.
- The 1,024-token chunk arm FITS at batch 6 per GPU: 50/50 steps, exit 0, ~71 GB per GPU, 42.1 s per step (`pt_text_long_188421.out`) → full stage 1 = 60.9 h — job 188421, closed.
- No accuracy result yet: no arm has a judged number.

**Open, and running**
- Control stage 1 (256-token chunks, 1M documents): job 188422 RUNNING on homer-019. Health check at step 2,000 (.384 epoch, 384,000 documents, ≈ 0.17 B text tokens) PASSED: held-out loss .801, Rouge-L .783 on 64 held-out rows (`pt_text_long_188422.out`; probe recorded). Train loss fell 2.30 → .95 and sits at .95–.97 since step ~1,000, slowly falling; no NaN. 11.3–11.6 s per step → ends ≈ 20:45 today. Scale only (other corpus, not a comparison): the SmolLM2 reference 180486 had held-out loss .924 at the same epoch.
- Chained behind it, all PENDING on dependencies:
  - 1,024-token arm stage 1 — job 188466, starts when the control's stage 1 succeeds, 76 h limit.
  - provisional: control stage 2 (images) — job 188467 → its two finetunes, seed 42 job 188468 and seed 43 job 188469. The argv equals the reference runs'; the data env vars (`PT_DATA`, `FT_DATA`) were passed at submit but Slurm shows a job's environment only once it runs, so they are confirmed when 188467 writes `expQ/longchunk/pt_mm_cml256/training_config.yaml` (`training_dataset` must list pdfa_1m_desc:50000, wiki_1m_desc:19000, natural_desc:31000).
  - Slot-ownership health probe on the control's stage-1 model — job 188470 (reference .977).

## Owned jobs (jobid | arm / stage | state | progress | eta | note)
| job | arm / stage | state | progress | eta | note |
|---|---|---|---|---|---|
| 188422 | C256 stage 1 | RUNNING (gpu, homer-019) | 2,068 / 5,208 steps ≈ 397,000 / 1M documents at 10:37 | ≈ 20:45 today | `pt_text_4b_long.sh 1000000 256 512 24 2`; out `expQ/longchunk/pt_text_1M_cml256_188422`; limit 23 h (16.8 h needed); held-out loss .801 @ 2,000 |
| 188466 | L1024 stage 1 | PENDING (afterok 188422) | 0 / 1M documents | ≈ 2026-11-14 10:00 | `… 1000000 1024 2048 6 8`; limit 3-04:15 = 42.1 s × 5,208 × 1.25; out `expQ/longchunk/pt_text_1M_cml1024_188466` |
| 188467 | C256 stage 2 | PENDING (afterok 188422) | 0 / 100k rows | ≈ 2026-11-12 16:00 | 183537's argv + env `PT_DATA`; out `expQ/longchunk/pt_mm_cml256`; limit 23:51 |
| 188468 | C256 finetune s42 | PENDING (afterok 188467) | 0 / 28k rows | ≈ 2026-11-13 03:00 | 183814's argv + env `FT_DATA` (docpages 20k + natural 8k); out `expQ/longchunk/ft_cml256_s42`; limit 14:03 |
| 188469 | C256 finetune s43 | PENDING (afterok 188467) | 0 / 28k rows | ≈ 2026-11-13 03:00 | same + `+hf_training.seed=43`; out `expQ/longchunk/ft_cml256_s43` |
| 188470 | C256 ownership probe | PENDING (afterok 188422) | — | ≈ 21:00 today | `SETS=text probe_mixer.sh <stage-1 model> 40` (= 184705, reference .977); limit 10 min |

Recipe file `expQ/longchunk/pt_text_4b_long.sh` md5 8d3745e9cc343b6fb30d8d4a2bc2cd8c (unchanged; not in git — `expQ/` is gitignored, griped 2026-11-11 02:00).
Resolved-config diff, control (188422) vs the 1,024-token arm (smoke 188421): `compressor_max_length` 256→1024, `decoder_max_length` 512→2048, `per_device_train_batch_size` 24→6, `gradient_accumulation_steps` 2→8 (effective batch 192 held); the smoke's `max_steps: 50` only. The 2,048-token arm adds 1024→2048, 2048→4096, 6→3, 8→16. Nothing else differs. Re-diff 188466's own `training_config.yaml` against 188422's when it starts.

## Adopted jobs (found running in this project, not launched by the campaign — never killed, never duplicated)
(none. ft_freeze / check_module_move / eval_mm jobs 188582–188599 belong to the sibling campaign `freeze`.)

## Open arms and the next decision for each
| arm | chunk / text kept per doc (tokens) | text tokens at 1M docs | stage-1 cost (measured) | status | next decision |
|---|---|---|---|---|---|
| C256 control | 256 / 512 | ≈ 0.45 B | 67 GPU-h | stage 1 at 40%, healthy at step 2,000; stage 2 + finetunes + probe chained | when stage 1 ends: step-4,000 and final held-out loss, ownership ≥ .95, `training_dataset` of 188467; then evals of both finetunes (g10, g1, MRQA, documents, photos + judge) once 188468/188469 end |
| L1024 | 1,024 / 2,048 | ≈ 1.04 B | 244 GPU-h | stage 1 chained (188466) | when it starts: config diff vs 188422; after it: ownership probe, then the control's exact stage 2 → finetunes → evals |
| L2048 | 2,048 / 4,096 | ≈ 1.30 B | 339 GPU-h | fits; not launched (one stage 1 at a time) | launch after L1024's stage 1 only if spent + projected ≤ 38 GPU-days (law) |
| SmolLM2 reference | 256 / 512 | ≈ 0.39 B | done | reported as a row only | — |

Known confound (report, do not fix): the long arms see 2.3× / 2.9× more text tokens than the control at the same 1M documents.

## Budget (38 GPU-days in the law)
- Spent: ≈ 7 GPU-h (the two fit tests) + ≈ 27 GPU-h of the control's stage 1 (6.75 h × 4 GPUs) ≈ 34 GPU-h = 1.4 GPU-days.
- Projected with measured speeds: ≈ 841 GPU-h = 35.0 GPU-days for the full plan → 3 GPU-days of slack under 38 (a tie's third seed ≈ 3).

## Health evidence this wake (numbers, not adjectives)
- Control step 2,000: held-out loss .801, Rouge-L .783, precision .814, recall .804 (64 rows); train loss .952 at step 2,050, grad norm .26, lr 3.19e-5; speed 11.3–11.6 s per step.
- One harmless tokenizer warning in the log (a 262,728-token document over the tokenizer's 262,144 limit): the collator truncates documents to 512 tokens, so it is only a warning.

## Blockers / needs the owner
- Nothing now. Budget check for L2048 is due ≈ 2026-11-14 when L1024's stage 1 ends.
