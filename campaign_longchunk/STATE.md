# STATE — campaign longchunk   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-11 04:00 (woken because both fit-and-speed tests left the queue)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Stage-1 corpus = MS MARCO documents: 43.4% of documents ≥ 1,024 tokens, 20.4% ≥ 2,048 (107,120 of 3.2M sampled) — data job 188358, closed. SmolLM2 rejected (23.8% / 8.3% of 100,000 sampled) — data job 188339, closed.
- The 2,048-token chunk arm FITS on 4 × A100-80GB at batch 3 per GPU: 50/50 steps, exit 0, peak GPU memory 288 GB summed over 4 GPUs (~72 GB each, sacct), 58.6 s per step (steps 10–49 of `pt_text_long_188420.out`) → a full stage 1 (5,208 steps, 1M documents) = 84.8 h — job 188420, closed.
- The 1,024-token chunk arm FITS at batch 6 per GPU: 50/50 steps, exit 0, ~71 GB per GPU, 42.1 s per step (`pt_text_long_188421.out`) → full stage 1 = 60.9 h — job 188421, closed.
- No accuracy result yet: no arm has a judged number.

**Open, and running**
- Control stage 1 (256-token chunks, 1M documents): job 188422 RUNNING since 03:47, 67 of 5,208 steps (12,864 documents) at 04:05, 11.6 s per step → ends ≈ 2026-11-11 20:45; first held-out loss at step 2,000 ≈ 10:15.
- Chained behind it, all waiting on dependencies (submitted this wake):
  - 1,024-token arm stage 1 — job 188466, starts when the control's stage 1 succeeds, 76 h limit.
  - Control stage 2 (images) — job 188467 → its two finetunes, seed 42 job 188468 and seed 43 job 188469.
  - Slot-ownership health probe on the control's stage-1 model — job 188470.

## Owned jobs (jobid | arm / stage | state | progress | eta | note)
| job | arm / stage | state | progress | eta | note |
|---|---|---|---|---|---|
| 188422 | C256 stage 1 | RUNNING (gpu, homer-019) | 67 / 5,208 steps = 12,864 / 1M docs | ≈ 20:45 today | `pt_text_4b_long.sh 1000000 256 512 24 2`; out `expQ/longchunk/pt_text_1M_cml256_188422`; limit 23 h (16.8 h needed) |
| 188466 | L1024 stage 1 | PENDING (afterok 188422) | 0 / 1M docs | ≈ 2026-11-14 10:00 | `… 1000000 1024 2048 6 8`; limit 3-04:15 = 42.1 s × 5,208 × 1.25; out `expQ/longchunk/pt_text_1M_cml1024_188466` |
| 188467 | C256 stage 2 | PENDING (afterok 188422) | 0 / 100k rows | ≈ 2026-11-12 16:00 | 183537's argv + env `PT_DATA` (pdfa 50k, wiki 19k, natural 31k); out `expQ/longchunk/pt_mm_cml256`; limit 23:51 |
| 188468 | C256 finetune s42 | PENDING (afterok 188467) | 0 / 28k rows | ≈ 2026-11-13 03:00 | 183814's argv + env `FT_DATA` (docpages 20k + natural 8k); out `expQ/longchunk/ft_cml256_s42`; limit 14:03 |
| 188469 | C256 finetune s43 | PENDING (afterok 188467) | 0 / 28k rows | ≈ 2026-11-13 03:00 | same + `+hf_training.seed=43`; out `expQ/longchunk/ft_cml256_s43` |
| 188470 | C256 ownership probe | PENDING (afterok 188422) | — | ≈ 21:00 today | `SETS=text probe_mixer.sh <stage-1 model> 40` (= 184705, SmolLM2 reference .977); limit 10 min |

Recipe file `expQ/longchunk/pt_text_4b_long.sh` md5 8d3745e9cc343b6fb30d8d4a2bc2cd8c (unchanged; not in git — `expQ/` is gitignored, griped 2026-11-11 02:00).
Resolved-config diff, control (188422) vs the 1,024-token arm (smoke 188421): `compressor_max_length` 256→1024, `decoder_max_length` 512→2048, `per_device_train_batch_size` 24→6, `gradient_accumulation_steps` 2→8 (effective batch 192 held); the smoke's `max_steps: 50` only. The 2,048-token arm adds 1024→2048, 2048→4096, 6→3, 8→16. Nothing else differs.

## Adopted jobs (found running in this project, not launched by the campaign — never killed, never duplicated)
(none. The eval_mm / probe_attn_route / route_fusion jobs 188439–188461 belong to the sibling campaign `probes`.)

## Open arms and the next decision for each
| arm | chunk / text kept per doc (tokens) | text tokens at 1M docs | stage-1 cost (measured) | status | next decision |
|---|---|---|---|---|---|
| C256 control | 256 / 512 | ≈ 0.45 B | 67 GPU-h | stage 1 running, stage 2 + finetunes + probe chained | step 2,000 held-out loss: collapse (flat / NaN) → kill 188422 and every dependent; else evals of both finetunes (g10, g1, MRQA, documents, photos + judge) once 188468/188469 exist |
| L1024 | 1,024 / 2,048 | ≈ 1.04 B | 244 GPU-h | stage 1 chained (188466) | after its stage 1: ownership probe, then the control's exact stage 2 → finetunes → evals |
| L2048 | 2,048 / 4,096 | ≈ 1.30 B | 339 GPU-h | fits; not launched (one stage 1 at a time) | launch after L1024's stage 1 ONLY if the budget allows (see below) |
| SmolLM2 reference | 256 / 512 | ≈ 0.39 B | done | reported as a row only | — |

Known confound (report, do not fix): the long arms see 2.3× / 2.9× more text tokens than the control at the same 1M documents.

## Budget (35 GPU-days in the law)
- Spent: ≈ 7 GPU-h (the two fit tests) + the control's stage 1 in progress.
- Projected with measured speeds: control 67 + L1024 244 + L2048 339 GPU-h of stage 1, ≈ 61 GPU-h downstream per arm, → ≈ 841 GPU-h = **35.0 GPU-days = the whole budget, no slack** (a third seed after a tie costs ≈ 3 more).
- Control + L1024 alone ≈ 18.4 GPU-days. The law now says (tightened this wake): L2048 launches only if spent + projected ≤ 35 GPU-days at that moment, else BLOCKED.md.

## Health evidence this wake (numbers, not adjectives)
- Both fit tests: GPU utilisation 99.5–99.7% (sacct gpuutil 398–399 of 400), host RAM 96–108 GB of 160 GB.
- Control: train loss 2.301 at step 50 (warm-up, lr 9.4e-6); no held-out loss yet (first at step 2,000).
- The control's 23 h limit holds: measured 11.6 s per step → 16.8 h plus evals.
- Stage-2 and finetune env vars (`PT_DATA`, `FT_DATA`) were recovered from the reference runs' `training_config.yaml`; the law's command lines omitted them (law clarified). Verify `training_dataset` in 188467's `training_config.yaml` when it starts.

## Blockers / needs the owner
- Nothing now. Heads-up, decision due ≈ 2026-11-14 when L1024's stage 1 ends: with measured speeds the full plan uses the whole 35 GPU-day budget. If the projection still exceeds 35 then, the wake writes BLOCKED.md: raise the budget (≈ +3 for slack) or close L2048 as over budget.
