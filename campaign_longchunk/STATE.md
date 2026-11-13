# STATE — campaign longchunk   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-13 19:00 (woken because the 1,024-token arm's stage 1 wrote its step-4,000 checkpoint. It is healthy. 5 of the control's 6 gating evals now have a judged result. No launch, no kill.)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- Stage-1 corpus = MS MARCO documents: 43.4% of documents ≥ 1,024 tokens, 20.4% ≥ 2,048 (107,120 of 3.2M sampled) — data job 188358, closed. SmolLM2 rejected (23.8% / 8.3% of 100,000 sampled) — data job 188339, closed.
- The 2,048-token chunk arm FITS on 4 × A100-80GB at batch 3 per GPU: 50/50 steps, ~72 GB per GPU, 58.6 s per step → full stage 1 = 84.8 h — job 188420, closed.
- The 1,024-token chunk arm FITS at batch 6 per GPU: 50/50 steps, ~71 GB per GPU, 42.1 s per step → full stage 1 = 60.9 h — job 188421, closed.
- Control stage 1 (256-token chunks) FINISHED: 1M documents (5,208 steps, 1.0 epoch) in 16:43 h, exit 0 — job 188422, closed. Held-out loss .801 / .877 / 1.007 at steps 2,000 / 4,000 / 5,208 (64 held-out rows); train loss .91–.96 at the end, no NaN.
- Control stage-1 slot ownership = .972 on text (3,053 test slots; gate ≥ .95; SmolLM2 reference .977) — job 188470, closed.
- Control stage 2 (images) FINISHED: 100k rows (1,562 steps, 1.0 epoch) in 19:12 h, exit 0; held-out loss .785 at the end, no NaN — job 188467, closed.
- Control finetune seed 42 FINISHED: 28k rows (437 steps, 1.0 epoch), exit 0; held-out loss .464 / .457 / .385 / .360 / .367 at steps 100 / 200 / 300 / 400 / 437 — job 188468, closed.
- Control finetune seed 43 FINISHED: 28k rows (437 steps, 1.0 epoch), exit 0; held-out loss .468 / .397 / .351 / .386 / .370 at the same steps — job 188469, closed.
- Control gating scores, judge accuracy ↑ (Qwen3-8B rubric judge, n = 1,500 questions per cell, rate 8 tokens per slot, 0 unparseable verdicts each). These are the comparison points; no arm has a score yet, so nothing is decided:

  | set | seed 42 | seed 43 |
  |---|---|---|
  | grouped HotpotQA g10 (one long chunk per question) | queued (188707) | .607 — job 188710, homer-019 |
  | documents `mm_docqa_1500_1m` | .615 — job 188708, homer-013 | .603 — job 188711, homer-016 |
  | photos `mm_naturalqa_1500` | .667 — job 188709, homer-014 | .670 — job 188712, attempt 16:02–16:22 on homer-013 |

  The photos s43 file was written 2 s before that attempt was preempted (file 16:22:26, preemption 16:22:28; 1,500 rows judged). The job requeued and will regenerate and overwrite it, so a byte-identical copy is kept: `expQ/longchunk/ft_cml256_s43/eval/mm_docqa_pisco_px1003520_mm_naturalqa_1500.json.kept_188712_homer013_1622`.

**Open, and running**
- provisional (unpaired, other corpus — never a comparison under the law): the MS MARCO control scores like the SmolLM2 reference (documents .615 / .603 vs .599 / .618; photos .667 / .670 vs .671 / .671; g10 s43 .607 vs reference s42 .617). Consistent with the MS MARCO stage 1 costing nothing downstream.
- provisional (n = 300 per cell, pairing points only): the control's step-100 previews (6,400 of 28k finetune rows): g10 .603 / .570, documents .597 / .603 (s42 / s43) — jobs 188981 / 188983 / 188980 / 188982. Each later arm's preview pairs against these.
- provisional: why the control's stage-1 held-out loss rises (+.077 by step 4,000, SmolLM2 reference −.003 at the same step). The 1,024-token arm does NOT rise over the same steps (.611 → .612). Cause untested; 64 held-out rows; losses are not comparable across chunk lengths.
- 1,024-token arm stage 1 — job 188466 RUNNING on homer-019, no restart: step 4,000 of 5,208 = 768,000 of 1M documents (.77 epoch), ≈ 0.80 B text tokens (estimate). Held-out loss .611 / .612 at steps 2,000 / 4,000, Rouge-L .790 / .767; train loss 2.17 (step 50) → .79 (step 4,000), grad norm .048, no NaN → healthy. ~41.7 s per step → ends ≈ 2026-11-14 08:50 (limit 2026-11-15 00:46).
- Chained behind it, PENDING on Dependency, they start by themselves when it exits 0: L1024 ownership probe 189691 → L1024 stage 2 189692 → finetune s42 189694 → finetune s43 189695; and the 2,048-token arm's stage 1 189693.
- Control g10 s42 (188707) and photos s43 (188712) are queued again on gpu-be after 4 and 5 preemptions (known_issues row 1). Not on the critical path: the 1,024-token arm's evals cannot start before ≈ 2026-11-15 evening.

## Owned jobs (jobid | arm / stage | state | progress | eta | note)
| job | arm / stage | state | progress | eta | note |
|---|---|---|---|---|---|
| 188466 | L1024 stage 1 | RUNNING (gpu, homer-019) | 4,000 / 5,208 steps = 768,000 / 1M documents | ≈ 2026-11-14 08:50 | `pt_text_4b_long.sh 1000000 1024 2048 6 8`; out `expQ/longchunk/pt_text_1M_cml1024_188466`; held-out eval + checkpoint every 2,000 steps, last one at the final step 5,208 |
| 189691 | L1024 ownership probe | PENDING afterok:188466 | — | ≈ 11-14 09:15 | `SETS=text probe_mixer.sh …188466/model 40` = the control's 188470 command; gpu-be a100, 0:10:00. Gate: ≥ .95 (control .972) |
| 189692 | L1024 stage 2 | PENDING afterok:188466 | — | start ≈ 11-14 09:00, end ≈ 11-15 04:30 | the control's 188467 argv + env `PT_DATA` (pdfa_1m_desc:50000, wiki_1m_desc:19000, natural_desc:31000); `PT_OUT=expQ/longchunk/pt_mm_cml1024`; 24:00:00 = 19:12 h × 1.25. **Check `training_dataset` in its `training_config.yaml` once it starts.** |
| 189694 | L1024 finetune s42 | PENDING afterok:189692 | — | ≈ 11-15 04:30 → 17:30 | the control's 188468 argv + env `FT_DATA` (docpages_20k_1m_teacher:20000, natural_8k_teacher:8000); `FT_OUT=expQ/longchunk/ft_cml1024_s42`; 16:06:00. Copy checkpoint-100 before step 300 |
| 189695 | L1024 finetune s43 | PENDING afterok:189692,afterany:189694 | — | ≈ 11-15 17:30 → 11-16 06:30 | same + `+hf_training.seed=43`; `FT_OUT=…/ft_cml1024_s43` |
| 189693 | L2048 stage 1 | PENDING afterok:188466 | — | ≈ 11-14 09:00 → 11-17 22:00 | `pt_text_4b_long.sh 1000000 2048 4096 3 16`; out `expQ/longchunk/pt_text_1M_cml2048_189693`; 4-10:00:00 = 58.6 s × 5,208 × 1.25. **Diff its `training_config.yaml` against 188422 once it starts.** |
| 188707 | C256 eval g10 s42 | PENDING (preempted 4 times) | — | — | `CML=4096 TOPK=10`, `text_hotpotqa_1500_g10`, limit 0:26 |
| 188712 | C256 eval photos s43 | PENDING (preempted 5 times; result already on disk, see TL;DR) | — | — | `PX=1003520`, `mm_naturalqa_1500`, limit 0:24 |
| 188708 / 188709 / 188710 / 188711 | C256 evals docs s42 / photos s42 / g10 s43 / docs s43 | COMPLETED exit 0 | 1,500 Q each, judged | done | results in TL;DR |

All evals: `--partition gpu-be --exclude 'homer-[005-009]'` (same-node rule: image cells pair only between agreeing el9_8 nodes; record the node of the attempt that wrote each file with `sacct -j <id> -D -o NodeList,End`). The law has no row to widen an eval to `gpu`.
Code md5, re-checked 2026-11-13 19:10 (unchanged since the control evals were submitted): `eval_mm_4b.sh` 50abc167, `scripts/eval_mm_docqa.py` dfe32aa0, `scripts/judge_mm_docqa.py` 02bc8748, `pisco/model.py` 994cff3d, `pt_mm_4b.sh` 3b63ee99, `ft_mm_4b.sh` 4522a9fc, recipe `expQ/longchunk/pt_text_4b_long.sh` 8d3745e9 (not in git — `expQ/` is gitignored). If the judge-only-resume hand-off changes `eval_mm_4b.sh`, the arm's evals run another md5 than the control's: record both and say so before pairing.
Preserved step-100 checkpoints: `expQ/longchunk/ft_cml256_s42_step100/checkpoint-100` and `…_s43_step100/checkpoint-100` (3.1 GB each, model files only).
Resolved-config diff, control 188422 vs the 1,024-token arm 188466: `compressor_max_length` 256→1024, `decoder_max_length` 512→2048, `per_device_train_batch_size` 24→6, `gradient_accumulation_steps` 2→8 (effective batch 192 held). Nothing else differs.

## Adopted jobs (found running in this project, not launched by the campaign — never killed, never duplicated)
(none. The freeze and probes campaigns' jobs belong to those sibling campaigns.)

## Hand-offs
- 5f4e2f → the pisco chats (2026-11-13 07:35): make `eval_mm_4b.sh` skip generation and only judge when the result file already holds every row. Not blocking. Today's preemptions cost ≈ 3.6 GPU-h of regenerated evals.

## Open arms and the next decision for each
| arm | chunk / text kept per doc (tokens) | text tokens at 1M docs | stage-1 cost (measured) | status | next decision |
|---|---|---|---|---|---|
| C256 control | 256 / 512 | ≈ 0.45 B | 67 GPU-h (done) | all training done; 5 of 6 gating evals judged; g10 s42 queued | read g10 s42 when it ends; g1 + MRQA (4 cells) only if an arm adopts |
| L1024 | 1,024 / 2,048 | ≈ 1.04 B (≈ 0.80 B so far, estimate) | 244 GPU-h (projected) | stage 1 at 77%, healthy; probe + stage 2 + 2 finetunes chained | when stage 2 starts: check `training_dataset`; ownership ≥ .95; after the s42 finetune writes checkpoint-100: copy, 2 previews, then the 3 s42 evals |
| L2048 | 2,048 / 4,096 | ≈ 1.30 B | 339 GPU-h (projected) | stage 1 chained (189693), budget rule passed at submit | when it starts: config diff vs 188422; held-out checks every 2,000 steps |
| SmolLM2 reference | 256 / 512 | ≈ 0.39 B | done | reported as a row only | — |

Known confound (report, do not fix): the long arms see 2.3× / 2.9× more text tokens than the control at the same 1M documents.

## Results table (deliverable; judge accuracy ↑, n = 1,500 Q per set unless marked; rate 8 tokens per slot)
| row | corpus | chunk (tokens) | slots per chunk | ownership ↑ | g10 s42 / s43 | documents s42 / s43 | photos s42 / s43 |
|---|---|---|---|---|---|---|---|
| SmolLM2 reference (183814 / 183816) | SmolLM2 | 256 | 33 | .977 | .617 / — | .599 / .618 | .671 / .671 |
| C256 control (188468 / 188469) | MS MARCO | 256 | 33 | .972 | queued / .607 | .615 / .603 | .667 / .670 |
| L1024 | MS MARCO | 1,024 | 129 | — | — | — | — |
| L2048 | MS MARCO | 2,048 | 257 | — | — | — | — |
| uncompressed decoder | — | — | — | — | — | .809 | .717 |

Paired difference to the control (arm − C256, same seed, `scripts/paired_judge.py`): none yet.

## Budget (38 GPU-days in the law)
- Spent: ≈ 7 GPU-h (fit tests) + 66.9 (control stage 1) + 0.05 (ownership probe) + ≈ 186 (L1024 so far, 46.5 h × 4) + 19.2 (control stage 2) + 23.9 (control finetunes) + ≈ 5.2 (previews and eval attempts, preempted ones included) ≈ 308 GPU-h = 12.8 GPU-days.
- Projected left with measured speeds: L1024 stage 1 ≈ 56 + L1024 downstream ≈ 61 + L2048 ≈ 400 + control evals ≈ 1 ≈ 518 → total ≈ 826 GPU-h = 34.4 GPU-days ≤ 38. Slack ≈ 3.6 GPU-days: a tie's third seed (≈ 3) fits, just.
- Train cap (CAMP_MAX_TRAIN = 2 running at once): L2048 stage 1 runs alongside L1024 stage 2, then alongside ONE L1024 finetune at a time (s43 waits for s42).

## Blockers / needs the owner
- Nothing now.
- Next wake: when L1024's stage 2 or L2048's stage 1 writes its `training_config.yaml` (both start only after stage 1 188466 exits 0, ≈ 11-14 09:00), max 16 h; a failure of 188466 wakes earlier (new failure). At that wake: check stage 2's `training_dataset`; diff L2048's config against 188422; read the ownership probe 189691; `expmon probe` 188466's final held-out loss.
