# STATE — campaign longchunk   (rewritten WHOLESALE by every LLM wake; never append)

last wake: 2026-11-13 20:30 (woken by "problems" = the control's evals being preempted and requeued on gpu-be, a known benign issue. The photos seed-43 eval then FINISHED cleanly; the g10 seed-42 eval wrote a complete result just before another preemption. The 1,024-token arm's stage 1 is at 80%, healthy. No launch, no kill.)

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
- Control gating scores from FINISHED eval jobs (exit 0), judge accuracy ↑ (Qwen3-8B rubric judge, n = 1,500 questions per cell, rate 8 tokens per slot, 0 unparseable verdicts each). These are the comparison points; no arm has a score yet, so nothing is decided:

  | set | seed 42 | seed 43 |
  |---|---|---|
  | grouped HotpotQA g10 (one long chunk per question) | see Open (job 188707 not finished) | .607 — job 188710, homer-019 |
  | documents `mm_docqa_1500_1m` | .615 — job 188708, homer-013 | .603 — job 188711, homer-016 |
  | photos `mm_naturalqa_1500` | .667 — job 188709, homer-014 | .670 — job 188712, homer-014 (COMPLETED 20:19 after 6 preemptions) |

  Photos seed 43: the finished run on homer-014 gives 1,500 / 1,500 answers identical to the earlier attempt on homer-013 (kept copy `…mm_naturalqa_1500.json.kept_188712_homer013_1622`), so those two nodes agree on this set.

**Open, and running**
- provisional (the job has not ended): control g10 seed 42 = .611 (n = 1,500, 0 unparseable, EM .345). The file was written 20:10:37 by the attempt 19:52–20:10 on homer-014, 2 s before that attempt was preempted. Job 188707 is queued again (6th requeue, estimated start 11-14 00:30) and will overwrite it; a byte-identical copy is kept as `expQ/longchunk/ft_cml256_s42/eval/mm_docqa_pisco_k10_cml4096_text_hotpotqa_1500_g10.json.kept_188707_homer014_2010`.
- provisional (unpaired, other corpus — never a comparison under the law): the MS MARCO control scores like the SmolLM2 reference (documents .615 / .603 vs .599 / .618; photos .667 / .670 vs .671 / .671; g10 .611 (provisional) / .607 vs reference s42 .617). Consistent with the MS MARCO stage 1 costing nothing downstream.
- provisional (n = 300 per cell, pairing points only): the control's step-100 previews (6,400 of 28k finetune rows): g10 .603 / .570, documents .597 / .603 (s42 / s43) — jobs 188981 / 188983 / 188980 / 188982. Each later arm's preview pairs against these.
- provisional: why the control's stage-1 held-out loss rises (+.077 by step 4,000, SmolLM2 reference −.003 at the same step). The 1,024-token arm does NOT rise over the same steps (.611 → .612). Cause untested; 64 held-out rows; losses are not comparable across chunk lengths.
- 1,024-token arm stage 1 — job 188466 RUNNING on homer-019, no restart: step 4,152 of 5,208 at 20:35 = 797,184 of 1M documents (.80 epoch), ≈ 0.83 B text tokens (estimate). Last held-out check (step 4,000): loss .612, Rouge-L .767; train loss .79, grad norm .048, no NaN → healthy. 41.8 s per step → ends ≈ 2026-11-14 08:50 (limit 2026-11-15 00:46).
- Chained behind it, PENDING on Dependency, they start by themselves when it exits 0: L1024 ownership probe 189691 → L1024 stage 2 189692 → finetune s42 189694 → finetune s43 189695; and the 2,048-token arm's stage 1 189693.

## Owned jobs (jobid | arm / stage | state | progress | eta | note)
| job | arm / stage | state | progress | eta | note |
|---|---|---|---|---|---|
| 188466 | L1024 stage 1 | RUNNING (gpu, homer-019) | 4,152 / 5,208 steps = 797,184 / 1M documents | ≈ 2026-11-14 08:50 | `pt_text_4b_long.sh 1000000 1024 2048 6 8`; out `expQ/longchunk/pt_text_1M_cml1024_188466`; held-out eval + checkpoint every 2,000 steps, last one at the final step 5,208 |
| 189691 | L1024 ownership probe | PENDING afterok:188466 | — | ≈ 11-14 09:15 | `SETS=text probe_mixer.sh …188466/model 40` = the control's 188470 command; gpu-be a100, 0:10:00. Gate: ≥ .95 (control .972) |
| 189692 | L1024 stage 2 | PENDING afterok:188466 | — | start ≈ 11-14 09:00, end ≈ 11-15 04:30 | the control's 188467 argv + env `PT_DATA` (pdfa_1m_desc:50000, wiki_1m_desc:19000, natural_desc:31000); `PT_OUT=expQ/longchunk/pt_mm_cml1024`; 24:00:00 = 19:12 h × 1.25. **Check `training_dataset` in its `training_config.yaml` once it starts.** |
| 189694 | L1024 finetune s42 | PENDING afterok:189692 | — | ≈ 11-15 04:30 → 17:30 | the control's 188468 argv + env `FT_DATA` (docpages_20k_1m_teacher:20000, natural_8k_teacher:8000); `FT_OUT=expQ/longchunk/ft_cml1024_s42`; 16:06:00. Copy checkpoint-100 before step 300 |
| 189695 | L1024 finetune s43 | PENDING afterok:189692,afterany:189694 | — | ≈ 11-15 17:30 → 11-16 06:30 | same + `+hf_training.seed=43`; `FT_OUT=…/ft_cml1024_s43` |
| 189693 | L2048 stage 1 | PENDING afterok:188466 | — | ≈ 11-14 09:00 → 11-17 22:00 | `pt_text_4b_long.sh 1000000 2048 4096 3 16`; out `expQ/longchunk/pt_text_1M_cml2048_189693`; 4-10:00:00 = 58.6 s × 5,208 × 1.25. **Diff its `training_config.yaml` against 188422 once it starts.** |
| 188707 | C256 eval g10 s42 | PENDING (preempted 6 times, Restarts=5; complete result already on disk, see Open) | — | est. start 11-14 00:30 | `CML=4096 TOPK=10`, `text_hotpotqa_1500_g10`, limit 0:26. Left queued so the cell ends COMPLETED |
| 188708 / 188709 / 188710 / 188711 / 188712 | C256 evals docs s42 / photos s42 / g10 s43 / docs s43 / photos s43 | COMPLETED exit 0 | 1,500 Q each, judged | done | results in TL;DR |

All evals: `--partition gpu-be --exclude 'homer-[005-009]'` (same-node rule: image cells pair only between agreeing el9_8 nodes; record the node of the attempt that wrote each file with `sacct -j <id> -D -X -o NodeList,Start,End`). The law has no row to widen an eval to `gpu`.
Code md5, re-checked 2026-11-13 20:35 (unchanged since the control evals were submitted): `eval_mm_4b.sh` 50abc167, `scripts/eval_mm_docqa.py` dfe32aa0, `scripts/judge_mm_docqa.py` 02bc8748, `pisco/model.py` 994cff3d, `pt_mm_4b.sh` 3b63ee99, `ft_mm_4b.sh` 4522a9fc, recipe `expQ/longchunk/pt_text_4b_long.sh` 8d3745e9 (not in git — `expQ/` is gitignored). `git status` shows uncommitted edits to `eval_mm_4b.sh`, `scripts/eval_mm_docqa.py` and `pisco/model.py` from the sibling chats: if their md5 differs when the L1024 evals are submitted, record both and do not pair until both sides ran the same code.
Preserved step-100 checkpoints: `expQ/longchunk/ft_cml256_s42_step100/checkpoint-100` and `…_s43_step100/checkpoint-100` (3.1 GB each, model files only).
Resolved-config diff, control 188422 vs the 1,024-token arm 188466: `compressor_max_length` 256→1024, `decoder_max_length` 512→2048, `per_device_train_batch_size` 24→6, `gradient_accumulation_steps` 2→8 (effective batch 192 held). Nothing else differs.

## Adopted jobs (found running in this project, not launched by the campaign — never killed, never duplicated)
(none. The freeze and probes campaigns' jobs belong to those sibling campaigns.)

## Hand-offs
- 5f4e2f → the pisco chats (2026-11-13 07:35): make `eval_mm_4b.sh` skip generation and only judge when the result file already holds every row. Not blocking. Preemptions have now cost ≈ 4.7 GPU-h of regenerated control evals.

## Open arms and the next decision for each
| arm | chunk / text kept per doc (tokens) | text tokens at 1M docs | stage-1 cost (measured) | status | next decision |
|---|---|---|---|---|---|
| C256 control | 256 / 512 | ≈ 0.45 B | 67 GPU-h (done) | all training done; 5 of 6 gating evals from finished jobs, g10 s42 .611 on disk (provisional) | confirm g10 s42 when 188707 ends; g1 + MRQA (4 cells) only if an arm adopts |
| L1024 | 1,024 / 2,048 | ≈ 1.04 B (≈ 0.83 B so far, estimate) | 244 GPU-h (projected) | stage 1 at 80%, healthy; probe + stage 2 + 2 finetunes chained | when stage 2 starts: check `training_dataset`; ownership ≥ .95; after the s42 finetune writes checkpoint-100: copy, 2 previews, then the 3 s42 evals |
| L2048 | 2,048 / 4,096 | ≈ 1.30 B | 339 GPU-h (projected) | stage 1 chained (189693), budget rule passed at submit | when it starts: config diff vs 188422; held-out checks every 2,000 steps |
| SmolLM2 reference | 256 / 512 | ≈ 0.39 B | done | reported as a row only | — |

Known confound (report, do not fix): the long arms see 2.3× / 2.9× more text tokens than the control at the same 1M documents.

## Results table (deliverable; judge accuracy ↑, n = 1,500 Q per set unless marked; rate 8 tokens per slot)
| row | corpus | chunk (tokens) | slots per chunk | ownership ↑ | g10 s42 / s43 | documents s42 / s43 | photos s42 / s43 |
|---|---|---|---|---|---|---|---|
| SmolLM2 reference (183814 / 183816) | SmolLM2 | 256 | 33 | .977 | .617 / — | .599 / .618 | .671 / .671 |
| C256 control (188468 / 188469) | MS MARCO | 256 | 33 | .972 | .611 (provisional) / .607 | .615 / .603 | .667 / .670 |
| L1024 | MS MARCO | 1,024 | 129 | — | — | — | — |
| L2048 | MS MARCO | 2,048 | 257 | — | — | — | — |
| uncompressed decoder | — | — | — | — | — | .809 | .717 |

Paired difference to the control (arm − C256, same seed, `scripts/paired_judge.py`): none yet.

## Budget (38 GPU-days in the law)
- Spent: ≈ 7 GPU-h (fit tests) + 66.9 (control stage 1) + 0.05 (ownership probe) + ≈ 192 (L1024 so far, 48 h × 4) + 19.2 (control stage 2) + 23.9 (control finetunes) + ≈ 6.3 (previews and eval attempts, preempted ones included) ≈ 315 GPU-h = 13.1 GPU-days.
- Projected left with measured speeds: L1024 stage 1 ≈ 49 + L1024 downstream ≈ 61 + L2048 ≈ 400 + control eval ≈ 0.5 ≈ 511 → total ≈ 826 GPU-h = 34.4 GPU-days ≤ 38. Slack ≈ 3.6 GPU-days: a tie's third seed (≈ 3) fits, just.
- Train cap (CAMP_MAX_TRAIN = 2 running at once): L2048 stage 1 runs alongside L1024 stage 2, then alongside ONE L1024 finetune at a time (s43 waits for s42).

## Blockers / needs the owner
- Nothing now.
- Next wake: when L1024's stage 2 or L2048's stage 1 writes its `training_config.yaml` (both start only after stage 1 188466 exits 0, ≈ 11-14 09:00), max 16 h; a failure of 188466 wakes earlier (new failure). At that wake: check stage 2's `training_dataset`; diff L2048's config against 188422; read the ownership probe 189691; `expmon probe` 188466's final held-out loss; confirm g10 s42 if 188707 has ended.
