# STATE — campaign qafreeze   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-18 17:30-17:50. Question: can a compressor that HAS had one QA finetune (R = 184778 / 184780)
stay frozen while a fresh decoder LoRA trains on the same 28k rows? (arm Q)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- The hybrid start model can be built from saved checkpoints, no new code: init_s42 / init_s43 = R's compressor + connector + config byte-identical (md5), decoder = the stage-2 decoder byte-identical (md5 dca63864); same 696 decoder tensor names, shapes, dtypes as R — job 191450, closed.
- Feasibility gate PASS (all three checks, every tensor compared): R s42 → init_s42 compressor and connector rows changed 0.0%, decoder lora_B 68.6% (job 191451); stage-2 → init_s42 decoder rows 0.0%, compressor lora_B 73.8% (job 191452); seed 43 same picture, decoder lora_B 67.5% (job 191453).
- Plumbing check PASS on the 20-step smoke, seed 42 (1,280 rows = .046 epoch; n = 1 smoke): froze 506 compressor + connector tensors, compressor trainable 0; loss step 10 .555 / step 20 .515 vs 188528 .594 / .584 (inside ±.10) — job 191493, closed; smoke weights: compressor / connector 0.0%, decoder lora_B 76.0% — job 191494, closed.
- After the full epoch (28k rows) the frozen side did not move: every compressor and connector row changed 0.0%, decoder lora_B 66.9% at both seeds (F full runs 66.3-66.7%) — jobs 191625 (s42), 191627 (s43), closed.
- Reference R re-evaluated on today's eval launcher (md5 b3d1cda1), 1,500 questions each, every row judged, 0 unparseable: documents .610 (s42, job 191495) / .615 (s43, job 191496), the same answers as the recorded files (0 of 1,500 differ at both seeds); photos .679 (s42, job 191531) / .685 (s43, job 191532).
- Arm Q documents, 1,500 questions each, healthy (every row judged, 0 unparseable, px 1003520, homer-017): judge .615 (s42, job 191626) / .613 (s43, job 191628), closed.
- No-op control of the instrument: R s42 documents paired with itself = +0.000, 0 discordant (campaign_qafreeze/pairs/docs_noop_R42.txt).

**Open, and running**
- provisional: documents Q − R = +0.5 pt [−0.6, +1.6], McNemar p .470 (s42) and −0.1 pt [−1.1, +0.9], p .896 (s43), n = 1,500 pairs each (jobs 191626 vs 191495, 191628 vs 191496; pairs/docs_QminusR_s4x.txt). Inside the gate (≥ −2 pt, CI not wholly below 0) at both seeds and inside the prediction [−1, +2] pt. Not a gate result until photos and MRQA are in.
- provisional: documents Q − F = +3.3 pt [+1.1, +5.3] (s42) and +3.3 pt [+1.1, +5.7] (s43), p .003 each, n = 1,500 — F's files ran on the older launcher (50abc167), so the law's control waits for the F re-runs.
- provisional: the held-out loss of Q ends below R's at both seeds and well below F's (table below). The loss is not the gate.
- Arm Q photos s42 / s43 — jobs 191659 / 191660 RUNNING (gpu-be); pair with R re-runs 191531 / 191532.
- Arm Q MRQA text s42 / s43 — jobs 191661 / 191662 RUNNING (gpu-be); the R side must be re-run on today's launcher (next wake).
- Arm Q trainings 191527 / 191530: COMPLETED, no verdict until the gate.

## Deliverable table (filled as cells land)
Judge accuracy ↑ (Qwen3-8B rubric), seed 42 / seed 43; paired difference to R in pt (95% CI, McNemar p); n = 1,500 per set.
Compression: 8 tokens per slot, ~123 slots per 1M-px page.
| model | what trains | documents ↑ | documents − R (pt) | photos ↑ | MRQA text ↑ | HotpotQA ↑ (not gating) |
|---|---|---|---|---|---|---|
| R reference | compressor LoRA + connector + decoder LoRA | .610 / .615 (today's code) | — | .679 / .685 (today's code) | recorded .718 / .724 (old code; re-run next wake) | recorded .603 / .593 (old code) |
| F frozen never-QA compressor (campaign freeze) | decoder LoRA | .582 / .580 (old code; re-run next wake) | recorded −2.8 / −3.5 | — | — | recorded −3.0 / −2.8 pt |
| **Q frozen QA-trained compressor** | decoder LoRA | **.615 / .613** | provisional: +0.5 [−0.6, +1.6] p .47 / −0.1 [−1.1, +0.9] p .90 | running | running | not launched |
| uncompressed decoder | — | recorded .809 | — | recorded .717 | recorded .807 | — |

Per source, documents Q − R (pt): infovqa −0.2 / −1.2, mpdocvqa +1.8 / 0.0, slidevqa −0.2 / +0.8 (500 each, no CI excludes 0).
Q and R answer almost the same way: Q right where R is wrong on 38 / 28 questions, the reverse on 31 / 30.

## Training health — held-out eval_loss by epoch (lower is better; 1 epoch = 28k rows)
| run | .23 ep | .46 ep | .69 ep | .92 ep | 1.0 ep |
|---|---|---|---|---|---|
| Q s42 (191527) | .413 | .411 | .398 | .400 | .400 |
| R s42 (184778) | .513 | .455 | .393 | .394 | .414 |
| F s42 (freeze ft_frozen_s42) | .455 | .444 | .438 | .434 | .435 |
| Q s43 (191530) | .399 | .382 | .375 | .372 | .371 |
| R s43 (184780) | .471 | .439 | .411 | .441 | .375 |
| F s43 (freeze ft_frozen_s43) | .442 | .431 | .416 | .415 | .413 |
No NaN; no probe above the run's first probe. Read from each run's `checkpoint-437/trainer_state.json`.

## Module movement (entries changed %, against that run's init)
| run | compressor.* | connector.* | decoder lora_B | decoder lora_A |
|---|---|---|---|---|
| Q s42 (check 191625) | 0.0 | 0.0 | 66.9 | 14.8 |
| Q s43 (check 191627) | 0.0 | 0.0 | 66.9 | 14.8 |
| F full runs (campaign freeze) | 0.0 | 0.0 | 66.3-66.7 | — |

## Training cost
| run | wall | s/step | peak GPU memory |
|---|---|---|---|
| Q s42 (191527) | 8:58:42 | 73.6 | 49.9 GB (sacct gpumem; GPU util max 92%) |
| Q s43 (191530) | 9:17:07 | 76.2 | 49.9 GB (GPU util max 93%) |
| F (campaign freeze) | 8.7 / 9.3 / 8.7 h | 71.5-76.1 | 49.7 GB (smoke) |
| R (184778 / 184780) | 11.0 / 13.8 h | 88-90 (smoke) | 31.9 GB (smoke) |

## Config diff of the full cells (law: only the allowed fields)
`diff expQ/ft_mm_4b_184778/training_config.yaml expQ/qafreeze/ft_frozen_s42/training_config.yaml` → `+freeze_compressor: true`,
`model_name_or_path: …/archq3/pt_mm_5M_spread/model → …/qafreeze/init_s42`. Same two lines for s43 vs 184780 (init_s43). Nothing else.

## Owned jobs
| job | arm | state | elapsed | note |
|---|---|---|---|---|
| 191450-191453 | B build + checks | COMPLETED 0:0 | ≤ 3:07 | closed; gate PASS |
| 191493 / 191494 | Qs smoke + check | COMPLETED 0:0 | 29:19 / 0:44 | closed; PASS |
| 191495 / 191496 | R documents s42 / s43 | COMPLETED 0:0 | 27:13 / 28:04 | closed; .610 / .615 |
| 191531 / 191532 | R photos s42 / s43 | COMPLETED 0:0 | 18:59 / 18:10 | closed; .679 / .685 |
| 191527 / 191530 | Q full s42 / s43 | COMPLETED 0:0 | 8:58:42 / 9:17:07 | open until the gate |
| 191625 / 191627 | Q weight checks | COMPLETED 0:0 | ~1 min | closed; PASS |
| 191626 / 191628 | Q documents s42 / s43 | COMPLETED 0:0 | 28:51 / 28:42 | closed; .615 / .613 |
| 191659 / 191660 | Q photos s42 / s43 | RUNNING homer-017 / homer-020 | started 17:32 | --time 0:27; PX=1003520 |
| 191661 / 191662 | Q MRQA s42 / s43 | RUNNING homer-020 / homer-010 | started 17:33 | --time 0:27; PX=1003520 CML=4096 |

## Adopted jobs
(none) — other jobs in the queue belong to sibling campaigns (never touched).

## Next — the remaining 8 eval cells, then the gate
All `eval_mm_4b.sh <model> pisco '' eval_data/<set>`, `--partition gpu-be --exclude 'homer-[005-009]'`, env as the
recorded cells (photos 188602/188604, MRQA 188606/188613, HotpotQA 188615/188617):
| wake | cells | env | --time |
|---|---|---|---|
| next (when 191659-191662 end) | R MRQA `expQ/qafreeze/ref_s42/model`, `ref_s43/model` (`text_mrqa_ood_1500`); F documents `expQ/qafreeze/f_s42/model`, `f_s43/model` (`mm_docqa_1500_1m`); pair Q − R photos | MRQA `PX=1003520 CML=4096`; documents `PX=1003520` | 0:27 / 0:40 |
| +1 | gate read on documents, photos, MRQA; Q − F control; launch Q HotpotQA s42/s43 + R HotpotQA ref_s42/ref_s43 (`text_hotpotqa_1500`, reported only) | `TOPK=10 PX=1003520 CML=4096` | 0:29 |
| +2 | HotpotQA pairs; final table; on PASS the hand-off + ONE owner question (CAMPAIGN.md "On acceptance") | — | — |

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| Q s42, s43 | documents provisional inside the gate; photos + MRQA running | gate when documents, photos, MRQA are judged at both seeds |
| R reference | documents + photos re-run; MRQA + HotpotQA to re-run | next wake / +1 |
| F re-runs (documents s42 / s43) | not launched | next wake; needed for the Q − F control |
| Q3 seed 44 | only on a tie | not needed if photos + MRQA pass at both seeds |
| H lr 1e-4 | only after the owner checkpoint | no ARM case yet |

## Health evidence this wake (numbers, not adjectives)
- Code md5 now = last wake (`campaign_qafreeze/code_md5.txt`): eval_mm_4b.sh b3d1cda1, eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py 02bc8748, paired_judge.py a3cc13b2, pisco/model.py 994cff3d, ft_qafreeze.sbatch fd29d048, check_module_move.sh b6c5f6f9.
- `paired_judge.py` prints `diff = A − B`: every pair is run with Q as A (files in campaign_qafreeze/pairs/).
- GPU-h spent ≈ 21.3 of 55 (sensor gpuh24 21.1 + the two documents cells). The 4 running + 8 remaining evals ≈ 4 → ~25.5 expected.
- Disk 7.7 T free.

## Blockers / needs the owner
(none)
