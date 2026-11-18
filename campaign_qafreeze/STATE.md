# STATE — campaign qafreeze   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-18 16:30-16:50. Question: can a compressor that HAS had one QA finetune (R = 184778 / 184780)
stay frozen while a fresh decoder LoRA trains on the same 28k rows? (arm Q)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- The hybrid start model can be built from saved checkpoints, no new code: init_s42 / init_s43 = R's compressor + connector + config byte-identical (md5), decoder = the stage-2 decoder byte-identical (md5 dca63864); same 696 decoder tensor names, shapes, dtypes as R — job 191450, closed.
- Feasibility gate PASS (all three checks, every tensor compared): R s42 → init_s42 compressor and connector rows changed 0.0%, decoder lora_B 68.6% (job 191451); stage-2 → init_s42 decoder rows 0.0%, compressor lora_B 73.8% (job 191452); seed 43 same picture, decoder lora_B 67.5% (job 191453).
- Plumbing check PASS on the 20-step smoke, seed 42 (1,280 rows = .046 epoch; n = 1 smoke): froze 506 compressor + connector tensors, compressor trainable 0; loss step 10 .555 / step 20 .515 vs 188528 .594 / .584 (inside ±.10) — job 191493, closed; smoke weights: compressor / connector 0.0%, decoder lora_B 76.0% — job 191494, closed.
- Arm Q trained one full epoch at both seeds (28k rows each, COMPLETED 0:0): log `froze 506 compressor + connector tensors`, `Compressor trainable parameters: 0`, trainable 155,978,240 = decoder only; 73.6 s/step (s42, 8:58:42) and 76.2 s/step (s43, 9:17:07) — jobs 191527, 191530 (no verdict yet: open until judged).
- After the full epoch the frozen side did not move: every compressor and connector row changed 0.0%, decoder lora_B 66.9% at both seeds (F full runs 66.3-66.7%) — jobs 191625 (s42), 191627 (s43), closed.
- Reference R re-evaluated on today's eval launcher (md5 b3d1cda1), 1,500 questions each, every row judged, 0 unparseable: documents .610 (s42, job 191495) / .615 (s43, job 191496), equal to the recorded .610 / .615; photos .679 (s42, job 191531, recorded .673) / .685 (s43, job 191532, recorded .685).

**Open, and running**
- Arm Q documents eval, seed 42 — job 191626 RUNNING (gpu-be, homer-017); pairs with R re-run 191495.
- Arm Q documents eval, seed 43 — job 191628 RUNNING (gpu-be, homer-017); pairs with R re-run 191496.
- provisional: the held-out loss of Q ends below R's at both seeds and well below F's (table below) — consistent with the prediction that Q ties R; the loss is not the gate.

## Training health — held-out eval_loss by epoch (lower is better; 1 epoch = 28k rows)
| run | .23 ep | .46 ep | .69 ep | .92 ep | 1.0 ep |
|---|---|---|---|---|---|
| Q s42 (191527) | .413 | .411 | .398 | .400 | .400 |
| R s42 (184778) | .513 | .455 | .393 | .394 | .414 |
| F s42 (freeze ft_frozen_s42) | .455 | .444 | .438 | .434 | .435 |
| Q s43 (191530) | .399 | .382 | .375 | .372 | .371 |
| R s43 (184780) | .471 | .439 | .411 | .441 | .375 |
| F s43 (freeze ft_frozen_s43) | .442 | .431 | .416 | .415 | .413 |
No NaN; no probe above the run's first probe. Read from each run's `checkpoint-437/trainer_state.json`; Q values recorded with `expmon probe`.
The step-~100 preview (pre-authorised) was not run: the trainings finished during the 16 h pace hold, so the full evals replace it.

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
| 191527 / 191530 | Q full s42 / s43 | COMPLETED 0:0 | 8:58:42 / 9:17:07 | open; homer-019; out expQ/qafreeze/ft_frozen_s4x/model |
| 191625 / 191627 | Q weight checks s42 / s43 | COMPLETED 0:0 | ~1 min | closed; PASS |
| 191626 | Q documents s42 | RUNNING homer-017 | started 16:37 | --time 0:40; → expQ/qafreeze/ft_frozen_s42/eval/mm_docqa_pisco_px1003520_mm_docqa_1500_1m.json |
| 191628 | Q documents s43 | RUNNING homer-017 | started 16:37 | same, ft_frozen_s43 |

## Adopted jobs
(none) — 191629-191633 in the queue are sibling campaigns' (never touched).

## Next — the remaining 12 eval cells, 4 per wake (CAMP_MAX_SUBMIT_PER_WAKE = 4)
All `eval_mm_4b.sh <model> pisco '' eval_data/<set>`, `--partition gpu-be --exclude 'homer-[005-009]'`, env as recorded (FACTORS lines of 188597 / 188602 / 188606 / 188615):
| wake | cells | env | --time |
|---|---|---|---|
| next (when 191626 / 191628 end) | Q photos s42, s43 (`mm_naturalqa_1500`); Q MRQA s42, s43 (`text_mrqa_ood_1500`); pair Q − R documents | photos `PX=1003520`; MRQA `PX=1003520 CML=4096` | 0:27 / 0:27 |
| +1 | R MRQA ref_s42, ref_s43; F documents f_s42, f_s43 (F files are from 11-11 / 11-12, before eval_mm_4b.sh changed on 11-17) | MRQA as above; documents `PX=1003520` | 0:27 / 0:40 |
| +2 | Q HotpotQA s42, s43; R HotpotQA ref_s42, ref_s43 (`text_hotpotqa_1500`, reported only) | `TOPK=10 PX=1003520 CML=4096` | 0:29 |
| gate wake | pair every set Q − R per seed (`scripts/paired_judge.py`), no-op control R docs vs itself = 0.000, Q − F documents next to the gate; Validated / Failed / Tie per CAMPAIGN.md "Decision gates" | — | — |

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| Q s42, s43 | trained, health PASS, documents evals running | the table above; gate when documents, photos, MRQA are judged at both seeds |
| R reference | documents + photos re-run (4 cells, healthy); MRQA + HotpotQA × 2 seeds to re-run | wakes +1 / +2 |
| F re-runs (documents s42 / s43) | not launched | wake +1; needed for the Q − F control |
| Q3 seed 44 | only on a tie | init_s44 not built (builder takes `44`) |
| H lr 1e-4 | only after the owner checkpoint | no ARM case yet |

## Deliverable table (filled as cells land)
Judge accuracy ↑ (Qwen3-8B rubric), seed 42 / seed 43; compression 8 tokens per slot, ~123 slots per 1M-px page; n = 1,500 per set.
| model | what trains | documents ↑ | photos ↑ | MRQA text ↑ | HotpotQA ↑ (not gating) |
|---|---|---|---|---|---|
| R reference | compressor LoRA + connector + decoder LoRA | .610 / .615 (re-run, today's code) | .679 / .685 (re-run) | recorded .718 / .724 (old code; re-run pending) | recorded .603 / .593 (old code) |
| F frozen (never-QA compressor) | decoder LoRA | recorded R −.028 / −.035 (old code) | — | — | recorded −3.0 / −2.8 pt |
| **Q (QA-trained compressor frozen)** | decoder LoRA | running | — | — | — |
| uncompressed decoder | — | recorded .809 | recorded .717 | recorded .807 | — |

## Health evidence this wake (numbers, not adjectives)
- Code md5 now = last wake (`campaign_qafreeze/code_md5.txt`): eval_mm_4b.sh b3d1cda1, eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py 02bc8748, paired_judge.py a3cc13b2, pisco/model.py 994cff3d, ft_qafreeze.sbatch fd29d048, check_module_move.sh b6c5f6f9.
- GPU-h spent ≈ 20.3 of 55 (smoke 0.5 + two trainings 18.3 + four R evals 1.5). The 12 remaining evals + 2 Q documents ≈ 5 → ~25.5 expected.
- Disk 7.7 T free.

## Efficiency line from the brief ("nothing running for 9.8 h")
Not a loop choice: the wait named last wake (R evals) was met at 00:27 and the trainings ended 08:24, but the weekly pace hold (sensor `paced=1`) kept the wake until the owner set `CAMP_PACE=0` and touched GATE at 16:25. Griped as a misattribution. What the loop changed: each chain now puts the weight check and its eval in one submission pair (`--dependency afterok`), and the remaining cells are planned at the cap of 4 per wake (griped as +1 of 65252f7b).

## Blockers / needs the owner
(none)
