# STATE — campaign qafreeze   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-17 18:30-18:45 (first wake). Question: can a compressor that HAS had one QA finetune (R = 184778 / 184780)
stay frozen while a fresh decoder LoRA trains on the same 28k rows? (arm Q)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- The hybrid start model can be built from saved checkpoints, no new code: init_s42 / init_s43 = R's compressor + connector + config byte-identical (md5), decoder = the stage-2 decoder byte-identical (md5 dca63864); same 696 decoder tensor names, shapes, dtypes as R — job 191450, closed.
- Feasibility gate PASS (all three checks, every tensor compared): R s42 → init_s42 compressor and connector rows changed 0.0%, decoder lora_B 68.6% (job 191451); stage-2 → init_s42 decoder rows 0.0%, compressor lora_B 73.8% (the known-positive control sees movement; job 191452); seed 43 same picture, decoder lora_B 67.5% (job 191453).

**Open, and running**
- (none running) Next: the 20-step plumbing smoke of arm Q seed 42 + its module-move check, and the reference re-evaluations (the eval launcher changed since R was scored, see Health).

## Owned jobs (jobid | arm | state | elapsed | epoch or step/total | eta | note)
| job | arm | state | elapsed | note |
|---|---|---|---|---|
| 191450 | B build | COMPLETED 0:0 | 3:07 | homer-012; init_s42 + init_s43 under expQ/qafreeze/ |
| 191451 | B check (1) s42 | COMPLETED 0:0 | 1:02 | gate (1) PASS |
| 191452 | B check (2) s42 | COMPLETED 0:0 | 0:48 | gate (2) PASS |
| 191453 | B check (1) s43 | COMPLETED 0:0 | 0:57 | s43 init OK |

## Adopted jobs
(none) — 191392 / 191393 in the queue are longdoc's (never touched).

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| R reference (184778 / 184780) | done; its judged files ran on eval_mm_4b.sh md5 50abc167 | re-run its 4 sets × 2 seeds on the current md5 (b3d1cda1) before any pair: stage eval-ref, models via symlinks expQ/qafreeze/ref_s42/model → 184778, ref_s43/model → 184780 (outputs land in expQ/qafreeze/ref_s4x/eval) |
| B init build + check | **closed, PASS** | — |
| Qs smoke (s42, 20 steps) | to launch next wake | PASS = log `freeze_compressor=True: froze N …` + `Compressor trainable parameters: 0`; module move init_s42 → smoke: compressor/connector 0.0%, decoder lora_B > 0%; config diff vs 184778 clean; loss steps 10/20 within .10 of 188528 (.594 / .584) |
| Q s42, s43 | after the smoke | full cells, `--time` = smoke s/step × 438 × 1.25; evals chained afterok |
| Q3 seed 44 | only on a tie | init_s44 not built (builder takes `44` → expQ/freeze/ft_unfrozen_s44/model) |
| H lr 1e-4 | only after the owner checkpoint | no ARM case yet (law: one new case when released) |
| F re-runs (documents s42 / s43) | needed for the Q − F control | F's files ran on md5 50abc167; symlinks expQ/qafreeze/f_s42/model, f_s43/model ready |

## Health evidence this wake (numbers, not adjectives)
- Code md5 now: eval_mm_4b.sh **b3d1cda1** (law/recorded 50abc167 → mismatch, as the law expected after commit 5955890; 50abc167 was an uncommitted copy — the commit's parent is 59bbae57 — so the exact change cannot be read from git, hence the re-runs); scripts/eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py 02bc8748, paired_judge.py a3cc13b2, pisco/model.py 994cff3d (all = recorded; provisional: the verifier could not re-check the three non-launcher md5s read-only, they match campaign_freeze/STATE.md:29 — re-check before the first pairing). Light bookkeeping wake 19:30: nothing changed, launches pending for the pinned model (NEXT = wake-in 1).
- Launcher ft_qafreeze.sbatch: `bash -n` OK; its training command is byte-identical to expQ/freeze/ft_freeze.sh's (diff of the `pixi run python … EXIT CODE` block empty); only `model_name_or_path` changes (expQ/qafreeze/init_s<seed>), output dir expQ/qafreeze/ft_frozen_s<seed>[_smoke].
- The two model configs (R vs stage-2) differ in connector_out_norm (true/.461 vs false/.4) and image_processor_source only; R's config travels with R's connector (md5 equal), and finetune_mm.yaml sets the same connector_out_norm / .461 (184778 training_config.yaml), so the override is a no-op on the hybrid.
- Measured costs for the next launches (sacct): frozen smoke 188528 31:05 wall (85.1 s/step overall), MaxRSS 21.8 GB → smoke `--time 0:40:00`; reference evals wall documents 29:04 / photos 18:47 / MRQA 16:35 / HotpotQA 21:41 → repeat the recorded `--time` 0:40 / 0:27 / 0:27 / 0:29.
- Reference eval env (from the recorded file names + FACTORS; runs.jsonl kept argv only): documents and photos `PX=1003520`; MRQA `PX=1003520 CML=4096`; HotpotQA `TOPK=10 PX=1003520 CML=4096`; argv `<model> pisco '' eval_data/<set>`; `-p gpu-be --exclude=homer-[005-009]`.
- Disk 8.1 T free; 24 free GPUs (gpu + gpu-be) at 18:30. GPU-h spent: 0 (CPU jobs only).

## Blockers / needs the owner
(none)
