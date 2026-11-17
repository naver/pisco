# STATE — campaign qafreeze   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-17 22:30-22:50. Question: can a compressor that HAS had one QA finetune (R = 184778 / 184780)
stay frozen while a fresh decoder LoRA trains on the same 28k rows? (arm Q)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- The hybrid start model can be built from saved checkpoints, no new code: init_s42 / init_s43 = R's compressor + connector + config byte-identical (md5), decoder = the stage-2 decoder byte-identical (md5 dca63864); same 696 decoder tensor names, shapes, dtypes as R — job 191450, closed.
- Feasibility gate PASS (all three checks, every tensor compared): R s42 → init_s42 compressor and connector rows changed 0.0%, decoder lora_B 68.6% (job 191451); stage-2 → init_s42 decoder rows 0.0%, compressor lora_B 73.8% (the known-positive control sees movement; job 191452); seed 43 same picture, decoder lora_B 67.5% (job 191453).
- Plumbing check PASS on the 20-step smoke, seed 42 (1,280 rows = .046 epoch; n = 1 smoke): log `froze 506 compressor + connector tensors`, `Compressor trainable parameters: 0`, trainable 155,978,240 = decoder only; loss step 10 .555 / step 20 .515 vs the earlier frozen smoke 188528 .594 / .584 (inside the ±.10 band); 80.97 s/step overall (70-73 steady); peak GPU memory 49.7 GB — job 191493, closed. Module movement init_s42 → smoke: every compressor / connector row 0.0%, decoder lora_B 76.0% — job 191494, closed.

**Open, and running**
- Arm Q full training, seed 42 (1 epoch = 28k rows) — job 191527 PENDING on `gpu` (Resources; scheduler estimate 23:48 tonight).
- Arm Q full training, seed 43 — job 191530 PENDING on `gpu` (Priority; estimate 11-18 12:00, worst case).
- Reference R re-evaluated with today's eval launcher: documents s42 (job 191495) and s43 (job 191496) RUNNING on `gpu-be`; photos s42 (job 191531) and s43 (job 191532) PENDING on `gpu-be`.

## Plumbing check — config diff (law: paste it here before the first full cell)
`diff expQ/ft_mm_4b_184778/training_config.yaml expQ/qafreeze/ft_frozen_s42_smoke/training_config.yaml`:
```
23a24  >   max_steps: 20
36a38  >   freeze_compressor: true
38c40  < model_name_or_path: …/expQ/archq3/pt_mm_5M_spread/model
       > model_name_or_path: …/expQ/qafreeze/init_s42
```
Only the three allowed fields differ (no output path appears in the yaml). The full cells drop `max_steps`.

## Owned jobs
| job | arm | state | elapsed | note |
|---|---|---|---|---|
| 191450 | B build | COMPLETED 0:0 | 3:07 | closed; init_s42 + init_s43 |
| 191451-191453 | B checks | COMPLETED 0:0 | ≤ 1:02 | closed; gate PASS |
| 191493 | Qs smoke s42 (20 steps) | COMPLETED 0:0 | 29:19 | closed; homer-010; PASS |
| 191494 | Qs module-move check | COMPLETED 0:0 | 0:44 | closed; homer-019; PASS |
| 191495 | R re-eval documents s42 | RUNNING (homer-017) | ~0:11 at 22:34 | → expQ/qafreeze/ref_s42/eval/mm_docqa_pisco_px1003520_mm_docqa_1500_1m.json |
| 191496 | R re-eval documents s43 | RUNNING (homer-019) | ~0:07 at 22:34 | same, ref_s43 |
| 191527 | Q full s42 | PENDING (Resources) | 0 | `gpu`, --time 12:20 = 80.97 s/step × 438 × 1.25; out expQ/qafreeze/ft_frozen_s42 |
| 191530 | Q full s43 | PENDING (Priority) | 0 | same, ft_frozen_s43 |
| 191531 | R re-eval photos s42 | PENDING (Priority) | 0 | PX=1003520, --time 0:27 = recorded 188602 command; → ref_s42/eval/mm_docqa_pisco_px1003520_mm_naturalqa_1500.json |
| 191532 | R re-eval photos s43 | PENDING (Priority) | 0 | same, ref_s43 |

## Adopted jobs
(none) — 191392 / 191516 / 191517 are longdoc's (never touched).

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| R reference (184778 / 184780) | documents running, photos queued (4 cells); MRQA + HotpotQA × 2 seeds still to launch (4 cells) | next wake: launch MRQA `PX=1003520 CML=4096` (--time 0:27, recorded 188606 wall 16:35) and HotpotQA `TOPK=10 PX=1003520 CML=4096` (--time 0:29, recorded 188615 wall 21:41) for ref_s42 / ref_s43, argv `expQ/qafreeze/ref_s4x/model pisco '' eval_data/<set>`; health-check the documents / photos files |
| Q s42, s43 | training queued | mid-run: `expmon probe` eval_loss at .23 / .46 / .69 / .92 epoch vs F (.455 / .444 / .438 / .434) and R (.513 / .455 / .393 / .394); at step ~100 the pre-authorised preview (documents, LIMIT=300, judged, paired vs R on the same 300 ids, `provisional:`); after the run: check_module_move init_s4x → model, then 4 judged evals per seed |
| F re-runs (documents s42 / s43) | not launched | needed for the Q − F control in the gate wake; symlinks expQ/qafreeze/f_s42/model, f_s43/model; PX=1003520, --time 0:40 |
| Q3 seed 44 | only on a tie | init_s44 not built (builder takes `44`) |
| H lr 1e-4 | only after the owner checkpoint | no ARM case yet |

## Deliverable table (skeleton; filled as cells land)
Judge accuracy ↑ (Qwen3-8B rubric), seed 42 / seed 43; compression 8 tokens per slot, 123 slots per 1M-px page; n = 1,500 per set.
Rows R / F / A2 / uncompressed are the numbers recorded by campaign freeze on the OLD eval launcher (md5 50abc167); they are not re-derived here and are replaced by the re-run cells as they land.

| model | what trains | documents ↑ | photos ↑ | MRQA text ↑ | HotpotQA ↑ (not gating) |
|---|---|---|---|---|---|
| R reference | compressor LoRA + connector + decoder LoRA | recorded .610 / .615 (re-run 191495 / 191496) | recorded .673 / .685 (re-run 191531 / 191532) | recorded .718 / .724 | recorded .603 / .593 |
| F frozen (never-QA compressor) | decoder LoRA | recorded R −.028 / −.035 | — | — | recorded −3.0 / −2.8 pt |
| A2 connector frozen | compressor LoRA + decoder LoRA | recorded R +.023 / −.006 | — | — | — |
| **Q (QA-trained compressor frozen)** | decoder LoRA | training | — | — | — |
| uncompressed decoder | — | recorded .809 | recorded .717 | recorded .807 | — |

Training cost so far: Q smoke 80.97 s/step overall, 70-73 steady, 49.7 GB peak (F smoke 188528: 85.1 / 72-76 / 49.7 GB; R unfrozen smoke 100.7 / 88-90 / 31.9 GB).

## Health evidence this wake (numbers, not adjectives)
- Code md5 (full hashes in `campaign_qafreeze/code_md5.txt`, written by `md5sum` this wake): `eval_mm_4b.sh` b3d1cda1 — DIFFERS from the recorded 50abc167, hence the reference re-runs; `scripts/eval_mm_docqa.py` dfe32aa0, `scripts/judge_mm_docqa.py` 02bc8748, `scripts/paired_judge.py` a3cc13b2 — these three EQUAL the recorded values; `pisco/model.py` 994cff3d (recorded in every intent note). ft_qafreeze.sbatch fd29d048, build_init.sbatch 6fa65d2c, check_module_move.sh b6c5f6f9.
- Queue at 22:33: `gpucap land a100 1` → yes, 8 free nodes on `gpu`, owner cap 8, this account using 0 on `gpu` before these two launches (2 now). `sinfo -p gpu`: 8 nodes 28/28 CPUs allocated, homer-009 / 019 26/28 — the 8-CPU training may pend Resources (known_issues row 26).
- GPU-h spent: smoke 0.49 (29:19 on 1 GPU); evals so far run on gpu-be. Budget 55.
- Disk 8.0 T free.

## Blockers / needs the owner
(none)
