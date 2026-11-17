# STATE — campaign qafreeze   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-17 20:00-20:10. Question: can a compressor that HAS had one QA finetune (R = 184778 / 184780)
stay frozen while a fresh decoder LoRA trains on the same 28k rows? (arm Q)

## TL;DR — where this leaves the question (readable as an email)
**Settled**
- The hybrid start model can be built from saved checkpoints, no new code: init_s42 / init_s43 = R's compressor + connector + config byte-identical (md5), decoder = the stage-2 decoder byte-identical (md5 dca63864); same 696 decoder tensor names, shapes, dtypes as R — job 191450, closed.
- Feasibility gate PASS (all three checks, every tensor compared): R s42 → init_s42 compressor and connector rows changed 0.0%, decoder lora_B 68.6% (job 191451); stage-2 → init_s42 decoder rows 0.0%, compressor lora_B 73.8% (the known-positive control sees movement; job 191452); seed 43 same picture, decoder lora_B 67.5% (job 191453).

**Open, and running**
- Arm Q plumbing smoke, seed 42, 20 steps (1,280 rows = .046 epoch) — job 191493 PENDING on `gpu` (Resources; scheduler estimate 23:48 tonight, worst case). Its module-movement check job 191494 waits on it (afterok).
- Reference R re-evaluated on documents with today's eval launcher, seed 42 (job 191495) and seed 43 (job 191496) — PENDING on `gpu-be` (Priority; scheduler estimate 11-22, worst case; known_issues row "gpu-be evals PENDING Priority … ~3 days out": wait, train meanwhile).

## Owned jobs
| job | arm | state | elapsed | note |
|---|---|---|---|---|
| 191450 | B build | COMPLETED 0:0 | 3:07 | homer-012; init_s42 + init_s43 under expQ/qafreeze/ |
| 191451 | B check (1) s42 | COMPLETED 0:0 | 1:02 | gate (1) PASS |
| 191452 | B check (2) s42 | COMPLETED 0:0 | 0:48 | gate (2) PASS |
| 191453 | B check (1) s43 | COMPLETED 0:0 | 0:57 | s43 init OK |
| 191493 | Qs smoke s42 (20 steps) | PENDING (Resources) | 0 | `gpu`, --time 0:40 (188528 wall 31:05 × 1.25); out expQ/qafreeze/ft_frozen_s42_smoke |
| 191494 | Qs module-move check | PENDING (afterok 191493) | 0 | init_s42 → smoke model; gpu-be CPU, --time 0:20 |
| 191495 | R re-eval documents s42 | PENDING (Priority) | 0 | PX=1003520 → expQ/qafreeze/ref_s42/eval/mm_docqa_pisco_px1003520_mm_docqa_1500_1m.json |
| 191496 | R re-eval documents s43 | PENDING (Priority) | 0 | same, ref_s43 |

## Adopted jobs
(none) — 191392 / 191393 are longdoc's, 191455-191457 are spread's (never touched).

## Open arms and the next decision for each
| arm | status | next decision |
|---|---|---|
| R reference (184778 / 184780) | documents re-runs queued (191495 / 191496); photos, MRQA, HotpotQA × 2 seeds still to launch (6 cells) | launch the rest as submission slots allow; env from the recorded file names: photos `PX=1003520`; MRQA `PX=1003520 CML=4096`; HotpotQA `TOPK=10 PX=1003520 CML=4096`; argv `expQ/qafreeze/ref_s4x/model pisco '' eval_data/<set>`; --time 0:27 / 0:27 / 0:29 |
| B init build + check | **closed, PASS** | — |
| Qs smoke (s42, 20 steps) | queued (191493 → check 191494) | PASS = log `freeze_compressor=True: froze N …` + `Compressor trainable parameters: 0`; check: compressor/connector 0.0%, decoder lora_B > 0%; `training_config.yaml` diff vs expQ/ft_mm_4b_184778 = model_name_or_path + freeze_compressor + paths + max_steps only (paste it here); loss steps 10 / 20 within .10 of 188528 (.594 / .584); record s/step and peak GPU memory |
| Q s42, s43 | after the smoke passes | both full cells in one wake (CAMP_MAX_TRAIN 2), `gpucap land a100 1` before each; `--time` = smoke steady s/step × 438 × 1.25; evals of the 4 sets chained afterok on the training are allowed once its model dir path is known |
| Q3 seed 44 | only on a tie | init_s44 not built (builder takes `44` → expQ/freeze/ft_unfrozen_s44/model) |
| H lr 1e-4 | only after the owner checkpoint | no ARM case yet (law: one new case when released) |
| F re-runs (documents s42 / s43) | needed for the Q − F control | F's files ran on md5 50abc167; symlinks expQ/qafreeze/f_s42/model, f_s43/model ready; 2 cells, PX=1003520 |

## Health evidence this wake (numbers, not adjectives)
- Code md5 re-checked now (md5sum on the files): eval_mm_4b.sh b3d1cda1 (≠ recorded 50abc167 → reference re-runs, as the law requires), scripts/eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py 02bc8748, paired_judge.py a3cc13b2, pisco/model.py 994cff3d — the four = recorded (this closes the verifier's "provisional" on them). ft_qafreeze.sbatch fd29d048, build_init.sbatch 6fa65d2c, check_module_move.sh b6c5f6f9.
- Reference eval env confirmed from the recorded output names in expQ/freeze/ref_s42/eval: `mm_docqa_pisco_px1003520_mm_docqa_1500_1m`, `…_px1003520_mm_naturalqa_1500`, `…_px1003520_cml4096_text_mrqa_ood_1500`, `…_k10_px1003520_cml4096_text_hotpotqa_1500`. runs.jsonl now records the env (191495: `{"PX": "1003520"}`).
- Budget vs wall-time ceiling (the verifier's point): the per-cell budget counts hours USED — F's three full frozen runs used 8.7 / 9.3 / 8.7 h at 71.5-76.1 s/step steady. The `--time` rule (s/step × 438 × 1.25) is a ceiling: at the 188528 overall 85.1 s/step it gives 12.9 h, at a steady 76 s/step 11.6 h. Read the new smoke's steady s/step; if a full Q run USES more than 10 h, record the overrun against the 11 h slack.
- Queue at 20:02: `gpucap land a100 1` → yes, 8 free nodes on `gpu`, owner cap 8 GPUs, this account using 0 on `gpu`; but every `gpu` node shows 26-28 of 28 CPUs allocated (sinfo), so the 8-CPU smoke pends Resources (known_issues row 26). GPU-h spent: 0 (CPU jobs only).
- Disk 8.1 T free.

## Blockers / needs the owner
(none)
