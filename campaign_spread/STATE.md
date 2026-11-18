# STATE — campaign spread   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-18 17:30-17:55 (fourth wake; woken because the four cells it waited on ended). Deliverable:
`docs/spread_cases.md`. Audit tables: `expQ/spread/audit_191454/audit.md`. Seed-pooled test:
`expQ/spread/pooled_20261118_1730/pooled.md` (spec `scripts/spread/pooled_pairs.tsv`, today's-code pairs only).
Re-run identity: `expQ/spread/rerun_checks/rerun_20261118.md` (and the older `photo_drift.md`).

## TL;DR — where this leaves the question
**Settled**
- Arm A feasibility gate PASS (job 191454, CPU, closed): the 8 recorded spread − tail judge differences re-derive exactly; 39 self-pairs 0.000; 184778's document file 1,500 rows, judge on all, 0 unparseable, mems_used 646,042.
- Seed-pooled "equal" test (owner, option A) controls PASS again at 17:45: reduced-budget documents +.084 [+.065, +.103] (expected +.084), every self-pair 0.000.
- **Arm B closed.** ONE chunk (ten HotpotQA passages as one ~1,380-token chunk, 173 slots per question, uncompressed .752): spread better at both seeds, s42 .611 vs .351 = +.260 [+.233, +.289], McNemar 488 / 98 (191457 − 191456); s43 .616 vs .369 = +.247 [+.219, +.275], 477 / 106 (191528 − 191526); n 1,500 each; EM +.151 / +.152. The project-prompt check runs in 191665 (arm J gate still open for this case).
- Tail production loses going from ten chunks to one at both seeds: −.253 [−.281, −.225] (s42, 191456 − 191455), −.229 [−.255, −.200] (s43, 191526 − 191519). Spread loses nothing beyond 2 pt at either seed: +.006 [−.019, +.029] (s42, 191457 − 191518), +.023 [+.000, +.045], 178 / 144, p .066 (s43, 191528 − 191630).
- TEN chunks, both seeds on today's code: s42 .605 vs .603 = +.001 [−.021, +.025] (191518 − 191455); s43 .593 vs .597 = −.004 [−.027, +.019], 154 / 160 (191630 − 191519); seed-pooled −.001 [−.0203, +.018], width .038 → no difference shown, "equal" NOT certified (the CI misses the ±2 pt band by .03 pt). EM −.027 / −.020 (tail ahead) → judge-sensitive candidate until 191665.
- Full-budget PHOTOS on today's code, **equal** by the seed-pooled test: s42 .679 vs .687 = −.007 [−.027, +.011], 105 / 116 (191531 − 191631); s43 .685 vs .675 = +.010 [−.011, +.030], 125 / 110 (191532 − 191632); pooled +.001 [−.014, +.018], width .031; n 1,500; 53 slots per question; uncompressed .717.
- Full-budget DOCUMENTS (4 pages, 431 slots per question), same code at both seeds: s42 .610 vs .599 = +.011 [−.014, +.035], 177 / 161 (184778 vs 174908 recorded files, both 24b7a2c; 184778 re-run 191495 identical); s43 .615 vs .591 = +.024 [−.002, +.048], 193 / 157, p .061 (191496 − 191633); pooled +.017 [−.002, +.036], width .038 → **not resolvable here** (neither equal nor better). Uncompressed .809.
- Re-runs on today's code (rerun_20261118.md): documents 184778, 184780, 176242 0 / 1,500 answers differ; photos 184780 and 176242 0 / 1,500; photos 184778 223 / 1,500 (68 judge flips, 32 / 36, .677 vs .679); photos 174908 135 / 1,500 (34 flips, 16 / 18, .685 vs .687); 184780 ten chunks 0 / 1,500 (191630 = freeze file).
- `PX` is inert on photos: every `mm_naturalqa_1500` file has 79,322 slots at 451,584 and 1,003,520 px.

**Open, and running**
- provisional: spread better on infographics at full budget, both seeds on the same code now: +.050 [+.006, +.094] 80 / 55 (s42) / +.050 [+.008, +.094] 76 / 51 (s43, 191496 − 191633), n 500 each. A post-hoc pick among 3 sources, and both seeds use the same 500 questions → one repeat on new infographic questions before it counts.
- provisional: spread better on one long MRQA passage at full budget, s42 only: .718 vs .631 = +.087 [+.063, +.112] (188606 vs 185780, two code versions → reported only). Needs 4 cells on today's code (both models, both seeds).
- provisional: the gain did not vanish because spread got worse: spread full − reduced documents +.011 [−.011, +.033] / −.003 [−.026, +.020]; tail production − reduced +.093 [+.068, +.117] / +.048 [+.025, +.072] (job 191454). NOT one variable; the tail s42 reduced file ran on homer-009 and the spread s43 reduced file 183816 on homer-008 (image outlier node); the s43 pairs cross code eras.
- provisional: the tail models' budget gain sits mostly on MP-DocVQA +.162 / +.090, slides +.076 / +.040, infographics +.040 / +.014 (n 500 per source, job 191454); no between-source test; same node and code-era caveats.
- Arm J: project-prompt re-judge of COPIES of all 16 today's-code files (one chunk, ten chunks, documents, photos × 2 seeds) = job 191665 RUNNING.
- Arm C (many pages): tail at 10 pages s42 191666 / s43 191667, tail at 20 pages s42 191668, all RUNNING. Spread side read-only from longdoc: 10 pages .607 (191378) / .608 (191390), 20 pages s42 .599 (191377); spread 20 pages s43 is missing → hand-off 19ac33 to longdoc.

## Owned jobs (jobid | arm | state | rows | eta | note)
| 191454 | A audit | COMPLETED | 23 pairs | — | closed: PASS |
| 191455 / 191456 / 191457 | B s42 tail ten / tail one / spread one | COMPLETED | 1,500 each | — | closed healthy: .603 / .351 / .611 |
| 191518 / 191519 / 191526 / 191528 | B spread ten s42 / tail ten s43 / tail one s43 / spread one s43 | COMPLETED | 1,500 each | — | closed healthy: .605 / .597 / .369 / .616 |
| 191630 | B spread 184780 ten chunks s43 | COMPLETED 0:0 homer-017 22:19 | 1,500 | — | closed healthy, .593 |
| 191631 / 191632 | G tail photos s42 / s43 | COMPLETED 0:0 homer-010 17:39 / 18:56 | 1,500 each | — | closed healthy, .687 / .675 |
| 191633 | G tail documents s43 | COMPLETED 0:0 homer-010 28:20 | 1,500 | — | closed healthy, .591 |
| 191665 | J re-judge, project prompt, 16 copies → `expQ/spread/rejudge/gating16/` | RUNNING | 16 × 1,500 | ~8 min | --time 0:25, --mem 4G |
| 191666 | C tail 174908 10 pages (XD=6 TOPK=10 DML=1578) | RUNNING | 1,500 | ~55 min | → `expQ/spread/b_tail_s42/eval/mm_docqa_pisco_k10_px1003520_xd6_mm_docqa_1500_1m.json`; --time 2:46, 18G |
| 191667 | C tail 176242 10 pages | RUNNING | 1,500 | ~55 min | → `expQ/spread/b_tail_s43/eval/…xd6…`; --time 2:46, 18G |
| 191668 | C tail 174908 20 pages (XD=16 TOPK=20 DML=2823) | RUNNING | 1,500 | ~1.5 h | → `expQ/spread/b_tail_s42/eval/mm_docqa_pisco_k20_px1003520_xd16_mm_docqa_1500_1m.json`; --time 4:27, 18G |
All four new jobs: gpu-be `--exclude=homer-[005-009]`, --parent 191454. `--time` = slowest measured wall of the kind × 3
(judge-only 191629 5:46 for 12 files → 16 files; 10 pages 174872 55:11; 20 pages 191377 1:28:50); `--mem` = largest
measured MaxRSS × 1.5 (10 pages 174872 11.9 GB; judge-only 1.39 GB).

## Adopted jobs
(none) — the queue held only the daemon 188764 and another project's job.

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| A. case audit | **decided: PASS** (191454) | re-run `--pooled` on every new pair set |
| B. long chunks | **closed**: one chunk better at both seeds; ten chunks no difference shown, equal not certified | only arm J's re-read can change the ten-chunk wording |
| G. full-budget gating pairs | **decided**: photos equal; documents not resolvable (+.017 [−.002, +.036]) | goes into the checkpoint question (option B, a second document set) |
| J. instrument | 191665 RUNNING | pair the 16 copies with `scripts/paired_judge.py` exactly as the rubric pairs; gate: every deciding case keeps its sign and stays within 2 pt under rubric / project / EM; ten chunks already differ by 2.8 pt rubric vs EM at s42 |
| C. many pages | tail 10 p ×2 and 20 p s42 RUNNING; spread 20 p s43 via hand-off 19ac33 | pair each with longdoc's spread file (mems_used must match the twin); then tail 20 p s43 (`--time 4:27`, 18G) once longdoc answers; gate: (spread − tail at 20 p) − (at 4 p) ≥ +3 pt, CI > 0, both seeds |
| MRQA one long passage | missing on today's code | 4 cells: 184778 / 184780 / 174908 / 176242 on `text_mrqa_ood_1500`, PX=1003520 CML=4096 (cell names under `expQ/spread/b_*`) |
| D. what slots hold | not started | write `scripts/spread/mem_recall.sbatch` (runs scripts/longdoc/mem_recall.py read-only) |
| E / F | owner checkpoint after C and D | question includes option B (second document set) and the infographics repeat |

Budget used: 16 cells of ~50 (1 CPU + 15 GPU), GPU-h ≈ 3.7 done + ≈ 3.4 running ≈ 7.1 of 30. LLM wakes 4 of ≤ 40.

## Health evidence this wake
- 191630-191633: COMPLETED 0:0, one attempt each, homer-010 / homer-017 (el9_8); 1,500 rows, judge on every row, judge_unparseable 0, no error in the log tail, mems_used equal to the twin (265,880 / 79,322 / 646,042). Probes recorded (`expmon probe --from-json`, one `--map`).
- md5 at launch of 191665-191668 = every earlier cell: eval_mm_4b.sh b3d1cda1, eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py 02bc8748, paired_judge.py a3cc13b2, pisco/model.py 994cff3d, judge_prompts.py bae44fa7. New launcher scripts/spread/rejudge.sbatch 15a378a7 (`bash -n` OK; copy of scripts/longdoc/rejudge.sbatch 9e2bc7d6 with the output dir changed).
- Arm C pairs: longdoc's spread page cells ran eval_mm_4b.sh 50abc167 (requeue markers only differ) with the same eval / judge / model md5s; the distractor pages are a deterministic function of the row (`with_distractors`, shuffle seeded by query_id).
- Verifier items fixed: "spread does not lose" now has both seeds with CIs; the 68-flip count lives in rerun_20261118.md; full-budget pairs are now same-code at both seeds; the homer-008 spread s43 reduced file is disclosed.

## Proposed replacement text (the owner's chats write these files; provisional until arms J and C)
- `docs/architecture_review.md:252-256`, replace "the +.08 of the reduced chains is what the extra stage-2 data was buying" with: "At full budget the spread chain scores what it scored at reduced budget (+.011 / −.003 on documents); the tail models gained +.093 / +.048 from the larger budget, most on MP-DocVQA. On today's code, full budget: photos equal (pooled +.001 [−.014, +.018]); 4-page documents +.017 [−.002, +.036], not resolvable with 1,500 questions × 2 seeds. Where the input leaves the finetune's regime the layout decides: ten HotpotQA passages compressed as ONE chunk give tail .351 / .369 and spread .611 / .616 (+26 / +25 pt, both seeds). Whether the layout changes the document plateau is untested (campaign spread, arm E)."
- HANDOFF.md thread 1: "spread vs tail: the reduced-budget gain was the tail models catching up with budget, not spread losing it; spread wins by +25-26 pt when long text is compressed as one chunk (both seeds); on ten short chunks and on photos they tie; on 4-page documents +1.7 pt, too small for the eval to resolve."

## Blockers / needs the owner
- None now. The owner checkpoint (before arm E) comes after arms C, J and D.
