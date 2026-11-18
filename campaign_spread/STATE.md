# STATE — campaign spread   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-18 16:30-16:55 (third wake; woken by the owner's REQUEST + arm B cells ended; the 15.8 h gap before it
was the weekly-pace hold, now off). Deliverable: `docs/spread_cases.md`. Audit tables: `expQ/spread/audit_191454/audit.md`.
Seed-pooled test table: `expQ/spread/pooled_20261118/pooled.md`. Re-run checks: `expQ/spread/rerun_checks/photo_drift.md`.

## TL;DR — where this leaves the question
**Settled**
- Arm A feasibility gate PASS (job 191454, CPU, closed): the 8 recorded spread − tail judge differences re-derive exactly (documents +.093 / +.075 reduced budget, photos +.023 / +.021, full budget documents +.011 / +.024, photos −.009 / +.010); 39 self-pairs 0.000; 184778's document file 1,500 rows, judge on all, 0 unparseable, mems_used 646,042.
- Owner decision executed (REQUEST 2026-11-18, option A): "equal" = the 95% CI of the per-question spread − tail difference averaged over seeds 42 and 43 inside ±2 pt AND each seed inside ±2 pt. Code: `scripts/spread/slice_pairs.py --pooled <tsv>` (+40 lines, stdlib), spec `scripts/spread/pooled_pairs.tsv`. Controls PASS: reduced-budget documents pooled +.084 [+.065, +.103] (expect +.084; seeds +.093 / +.075), every self-pair 0.000 (n 1,500 each). Law updated (CAMPAIGN.md gates). Option B (a second document set) waits for the checkpoint before arm E.
- Arm B, ONE chunk (ten HotpotQA passages compressed as one ~1,380-token chunk, ~173 slots per question at 8 tokens per slot, uncompressed .752): spread **better** at both seeds — s42 .611 vs .351 = +.260 [+.233, +.289], McNemar 488 / 98 (191457 − 191456); s43 .616 vs .369 = +.247 [+.219, +.275], 477 / 106 (191528 − 191526); p < .001 each; n 1,500 each; pooled +.254 [+.227, +.279]. EM same sign (s42 +.151, s43 +.152). All four cells healthy (1,500 rows, judge on all, 0 unparseable, mems_used 259,986 on both sides, el9_8 nodes, today's code). The project-prompt judge check (arm J) is still to run.
- Tail production loses on one chunk vs ten at both seeds: −.253 [−.281, −.225] (s42, 191456 − 191455), −.229 [−.255, −.200] (s43, 191526 − 191519); spread does not: +.006 [−.019, +.029] (s42, 191457 − 191518).
- Arm B, TEN chunks, seed 42 on today's code: spread .605 (191518) vs tail .603 (191455) = +.001 [−.021, +.025], 156 / 154, p .96 (n 1,500, both healthy, mems_used 265,880 both).
- Text answers moved across code dates once: 184778 ten chunks, freeze file 188615 (homer-016, 11-11) vs 191518 (homer-011, 11-17): 58 / 1,500 answers differ, 19 judge flips (8 / 11), .603 vs .605. Law and known_issues updated (text pairs too need today's code).
- Today's-code re-runs of the spread reference (qafreeze jobs 191495 / 191496 / 191531 / 191532, read only) vs the recorded files: documents 184778 and 184780 0 / 1,500 answers differ; photos 184780 0 / 1,500; photos 184778 223 / 1,500 differ, 68 judge flips, .677 vs .679.
- `PX` is inert on photos: every `mm_naturalqa_1500` file (spread at 1,003,520 px, tail at 451,584 px) has 79,322 slots, so that recorded pixel difference is not a confound.
- Re-run identity (photo_drift.md): 184778 photos 184783 vs 188602 differ on 233 / 1,500 answers, 85 judge flips, .677 vs .673.

**Open, and running**
- Arm B ten chunks s43: spread 184780 on today's code = job 191630 (RUNNING) → pairs with tail 191519. Then the seed-pooled ten-chunk test decides "equal". Reported meanwhile: with the older-code s43 spread file (expQ/freeze/ref_s43) the pooled difference is −.001 [−.020, +.018] — at the band edge, not decided.
- Full-budget gating pairs on today's code: tail 174908 photos (191631), tail 176242 photos (191632), tail 176242 documents (191633), all RUNNING, PX=1003520. They pair with the qafreeze re-runs of the spread models. Reported meanwhile (recorded files, mixed code dates): documents pooled +.017 [−.002, +.036] (width .038 → neither equal nor better), photos pooled +.001 [−.015, +.017] → would be "equal", provisional until the today's-code pairs land.
- provisional: spread better on infographics at full budget, +.050 [+.004, +.096] / +.050 [+.008, +.096] (n 500, 135 / 127 discordant, job 191454) — post-hoc pick among 3 sources; the s43 tail file is older code → 191633 re-decides it.
- provisional: spread better on one long MRQA passage at full budget, s42 only: .718 vs .631 = +.087 [+.063, +.112] (188606 vs 185780, two code versions → reported only); s43 needs a 176242 cell and both sides on today's code.
- provisional: the gain did not vanish because spread got worse: spread full − reduced documents +.011 [−.011, +.033] / −.003 [−.026, +.020]; tail production − reduced +.093 [+.068, +.117] / +.048 [+.025, +.072] (job 191454; NOT one variable; tail s42 reduced side ran on homer-009, s43 pairs cross code eras).
- provisional: the tail models' budget gain sits mostly on MP-DocVQA +.162 / +.090, slides +.076 / +.040, infographics +.040 / +.014 (n 500 per source, job 191454); no between-source test.
- Resolving power even pooled: on documents and text (width .038) "equal" is reachable only when the difference is under ~.1 pt; photos (width .032) under ~.4 pt. This is the case for option B at the checkpoint.

## Owned jobs (jobid | arm | state | rows | eta | note)
| 191454 | A audit | COMPLETED | 23 pairs | — | closed: PASS |
| 191455 / 191456 / 191457 | B s42 tail ten / tail one / spread one | COMPLETED | 1,500 each | — | closed healthy: .603 / .351 / .611 |
| 191518 | B spread 184778 ten chunks s42, today's code | COMPLETED 21:01 homer-011 | 1,500 | — | closed healthy, .605 |
| 191519 | B tail 176242 ten chunks s43 | COMPLETED 21:11 homer-017 | 1,500 | — | closed healthy, .597 |
| 191526 | B tail 176242 one chunk s43 | COMPLETED 18:52 homer-018 | 1,500 | — | closed healthy, .369 |
| 191528 | B spread 184780 one chunk s43 | COMPLETED 18:00 homer-011 | 1,500 | — | closed healthy, .616 |
| 191630 | B spread 184780 ten chunks s43, today's code | RUNNING | 1,500 | ~21 min | → expQ/spread/b_spread_s43/eval/mm_docqa_pisco_k10_cml4096_text_hotpotqa_1500.json; --time 1:15 |
| 191631 | G tail 174908 photos, today's code | RUNNING | 1,500 | ~19 min | → expQ/spread/b_tail_s42/eval/mm_docqa_pisco_px1003520_mm_naturalqa_1500.json; --time 1:00 |
| 191632 | G tail 176242 photos, today's code | RUNNING | 1,500 | ~19 min | → expQ/spread/b_tail_s43/eval/mm_docqa_pisco_px1003520_mm_naturalqa_1500.json; --time 1:00 |
| 191633 | G tail 176242 documents, today's code | RUNNING | 1,500 | ~29 min | → expQ/spread/b_tail_s43/eval/mm_docqa_pisco_px1003520_mm_docqa_1500_1m.json; --time 1:30 |
All four new cells: gpu-be `--exclude=homer-[005-009]`, `--mem 8G` (MaxRSS 3.98-4.29 GB × 1.5); `--time` = slowest measured wall of the cell kind × 3 (text 24:43, photos 19:25, 4-page documents 29:04). They started at once (no queue wait). Stage G = re-runs of the full-budget gating pairs.

## Adopted jobs
(none) — the queue holds only the daemon 188764 and another project's job.

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| A. case audit | **decided: PASS** (191454); seed-pooled statistic added with controls | re-run `--pooled` on every new pair set |
| B. long chunks, full budget | one chunk **better** at both seeds; ten chunks s42 +.001 | when 191630 ends: pair 191630 − 191519 (s43), then `--pooled` with s42 191518 − 191455 → ten chunks equal / not resolvable → arm B closes (gate: one chunk better AND ten chunks equal) |
| G. full-budget gating pairs on today's code | 3 tail re-runs RUNNING | pair photos 191631 − qafreeze 191531 (s42), 191632 − 191532 (s43); documents s43 191633 − 191496; documents s42 = recorded 184778 vs 174908, both on 24b7a2c (same code) → `--pooled` on documents and photos → case verdicts; re-slice infographics / MP-DocVQA |
| J. instrument | not started | write `scripts/spread/rejudge.sbatch` (project prompt on COPIES under `expQ/spread/rejudge/`); files: arm B one-chunk pairs (4) first, then the gating pairs |
| C. many pages | not started | tail 174908 / 176242 at 10 pages (XD=6 TOPK=10) and 20 pages (XD=16 TOPK=20), PX=1003520, DML from longdoc's audit; spread side = longdoc's 191378 / 191390 / 191377 (20-page s43 missing on longdoc's side) |
| D. what slots hold | not started | write `scripts/spread/mem_recall.sbatch` (runs scripts/longdoc/mem_recall.py read-only) |
| MRQA one long passage s43 | missing | tail 176242 + spread 184780 on today's code (`PX=1003520 CML=4096`, text_mrqa_ood_1500) |
| E / F | owner checkpoint | includes option B (second document set) |

Budget used: 12 cells of ~50 (1 CPU + 11 evals), GPU-h ≈ 2.2 done + ≈ 1.5 running. LLM wakes 3 of ≤ 40.

## Health evidence this wake
- md5 at launch of 191630-191633 = at every earlier cell: eval_mm_4b.sh b3d1cda1, eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py 02bc8748, paired_judge.py a3cc13b2, pisco/model.py 994cff3d, judge_prompts.py bae44fa7. slice_pairs.py now has the `--pooled` mode (the audit path is unchanged).
- Verifier items fixed: CI ranges in the law now .044-.053 documents / .036-.041 photos; the ten-chunk s42 pair is now decided on today's code (191518), the freeze file is only a re-run check.
- Efficiency line (18 % of the last day in the queue): this wake's four cells started immediately on gpu-be; `gpucap free a100` showed 36 free.

## Proposed replacement text (the owner's chats write these files; provisional until stage G and arm J)
- `docs/architecture_review.md:252-256`, replace "the +.08 of the reduced chains is what the extra stage-2 data was buying" with: "At full budget the spread chain scores what it scored at reduced budget (+.011 / −.003 on documents); the tail models gained +.093 / +.048 from the larger budget, most on MP-DocVQA. The layout reaches the plateau with less data; whether it changes the plateau is untested (campaign spread, arm E). Where the input leaves the finetune's regime the layout still decides: ten HotpotQA passages compressed as ONE chunk give tail .351 / .369 and spread .611 / .616 (+26 / +25 pt, both seeds). A 1,500-Q eval cannot show 'equal' on documents even with two seeds pooled unless the difference is ~0."
- HANDOFF.md thread 1: "spread vs tail: the reduced-budget gain was the tail models catching up with budget, not spread losing it; spread wins by +25-26 pt when long text is compressed as one chunk (both seeds); on ten short chunks they tie (s42 +.001, s43 running)."

## Blockers / needs the owner
- None now. BLOCKED.md removed (the owner answered: option A applied; option B goes to the checkpoint before arm E).
