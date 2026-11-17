# STATE — campaign spread   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-17 22:30-22:50 (second wake; woken because arm B seed 42 ended). Deliverable: `docs/spread_cases.md`.
Audit tables: `expQ/spread/audit_191454/audit.md`. Re-run checks: `expQ/spread/rerun_checks/photo_drift.md`.

## TL;DR — where this leaves the question
**Settled**
- Arm A feasibility gate PASS (job 191454, CPU, closed): the 8 recorded spread − tail judge differences re-derive exactly from the recorded files — documents +.093 / +.075 (reduced budget, seeds 42 / 43), photos +.023 / +.021, full budget documents +.011 / +.024, photos −.009 / +.010, with the recorded CIs; 39 self-pairs give 0.000; 184778's document file has 1,500 rows, a judge on every row, 0 unparseable, mems_used 646,042. (Ran slice_pairs.py md5 229600a9; now 8acfcab7 = same tables + exit 3 on FAIL.)
- Arm B cells, seed 42, healthy (191455 / 191456 / 191457, closed; COMPLETED 0:0 on homer-017 / homer-017 / homer-019, one attempt each; 1,500 rows, judge on all, 0 unparseable, no error in the log; mems_used 265,880 ten chunks = the freeze twin, 259,986 one chunk on both sides). Judge accuracy, HotpotQA 10 passages, full budget: tail production 174908 ten chunks .603, ONE chunk .351; spread 184778 ONE chunk .611 (grid probes).
- Re-run identity (`expQ/spread/rerun_checks/photo_drift.md`, `scripts/spread/compare_answers.py` md5 2414d9c9): 184778 photos 184783 (homer-011, 11-09) vs 188602 (homer-016, 11-11) differ on 233 / 1,500 answers, 85 judge flips (45 / 40), .677 vs .673; 184780 photos 184785 (homer-008) vs its re-run: 0 / 1,500; 184778 documents recorded vs re-run: 0 / 1,500.
- provisional: spread better on ONE long text chunk at full budget, seed 42 only: spread − tail = +.260 [+.233, +.289], McNemar 488 / 98, p < .001 (n 1,500, 191457 − 191456, same code, ~173 slots per question at 8 tokens per slot; uncompressed .752). Same sign under EM +.151, match +.199, ANLS +.174. Tail one − ten chunks = −.253 [−.281, −.225] (191456 − 191455); spread one chunk − the freeze ten-chunk file = +.008 [−.016, +.031]. The reduced-budget collapse of tail on one chunk (.568 → .399) is not cured by full budget (.603 → .351). Seed 43 running.
- provisional: the gain did not vanish because spread got worse: spread full − spread reduced = documents +.011 [−.011, +.033] / −.003 [−.026, +.020], photos +.006 (s42); tail production − tail reduced = documents +.093 [+.068, +.117] / +.048 [+.025, +.072], photos +.037 [+.017, +.058] (n 1,500, job 191454). NOT one variable (stage 1 5M vs 1M, stage 2 160k vs 100k for spread; three stage-2 passes for production); the tail photo pair spans two code eras, the s43 spread pair has a homer-008 side; so "not worse", never "equal".
- provisional: the tail models' budget gain sits mostly on multi-page documents: MP-DocVQA +.162 / +.090, slides +.076 / +.040, infographics +.040 / +.014 (n 500 per source, job 191454). Not tested as a between-source difference; infographics CIs touch 0; same confound as above.
- provisional: one seed's 95% CI on 1,500 Q is .044-.050 wide on documents, .041-.056 on text, .037-.041 on photos (job 191454 audit.md, 23 pairs; arm B s42 .045 / .056). On documents and text it is wider than the ±2-pt "equal" band; photos sit at the band ("equal" only if the difference is ~0). Owner question in BLOCKED.md.

**Open, and running**
- Arm B ten chunks, seed 42: spread (freeze file 188615) − tail (191455) = −.001 [−.023, +.022], 151 / 152 — REPORTED only: 188615 ran eval_mm_4b.sh 50abc167, model.py before freeze_part, and PX=1003520 (verifier). Today's-code re-run of the spread side = 191518 (PENDING). Even then CI width .045 > .040: "not resolvable here" under the current law.
- Arm B seed 43: 191519 tail 176242 ten chunks, 191526 tail 176242 one chunk, 191528 spread 184780 one chunk — PENDING (Priority), gpu-be. Spread ten chunks s43 = re-run of 184780 on today's code, next wake (cap 4 reached).
- provisional: spread better on infographics at full budget, +.050 [+.004, +.096] / +.050 [+.008, +.096] (n 500, 135 / 127 discordant, job 191454) — post-hoc pick among 3 sources; MP-DocVQA −.036 / −.004; the s43 tail file (176244, ea0c591) is from an older code era → needs a re-run on today's code.
- provisional: spread better on one long MRQA passage at full budget, s42 only: .718 vs .631 = +.087 [+.063, +.112] (n 1,500; 188606 vs 185780, commits 0ceedcb vs e936195 → reported, not decided); s43 needs a 176242 cell and both sides on today's code.

## Owned jobs (jobid | arm | state | rows | eta | note)
| 191454 | A audit | COMPLETED 1:45 | 23 pairs, 39 files | — | closed: PASS |
| 191455 | B tail 174908 s42, ten chunks | COMPLETED 20:26 | 1,500 | — | closed healthy, .603 |
| 191456 | B tail 174908 s42, one chunk | COMPLETED 16:41 | 1,500 | — | closed healthy, .351 |
| 191457 | B spread 184778 s42, one chunk | COMPLETED 17:31 | 1,500 | — | closed healthy, .611 |
| 191518 | B spread 184778 s42, ten chunks (today's code) | PENDING | 1,500 | ~20 min once running | out expQ/spread/b_spread_s42/eval/mm_docqa_pisco_k10_cml4096_text_hotpotqa_1500.json |
| 191519 | B tail 176242 s43, ten chunks | PENDING | 1,500 | ~20 min | out expQ/spread/b_tail_s43/eval/…_text_hotpotqa_1500.json |
| 191526 | B tail 176242 s43, one chunk | PENDING | 1,500 | ~20 min | out expQ/spread/b_tail_s43/eval/…_text_hotpotqa_1500_g10.json |
| 191528 | B spread 184780 s43, one chunk | PENDING | 1,500 | ~20 min | out expQ/spread/b_spread_s43/eval/…_text_hotpotqa_1500_g10.json |
All four: TOPK=10 CML=4096, gpu-be `--exclude=homer-[005-009]`, `--time 1:15` (slowest text cell 24:43 × 3), `--mem 8G` (MaxRSS 4.43 GB × 1.5); symlinks `expQ/spread/b_tail_s43/model → ft_mm_4b_176242/model`, `b_spread_s43/model → ft_mm_4b_184780/model` (176242 has no `compressor_mem_layout` key = tail; 184780 spread, stride 8).

## Adopted jobs
(none) — the queue holds longdoc's 191392 / 191516 / 191517 and qafreeze's 191495 / 191496 (not ours, never touched) and the daemon 188764.

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| A. case audit | **decided: PASS** (191454) | add a seed-pooled statistic if the owner picks option A in BLOCKED.md |
| B. long chunks, full budget | s42 done; s43 + s42 ten-chunk re-run PENDING | when 191518-191528 end: pair one chunk s43 (191528 − 191526); pair ten chunks s42 (191518 − 191455); `compare_answers.py` 191518 vs 188615 (text re-run identity); launch the spread ten-chunk s43 re-run (184780, text_hotpotqa_1500). Verdict: one chunk "better" if s43 ≥ +3 pt CI > 0; ten chunks "not resolvable here (CI width w)" unless the owner answers BLOCKED.md |
| J. instrument | not started | write `scripts/spread/rejudge.sbatch` = `judge_mm_docqa.py --prompt project` on COPIES under `expQ/spread/rejudge/` (the longdoc twin `scripts/longdoc/rejudge.sbatch` is a model to read, not to run); files: the 8 gate pairs + the arm B one-chunk pairs; EM / match / ANLS already read (one chunk s42: all same sign) |
| C. many pages | not started | longdoc's spread cells exist (10 pages s42 191378 / s43 191390; 20 pages s42 191377); launch TAIL 174908 / 176242 at 10 pages (XD=6 TOPK=10) and 20 pages (XD=16 TOPK=20), PX=1003520, DML from longdoc's audit; 20-page spread s43 is missing on longdoc's side |
| D. what slots hold | not started | write `scripts/spread/mem_recall.sbatch` (OUT under expQ/spread, runs scripts/longdoc/mem_recall.py read-only) |
| re-runs for the full-budget gating pairs | proposed | tail production 174908 / 176242 on documents + photos on today's code, el9_8 node: makes the infographics claim and the photo pairs decidable (4 cells, ≈ 2-4 GPU-h) |
| E / F | owner checkpoint | — |

Budget used: 8 cells of ~50 (1 CPU + 7 text evals), GPU-h ≈ 0.9 done (3 × ~18 min) + ≈ 1.3 pending. LLM wakes 2 of ≤ 40.

## Health evidence this wake
- md5 at launch of 191518-191528 = at launch of 191455-191457: eval_mm_4b.sh b3d1cda1, eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py 02bc8748, paired_judge.py a3cc13b2, pisco/model.py 994cff3d, judge_prompts.py bae44fa7.
- Verifier fixes done this wake: text CI range corrected to .041-.056 and photos to .037-.041 (STATE, CAMPAIGN.md gate line, BLOCKED.md, spread_cases.md); photo-drift counts now in a file; the arm B ten-chunk pair marked "reported" with its three code differences, and a today's-code re-run launched.

## Proposed replacement text (the owner's chats write these files; provisional until arm B s43 / the re-runs)
- `docs/architecture_review.md:252-256`, replace "the +.08 of the reduced chains is what the extra stage-2 data was buying" with: "At full budget the spread chain scores what it scored at reduced budget (+.011 / −.003 on documents); the tail models gained +.093 / +.048 from the larger budget (stage 1 5M and three stage-2 passes), most on MP-DocVQA. The layout reaches the plateau with less data; whether it changes the plateau is untested (campaign spread, arm E). Where the input leaves the finetune's regime the layout still decides: on ten HotpotQA passages compressed as ONE chunk the production tail model falls to .351 and spread holds .611 (s42). A 1,500-Q document eval cannot resolve differences under ~3.4 pt (80% power)."
- HANDOFF.md thread 1: "spread vs tail: the reduced-budget gain was the tail models catching up with budget, not spread losing it; spread still wins by +26 pt when long text is compressed as one chunk (s42, s43 running); infographics +5 pt and one long passage +8.7 pt (s42) favour spread at full budget, provisional."

## Blockers / needs the owner
- BLOCKED.md: how "equal" is tested (recommendation: seed-pooled CI, 0 GPU-h). The campaign keeps running meanwhile.
