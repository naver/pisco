# STATE — campaign spread   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-17 18:30-19:15 (first wake). Deliverable: `docs/spread_cases.md`. Audit tables: `expQ/spread/audit_191454/audit.md`.

## TL;DR — where this leaves the question
**Settled**
- Arm A feasibility gate PASS (job 191454, CPU, closed): the 8 recorded spread − tail judge differences re-derive exactly from the recorded files — documents +.093 / +.075 (reduced budget, seeds 42 / 43), photos +.023 / +.021, full budget documents +.011 / +.024, photos −.009 / +.010, with the recorded CIs; 39 self-pairs give 0.000; 184778's document file has 1,500 rows, a judge on every row, 0 unparseable, mems_used 646,042.
- provisional: the gain did not vanish because spread got worse: spread full − spread reduced = documents +.011 [−.011, +.033] / −.003 [−.026, +.020], photos +.006 (s42); tail production − tail reduced = documents +.093 [+.068, +.117] / +.048 [+.025, +.072], photos +.037 [+.017, +.058] (n 1,500, job 191454). These pairs are NOT one variable (stage 1 5M vs 1M, stage 2 160k vs 100k for spread; three stage-2 passes for production).
- provisional: the tail models' budget gain is largest on multi-page documents: MP-DocVQA +.162 / +.090, slides +.076 / +.040, infographics +.040 / +.014 (n 500 per source, job 191454).
- The gating sets cannot show "equal": one seed's 95% CI on 1,500 Q is .044-.050 wide on documents and text, .036-.041 on photos (job 191454, 23 pairs) — wider than the ±2-pt band. Owner question in BLOCKED.md.
- Photo answers drift across code dates: 184778's photo file (184783, homer-011, 11-09) vs its re-run (188602, homer-016, 11-11): 233/1,500 answers differ, judge .677 vs .673, 85 discordant; 184780's photo file from homer-008 equals its re-run 1,500/1,500; document re-runs identical (checked this wake on the recorded files).

**Open, and running**
- provisional: spread better on infographics at full budget, +.050 [+.004, +.096] / +.050 [+.008, +.096] (n 500, job 191454); the s43 tail file (176242) is from an older code era → needs a re-run on today's code.
- provisional: spread better on one long MRQA passage at full budget, s42 only: .718 vs .631 = +.087 [+.063, +.112] (n 1,500; freeze re-run 188606 vs 185780); s43 needs a 176242 cell.
- Arm B (long text chunks at full budget), seed 42: 191455 tail 174908 ten chunks, 191456 tail 174908 one chunk, 191457 spread 184778 one chunk — all PENDING (Priority) on gpu-be. Spread ten chunks s42 = the freeze re-run `expQ/freeze/ref_s42/eval/mm_docqa_pisco_k10_px1003520_cml4096_text_hotpotqa_1500.json` (.603, same eval code, homer-016).

## Owned jobs (jobid | arm | state | elapsed | rows | eta | note)
| 191454 | A audit | COMPLETED 0:01:45 | 23 pairs, 39 files | — | closed: PASS |
| 191455 | B tail s42, ten chunks | PENDING | 1,500 Q | ~25 min once running | out expQ/spread/b_tail_s42/eval/mm_docqa_pisco_k10_cml4096_text_hotpotqa_1500.json |
| 191456 | B tail s42, one chunk | PENDING | 1,500 Q | ~25 min | out expQ/spread/b_tail_s42/eval/…_text_hotpotqa_1500_g10.json |
| 191457 | B spread s42, one chunk | PENDING | 1,500 Q | ~25 min | out expQ/spread/b_spread_s42/eval/…_text_hotpotqa_1500_g10.json |

## Adopted jobs
(none) — the queue holds longdoc's 191392 / 191393 (not ours, never touched) and the daemon 188764.

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| A. case audit | **decided: PASS** (191454) | add a seed-pooled statistic if the owner picks option A in BLOCKED.md |
| J. instrument | not started | one cell: `scripts/spread/rejudge.sbatch` (to write) re-judges COPIES of the 8 gate files with `--prompt project` (≈12k records; the 2026-10 rubric rejudge did ~16k records in 13:32 → `--time` 0:45) |
| B. long chunks, full budget | 3 of 6 cells PENDING (s42) | when they end: pair (spread − tail at one chunk; at ten chunks vs the freeze s42 file); launch s43: 176242 ten + one chunk, 184780 one chunk (ten chunks s43 = `expQ/freeze/ref_s43` file) |
| C. many pages | not started | wait for longdoc's spread cells (10 pages: l_r42 / l_r43 exist; 20 pages: 191377), then the TAIL cells on 174908 / 176242 |
| D. what slots hold | not started | write `scripts/spread/mem_recall.sbatch` (OUT under expQ/spread, runs scripts/longdoc/mem_recall.py read-only); probe_mixer / probe_connector write into `expQ/archq2/<tool>_<jid>` by design — new dirs, never edits |
| re-runs for the full-budget gating pairs | proposed | tail production 174908 / 176242 on documents + photos on today's code, el9_8 node: makes the infographics claim and the photo pairs decidable (4 cells, ≈ 2-4 GPU-h) |
| E / F | owner checkpoint | — |

Budget used: 4 cells of ~50 (1 CPU + 3 text evals), GPU-h ≈ 0 so far (B s42 will be ≈ 1.2).

## Health evidence this wake
- 191454: COMPLETED exit 0 in 1:45 on homer-016, md5 slice_pairs.py 229600a9 (the version before the exit-code line; now 8acfcab7, same tables + exit 3 on FAIL), audit_pairs.tsv 67add90f, paired_judge.py a3cc13b2.
- Eval code md5 at launch of 191455-191457: eval_mm_4b.sh b3d1cda1 (the law's 50abc167 + requeue markers and an off-by-default DENSE env, same python call — `git diff 2c2333d HEAD -- eval_mm_4b.sh`), eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py 02bc8748, pisco/model.py 994cff3d.
- Same-node rule on the 8 gate pairs (inventory): red_docs_s43 (spread 183816 on homer-008) and full_photos_s43 (spread 184780 on homer-008) carry a homer-008 side; red_docs_s42 / red_photos_s42 one side on untested homer-009; full_docs_s43 and full_photos_s42 span two code eras (24b7a2c vs ea0c591); full_docs_s42 is clean (homer-011 vs homer-012, same era).

## Proposed replacement text (the owner's chats write these files; provisional until arm B / the re-runs)
- `docs/architecture_review.md:252-256`, replace "the +.08 of the reduced chains is what the extra stage-2 data was buying" with: "At full budget the spread chain scores what it scored at reduced budget (+.011 / −.003 on documents); the tail models gained +.093 / +.048 from the larger budget (stage 1 5M and three stage-2 passes), most on MP-DocVQA. The layout reaches the plateau with less data; whether it changes the plateau is untested (campaign spread, arm E). A 1,500-Q document eval cannot resolve differences under ~3.4 pt (80% power)."
- HANDOFF.md thread 1: "spread vs tail: the reduced-budget gain was the tail models catching up with budget, not spread losing it; infographics +5 pt and one long passage +8.7 pt (s42) favour spread at full budget, provisional."

## Blockers / needs the owner
- BLOCKED.md: how "equal" is tested (recommendation: seed-pooled CI, 0 GPU-h). The campaign keeps running meanwhile.
