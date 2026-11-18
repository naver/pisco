# STATE — campaign spread   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-18 21:00-21:40 (fifth wake; the 19:30 wake died on the account's 5-hour usage limit, so this one read
the four jobs that ended meanwhile). Deliverable: `docs/spread_cases.md`. Tables: audit `expQ/spread/audit_191454/audit.md`;
seed-pooled test `expQ/spread/pooled_20261118_1730/pooled.md`; arm J per scoring
`expQ/spread/rejudge/pooled_20261118_2100/pooled{,_judge_rubric,_em,_anls}.md`; arm C growth test
`expQ/spread/did_20261118_2100/did{,_em}.md`; re-run identity `expQ/spread/rerun_checks/rerun_20261118.md`.

## TL;DR — where this leaves the question
**Settled**
- Arm A feasibility gate PASS (job 191454, CPU, closed): the 8 recorded spread − tail judge differences re-derive exactly; 39 self-pairs 0.000; 184778's document file 1,500 rows, judge on all, 0 unparseable, mems_used 646,042.
- Seed-pooled "equal" test controls PASS (17:45, and again at 21:10 on the arm J copies: re-pairing the copies' rubric verdicts reproduces the rubric table to the third decimal): reduced-budget documents +.084 [+.065, +.103] (expected +.084), every self-pair 0.000.
- **Arm B closed + arm J checked.** ONE chunk (ten HotpotQA passages as one ~1,380-token chunk, 173 slots per question, uncompressed .752), n 1,500 per seed, spread − tail: rubric judge +.260 [+.233, +.289] (s42, 191457 − 191456, McNemar 488 / 98) / +.247 [+.219, +.275] (s43, 191528 − 191526, 477 / 106); project prompt +.271 [+.243, +.301] / +.253 [+.224, +.281]; EM +.151 [+.127, +.175] / +.152 [+.126, +.177] (job 191665). Better by the bar (≥ +3 pt, CI > 0, both seeds) under every scoring. Arm J's literal rule ("within 2 pt across scorings") fails on SIZE only: EM is 10 pt smaller than the judges, because both models rarely hit the exact string.
- Tail production loses going from ten chunks to one at both seeds: −.253 [−.281, −.225] (s42, 191456 − 191455), −.229 [−.255, −.200] (s43, 191526 − 191519). Spread does not: +.006 [−.019, +.029] (s42, 191457 − 191518), +.023 [+.000, +.045], 178 / 144, p .066 (s43, 191528 − 191630).
- TEN chunks (today's code, n 1,500 per seed) are **judge-sensitive** (arm J, job 191665): rubric +.001 [−.021, +.025] / −.004 [−.027, +.019] (191518 − 191455, 191630 − 191519), pooled −.001 [−.0203, +.018]; project prompt pooled −.003 [−.022, +.016]; EM −.027 [−.047, −.007] (p .010) / −.020 [−.039, +.000] (p .058), pooled −.023 [−.039, −.007]; ANLS pooled −.030 [−.045, −.014]. Spread's answers are longer (3.43 vs 3.01 words s42, 3.91 vs 3.35 s43) with 42 / 24 more judged-right-but-EM-0 answers: a wording difference, not a reading one.
- Full-budget PHOTOS (stage G = the law's re-run rule, CAMPAIGN.md:70-71, applied to the gating pair), n 1,500: rubric −.007 [−.027, +.011], 105 / 116 (s42, 191531 − 191631); +.010 [−.011, +.030], 125 / 110 (s43, 191532 − 191632); pooled +.001 [−.014, +.018] → **equal** under the deciding rubric judge; EM pooled −.003 [−.017, +.012] equal too. Arm J: same sign per seed and within 2 pt under every scoring, but under the project prompt alone "equal" would not be certified (s42 −.023 [−.043, −.004], pooled −.006 [−.022, +.010]).
- Full-budget DOCUMENTS, 4 pages (431 slots per question), n 1,500: rubric +.011 [−.014, +.035], 177 / 161 (s42, 184778 vs 174908, both on 24b7a2c; 184778 re-run 191495 identical) / +.024 [−.002, +.048], 193 / 157 (s43, 191496 − 191633); pooled +.017 [−.002, +.036], width .038 → **not resolvable here**. Arm J: same sign and within 2 pt under every scoring (project +.019 / +.029, pooled +.024 [+.006, +.042]; EM +.001 / +.022; ANLS pooled +.029 [+.012, +.046]) — a 1-3 pt lean to spread, below the +3 pt bar at both seeds under every scoring. Uncompressed .809.
- **Arm C closed (many pages): the layout does not pay more as pages are added.** Per-question growth (spread − tail at N pages) − (at 4 pages), rubric judge, n 1,500: 10 pages −.004 [−.027, +.019] (s42, longdoc 191378 vs 191666) / −.007 [−.031, +.015] (s43, longdoc 191390 vs 191667); 20 pages s42 +.009 [−.015, +.031] (longdoc 191377 vs 191668); EM +.003 / +.007 / +.013. Control: arm B's one-vs-ten-chunk growth recovered +.259 / +.251, self-pairs 0.000. Plain differences: 10 pages +.007 [−.017, +.031] / +.017 [−.007, +.040]; 20 pages s42 +.019 [−.006, +.043]. Slot counts equal to the spread twins (1,614,940 / 3,229,884). The 20-page s43 pair is not needed (the gate needs both seeds; s42 and both 10-page seeds already miss it) → hand-off 19ac33 withdrawn.
- Re-runs on today's code (rerun_20261118.md): documents 184778, 184780, 176242 0 / 1,500 answers differ; photos 184780 and 176242 0 / 1,500; photos 184778 223 / 1,500 (68 judge flips); photos 174908 135 / 1,500 (34 flips); 184780 ten chunks 0 / 1,500.
- `PX` is inert on photos: every `mm_naturalqa_1500` file has 79,322 slots at 451,584 and 1,003,520 px.

**Open, and running**
- provisional: spread better on infographics at full budget, 4 pages: +.050 [+.004, +.096] 80 / 55 (s42) / +.050 [+.008, +.096] 76 / 51 (s43), n 500 each (audit.md:198 / :230). It persists at 10 pages (+.028 [−.014, +.070] / +.070 [+.030, +.114]) and 20 pages (s42 +.056 [+.016, +.096]), but these are the SAME 500 questions with distractors, so they are not a repeat; MP-DocVQA leans tail at the same sizes (−.036 / −.004; −.042). One repeat on new infographic questions before it counts.
- provisional: the gain did not vanish because spread got worse: spread full − reduced documents +.011 [−.011, +.033] / −.003 [−.026, +.020]; tail production − reduced +.093 [+.068, +.117] / +.048 [+.025, +.072] (job 191454). NOT one variable; the tail s42 reduced file ran on homer-009 and the spread s43 reduced file 183816 on homer-008 (image outlier node).
- provisional: the tail models' budget gain sits mostly on MP-DocVQA +.162 / +.090, slides +.076 / +.040, infographics +.040 / +.014 (n 500 per source, job 191454); no between-source test.
- MRQA one long passage (~900 tokens, 53 slots per question): spread on today's code .715 / .724 (qafreeze 191675 / 191676, read-only); tail re-runs on today's code 191779 (174908) / 191780 (176242) PENDING. The old cross-code pair (+.087 [+.063, +.112], 188606 vs 185780) stays reported only.
- Arm D (what the slots hold): job 191785 PENDING — ownership + answer loss of the four stage-2 models; the reduced pair is the control.

## Owned jobs (jobid | arm | state | rows | eta | note)
| 191454 | A audit | COMPLETED | 23 pairs | — | closed: PASS |
| 191455-191457, 191518, 191519, 191526, 191528, 191630 | B (one / ten chunks × 2 layouts × 2 seeds) | COMPLETED | 1,500 each | — | closed healthy |
| 191631 / 191632 / 191633 | G tail photos s42 / s43, documents s43 | COMPLETED | 1,500 each | — | closed healthy |
| 191665 | J re-judge, project prompt, 16 copies | COMPLETED 0:0 7:27 homer-017 | 16 × 1,500 | — | closed healthy (unparseable 0 × 16) |
| 191666 / 191667 | C tail 10 pages s42 / s43 | COMPLETED 0:0 51:36 / 50:32 homer-010 | 1,500 each | — | closed healthy, .600 / .591 |
| 191668 | C tail 20 pages s42 | COMPLETED 0:0 1:30:50 homer-011 | 1,500 | — | closed healthy, .579 |
| 191779 | MRQA tail 174908 s42, PX=1003520 CML=4096 | PENDING (Priority) | 1,500 | ~17 min once running | → `expQ/spread/b_tail_s42/eval/mm_docqa_pisco_px1003520_cml4096_text_mrqa_ood_1500.json`; --time 0:52, 9G |
| 191780 | MRQA tail 176242 s43 | PENDING (Priority) | 1,500 | ~17 min | → `expQ/spread/b_tail_s43/eval/…cml4096_text_mrqa_ood_1500.json`; --time 0:52, 9G |
| 191785 | D mechanism probes, 4 stage-2 models | PENDING (Priority) | 40 rows × 3 sets × 4 models | ~1.1 h | → `expQ/spread/mech/<name>_191785/probe_{mixer,connector}.json`, log `expQ/spread/mech_191785.out`; --time 3:20, 8G, gpu_a100 |
All three: gpu-be `--exclude=homer-[005-009]`, --parent 191454. `--time` = slowest measured wall × 3 (MRQA 188613 17:11;
probes 183828 12:00 + 183832 4:28 per model × 4). `--mem` = MaxRSS × 1.5 (MRQA 188613 5.7 GB; probes 183831 4.24 GB).

## Adopted jobs
(none) — the queue held only the daemon 188764, other projects' jobs and qafreeze's 191766-191768.

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| A. case audit | **decided: PASS** (191454) | re-run `--pooled` on every new pair set |
| B. long chunks | **closed**: one chunk better under every scoring; ten chunks judge-sensitive | — |
| G. full-budget gating pairs | **decided**: photos equal (rubric, EM); documents not resolvable | into the checkpoint question (option B, a second document set) |
| J. instrument | **decided** (191665) | — |
| C. many pages | **closed**: no growth with pages | — |
| MRQA one long passage | tail re-runs PENDING (191779 / 191780) | pair with qafreeze 191675 / 191676 (`scripts/paired_judge.py`), pooled test; health: mems_used 80,003 = spread twin |
| D. what slots hold | 191785 PENDING | control first: reduced pair must give ownership .977 / .799 (±.02) and pages NLL 1.138 / 1.334; then the gate pair pt_mm_5M_spread vs pt_mm_v4kd_4b: does the reduced-budget advantage shrink inside ±2 pt? Transcription (mem_recall) only if the owner wants it in the mechanism table |
| E / F | owner checkpoint | write ONE BLOCKED.md question once MRQA and D are read: case table, the stage-1 config diff (`expQ/qwen35_4b_pt_5M_143228` vs `expQ/archq3/pt_text_5M_spread` training_config.yaml lines 46-47), arm E cost (≈ 60 GPU-h), option B, the infographics repeat, the judge-sensitivity note on the 2-pt rule, up to 2 fixes |

Budget used: 19 cells of ~50 (1 CPU + 18 GPU); GPU-h ≈ 7.0 done + ≈ 1.7 queued ≈ 8.7 of 30 before the checkpoint.
LLM wakes 5 of ≤ 40 (one more died on the usage limit at 19:30).

## Health evidence this wake
- 191665: 16 / 16 files `unparseable: 0`; copies' `judge_rubric` and answers = the source files on 16 × 1,500 rows (checked record by record).
- 191666-191668: COMPLETED 0:0, one attempt each; 1,500 rows, judge on every row, unparseable 0, no "exceeds" line in the log, mems_used equal to the spread twin. Probes recorded with one `--map`.
- Spread twins (longdoc 191377 / 191378 / 191390): homer-017 (el9_8), eval_mm_4b.sh 50abc167 (requeue markers only differ), same python md5s.
- MRQA: qafreeze 191675 / 191676 wrote their files in earlier attempts (sacct -X shows the 0 s requeue that reused them; known_issues row added); their summary settings equal the recorded tail file 185780 except the checkpoint.
- md5 at launch of 191779 / 191780 / 191785 = every earlier cell: eval_mm_4b.sh b3d1cda1, eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py 02bc8748, paired_judge.py a3cc13b2, pisco/model.py 994cff3d, judge_prompts.py bae44fa7; probes probe_mixer.py e2463c57, probe_connector.py ff63deeb.
- Code written this wake (all under scripts/spread, freely patchable): `slice_pairs.py` gains `--metric` for `--pooled` (default `judge`, so earlier calls are unchanged; the rubric re-pairing of the copies reproduces pooled.md exactly); new `did_pairs.py` + `did_pairs.tsv` (growth test, control = arm B recovered); new `rejudge_pairs.tsv`; new `mech_probes.sbatch` de7a50c3 (`bash -n` OK; runs the two archq2 probes unchanged, only OUT moves to expQ/spread/mech/).
- Verifier items fixed: one-chunk "better" now has its arm J reading (better under every scoring; the 2-pt rule fails on size only); photos "equal" now names the law rule behind stage G and the arm J result; infographics CIs now quote audit.md (+.004, +.096) / (+.008, +.096) here and in docs/spread_cases.md.

## Proposed replacement text (the owner's chats write these files; provisional until MRQA and arm D)
- `docs/architecture_review.md:252-256`, replace "the +.08 of the reduced chains is what the extra stage-2 data was buying" with: "At full budget the spread chain scores what it scored at reduced budget (+.011 / −.003 on documents); the tail models gained +.093 / +.048 from the larger budget, most on MP-DocVQA. On today's code, full budget: photos equal (pooled +.001 [−.014, +.018]); 4-page documents +.017 [−.002, +.036], a 1-3 pt lean to spread under every scoring that 1,500 questions × 2 seeds cannot resolve; adding pages (10, 20) does not widen it. Where the input leaves the finetune's regime the layout decides: ten HotpotQA passages compressed as ONE chunk give tail .351 / .369 and spread .611 / .616 (+26 / +25 pt, both seeds, under both judge prompts and EM). Whether the layout changes the document plateau is untested (campaign spread, arm E)."
- HANDOFF.md thread 1: "spread vs tail: the reduced-budget gain was the tail models catching up with budget, not spread losing it; spread wins by +25-26 pt when long text is compressed as one chunk (both seeds, every scoring); on ten short chunks the judges tie and EM favours tail (spread answers are longer); photos tie; 4-page documents +1.7 pt and more pages do not widen it."

## Blockers / needs the owner
- None now. The owner checkpoint (before arm E) is the next wake's question, after MRQA and arm D are read.
