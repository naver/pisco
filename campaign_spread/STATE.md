# STATE — campaign spread   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-19 02:00-02:40 (sixth wake; the 23:30 wake died on the account's 5-hour usage limit, so this one
read the three jobs that ended meanwhile). **Every inference arm is read; the campaign waits at the owner checkpoint
(BLOCKED.md: arm E, and the arm J size rule).** Deliverable: `docs/spread_cases.md`. Tables: audit
`expQ/spread/audit_191454/audit.md`; seed-pooled test `expQ/spread/pooled_20261118_1730/pooled.md`; arm J per scoring
`expQ/spread/rejudge/pooled_20261118_2100/pooled{,_judge_rubric,_em,_anls}.md`; arm C growth test
`expQ/spread/did_20261118_2100/did{,_em}.md`; arm D `expQ/spread/mech/*_191785/probe_{mixer,connector}.json`.

## TL;DR — where this leaves the question
**Settled**
- Arm A feasibility gate PASS (job 191454, CPU, closed): the 8 recorded spread − tail judge differences re-derive exactly; 39 self-pairs 0.000; 184778's document file 1,500 rows, judge on all, 0 unparseable, mems_used 646,042.
- Seed-pooled "equal" test controls PASS (17:45 on the rubric files): reduced-budget documents +.084 [+.065, +.103] (expected +.084), every self-pair 0.000. The 21:10 runs on the arm J copies (`rejudge_pairs.tsv`) carry self-pairs only: no project-prompt copy of the reduced files exists, so the +.084 row cannot be added there; re-pairing the copies' rubric verdicts reproduced pooled.md exactly.
- Tail production loses going from ten HotpotQA chunks to one at both seeds: −.253 [−.281, −.225] (s42, 191456 − 191455), −.229 [−.255, −.200] (s43, 191526 − 191519). Spread does not: +.006 [−.019, +.029] (s42, 191457 − 191518), +.023 [+.000, +.045], 178 / 144, p .066 (s43, 191528 − 191630).
- TEN HotpotQA chunks (today's code, n 1,500 per seed) are **judge-sensitive** (arm J, job 191665): rubric +.001 [−.021, +.025] / −.004 [−.027, +.019] (191518 − 191455, 191630 − 191519), pooled −.001 [−.0203, +.018]; project prompt pooled −.003 [−.022, +.016]; EM −.027 [−.047, −.007] (p .010) / −.020 [−.039, +.000] (p .058), pooled −.023 [−.039, −.007]; ANLS pooled −.030 [−.045, −.014]. Spread's answers are longer (3.43 vs 3.01 words s42, 3.91 vs 3.35 s43) with 42 / 24 more judged-right-but-EM-0 answers.
- Full-budget PHOTOS (stage G = the law's re-run rule, CAMPAIGN.md:70-71), n 1,500: rubric −.007 [−.027, +.011], 105 / 116 (s42, 191531 − 191631); +.010 [−.011, +.030], 125 / 110 (s43, 191532 − 191632); pooled +.001 [−.014, +.018] → **equal** under the rubric judge; EM pooled −.003 [−.017, +.012] equal too. Under the project prompt alone "equal" is not certified (s42 −.023 [−.043, −.004], pooled −.006 [−.022, +.010]).
- Full-budget DOCUMENTS, 4 pages (431 slots per question), n 1,500: rubric +.011 [−.014, +.035], 177 / 161 (s42, 184778 vs 174908, both on 24b7a2c; 184778 re-run 191495 identical) / +.024 [−.002, +.048], 193 / 157 (s43, 191496 − 191633); pooled +.017 [−.002, +.036], width .038 → **not resolvable here**. Same sign and within 2 pt under every scoring (project pooled +.024 [+.006, +.042]; EM +.001 / +.022; ANLS pooled +.029 [+.012, +.046]). Uncompressed .809.
- **Arm C closed (many pages): the layout does not pay more as pages are added.** Growth (spread − tail at N pages) − (at 4 pages), rubric, n 1,500: 10 pages −.004 [−.027, +.019] (s42, longdoc 191378 vs 191666) / −.007 [−.031, +.015] (s43, longdoc 191390 vs 191667); 20 pages s42 +.009 [−.015, +.031] (longdoc 191377 vs 191668). Control (arm B one-vs-ten-chunk growth) recovered +.259 / +.251, self-pairs 0.000.
- **MRQA one long passage, today's code both sides** (53 slots per question, slot counts equal 80,003, uncompressed .807): spread 184778 / 184780 .715 / .724 (qafreeze 191675 / 191676, read-only) vs tail 174908 / 176242 .631 / .626 (191779 / 191780) → rubric **+.084 [+.060, +.109], 239 / 113** (s42) / **+.098 [+.075, +.122], 249 / 102** (s43), p < .001 each: the better bar holds at both seeds. EM +.057 [+.034, +.081] / +.074 [+.051, +.096]; ANLS +.076 / +.078; project prompt +.070 [+.046, +.095] / +.103 [+.079, +.128] (job 192006, 4 copies, unparseable 0; the copies' rubric verdicts reproduce +.084 / +.098).
- **Arm D closed (mechanism, job 191785, no accuracy claim).** Control PASS: the reduced pair reproduces ownership .977 / .799 and answer loss pages 1.138 / 1.334, text 1.812 / 1.927, photos 1.765 / 1.802 to the third decimal. Full-budget gate pair `pt_mm_5M_spread` vs `pt_mm_v4kd_4b`: text ownership .972 vs .763 (+.208; reduced +.178; 3,053 slots); answer loss (nats, 40 rows, unpaired means) pages 1.102 vs 1.226 (−.124; reduced −.196), text 1.844 vs 1.788 (+.056; reduced −.115), photos 1.736 vs 1.813 (−.077; reduced −.037). The advantage does not shrink inside ±2 pt on ownership, pages or photos (text reverses) → the full-budget accuracy tie is not a tie of what the slots hold.
- No disjoint document questions are left in the gating sources: VisRAG slides 556 / infographics 718 / MP-DocVQA 591 questions (`scripts/build_mm_eval.py:5`), the gating set uses 500 each → 56 + 218 + 91 left; `mm_docqa_retr10_1m`, `retr4_1m`, `longdoc_1m` reuse the gating question ids 1,500 / 1,500 (500 / 500), checked by query_id this wake.
- Re-runs on today's code (rerun_20261118.md): documents 184778, 184780, 176242 0 / 1,500 answers differ; photos 184780 and 176242 0 / 1,500; photos 184778 223 / 1,500 (68 judge flips); photos 174908 135 / 1,500 (34 flips); 184780 ten chunks 0 / 1,500. `PX` is inert on photos (79,322 slots at both budgets).

**Open, and running**
- provisional (owner question 2): ONE HotpotQA chunk is "better" under every scoring — rubric +.260 [+.233, +.289] (s42, 191457 − 191456, 488 / 98) / +.247 [+.219, +.275] (s43, 191528 − 191526, 477 / 106); project +.271 / +.253; EM +.151 [+.127, +.175] / +.152 [+.126, +.177] (job 191665) — but arm J's literal rule (within 2 pt across scorings) fails on size: EM is 10 pt smaller. MRQA has the same pattern (project prompt within 1.4 pt of the rubric, EM 2.4-2.7 pt below it). Until the owner rules, both stay "better by every scoring, size judge-sensitive".
- provisional (untested): the ten-chunk EM gap is answer wording, not reading (longer spread answers, more judged-right-EM-0).
- provisional: MRQA by passage length (slice chosen after the data): > 100 slots (DuoRC, TextbookQA, n 343) +.187 [+.131, +.245] / +.230 [+.169, +.289]; ≤ 30 slots (mostly RelationExtraction, n 429) +.082 / +.065; 31-60 (n 568) +.007 / +.048; 61-100 n 160 not resolvable. Largest on the longest passages, but not length alone.
- provisional: spread better on infographics at full budget, 4 pages: +.050 [+.004, +.096] 80 / 55 (s42) / +.050 [+.008, +.096] 76 / 51 (s43), n 500 each (audit.md:198 / :230); post-hoc pick among 3 sources. A repeat on new questions could use only 218 → under the 300-question floor → not run, stays provisional.
- provisional: the gain did not vanish because spread got worse: spread full − reduced documents +.011 / −.003; tail production − reduced +.093 / +.048 (job 191454). NOT one variable (three stage-2 passes vs one); tail s42 reduced file on homer-009, spread s43 reduced file 183816 on homer-008.
- provisional: the tail models' budget gain sits mostly on MP-DocVQA +.162 / +.090, slides +.076 / +.040, infographics +.040 / +.014 (n 500 per source, job 191454); no between-source test.

## Owned jobs (jobid | arm | state | rows | eta | note)
| 191454 | A audit | COMPLETED | 23 pairs | — | closed: PASS |
| 191455-191457, 191518, 191519, 191526, 191528, 191630 | B (one / ten chunks × 2 layouts × 2 seeds) | COMPLETED | 1,500 each | — | closed healthy |
| 191631 / 191632 / 191633 | G tail photos s42 / s43, documents s43 | COMPLETED | 1,500 each | — | closed healthy |
| 191665 | J re-judge, 16 gating copies | COMPLETED | 16 × 1,500 | — | closed healthy |
| 191666 / 191667 / 191668 | C tail 10 pages s42 / s43, 20 pages s42 | COMPLETED | 1,500 each | — | closed healthy |
| 191779 / 191780 | MRQA tail 174908 / 176242 | COMPLETED 0:0 16:17 / 16:54 homer-011 | 1,500 each | — | closed healthy |
| 191785 | D mechanism probes, 4 stage-2 models | COMPLETED 0:0 56:54 homer-011 | 40 rows × 3 sets × 4 | — | closed, control PASS |
| 192006 | J re-judge, project prompt, 4 MRQA copies | COMPLETED 0:0 2:23 homer-019 | 4 × 1,500 | — | closed healthy, unparseable 0 × 4 |

## Adopted jobs
(none) — the queue held only the daemon 188764 and other projects' jobs (slotcheck 192002 / 192003 / 192009 / 192010, longdoc 192007 / 192008).

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| A. case audit | **decided: PASS** (191454) | re-run `--pooled` on every new pair set |
| B. long chunks | **closed**; one chunk better under every scoring (size rule: owner question 2) | — |
| G. full-budget gating pairs | **decided**: photos equal; documents not resolvable | — |
| J. instrument | **decided** (191665, MRQA 192006) | — |
| C. many pages | **closed**: no growth with pages | — |
| MRQA one long passage | **decided: better** at both seeds under every scoring (size rule: owner question 2) | — |
| D. what slots hold | **closed** (191785): the tie is not a representation tie | page transcription not run (≈ 2 GPU-h per model; not asked) |
| E. tail control, one stage-2 pass | **waiting for the owner** (BLOCKED.md question 1; recommendation: option A, s43 only if s42 ≥ +3 pt) | on a REQUEST: stage-2 smoke `pt_mm_tail5m.sbatch e_pt_smoke 20` (`--time 0:55:00`, `gpucap land a100 1` first) → check INVARIANT 0 → full stage 2 `--time 37:45:00` → ft s42 `--time 17:17:00` (afterok) → judged evals; preview at a quarter of stage 2 is pre-authorised |
| F. fixes | none proposed (BLOCKED.md question 1) | — |

Budget used: 20 cells of ~50 (1 CPU + 19 GPU); GPU-h 8.6 (sacct over the 20 runs) of 30 before the checkpoint / 95 in
all. LLM wakes 6 of ≤ 40 (two more died on the usage limit, 19:30 and 23:30).

## Health evidence this wake
- 191779 / 191780: COMPLETED 0:0, one attempt each, homer-011; 1,500 rows, `judge` on every row, `judge_unparseable` 0, no "exceeds" line in the log (0 / 0), mems_used 80,003 = spread twin files (qafreeze 191675 / 191676, same 1,500 query ids); summary settings equal to the twin's except the checkpoint (topk 4, PX 1,003,520, CML 4096, brief). Probes recorded with one `--map`.
- 192006: COMPLETED 0:0 2:23 homer-019, one attempt; 4 copies × 1,500 rows, project prompt on every row, unparseable 0 (4 / 4); the copies' `judge_rubric` re-pair to +.084 / +.098 = the source files.
- 191785: COMPLETED 0:0 56:54, MaxRSS 4.5 GB, 8 JSONs; probe scripts unchanged (probe_mixer.py e2463c57, probe_connector.py ff63deeb via mech_probes.sbatch de7a50c3).
- md5 today (= at launch of every cell): eval_mm_4b.sh b3d1cda1, eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py 02bc8748, paired_judge.py a3cc13b2, pisco/model.py 994cff3d, judge_prompts.py bae44fa7; git log shows no change to them since 2026-11-17 17:17.
- New this wake (scripts/spread, freely patchable): `pt_mm_tail5m.sbatch` 2de3a08b and `ft_tail5m.sbatch` 3a1b4f7c, arm E launchers, `bash -n` OK, NOT run (no training before the owner's REQUEST). They replay the recorded argv of 184777 / 184778 / 184780 (`.expmon/runs.jsonl`) with the data env read from each `training_config.yaml` line 25 (PT_DATA / FT_DATA were set at launch, not in argv), `+key` written `++key`, and end with an automatic invariant check (diff vs the recipe yaml minus model_name_or_path / max_steps / output keys must be 0 lines, else exit 5). Code since the spread chain (24b7a2c..HEAD on pisco/, pt_mm_4b.sh, ft_mm_4b.sh): only off-by-default options (freeze_compressor, freeze_part, connector_lr — 184778's yaml already has `connector_lr: null`).
- Verifier items fixed: one-chunk "better" moved to provisional pending owner question 2; the ten-chunk wording explanation marked untested; the missing +.084 control in the 21:10 copy runs explained (no project-prompt copies of the reduced files).
- Efficiency lines: SPEED (nothing running 70% of 24 h) — no ready work was left unlaunched: every inference arm is read and arm E needs the owner. Usage limit (3.0 h) — the login is shared by every pisco loop; moving it is the owner's (`cc-team`), not asked here.

## Proposed replacement text (the owner's chats write these files; provisional until arm E or the owner closes it)
- `docs/architecture_review.md:252-256`, replace "the +.08 of the reduced chains is what the extra stage-2 data was buying" with: "At full budget the spread chain scores what it scored at reduced budget (+.011 / −.003 on documents); the tail models gained +.093 / +.048 from the larger budget (three stage-2 passes), most on MP-DocVQA. On today's code, full budget: photos equal (pooled +.001 [−.014, +.018]); 4-page documents +.017 [−.002, +.036], a 1-3 pt lean to spread that 1,500 questions × 2 seeds cannot resolve; adding pages (10, 20) does not widen it. Where text is compressed in long chunks the layout decides: ten HotpotQA passages as ONE chunk give tail .351 / .369 and spread .611 / .616 (+26 / +25 pt), MRQA single passages +8 / +10 pt, both seeds, under both judge prompts and EM. Spread slots stay tied to their own chunk at full budget (ownership .972 vs .763), so the tie is not a representation tie. Whether the layout changes the document plateau with one stage-2 pass each is arm E of campaign spread."
- HANDOFF.md thread 1: "spread vs tail: the reduced-budget gain was the tail models catching up with budget (three stage-2 passes), not spread losing it; spread wins by +25 pt on one long HotpotQA chunk and +8-10 pt on MRQA (both seeds, every scoring); on ten short chunks the judges tie and EM favours tail; photos tie; 4-page documents +1.7 pt, more pages do not widen it."

## Blockers / needs the owner
- BLOCKED.md question 1: arm E (recommendation A, ≈ 44 GPU-h, ≈ 60 if seed 42 shows the gain). Question 2: the arm J size rule (recommendation: count "better under every scoring" as better, 0 GPU-h).
