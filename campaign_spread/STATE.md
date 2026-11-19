# STATE — campaign spread   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-19 18:00 (seventh, closing wake). **The campaign is CLOSED by the owner** (REQUEST 2026-11-19 17:05,
executed this wake): no tail-control training (arm E, option C), and "better under every scoring" counts as better
(arm J size rule, option (a)). Nothing is left to launch or decide; this wake wrote the final deliverable, the retro
and STAND-DOWN. Deliverable: `docs/spread_cases.md` (final). Tables: audit `expQ/spread/audit_191454/audit.md`;
seed-pooled test `expQ/spread/pooled_20261118_1730/pooled.md`; arm J per scoring
`expQ/spread/rejudge/pooled_20261118_2100/pooled{,_judge_rubric,_em,_anls}.md`; arm C growth test
`expQ/spread/did_20261118_2100/did{,_em}.md`; arm D `expQ/spread/mech/*_191785/probe_{mixer,connector}.json`.

## Owner REQUEST executed this wake (2026-11-19 17:05)
- Question 1 (arm E): option C — no arm E; close with the case table; spread stays the base layout; GPU time goes to
  the stage-2 rebuild. Note: BLOCKED.md had recommended option A; the relayed answer names option C explicitly, so it
  was executed as written. Launchers `scripts/spread/pt_mm_tail5m.sbatch` / `ft_tail5m.sbatch` stay unrun.
- Question 2 (arm J size rule): option (a) — a case is "better" when every scoring (rubric judge, project prompt, EM)
  gives "better"; the size range is reported next to it. Applied to HotpotQA one chunk and MRQA (verdicts 191528,
  191780 re-recorded).
- REQUEST and BLOCKED.md deleted (both questions answered).

## TL;DR — where this leaves the question
**Settled**
- Arm A feasibility gate PASS (job 191454, CPU, closed): the 8 recorded spread − tail judge differences re-derive exactly; 39 self-pairs 0.000; 184778's document file 1,500 rows, judge on all, 0 unparseable, mems_used 646,042.
- Seed-pooled "equal" test controls PASS (17:45 on the rubric files, `pooled_20261118_1730/pooled.md:3`): reduced-budget documents +.084 [+.065, +.103] (expected +.084), every self-pair 0.000. The 21:10 runs on the arm J copies carry self-pairs only (no project-prompt copy of the reduced files exists); re-pairing the copies' rubric verdicts reproduced pooled.md exactly.
- **HotpotQA ten passages as ONE chunk: spread BETTER** (owner rule (a)), n 1,500 per seed: rubric +.260 [+.233, +.289], 488 / 98 (s42, 191457 − 191456) / +.247 [+.219, +.275], 477 / 106 (s43, 191528 − 191526); project prompt +.271 / +.253, pooled +.262 [+.236, +.288]; EM +.151 [+.127, +.175] / +.152 [+.126, +.177] (job 191665). Size range +.15 (EM) to +.27 (project prompt).
- Tail production loses going from ten HotpotQA chunks to one at both seeds: −.253 [−.281, −.225] (s42, 191456 − 191455), −.229 [−.255, −.200] (s43, 191526 − 191519). Spread does not: +.006 [−.019, +.029] (s42, 191457 − 191518), +.023 [+.000, +.045], 178 / 144, p .066 (s43, 191528 − 191630).
- **MRQA one long passage: spread BETTER** (owner rule (a)), today's code both sides, 53 slots per question, slot counts equal 80,003, uncompressed .807: spread 184778 / 184780 .715 / .724 (qafreeze 191675 / 191676, read-only) vs tail 174908 / 176242 .631 / .626 (191779 / 191780) → rubric +.084 [+.060, +.109], 239 / 113 (s42) / +.098 [+.075, +.122], 249 / 102 (s43); project prompt +.070 [+.046, +.095] / +.103 [+.079, +.128] (job 192006); EM +.057 [+.034, +.081] / +.074 [+.051, +.096]. Size range +.057 (EM) to +.103 (project prompt).
- TEN HotpotQA chunks (today's code, n 1,500 per seed) are **judge-sensitive** (arm J, job 191665): rubric +.001 [−.021, +.025] / −.004 [−.027, +.019] (191518 − 191455, 191630 − 191519), pooled −.001 [−.0203, +.018]; project prompt pooled −.003 [−.022, +.016]; EM −.027 [−.047, −.007] (p .010) / −.020 [−.039, +.000] (p .058), pooled −.023 [−.039, −.007]; ANLS pooled −.030 [−.045, −.014]. Spread's answers are longer (3.43 vs 3.01 words s42, 3.91 vs 3.35 s43) with 42 / 24 more judged-right-but-EM-0 answers.
- Full-budget PHOTOS (stage G), n 1,500: rubric −.007 [−.027, +.011], 105 / 116 (s42, 191531 − 191631); +.010 [−.011, +.030], 125 / 110 (s43, 191532 − 191632); pooled +.001 [−.014, +.018] → **equal** under the rubric judge; EM pooled −.003 [−.017, +.012] equal too. Under the project prompt alone "equal" is not certified (s42 −.023 [−.043, −.004], pooled −.006 [−.022, +.010]).
- Full-budget DOCUMENTS, 4 pages (431 slots per question), n 1,500: rubric +.011 [−.014, +.035], 177 / 161 (s42, 184778 vs 174908, both on 24b7a2c; 184778 re-run 191495 identical) / +.024 [−.002, +.048], 193 / 157 (s43, 191496 − 191633); pooled +.017 [−.002, +.036], width .038 → **not resolvable here**. Same sign and within 2 pt under every scoring (project pooled +.024 [+.006, +.042]; EM +.001 / +.022; ANLS pooled +.029 [+.012, +.046]). Uncompressed .809.
- **Arm C closed (many pages): gate not met — the layout does not pay more as pages are added.** Growth (spread − tail at N pages) − (at 4 pages), rubric, n 1,500: 10 pages −.004 [−.027, +.019] (s42, longdoc 191378 vs 191666) / −.007 [−.031, +.015] (s43, longdoc 191390 vs 191667); 20 pages s42 +.009 [−.015, +.031] (longdoc 191377 vs 191668). Control (arm B one-vs-ten-chunk growth) recovered +.259 / +.251, self-pairs 0.000. The 20-page s43 replicate was not run (the gate already fails at s42 and at both 10-page seeds).
- Re-runs on today's code (rerun_20261118.md): documents 184778, 184780, 176242 0 / 1,500 answers differ; photos 184780 and 176242 0 / 1,500; photos 184778 223 / 1,500 (68 judge flips); photos 174908 135 / 1,500 (34 flips); 184780 ten chunks 0 / 1,500. `PX` is inert on photos (79,322 slots at both budgets).
- Arm E (tail control, one stage-2 pass) **not run**: owner option C, 2026-11-19. Arm F (fixes): none proposed.

**Open (closed without resolution)**
- provisional (arm D, job 191785, mechanism only, INCOMPLETE against the law): on the four stage-2 models, control PASS (the reduced pair reproduces ownership .977 / .799 and answer loss pages 1.138 / 1.334, text 1.812 / 1.927, photos 1.765 / 1.802); full-budget pair `pt_mm_5M_spread` vs `pt_mm_v4kd_4b`: text ownership .972 vs .763 (+.208; reduced +.178; 3,053 slots); answer loss (nats, 40 rows, unpaired means) pages 1.102 vs 1.226, text 1.844 vs 1.788, photos 1.736 vs 1.813. Not run: the four finetunes, and the page transcription with its control (within ±2 pt of the legacy 70.5%). Reading: the full-budget accuracy tie is not a tie of what the stage-2 slots hold.
- provisional: no disjoint document question set can be built in the gating sources: VisRAG slides 556 / infographics 718 / MP-DocVQA 591 questions (`scripts/build_mm_eval.py:5`), the gating set uses 500 each → 56 + 218 + 91 = 365 left; no source alone reaches the 300-question floor, pooled they are 60% infographics; `mm_docqa_retr10_1m`, `retr4_1m`, `longdoc_1m` reuse the gating question ids.
- provisional (untested): why the reduced-budget document gain (+.093 / +.075) vanished at full budget. Consistent with the tail models catching up through budget: spread full − reduced documents +.011 / −.003, tail production − reduced +.093 / +.048 (job 191454; NOT one variable — three stage-2 passes vs one).
- provisional: the tail models' budget gain sits mostly on MP-DocVQA +.162 / +.090, slides +.076 / +.040, infographics +.040 / +.014 (n 500 per source, job 191454); no between-source test.
- provisional: spread better on infographics at full budget, 4 pages: +.050 [+.004, +.096] 80 / 55 (s42) / +.050 [+.008, +.096] 76 / 51 (s43), n 500 each (audit.md:198 / :230); post-hoc pick among 3 sources; a repeat could use only 218 new questions (< 300 floor), not run.
- provisional (untested): the ten-chunk EM gap is answer wording, not reading (longer spread answers, more judged-right-EM-0).
- provisional: MRQA by passage length (slice chosen after the data, no slice file): > 100 slots (DuoRC, TextbookQA, n 343) +.187 [+.131, +.245] / +.230 [+.169, +.289]; ≤ 30 slots (n 429) +.082 / +.065; 31-60 (n 568) +.007 / +.048; 61-100 n 160 not resolvable.

## Owned jobs (jobid | arm | state | rows | eta | note)
| 191454 | A audit | COMPLETED | 23 pairs | — | closed: PASS |
| 191455-191457, 191518, 191519, 191526, 191528, 191630 | B (one / ten chunks × 2 layouts × 2 seeds) | COMPLETED | 1,500 each | — | closed healthy; one chunk better (191528 verdict) |
| 191631 / 191632 / 191633 | G tail photos s42 / s43, documents s43 | COMPLETED | 1,500 each | — | closed healthy |
| 191665 | J re-judge, 16 gating copies | COMPLETED | 16 × 1,500 | — | closed healthy |
| 191666 / 191667 / 191668 | C tail 10 pages s42 / s43, 20 pages s42 | COMPLETED | 1,500 each | — | closed healthy |
| 191779 / 191780 | MRQA tail 174908 / 176242 | COMPLETED | 1,500 each | — | closed healthy; better (191780 verdict) |
| 191785 | D mechanism probes, 4 stage-2 models | COMPLETED | 40 rows × 3 sets × 4 | — | closed, control PASS |
| 192006 | J re-judge, project prompt, 4 MRQA copies | COMPLETED | 4 × 1,500 | — | closed healthy |

## Adopted jobs
(none) — the queue held only the daemon 188764 and campaign slotcheck's 192237-192239 (not ours).

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| A. case audit | decided: PASS (191454) | — |
| B. long chunks | **closed: one chunk better** under every scoring | — |
| G. full-budget gating pairs | decided: photos equal; documents not resolvable | — |
| J. instrument | decided (191665, 192006); size rule (a) applied | — |
| C. many pages | closed: gate not met | — |
| MRQA one long passage | **closed: better** under every scoring | — |
| D. what slots hold | closed incomplete (stage-2 models only; no finetunes, no transcription) | — |
| E. tail control, one stage-2 pass | **not run** (owner option C) | — |
| F. fixes | none | — |

Budget used: 20 cells (1 CPU + 19 GPU), GPU-h 8.6 (sacct over the 20 runs) of 95. LLM wakes 7 of ≤ 40 (two more died
on the account's usage limit).

## Health evidence this wake
- No job ran or ended for this campaign since the 02:00 wake (brief: new result files none; grid unchanged).
- `expmon display --exp spread` refreshed: `display_spread.html`, 21 `<tr>` rows (the summary line says 0 rows: gripe 43db294f +1).
- Verifier items fixed: MRQA and one-chunk now settled by the owner's rule (a); arm D moved to Open as incomplete (no finetunes, no transcription control); the "no disjoint questions" bullet restated (365 pooled, no source ≥ 300); the "every inference arm is read" claim dropped; the 20-page s43 replicate named as not run.

## Proposed replacement text (the owner's chats write these files; hand-off sent to the pisco chat)
- `docs/architecture_review.md:252-256`, replace "the +.08 of the reduced chains is what the extra stage-2 data was buying" with: "At full budget the spread chain scores what it scored at reduced budget (+.011 / −.003 on documents); the tail models gained +.093 / +.048 from the larger budget (three stage-2 passes), most on MP-DocVQA. On today's code, full budget: photos equal (pooled +.001 [−.014, +.018]); 4-page documents +.017 [−.002, +.036], a 1-3 pt lean to spread that 1,500 questions × 2 seeds cannot resolve; adding pages (10, 20) does not widen it. Where text is compressed in long chunks the layout decides: ten HotpotQA passages as ONE chunk give tail .351 / .369 and spread .611 / .616 (+25 to +27 pt under the judges, +15 by EM), MRQA single passages +7 to +10 pt, both seeds, better under every scoring. Spread slots stay tied to their own chunk at full budget (ownership .972 vs .763), so the tie is not a representation tie. Whether the layout changes the document plateau with one stage-2 pass each was not tested (campaign spread closed 2026-11-19; `docs/spread_cases.md`)."
- HANDOFF.md thread 1: "spread vs tail (campaign spread, closed 2026-11-19): spread is better on long text in long chunks (+25 pt on one HotpotQA chunk, +7-10 pt on MRQA, both seeds, every scoring); on ten short chunks the judges tie and EM favours tail; photos equal; 4-page documents +1.7 pt, not resolvable, more pages do not widen it. The vanished reduced-budget document gain is consistent with tail catching up through budget (three stage-2 passes), untested. Spread stays the base layout."

## Blockers / needs the owner
None. Closed: retro in RETRO.md, STAND-DOWN written; `expmon campaign disarm` and pasting this dir's SYNC.md into
`~/.claude/SYNC.md` are the owner's.
