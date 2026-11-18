# STATE — campaign longdoc   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-19 08:30 (seventeenth LLM wake; woken because arm L seed-43 cells 192007 / 192008 ended).
NEXT: waiting-owner. BLOCKED.md holds ONE owner question (arm E model-cell budget, rewritten this wake: +54 GPU-h,
+17 cells; new option 2 adds the arm L re-run at the 512-token cap).
Question: where do long inputs lose accuracy (eval, training data, length, `<MEM>` reading, positions, long-chunk
training)? Diagnosis only: inference probes and CPU audits, no training.

## This wake
- 192007 (R s43 documents 20 pages) healthy: COMPLETED 0:0; preempted 4 times, generation + judge by attempt 4 on
  homer-017 (06:37-08:09, file 08:09), attempt 5 reused it. 192008 (R s43 text 40 passages) healthy: one attempt,
  homer-018, 46 min. Both 1,500 rows, judge on all, unparseable 0, no "exceeds" error, `mems_used` = audit = s42
  (3,229,884 / 1,075,610). Verdicts closed, probes set.
- Launched (1 of cap 4): 192142, the project-prompt re-judge (arm J rule) of the two s43 files + the borrowed s43
  10-passage file. COMPLETED 0:0 in 2:21 on homer-014, unparseable 0; verdict closed.
- Verifier fix: it flagged that arm L ran at the default 32-token answer cap. Counted (frontend, tokenizer rule of arm
  E, no job; `expQ/longdoc/length/cap32_sensitivity.json`): EVERY arm L uncompressed file fails the 5% bar, worst at
  the long sizes. Detail in TL;DR. "The gap shrinks at 20 pages" withdrawn as a cap artefact.
- Law tightened (CAMPAIGN.md Δ_len gate): a Δ_len decides only if all four files have < 5% of answers at the cap; arm
  L decides only from cells re-run at `MNT=512`. known_issues row added.
- `docs/longdoc_diagnosis.md` rewritten (bottleneck rows for the cap, arm J, arm L; both length tables with s43, the
  at-cap share and the no-cap read). BLOCKED.md rewritten with the new numbers and budget (32.0 GPU-h used by sacct
  over all 57 jobs and attempts).
- Verifier's other notes, answered here: the 7 / 300 cap recount (191777) is a frontend tokenizer count with no job
  (verdict texts still say 8 / 300, superseded by the ledger note of 02:07); the needle checks score "code string in
  the answer", not EM (the base decoder answers in a sentence); the borrowed 10-passage files (spread 191518 / 191630)
  ran the same models (184778 / 184780) and `git diff 5923159 32f9a78` shows no committed change to the eval, judge,
  model or collator files (both trees were dirty: identity of uncommitted code not proven).

## TL;DR — where this leaves the question
**Settled**
- provisional: training does not show the decoder long inputs (a count, not yet a cause). In a 2,000-row sample of
  finetune rows (job 191147) the maximum is 488 `<MEM>`s and a position span of 4,017 (a sample bound). Beyond that
  range (191147): 10- and 20-page documents 1,500 / 1,500, 40 text passages 1,498 / 1,500, 20 passages 35 / 1,500;
  4-page documents 0 / 1,500. Arm E rows above 488 `<MEM>`s (191516): text 2,246 / 2,246, pages 4,221 / 4,221, mixed
  4,212 / 4,221.
- The transcription-probe port is validated (job 191169): 360 / 360 legacy pages byte-identical, word recall 70.5% =
  70.5% (gate ±2 pt).
- Arm E sets exist (job 191515): LongDocURL (Apache-2.0), 1,407 usable questions on 387 PDFs; pages and mixed 4,221 rows
  each (1,407 × 10 / 20 / 40 pages), text 2,246 rows (t4k 637 / t8k 610 / t16k 588 / t32k 411), needles text 300 /
  pages 225 / mixed 225. Closed-book set: 1,407 questions (job 191774).
- provisional: the arm E answer cap is 512 tokens: uncompressed text answers at the cap 7 / 300 (2.3%, bar 5%;
  frontend tokenizer recount of 191777's file, padding off, no job; its verdict says 8 / 300 from the padded count);
  mixed 7 / 300 (191778), closed-book 5 / 1,407 (191776), R mixed 4 / 300 (191887). By characters 191777 gives 6.3%.
- provisional: arm E soundness check 3 (needle) PASS in every bucket, uncompressed, cap 512, 25 documents × 3 depths
  per length, scored as "code string in the answer": text 75 / 75 at ~4k / 8k / 16k / 32k tokens (191877); pages
  75 / 75, 73 / 75, 74 / 75 at 10 / 20 / 40 pages (191878); mixed 75 / 75, 75 / 75, 74 / 75 (191879); bar 90%.
- provisional: arm E input check PASS (job 191516, every row, CML 1,024): cut 0 and parity 1.0000 on text (2,246),
  pages (4,221), mixed (4,221) and the three needle sets; the CML 256 control cuts 235 / 2,246. Parity cannot fail by
  construction (one row builder for both modes).
- provisional: the evaluation feeds the long inputs it claims on the gating sets (arm A, gate passed; job 191147): cut
  0 / 1,500 and parity 1,500 / 1,500 on every gating variant; health 646,042 = 646,042; CML 256 controls cut 924 / 1,500
  and 275 / 1,500. The gating sets are not long documents (owner, 2026-11-17) → arm E.
- provisional: the shared 32-token answer cap distorts every uncompressed reference (counted with the tokenizer, answer
  ≥ 30 tokens, `expQ/longdoc/length/cap32_sensitivity.json`, frontend, no job):
  - LongDocURL, 300 rows per set: uncompressed pages .340 → .717 from cap 32 to 256, +37.7 [+32.0, +43.0], sign 114 / 1
    (191634 → 191669); text .423 → .580, +15.7 [+10.7, +20.7], 58 / 11 (191635 → 191670); R pages .517 → .547, +3.0
    [+0.7, +5.3], 11 / 2 (191636 → 191673).
  - arm L files, 1,500 Q each: uncompressed documents 4 / 10 / 20 pages 86 / 103 / 245 answers at the cap (5.7 / 6.9 /
    16.3%; 191376 is the 20-page file), text 10 / 40 passages 494 / 703 (32.9 / 46.9%; 191393 is the 40-passage file);
    R files 0.6-1.7% on documents, 2.7-5.9% on text (191377, 191378, 191390, 191392, 192007, 192008).
- provisional (first read, trial rows; the full uncompressed cells decide): arm E soundness check 1 — the document is
  needed. Closed-book judge .164 on 1,407 questions (191776, cap 512). With the document − closed-book: text +33 / +33
  / +33 / +32 (75 Q each, 191777; CI lower bounds .173-.200, at or under the 20-pt bar); mixed +44 / +47 / +48 (100 Q
  each, 191778). Pages +52 / +52 / +54 is indicative only (its with-document file ran at cap 256, 191669).
- provisional (plumbing, 75 Q per bucket, no paired test between buckets): no drop visible for the uncompressed reader
  with length on LongDocURL text at cap 512 (191777): .587 / .587 / .587 / .573; mixed .63 / .66 / .67 (191778).
- provisional: arm J — the length-effect calculator recovers −.1687 [−.1960, −.1407] (target −.169 ± .005), self-pair
  0 (frontend, no job; EXPERIMENTS.md:1116). Second judge prompt (191629, 192142): see the arm L bullets.
- provisional: arm T CLOSED — the long-chunk (L1024) loss on documents sits IN THE COMPRESSED IMAGE PAGE (gate met at
  both seeds; one stage-2 model per chain).
  - Gold page only L1024 − C256 −6.7 [−8.9, −4.5] McNemar 106 / 207 (s42; 191153 .568 vs 191152 .635), −4.9 [−7.1,
    −2.5] 113 / 186 (s43; 191155 .579 vs 191154 .628); picked page raw +0.1 [−0.8, +1.1] (s42; 191166 vs 191165),
    −0.1 [−1.1, +0.9] (s43; 191168 vs 191167). 1,500 pairs per seed, 1,003,520 px.
  - Page transcription, stage-2 models (191219 / 191220 vs 191170): .566 vs .647 of the page's words, −8.0 pt [−10.6,
    −5.5], sign 147 / 322, p 4e-16, 500 pages.
  - Text autoencoding at 256 tokens, d = L1024 − C256: stage 1 −.0018 [−.0031, −.0006], 271 / 229, p .067 (191218);
    stage 2 +.0006 [−.0008, +.0019], 247 / 253, p .82 (191308): consistent with an image-specific loss.
  - Text QA (`text_hotpotqa_1500`, CML 4,096): L1024 − C256 +2.3 [+0.3, +4.4] 136 / 101 (s42; 191305 vs 191221), −0.6
    [−2.7, +1.3] 121 / 130 (s43; 191307 vs 191306): seeds disagree; moves to arm E.
  - Proposed to the owner (untested): keep the 256-token stage 1; if retried, vary stage 2 from the L1024 start and
    check page transcription before any finetune.

**Open, and running** (nothing running; waiting on the owner)
- provisional, NOT DECIDED (32-token cap confound, law rule of 08:30): arm L documents, Δ_len vs 4 pages, 1,500 pairs:
  - 10 pages: +0.5 [−1.8, +2.7] p .77 (s42, 191378), +0.1 [−2.2, +2.5] p .86 (s43, 191390); questions with no capped
    answer −0.1 / −0.4 (n 1,357 / 1,354).
  - 20 pages: +4.3 [+1.8, +6.9] 197 / 146 p .007 (s42, 191377), +3.1 [+0.6, +5.7] 195 / 153 p .028 (s43, 192007);
    project prompt +5.1 / +3.1 [+0.5, +5.6] (191629 / 192142); EM +7.7 / +6.1. R .599 / .591 vs .610 / .615 at 4
    pages; uncompressed .754 vs .809. No capped answer: −0.6 [−3.3, +2.0] (n 1,221) / −2.0 [−4.5, +0.4] (n 1,220).
    "The gap shrinks at 20 pages" is withdrawn: it is the cap.
- provisional, NOT DECIDED (same confound): arm L text 40 vs 10 passages, 1,500 pairs: −2.6 [−5.6, +0.3] p .084 (s42,
  191392), +0.1 [−3.0, +3.1] 220 / 224 p .89 (s43, 192008); project −1.5 / +0.9 [−2.1, +4.1]; EM +0.8 / +2.7. R .534 /
  .549 at 40 vs .605 / .593 at 10 (borrowed: spread 191518 / 191630); uncompressed .707 vs .752. No capped answer: −4.8
  [−8.3, −1.5] (n 731) / −3.4 [−6.8, 0.0] (n 733): a length loss on text is plausible, the subset is chosen on answers.
- provisional (plumbing, 300 rows, never decides): R s42 − uncompressed on LongDocURL. Pages cap 256 −17.0 [−23.0,
  −11.0] (24 / 75; −13 / −15 / −23 at 10 / 20 / 40 pages; 191673 vs 191669); text cap 256 −16.3 [−21.7, −11.0] (15 / 64;
  −16 / −19 / −12 / −19 at 4k-32k; 191674 vs 191670); mixed cap 512 −15.7 [−22.0, −9.3] (31 / 78; −12 / −14 / −21 at 10
  / 20 / 40 items; 191887 vs 191778).
- Waiting on the owner: BLOCKED.md (arm E budget; option 2 adds the arm L re-run at 512).

## Arm E — the long-document yardstick
Source: LongDocURL (`dengchao/LongDocURL`, Apache-2.0). Builder `scripts/longdoc/longdoc_build.py` (md5 f14e4640) +
`build_long.sbatch`. Rows are interleaved by question, so `LIMIT=300` = 100 whole questions at every page size, or 75 at
every text size. `<MEM>` slots p50 per bucket: pages 1,198 / 2,392 / 4,795; text 540 / 1,046 / 2,055 / 4,079; mixed
977 / 1,922 / 3,821 (max 14,688). Uncompressed prompt max: pages 39,562; text 36,965; mixed 117,451 (measured up to
89,311 on one GPU, 191778). Measured walls per 300 judged rows: uncompressed text 23:26 (cap 512), mixed 43:47 (512),
pages 31:01 (256); R pages 31:05 (256), text 30:22 (256), mixed 31:19 (512). MaxRSS ≤ 4.4 GB everywhere.
Next, in order, once the owner answers BLOCKED.md:
1. Option 1 or 2: one CPU job writes per-bucket sets `eval_data/longdoc_{pages,mixed}_1500_{p10,p20,p40}` (same rows,
   split by the `__` suffix of the query id); then the uncompressed cells (pages 3 bucket cells, text 1, mixed 3), then
   R s42, R s43 (symlink `expQ/longdoc/e_r43/model → expQ/ft_mm_4b_184780/model` first), then C256 / L1024 at CML 1,024
   on text. ≤ 4 submissions per wake, chained with `--dependency afterok` where it saves a wake.
2. Each full cell: health (now including the share of answers at the cap, < 5%); soundness check 1 per bucket with
   `closedbook_pair.py` (same cap 512); gap and Δ_len per bucket with `length_effect.py`; the project-prompt re-judge on
   any number that decides.
3. Option 2 only: arm L re-run at `MNT=512` through `eval_mnt.sbatch` into NEW cell dirs (e.g. `expQ/longdoc/l512_r42`),
   uncompressed under new file names: documents 4 / 20 pages, text 10 / 40 passages, uncompressed + R s42 + R s43.

## Owned jobs
| job | arm | state | rows | note |
|---|---|---|---|---|
| 192142 | J, project re-judge of the s43 length files | COMPLETED, verdict closed | 3 files × 1,500 | copies in `expQ/longdoc/rejudge/dlen_docs_text/` |
| 192007 / 192008 | L, R s43 documents 20 pages / text 40 passages | COMPLETED, verdict closed | 1,500 each | `expQ/longdoc/l_r43/eval/` |
| 191877 / 191878 / 191879 / 191887 / 191776 / 191777 / 191778 / 191774 / 191669 / 191670 / 191673 / 191674 / 191629 / 191634-191636 | E, J | COMPLETED, verdict closed | | |
| 191517 | J re-judge (first try) | FAILED, verdict closed "launch fault" | — | replaced by 191629 |
| 191515 / 191516 / 191392 / 191393 / 191375-191378 / 191390 / 191391 / 191305-191308 / 191218-191221 / 191169-191172 / 191148-191155 / 191165-191168 / 191145-191147 | E, L, T, A | COMPLETED, verdict closed | | |

Cells used: 57 of 70. GPU-h: 32.0 of 60 (sacct over all 57 jobs, every attempt, GPUs × elapsed). CPU jobs count 0.
Code md5 at launch of 192007 / 192008: eval_mm_4b.sh b3d1cda1, eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py 02bc8748,
judge_prompts.py bae44fa7, pisco/model.py 994cff3d. The s42 arm L cells ran eval_mm_4b.sh 50abc167; b3d1cda1 adds only
the requeue resume.

## Adopted jobs
(none — the other jobs in the queue are the expmon daemons 188764 / 162472 and campaign slotcheck's 192003)

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| E long-document yardstick | sets built; soundness checks 2 and 3 PASS, check 1 passes on trial rows; cap 512 | owner answer (BLOCKED.md), then the Arm E section |
| L length on a fixed model | NOT DECIDED: every uncompressed file fails the cap bar | re-run at `MNT=512` only if the owner picks option 2; otherwise arm E answers the length question |
| J instrument | control 1 PASS (frontend); second prompt done for every arm L number (191629, 192142) | re-run the project prompt on any arm E number that decides |
| U reading many `<MEM>`s | not started | after E (owner); only where a length loss shows |
| P positions | not started | after E (owner); needs `--mem_position_offset` (patchable IF, only while no sibling eval runs) |
| A audit | DECIDED (provisional); E re-run PASS (191516) | — |
| T long-chunk training | CLOSED (provisional) | text half in arm E's grid (C256 vs L1024 at CML 1,024) |

## Tooling and controls
- `scripts/longdoc/eval_mnt.sbatch` (md5 1c874b32): arm E cells (and any arm L re-run). `--env MNT=<N≠32>` required;
  STYLE=expand refused; writes the derived launcher to `expQ/longdoc/launchers/eval_mm_4b_mnt.<jobid>.sh`; exits 3 if
  `eval_mm_4b.sh` changed shape.
- `scripts/longdoc/closedbook_pair.py CLOSEDBOOK DOC… [--json OUT]` (stdlib): soundness check 1 per bucket.
- `scripts/longdoc/length_effect.py +A -B …` (judge default, `--metric em|match`, `--json`); tables in `expQ/longdoc/length/`.
- `scripts/longdoc/rejudge.sbatch NAME FILE…` (md5 9e2bc7d6): project-prompt judge on copies; requeue-safe; skips a copy
  already re-judged. Pass files as separate arguments. 3 files took 2:21.
- Cap count (do it for EVERY file of a contrast before reading a Δ_len): `.pixi/envs/default/bin/python -I`,
  `t = tokenizers.Tokenizer.from_file("expQ/ft_mm_4b_184778/model/tokenizer.json")`, `t.no_padding(); t.no_truncation()`,
  answer at the cap = `len(t.encode(pred).ids) >= cap - 2`. Example output: `expQ/longdoc/length/cap32_sensitivity.json`.
- `audit_inputs.py --only E_`; per-row output `expQ/longdoc/audit_191516/rows.jsonl`.
- Arm E launch shape: `expmon launch --exp longdoc --stage E --time … --mem … --exclude='homer-[005-009]' -p gpu-be
  --env MNT=512 --env PX=1003520 --env TOPK=1000 --env CML=1024 [--env DML=<pisco prompt max + 256>]
  scripts/longdoc/eval_mnt.sbatch expQ/longdoc/e_r4{2,3}/model pisco|base [LIMIT|''] /beegfs/scratch/user/hdejean/pisco/eval_data/longdoc_<set>`.
  DML per set: pages 5,612; text 5,232; mixed 15,672. Check `len(argv)` in `.expmon/runs.jsonl` after each launch.
- The guard refuses any command containing `rm -r`, even on a `/tmp` scratch dir.
- `expmon display --exp longdoc` prints "0 table rows" but the page holds the grid (58 rows): known gripe be8057a1.

## Blockers / needs the owner
- BLOCKED.md: approve ≈ +54 GPU-hours (60 → 114) and +17 cells (70 → 87) for arm E's model cells (options 1-4;
  recommendation option 1; option 2 also re-runs arm L at the 512 cap).
