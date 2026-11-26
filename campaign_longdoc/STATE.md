# STATE — campaign longdoc   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-27 01:30 (full wake, woken by an ALERT; no job ran, nothing launched).
NEXT: `waiting-owner` — the final checkpoint is in BLOCKED.md (close now, or fund one repeat of arm U).
Question: where do long inputs lose accuracy (eval, training data, length, `<MEM>` reading, positions, long-chunk
training)? Diagnosis only: inference probes and CPU audits, no training.

## This wake
- ALERT line 1 (01:17) "expmon daemon not running": resolved by the tick — daemon 193847 RUNNING on homer-020 at 01:30.
- ALERT line 2 (01:30) "crontab has no 'campaign tick' line — the loop is not armed": a false alarm. The same tick's
  sensor line reads `cron=6`, and that tick launched this wake. Known expmon defect (gripe e8d30dc9: a failed
  crontab read counts as 0); +1 filed. The guard refused my read-only `crontab -l` (hard rule 3), so I did not check
  the crontab myself. ALERT file deleted. New row in known_issues.md.
- Verifier fixes: s43 40-page eighth CI lower bound 1.8 → 1.9 (verdict 193608); pages Δ_len string match s42 −4.5 →
  −4.6 (verdict 192405; file −.0455); "positions explain none of the 40-page loss" → "at most ~1.4 pt" (arm P CI).
- Time spent waiting for the owner (93% of the last 24 h, the tick's SPEED line): does not apply to the loop. The one
  question has one recommendation (close, 0 GPU-h), but the escalation check (2026-11-26 03:30) ruled it the owner's:
  it touches the run cap and the one-repeat rule. Nothing the loop can do shortens it.

## TL;DR — where this leaves the question
**Settled**
- The arm E answer cap is 512 tokens: uncompressed full cells at the cap text 2.1% (47 / 2,246, 192266), pages 0.6 / 1.0
  / 1.0%, mixed 1.1 / 1.8 / 2.2% (1,407 each); bar 5% (tokenizer count, padding off). R at the cap: s42 pages 1.4 / 1.7 /
  3.6%, mixed 1.8 / 1.9 / 3.2%, text 3.2%; s43 pages 1.1 / 2.3 / 3.7%, mixed 1.8 / 2.8 / 3.8%, text 3.6%.
- provisional: arm P DECIDED — positions are NOT a cause at both seeds, under both judge prompts. Each 10-page
  question's `<MEM>`s and question were moved to the position ids of its own 40-page version (offset p50 28,934, min
  27,990, max 29,670), slot count unchanged. Shifted − plain, n 1,407 pairs (`length_effect.py`, tables
  `expQ/longdoc/length/P_shift_p10_s4{2,3}_*.json`): rubric s42 −0.6 [−1.4, +0.3] McNemar 16 / 24 p .27 (193604 .501 vs
  192326 .507), s43 −0.1 [−0.8, +0.6] 11 / 13 p .84 (193605 .488 vs 192360 .489); project prompt (re-judge 193645 on
  copies, paired with 192405's copies) −0.4 [−1.4, +0.6] 24 / 29 p .58 / −0.1 [−0.8, +0.6] 13 / 14; match +0.2 / +0.1,
  EM −0.6 [−1.2, −0.1] / −0.4. Every CI inside ±3. The shift is applied: offsets = the map on 1,407 / 1,407 rows, 207 /
  194 answers change wording, `mems_used` 1,689,350 = the plain files'. Answers at the cap 1.63 / 1.28%.
  Prediction written at commit 622ae40 before any U / P cell: "P costs ≥ 3 pt at both seeds" — WRONG.
- provisional: arm U BETWEEN THE BARS (not confirmed, not ruled out). "Eighth" = keep the eighth of the slots the last
  prompt position attends to most over the whole document (`KEEPF=0.125 KEEPM=attn_global`; `mems_used` ratio .121 in
  every eighth file). D = (eighth − plain at 40 pages) − (eighth − plain at 10 pages), n 1,407 pairs
  (`length_effect.py --strip_bucket`, tables `expQ/longdoc/length/U_D_pages_s4{2,3}_{judge,match,em,project}.json`):
  - s42: 40 pages +2.5 [+0.8, +4.3], eighth-only / plain-only 105 / 70 (193536 .453 vs 192324 .429); 10 pages −0.4
    [−2.1, +1.4] 77 / 82 (193606 .503 vs 192326 .507); D +2.8 [+0.4, +5.3] sign 152 / 113 p .019.
  - s43: 40 pages +3.8 [+1.9, +5.9] 136 / 82 (193608 .461 vs 192364 .422); 10 pages +0.5 [−1.4, +2.3] 91 / 84 (193607
    .494 vs 192360 .489); D +3.3 [+0.8, +6.0] sign 173 / 135 p .035.
  - other scorers, D s42 / s43: project prompt +1.8 [−0.9, +4.5] p .24 / +3.8 [+0.9, +6.6] p .012 (193645 copies vs
    192405 copies); match +1.9 [−0.1, +3.9] / +1.9 [−0.1, +3.9]; EM +1.8 [−0.1, +3.6] / +1.9 [−0.1, +3.7]. Sign kept and
    within 2 pt of the rubric at each seed, so D is not judge-sensitive.
  - Gate (CAMPAIGN.md:144-146): cause = D ≥ +3 with the CI above 0 at both seeds → missed at s42 by 0.2 pt; not this
    cause = CI inside ±3 at both → no (upper bounds +5.3 / +6.0). By the "Promising, then ONE repeat" rule a repeat on
    new items comes next; it needs runs beyond 96 → the owner question in BLOCKED.md. My reading, not the law's letter:
    that rule is written for a gain below the noise floor; D is above the 1.2 pt band, so the law's "above it once
    still needs the repeat to be adopted" applies too. Both lead to the same repeat.
  - Prediction at 622ae40 "U's D lies inside ±3": WRONG at both seeds (CIs reach +5.3 / +6.0).
- provisional: the position probe's `MPOS=0` path reproduces the plain eval (job 193603): 300 rows of pages p10, R
  s42, model.py 2a1cc007, 0 of 300 predictions differ from 192326 (log `IDENTITY OK`). This covers the P path on
  p10 only; the U cells (pruning branch, `eval_mnt.sbatch`, 40 pages) pair with plain cells run on model.py f8ae6efe,
  bridged by the code diff (an opt-in pad guard, off by default), not by an identity run.
- The transcription-probe port is validated (job 191169): 360 / 360 legacy pages byte-identical, word recall 70.5% =
  70.5% (gate ±2 pt).
- Arm E sets exist (job 191515): LongDocURL (Apache-2.0), 1,407 usable questions on 387 PDFs; pages and mixed 4,221 rows
  each (1,407 × 10 / 20 / 40 pages), text 2,246 rows (t4k 637 / t8k 610 / t16k 588 / t32k 411), needles text 300 /
  pages 225 / mixed 225. Closed-book set: 1,407 questions (job 191774). Bucket sets (job 192265): 6 × 1,407 rows.
- provisional: R − uncompressed on one long document, cap 512, same code, paired per question (`paired_judge.py`),
  both seeds, n 1,407 per pages / mixed bucket (McNemar R-only / U-only):
  - pages 10: R s42 .507 (192326) vs U .672 (192287) −16.6 [−19.5, −13.9] 120 / 353; R s43 .489 (192360) −18.3 [−21.3,
    −15.4] 119 / 377. pages 20: s42 .471 (192356) vs U .665 (192288) −19.4 [−22.4, −16.4] 121 / 394; s43 .462 (192362)
    −20.3 [−23.2, −17.5] 109 / 394. pages 40: s42 .429 (192324) vs U .659 (192267) −23.0 [−26.1, −19.9] 123 / 447; s43
    .422 (192364) −23.7 [−26.8, −20.5] 114 / 447.
  - mixed 10: s42 .449 (192357) vs U .654 (192289) −20.5 [−23.3, −17.6] 100 / 388; s43 .459 (192361) −19.5 [−22.3,
    −16.5] 110 / 384. mixed 20: s42 .435 (192358) vs U .643 (192290) −20.8 [−23.8, −17.8] 109 / 401; s43 .436 (192363)
    −20.7 [−23.7, −17.7] 102 / 393. mixed 40: s42 .412 (192325) vs U .631 (192268) −21.9 [−24.7, −18.9] 105 / 413; s43
    .407 (192365) −22.5 [−25.4, −19.6] 110 / 426.
  - text whole set (n 2,246): s42 .415 (192323) vs U .594 (192266) −17.9 [−20.0, −15.8] 147 / 549; s43 .422 (192359)
    −17.3 [−19.4, −15.1] 149 / 537. Per bucket s42 / s43 gap: t4k −19.0 / −15.7 (637), t8k −19.2 / −19.0 (610), t16k
    −16.0 / −16.8 (588), t32k −17.0 / −17.8 (411).
  - R s43 − R s42: pages −1.8 / −0.9 / −0.6, mixed +1.0 / +0.1 / −0.6, text whole set +0.7: within 2 pt on every
    whole bucket; the text ~4k sub-bucket differs +3.3 [+0.8, +5.8] (192359 vs 192323).
- provisional: DECIDED for page images — a length loss at 40 pages. Δ_len 40 vs 10 pages (`length_effect.py
  --strip_bucket`, n 1,407, every file < 5% at the cap): rubric judge −6.5 [−9.5, −3.5] sign 165 / 249 p 4e-5 (s42;
  192324 / 192267 vs 192326 / 192287), −5.3 [−8.4, −2.3] 177 / 245 p .001 (s43; 192364 / 192267 vs 192360 / 192287);
  project prompt −5.7 [−8.7, −2.6] 178 / 257 / −6.1 [−9.2, −3.0] 183 / 271 (192405); match −4.6 [−7.1, −2.1] / −3.6
  [−6.0, −1.1]; EM +6.0 / +7.2 flips (uncompressed EM .190 → .080 while its judge .672 → .659: EM fails on the accepted
  reference). By arm J's letter the EM flip makes it "judge-sensitive"; sourced to EM's fault and stated in the final
  checkpoint, which the owner answered 2026-11-25 (option A, fund U / P). Tables `expQ/longdoc/length/E_dlen_pages_p40_s4{2,3}_{judge,match,em,project}.json`.
- provisional, NOT DECIDED (no bar at both seeds): pages 20 vs 10 −2.8 [−5.6, −0.1] (s42), −1.9 [−4.8, +0.9] sign 174 /
  197 (s43); mixed 20 vs 10 −0.3 [−3.1, +2.5] / −1.2 [−4.0, +1.6]; mixed 40 vs 10 −1.4 [−4.5, +1.6] / −3.0 [−5.9, 0.0]
  sign 186 / 235 p .02; text t8k / t16k / t32k vs t4k +0.7 / +3.6 / +3.9 (s42), −2.5 / −0.7 / −1.7 (s43), every CI
  wider than ±3 (n 610 / 588 / 411: too few questions for the 3-pt bar).
- provisional: long-chunk training on long text (arm E, `longdoc_text_1500`, CML 1,024 both, cap 512, n 2,246):
  L1024 − C256 +4.6 [+3.1, +6.2] McNemar 231 / 127 (s42; 192367 .412 vs 192366 .366; t4k +1.4 / t8k +3.3 / t16k +6.5 /
  t32k +9.0; the L1024 s42 file has 114 / 2,246 = 5.08% of answers at the cap, over the 5% bar: +4.9 on the questions
  neither file caps), −0.0 [−1.7, +1.7] 184 / 185 (s43; 192375 .411 vs 192374 .411; every bucket within ±1.3). Seeds
  disagree: not adopted. The s42 margin is one weak model: C256 s42 − R s42 −5.0 [−6.5, −3.3], L1024 s42 − R s42 −0.4
  [−2.0, +1.4]; at s43 both are within 1.1 pt of R. The CML 256 second round opens only on a tie at both seeds: not
  opened.
- provisional: arm E soundness check 1 PASS in every bucket, now on the same code: the closed-book re-run 192376
  (model.py f8ae6efe) scores .164 = 191776 with 0 discordant pairs, so the margins stand (`closedbook_pair.py`,
  `expQ/longdoc/length/closedbook_check1_f8ae6efe.json`, bar 20 pt): text (192266) ~4k +41.0 [+36.3, +45.5] (637), ~8k
  +41.3 (610), ~16k +38.9 (588), ~32k +32.8 [+27.0, +38.9] (411); pages 10 / 20 / 40 +50.8 / +50.0 / +49.5 (1,407 each);
  mixed 10 / 20 / 40 +49.0 / +47.8 / +46.7 (1,407 each).
- provisional: training does not show the decoder long inputs (a count, not yet a cause). In a 2,000-row sample of
  finetune rows (job 191147) the maximum is 488 `<MEM>`s and a position span of 4,017 (a sample bound). Beyond that
  range (191147): 10- and 20-page documents 1,500 / 1,500, 40 text passages 1,498 / 1,500, 20 passages 35 / 1,500. Arm E rows above 488 `<MEM>`s: text 2,246 / 2,246, pages 4,221 / 4,221, mixed 4,212 /
  4,221 (`expQ/longdoc/audit_191516/out_of_range.json`, counted by a wake from the job's `rows.jsonl`).
- provisional: arm E soundness check 3 (needle) passes in every bucket, scored "exact code string in the answer" AND
  judge, both ≥ 90% (uncompressed, cap 512, 25 documents × 3 depths per length): text 75 / 75 at ~4k / 8k / 16k / 32k,
  judge 1.00 each (191877); pages 75 / 75, 73 / 75, 74 / 75 at 10 / 20 / 40 pages, judge 1.00 / .95 / .97 (191878);
  mixed 75 / 75, 75 / 75, 74 / 75, judge 1.00 / 1.00 / .99 (191879). Whole-answer EM fails everywhere (text .53 / .36 /
  .43 / .15); the scoring rule was written after the data.
- provisional: arm E input check passes (job 191516, every row, CML 1,024): cut 0 and parity 1.0000 on text (2,246),
  pages (4,221), mixed (4,221) and the three needle sets; the CML 256 control cuts 235 / 2,246. Parity has no failing
  control: one row builder serves both modes.
- provisional: the evaluation feeds the long inputs it claims on the gating sets (arm A, job 191147): cut 0 / 1,500 and
  parity 1,500 / 1,500 on every gating variant; health 646,042 = 646,042; CML 256 controls cut 924 / 1,500 and 275 /
  1,500. The gating sets are not long documents (owner, 2026-11-17) → arm E.
- provisional: the shared 32-token answer cap distorts the uncompressed references (handed off, 8d45d0): 300 plumbing
  rows, uncompressed pages .340 → .717 from cap 32 to 256, +37.7 [+32.0, +43.0], sign 114 / 1 (191634 → 191669); text
  .423 → .580, +15.7 [+10.7, +20.7], 58 / 11 (191635 → 191670); R pages .517 → .547, +3.0 [+0.7, +5.3], 11 / 2 (191636 →
  191673). Arm L files at the cap (a count): uncompressed documents 4 / 10 / 20 pages 5.7 / 6.9 / 16.3%, text 10 / 40
  passages 32.9 / 46.9% (`expQ/longdoc/length/cap32_sensitivity.json`).
- provisional: arm J — the length-effect calculator recovers −.1687 [−.1960, −.1407] (target −.169 ± .005), self-pair
  0 (frontend stdlib, no job; EXPERIMENTS.md:1116).
- provisional: arm T CLOSED — the long-chunk (L1024) loss on documents sits IN THE COMPRESSED IMAGE PAGE (gate met at
  both seeds; one stage-2 model per chain).
  - Gold page only L1024 − C256 −6.7 [−8.9, −4.5] McNemar 106 / 207 (s42; 191153 .568 vs 191152 .635), −4.9 [−7.1,
    −2.5] 113 / 186 (s43; 191155 .579 vs 191154 .628); picked page raw +0.1 [−0.8, +1.1] (s42; 191166 vs 191165),
    −0.1 [−1.1, +0.9] (s43; 191168 vs 191167). 1,500 pairs per seed, 1,003,520 px.
  - Page transcription, stage-2 models (191219 / 191220 vs 191170): .566 vs .647 of the page's words, −8.0 pt [−10.6,
    −5.5], sign 147 / 322, p 4e-16, 500 pages.
  - Text autoencoding at 256 tokens, d = L1024 − C256: stage 1 −.0018 [−.0031, −.0006], 271 / 229, p .067 (191218);
    stage 2 +.0006 [−.0008, +.0019], 247 / 253, p .82 (191308): consistent with an image-specific loss (rests on a null).
  - Text QA (`text_hotpotqa_1500`, CML 4,096, cap 32): L1024 − C256 +2.3 [+0.3, +4.4] 136 / 101 (s42; 191305 vs
    191221), −0.6 [−2.7, +1.3] 121 / 130 (s43; 191307 vs 191306): the same seed pattern as arm E's long text.

**Open, and running**
- Nothing running. Every funded cell is in: 96 of 96 runs (`.expmon/runs.jsonl`, exp longdoc), 98.3 of 114 GPU-h by
  sacct at 03:05 (96 jobs, elapsed × allocated GPUs).
- Final checkpoint written to BLOCKED.md: close now (recommended, 0 GPU-h) or fund one repeat of arm U on new
  documents (~15 runs, ~17 GPU-h, MMLongBench-Doc, licence unchecked, under-powered at ~700-1,000 questions).
- The 40-page loss to explain: Δ_len −6.5 (s42) / −5.3 (s43); R's own 10 → 40 page drop .507 → .429 = −7.8 (s42,
  192326 → 192324), .489 → .422 = −6.7 (s43, 192360 → 192364). Positions explain at most ~1.4 pt of it (arm P CI, page images at 10 pages only). The eighth
  recovers +2.5 / +3.8 of R's drop at 40 pages and nothing at 10 pages.
- Health for every U / P cell: md5 eval_mm_4b.sh b3d1cda1, eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py 02bc8748,
  pisco/model.py 2a1cc007. 193608: 1,407 / 1,407 judged, unparseable 0, at cap 28 (1.99%), `mems_used` 815,874 = .121 of
  6,757,206 (= its s42 twin), homer-014, COMPLETED 0:0 3:13 (`expQ/longdoc/length/health_U43p40_2026-11-26.json`).
  193645: 6 × 1,407 rows, unparseable 0 in each, COMPLETED 0:0 5:16.
- provisional (plumbing, 300 rows, never decides): R s42 − uncompressed on LongDocURL. Pages cap 256 −17.0 [−23.0,
  −11.0] (24 / 75; 191673 vs 191669); text cap 256 −16.3 [−21.7, −11.0] (15 / 64; 191674 vs 191670); mixed cap 512
  −15.7 [−22.0, −9.3] (31 / 78; 191887 vs 191778).
- provisional, NOT DECIDED (32-token cap confound; the 512 re-run was not approved; arm E answered the length question):
  arm L documents, Δ_len vs 4 pages, 1,500 pairs: 10 pages +0.5 [−1.8, +2.7] (s42, 191378), +0.1 [−2.2, +2.5] (s43,
  191390); 20 pages +4.3 [+1.8, +6.9] (s42, 191377), +3.1 [+0.5, +5.7] (s43, 192007), questions with no capped answer
  −0.6 / −2.0 (n 1,221 / 1,220). Text 40 vs 10 passages: −2.6 [−5.6, +0.3] (s42, 191392), +0.1 [−3.0, +3.1] (s43,
  192008); no capped answer −4.8 / −3.4 (n 731 / 733, a subset chosen on answers).


## Arm E — the long-document yardstick
Source: LongDocURL (`dengchao/LongDocURL`, Apache-2.0). Builder `scripts/longdoc/longdoc_build.py` (md5 f14e4640) +
`build_long.sbatch`. `<MEM>` slots p50 per bucket: pages 1,198 / 2,392 / 4,795; text 540 / 1,046 / 2,055 / 4,079; mixed
977 / 1,922 / 3,821 (max 14,688). Pisco prompt max (audit 191516): text 4,976, pages 5,356, mixed 15,416 → DML 5,232 /
5,612 / 15,672. Measured walls (judge included): uncompressed text 2:09; pages p10 1:22, p20 2:01, p40 3:22; mixed m10
1:27, m20 2:24, m40 3:00. R: pages p10 1:13, p20 1:56-1:58, p40 3:49-3:53; mixed m10 1:12-1:14, m20 1:45-1:52, m40
2:55-2:58; text 2:38-2:48 (C256 / L1024 2:35-3:02). Project-prompt re-judge: 6 files × 1,407 rows in 5:58.
Host RAM (MaxRSS): R p10 5.6-5.7, p20 4.1-7.5, p40 5.1-13.7, m10 4.3, m20 4.1-5.5, m40 4.5-9.8, text 3.3-11.2 GB
(`--mem` is not enforced on gpu-be).
Submission rule (if more cells are approved): `--time` = the measured R wall of that bucket × 3 (pages 3:40 / 5:48 /
11:26, mixed 3:36 / 5:15 / 8:45, text 7:54); `--mem` = measured MaxRSS × 1.5 (pages 9G / 12G / 22G, mixed 8G / 9G /
15G, text 8G); re-judge `--time 0:18`, `--mem 3G`.
Launch shape (write each launch out in full; never `eval` it — zsh globs the bracket): `expmon launch --exp longdoc
--stage E -p gpu-be --mem <rule> --time <rule> --exclude='homer-[005-009]' --env MNT=512 --env PX=1003520 --env
TOPK=1000 --env CML=1024 --env DML=<set> scripts/longdoc/eval_mnt.sbatch expQ/longdoc/<cell>/model pisco ''
/beegfs/scratch/user/hdejean/pisco/eval_data/longdoc_<set>[_<b>]`. Check `len(argv)` = 4 in `.expmon/runs.jsonl`.
Results land in `expQ/longdoc/<cell>/eval/` (names `mm_docqa_pisco_k1000_px1003520_cml1024_mnt512_longdoc_<set>.json`);
logs `expQ/longdoc/eval_mnt_<jobid>.out`.

## Owned jobs
| job | arm | state | rows | note |
|---|---|---|---|---|
| 193645 | J project-prompt re-judge of the 6 U / P files | COMPLETED, verdict closed "health OK" | 6 × 1,407 | copies in `expQ/longdoc/rejudge/UP_pages_p10_p40/` |
| 193608 | U R s43 pages p40 eighth | COMPLETED, verdict closed "health OK" | 1,407 | +3.8 [+1.9, +5.9] vs 192364 |
| 193604 / 193605 | P R s42 / s43 pages p10 shifted | COMPLETED, verdict closed "health OK" | 1,407 each | −0.6 / −0.1 |
| 193606 / 193607 | U R s42 / s43 pages p10 eighth | COMPLETED, verdict closed "health OK" | 1,407 each | −0.4 / +0.5 |
| 193536 | U R s42 pages p40 eighth | COMPLETED, verdict closed "health OK" | 1,407 | +2.5 [+0.8, +4.3] |
| 193603 | P identity check, R s42 pages p10, `MPOS=0` | COMPLETED, verdict closed "PASS" | 300 | 0 of 300 differ from 192326 |
| 193533 | P identity check (first try) | TIMEOUT, verdict closed "launch fault" | 930 of 1,407 run, no output | row limit passed as env, not arg 3 |
| 193534 / 193535 | P shifted (first try) | CANCELLED, verdict closed | — | DependencyNeverSatisfied |
| 192405 | J project-prompt re-judge, pages 10 / 40 | COMPLETED, verdict closed | 6 × 1,407 | copies in `expQ/longdoc/rejudge/E_pages_p10_p40/` |
| 192362 / 192363 / 192364 / 192365 | E R s43 pages 20 / mixed 20 / pages 40 / mixed 40 | COMPLETED, verdict closed | 1,407 each | health OK |
| 192366 / 192367 / 192374 / 192375 | E long-chunk text C256 / L1024 s42, s43 | COMPLETED, verdict closed | 2,246 each | health OK (L1024 s42 cap 5.08%) |
| 192376 | E closed-book re-run on f8ae6efe | COMPLETED, verdict closed | 1,407 | = 191776 |
| 192359 / 192360 / 192361, 192323 / 192326 / 192356 / 192324 / 192357 / 192358 / 192325 | E R s43 / s42 | COMPLETED, verdict closed | | health OK |
| 192266-192268, 192287-192290 | E uncompressed reference | COMPLETED, verdict closed | 1,407-2,246 | health OK |
| 192265, 192142, 192007, 192008, 191877-191879, 191887, 191774, 191776-191778, 191669, 191670, 191673, 191674, 191629, 191634-191636 | E, J, L | COMPLETED, verdict closed | | |
| 191517 | J re-judge (first try) | FAILED, verdict closed "launch fault" | — | replaced by 191629 |
| 191515 / 191516 / 191392 / 191393 / 191375-191378 / 191390 / 191391 / 191305-191308 / 191218-191221 / 191169-191172 / 191148-191155 / 191165-191168 / 191145-191147 | E, L, T, A | COMPLETED, verdict closed | | |

Cells used: 96 of 96 (`.expmon/runs.jsonl`, exp longdoc). Code md5 for the arm E grid: eval_mm_4b.sh b3d1cda1,
eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py 02bc8748, pisco/model.py f8ae6efe, eval_mnt.sbatch 1c874b32; U / P cells
the same but model.py 2a1cc007.

## Adopted jobs
(none — the other jobs in the queue are the expmon daemons 188764 / 162472)

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| E long-document yardstick | DONE (provisional): gap −16 to −24 pt at every size, seeds within 2 pt; length loss on page images at 40 pages (both seeds, judge / project / match); no bar elsewhere; long-chunk text: seeds disagree | — |
| U reading many `<MEM>`s | BETWEEN THE BARS (provisional): D +2.8 / +3.3, positive under every scorer, s42 below +3 | owner: close, or fund one repeat on new items (BLOCKED.md) |
| P positions | DECIDED (provisional): not this cause, −0.6 / −0.1 (project −0.4 / −0.1) | — |
| L length on a fixed model | NOT DECIDED (cap confound); 512 re-run not approved | answered by arm E |
| J instrument | control 1 PASS; project prompt run on every number that decides | — |
| A audit | DECIDED (provisional); E re-run PASS (191516) | — |
| T long-chunk training | CLOSED (provisional); text half answered by arm E (seeds disagree) | — |

## Tooling and controls
- `scripts/longdoc/eval_mnt.sbatch` (md5 1c874b32): arm E cells. `--env MNT=<N≠32>` required; STYLE=expand refused;
  writes the derived launcher to `expQ/longdoc/launchers/eval_mm_4b_mnt.<jobid>.sh`; exits 3 if `eval_mm_4b.sh`
  changed shape. Base-mode results land in `expQ/base_qwen35_4b/eval/` (names `_mnt512_<set>`).
- `scripts/longdoc/split_buckets.py` / `.sbatch`: per-bucket sets. `scripts/longdoc/closedbook_pair.py CLOSEDBOOK DOC…`
  (stdlib): soundness check 1 per bucket.
- `scripts/longdoc/length_effect.py +A -B …` (judge default, `--metric em|match`, `--json`, `--strip_bucket`); tables
  in `expQ/longdoc/length/`. `scripts/paired_judge.py A B` (stdlib): gap per source/bucket with CI and McNemar.
- `scripts/longdoc/rejudge.sbatch NAME FILE…` (md5 9e2bc7d6): project-prompt judge on copies (files as separate
  arguments). `scripts/longdoc/cell_health.py [--cap 512] [--json OUT] FILE…` (pixi python `-I`): health per file.
- `scripts/longdoc/plot_length.py [OUT.svg]` (stdlib): the arm E length figure.
- The guard refuses any command containing `rm -r`, even on a `/tmp` scratch dir.
- `expmon display --exp longdoc` prints "0 table rows" but the page holds the grid: known gripe be8057a1.
- Row limit = positional arg 3 of every eval launcher (`… pisco 300 DATA`), never `--env LIMIT`.
- `scripts/longdoc/eval_mpos.py` (md5 d2abdd65, tracked): arm P. Imports `scripts/eval_mm_docqa.py` unchanged and raises the
  decoder position ids of the first `<MEM>` and of everything after it by env `MPOS` (a constant, or a {query_id: N}
  json); `MPOS_REF=<recorded json>` makes the job exit 4 unless every prediction equals the reference's. Writes
  `<output>.mpos.json` (offset per row). Launcher `scripts/longdoc/eval_mpos.sbatch` (md5 e970ba7f) = eval_mnt.sbatch
  plus that script swap; output names `…_mnt512_mpos<tag>_…`; exits 3 if `eval_mm_4b.sh` changed shape (sed checked on
  the frontend: exactly 3 changed lines).
- `scripts/longdoc/mpos_map.py ROWS VARIANT SHORT LONG OUT` (stdlib): per-question offsets from the audit rows.
  `expQ/longdoc/mpos/pages_p40_minus_p10.json` (md5 ca3c56de): 1,407 questions, N min 27,990 / p50 28,934 / max 29,670.
- `--dependency` is not recorded in `.expmon/runs.jsonl` (gripe 04e7f373): a chain lives only in SLURM.


## Blockers / needs the owner
- BLOCKED.md, final checkpoint (one question): close now (recommended: 0 runs, 0 GPU-h) or fund one repeat of arm U on
  new documents (~15 runs over the cap, ~17 GPU-h, ~2 over 114). If "close": write RETRO.md Lessons, the retro gripe,
  then `expmon campaign standby`. If "repeat": check the MMLongBench-Doc licence first; a licence that forbids research
  use stops it.
