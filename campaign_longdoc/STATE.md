# STATE — campaign longdoc   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-20 04:30 (twenty-fourth LLM wake; woken because R s43 text, pages 10 and mixed 10 ended).
NEXT: `wake-when job:192362,192363,192364,192365,192366,192367,192374,192375,192376 max 5h` (every running cell; R s43
pages 20 / mixed 20 and the closed-book re-run end ~05:05, text C256 / L1024 s42 ~06:45, R s43 mixed 40 ~07:00, C256 /
L1024 s43 text ~07:15, R s43 pages 40 ~07:55; one wake then closes all nine and decides the page / mixed length question).
Question: where do long inputs lose accuracy (eval, training data, length, `<MEM>` reading, positions, long-chunk
training)? Diagnosis only: inference probes and CPU audits, no training.

## This wake
- R s43 text 192359 (homer-016, 2:44:37), pages 10 192360 (homer-011, 1:13:08), mixed 10 192361 (homer-014, 1:14:01):
  all COMPLETED 0:0, one attempt, code md5 b3d1cda1 / dfe32aa0 / 02bc8748 / f8ae6efe (unchanged), no "exceeds" in the
  logs. Health OK (`cell_health.py --cap 512`): every row judged, 0 unparseable, answers at the cap 3.6% / 1.1% / 1.8%,
  `<MEM>`s used 3,887,462 / 1,689,350 / 1,406,369 = the audit's predictions exactly. MaxRSS 4.3 / 5.7 / 4.3 GB.
  Verdicts closed; probes judge + gap recorded.
- Paired with U and with R s42 (`paired_judge.py`); text split per bucket (`expQ/longdoc/length/split/text_R43_*.json`)
  and Δ_len s43 tables `expQ/longdoc/length/E_dlen_text_{t8k,t16k,t32k}_s43_judge.json`.
- Launched 3 of cap 4, all RUNNING at 04:36, argv length 4 each: C256 s43 text 192374 (homer-011) and L1024 s43 text
  192375 (homer-013), `--time 7:54`, `--mem 8G`; closed-book re-run 192376 (homer-014, base, `TOPK=0`, `--env CML=1024`
  only renames the file: `gen_base` never receives `compressor_max_length`, `scripts/eval_mm_docqa.py:723`; new name
  `mm_docqa_base_k0_px1003520_cml1024_mnt512_longdoc_closedbook.json` checked absent; `--time 1:31` = 191776's 0:30 × 3,
  `--mem 4G` = its 2.1 GB × 1.5). The fourth slot (the project-prompt re-judge) waits for R s43 pages 40.
- `docs/longdoc_diagnosis.md`: arm E table gets the s43 column for text / pages 10 / mixed 10, reading rewritten.

## TL;DR — where this leaves the question
**Settled**
- The arm E answer cap is 512 tokens: uncompressed full cells at the cap text 2.1% (47 / 2,246, 192266), pages 0.6 / 1.0
  / 1.0%, mixed 1.1 / 1.8 / 2.2% (1,407 each); bar 5% (counted from the jobs' files, tokenizer, padding off). R s42 at
  the cap: pages 10 / 20 / 40 1.4 / 1.7 / 3.6% (192326 / 192356 / 192324), mixed 10 / 20 / 40 1.8 / 1.9 / 3.2%
  (192357 / 192358 / 192325), text 3.2% (192323).
- The transcription-probe port is validated (job 191169): 360 / 360 legacy pages byte-identical, word recall 70.5% =
  70.5% (gate ±2 pt).
- Arm E sets exist (job 191515): LongDocURL (Apache-2.0), 1,407 usable questions on 387 PDFs; pages and mixed 4,221 rows
  each (1,407 × 10 / 20 / 40 pages), text 2,246 rows (t4k 637 / t8k 610 / t16k 588 / t32k 411), needles text 300 /
  pages 225 / mixed 225. Closed-book set: 1,407 questions (job 191774). Bucket sets (job 192265): 6 × 1,407 rows.
- provisional (s43 in for text / pages 10 / mixed 10, the rest running): R − uncompressed on one long document, cap
  512, same code, paired per question (`paired_judge.py`), health OK on every file:
  - pages 10: R s42 .507 (192326) vs U .672 (192287), −16.6 [−19.5, −13.9], McNemar 120 / 353, n 1,407; R s43 .489
    (192360), −18.3 [−21.3, −15.4], 119 / 377; s43 − s42 −1.8 [−3.8, +0.2].
  - pages 20: R .471 (192356) vs U .665 (192288), −19.4 [−22.4, −16.4], 121 / 394, n 1,407.
  - pages 40: R .429 (192324) vs U .659 (192267), −23.0 [−26.1, −19.9], 123 / 447, n 1,407.
  - mixed 10: R s42 .449 (192357) vs U .654 (192289), −20.5 [−23.3, −17.6], 100 / 388, n 1,407; R s43 .459
    (192361), −19.5 [−22.3, −16.5], 110 / 384; s43 − s42 +1.0 [−0.9, +3.0].
  - mixed 20: R .435 (192358) vs U .643 (192290), −20.8 [−23.8, −17.8], 109 / 401, n 1,407.
  - mixed 40: R .412 (192325) vs U .631 (192268), −21.9 [−24.7, −18.9], 105 / 413, n 1,407.
  - text whole set: R s42 .415 (192323) vs U .594 (192266), −17.9 [−20.0, −15.8], 147 / 549, n 2,246; per bucket
    t4k .422 vs .612 −19.0 [−22.9, −15.1] (637), t8k .423 vs .615 −19.2 [−23.6, −14.9] (610), t16k .427 vs .587 −16.0
    [−20.1, −11.7] (588), t32k .377 vs .547 −17.0 [−22.6, −11.9] (411). R s43 .422 (192359), −17.3 [−19.4, −15.1],
    149 / 537; per bucket t4k .455 −15.7 [−19.6, −11.8], t8k .425 −19.0 [−23.1, −15.1], t16k .418 −16.8 [−21.3,
    −12.8], t32k .370 −17.8 [−23.4, −12.9]; s43 − s42 t4k +3.3 [+0.8, +5.8] (44 / 23), other buckets within ±1.
- provisional (NOT DECIDED: pages / mixed need s43; text meets no bar at either seed): Δ_len for R on arm E, each question paired
  with itself at the shortest size (`length_effect.py --strip_bucket`, judge), n 1,407 per pages / mixed contrast:
  - pages 20 vs 10 −2.8 [−5.6, −0.1], sign 157 / 193, p .06 (192356 / 192288 vs 192326 / 192287); pages 40 vs 10
    −6.5 [−9.5, −3.5], sign 165 / 249, p 4e-5 (192324 / 192267 vs the same): past the length-loss bar at s42 only.
    Scorers on 40 vs 10: match −4.6 [−7.1, −2.1]; EM +6.0 [+3.6, +8.3] (EM fails on the uncompressed answers, which
    grow longer at 40 pages: U EM .190 → .080, U judge .672 → .659). Project prompt not run yet.
  - mixed 20 vs 10 −0.3 [−3.1, +2.5], sign 174 / 176, p .96 (192358 / 192290 vs 192357 / 192289); mixed 40 vs 10
    −1.4 [−4.5, +1.6], 206 / 223, p .44 (192325 / 192268 vs the same).
  - text t8k / t16k / t32k vs t4k, s42: +0.7 [−3.1, +4.4] (n 610), +3.6 [−1.0, +8.0] (n 588), +3.9 [−1.7, +9.7]
    (n 411) (192323 vs 192266); s43: −2.5 [−6.2, +1.3] sign 57 / 69, −0.7 [−5.3, +3.9] 73 / 75, −1.7 [−7.5, +4.1]
    63 / 68 (192359 vs 192266). No length loss at either seed, but no CI inside ±3 either: n 411-610 per bucket is
    too few for the 3-pt bar, so "text: no length loss" cannot be decided on this set.
  Tables `expQ/longdoc/length/E_dlen_*_s4{2,3}_judge.json`.
- provisional (closed-book 191776 ran on model.py 994cff3d, the with-document cells on f8ae6efe; the change is in
  `PISCO.compress`, unused by base mode): arm E soundness check 1 PASS in every bucket: the uncompressed decoder with the
  document beats closed-book by ≥ 20 pt (bar), cap 512 both files, paired per question. Closed-book 191776 (1,407 Q,
  judge .164).
  - text (192266): ~4k .612 vs .203, +41.0 [+36.3, +45.5] (n 637); ~8k +41.3 [+36.4, +46.1] (610); ~16k +38.9
    [+34.2, +43.9] (588); ~32k .547 vs .219, +32.8 [+27.0, +38.9] (411).
  - pages (192287 / 192288 / 192267, n 1,407 each): 10 .672 +50.8 [+47.8, +53.8]; 20 .665 +50.0 [+47.1, +53.0];
    40 .659 +49.5 [+46.4, +52.5].
  - mixed (192289 / 192290 / 192268, n 1,407 each): 10 .654 +49.0 [+45.9, +52.0]; 20 .643 +47.8 [+44.8, +50.7];
    40 .631 +46.7 [+43.7, +49.7].
- provisional: training does not show the decoder long inputs (a count, not yet a cause). In a 2,000-row sample of
  finetune rows (job 191147) the maximum is 488 `<MEM>`s and a position span of 4,017 (a sample bound, not the
  population maximum). Beyond that range (191147): 10- and 20-page documents 1,500 / 1,500, 40 text passages
  1,498 / 1,500, 20 passages 35 / 1,500; 4-page documents 0 / 1,500. Arm E rows above 488 `<MEM>`s: text 2,246 / 2,246,
  pages 4,221 / 4,221, mixed 4,212 / 4,221 (`expQ/longdoc/audit_191516/out_of_range.json`, counted by a wake from the
  job's `rows.jsonl`, not by the job).
- provisional: arm E soundness check 3 (needle) passes in every bucket, scored "exact code string in the answer" AND
  judge, both ≥ 90% (uncompressed, cap 512, 25 documents × 3 depths per length): text 75 / 75 at ~4k / 8k / 16k / 32k
  tokens, judge 1.00 each (191877); pages 75 / 75, 73 / 75, 74 / 75 at 10 / 20 / 40 pages, judge 1.00 / .95 / .97
  (191878); mixed 75 / 75, 75 / 75, 74 / 75, judge 1.00 / 1.00 / .99 (191879). Whole-answer EM fails everywhere (text
  .52 / .36 / .43 / .15) because the base decoder answers in a sentence; the scoring rule was written after the data.
- provisional: arm E input check passes (job 191516, every row, CML 1,024): cut 0 and parity 1.0000 on text (2,246),
  pages (4,221), mixed (4,221) and the three needle sets; the CML 256 control cuts 235 / 2,246 (the cut check can
  fail). Parity has no failing control: one row builder serves both modes.
- provisional: the evaluation feeds the long inputs it claims on the gating sets (arm A, gate passed; job 191147): cut
  0 / 1,500 and parity 1,500 / 1,500 on every gating variant; health 646,042 = 646,042; CML 256 controls cut 924 / 1,500
  and 275 / 1,500. Parity has no failing control. The gating sets are not long documents (owner, 2026-11-17) → arm E.
- provisional: the shared 32-token answer cap distorts every uncompressed reference (tokenizer count, answer ≥ 30
  tokens, `expQ/longdoc/length/cap32_sensitivity.json`, frontend, no job):
  - LongDocURL, 300 plumbing rows per set: uncompressed pages .340 → .717 from cap 32 to 256, +37.7 [+32.0, +43.0], sign
    114 / 1 (191634 → 191669); text .423 → .580, +15.7 [+10.7, +20.7], 58 / 11 (191635 → 191670); R pages .517 → .547,
    +3.0 [+0.7, +5.3], 11 / 2 (191636 → 191673).
  - arm L files, 1,500 Q each, answers at the cap (a count, not a score shift): uncompressed documents 4 / 10 / 20 pages
    86 / 103 / 245 (5.7 / 6.9 / 16.3%; 191376 is the 20-page file), text 10 / 40 passages 494 / 703 (32.9 / 46.9%;
    191393 is the 40-passage file); R files 0.6-1.7% on documents, 2.7-5.9% on text.
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
    191221), −0.6 [−2.7, +1.3] 121 / 130 (s43; 191307 vs 191306): seeds disagree; answered by arm E's text cells.

**Open, and running**
- Arm E full grid (owner-approved 2026-11-19): 26 of 26 grid cells submitted; uncompressed reference DONE on all 7
  inputs; R s42 DONE on all 7; R s43 DONE on text / pages 10 / mixed 10.
  - Running R s43: pages 20 192362 and mixed 20 192363 (homer-016, started 03:04), pages 40 192364 (homer-012) and
    mixed 40 192365 (homer-014, started 04:05).
  - Running long-chunk text at CML 1,024: C256 / L1024 s42 192366 / 192367 (homer-016, started 04:05); C256 / L1024
    s43 192374 / 192375 (homer-011 / homer-013, started 04:36).
  - Running closed-book re-run on f8ae6efe 192376 (homer-014, started 04:36).
  - Left: ONE project-prompt re-judge job (`rejudge.sbatch`) on copies of every file of each Δ_len that decides (pages
    10 / 20 / 40 R and U, both seeds, if pages decides; mixed likewise), after R s43 pages 40.
- provisional (plumbing, 300 rows, never decides): R s42 − uncompressed on LongDocURL. Pages cap 256 −17.0 [−23.0,
  −11.0] (24 / 75; −13 / −15 / −23 at 10 / 20 / 40 pages; 191673 vs 191669); text cap 256 −16.3 [−21.7, −11.0] (15 / 64;
  −16 / −19 / −12 / −19 at 4k-32k; 191674 vs 191670); mixed cap 512 −15.7 [−22.0, −9.3] (31 / 78; −12 / −14 / −21 at 10
  / 20 / 40 items; 191887 vs 191778).
- provisional, NOT DECIDED (32-token cap confound; the 512 re-run was not approved; arm E answers the length question):
  arm L documents, Δ_len vs 4 pages, 1,500 pairs: 10 pages +0.5 [−1.8, +2.7] (s42, 191378), +0.1 [−2.2, +2.5] (s43,
  191390); 20 pages +4.3 [+1.8, +6.9] (s42, 191377), +3.1 [+0.5, +5.7] (s43, 192007), questions with no capped answer
  −0.6 / −2.0 (n 1,221 / 1,220). "The gap shrinks at 20 pages" is withdrawn: it is the cap. Text 40 vs 10 passages:
  −2.6 [−5.6, +0.3] (s42, 191392), +0.1 [−3.0, +3.1] (s43, 192008); no capped answer −4.8 / −3.4 (n 731 / 733, a subset
  chosen on answers).

## Arm E — the long-document yardstick
Source: LongDocURL (`dengchao/LongDocURL`, Apache-2.0). Builder `scripts/longdoc/longdoc_build.py` (md5 f14e4640) +
`build_long.sbatch`. `<MEM>` slots p50 per bucket: pages 1,198 / 2,392 / 4,795; text 540 / 1,046 / 2,055 / 4,079; mixed
977 / 1,922 / 3,821 (max 14,688). Predicted `<MEM>` totals per bucket (rows.jsonl of 191516): pages 1,689,350 /
3,378,666 / 6,757,206; mixed 1,406,369 / 2,815,326 / 5,635,667; text t4k 350,690, t8k 641,990, t16k 1,213,521, t32k
1,681,261 (whole set 3,887,462). Pisco prompt max (audit 191516): text 4,976, pages 5,356, mixed 15,416 → DML = max +
256 (5,232 / 5,612 / 15,672). Uncompressed prompt max: pages 39,562; text 36,965; mixed 117,451 (all ran on one GPU).
Measured walls (judge included): uncompressed text 2:09; pages p10 1:22, p20 2:01, p40 3:22; mixed m10 1:27, m20 2:24,
m40 3:00. R s42: pages p10 1:13, p20 1:56, p40 3:49; mixed m10 1:12, m20 1:45, m40 2:55; text 2:38.
Host RAM (MaxRSS): uncompressed p10 4.8 / p20 7.6 / p40 14.3 GB, m10 4.3 / m20 6.5 / m40 8.9 GB, text 2.4 GB; R s42
p10 5.6, p20 7.5, p40 13.7, m10 4.3, m20 5.5, m40 9.8, text 3.9 GB (`--mem` is not enforced on gpu-be).
Submission rule: `--time` = the measured R s42 wall of that bucket × 3 (pages 3:40 / 5:48 / 11:26, mixed 3:36 / 5:15 /
8:45, text 7:54). `--mem` = measured R MaxRSS × 1.5: pages 9G / 12G / 22G, mixed 8G / 9G / 15G, text 8G. C256 / L1024
text cells: as R text (7:54, 8G).
1. DONE: uncompressed on all 7 inputs; R s42 on all 7; R s43 text / pages 10 / mixed 10. RUNNING: R s43 pages / mixed
   20 and 40; C256 / L1024 text s42 and s43; closed-book re-run.
2. Next wake (all nine ended): health + verdict each. Pair R s43 pages 20 / 40 and mixed 20 / 40 with U and R s42; Δ_len
   s43 vs the 10 bucket (`length_effect.py --strip_bucket`, tables `E_dlen_{pages_p20,pages_p40,mixed_m20,mixed_m40}_s43`);
   decide pages and mixed length per the bars (both seeds). Long-chunk text: L1024 − C256 per seed, overall and per
   bucket (`paired_judge.py`), decide per the arm E long-chunk bars. Closed-book: soundness check 1 per bucket with
   `closedbook_pair.py` on the new file (replaces the 994cff3d one; check 1 loses its `provisional: md5` caveat if every
   margin stays ≥ 20 pt). Then submit ONE `rejudge.sbatch` job on copies of the files of every Δ_len that decides
   (files as separate arguments; check `len(argv)`), and the scorer check (EM, match) with `length_effect.py --metric`.
3. Each full cell: health (n rows, judge on all, unparseable 0, no "exceeds", answers at the cap < 5%, `mems_used`
   = the predicted total above ±1%, node, one attempt, model.py md5 f8ae6efe); gap R − U per bucket with
   `paired_judge.py`; Δ_len vs the shortest bucket with `length_effect.py --strip_bucket` (text: the per-bucket files
   in `expQ/longdoc/length/split/`); the project-prompt re-judge on any number that decides.
Launch shape (write each launch out in full; never `eval` it — zsh globs the bracket): `expmon launch --exp longdoc
--stage E -p gpu-be --mem <rule> --time <rule> --exclude='homer-[005-009]' --env MNT=512 --env PX=1003520 --env
TOPK=1000 --env CML=1024 --env DML=<set> scripts/longdoc/eval_mnt.sbatch expQ/longdoc/<cell>/model pisco ''
/beegfs/scratch/user/hdejean/pisco/eval_data/longdoc_<set>[_<b>]`. Check `len(argv)` = 4 in `.expmon/runs.jsonl` after
each launch (192360-192363: 4). Pisco results land in `expQ/longdoc/<cell>/eval/` (names
`mm_docqa_pisco_k1000_px1003520_cml1024_mnt512_longdoc_<set>.json`); logs `expQ/longdoc/eval_mnt_<jobid>.out`.

## Owned jobs
| job | arm | state | rows | note |
|---|---|---|---|---|
| 192364 / 192365 | E R s43 pages p40 / mixed m40 | RUNNING homer-012 / homer-014 | 1,407 each | pair with 192267 / 192268 and R s42 192324 / 192325 |
| 192362 / 192363 | E R s43 pages p20 / mixed m20 | RUNNING homer-016 | 1,407 each | pair with 192288 / 192290 and R s42 192356 / 192358 |
| 192366 / 192367 | E long-chunk text C256 / L1024 s42, CML 1,024 | RUNNING homer-016 | 2,246 each | pair with each other |
| 192374 / 192375 | E long-chunk text C256 / L1024 s43, CML 1,024 | RUNNING homer-011 / homer-013 | 2,246 each | pair with each other |
| 192376 | E closed-book re-run on f8ae6efe | RUNNING homer-014 | 1,407 | replaces 191776 in soundness check 1 |
| 192359 / 192360 / 192361 | E R s43 text / p10 / m10 | COMPLETED, verdict closed | 2,246 / 1,407 / 1,407 | health OK; gaps above |
| 192323 / 192326 / 192356 / 192324 / 192357 / 192358 / 192325 | E R s42 text / p10 / p20 / p40 / m10 / m20 / m40 | COMPLETED, verdict closed | 2,246 / 1,407 × 6 | health OK; gaps above |
| 192266-192268, 192287-192290 | E uncompressed reference | COMPLETED, verdict closed | 1,407-2,246 | health OK |
| 192265, 192142, 192007, 192008, 191877-191879, 191887, 191774, 191776-191778, 191669, 191670, 191673, 191674, 191629, 191634-191636 | E, J, L | COMPLETED, verdict closed | | |
| 191517 | J re-judge (first try) | FAILED, verdict closed "launch fault" | — | replaced by 191629 |
| 191515 / 191516 / 191392 / 191393 / 191375-191378 / 191390 / 191391 / 191305-191308 / 191218-191221 / 191169-191172 / 191148-191155 / 191165-191168 / 191145-191147 | E, L, T, A | COMPLETED, verdict closed | | |

Cells used: 84 of 87. GPU-h: 68.1 by `sacct -X` over the 84 campaign jobs (last attempts, running cells included, to
04:40) + about 4.5 in requeued attempts ≈ 73. Left ≈ 17 GPU-h in the running cells + 0.5 for the re-judge → ≈ 90 of
114. Cells left: 1 planned (the re-judge) = 85 of 87.
Code md5 for the arm E full grid: eval_mm_4b.sh b3d1cda1, eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py 02bc8748,
pisco/model.py f8ae6efe, eval_mnt.sbatch 1c874b32.

## Adopted jobs
(none — the other jobs in the queue are the expmon daemons 188764 / 162472)

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| E long-document yardstick | soundness checks 1 (provisional: md5, re-run 192376) and 2 pass, 3 provisional; 26 / 26 grid cells submitted; R gap −16 to −23 pt at every size, s43 = s42 within 2 pt where in; grows with length on pages only (40 vs 10 −6.5, s42); text length undecidable at ±3 (n per bucket) | step 2 of the arm E plan |
| L length on a fixed model | NOT DECIDED (cap confound); 512 re-run not approved | arm E answers the length question |
| J instrument | control 1 PASS (frontend); project prompt done for every arm L number | re-run the project prompt on any arm E number that decides |
| U reading many `<MEM>`s | not started | after E; only where a length loss shows |
| P positions | not started | after E; needs `--mem_position_offset` (patchable IF, only while no sibling eval runs) |
| A audit | DECIDED (provisional); E re-run PASS (191516) | — |
| T long-chunk training | CLOSED (provisional) | text half = arm E's C256 vs L1024 cells |

## Tooling and controls
- `scripts/longdoc/eval_mnt.sbatch` (md5 1c874b32): arm E cells. `--env MNT=<N≠32>` required; STYLE=expand refused;
  writes the derived launcher to `expQ/longdoc/launchers/eval_mm_4b_mnt.<jobid>.sh`; exits 3 if `eval_mm_4b.sh`
  changed shape. Base-mode results land in `expQ/base_qwen35_4b/eval/` (names `_mnt512_<set>`).
- `scripts/longdoc/split_buckets.py` (md5 3ef018be) / `.sbatch` (md5 d5430dc9): per-bucket sets.
- `scripts/longdoc/closedbook_pair.py CLOSEDBOOK DOC… [--json OUT]` (stdlib): soundness check 1 per bucket.
- `scripts/longdoc/length_effect.py +A -B …` (judge default, `--metric em|match`, `--json`, `--strip_bucket` to pair
  `<qid>__<bucket>` across buckets); tables in `expQ/longdoc/length/`.
- `scripts/paired_judge.py A B` (stdlib): gap per source/bucket with CI and McNemar counts.
- `scripts/longdoc/rejudge.sbatch NAME FILE…` (md5 9e2bc7d6): project-prompt judge on copies. Pass files as separate
  arguments (zsh does not word-split `$F`).
- Cell health: `.pixi/envs/default/bin/python -I scripts/longdoc/cell_health.py [--cap 512] [--json OUT] FILE…` (rows,
  judged, unparseable, judge, answers at the cap by the tokenizer rule, prompt max, `mems_used` total, judge per
  bucket). Compare `mems_used` with the predicted totals in the arm E section.
- The guard refuses any command containing `rm -r`, even on a `/tmp` scratch dir.
- `expmon display --exp longdoc` prints "0 table rows" but the page holds the grid: known gripe be8057a1.

## Blockers / needs the owner
(none)
