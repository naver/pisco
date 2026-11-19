# STATE — campaign longdoc   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-20 00:00 (twentieth LLM wake; woken because the uncompressed text cell 192266 ended).
NEXT: `wake-when job:192326 max 4h` (R s42 pages 10, the shortest of the four R cells; when it ends, pair it and submit 4 more).
Question: where do long inputs lose accuracy (eval, training data, length, `<MEM>` reading, positions, long-chunk
training)? Diagnosis only: inference probes and CPU audits, no training.

## This wake
- The full uncompressed arm E reference is in: 7 cells COMPLETED 0:0, one attempt each, nodes homer-011/012/014/016/017,
  code md5 b3d1cda1 / dfe32aa0 / 02bc8748 / f8ae6efe in every log. Health per cell: every row judged, 0 unparseable,
  no "exceeds"; answers at the 512 cap (tokenizer, padding off) text 47 / 2,246 = 2.1%, pages 8 / 14 / 14 of 1,407,
  mixed 15 / 26 / 31 of 1,407 (all < 5%). Verdicts closed (192266-192268, 192287-192290). Table:
  `expQ/longdoc/soundness/uncompressed_full_health.json`.
- Soundness check 1 decided on the full cells (table `expQ/longdoc/soundness/closedbook_full.json`): PASS in all 10 buckets.
- Launched 4 of cap 4: R s42 text 192323 (`--time 11:24`, DML 5232), pages p40 192324 (`--time 12:30`, DML 5612),
  mixed m40 192325 (`--time 12:30`, DML 15672), pages p10 192326 (`--time 7:18`, DML 5612). All RUNNING (homer-011 / 012
  / 014 / 014), argv length 4 in runs.jsonl, the same md5 as the uncompressed reference in their logs.
- Verifier point on the needle check: sourced as a fault of the check (law rule "a check that fails on almost every item
  is a fault of the check"). Whole-answer EM fails even at ~4k text (.52) where the code is found 100% and the judge
  gives 1.00. CAMPAIGN.md now says: a needle bucket passes only if BOTH "exact code in the answer" and the judge are
  ≥ 90%. Written after the data, so check 3 stays provisional.
- Cap 512 is now backed by full jobs (192266-192268, 192287-192290), not only the frontend recount of 300 trial rows.
- `docs/longdoc_diagnosis.md` rewritten (arm E soundness section, the cap row of the bottleneck table).
- Gripe +1 of ee961307 (wake-when waits for ALL jobs; no "first of").
- SPEED line of the brief (40% waiting for the owner): the cause was the arm E budget question, answered 2026-11-19
  17:05. No question is open now.

## TL;DR — where this leaves the question
**Settled**
- Arm E soundness check 1 PASS in every bucket: the uncompressed decoder with the document beats closed-book by ≥ 20 pt
  (bar), cap 512 both files, paired per question. Closed-book 191776 (1,407 Q, judge .164).
  - text (192266): ~4k .612 vs .203, +41.0 [+36.3, +45.5] (n 637); ~8k +41.3 [+36.4, +46.1] (610); ~16k +38.9
    [+34.2, +43.9] (588); ~32k .547 vs .219, +32.8 [+27.0, +38.9] (411).
  - pages (192287 / 192288 / 192267, n 1,407 each): 10 .672 +50.8 [+47.8, +53.8]; 20 .665 +50.0 [+47.1, +53.0];
    40 .659 +49.5 [+46.4, +52.5].
  - mixed (192289 / 192290 / 192268, n 1,407 each): 10 .654 +49.0 [+45.9, +52.0]; 20 .643 +47.8 [+44.8, +50.7];
    40 .631 +46.7 [+43.7, +49.7].
- The arm E answer cap is 512 tokens: uncompressed full cells at the cap text 2.1% (47 / 2,246, 192266), pages 0.6 / 1.0
  / 1.0%, mixed 1.1 / 1.8 / 2.2% (1,407 each); bar 5% (counted by the wake from the jobs' files, tokenizer, padding off).
- The transcription-probe port is validated (job 191169): 360 / 360 legacy pages byte-identical, word recall 70.5% =
  70.5% (gate ±2 pt).
- Arm E sets exist (job 191515): LongDocURL (Apache-2.0), 1,407 usable questions on 387 PDFs; pages and mixed 4,221 rows
  each (1,407 × 10 / 20 / 40 pages), text 2,246 rows (t4k 637 / t8k 610 / t16k 588 / t32k 411), needles text 300 /
  pages 225 / mixed 225. Closed-book set: 1,407 questions (job 191774). Bucket sets (job 192265): 6 × 1,407 rows.
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
- Arm E full grid (owner-approved 2026-11-19): 12 of 26 cells submitted. Uncompressed reference DONE on all 7 inputs.
  Running: R s42 text, pages 40, mixed 40, pages 10 (192323-192326). Left (14 cells): R s42 pages 20, mixed 10, mixed 20;
  R s43 × 7; C256 / L1024 at CML 1,024 on text, s42 and s43 (4 cells; symlinks `expQ/longdoc/t_{c256,l1024}_s4{2,3}/model`).
- provisional (plumbing, 300 rows, never decides): R s42 − uncompressed on LongDocURL. Pages cap 256 −17.0 [−23.0,
  −11.0] (24 / 75; −13 / −15 / −23 at 10 / 20 / 40 pages; 191673 vs 191669); text cap 256 −16.3 [−21.7, −11.0] (15 / 64;
  −16 / −19 / −12 / −19 at 4k-32k; 191674 vs 191670); mixed cap 512 −15.7 [−22.0, −9.3] (31 / 78; −12 / −14 / −21 at 10
  / 20 / 40 items; 191887 vs 191778).
- provisional, NOT DECIDED (32-token cap confound; the 512 re-run was not approved): arm L documents, Δ_len vs 4
  pages, 1,500 pairs: 10 pages +0.5 [−1.8, +2.7] (s42, 191378), +0.1 [−2.2, +2.5] (s43, 191390); 20 pages +4.3 [+1.8,
  +6.9] (s42, 191377), +3.1 [+0.5 to +0.6, +5.7] (s43, 192007; the lower bound is +0.5 in the verdict, +0.6 in
  cap32_sensitivity.json), questions with no capped answer −0.6 / −2.0 (n 1,221 / 1,220). "The gap shrinks at 20
  pages" is withdrawn: it is the cap. Text 40 vs 10 passages: −2.6 [−5.6, +0.3] (s42, 191392), +0.1 [−3.0, +3.1] (s43,
  192008); no capped answer −4.8 / −3.4 (n 731 / 733, a subset chosen on answers).

## Arm E — the long-document yardstick
Source: LongDocURL (`dengchao/LongDocURL`, Apache-2.0). Builder `scripts/longdoc/longdoc_build.py` (md5 f14e4640) +
`build_long.sbatch`. `<MEM>` slots p50 per bucket: pages 1,198 / 2,392 / 4,795; text 540 / 1,046 / 2,055 / 4,079; mixed
977 / 1,922 / 3,821 (max 14,688). Pisco prompt max (audit 191516): text 4,976, pages 5,356, mixed 15,416 → DML = max +
256 (5,232 / 5,612 / 15,672). Uncompressed prompt max: pages 39,562; text 36,965; mixed 117,451 (all ran on one GPU).
Measured uncompressed full walls (judge included): text 2:09; pages p10 1:22, p20 2:01, p40 3:22; mixed m10 1:27, m20
2:24, m40 3:00 (faster than the plan's estimate by ~25%). R walls are measured on 300 plumbing rows only: pages 31:05
(cap 256), text 30:22 (256), mixed 31:19 (512).
Submission plan (≤ 4 per wave, `--time` = whole-set wall × max(1/3, the bucket's prompt-token share 1/7, 2/7, 4/7) × 3;
R whole-set walls: pages 7.3 h, mixed 7.3 h, text 3.8 h; C256 / L1024 text 3.8 h). Once a first R bucket cell ends,
re-derive the R `--time` from it (× 3).
1. DONE: uncompressed text, p10 / p20 / p40, m10 / m20 / m40. RUNNING: R s42 text, p40, m40, p10.
2. Next wake: R s42 p20 (DML 5612, `--time 7:18` = 7.3 × 2/7 → 1/3 × 3), m10 and m20 (DML 15672, `--time 7:18`),
   R s43 text (DML 5232, `--time 11:24`, model `expQ/longdoc/e_r43/model`). Then R s43 per bucket (6 cells); then
   C256 / L1024 text at CML 1,024 (4 cells).
3. Each full cell: health (n rows, judge on all, unparseable 0, no "exceeds", answers at the cap < 5%, `mems_used`
   vs audit ±1%, node, model.py md5 f8ae6efe); gap R − U per bucket with `paired_judge.py`; Δ_len vs the shortest
   bucket with `length_effect.py`; the project-prompt re-judge on any number that decides.
Launch shape: `expmon launch --exp longdoc --stage E -p gpu-be --mem 8G --time … --exclude='homer-[005-009]'
--env MNT=512 --env PX=1003520 --env TOPK=1000 --env CML=1024 --env DML=<set> scripts/longdoc/eval_mnt.sbatch
expQ/longdoc/e_r4{2,3}/model pisco '' /beegfs/scratch/user/hdejean/pisco/eval_data/longdoc_<set>[_<b>]`.
Check `len(argv)` = 4 in `.expmon/runs.jsonl` after each launch (192323-192326: 4).
Pisco results land in `expQ/longdoc/e_r42/eval/` (names `mm_docqa_pisco_k1000_px1003520_cml1024_mnt512_longdoc_<set>.json`).

## Owned jobs
| job | arm | state | rows | note |
|---|---|---|---|---|
| 192323 | E R s42 text, cap 512 | RUNNING homer-011 | 2,246 | pairs with 192266 |
| 192324 / 192325 | E R s42 pages p40 / mixed m40 | RUNNING homer-012 / homer-014 | 1,407 each | pair with 192267 / 192268 |
| 192326 | E R s42 pages p10 | RUNNING homer-014 | 1,407 | pairs with 192287 |
| 192266-192268, 192287-192290 | E uncompressed reference | COMPLETED, verdict closed | 1,407-2,246 | health OK |
| 192265, 192142, 192007, 192008, 191877-191879, 191887, 191774, 191776-191778, 191669, 191670, 191673, 191674, 191629, 191634-191636 | E, J, L | COMPLETED, verdict closed | | |
| 191517 | J re-judge (first try) | FAILED, verdict closed "launch fault" | — | replaced by 191629 |
| 191515 / 191516 / 191392 / 191393 / 191375-191378 / 191390 / 191391 / 191305-191308 / 191218-191221 / 191169-191172 / 191148-191155 / 191165-191168 / 191145-191147 | E, L, T, A | COMPLETED, verdict closed | | |

Cells used: 69 of 87. GPU-h: 42.5 by `sacct -X` (last attempts only, to 2026-11-20 00:15); about 4.5 more in requeued
attempts (the 08:30 count). Left to run ≈ 52 GPU-h (R 18.4 per seed × 2 + C256 / L1024 15.2) → ≈ 99 of 114.
Code md5 for the arm E full grid: eval_mm_4b.sh b3d1cda1, eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py 02bc8748,
pisco/model.py f8ae6efe, eval_mnt.sbatch 1c874b32.

## Adopted jobs
(none — the other jobs in the queue are the expmon daemons 188764 / 162472)

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| E long-document yardstick | soundness checks 1 (decided) and 2 pass, 3 provisional; 12 / 26 cells submitted | the submission plan above |
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
- `scripts/longdoc/length_effect.py +A -B …` (judge default, `--metric em|match`, `--json`); tables in `expQ/longdoc/length/`.
- `scripts/longdoc/rejudge.sbatch NAME FILE…` (md5 9e2bc7d6): project-prompt judge on copies. Pass files as separate
  arguments (zsh does not word-split `$F`).
- Cap count (every file of a contrast, before reading a Δ_len): `.pixi/envs/default/bin/python -I`,
  `t = tokenizers.Tokenizer.from_file("expQ/ft_mm_4b_184778/model/tokenizer.json")`, `t.no_padding(); t.no_truncation()`,
  answer at the cap = `len(t.encode(pred).ids) >= cap - 2`.
- Cell health: `.pixi/envs/default/bin/python -I scripts/longdoc/cell_health.py [--cap 512] [--json OUT] FILE…` (rows,
  judged, unparseable, judge, answers at the cap by the rule above, prompt max, `mems_used` total, judge per bucket).
  Checked on 191887's file: 715,709 `<MEM>`s = the audit's prediction; at cap 4 / 300 = verdict 191887.
- The guard refuses any command containing `rm -r`, even on a `/tmp` scratch dir.
- `expmon display --exp longdoc` prints "0 table rows" but the page holds the grid: known gripe be8057a1.

## Blockers / needs the owner
(none)
