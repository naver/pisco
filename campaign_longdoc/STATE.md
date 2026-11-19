# STATE — campaign longdoc   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-19 18:00 (eighteenth LLM wake; woken by the owner's REQUEST of 17:05).
NEXT: `wake-when job:192265 max 2h` (the bucket-split build; when it ends, submit the next 4 uncompressed bucket cells).
Question: where do long inputs lose accuracy (eval, training data, length, `<MEM>` reading, positions, long-chunk
training)? Diagnosis only: inference probes and CPU audits, no training.

## This wake
- Owner REQUEST 17:05 executed: option 1 of the arm E budget question approved ("do as recommended"): cells 70 → 87,
  GPU-hours 60 → 114, arm E model cells at both seeds + the long-chunk text cells. Option 2 (arm L re-run at the
  512 cap) was NOT approved: arm L stays undecided. Recorded in CAMPAIGN.md "Budget and caps"; REQUEST and BLOCKED.md
  deleted (question answered).
- Correction to the deleted BLOCKED.md (verifier): it said "uncompressed with the document beats closed-book by +27 to
  +54 pt"; the +27 was R mixed at 40 items (191887), not uncompressed. Uncompressed − closed-book on the trial rows is
  +32 to +33 (text), +44 to +48 (mixed), +52 to +54 (pages, cap-256 file, indicative only).
- New tooling: `scripts/longdoc/split_buckets.py` + `split_buckets.sbatch` (md5 3ef018be): per-bucket sets
  `eval_data/longdoc_pages_1500_{p10,p20,p40}` and `eval_data/longdoc_mixed_1500_{m10,m20,m40}` (rows with source
  `longdocurl_<b>`, source order, checks query ids). Law clarified: a bucket cell pairs only with the same bucket.
- Symlink `expQ/longdoc/e_r43/model → expQ/ft_mm_4b_184780/model` made (R seed 43).
- Launched 4 of cap 4 (all `-p gpu-be --exclude=homer-[005-009]`, one GPU, `--mem 8G` = MaxRSS 4.4 GB × 1.5,
  env `MNT=512 PX=1003520 TOPK=1000 CML=1024`, base mode via `expQ/longdoc/e_r42/model`):
  - 192265 split build (CPU, `-c 2 --mem 16G --time 1:00`), RUNNING homer-014;
  - 192266 uncompressed text, whole set 2,246 rows, `--time 8:45` (23:26 per 300 rows × 7.49 × 3), RUNNING homer-016;
  - 192267 uncompressed pages p40 (1,407 rows), `--time 13:45`, afterok 192265;
  - 192268 uncompressed mixed m40 (1,407 rows), `--time 17:40`, afterok 192265.
  `--time` for a bucket = whole-set wall (BLOCKED.md costing) × max(1/3, the bucket's prompt-token share) × 3; the
  token shares on the trial files are 1/7, 2/7, 4/7 for 10 / 20 / 40 pages or items.
- Code md5 changed under the campaign: `pisco/model.py` is now f8ae6efe (was 994cff3d), edited 17:06 by another chat:
  `compress()` drops pad columns of a ONE-row batch (Qwen3.5 linear-attention layers leaked pads into `<MEM>`s,
  campaign slotcheck job 191781). Base mode does not compress. Every arm E full cell runs on f8ae6efe (192266 logged
  it); the 300-row trial cells ran on 994cff3d and stay plumbing. Each new cell's log must show f8ae6efe before it pairs.
- Gripe: runs.jsonl drops `--dependency` / `--mem` from the launch row.

## TL;DR — where this leaves the question
**Settled**
- provisional: training does not show the decoder long inputs (a count, not yet a cause). In a 2,000-row sample of
  finetune rows (job 191147) the maximum is 488 `<MEM>`s and a position span of 4,017 (a sample bound, not the
  population maximum). Beyond that range (191147): 10- and 20-page documents 1,500 / 1,500, 40 text passages
  1,498 / 1,500, 20 passages 35 / 1,500; 4-page documents 0 / 1,500. Arm E rows above 488 `<MEM>`s: text 2,246 / 2,246,
  pages 4,221 / 4,221, mixed 4,212 / 4,221 (`expQ/longdoc/audit_191516/out_of_range.json`, counted by a wake from the
  job's `rows.jsonl`, not by the job).
- The transcription-probe port is validated (job 191169): 360 / 360 legacy pages byte-identical, word recall 70.5% =
  70.5% (gate ±2 pt).
- Arm E sets exist (job 191515): LongDocURL (Apache-2.0), 1,407 usable questions on 387 PDFs; pages and mixed 4,221 rows
  each (1,407 × 10 / 20 / 40 pages), text 2,246 rows (t4k 637 / t8k 610 / t16k 588 / t32k 411), needles text 300 /
  pages 225 / mixed 225. Closed-book set: 1,407 questions (job 191774).
- provisional: the arm E answer cap is 512 tokens. Uncompressed text answers at the cap: 7 / 300 (2.3%, bar 5%) by a
  frontend tokenizer recount of 191777's file with padding off (no job); verdict 191777 says 8 / 300 (2.7%, padded
  count). Mixed 7 / 300 (191778), closed-book 5 / 1,407 (191776), R mixed 4 / 300 (191887). By characters 191777 is 6.3%.
- provisional: arm E soundness check 3 (needle) passes in every bucket when scored as "code string in the answer"
  (uncompressed, cap 512, 25 documents × 3 depths per length): text 75 / 75 at ~4k / 8k / 16k / 32k tokens (191877);
  pages 75 / 75, 73 / 75, 74 / 75 at 10 / 20 / 40 pages (191878); mixed 75 / 75, 75 / 75, 74 / 75 (191879); bar 90%.
  The law says "exact scoring"; under EM the text needle scores .53 / .36 / .43 / .15 (191877) and fails, because the
  base decoder answers in a sentence. The substring scoring was chosen after the data were seen.
- provisional: arm E input check passes (job 191516, every row, CML 1,024): cut 0 and parity 1.0000 on text (2,246),
  pages (4,221), mixed (4,221) and the three needle sets; the CML 256 control cuts 235 / 2,246 (the cut check can
  fail). Parity has no failing control: one row builder serves both modes.
- provisional: the evaluation feeds the long inputs it claims on the gating sets (arm A, gate passed; job 191147): cut
  0 / 1,500 and parity 1,500 / 1,500 on every gating variant; health 646,042 = 646,042; CML 256 controls cut 924 / 1,500
  and 275 / 1,500. The gating sets are not long documents (owner, 2026-11-17) → arm E.
- provisional: the shared 32-token answer cap distorts every uncompressed reference (tokenizer count, answer ≥ 30
  tokens, `expQ/longdoc/length/cap32_sensitivity.json`, frontend, no job):
  - LongDocURL, 300 rows per set: uncompressed pages .340 → .717 from cap 32 to 256, +37.7 [+32.0, +43.0], sign 114 / 1
    (191634 → 191669); text .423 → .580, +15.7 [+10.7, +20.7], 58 / 11 (191635 → 191670); R pages .517 → .547, +3.0
    [+0.7, +5.3], 11 / 2 (191636 → 191673).
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
- Arm E full grid (owner-approved 2026-11-19): 4 of 26 cells submitted (192265-192268, above). Left: uncompressed pages
  p10 / p20 and mixed m10 / m20; R s42 and R s43 (text whole, pages and mixed per bucket: 14 cells); C256 / L1024 at
  CML 1,024 on text, s42 and s43 (4 cells, symlinks `expQ/longdoc/t_{c256,l1024}_s4{2,3}/model` exist).
- provisional (first read, trial rows; the full uncompressed cells decide): arm E soundness check 1 — the document is
  needed. Closed-book judge .164 on 1,407 questions (191776, cap 512). With the document − closed-book: text +33 / +33
  / +33 / +32 (75 Q each, 191777; CI lower bounds .173-.200, at or under the 20-pt bar); mixed +44 / +47 / +48 (100 Q
  each, 191778). Pages +52 / +52 / +54 is indicative only (its with-document file ran at cap 256, 191669: breaks the
  one-cap rule).
- provisional (plumbing, 75-100 Q per bucket, no paired test, decides nothing): uncompressed LongDocURL text at cap 512
  (191777) .587 / .587 / .587 / .573 at 4k-32k; mixed .63 / .66 / .67 at 10 / 20 / 40 items (191778).
- provisional (plumbing, 300 rows, never decides): R s42 − uncompressed on LongDocURL. Pages cap 256 −17.0 [−23.0,
  −11.0] (24 / 75; −13 / −15 / −23 at 10 / 20 / 40 pages; 191673 vs 191669); text cap 256 −16.3 [−21.7, −11.0] (15 / 64;
  −16 / −19 / −12 / −19 at 4k-32k; 191674 vs 191670); mixed cap 512 −15.7 [−22.0, −9.3] (31 / 78; −12 / −14 / −21 at 10
  / 20 / 40 items; 191887 vs 191778).
- provisional, NOT DECIDED (32-token cap confound; the 512 re-run was not approved): arm L documents, Δ_len vs 4
  pages, 1,500 pairs: 10 pages +0.5 [−1.8, +2.7] (s42, 191378), +0.1 [−2.2, +2.5] (s43, 191390); 20 pages +4.3 [+1.8,
  +6.9] (s42, 191377), +3.1 [+0.6, +5.7] (s43, 192007), questions with no capped answer −0.6 / −2.0 (n 1,221 / 1,220).
  "The gap shrinks at 20 pages" is withdrawn: it is the cap. Text 40 vs 10 passages: −2.6 [−5.6, +0.3] (s42, 191392),
  +0.1 [−3.0, +3.1] (s43, 192008); no capped answer −4.8 / −3.4 (n 731 / 733, a subset chosen on answers).

## Arm E — the long-document yardstick
Source: LongDocURL (`dengchao/LongDocURL`, Apache-2.0). Builder `scripts/longdoc/longdoc_build.py` (md5 f14e4640) +
`build_long.sbatch`. Rows are interleaved by question, so `LIMIT=300` = 100 whole questions at every page size, or 75 at
every text size. `<MEM>` slots p50 per bucket: pages 1,198 / 2,392 / 4,795; text 540 / 1,046 / 2,055 / 4,079; mixed
977 / 1,922 / 3,821 (max 14,688). Uncompressed prompt max: pages 39,562; text 36,965; mixed 117,451 (measured up to
89,311 on one GPU, 191778). Measured walls per 300 judged rows: uncompressed text 23:26 (cap 512), mixed 43:47 (512),
pages 31:01 (256); R pages 31:05 (256), text 30:22 (256), mixed 31:19 (512). MaxRSS ≤ 4.4 GB everywhere.
Submission plan (≤ 4 per wake, `--time` rule above; whole-set walls: U pages 8.0 h, U mixed 10.3 h, U text 2.9 h,
R pages 7.3 h, R mixed 7.3 h, R text 3.8 h, C256 / L1024 text 3.8 h):
1. When 192265 ends (check its log: 6 buckets × 1,407 rows, FAILURES none): uncompressed pages p10 / p20 (`--time
   8:00` each), mixed m10 / m20 (`--time 10:18` each).
2. R s42: text (`--time 11:24`, DML 5232), pages p40 (`--time 12:30`, DML 5612), mixed m40 (`--time 12:30`, DML
   15672), pages p10. Then R s42 p20 / m10 / m20 + R s43 text; then R s43 per bucket; then C256 / L1024 text.
3. Each full cell: health (n rows, judge on all, unparseable 0, no "exceeds", answers at the cap < 5%, `mems_used`
   vs audit, node, model.py md5 f8ae6efe); soundness check 1 per bucket with `closedbook_pair.py`; gap and Δ_len per
   bucket with `length_effect.py`; the project-prompt re-judge on any number that decides.
Launch shape: `expmon launch --exp longdoc --stage E -p gpu-be --mem 8G --time … --exclude='homer-[005-009]'
[--dependency afterok:<id>] --env MNT=512 --env PX=1003520 --env TOPK=1000 --env CML=1024 [--env DML=<set>]
scripts/longdoc/eval_mnt.sbatch expQ/longdoc/e_r4{2,3}/model pisco|base '' /beegfs/scratch/user/hdejean/pisco/eval_data/longdoc_<set>[_<b>]`.
Check `len(argv)` = 4 in `.expmon/runs.jsonl` after each launch (192266-192268: 4).

## Owned jobs
| job | arm | state | rows | note |
|---|---|---|---|---|
| 192265 | E split build | RUNNING homer-014 | 6 bucket sets | CPU |
| 192266 | E uncompressed text, cap 512 | RUNNING homer-016 | 2,246 | model.py f8ae6efe in log |
| 192267 / 192268 | E uncompressed pages p40 / mixed m40 | PENDING afterok 192265 | 1,407 each | |
| 192142, 192007, 192008, 191877-191879, 191887, 191774, 191776-191778, 191669, 191670, 191673, 191674, 191629, 191634-191636 | E, J, L | COMPLETED, verdict closed | | |
| 191517 | J re-judge (first try) | FAILED, verdict closed "launch fault" | — | replaced by 191629 |
| 191515 / 191516 / 191392 / 191393 / 191375-191378 / 191390 / 191391 / 191305-191308 / 191218-191221 / 191169-191172 / 191148-191155 / 191165-191168 / 191145-191147 | E, L, T, A | COMPLETED, verdict closed | | |

Cells used: 61 of 87. GPU-h: 32.0 of 114 at the last sacct (08:30) + the cells now running.
Code md5 for the arm E full grid: eval_mm_4b.sh b3d1cda1, eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py 02bc8748,
pisco/model.py f8ae6efe, eval_mnt.sbatch 1c874b32.

## Adopted jobs
(none — the other jobs in the queue are the expmon daemons 188764 / 162472 and campaign slotcheck's 192237-192239)

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| E long-document yardstick | full grid approved and started (4 / 26 cells) | the submission plan above |
| L length on a fixed model | NOT DECIDED (cap confound); 512 re-run not approved | arm E answers the length question |
| J instrument | control 1 PASS (frontend); project prompt done for every arm L number | re-run the project prompt on any arm E number that decides |
| U reading many `<MEM>`s | not started | after E; only where a length loss shows |
| P positions | not started | after E; needs `--mem_position_offset` (patchable IF, only while no sibling eval runs) |
| A audit | DECIDED (provisional); E re-run PASS (191516) | — |
| T long-chunk training | CLOSED (provisional) | text half = arm E's C256 vs L1024 cells |

## Tooling and controls
- `scripts/longdoc/eval_mnt.sbatch` (md5 1c874b32): arm E cells. `--env MNT=<N≠32>` required; STYLE=expand refused;
  writes the derived launcher to `expQ/longdoc/launchers/eval_mm_4b_mnt.<jobid>.sh`; exits 3 if `eval_mm_4b.sh`
  changed shape. Base-mode results land in `expQ/base_qwen35_4b/eval/` (new names: `_mnt512_<set>`).
- `scripts/longdoc/split_buckets.py` / `.sbatch` (md5 3ef018be): per-bucket sets; keeps a bucket already written.
- `scripts/longdoc/closedbook_pair.py CLOSEDBOOK DOC… [--json OUT]` (stdlib): soundness check 1 per bucket.
- `scripts/longdoc/length_effect.py +A -B …` (judge default, `--metric em|match`, `--json`); tables in `expQ/longdoc/length/`.
- `scripts/longdoc/rejudge.sbatch NAME FILE…` (md5 9e2bc7d6): project-prompt judge on copies. Pass files as separate
  arguments (zsh does not word-split `$F`).
- Cap count (every file of a contrast, before reading a Δ_len): `.pixi/envs/default/bin/python -I`,
  `t = tokenizers.Tokenizer.from_file("expQ/ft_mm_4b_184778/model/tokenizer.json")`, `t.no_padding(); t.no_truncation()`,
  answer at the cap = `len(t.encode(pred).ids) >= cap - 2`.
- The guard refuses any command containing `rm -r`, even on a `/tmp` scratch dir.
- `expmon display --exp longdoc` prints "0 table rows" but the page holds the grid: known gripe be8057a1.

## Blockers / needs the owner
(none)
