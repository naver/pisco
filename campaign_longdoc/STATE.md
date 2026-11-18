# STATE — campaign longdoc   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-18 22:00 (fifteenth LLM wake; woken because closed-book 191776, text cap check 191777 and the first
mixed run 191778 ended).
NEXT: wake-when job:191877,191878,191879,191887 max 6h.
Question: where do long inputs lose accuracy (eval, training data, length, `<MEM>` reading, positions, long-chunk
training)? Diagnosis only: inference probes and CPU audits, no training.

## This wake
- 191776 / 191777 / 191778 healthy (one attempt each, COMPLETED 0:0, homer-011 / 019 / 020; rows = set size, judge on
  all, unparseable 0, no "exceeds" error), verdicts closed, grid probes set. Walls / MaxRSS: closed-book 30:07 /
  2.1 GB; uncompressed text 300 rows at 512 23:26 / 2.3 GB; uncompressed mixed 300 rows 43:47 / 3.9 GB (prompt max
  89,311 tokens, 40 images, one GPU).
- Answer cap: counted EXACTLY with the tokenizer (answer ≥ cap − 2 tokens; 2 s on the frontend, pixi python `-I`):
  uncompressed text at 512: 8 / 300 = 2.7% < 5% → the cap stays 512. The character heuristic was wrong both ways
  (256: 8.3% vs 11.0% exact; 512: 6.3% vs 2.7%). CAMPAIGN.md (clarified) and known_issues.md row 38 now require the
  tokenizer count.
- New `scripts/longdoc/closedbook_pair.py` (stdlib): pairs any with-document file with the closed-book file by the
  question id, per bucket. Soundness check 1, first read on the trial rows: PASS everywhere (TL;DR).
- Verifier fix: the "every arm E bucket out of range" count now has a file behind it,
  `expQ/longdoc/audit_191516/out_of_range.json` (from that audit's rows.jsonl).
- Launched (cap 4): needle checks uncompressed text 191877 / pages 191878 / mixed 191879 (soundness check 3) and the
  first compressed mixed run, R s42, 300 rows, 191887 (measures the last unmeasured wall for the cost question).
- `docs/longdoc_diagnosis.md` rewritten (cap row, soundness table). Gripes: +1 b496ac4c (health verdicts fill the
  Settled list); new gap (pending cells in the report show only env).

## TL;DR — where this leaves the question
**Settled**
- provisional: training never shows the decoder long inputs (a count, not yet a cause). Finetune rows reach at most 488
  `<MEM>`s and a position span of 4,017 (2,000 sampled rows, job 191147). Out of that range (191147): 10- and 20-page
  documents 1,500 / 1,500, 40 text passages 1,498 / 1,500, 20 passages 35 / 1,500; 4-page documents 0 / 1,500. Arm E
  rows above 488 `<MEM>`s (job 191516, `audit_191516/out_of_range.json`): text 2,246 / 2,246, pages 4,221 / 4,221,
  mixed 4,212 / 4,221.
- The transcription-probe port is validated (job 191169): 360 / 360 legacy pages byte-identical, word recall 70.5% =
  70.5% (gate ±2 pt).
- Arm E sets exist (job 191515): LongDocURL (Apache-2.0), 1,407 usable questions on 387 PDFs; pages and mixed 4,221 rows
  each (1,407 × 10 / 20 / 40 pages of the question's own PDF), text 2,246 rows (t4k 637 / t8k 610 / t16k 588 / t32k
  411), needles text 300 / pages 225 / mixed 225. Closed-book set: 1,407 questions (job 191774).
- The arm E answer cap is 512 tokens (job 191777): 8 / 300 uncompressed text answers reach it (2.7%, bar 5%, exact
  tokenizer count); mixed 8 / 300 (191778), closed-book 5 / 1,407 (191776).
- provisional: arm E input check PASS (job 191516, every row, CML 1,024): cut 0 and parity 1.0000 on text (2,246), pages
  (4,221), mixed (4,221) and the three needle sets; the CML 256 control cuts 235 / 2,246 text rows (10.5%). Parity uses
  one row builder for both modes, so it cannot fail by construction.
- provisional: the evaluation feeds the long inputs it claims on the gating sets (arm A, gate passed; job 191147):
  truncation cut 0 / 1,500 and pisco/base parity 1,500 / 1,500 on every gating variant; health 646,042 = 646,042; the
  CML 256 controls fail as they should (924 / 1,500 and 275 / 1,500). On documents the cut check can only fail on
  prompt length. The gating sets are not long documents (owner, 2026-11-17) → arm E.
- provisional: on LongDocURL the shared 32-token answer cap made the uncompressed reader look worse than the compressed
  one (300 rows per set, same rows): uncompressed pages .340 → .717 at 256 tokens, +37.7 [+32.0, +43.0], sign 114 / 1
  (191634 → 191669); uncompressed text .423 → .580, +15.7 [+10.7, +20.7], 58 / 11 (191635 → 191670); R pages .517 →
  .547, +3.0 [+0.7, +5.3], 11 / 2 (191636 → 191673). Plumbing rows: this fixes the instrument, it decides no arm.
- provisional (first read, trial rows; the full uncompressed cells decide): arm E soundness check 1 PASS — the document
  is needed in every bucket. Closed-book judge .164 on all 1,407 questions (191776). With the document − closed-book,
  same questions (`expQ/longdoc/soundness/closedbook_trial300.json`): pages +52 / +52 / +54 pt at 10 / 20 / 40 pages
  (100 Q each, cap 256, 191669); text +33 / +33 / +33 / +32 at ~4k / 8k / 16k / 32k tokens (75 Q each, 191777); mixed
  +44 / +47 / +48 (100 Q each, 191778); lowest CI bound +17; bar 20 pt.
- provisional: the uncompressed reader is flat with length on LongDocURL text at cap 512 (191777, 75 Q per bucket):
  .587 / .587 / .587 / .573 from ~4.5k to ~32.7k prompt tokens (only 5 of 75 questions give the same answer in every
  bucket, so the inputs do differ). Mixed .63 / .66 / .67 at 10 / 20 / 40 items (191778).
- provisional: arm J — the length-effect calculator recovers the tail model's one-chunk minus ten-chunk HotpotQA
  difference (183538, 1,500 pairs): −.1687 [−.1960, −.1407] (target −.169 ± .005), self-pair 0 (computed on the
  frontend with stdlib, no job backs it; EXPERIMENTS.md:1116). Second judge prompt (job 191629, 1,500 pairs per number),
  Δ_len rubric / project / EM: documents 10 vs 4 pages +0.5 / +0.6 / +1.6 (s42), +0.1 / −0.3 / +0.9 (s43); 20 vs 4
  pages s42 +4.3 / +5.1 / +7.7; text 40 vs 10 passages s42 −2.6 / −1.5 / +0.8. By the law's rule (same sign AND within
  2 pt) all three are judge-sensitive.
- provisional: arm T CLOSED — the long-chunk (L1024) loss on documents sits IN THE COMPRESSED IMAGE PAGE (gate met at
  both seeds; one stage-2 model per chain).
  - Documents `mm_docqa_1500_1m`, 1,003,520 px, 1,500 pairs per seed: gold page only L1024 − C256 −6.7 [−8.9, −4.5]
    McNemar 106 / 207 (s42; 191153 .568 vs 191152 .635), −4.9 [−7.1, −2.5] 113 / 186 (s43; 191155 .579 vs 191154
    .628). Picked page raw +0.1 [−0.8, +1.1] 29 / 27 (s42; 191166 vs 191165), −0.1 [−1.1, +0.9] 29 / 31 (s43; 191168
    vs 191167).
  - Page transcription, stage-2 models (191219 / 191220 vs reference 191170): .566 vs .647 of the page's words, −8.0 pt
    [−10.6, −5.5], sign 147 / 322, p 4e-16, 500 pages scored of 501 paired.
  - Text autoencoding at 256 tokens, 500 held-out MS MARCO documents, d = L1024 − C256: stage 1 −.0018 [−.0031,
    −.0006], sign 271 / 229, p .067 (191218); stage 2 +.0006 [−.0008, +.0019], 247 / 253, p .82 (191308). Consistent
    with an image-specific loss (a null with no stated margin).
  - Text QA (`text_hotpotqa_1500`, CML 4,096, 1,500 pairs per seed): ten short chunks L1024 − C256 +2.3 [+0.3, +4.4]
    McNemar 136 / 101 (s42; 191305 .603 vs 191221 .579), −0.6 [−2.7, +1.3] 121 / 130 (s43; 191307 .589 vs 191306
    .595): the seeds disagree; not decided. The long-chunk question on text moves to arm E (CML 1,024).
  - Proposed to the owner (a recommendation, untested by any arm): keep the 256-token stage 1; if a long-chunk stage 1
    is retried, vary stage 2 from the L1024 start and check page transcription before any finetune.
- provisional (judge-sensitive by the letter at s43, see arm J): documents (`mm_docqa_1500_1m`, distractor pages from
  OTHER documents), 10 vs 4 pages — no length loss at either seed: Δ_len +0.5 [−1.8, +2.7], sign 144 / 138, p .77
  (s42, 191378); +0.1 [−2.2, +2.5], 146 / 142, p .86 (s43, 191390); 1,500 pairs each; project prompt +0.6 [−1.7, +2.9]
  / −0.3 [−2.7, +2.1]. R .607 / .608 at 10 pages vs .610 / .615 at 4; uncompressed .801 vs .809.

**Open, and running**
- provisional (one seed, judge-sensitive): documents 20 vs 4 pages, s42 (191377 R .599, 191376 uncompressed .754):
  Δ_len +4.3 [+1.8, +6.9], sign 197 / 146, p .007, 1,500 pairs; project prompt +5.1 [+2.6, +7.7], p .0009; EM +7.7.
  The gap SHRINKS (−19.9 → −15.5 pt) because the uncompressed reader falls more. Needs R s43 at 20 pages (this
  campaign's own cell now: campaign spread withdrew hand-off 19ac33).
- provisional (one seed, undecided, judge-sensitive): text 40 vs 10 passages s42, 1,500 pairs (R40 191392 .534, U40
  191393 .707, R10 .605 = campaign spread's re-run 191518 of model 184778, borrowed, not a cell of this grid; U10 .752):
  Δ_len −2.6 [−5.6, +0.3], p .084; project −1.5 [−4.6, +1.5], p .35; EM +0.8.
- provisional (plumbing, 300 rows, never decides): at 256 tokens on LongDocURL, R s42 − uncompressed = −17.0 pt
  [−23.0, −11.0] on pages (24 / 75; −13 / −15 / −23 at 10 / 20 / 40 pages, 100 Q each; 191673 vs 191669) and −16.3 pt
  [−21.7, −11.0] on text (15 / 64; −16 / −19 / −12 / −19 at 4k / 8k / 16k / 32k, 75 Q each; 191674 vs 191670).
- Running / queued: needle checks 191877 (text) / 191878 (pages) / 191879 (mixed), R s42 mixed trial 191887.

## Arm E — the long-document yardstick
Source: LongDocURL (`dengchao/LongDocURL`, Apache-2.0). Builder `scripts/longdoc/longdoc_build.py` (md5 f14e4640) +
`build_long.sbatch`. Rows are interleaved by question, so `LIMIT=300` = 100 whole questions at every page size, or 75 at
every text size. `<MEM>` slots p50 per bucket: pages 1,198 / 2,392 / 4,795; text 540 / 1,046 / 2,055 / 4,079; mixed
977 / 1,922 / 3,821 (max 14,688). Uncompressed prompt max: pages 39,562; text 36,965; mixed 117,451 (measured so far up
to 89,311 on one GPU, 191778). Needles: 25 PDFs × buckets × depth 10 / 50 / 90%; uncompressed prompt max text 33,504,
pages 40,138, mixed 68,714.
Next, in order:
1. When 191877-191879 / 191887 end: health (n rows, judge on all, unparseable 0, no "exceeds" error, one attempt, node);
   needles: exact-match share per length bucket (and per depth), bar ≥ 90% → a bucket under it is dropped; R mixed:
   `mems_used` vs `audit_191516/rows.jsonl` `E_mixed_cml1024` first 300 rows (expect ±1%), wall, MaxRSS, answers at
   the cap (tokenizer count).
2. Same wake, cap 4: R s43 at 20 pages for arm L (`eval_mm_4b.sh expQ/longdoc/l_r43/model pisco ""
   /beegfs/…/eval_data/mm_docqa_1500_1m`, env DML=2823 PX=1003520 TOPK=20 XD=16 = 191377's row, `--time 4:27` = 191377
   wall 1:29 × 3, `--mem 8G`); text 40 passages s43 (env XD=30 TOPK=40 CML=4096 DML=1545 as 191392, `--time 2:04`).
3. Cost the arm E model grid and write ONE BLOCKED.md question (law: never trim the grid to fit). Draft from measured
   walls per 300 judged rows: uncompressed pages 31:01 (cap 256), text 23:26, mixed 43:47; R pages 31:05, text 30:22
   (cap 256); R mixed = 191887. Full cells: uncompressed pages 4,221 rows ≈ 7.3 h, text 2,246 ≈ 2.9 h, mixed 4,221 ≈
   10.3 h (soundness check 1 reference, also the gap reference); R s42 + s43 pages ≈ 14.6 h, text ≈ 7.6 h, mixed (from
   191887); C256 vs L1024 at CML 1,024 on text, s42 / s43, 4 cells ≈ 15 h. Total ≈ 75 GPU-h in ~13 cells against
   ≈ 30 GPU-h and 16 cells left → over budget → the owner decides. Raise in the question: a 10-h gpu-be cell has no
   resume (`eval_mm_4b.sh` regenerates on requeue), so the mixed and pages cells may need `gpu` (≤ 2 at once) or a
   split by rows.

## Owned jobs
| job | arm | state | rows | note |
|---|---|---|---|---|
| 191877 | E needle text, uncompressed, cap 512 (`--time 1:10`, 4G) | PENDING | 300 | out `expQ/base_qwen35_4b/eval/mm_docqa_base_k1000_px1003520_cml1024_mnt512_longdoc_needle_text.json` |
| 191878 | E needle pages, uncompressed, cap 512 (`--time 1:35`, 5G) | PENDING | 225 | `…_mnt512_longdoc_needle_pages.json` |
| 191879 | E needle mixed, uncompressed, cap 512 (`--time 2:15`, 6G) | PENDING | 225 | `…_mnt512_longdoc_needle_mixed.json` |
| 191887 | E first R s42 mixed run, cap 512, DML 15,672 (`--time 2:15`, 9G) | PENDING | 300 | `expQ/longdoc/e_r42/eval/mm_docqa_pisco_k1000_px1003520_cml1024_mnt512_longdoc_mixed_1500_lim300.json` |
| 191776 / 191777 / 191778 | E closed-book, text cap 512, mixed uncompressed trial | COMPLETED, verdict closed this wake | 1,407 / 300 / 300 | health in the verdicts |
| 191774 / 191669 / 191670 / 191673 / 191674 / 191629 / 191634-191636 | E, J | COMPLETED, verdict closed | | |
| 191517 | J re-judge (first try) | FAILED, verdict closed "launch fault" | — | replaced by 191629 |
| 191515 / 191516 / 191392 / 191393 / 191375-191378 / 191390 / 191391 / 191305-191308 / 191218-191221 / 191169-191172 / 191148-191155 / 191165-191168 / 191145-191147 | E, L, T, A | COMPLETED, verdict closed | | |

Cells used: 54 of 70. GPU-h: ≈ 26 of 60 (≈ 24 before the last wake's three cells + ≈ 1.6 for them); the four cells
now queued add ≈ 2.5-3. CPU jobs count 0 GPU-h.

## Adopted jobs
(none — the other jobs in the queue are the expmon daemons 188764 / 162472, campaign slotcheck's 191783 / 191784 and
campaign spread's 191785)

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| E long-document yardstick | sets built, input check PASS; cap 512 fixed; closed-book check passes on the trial rows; needles + R mixed trial running | Arm E section, steps 1-3 |
| L length on a fixed model | docs 10 pages no loss (provisional, both seeds; judge-sensitive by the letter at s43); 20 pages one seed; text 40 one seed, undecided + judge-sensitive | R s43 20 pages + text 40 s43 next wake (shared launcher, gating sets, 32-token cap as recorded) |
| J instrument | control 1 PASS (frontend, no job); second judge prompt done (191629) | re-run the project prompt on any new number that decides an arm |
| U reading many `<MEM>`s | not started | after E (owner); only where a length loss shows |
| P positions | not started | after E (owner); needs `--mem_position_offset` (patchable IF, only while no sibling eval runs) |
| A audit | DECIDED (provisional); E re-run PASS (191516) | — |
| T long-chunk training | CLOSED (provisional) | the text half moves to arm E (C256 vs L1024 at CML 1,024) |

## Tooling and controls
- `scripts/longdoc/eval_mnt.sbatch` (md5 1c874b32): arm E cells. `--env MNT=<N≠32>` required; STYLE=expand refused;
  writes the derived launcher to `expQ/longdoc/launchers/eval_mm_4b_mnt.<jobid>.sh` and logs it to
  `expQ/longdoc/eval_mnt_<jobid>.out`; exits 3 if `eval_mm_4b.sh` changed shape.
- `scripts/longdoc/closedbook_pair.py CLOSEDBOOK DOC… [--json OUT]` (stdlib): soundness check 1 per bucket.
- `scripts/longdoc/build_closedbook.py` + `.sbatch` (md5 42758c8b): the closed-book set; exits 1 on a query mismatch.
- `scripts/longdoc/length_effect.py +A -B …` (judge default, `--metric em|match`, `--json`); tables in `expQ/longdoc/length/`.
- `scripts/longdoc/rejudge.sbatch NAME FILE…` (md5 9e2bc7d6): project-prompt judge on copies; requeue-safe. Pass the
  files as separate arguments (zsh: `F=(${=LIST})`, then `"${F[@]}"`).
- `audit_inputs.py --only E_`: the arm E variants only. Per-row output `expQ/longdoc/audit_191516/rows.jsonl` (keys `v`,
  `qid`, `n_mem`, `prompt`, `base_prompt`).
- Cap count: `.pixi/envs/default/bin/python -I` + `tokenizers.Tokenizer.from_file("expQ/ft_mm_4b_184778/model/tokenizer.json")`,
  answer at the cap = `len(ids) >= cap - 2`.
- Arm E launch shape: `expmon launch --exp longdoc --stage E --time … --mem … --exclude='homer-[005-009]' -p gpu-be
  --env MNT=512 --env PX=1003520 --env TOPK=1000 --env CML=1024 [--env DML=<pisco prompt max + 256>]
  scripts/longdoc/eval_mnt.sbatch expQ/longdoc/e_r4{2,3}/model pisco|base [LIMIT|''] /beegfs/scratch/user/hdejean/pisco/eval_data/longdoc_<set>`.
  DML per set: pages 5,612; text 5,232; mixed 15,672. Base mode ignores DML. `expQ/longdoc/e_r43/model` does not exist
  yet (symlink to `expQ/ft_mm_4b_184780/model` before the first s43 arm E cell). Check `len(argv)` = 4 in
  `.expmon/runs.jsonl` after each launch.
- Code md5 at launch of 191877-191887: eval_mm_4b.sh b3d1cda1 (wrapped), eval_mnt.sbatch 1c874b32, eval_mm_docqa.py
  dfe32aa0, judge_mm_docqa.py 02bc8748, judge_prompts.py bae44fa7, pisco/model.py 994cff3d.
- The guard refuses any command containing `rm -r`, even on a `/tmp` scratch dir: leave scratch dirs in /tmp.

## Blockers / needs the owner
- None now. Expected next wake: one BLOCKED.md question with the measured cost of arm E's model grid (draft in the
  Arm E section, step 3).
