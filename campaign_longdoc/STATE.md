# STATE — campaign longdoc   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-18 17:30 (thirteenth LLM wake; woken because the four jobs it waited on ended: re-judge 191629 and
arm E trial runs 191634 / 191635 / 191636).
NEXT: wake-when job:191669,191670,191673,191674 max 3h.
Question: where do long inputs lose accuracy (eval, training data, length, `<MEM>` reading, positions, long-chunk
training)? Diagnosis only: inference probes and CPU audits, no training.

## This wake
- Arm J re-judge 191629 healthy (12 copies × 1,500 rows, project prompt on all, unparseable 0; judge-only wall 5 min 46 s
  for 18,000 rows), verdict closed. Δ_len recomputed under the project prompt: `expQ/longdoc/length/*_project.json`.
- Arm E trial runs 191634 / 191635 / 191636 healthy (300 rows each, judge on all, unparseable 0, no "exceeds" error, one
  attempt each on homer-011 A100-80GB; R `mems_used` 840,688 = audit 840,688 on 300 / 300 rows; prompt tokens = audit
  300 / 300 in base mode), verdicts closed. Walls / MaxRSS: uncompressed pages 24:10 / 4.55 GB, uncompressed text
  12:09 / 2.0 GB, R pages 26:45 / 4.50 GB.
- FINDING in those trial runs: the uncompressed decoder scores BELOW R (.34 vs .52 on the same 300 page rows) because
  the shared 32-token answer cap cuts most of its sentence-style answers (Arm E section). Fix inside the campaign:
  `scripts/longdoc/eval_mnt.sbatch` (md5 1c874b32) = a runtime copy of the shared `eval_mm_4b.sh` with only `MNT=32` →
  `MNT=$MNT` and a `_mnt<N>` file-name suffix; the substitution is checked (exactly 2 lines differ) and both md5s are
  logged. The shared launcher was NOT patched: sibling evals (191659-191662) were running, and the law forbids it then.
- Re-ran the same 300 rows at a 256-token cap (cap 4 reached): uncompressed pages 191669, uncompressed text 191670,
  R pages 191673, R text 191674 (the first compressed run on the long text set).
- CAMPAIGN.md tightened: every arm E cell runs through `eval_mnt.sbatch` with `MNT=256`; cells pair only at the same
  cap; if uncompressed answers still hit the cap on ≥ 5% of the 300 rows, raise it again before any soundness cell.
- known_issues.md: one row (uncompressed below compressed on a new set → answer cap). Gripes: +1 on c0b20b25 (the
  guard refused a whole command for `rm -r` of a `mktemp -d /tmp` dir; the dir was left in /tmp), one wish (derived
  launchers).
- `docs/longdoc_diagnosis.md` rewritten in place: arm J row, the answer-cap row, project-prompt numbers in the length
  tables, an arm E trial-run table.

## TL;DR — where this leaves the question
**Settled**
- Training never shows the decoder long inputs (a count, not yet a cause; job 191147): finetune rows reach at most 488
  `<MEM>`s and a position span of 4,017 (2,000 sampled rows). Out of that range: 10- and 20-page documents 1,500 /
  1,500, 40 text passages 1,498 / 1,500, 20 passages 35 / 1,500, whole documents 198 / 500; 4-page documents 0 / 1,500.
  Every arm E bucket is out of range too (job 191516).
- The transcription-probe port is validated (job 191169): 360 / 360 legacy pages byte-identical, word recall 70.5% =
  70.5% (gate ±2 pt).
- Arm E sets exist (job 191515): LongDocURL (Apache-2.0), 1,407 usable questions on 387 PDFs; pages and mixed 4,221 rows
  each (1,407 × 10 / 20 / 40 pages of the question's own PDF), text 2,246 rows (t4k 637 / t8k 610 / t16k 588 / t32k
  411), needles text 300 / pages 225 / mixed 225.
- provisional: arm E input check PASS (job 191516, every row, CML 1,024): cut 0 and parity 1.0000 on text (2,246), pages
  (4,221), mixed (4,221) and the three needle sets; the CML 256 control cuts 235 / 2,246 text rows (10.5%). Its
  `audit.json` health field reads false only because `--only E_` skips the 4-page variant it is computed on (health
  passed in 191147); the E variants were added to `audit_inputs.py` after 191147.
- provisional: the evaluation feeds the long inputs it claims on the gating sets (arm A, gate passed; job 191147):
  truncation cut 0 / 1,500 and pisco/base parity 1,500 / 1,500 on every gating variant; health 646,042 = 646,042; the
  CML 256 controls fail as they should (924 / 1,500 and 275 / 1,500). Limits: on documents the cut check can only fail
  on prompt length; the audit's parity used one row builder for both modes. The gating sets are not long documents
  (owner, 2026-11-17) → arm E.
- provisional: arm J — the length-effect calculator recovers the tail model's one-chunk minus ten-chunk HotpotQA
  difference (183538, 1,500 pairs): −.1687 [−.1960, −.1407] (target −.169 ± .005), self-pair 0 (computed on the
  frontend, no job). Second judge prompt (job 191629, 1,500 pairs per number), Δ_len rubric / project / EM:
  documents 10 vs 4 pages +0.5 / +0.6 / +1.6 (s42), +0.1 / −0.3 / +0.9 (s43): the no-loss reading holds under all
  three (every CI inside ±3; the s43 sign flip is between +0.1 and −0.3); 20 vs 4 pages s42 +4.3 / +5.1 / +7.7:
  positive under all three, EM 3.4 pt above the judges; text 40 vs 10 passages s42 −2.6 / −1.5 / +0.8: judge-sensitive
  (sign flips with EM).
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
- provisional: documents (`mm_docqa_1500_1m`, distractor pages from OTHER documents), 10 vs 4 pages — no length loss
  at either seed: Δ_len +0.5 [−1.8, +2.7], sign 144 / 138, p .77 (s42, 191378); +0.1 [−2.2, +2.5], 146 / 142, p .86
  (s43, 191390); 1,500 pairs each; project prompt +0.6 [−1.7, +2.9] / −0.3 [−2.7, +2.1]. R .607 / .608 at 10 pages
  vs .610 / .615 at 4; uncompressed .801 vs .809.

**Open, and running**
- provisional (one seed): documents 20 vs 4 pages, s42 (191377 R .599, 191376 uncompressed .754): Δ_len +4.3 [+1.8,
  +6.9], sign 197 / 146, p .007, 1,500 pairs; project prompt +5.1 [+2.6, +7.7], p .0009; EM +7.7. The gap SHRINKS
  (−19.9 → −15.5 pt) because the uncompressed reader falls more. Needs R s43 at 20 pages (after arm E's costing).
- provisional (one seed, undecided, judge-sensitive): text 40 vs 10 passages s42, 1,500 pairs (R40 191392 .534, U40
  191393 .707, R10 191518 .605, U10 .752): Δ_len −2.6 [−5.6, +0.3], p .084; project −1.5 [−4.6, +1.5], p .35; EM +0.8.
- provisional (plumbing, 300 rows, never decides): on LongDocURL pages the uncompressed decoder's answers run into
  the 32-token answer cap on 61 / 67 / 74% of rows at 10 / 20 / 40 pages (heuristic: ≥ 90 characters, no final
  punctuation); judge .16-.21 on those rows vs .58-.72 on finished ones; text 47-64%; R 8-17%. Uncompressed .41 / .34
  / .27 vs R .54 / .55 / .46 at 10 / 20 / 40 pages (191634 vs 191636). On the gating documents set only 4% (SlideVQA
  11%) of uncompressed answers reach the cap (recorded file of the 4-page uncompressed cell).
- Running (all at a 256-token cap, same 300 rows): 191669 uncompressed pages, 191670 uncompressed text, 191673 R pages,
  191674 R text.

## Arm E — the long-document yardstick
Source: LongDocURL (`dengchao/LongDocURL`, Apache-2.0). Builder `scripts/longdoc/longdoc_build.py` (md5 f14e4640) +
`build_long.sbatch`. Rows are interleaved by question, so `LIMIT=300` = 100 whole questions at every page size, or 75 at
every text size. `<MEM>` slots p50 per bucket: pages 1,198 / 2,392 / 4,795; text 540 / 1,046 / 2,055 / 4,079; mixed
977 / 1,922 / 3,821 (max 14,688). Uncompressed prompt max: pages 39,562; text 36,965; mixed 117,451.
Many LongDocURL answers are yes/no or one option letter (closed-book may score well: soundness check 1 measures it).
Next, in order:
1. When 191669 / 191670 / 191673 / 191674 end: health (n 300, judge on all, unparseable 0, no "exceeds" error; R
   `mems_used` vs audit rows `E_pages_cml1024` / `E_text_cml1024`, `expQ/longdoc/audit_191516/rows.jsonl`, key `qid`);
   uncompressed cut share at 256 (a cut answer is ~1,000 characters with no final punctuation); paired 32 vs 256 on
   the same rows (`length_effect.py +new -old`); walls and MaxRSS.
2. If the cut share is < 5%: soundness cells (full sets, uncompressed, `eval_mnt.sbatch` `MNT=256`): closed-book
   (`TOPK=0`: `gen_base` slices `docs[:topk]`, so the background is empty — read in `eval_mm_docqa.py:526-541,723`;
   the XD check near line 596 not read yet) on pages, text, mixed; needles uncompressed on text, pages, mixed; and the
   mixed plumbing pair (uncompressed + R, LIMIT=300; the 117k-token prompt needs its memory measured).
3. Cost the model grid from the measured walls (R s42 / s43 on pages, text, mixed; uncompressed full cells; C256 vs
   L1024 at CML 1,024 on text, s42 / s43). Rough prior: a full pages cell is 4,221 rows ≈ 14 × the 300-row wall ≈ 6 h
   per model; the grid is likely ~50 GPU-h against ~35 left and 24 cells left → ONE BLOCKED.md question with the
   measured cost (law: never trim the grid to fit).

## Owned jobs
| job | arm | state | rows | note |
|---|---|---|---|---|
| 191669 | E cap check, uncompressed pages, `MNT=256` (`--time 1:15`, 8G) | RUNNING | 300 | out `expQ/base_qwen35_4b/eval/mm_docqa_base_k1000_px1003520_cml1024_mnt256_longdoc_pages_1500_lim300.json`; pairs with 191634 |
| 191670 | E cap check, uncompressed text, `MNT=256` (`--time 1:00`, 8G) | RUNNING | 300 | `…_cml1024_mnt256_longdoc_text_1500_lim300.json`; pairs with 191635 |
| 191673 | E cap check, R s42 pages, `MNT=256`, DML 5,612 (`--time 1:20`, 8G) | RUNNING | 300 | `expQ/longdoc/e_r42/eval/mm_docqa_pisco_k1000_px1003520_cml1024_mnt256_longdoc_pages_1500_lim300.json`; pairs with 191636 |
| 191674 | E plumbing, R s42 text, `MNT=256`, DML 5,232 (`--time 1:15`, 8G) | RUNNING | 300 | `expQ/longdoc/e_r42/eval/…_cml1024_mnt256_longdoc_text_1500_lim300.json`; wall unmeasured |
| 191629 / 191634 / 191635 / 191636 | J, E | COMPLETED, verdict closed this wake | | health in the verdicts |
| 191517 | J re-judge (first try) | FAILED, verdict closed "launch fault" | — | replaced by 191629 |
| 191515 / 191516 / 191392 / 191393 / 191375-191378 / 191390 / 191391 / 191305-191308 / 191218-191221 / 191169-191172 / 191148-191155 / 191165-191168 / 191145-191147 | E, L, T, A | COMPLETED, verdict closed | | |

Cells used: 46 of 70. GPU-h: ≈ 22 of 60 (≈ 21 before this wake + 191629 0.1 + 191634 0.4 + 191635 0.2 + 191636
0.45); the four cells now running add ≈ 2-4. Arm E build / cut check are CPU (0 GPU-h).

## Adopted jobs
(none — the other jobs in the queue are the expmon daemons 188764 / 162472 and the spread / qafreeze campaigns' cells
191659-191662)

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| E long-document yardstick | sets built, input check PASS; trial runs healthy but the uncompressed side is capped; re-run at 256 tokens running | Arm E section, steps 1-3 |
| L length on a fixed model | docs 10 pages no loss (provisional, both seeds, both judges); 20 pages one seed; text 40 one seed, undecided + judge-sensitive | R s43 at 20 pages and text 40 s43 after arm E's costing (20 pages: XD=16 TOPK=20 PX=1003520 DML=2823, `--time` 4:27 = 191377 wall 1:29 × 3, `--mem 8G`; text 40: XD=30 TOPK=40 CML=4096 DML=1545, `--time` 2:04 = 191392 wall 41 min × 3). These run on the shared launcher (gating sets, 32-token cap, as recorded) |
| J instrument | control 1 PASS; second judge prompt done (191629) | closed for the recorded numbers; re-run the project prompt on any new number that decides an arm |
| U reading many `<MEM>`s | not started | after E (owner); only where a length loss shows |
| P positions | not started | after E (owner); needs `--mem_position_offset` (patchable IF, only while no sibling eval runs) |
| A audit | DECIDED (provisional); E re-run PASS (191516) | — |
| T long-chunk training | CLOSED (provisional) | the text half moves to arm E (C256 vs L1024 at CML 1,024) |

## Tooling and controls
- `scripts/longdoc/eval_mnt.sbatch` (md5 1c874b32): arm E cells. `--env MNT=<N≠32>` required; STYLE=expand refused;
  writes the derived launcher to `expQ/longdoc/launchers/eval_mm_4b_mnt.<jobid>.sh` and logs it to
  `expQ/longdoc/eval_mnt_<jobid>.out`; exits 3 if `eval_mm_4b.sh` changed shape.
- `scripts/longdoc/length_effect.py +A -B …` (judge default, `--metric em|match`, `--json`); tables in `expQ/longdoc/length/`.
- `scripts/longdoc/rejudge.sbatch NAME FILE…` (md5 9e2bc7d6): project-prompt judge on copies; requeue-safe; 18,000
  rows in 5 min 46 s. Pass the files as separate arguments (zsh: `F=(${=LIST})`, then `"${F[@]}"`).
- `audit_inputs.py --only E_`: the arm E variants only. Per-row output `expQ/longdoc/audit_191516/rows.jsonl` (keys `v`,
  `qid`, `n_mem`, `prompt`, `base_prompt`).
- Arm E launch shape: `expmon launch --exp longdoc --stage E --time … --mem 8G --exclude='homer-[005-009]' -p gpu-be
  --env MNT=256 --env PX=1003520 --env TOPK=1000 --env CML=1024 [--env DML=<pisco prompt max + 256>]
  scripts/longdoc/eval_mnt.sbatch expQ/longdoc/e_r4{2,3}/model pisco|base [LIMIT] /beegfs/scratch/user/hdejean/pisco/eval_data/longdoc_<set>`.
  DML per set: pages 5,612; text 5,232; mixed 15,672. Base mode ignores DML.
- Code md5 at launch of 191669-191674: eval_mm_4b.sh b3d1cda1 (wrapped), eval_mnt.sbatch 1c874b32, eval_mm_docqa.py
  dfe32aa0, judge_mm_docqa.py 02bc8748, judge_prompts.py bae44fa7, pisco/model.py 994cff3d.
- The guard refuses any command containing `rm -r`, even on a `/tmp` scratch dir: leave scratch dirs in /tmp.

## Blockers / needs the owner
- None now. Expected after the cap check and soundness cells: one BLOCKED.md question with the measured cost of arm E's
  model grid.
