# STATE — campaign longdoc   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-18 21:00 (fourteenth LLM wake; the 18:30 wake died on the account's 5-hour usage limit, so this wake
read the four answer-cap cells that ended at 17:58-18:08).
NEXT: wake-when job:191776,191777,191778 max 6h.
Question: where do long inputs lose accuracy (eval, training data, length, `<MEM>` reading, positions, long-chunk
training)? Diagnosis only: inference probes and CPU audits, no training.

## This wake
- Answer-cap cells 191669 / 191670 / 191673 / 191674 healthy (300 rows each, judge on all, unparseable 0, no
  "exceeds" error, one attempt each on homer-011 / 011 / 019 / 010; R `mems_used` = audit on 300 / 300 rows for pages
  and text), verdicts closed. Walls / MaxRSS: uncompressed pages 31:01 / 3.2 GB, uncompressed text 21:20 / 1.8 GB,
  R pages 31:05 / 5.9 GB, R text 30:22 / 4.2 GB.
- At 256 tokens the uncompressed answers still reach the cap on 7 / 300 page rows (2.3%) and 25 / 300 text rows (8.3%;
  heuristic ≥ 600 characters, no final punctuation; t32k 11 / 75). Text is above the law's 5% bar → CAMPAIGN.md
  tightened: ONE cap, 512 tokens, for every arm E cell (closed-book and needles included); 1,024 if text still ≥ 5% at
  512. The 32- and 256-token cells stay plumbing.
- New set `eval_data/longdoc_closedbook` (job 191774, 15 s CPU; `scripts/longdoc/build_closedbook.py` md5 42758c8b):
  every arm E question once (1,407; the 637 text questions are among them), no document; query and answers identical
  across the three sets (0 mismatches). Without a document the base prompt is the same for every set and bucket, so one
  closed-book cell serves them all.
- Launched (cap 4): closed-book 191776 (chained afterok 191774), text cap check at 512 191777, first mixed run
  (uncompressed, 300 rows, 512) 191778.
- Cleared the ALERT: the verifier line was from 2026-11-17 (the 2026-11-18 17:30 verifier found 0 unsupported claims)
  and the usage limit reset at 21:00. Verifier asks applied: judge-sensitivity labels below and in
  `docs/longdoc_diagnosis.md`.
- Gripe: +1 on be8057a1 (`expmon display` prints "0 table rows … from 0 runs" while the page has 51 rows).

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
  411), needles text 300 / pages 225 / mixed 225. Closed-book set: 1,407 questions (job 191774).
- provisional: arm E input check PASS (job 191516, every row, CML 1,024): cut 0 and parity 1.0000 on text (2,246), pages
  (4,221), mixed (4,221) and the three needle sets; the CML 256 control cuts 235 / 2,246 text rows (10.5%). Its
  `audit.json` health field reads false only because `--only E_` skips the 4-page variant it is computed on (health
  passed in 191147). Parity uses one row builder for both modes, so it cannot fail by construction.
- provisional: the evaluation feeds the long inputs it claims on the gating sets (arm A, gate passed; job 191147):
  truncation cut 0 / 1,500 and pisco/base parity 1,500 / 1,500 on every gating variant; health 646,042 = 646,042; the
  CML 256 controls fail as they should (924 / 1,500 and 275 / 1,500). Limits: on documents the cut check can only fail
  on prompt length; the audit's parity used one row builder for both modes. The gating sets are not long documents
  (owner, 2026-11-17) → arm E.
- provisional: on LongDocURL the shared 32-token answer cap made the uncompressed reader look worse than the compressed
  one (300 rows per set, same rows): uncompressed pages .340 → .717 at 256 tokens, +37.7 [+32.0, +43.0], sign 114 / 1
  (191634 → 191669); uncompressed text .423 → .580, +15.7 [+10.7, +20.7], 58 / 11 (191635 → 191670); R pages .517 →
  .547, +3.0 [+0.7, +5.3], 11 / 2 (191636 → 191673). Plumbing rows: this fixes the instrument, it decides no arm.
- provisional: arm J — the length-effect calculator recovers the tail model's one-chunk minus ten-chunk HotpotQA
  difference (183538, 1,500 pairs): −.1687 [−.1960, −.1407] (target −.169 ± .005), self-pair 0 (computed on the
  frontend with stdlib, no job backs it). Second judge prompt (job 191629, 1,500 pairs per number), Δ_len rubric /
  project / EM: documents 10 vs 4 pages +0.5 / +0.6 / +1.6 (s42), +0.1 / −0.3 / +0.9 (s43); 20 vs 4 pages s42 +4.3 /
  +5.1 / +7.7; text 40 vs 10 passages s42 −2.6 / −1.5 / +0.8. By the law's rule (same sign AND within 2 pt) all three
  are judge-sensitive: 10 pages s43 flips sign (every CI inside ±3, so the no-loss reading survives), 20 pages keeps its
  sign but EM is 3.4 pt larger, text 40 flips sign with EM.
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
  The gap SHRINKS (−19.9 → −15.5 pt) because the uncompressed reader falls more. Needs R s43 at 20 pages (hand-off
  19ac33 from campaign spread asks for the same cell).
- provisional (one seed, undecided, judge-sensitive): text 40 vs 10 passages s42, 1,500 pairs (R40 191392 .534, U40
  191393 .707, R10 .605 = campaign spread's re-run 191518 of model 184778, borrowed, not a cell of this grid; U10 .752):
  Δ_len −2.6 [−5.6, +0.3], p .084; project −1.5 [−4.6, +1.5], p .35; EM +0.8.
- provisional (plumbing, 300 rows, never decides): at 256 tokens on LongDocURL, R s42 − uncompressed = −17.0 pt
  [−23.0, −11.0] on pages (24 / 75; −13 / −15 / −23 at 10 / 20 / 40 pages, 100 Q each; 191673 vs 191669) and −16.3 pt
  [−21.7, −11.0] on text (15 / 64; −16 / −19 / −12 / −19 at 4k / 8k / 16k / 32k, 75 Q each; 191674 vs 191670).
- Running: closed-book 191776, text cap check at 512 191777, first mixed run 191778 (uncompressed, 300 rows).

## Arm E — the long-document yardstick
Source: LongDocURL (`dengchao/LongDocURL`, Apache-2.0). Builder `scripts/longdoc/longdoc_build.py` (md5 f14e4640) +
`build_long.sbatch`. Rows are interleaved by question, so `LIMIT=300` = 100 whole questions at every page size, or 75 at
every text size. `<MEM>` slots p50 per bucket: pages 1,198 / 2,392 / 4,795; text 540 / 1,046 / 2,055 / 4,079; mixed
977 / 1,922 / 3,821 (max 14,688). Uncompressed prompt max: pages 39,562; text 36,965; mixed 117,451.
Answer cap: 512 tokens for every arm E cell (CAMPAIGN.md). The text stats.json says 650 questions; 637 have rows.
Next, in order:
1. When 191776 / 191777 / 191778 end: health (n rows, judge on all, unparseable 0, no "exceeds" error, one attempt,
   node); text cut share at 512 (< 5% → 512 stays; else 1,024 for all and closed-book re-runs); mixed wall, MaxRSS
   and GPU memory (`sacct … MaxRSS`, `expmon occupancy` if needed) and its cut share; closed-book judge overall and per
   bucket (expand the 1,407 answers to each set's rows by the id before the last `__`; a stdlib pairing script under
   `scripts/longdoc/` is still to write; compare with the 300 uncompressed trial rows for a first read only).
2. Launch (cap 4): needles uncompressed on text, pages, mixed at 512 (soundness check 3; exact scoring, ≥ 90% per
   length); R s43 at 20 pages for arm L (hand-off 19ac33: `eval_mm_4b.sh expQ/longdoc/l_r43/model pisco ""
   eval_data/mm_docqa_1500_1m`, env XD=16 TOPK=20 PX=1003520 DML=2823, `--time 4:27` = 191377 wall 1:29 × 3, `--mem
   8G`; then `expmon handoff done 19ac33 …` with the file path); R mixed plumbing (300 rows, DML 15,672) after that.
3. Cost the model grid from the measured walls (full uncompressed cells on pages / text / mixed = soundness check 1's
   reference; R s42 / s43 on each; C256 vs L1024 at CML 1,024 on text, s42 / s43). Measured per 300 rows: pages ~31
   min, text 21-30 min → a full pages cell (4,221 rows) ≈ 7.3 h, text (2,246) ≈ 2.7-3.8 h. Rough total ~75 GPU-h and
   ~20 cells against ~35 GPU-h and 20 cells left → ONE BLOCKED.md question with the measured cost (law: never trim
   the grid to fit).

## Owned jobs
| job | arm | state | rows | note |
|---|---|---|---|---|
| 191776 | E closed-book, uncompressed `TOPK=0`, cap 512 (`--time 5:00`, 4G) | RUNNING (homer-011) | 1,407 | out `expQ/base_qwen35_4b/eval/mm_docqa_base_k0_px1003520_mnt512_longdoc_closedbook.json` |
| 191777 | E text cap check at 512, uncompressed (`--time 1:05`, 4G) | PENDING | 300 | `…_k1000_px1003520_cml1024_mnt512_longdoc_text_1500_lim300.json`; pairs with 191670 |
| 191778 | E first mixed run, uncompressed, cap 512 (`--time 2:00`, 8G) | PENDING | 300 | `…_k1000_px1003520_cml1024_mnt512_longdoc_mixed_1500_lim300.json`; measures the 117k-token prompts |
| 191774 | E closed-book set builder (CPU) | COMPLETED, verdict closed | 1,407 | |
| 191669 / 191670 / 191673 / 191674 | E cap check at 256 | COMPLETED, verdict closed this wake | 300 each | health in the verdicts |
| 191629 / 191634 / 191635 / 191636 | J, E | COMPLETED, verdict closed | | |
| 191517 | J re-judge (first try) | FAILED, verdict closed "launch fault" | — | replaced by 191629 |
| 191515 / 191516 / 191392 / 191393 / 191375-191378 / 191390 / 191391 / 191305-191308 / 191218-191221 / 191169-191172 / 191148-191155 / 191165-191168 / 191145-191147 | E, L, T, A | COMPLETED, verdict closed | | |

Cells used: 50 of 70. GPU-h: ≈ 24 of 60 (≈ 22 before this wake + the four 256-token cells ≈ 1.9); the three cells now
running add ≈ 2.5-3.5. CPU jobs count 0 GPU-h.

## Adopted jobs
(none — the other jobs in the queue are the expmon daemons 188764 / 162472 and campaign qafreeze's evals 191766-191769)

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| E long-document yardstick | sets built, input check PASS; cap fixed at 512; closed-book, text 512 check and mixed trial running | Arm E section, steps 1-3 |
| L length on a fixed model | docs 10 pages no loss (provisional, both seeds; judge-sensitive by the letter at s43); 20 pages one seed; text 40 one seed, undecided + judge-sensitive | R s43 at 20 pages next wake (hand-off 19ac33); text 40 s43 (XD=30 TOPK=40 CML=4096 DML=1545, `--time` 2:04 = 191392 wall 41 min × 3) after arm E's costing. Both on the shared launcher (gating sets, 32-token cap, as recorded) |
| J instrument | control 1 PASS (frontend, no job); second judge prompt done (191629) | re-run the project prompt on any new number that decides an arm |
| U reading many `<MEM>`s | not started | after E (owner); only where a length loss shows |
| P positions | not started | after E (owner); needs `--mem_position_offset` (patchable IF, only while no sibling eval runs) |
| A audit | DECIDED (provisional); E re-run PASS (191516) | — |
| T long-chunk training | CLOSED (provisional) | the text half moves to arm E (C256 vs L1024 at CML 1,024) |

## Tooling and controls
- `scripts/longdoc/eval_mnt.sbatch` (md5 1c874b32): arm E cells. `--env MNT=<N≠32>` required; STYLE=expand refused;
  writes the derived launcher to `expQ/longdoc/launchers/eval_mm_4b_mnt.<jobid>.sh` and logs it to
  `expQ/longdoc/eval_mnt_<jobid>.out`; exits 3 if `eval_mm_4b.sh` changed shape.
- `scripts/longdoc/build_closedbook.py` + `.sbatch` (md5 42758c8b): the closed-book set; exits 1 on a query mismatch.
- `scripts/longdoc/length_effect.py +A -B …` (judge default, `--metric em|match`, `--json`); tables in `expQ/longdoc/length/`.
- `scripts/longdoc/rejudge.sbatch NAME FILE…` (md5 9e2bc7d6): project-prompt judge on copies; requeue-safe; 18,000
  rows in 5 min 46 s. Pass the files as separate arguments (zsh: `F=(${=LIST})`, then `"${F[@]}"`).
- `audit_inputs.py --only E_`: the arm E variants only. Per-row output `expQ/longdoc/audit_191516/rows.jsonl` (keys `v`,
  `qid`, `n_mem`, `prompt`, `base_prompt`).
- Arm E launch shape: `expmon launch --exp longdoc --stage E --time … --mem … --exclude='homer-[005-009]' -p gpu-be
  --env MNT=512 --env PX=1003520 --env TOPK=1000 --env CML=1024 [--env DML=<pisco prompt max + 256>]
  scripts/longdoc/eval_mnt.sbatch expQ/longdoc/e_r4{2,3}/model pisco|base [LIMIT] /beegfs/scratch/user/hdejean/pisco/eval_data/longdoc_<set>`.
  DML per set: pages 5,612; text 5,232; mixed 15,672. Base mode ignores DML. Closed-book: `TOPK=0`, no CML, set
  `longdoc_closedbook`, LIMIT "" (an empty argument; check `len(argv)` = 4 in `.expmon/runs.jsonl`).
- Code md5 at launch of 191776-191778: eval_mm_4b.sh b3d1cda1 (wrapped), eval_mnt.sbatch 1c874b32, eval_mm_docqa.py
  dfe32aa0, judge_mm_docqa.py 02bc8748, judge_prompts.py bae44fa7, pisco/model.py 994cff3d.
- The guard refuses any command containing `rm -r`, even on a `/tmp` scratch dir: leave scratch dirs in /tmp.

## Blockers / needs the owner
- None now. Expected after the soundness cells: one BLOCKED.md question with the measured cost of arm E's model grid.
