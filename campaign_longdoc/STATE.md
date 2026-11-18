# STATE — campaign longdoc   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-19 02:00 (sixteenth LLM wake; woken because the four arm E trial jobs ended and the 23:30 wake had
failed on the account's 5-hour usage limit, which reset at 02:00).
NEXT: wake-when job:192007,192008 max 8h. BLOCKED.md holds ONE owner question (arm E model-cell budget).
Question: where do long inputs lose accuracy (eval, training data, length, `<MEM>` reading, positions, long-chunk
training)? Diagnosis only: inference probes and CPU audits, no training.

## This wake
- 191877 / 191878 / 191879 / 191887 healthy (COMPLETED 0:0, one attempt each, homer-016 / 016 / 017 / 016; rows = set
  size, judge on all, unparseable 0, no "exceeds" error). Verdicts closed, probes set.
- Needle checks (arm E soundness check 3): PASS in every bucket (TL;DR).
- First compressed mixed run (R s42, 300 rows): `mems_used` 715,709 = the audit's prediction on 300 / 300 rows; wall
  31:19, MaxRSS 4.4 GB, decoder prompt max 11,739. Gap vs uncompressed in the TL;DR (plumbing).
- Cap-count correction: `expQ/ft_mm_4b_184778/model/tokenizer.json` pads to a multiple of 64 and truncates at 1,024, so
  the 22:00 "exact" counts were rounded up. Padding off: uncompressed text at 512 7 / 300 = 2.3% (was 8 / 300), mixed
  7 / 300, closed-book 5 / 1,407; at 256 text 24 / 300 = 8.0% (was 11.0%), pages 8 / 300 = 2.7%. The cap stays 512.
  CAMPAIGN.md bracket, known_issues row 38, `docs/longdoc_diagnosis.md` and the ledger corrected. The verdict texts of
  191777 / 191778 still say 8 / 300 (the ledger note of this wake supersedes them).
- Launched (2 of cap 4): arm L seed 43 — R s43 documents at 20 pages (192007) and text at 40 passages (192008), the
  exact rows of 191377 / 191392 with `l_r43`. No uncompressed pages trial at 512: its wall is estimated from the text
  ratio 256 → 512 (21:20 → 23:26, ×1.10), to keep cells for the grid.
- Cost of arm E's model cells: ≈ 73 GPU-h in 13 cells (26 when split per length bucket against preemption) vs ≈ 32 GPU-h
  and 14 cells left → BLOCKED.md, recommendation +50 GPU-h, +16 cells.
- Cleared the usage-limit ALERT (known_issues row 23: resolved by the 02:00 reset; this wake ran).
- Gripes: +1 af99bb93 (paired compare by hand, per bucket); new gap (no per-bucket view of summary keys in grid/results).
- Efficiency lines of the brief: idle 71% = the usage-limit hold (4 h) + waiting on four trial jobs that ran 11-31 min;
  the next large step is the owner's budget answer, so nothing more was ready to launch. Moving the loop to another
  login is the owner's (`cc-team`); not mine to change.

## TL;DR — where this leaves the question
**Settled**
- provisional: training does not show the decoder long inputs (a count, not yet a cause). In a 2,000-row sample of
  finetune rows (job 191147) the maximum is 488 `<MEM>`s and a position span of 4,017 (a sample bound, not a proven
  maximum). Beyond that range (191147): 10- and 20-page documents 1,500 / 1,500, 40 text passages 1,498 / 1,500, 20
  passages 35 / 1,500; 4-page documents 0 / 1,500. Arm E rows above 488 `<MEM>`s (job 191516,
  `expQ/longdoc/audit_191516/out_of_range.json`): text 2,246 / 2,246, pages 4,221 / 4,221, mixed 4,212 / 4,221.
- The transcription-probe port is validated (job 191169): 360 / 360 legacy pages byte-identical, word recall 70.5% =
  70.5% (gate ±2 pt).
- Arm E sets exist (job 191515): LongDocURL (Apache-2.0), 1,407 usable questions on 387 PDFs; pages and mixed 4,221 rows
  each (1,407 × 10 / 20 / 40 pages of the question's own PDF), text 2,246 rows (t4k 637 / t8k 610 / t16k 588 / t32k
  411), needles text 300 / pages 225 / mixed 225. Closed-book set: 1,407 questions (job 191774).
- provisional: the arm E answer cap is 512 tokens (job 191777): 7 / 300 uncompressed text answers reach it (2.3%, bar
  5%, tokenizer count with padding off); mixed 7 / 300 (191778), closed-book 5 / 1,407 (191776), R mixed 4 / 300
  (191887). The token-count rule was written after 191777 was seen; by characters the same file gives 6.3%.
- provisional: arm E soundness check 3 (needle) PASS in every bucket, uncompressed, cap 512, 25 documents × 3 depths per
  length, code string found in the answer: text 75 / 75 at ~4k / 8k / 16k / 32k tokens (191877); pages 75 / 75, 73 / 75
  (97.3%), 74 / 75 (98.7%) at 10 / 20 / 40 pages (191878); mixed 75 / 75, 75 / 75, 74 / 75 at 10 / 20 / 40 items
  (191879); bar 90%. No answer lists more than one code. EM is .00-.53 only because the base decoder answers in a sentence.
- provisional: arm E input check PASS (job 191516, every row, CML 1,024): cut 0 and parity 1.0000 on text (2,246), pages
  (4,221), mixed (4,221) and the three needle sets; the CML 256 control cuts 235 / 2,246 text rows (10.5%). Parity uses
  one row builder for both modes, so it cannot fail by construction.
- provisional: the evaluation feeds the long inputs it claims on the gating sets (arm A, gate passed; job 191147):
  truncation cut 0 / 1,500 and pisco/base parity 1,500 / 1,500 on every gating variant; health 646,042 = 646,042; the
  CML 256 controls fail as they should (924 / 1,500 and 275 / 1,500). Parity cannot fail by construction. The gating
  sets are not long documents (owner, 2026-11-17) → arm E.
- provisional (plumbing rows, fixes the instrument, decides no arm): on LongDocURL the shared 32-token answer cap cut
  the uncompressed reader (300 rows per set, same rows): uncompressed pages .340 → .717 at 256 tokens, +37.7 [+32.0,
  +43.0], sign 114 / 1 (191634 → 191669), so at 32 tokens it scored below R on pages (.340 vs .517, 191636);
  uncompressed text .423 → .580, +15.7 [+10.7, +20.7], 58 / 11 (191635 → 191670); R pages .517 → .547, +3.0 [+0.7,
  +5.3], 11 / 2 (191636 → 191673).
- provisional (first read, trial rows; the full uncompressed cells decide): arm E soundness check 1 — the document is
  needed. Closed-book judge .164 on all 1,407 questions (191776, cap 512). With the document − closed-book, same
  questions (`expQ/longdoc/soundness/closedbook_trial300.json`): text +33 / +33 / +33 / +32 at ~4k / 8k / 16k / 32k
  tokens (75 Q each, 191777, cap 512); mixed +44 / +47 / +48 (100 Q each, 191778, cap 512); lowest CI bound +17; bar 20
  pt. Pages +52 / +52 / +54 is INDICATIVE only: its with-document file ran at cap 256 (191669) and the law pairs cells
  only at the same cap.
- provisional (plumbing, 75 Q per bucket, no paired test between buckets): the uncompressed reader shows no drop with
  length on LongDocURL text at cap 512 (191777): .587 / .587 / .587 / .573 from ~4.5k to ~32.7k prompt tokens; mixed
  .63 / .66 / .67 at 10 / 20 / 40 items (191778).
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
  / −0.3 [−2.7, +2.1]. R .607 / .608 at 10 pages vs .610 / .615 at 4; uncompressed .801 (existing base file) vs .809.

**Open, and running**
- provisional (one seed, judge-sensitive): documents 20 vs 4 pages, s42 (191377 R .599, 191376 uncompressed .754):
  Δ_len +4.3 [+1.8, +6.9], sign 197 / 146, p .007, 1,500 pairs; project prompt +5.1 [+2.6, +7.7], p .0009; EM +7.7.
  The gap SHRINKS (−19.9 → −15.5 pt) because the uncompressed reader falls more. Seed 43 running: 192007.
- provisional (one seed, undecided, judge-sensitive): text 40 vs 10 passages s42, 1,500 pairs (R40 191392 .534, U40
  191393 .707, R10 .605 = campaign spread's re-run 191518 of model 184778, borrowed, not a cell of this grid; U10 .752):
  Δ_len −2.6 [−5.6, +0.3], p .084; project −1.5 [−4.6, +1.5], p .35; EM +0.8. Seed 43 running: 192008.
- provisional (plumbing, 300 rows, never decides): R s42 − uncompressed on LongDocURL. Pages at cap 256: −17.0 pt
  [−23.0, −11.0] (24 / 75; −13 / −15 / −23 at 10 / 20 / 40 pages, 100 Q each; 191673 vs 191669). Text at cap 256:
  −16.3 [−21.7, −11.0] (15 / 64; −16 / −19 / −12 / −19 at 4k / 8k / 16k / 32k, 75 Q each; 191674 vs 191670). Mixed at
  cap 512: −15.7 [−22.0, −9.3] (McNemar 31 / 78; −12 / −14 / −21 at 10 / 20 / 40 items, 100 Q each; 191887 vs 191778).
- Waiting on the owner: the arm E model-cell budget (BLOCKED.md).

## Arm E — the long-document yardstick
Source: LongDocURL (`dengchao/LongDocURL`, Apache-2.0). Builder `scripts/longdoc/longdoc_build.py` (md5 f14e4640) +
`build_long.sbatch`. Rows are interleaved by question, so `LIMIT=300` = 100 whole questions at every page size, or 75 at
every text size. `<MEM>` slots p50 per bucket: pages 1,198 / 2,392 / 4,795; text 540 / 1,046 / 2,055 / 4,079; mixed
977 / 1,922 / 3,821 (max 14,688). Uncompressed prompt max: pages 39,562; text 36,965; mixed 117,451 (measured so far up
to 89,311 on one GPU, 191778). Soundness: checks 2 (input) and 3 (needle) PASS; check 1 passes on the trial rows.
Measured walls per 300 judged rows: uncompressed text 23:26 (cap 512), mixed 43:47 (512), pages 31:01 (256); R pages
31:05 (256), text 30:22 (256), mixed 31:19 (512). MaxRSS ≤ 4.4 GB everywhere.
Next, in order, once the owner answers BLOCKED.md:
1. Option 1 or 2: one CPU job writes per-bucket sets `eval_data/longdoc_{pages,mixed}_1500_{p10,p20,p40}` (the same rows,
   split by the `__` suffix of the query id); then the uncompressed cells (pages 3 bucket cells, text 1, mixed 3), then R
   s42, R s43 (symlink `expQ/longdoc/e_r43/model → expQ/ft_mm_4b_184780/model` first), then C256 / L1024 at CML 1,024
   on text. ≤ 4 submissions per wake, chained with `--dependency afterok` where it saves a wake.
2. Each full cell: health; soundness check 1 per bucket with `closedbook_pair.py` (same cap 512); gap and Δ_len per
   bucket with `length_effect.py`; the project-prompt re-judge on any number that decides.

## Owned jobs
| job | arm | state | rows | note |
|---|---|---|---|---|
| 192007 | L, R s43 documents 20 pages (`--time 4:27`, 7G) | PENDING | 1,500 | out `expQ/longdoc/l_r43/eval/mm_docqa_pisco_k20_px1003520_xd16_mm_docqa_1500_1m.json`; pairs with 191376 and R s43 4 pages (`expQ/ft_mm_4b_184780/eval/mm_docqa_pisco_px1003520_mm_docqa_1500_1m.json`) |
| 192008 | L, R s43 text 40 passages (`--time 2:04`, 7G) | PENDING | 1,500 | out `expQ/longdoc/l_r43/eval/mm_docqa_pisco_k40_px1003520_xd30_cml4096_text_hotpotqa_1500.json`; pairs with 191393 and R s43 ten passages `expQ/spread/b_spread_s43/eval/mm_docqa_pisco_k10_cml4096_text_hotpotqa_1500.json` (spread's 191630, borrowed) |
| 191877 / 191878 / 191879 / 191887 | E needles text / pages / mixed; R s42 mixed trial | COMPLETED, verdict closed this wake | 300 / 225 / 225 / 300 | health in the verdicts |
| 191776 / 191777 / 191778 / 191774 / 191669 / 191670 / 191673 / 191674 / 191629 / 191634-191636 | E, J | COMPLETED, verdict closed | | |
| 191517 | J re-judge (first try) | FAILED, verdict closed "launch fault" | — | replaced by 191629 |
| 191515 / 191516 / 191392 / 191393 / 191375-191378 / 191390 / 191391 / 191305-191308 / 191218-191221 / 191169-191172 / 191148-191155 / 191165-191168 / 191145-191147 | E, L, T, A | COMPLETED, verdict closed | | |

Cells used: 56 of 70 (54 + 192007 / 192008). GPU-h: 25.8 of 60 by sacct over the 54 ended jobs; the two running cells
add ≈ 2.2. CPU jobs count 0 GPU-h.
Code md5 at launch of 192007 / 192008: eval_mm_4b.sh b3d1cda1, eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py 02bc8748,
judge_prompts.py bae44fa7, pisco/model.py 994cff3d. The s42 arm L cells ran eval_mm_4b.sh 50abc167; b3d1cda1 adds only
the requeue resume (R documents reproduced 1,500 / 1,500 on it, 191495 / 191496): the one listed difference in the s43
pairs.

## Adopted jobs
(none — the other jobs in the queue are the expmon daemons 188764 / 162472 and campaign slotcheck's 192002)

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| E long-document yardstick | sets built; soundness checks 2 and 3 PASS, check 1 passes on the trial rows; cap 512 | owner answer on the budget (BLOCKED.md), then the Arm E section |
| L length on a fixed model | docs 10 pages no loss (provisional, both seeds); 20 pages and text 40 one seed | read 192007 / 192008 → Δ_len at s43 with `length_effect.py`, re-judge with the project prompt (`rejudge.sbatch`) |
| J instrument | control 1 PASS (frontend, no job); second judge prompt done (191629) | re-run the project prompt on any new number that decides an arm |
| U reading many `<MEM>`s | not started | after E (owner); only where a length loss shows |
| P positions | not started | after E (owner); needs `--mem_position_offset` (patchable IF, only while no sibling eval runs) |
| A audit | DECIDED (provisional); E re-run PASS (191516) | — |
| T long-chunk training | CLOSED (provisional) | the text half is in arm E's grid (C256 vs L1024 at CML 1,024) |

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
- Needle scoring: summary key `match/needle_<set>_<len>_d<depth>` (code string in the answer); per bucket by grouping
  records on `source`.
- Cap count: `.pixi/envs/default/bin/python -I` + `t = tokenizers.Tokenizer.from_file("expQ/ft_mm_4b_184778/model/tokenizer.json")`,
  then `t.no_padding(); t.no_truncation()` (the file pads to a multiple of 64), answer at the cap = `len(ids) >= cap - 2`.
- Arm E launch shape: `expmon launch --exp longdoc --stage E --time … --mem … --exclude='homer-[005-009]' -p gpu-be
  --env MNT=512 --env PX=1003520 --env TOPK=1000 --env CML=1024 [--env DML=<pisco prompt max + 256>]
  scripts/longdoc/eval_mnt.sbatch expQ/longdoc/e_r4{2,3}/model pisco|base [LIMIT|''] /beegfs/scratch/user/hdejean/pisco/eval_data/longdoc_<set>`.
  DML per set: pages 5,612; text 5,232; mixed 15,672. Base mode ignores DML. Check `len(argv)` = 4 in
  `.expmon/runs.jsonl` after each launch.
- The guard refuses any command containing `rm -r`, even on a `/tmp` scratch dir: leave scratch dirs in /tmp.

## Blockers / needs the owner
- BLOCKED.md: approve ≈ +50 GPU-hours (60 → 110) and +16 cells (70 → 86) for arm E's model cells (options 1-4 there;
  recommendation option 1).
