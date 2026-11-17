# STATE — campaign longdoc   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-17 15:00 (eighth wake). NEXT: wake-when job:191375,191378 max 3h. Question: where do long inputs
lose accuracy (eval, training data, length, `<MEM>` reading, positions, long-chunk training)? Diagnosis only: inference
probes and CPU audits, no training.

## TL;DR — where this leaves the question
**Settled**
- Training never shows the decoder long inputs (a count, not yet a cause; job 191147): finetune rows reach at most 488
  `<MEM>`s and a position span of 4,017 (2,000 sampled rows). Out of that range: 10- and 20-page documents 1,500 /
  1,500, 40 text passages 1,498 / 1,500, 20 passages 35 / 1,500, whole documents 198 / 500; 4-page documents 0 / 1,500.
- provisional: the evaluation feeds the long inputs it claims (arm A, gate passed; job 191147): truncation cut 0 / 1,500
  and pisco/base parity 1,500 / 1,500 on every gating variant; health 646,042 = 646,042; the CML 256 controls fail as
  they should (924 / 1,500 and 275 / 1,500). Limits: on documents the cut check can only fail on prompt length; the
  audit's parity used one row builder for both modes.
- provisional: the length-effect calculator recovers the tail model's one-chunk minus ten-chunk HotpotQA difference
  (183538, 1,500 pairs): d = −.1687 [−.1960, −.1407], sign +121 / −374 (target −.169 ± .005), self-pair 0. Arm J
  control 1 of 2; control 2 (project prompt, EM) not run.
- provisional: arm T CLOSED — the long-chunk loss sits IN THE COMPRESSED IMAGE PAGE and enters in stage 2.
  - Documents `mm_docqa_1500_1m`, 1,003,520 px, 1,500 pairs per seed (tables `expQ/longdoc/locate/pair_*.txt`): gold
    page only L1024 − C256 −6.7 [−8.9, −4.5] McNemar 106 / 207 (s42; 191153 .568 vs 191152 .635), −4.9 [−7.1, −2.5]
    113 / 186 (s43; 191155 .579 vs 191154 .628). Picked page raw +0.1 [−0.8, +1.1] 29 / 27 (s42; 191166 vs 191165),
    −0.1 [−1.1, +0.9] 29 / 31 (s43; 191168 vs 191167).
  - Page transcription, stage-2 models (191219 / 191220 vs reference 191170; one stage-2 model per chain, no seed
    repeat): .566 vs .647 of the page's words, −8.0 pt [−10.6, −5.5], sign 147 / 322, p 4e-16, 500 pages scored (the
    table header says 501: one page was not scored). `expQ/longdoc/recall/pair_s2_c256_vs_l1024_n167.txt`.
  - Text autoencoding at 256 tokens, 500 held-out MS MARCO documents, nats per token, d = L1024 − C256: stage 1
    −.0018 [−.0031, −.0006], sign 271 / 229, p .067 (job 191218; both near ceiling, CI and sign test disagree);
    stage 2 +.0006 [−.0008, +.0019], sign 247 / 253, p .82 (job 191308). Neither model is worse on short text after
    either stage → the loss is specific to image pages. Controls: another document's slots 2.50 / 2.44 (stage 2),
    2.71 / 2.58 (stage 1); self-pair 0. Tables `expQ/longdoc/ae/ae_pair_n500.txt`, `ae_pair_s2_n500.txt`.
  - Text QA (`text_hotpotqa_1500`, CML 4,096, 1,500 pairs per seed): ten short chunks L1024 − C256 +2.3 [+0.3, +4.4]
    McNemar 136 / 101 (s42; 191305 .603 vs 191221 .579), −0.6 [−2.7, +1.3] 121 / 130 (s43; 191307 .589 vs 191306
    .595); one long chunk (g10, recorded) −1.1 [−3.2, +1.2] / −3.3 [−5.4, −1.1]. Difference of the two gaps +3.4
    [+0.5, +6.1] sign p .017 (s42), +2.7 [−0.3, +5.5] p .028 (s43): consistent with "L1024's text loss shows only on
    long chunks"; below the 3-pt bar at s43, so not decided. `expQ/longdoc/text10/`.
  - Training arm it motivates (owner): keep the 256-token stage 1; a long-chunk stage 1 needs its stage 2 changed and a
    page-transcription check before any finetune. Closes "longer stage-1 chunks help long inputs".
- The transcription-probe port is validated (job 191169): 360 / 360 legacy pages byte-identical, word recall 70.5% =
  70.5% (gate ±2 pt).

**Open, and running** (arm L, length on a fixed model; R = reference finetune 184778, seed 42, via `expQ/longdoc/l_r42`)
- 20-page documents, uncompressed, LIMIT=300 plumbing: job 191375, RUNNING on homer-011 (`--time 1:00`). Proves the
  19.5k-token prompt fits one A100 and measures the wall. Never decides.
- 20-page documents, uncompressed, full 1,500 Q: job 191376, PENDING (afterok 191375), `--time 4:54`.
- 20-page documents, R s42, full 1,500 Q: job 191377, PENDING (afterok 191375), `--time 5:30`, DML 2,823.
- 10-page documents, R s42, full 1,500 Q: job 191378, RUNNING on homer-017, `--time 2:45`, DML 1,578.
All four: XD (extra distractor pages) 16 or 6, TOPK 20 or 10, PX=1003520, `--exclude homer-[005-009]`, code md5
unchanged.

## Owned jobs
| job | arm | state | rows | note |
|---|---|---|---|---|
| 191375 | L plumbing, 20 pages, uncompressed | RUNNING | 300 Q | out `expQ/base_qwen35_4b/eval/mm_docqa_base_k20_px1003520_xd16_mm_docqa_1500_1m_lim300.json` |
| 191376 | L 20 pages, uncompressed | PENDING (dependency) | 1,500 Q | out `…/mm_docqa_base_k20_px1003520_xd16_mm_docqa_1500_1m.json` |
| 191377 | L 20 pages, R s42 | PENDING (dependency) | 1,500 Q | out `expQ/longdoc/l_r42/eval/mm_docqa_pisco_k20_px1003520_xd16_mm_docqa_1500_1m.json` |
| 191378 | L 10 pages, R s42 | RUNNING | 1,500 Q | out `expQ/longdoc/l_r42/eval/mm_docqa_pisco_k10_px1003520_xd6_mm_docqa_1500_1m.json` |
| 191305 / 191306 / 191307 / 191221 | T ten-chunk text | COMPLETED (191306 preempted twice, final attempt on homer-017) | 1,500 Q each | verdict closed |
| 191308 | T autoencoding on stage-2 models | first attempt COMPLETED 15:30 homer-017, requeue exited 3 (overwrite guard, benign) | 500 docs × 4 conditions × 2 models | verdict closed |
| 191218 / 191219 / 191220 / 191170 / 191169 | T probes | COMPLETED | | verdict closed |
| 191171 / 191172 | T plumbing | COMPLETED | | verdict closed |
| 191165-191168 / 191152-191155 / 191148-191151 | T document cells | COMPLETED | 1,500 Q each | verdict closed (191148-191151 off-regime) |
| 191147 / 191145 / 191146 | A audit, smokes | COMPLETED / CANCELLED / COMPLETED | | verdict closed |

Cells used: 31 of 70. GPU-h: tick sensor gpuh24 = 13.7 at 15:00 (whole campaign so far); the 4 new cells add ≈ 5.

## Adopted jobs
(none — the only other jobs are the expmon daemons 188764 / 162472)

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| A audit | DECIDED: gate PASS (provisional) | re-run once when the mixed set exists (arm L builds it) |
| T long-chunk training | CLOSED (provisional) | nothing; a critic pass re-derives it |
| L length on a fixed model | running, documents s42 | 1) when 191375 ends: health (300 rows, judge on every row, no "exceeds" error), node, wall, MaxRSS. If it failed: `expmon kill` 191376 / 191377 (they never start), diagnose. 2) when 191378 ends: health, then Δ_len at 10 pages = `length_effect.py +R10 -U10 -R4 +U4` (R4 = `expQ/ft_mm_4b_184778/eval/mm_docqa_pisco_px1003520_mm_docqa_1500_1m.json`, U4 = `expQ/base_qwen35_4b/eval/mm_docqa_base_px1003520_mm_docqa_1500_1m.json`, U10 = `…/mm_docqa_base_k10_px1003520_xd6_mm_docqa_1500_1m.json`). 3) launch the s43 repeats (symlink `expQ/longdoc/l_r43` ready): 10 pages (`--time 2:45 --mem 18G`, or the measured wall × 3), 20 pages (wall of 191377 unknown until it ends: use 5:30). 4) text: R s42 at 10 passages (TOPK=10 CML=4096), then 20 / 40 passages (XD 10 / 30) with DML = audit max + 256; uncompressed 20 / 40 passages are new files |
| U reading many `<MEM>`s | not started | after L's long size |
| P positions | not started | needs the `--mem_position_offset` flag (patchable IF); offset N from the audit spans (≈ 14,300) |
| J instrument | control 1 PASS (provisional) | after the first decided Δ_len: `rejudge.sbatch` (project prompt) on copies + EM, sign kept within 2 pt |

## Tooling and controls
- `scripts/longdoc/ae_nll.sbatch [N_DOCS] [SET]` (`s1` stage-1 models, `s2` stage-2 models), `ae_pair.py` (planted
  +0.30 recovered, self-pair 0), `mem_recall.py` + `mem_recall.sbatch` + `recall_pair.py`, `length_effect.py`.
- Base-mode cells take any PISCO checkpoint as `<CKPT>` (decoder name only) and always write to
  `expQ/base_qwen35_4b/eval/`; `DML` does not change the file name.
- Launch line shape used this wake: `expmon launch --exp longdoc --stage L --time … --mem … --exclude='homer-[005-009]'
  [--dependency=afterok:<id>] --env PX=1003520 --env XD=… --env TOPK=… --env DML=… eval_mm_4b.sh expQ/longdoc/l_r42/model
  pisco|base "" /beegfs/scratch/user/hdejean/pisco/eval_data/mm_docqa_1500_1m` (without `--exclude` the dry run shows
  `exclude=-`).

## Health evidence this wake
- 191305 / 191306 / 191307: COMPLETED exit 0 (homer-017 / homer-017 / homer-018), 1,500 rows, `judge` on 1,500,
  unparseable 0, no "exceeds" error, `mems_used` 265,880 = 191221's.
- 191308: `ae_nll_s2_n500.json` has 500 documents, 4 conditions, 2 models; written by the attempt whose batch step
  COMPLETED (known_issues.md, new row).
- Code md5 at 15:0x: eval_mm_4b.sh 50abc167, eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py 02bc8748, pisco/model.py
  994cff3d — unchanged. No longchunk cell queued or running.
- `expmon display --exp longdoc` again prints "0 table rows" while the page holds 32 `<tr>` rows (gripe 43db294f, +1).

## Blockers / needs the owner
- None.
