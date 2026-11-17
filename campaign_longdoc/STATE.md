# STATE — campaign longdoc   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-17 16:00 (ninth wake). NEXT: wake-when job:191376,191377,191390,191392,191393 max 4h. Question:
where do long inputs lose accuracy (eval, training data, length, `<MEM>` reading, positions, long-chunk training)?
Diagnosis only: inference probes and CPU audits, no training.

## TL;DR — where this leaves the question
**Settled**
- Training never shows the decoder long inputs (a count, not yet a cause; job 191147): finetune rows reach at most 488
  `<MEM>`s and a position span of 4,017 (2,000 sampled rows). Out of that range: 10- and 20-page documents 1,500 /
  1,500, 40 text passages 1,498 / 1,500, 20 passages 35 / 1,500, whole documents 198 / 500; 4-page documents 0 / 1,500.
- The transcription-probe port is validated (job 191169): 360 / 360 legacy pages byte-identical, word recall 70.5% =
  70.5% (gate ±2 pt).
- provisional: the evaluation feeds the long inputs it claims (arm A, gate passed; job 191147): truncation cut 0 / 1,500
  and pisco/base parity 1,500 / 1,500 on every gating variant; health 646,042 = 646,042; the CML 256 controls fail as
  they should (924 / 1,500 and 275 / 1,500). Limits: on documents the cut check can only fail on prompt length; the
  audit's parity used one row builder for both modes.
- provisional: the length-effect calculator recovers the tail model's one-chunk minus ten-chunk HotpotQA difference
  (183538, 1,500 pairs): judge d = −.1687 [−.1960, −.1407], sign +121 / −374 (target −.169 ± .005), EM d −.1087,
  self-pair 0. Arm J control 1 of 2; the project-prompt re-judge has not run.
- provisional: arm T CLOSED — the long-chunk (L1024) loss on documents sits IN THE COMPRESSED IMAGE PAGE (gate met at
  both seeds). Where it enters is an inference, not a measurement: stage-1 models are text-only, so no image probe
  exists before stage 2; the image-page loss was measured on stage-2 models only, one model per chain.
  - Documents `mm_docqa_1500_1m`, 1,003,520 px, 1,500 pairs per seed (tables `expQ/longdoc/locate/pair_*.txt`): gold
    page only L1024 − C256 −6.7 [−8.9, −4.5] McNemar 106 / 207 (s42; 191153 .568 vs 191152 .635), −4.9 [−7.1, −2.5]
    113 / 186 (s43; 191155 .579 vs 191154 .628). Picked page raw +0.1 [−0.8, +1.1] 29 / 27 (s42; 191166 vs 191165),
    −0.1 [−1.1, +0.9] 29 / 31 (s43; 191168 vs 191167).
  - Page transcription, stage-2 models (191219 / 191220 vs reference 191170; one stage-2 model per chain, so no seed
    repeat is possible): .566 vs .647 of the page's words, −8.0 pt [−10.6, −5.5], sign 147 / 322, p 4e-16, 500 pages
    scored of 501 paired. `expQ/longdoc/recall/pair_s2_c256_vs_l1024_n167.txt` (167 pages per source).
  - Text autoencoding at 256 tokens, 500 held-out MS MARCO documents, nats per token, d = L1024 − C256: stage 1
    −.0018 [−.0031, −.0006], sign 271 / 229, p .067 (job 191218); stage 2 +.0006 [−.0008, +.0019], sign 247 / 253,
    p .82 (job 191308). L1024 is not worse on short text after either stage: consistent with an image-specific loss
    (a null with no stated margin, not a proof). Controls: another document's slots 2.50 / 2.44 (stage 2), 2.71 / 2.58
    (stage 1); self-pair 0. Tables `expQ/longdoc/ae/ae_pair_n500.txt`, `ae_pair_s2_n500.txt`.
  - Text QA (`text_hotpotqa_1500`, CML 4,096, 1,500 pairs per seed): ten short chunks L1024 − C256 +2.3 [+0.3, +4.4]
    McNemar 136 / 101 (s42; 191305 .603 vs 191221 .579), −0.6 [−2.7, +1.3] 121 / 130 (s43; 191307 .589 vs 191306
    .595): the seeds disagree. One long chunk (g10, recorded) −1.1 [−3.2, +1.2] / −3.3 [−5.4, −1.1]. Difference of the
    two gaps +3.4 [+0.5, +6.1] sign p .017 (s42), +2.7 [−0.3, +5.5] p .028 (s43): consistent with "L1024's text loss
    shows only on long chunks"; below the 3-pt bar at s43, not decided. `expQ/longdoc/text10/`.
  - Proposed to the owner (a recommendation, untested by any arm): keep the 256-token stage 1; if a long-chunk stage 1
    is retried, vary stage 2 from the L1024 start and check page transcription before any finetune. No arm here ran the
    L1024 models on 10-20 pages or 20-40 passages.

**Open, and running** (arm L, length on a fixed model; R = reference finetune 184778 s42 via `expQ/longdoc/l_r42`,
184780 s43 via `l_r43`; uncompressed = Qwen3.5-4B on the same rows)
- provisional (one seed): documents, 10 vs 4 pages, s42 (job 191378): Δ_len +0.5 [−1.8, +2.7], sign 144 / 138, p .77,
  1,500 pairs; EM +1.6 [−0.8, +3.9], match +1.2 [−0.9, +3.3]. R .607 vs .610 at 4 pages, uncompressed .801 vs .809:
  the gap stays at about −19.5 pt. CI inside ±3 pt → on track to "no length loss at 10 pages"; needs s43.
- Documents 10 pages, R s43, 1,500 Q: job 191390, RUNNING (`--time 2:32`).
- Documents 20 pages, R s42 and uncompressed, 1,500 Q each: jobs 191377 / 191376, RUNNING since 15:23 (670 / 810 of
  1,500 rows at 16:00). The uncompressed 20-page plumbing (300 Q, job 191375) passed in 17:21.
- Text 40 passages (XD=30 TOPK=40 CML=4096), R s42: plumbing 300 Q job 191391 RUNNING; full cells R s42 191392 and
  uncompressed 191393 PENDING (afterok 191391). Short size = recorded R files at 10 passages: s42 .603, s43 .593
  (`expQ/freeze/ref_s4{2,3}/eval/mm_docqa_pisco_k10_px1003520_cml4096_text_hotpotqa_1500.json`, same code md5,
  `<MEM>` total 265,880 = audit); uncompressed 10 passages .752 (`mm_docqa_base_k10_text_hotpotqa_1500.json`).

## Owned jobs
| job | arm | state | rows | note |
|---|---|---|---|---|
| 191376 | L 20 pages, uncompressed | RUNNING homer-011 | 1,500 Q | out `expQ/base_qwen35_4b/eval/mm_docqa_base_k20_px1003520_xd16_mm_docqa_1500_1m.json` |
| 191377 | L 20 pages, R s42 | RUNNING homer-017 | 1,500 Q | out `expQ/longdoc/l_r42/eval/mm_docqa_pisco_k20_px1003520_xd16_mm_docqa_1500_1m.json`; health: mems_used = audit 3,229,884 |
| 191390 | L 10 pages, R s43 | RUNNING homer-017 | 1,500 Q | out `expQ/longdoc/l_r43/eval/mm_docqa_pisco_k10_px1003520_xd6_mm_docqa_1500_1m.json`; health: mems_used = 1,614,940 |
| 191391 | L text 40 passages plumbing, R s42 | RUNNING homer-017 | 300 Q | out `…/l_r42/eval/mm_docqa_pisco_k40_px1003520_cml4096_xd30_text_hotpotqa_1500_lim300.json`; distractors drawn from the first 300 rows only (LIMIT selects first), so mems_used is a range check (audit p50 713 / max 1,035) |
| 191392 | L text 40 passages, R s42 | PENDING afterok 191391 | 1,500 Q | out `…/l_r42/eval/mm_docqa_pisco_k40_px1003520_cml4096_xd30_text_hotpotqa_1500.json`; health: mems_used = audit 1,075,610 ±1% |
| 191393 | L text 40 passages, uncompressed | PENDING afterok 191391 | 1,500 Q | out `expQ/base_qwen35_4b/eval/mm_docqa_base_k40_xd30_text_hotpotqa_1500.json` (no PX/CML, matching the recorded base 10-passage file) |
| 191378 | L 10 pages, R s42 | COMPLETED homer-017, one attempt, 50:42 | 1,500 Q | healthy (judge 1,500, unparseable 0, mems_used 1,614,940 = audit); probe judge .6067, Δ_len +.0047 |
| 191375 | L plumbing 20 pages, uncompressed | COMPLETED homer-011, 17:21 | 300 Q | verdict closed (plumbing OK) |
| 191305-191308 / 191218-191221 / 191169-191172 | T probes and text cells | COMPLETED | | verdict closed |
| 191148-191155 / 191165-191168 | T document cells | COMPLETED | 1,500 Q each | verdict closed (191148-191151 off-regime) |
| 191145-191147 | A audit, smokes | COMPLETED / CANCELLED | | verdict closed |

Cells used: 35 of 70. GPU-h: tick sensor gpuh24 = 14.6 at 16:00 (whole campaign so far); the 4 new cells add ≈ 3.

## Adopted jobs
(none — the only other jobs are the expmon daemons 188764 / 162472)

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| A audit | DECIDED: gate PASS (provisional) | re-run once when the mixed set exists |
| T long-chunk training | CLOSED (provisional) | nothing; a critic pass re-derives it |
| L length on a fixed model | running | 1) 191390 ends: health, then Δ_len s43 10 pages = `length_effect.py +R10s43 -U10 -R4s43 +U4` (R4s43 = `expQ/ft_mm_4b_184780/eval/mm_docqa_pisco_px1003520_mm_docqa_1500_1m.json`, .615). Both seeds inside ±3 → verdict "no length loss at 10 pages". 2) 191376 / 191377 end: health, Δ_len s42 at 20 pages; launch R s43 20 pages (XD=16 TOPK=20 PX=1003520 DML=2823, `--time` = 191377 wall × 3, `--mem 32G`). 3) 191391: 300 rows, judge on all, no "exceeds" error, mems per row within the audit's range; 191392 / 191393 end: health, Δ_len text 40 vs 10 passages s42 (`+R40 -U40 -R10 +U10`). 4) then text 20 passages (XD=10 TOPK=20 CML=4096 PX=1003520 DML=1038; base new file `mm_docqa_base_k20_xd10_text_hotpotqa_1500.json`) and R s43 at 40 passages. 5) mixed set (`scripts/longdoc/build_mixed.py`, to write) |
| U reading many `<MEM>`s | not started | after L's long size; only worth it where L shows a loss |
| P positions | not started | needs the `--mem_position_offset` flag (patchable IF); offset N from the audit spans (≈ 14,300) |
| J instrument | control 1 PASS (provisional) | after the first decided Δ_len: `rejudge.sbatch` (project prompt, to write) on copies; EM already computed for every Δ_len |

## Tooling and controls
- `scripts/longdoc/length_effect.py +A -B …` (judge default, `--metric em|match`, `--json`); Δ_len tables in
  `expQ/longdoc/length/`. `ae_nll.sbatch`, `ae_pair.py`, `mem_recall.py` + `.sbatch` + `recall_pair.py`.
- `--extra_distractors` (env XD) reaches text rows: `with_distractors` runs before the mode split with no modality
  test (`scripts/eval_mm_docqa.py:709-714`); text distractors carry image None, and `gen_base` renders a None image as
  text (`:531-541`). `gen_base` ignores `--decoder_max_length`; the pisco path raises on it.
- Launch line shape: `expmon launch --exp longdoc --stage L --time … --mem … --exclude='homer-[005-009]'
  [--dependency=afterok:<id>] --env PX=1003520 --env XD=… --env TOPK=… [--env CML=4096] --env DML=… eval_mm_4b.sh
  expQ/longdoc/l_r4{2,3}/model pisco|base "" <abs path of set>`. Base cells always write to
  `expQ/base_qwen35_4b/eval/`; check the file name is new.

## Health evidence this wake
- 191378: COMPLETED exit 0, one attempt (homer-017, 15:06-15:56), 1,500 rows, judge on 1,500, unparseable 0, no
  "exceeds" error, mems_used 1,614,940 = audit `docs_p10` total.
- 191375: COMPLETED exit 0 (homer-011, 17:21), 300 rows, judge on 300, unparseable 0.
- Code md5 at 16:0x: eval_mm_4b.sh 50abc167, eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py 02bc8748, pisco/model.py
  994cff3d — unchanged. No longchunk cell queued or running.
- `expmon display --exp longdoc` again prints "0 table rows" (gripe 43db294f, already +1).

## Blockers / needs the owner
- None.
