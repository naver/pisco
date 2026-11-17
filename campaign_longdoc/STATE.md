# STATE — campaign longdoc   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-17 22:00 (eleventh LLM wake; woken by the verifier's ALERT). NEXT: wake-when job:191516 max 10h.
Question: where do long inputs lose accuracy (eval, training data, length, `<MEM>` reading, positions, long-chunk
training)? Diagnosis only: inference probes and CPU audits, no training.

## This wake
- Verifier ALERT fixed: job 191378 (documents, 10 pages, R s42) had no verdict. Re-read its result file
  (`expQ/longdoc/l_r42/eval/mm_docqa_pisco_k10_px1003520_xd6_mm_docqa_1500_1m.json`): COMPLETED 0:0, homer-017, 50:42,
  1,500 rows, judge on all, unparseable 0, mems_used 1,614,940 = audit → `expmon verdict 191378 closed` recorded.
- Arm E build (job 191515) checked at 22:00: 1,407 usable questions on 387 PDFs (650 with Text/Layout evidence);
  evidence pages are 1-based (answer on page e−1: 351, on page e: 217); 21,843 pages rendered; the page-image set was
  being saved (4,221 rows = 1,407 × 3 windows). Text, mixed and needle sets still to build.
- Arm J judge-sensitivity cell launched (job 191517, `scripts/longdoc/rejudge.sbatch`, new this wake): the project
  (Yes/No) prompt on COPIES of the 8 files behind the 10-page and 20-page Δ_len tables. Copies in
  `expQ/longdoc/rejudge/docs_p10_p20/<dir>__<file>`; originals untouched.

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
  audit's parity used one row builder for both modes. BUT (owner, 2026-11-17) the gating sets are not long documents:
  no recorded number measures one long coherent document → arm E.
- provisional: the length-effect calculator recovers the tail model's one-chunk minus ten-chunk HotpotQA difference
  (183538, 1,500 pairs): judge d = −.1687 [−.1960, −.1407], sign +121 / −374 (target −.169 ± .005), EM d −.1087,
  self-pair 0. Arm J control 1 of 2; the project-prompt re-judge is job 191517 (queued).
- provisional: arm T CLOSED — the long-chunk (L1024) loss on documents sits IN THE COMPRESSED IMAGE PAGE (gate met at
  both seeds). Where it enters is an inference: stage-1 models are text-only; the image-page loss was measured on
  stage-2 models only, one model per chain.
  - Documents `mm_docqa_1500_1m`, 1,003,520 px, 1,500 pairs per seed (`expQ/longdoc/locate/pair_*.txt`): gold page
    only L1024 − C256 −6.7 [−8.9, −4.5] McNemar 106 / 207 (s42; 191153 .568 vs 191152 .635), −4.9 [−7.1, −2.5]
    113 / 186 (s43; 191155 .579 vs 191154 .628). Picked page raw +0.1 [−0.8, +1.1] 29 / 27 (s42; 191166 vs 191165),
    −0.1 [−1.1, +0.9] 29 / 31 (s43; 191168 vs 191167).
  - Page transcription, stage-2 models (191219 / 191220 vs reference 191170; one stage-2 model per chain): .566 vs
    .647 of the page's words, −8.0 pt [−10.6, −5.5], sign 147 / 322, p 4e-16, 500 pages scored of 501 paired.
  - Text autoencoding at 256 tokens, 500 held-out MS MARCO documents, nats per token, d = L1024 − C256: stage 1
    −.0018 [−.0031, −.0006], sign 271 / 229, p .067 (191218); stage 2 +.0006 [−.0008, +.0019], 247 / 253, p .82
    (191308). Consistent with an image-specific loss (a null with no stated margin, not a proof).
  - Text QA (`text_hotpotqa_1500`, CML 4,096, 1,500 pairs per seed): ten short chunks L1024 − C256 +2.3 [+0.3, +4.4]
    McNemar 136 / 101 (s42; 191305 .603 vs 191221 .579), −0.6 [−2.7, +1.3] 121 / 130 (s43; 191307 .589 vs 191306
    .595): the seeds disagree. One long chunk (g10, recorded) −1.1 [−3.2, +1.2] / −3.3 [−5.4, −1.1]. Not decided;
    the long-chunk question on text moves to arm E (real long documents, CML 1,024).
  - Proposed to the owner (a recommendation, untested by any arm): keep the 256-token stage 1; if a long-chunk stage 1
    is retried, vary stage 2 from the L1024 start and check page transcription before any finetune.
- provisional: documents (`mm_docqa_1500_1m`, distractor pages from OTHER documents), 10 vs 4 pages — no length loss
  at either seed: Δ_len +0.5 [−1.8, +2.7], sign 144 / 138, p .77 (s42, job 191378); +0.1 [−2.2, +2.5], 146 / 142,
  p .86 (s43, job 191390); 1,500 pairs each. EM +1.6 / +0.9 (same sign). R .607 / .608 at 10 pages vs .610 / .615 at
  4; uncompressed .801 vs .809. Leaves provisional only after the project-prompt re-judge (job 191517).
  Tables `expQ/longdoc/length/dlen_docs_p10_s4{2,3}_{judge,em,match}.json`.

**Open, and running**
- provisional (one seed): documents 20 vs 4 pages, s42 (jobs 191377 R .599, 191376 uncompressed .754): Δ_len +4.3
  [+1.8, +6.9], sign 197 / 146, p .007, 1,500 pairs; EM +7.7 [+4.9, +10.4], match +4.6. The gap SHRINKS (−19.9 → −15.5
  pt) because the uncompressed reader falls more. The EM Δ_len is 3.4 pt above the judge's, so the judge-sensitivity
  check (job 191517) matters here too. Needs R s43 at 20 pages (not launched: arm E first).
- Text 40 passages, R s42 (job 191392, preempted 21:29 and requeued; PENDING for Priority since 21:34, est. start
  11-18 13:15) and uncompressed (job 191393, RUNNING since 21:41). Plumbing 191391 passed. Then Δ_len text 40 vs 10
  passages s42 (`+R40 -U40 -R10 +U10`; R10 =
  `expQ/freeze/ref_s42/eval/mm_docqa_pisco_k10_px1003520_cml4096_text_hotpotqa_1500.json` .603, U10 =
  `expQ/base_qwen35_4b/eval/mm_docqa_base_k10_text_hotpotqa_1500.json` .752).
- Arm E build (job 191515, CPU) → cut check (job 191516, `audit.sbatch 0 E_`, afterok).
- Arm J re-judge (job 191517, GPU, PENDING for Priority, est. start 11-18 16:00).

## Arm E — the long-document yardstick
Sources checked 2026-11-17 (HF API + GitHub): **LongDocURL** (`dengchao/LongDocURL`, Apache-2.0; 2,325 Q on 396 PDFs of
40-150 pages; evidence pages annotated) — USED. The build measured 1,407 usable questions on 387 PDFs after the
≤ 6-per-PDF cap (650 with Text/Layout evidence only), so the sets hold 1,407 Q (pages, mixed) and 650 Q (text), not
1,500. QuALITY (CC BY 4.0; 2k-8k-token articles) and Qasper (CC BY 4.0; papers ~3k-10k tokens) cannot fill the 16k /
32k buckets. NarrativeQA (Apache-2.0) reaches 32k but has no evidence location and invites memory answers. LooGLE
(CC BY-SA 4.0): only ≥ 24k-word documents. MMLongBench-Doc (CC BY-NC 4.0, research only): held in reserve. LongBench
MultiFieldQA stays reported-only.
Construction (`scripts/longdoc/longdoc_build.py`, md5 f14e4640; wrappers `build_long_{text,pages,mixed}.py`;
launcher `build_long.sbatch`): for each question, NESTED windows of consecutive pages of its own PDF that contain every
evidence page, so each question is asked at every length (paired Δ_len as in arm L). Rows interleaved by question:
`LIMIT=300` = 100 whole questions (pages, mixed) or 75 (text) at every bucket.
- `eval_data/longdoc_pages_1500`: page images at ≤ 1,003,520 px, windows of 10 / 20 / 40 pages (4,221 rows).
  Bucket in `source` (`longdocurl_p10|p20|p40`) and in the `query_id` suffix.
- `eval_data/longdoc_text_1500`: the PDF text layer, one doc per page (a page over 1,000 tokens split into pieces, so
  CML 1,024 keeps it whole), windows grown to ≥ 4k / 8k / 16k / 32k tokens; `stats.json` records the answer-in-text share.
- `eval_data/longdoc_mixed_1500`: the pages windows; each page is its text layer or its image (p = .5, fixed per page).
- `eval_data/longdoc_needle_{text,pages,mixed}`: a synthetic "access code" page at 10 / 50 / 90% depth of a real
  window, 25 PDFs × buckets × 3 depths, exact scoring (`match`).
Next, in order: read `stats.json` + build log (rows per bucket, answer-in-text share) and the cut check (cut 0% and
parity 100% on E_text / E_pages / E_mixed at CML 1,024; DML = prompt max + 256); ledger `resource:` notes for the sets;
then plumbing `LIMIT=300` cells: uncompressed on pages (`TOPK=1000`, measures the 40-page wall and memory) and on text
(32k tokens), R s42 on pages; then soundness: closed-book uncompressed (`TOPK=0`; `gen_base` builds a question-only
prompt, `scripts/eval_mm_docqa.py:531-551`) per set, needle uncompressed per modality. Then cost the model grid from
those walls (prior from arm L walls: R 50 / 89 min per 1,500 Q at 10 / 20 pages → ~5 GPU-h per full pages cell; full
grid likely ~40 GPU-h, above the ~37 left) → expect ONE BLOCKED.md question with the measured cost.

## Owned jobs
| job | arm | state | rows | note |
|---|---|---|---|---|
| 191515 | E build (CPU, 3 CPUs, 32G) | RUNNING (started 21:41) | — | log `expQ/longdoc/build_long_191515.out`; source cache `eval_data/longdoc_src/` |
| 191516 | E cut check (CPU, `audit.sbatch 0 E_`) | PENDING afterok 191515 | all rows | out `expQ/longdoc/audit_191516/audit.json` |
| 191517 | J re-judge, project prompt (GPU, 24G, `--time 1:00`) | PENDING (Priority) | 8 × 1,500 | log `expQ/longdoc/rejudge_191517.out`; copies `expQ/longdoc/rejudge/docs_p10_p20/` |
| 191392 | L text 40 passages, R s42 | PENDING (requeued after preemption 21:29) | 1,500 Q | out `…/l_r42/eval/mm_docqa_pisco_k40_px1003520_cml4096_xd30_text_hotpotqa_1500.json`; health: mems_used = audit 1,075,610 ±1% |
| 191393 | L text 40 passages, uncompressed | RUNNING since 21:41 | 1,500 Q | out `expQ/base_qwen35_4b/eval/mm_docqa_base_k40_xd30_text_hotpotqa_1500.json` |
| 191375 / 191376 / 191377 / 191378 / 191390 / 191391 | L | COMPLETED, verdict closed (191378 recorded this wake) | | health in the verdicts |
| 191305-191308 / 191218-191221 / 191169-191172 / 191148-191155 / 191165-191168 / 191145-191147 | T, A | COMPLETED | | verdict closed |

Cells used: 38 of 70 (191517 added). GPU-h: tick sensor gpuh24 = 18.6 at 22:00 (≈ the whole campaign); 191392 / 191393
add ≈ 3, 191517 ≤ 1; the two arm E jobs are CPU (0 GPU-h).

## Adopted jobs
(none — the only other jobs are the expmon daemons 188764 / 162472 and the spread / qafreeze campaigns' cells)

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| E long-document yardstick | building | stats + cut check → plumbing → soundness → cost → BLOCKED question if over budget |
| L length on a fixed model | 10 pages decided (provisional); 20 pages one seed; text 40 running | 191392 / 191393 end → health + Δ_len text 40 vs 10. If 191392 is still PENDING for Priority after 03:34 (6 h), move it to `gpu,gpu-be` (pre-authorised; ≤ 2 cells on `gpu`, count qafreeze's finetune against the 8-GPU cap). R s43 at 20 pages after arm E's costing (XD=16 TOPK=20 PX=1003520 DML=2823, `--time` 4:27 = 191377 wall 1:29 × 3, `--mem 32G`) |
| J instrument | control 1 PASS (provisional); re-judge queued (191517) | when 191517 ends: `length_effect.py` on the copies, same terms as `dlen_docs_p10_s4{2,3}` and `dlen_docs_p20_s42` → the 10-page result leaves provisional if Δ_len keeps its sign or stays within 2 pt at both seeds; record the measured judge wall for later `--time`s |
| U reading many `<MEM>`s | not started | after E (owner); only where a length loss shows |
| P positions | not started | after E (owner); needs `--mem_position_offset` (patchable IF) |
| A audit | DECIDED (provisional) | re-run on the E sets = job 191516 |
| T long-chunk training | CLOSED (provisional) | the text half moves to arm E (C256 vs L1024 at CML 1,024) |

## Tooling and controls
- `scripts/longdoc/length_effect.py +A -B …` (judge default, `--metric em|match`, `--json`); tables in `expQ/longdoc/length/`.
- `scripts/longdoc/rejudge.sbatch NAME FILE…` (new, md5 9e2bc7d6): project-prompt judge on copies; requeue-safe.
- `audit_inputs.py --only E_`: the arm E variants only (TOPK 1,000; CML 1,024 gating, CML 256 and needles reported).
- Launch line shape (arm L): `expmon launch --exp longdoc --stage L --time … --mem … --exclude='homer-[005-009]'
  --env PX=1003520 --env XD=… --env TOPK=… [--env CML=4096] --env DML=… eval_mm_4b.sh expQ/longdoc/l_r4{2,3}/model
  pisco|base "" <abs path of set>`. Base cells write to `expQ/base_qwen35_4b/eval/` (check the file name is new).
- Code md5 now: eval_mm_4b.sh b3d1cda1 (changed 16:54 by a sibling: 50abc167 + requeue markers + DENSE off, same
  python call; cells 191376 / 191377 / 191390 / 191392 / 191393 were submitted before it, so SLURM runs the 50abc167
  copy), eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py 02bc8748, judge_prompts.py bae44fa7, pisco/model.py 994cff3d.

## Blockers / needs the owner
- None now. Expected after the plumbing cells: one BLOCKED.md question with the measured cost of arm E's model grid.
