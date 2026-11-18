# STATE — campaign longdoc   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-18 16:30 (twelfth LLM wake; woken by the owner's GATE touch + job 191516 ending; the owner set
CAMP_PACE=0, "this constraint is gone", so the weekly-pace hold that kept this loop idle ~14 h is off).
NEXT: wake-when job:191629,191634,191635,191636 max 4h.
Question: where do long inputs lose accuracy (eval, training data, length, `<MEM>` reading, positions, long-chunk
training)? Diagnosis only: inference probes and CPU audits, no training.

## This wake
- Arm E sets built and checked (jobs 191515 / 191516, both verdict closed): see Settled. Ledger `resource:` note written.
- Arm L text 40 vs 10 passages s42: both cells healthy (191392 / 191393, verdicts closed) → Δ_len computed (Open).
- Arm J re-judge 191517 FAILED in 0 s: launch fault (zsh passed the 8 paths as one argument; known_issues row added,
  gripe +1 052cca8b). Relaunched as 191629 with 12 files (the 8 document files + the 4 text files behind the new
  text Δ_len), argv checked = 13 elements. Copies go to `expQ/longdoc/rejudge/dlen_docs_text/`.
- Arm E plumbing launched (LIMIT=300 each, all RUNNING by 16:40): uncompressed pages 191634, uncompressed text
  191635, R s42 pages 191636 (model dir `expQ/longdoc/e_r42/model` → 184778). Cap 4 reached.
- `docs/longdoc_diagnosis.md` rewritten in place (text length table, arm E audit rows, figure still pending).

## TL;DR — where this leaves the question
**Settled**
- Training never shows the decoder long inputs (a count, not yet a cause; job 191147): finetune rows reach at most 488
  `<MEM>`s and a position span of 4,017 (2,000 sampled rows). Out of that range: 10- and 20-page documents 1,500 /
  1,500, 40 text passages 1,498 / 1,500, 20 passages 35 / 1,500, whole documents 198 / 500; 4-page documents 0 / 1,500.
  Every arm E bucket is out of range too (job 191516 rows: pages 1,407 / 1,407 at 10 / 20 / 40 pages; text 637 / 637,
  610 / 610, 588 / 588, 411 / 411; mixed 1,398 / 1,407 at 10 pages, all at 20 / 40).
- The transcription-probe port is validated (job 191169): 360 / 360 legacy pages byte-identical, word recall 70.5% =
  70.5% (gate ±2 pt).
- Arm E sets exist (job 191515): LongDocURL (Apache-2.0), 1,407 usable questions on 387 PDFs; pages and mixed 4,221 rows
  each (1,407 × 10 / 20 / 40 pages of the question's own PDF), text 2,246 rows (t4k 637 / t8k 610 / t16k 588 / t32k
  411; 650 text-evidence questions), needles text 300 / pages 225 / mixed 225.
- Arm E input check PASS (job 191516, every row, CML 1,024): cut 0 and parity 1.0000 on text (2,246), pages (4,221),
  mixed (4,221) and the three needle sets; the CML 256 control cuts 235 / 2,246 text rows (10.5%), as it should.
  Its health number is null by design (`--only E_` skips the 4-page variant; health passed in 191147).
- provisional: the evaluation feeds the long inputs it claims (arm A, gate passed; job 191147): truncation cut 0 / 1,500
  and pisco/base parity 1,500 / 1,500 on every gating variant; health 646,042 = 646,042; the CML 256 controls fail as
  they should (924 / 1,500 and 275 / 1,500). Limits: on documents the cut check can only fail on prompt length; the
  audit's parity used one row builder for both modes. BUT (owner, 2026-11-17) the gating sets are not long documents
  → arm E.
- provisional: the length-effect calculator recovers the tail model's one-chunk minus ten-chunk HotpotQA difference
  (183538, 1,500 pairs): judge d = −.1687 [−.1960, −.1407], sign +121 / −374 (target −.169 ± .005), EM d −.1087,
  self-pair 0. Arm J control 1 of 2 (computed on the frontend, no job); the project-prompt re-judge is job 191629.
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
  4; uncompressed .801 vs .809. Leaves provisional only after the project-prompt re-judge (job 191629).
  Tables `expQ/longdoc/length/dlen_docs_p10_s4{2,3}_{judge,em,match}.json`.

**Open, and running**
- provisional (one seed): documents 20 vs 4 pages, s42 (jobs 191377 R .599, 191376 uncompressed .754): Δ_len +4.3
  [+1.8, +6.9], sign 197 / 146, p .007, 1,500 pairs; EM +7.7 [+4.9, +10.4], match +4.6. The gap SHRINKS (−19.9 → −15.5
  pt) because the uncompressed reader falls more. EM is 3.4 pt above the judge (> the 2-pt limit) → re-judge 191629.
  Needs R s43 at 20 pages (not launched: arm E first).
- provisional (one seed, undecided): text 40 vs 10 passages, s42, 1,500 pairs (R40 191392 .534, U40 191393 .707,
  R10 = 191518 .605 = the spread campaign's today's-code re-run of 184778, U10 `expQ/base_qwen35_4b/eval/
  mm_docqa_base_k10_text_hotpotqa_1500.json` .752): Δ_len −2.6 [−5.6, +0.3], sign 199 / 236, p .084. Neither bar met
  (loss needs the CI below 0; no-loss needs it inside ±3). EM +0.8 [−1.7, +3.3], match +1.5: opposite sign to the judge
  → judge-sensitive candidate until 191629 scores the project prompt. Gap −14.7 → −17.3. With the older R10 file
  (188615; 1,442 / 1,500 predictions identical to 191518): −2.4 [−5.3, +0.5]. Limit: U10 dates from 2026-10-28 (older
  code; base mode). Tables `expQ/longdoc/length/dlen_text_n40_s42_{judge,em,match}.json`.
- Arm J re-judge 191629 (project prompt on copies of 12 files) RUNNING since ~16:38.
- Arm E plumbing 191634 / 191635 / 191636 RUNNING since ~16:38 (see Owned jobs).

## Arm E — the long-document yardstick
Source: LongDocURL (`dengchao/LongDocURL`, Apache-2.0; 2,325 Q on 396 PDFs). Other candidates and why they were not
used: QuALITY / Qasper too short for 16k / 32k; NarrativeQA has no evidence location; LooGLE only ≥ 24k words;
MMLongBench-Doc CC BY-NC (reserve). Builder `scripts/longdoc/longdoc_build.py` (md5 f14e4640) + `build_long.sbatch`.
Rows are interleaved by question, so `LIMIT=300` = 100 whole questions at every page size, or 75 at every text size.
Per bucket (job 191516 rows): `<MEM>` slots p50 pages 1,198 / 2,392 / 4,795; text 540 / 1,046 / 2,055 / 4,079; mixed
977 / 1,922 / 3,821 (max 14,688). Uncompressed prompt max: pages 39,562; text 36,965; mixed 117,451 (one outlier PDF
with a huge text layer; p50 at 40 pages 30,649). The answer string appears verbatim in the text layer on 326 / 637
(t4k) … 231 / 411 (t32k) text rows.
Next, in order:
1. When 191634 / 191635 / 191636 end: health (n = 300, judge on all, unparseable 0, no "exceeds" error; R mems_used vs
   the audit rows of the same 300 query_ids), measured wall and MaxRSS per bucket → `--time` / `--mem` for full cells.
2. Soundness cells (6, full sets, uncompressed): closed-book (`TOPK=0`) on pages, text, mixed; needles uncompressed on
   text, pages, mixed. Plus plumbing for mixed (uncompressed + R; the 117k-token prompt needs its memory measured).
3. Cost the model grid (R s42 / s43 on pages, text, mixed; C256 vs L1024 at CML 1,024 on text, s42 / s43) from the
   measured walls. Prior from arm L walls: ~5 GPU-h per full pages cell; grid likely ~40 GPU-h, above what is left
   (~37) → expect ONE BLOCKED.md question with the measured cost (law: never trim the grid to fit).

## Owned jobs
| job | arm | state | rows | note |
|---|---|---|---|---|
| 191629 | J re-judge, project prompt (GPU, 12G, `--time 1:00`) | RUNNING | 12 × 1,500 | log `expQ/longdoc/rejudge_191629.out`; copies `expQ/longdoc/rejudge/dlen_docs_text/<dir>__<file>`; judge-only wall unmeasured: record it |
| 191634 | E plumbing, uncompressed, pages (`--time 1:30`, 13G) | RUNNING | 300 | out `expQ/base_qwen35_4b/eval/mm_docqa_base_k1000_px1003520_cml1024_longdoc_pages_1500_lim300.json`; measures the 40-page wall + GPU memory |
| 191635 | E plumbing, uncompressed, text (`--time 1:00`, 13G) | RUNNING | 300 | out `expQ/base_qwen35_4b/eval/mm_docqa_base_k1000_px1003520_cml1024_longdoc_text_1500_lim300.json`; measures 32k tokens |
| 191636 | E plumbing, R s42, pages (`--time 1:05`, 13G, DML 5,612) | RUNNING | 300 | out `expQ/longdoc/e_r42/eval/mm_docqa_pisco_k1000_px1003520_cml1024_longdoc_pages_1500_lim300.json` |
| 191517 | J re-judge (first try) | FAILED, verdict closed "launch fault" | — | replaced by 191629 |
| 191392 / 191393 / 191515 / 191516 | L, E | COMPLETED, verdict closed this wake | | health in the verdicts |
| 191375-191378 / 191390 / 191391 / 191305-191308 / 191218-191221 / 191169-191172 / 191148-191155 / 191165-191168 / 191145-191147 | L, T, A | COMPLETED, verdict closed | | |

Cells used: 42 of 70. GPU-h: ≈ 21 of 60 (18.9 at 2026-11-17 22:00 per the verifier + 191392's four attempts ≈ 1.9 +
191393 0.45); the four cells now running add ≈ 2-4. Arm E build / cut check are CPU (0 GPU-h).
Efficiency note (brief): the 14.5 h with nothing running was the weekly-pace hold (now off, CAMP_PACE=0) plus the
failed re-judge; the 5.5 h queued was 191392 / 191517 pending for Priority on gpu-be. This wake has 4 cells in flight
at once; next wake chains soundness cells behind measured plumbing.

## Adopted jobs
(none — the other jobs in the queue are the expmon daemons 188764 / 162472 and the spread / qafreeze campaigns' cells)

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| E long-document yardstick | sets built, input check PASS; plumbing running | health + walls → soundness (6) + mixed plumbing → cost → BLOCKED question if over budget |
| L length on a fixed model | docs 10 pages no loss (provisional, both seeds); 20 pages one seed; text 40 one seed, undecided | after 191629: re-read the three Δ_len under the project prompt. R s43 at 20 pages and text 40 s43 after arm E's costing (20 pages: XD=16 TOPK=20 PX=1003520 DML=2823, `--time` 4:27 = 191377 wall 1:29 × 3, `--mem 8G` = MaxRSS 4.25 GB × 1.5; text 40: XD=30 TOPK=40 CML=4096 DML=1545, `--time` 2:04 = 191392 attempt-4 wall 41 min × 3) |
| J instrument | control 1 PASS (provisional); re-judge running (191629) | when it ends: `length_effect.py` on the copies with the same terms as `dlen_docs_p10_s4{2,3}`, `dlen_docs_p20_s42`, `dlen_text_n40_s42` → a Δ_len leaves provisional if it keeps its sign and stays within 2 pt; else the arm is judge-sensitive (an evaluation bottleneck). Record the judge-only wall |
| U reading many `<MEM>`s | not started | after E (owner); only where a length loss shows |
| P positions | not started | after E (owner); needs `--mem_position_offset` (patchable IF) |
| A audit | DECIDED (provisional); E re-run PASS (191516) | — |
| T long-chunk training | CLOSED (provisional) | the text half moves to arm E (C256 vs L1024 at CML 1,024) |

## Tooling and controls
- `scripts/longdoc/length_effect.py +A -B …` (judge default, `--metric em|match`, `--json`); tables in `expQ/longdoc/length/`.
- `scripts/longdoc/rejudge.sbatch NAME FILE…` (md5 9e2bc7d6): project-prompt judge on copies; requeue-safe. Pass the
  files as separate arguments (zsh: `F=(${=LIST})`, then `"${F[@]}"`; check `len(argv)` in runs.jsonl).
- `audit_inputs.py --only E_`: the arm E variants only (TOPK 1,000; CML 1,024 gating, CML 256 and needles reported).
- Arm E launch shape: `expmon launch --exp longdoc --stage E --time … --mem … --exclude='homer-[005-009]' --env PX=1003520
  --env TOPK=1000 --env CML=1024 [--env DML=<pisco prompt max + 256>] eval_mm_4b.sh expQ/longdoc/e_r4{2,3}/model
  pisco|base [LIMIT] /beegfs/scratch/user/hdejean/pisco/eval_data/longdoc_<set>`. Base mode ignores DML (gen_base
  never checks it). DML per set: pages 5,612; text 5,232; mixed 15,672.
- Code md5 now = at launch of 191629-191636: eval_mm_4b.sh b3d1cda1, eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py
  02bc8748, judge_prompts.py bae44fa7, pisco/model.py 994cff3d. (191392 / 191393 ran the eval_mm_4b.sh 50abc167 copy
  taken at submission; same python call.)

## Blockers / needs the owner
- None now. Expected after the plumbing and soundness cells: one BLOCKED.md question with the measured cost of arm E's
  model grid.
