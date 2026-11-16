# STATE — campaign longdoc   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-17 02:00 (third wake). Question: where do long inputs lose accuracy (eval, training data, length,
`<MEM>` reading, positions, long-chunk training)? Diagnosis only: inference probes and CPU audits, no training.

## TL;DR — where this leaves the question
**Settled**
- Training never shows the decoder long inputs (a count, not yet a cause; job 191147, verifier: supported): finetune
  rows reach at most 488 `<MEM>`s and a position span of 4,017 (2,000 sampled rows). Out of that range: 10- and 20-page
  documents 1,500 / 1,500, 40 text passages 1,498 / 1,500, 20 passages 35 / 1,500, whole documents 198 / 500; 4-page
  documents 0 / 1,500 (max 485 `<MEM>`s, span 3,962).
- provisional: the evaluation feeds the long inputs it claims (arm A, gate passed; job 191147, verdict closed): truncation
  cut 0 / 1,500 and pisco/base parity 1,500 / 1,500 on every gating variant (documents 4 / 10 / 20 pages, text 10 / 20 /
  40 passages); health 646,042 = 646,042; the CML 256 controls do fail (924 / 1,500 and 275 / 1,500). Limits the
  verifier raised: on documents the cut check can only fail on prompt length (images are never cut); the audit's parity
  used one row builder for both modes. Independent re-check this wake on recorded runs: pages per question equal in both
  modes on 1,500 / 1,500 (4 pages, 184778 vs base), 1,500 / 1,500 (10 pages, 173550 vs base), 500 / 500 (whole
  documents, 174908 vs base) — a page count, page identity is not recorded. The text gold-only "cut" (16 / 1,500) is
  definitional (answer only in removed distractors; known_issues.md), and gold-only is not a gating variant in the law.
- provisional: the length-effect calculator recovers the tail model's one-chunk minus ten-chunk HotpotQA difference
  (183538, 1,500 pairs): d = −.1687 [−.1960, −.1407], sign test +121 / −374 (target −.169 ± .005), self-pair 0
  (stdlib on recorded files, no job; ledger 2026-11-17 00:56). Arm J control 1 of 2.

**Open, and running**
- Arm T (why long-chunk training lost): gold page only (distractor pages removed), documents, 1,500 Q, PX=1003520, on
  the four finetunes C256 s42 / L1024 s42 / C256 s43 / L1024 s43 = jobs 191152 / 191153 / 191154 / 191155, RUNNING since
  ~02:05 (~18 min generation + judge each, `--time 0:55`). These replace 191148-191151, which ran at the launcher's
  default 451,584 px because the previous wake omitted `--env PX=1003520` (verdict closed "off-regime").
- provisional, off-regime (451,584 px, NOT gating): gold page only, L1024 − C256 = −6.3 [−8.6, −3.8] McNemar 119 / 213
  (s42, 191149 vs 191148) and −4.7 [−7.0, −2.3] 128 / 198 (s43, 191151 vs 191150), n = 1,500 pairs each, every source
  negative. Plain 4-page cells at 1,003,520 px: −6.2 [−8.6, −3.9] (s42, 190271 vs 188708), −4.3 [−6.7, −1.8] (s43, 190274
  vs 188711). Consistent with the deficit sitting in the compressed page, at two pixel budgets.
- Gate (law, both seeds): IN THE COMPRESSED PAGE if the gold-only deficit stays ≤ −3 pt (CI below 0) AND it shrinks
  inside ±2 pt with the picked page raw (`RAW=route1`); IN MULTI-PAGE READING if it shrinks inside ±2 pt with gold only.

## Owned jobs
| job | arm | state | rows | eta | note |
|---|---|---|---|---|---|
| 191152 | T gold-only, C256 s42, 1,003,520 px | RUNNING | 1,500 Q | ~02:25 | out `expQ/longdoc/t_c256_s42/eval/mm_docqa_pisco_px1003520_gold_mm_docqa_1500_1m.json` |
| 191153 | T gold-only, L1024 s42, 1,003,520 px | RUNNING | 1,500 Q | ~02:25 | `expQ/longdoc/t_l1024_s42/eval/` same name |
| 191154 | T gold-only, C256 s43, 1,003,520 px | RUNNING | 1,500 Q | ~02:25 | `expQ/longdoc/t_c256_s43/eval/` same name |
| 191155 | T gold-only, L1024 s43, 1,003,520 px | RUNNING | 1,500 Q | ~02:25 | `expQ/longdoc/t_l1024_s43/eval/` same name |
| 191148-191151 | T gold-only at 451,584 px | COMPLETED 17:24-17:53, homer-017 | 1,500 Q each | | verdict closed off-regime; judge .528 / .465 / .530 / .483 (probe recorded); files `…/mm_docqa_pisco_gold_mm_docqa_1500_1m.json` |
| 191147 | A audit (full) | COMPLETED 7:10 | 20,000 eval + 8,000 training | | verdict closed (gate PASS, provisional) |
| 191146 | A audit smoke | COMPLETED 1:58 | 168 + 48 | | verdict closed (plumbing) |
| 191145 | A audit smoke | CANCELLED before start | | | verdict closed (16 CPUs never free) |

Cells used: 11 of 70 (3 audit jobs, 8 eval cells). GPU-h used: ~1.2 (four 18-min cells); the 4 running add ~1.2.

## Adopted jobs
(none — the only other jobs are the expmon daemons 188764 / 162472)

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| A audit | DECIDED: gate PASS (provisional, caveats above) | re-run once when the mixed set exists (arm L builds it) |
| T long-chunk training | gold-only at 1,003,520 px running | pair L1024 vs C256 per seed (`scripts/paired_judge.py` on the `_px1003520_gold` files). If the deficit stays ≤ −3 pt at both seeds: launch `RAW=route1` on the 4 finetunes with `--env RAW=route1 --env PX=1003520` (`--time 1:34:00` = 31:09 of 188441 × 3). If it shrinks inside ±2 pt: verdict "multi-page reading", skip route1. Then plain 10-chunk text (TOPK=10 CML=4096) on the 4 finetunes; then write `mem_recall.py` / `ae_nll.py` |
| J instrument | control 1 PASS (provisional) | after the first decided Δ_len: `rejudge.sbatch` (project prompt) on copies + EM, sign kept within 2 pt |
| L length on a fixed model | not started | DML per size = audit max + 256 (documents 10 pages pisco 1,578, base 9,983; 20 pages pisco 2,823, base 19,564); one `LIMIT=300` plumbing cell per new variant (XD on text rows unverified); every document cell carries `--env PX=1003520` |
| U reading many `<MEM>`s | not started | after L's long size |
| P positions | not started | needs the `--mem_position_offset` flag (patchable IF); offset N from the audit spans (documents 20 pages p50 18,027 − 4 pages p50 3,720 ≈ 14,300) |

## Health evidence this wake
- Code md5 (pairing key) unchanged at launch: eval_mm_4b.sh 50abc167, eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py
  02bc8748, pisco/model.py 994cff3d — identical to longchunk's plain cells.
- 191148-191151: n = 1,500, a judge on every row, `judge_unparseable` 0, homer-017 (allowed); summary
  `image_max_pixels` 451,584 — the only defect. Run rows of 191152-191155 carry `GOLD=1 PX=1003520` (grid PX column).
- No longchunk cell is queued or running.
- The four 02:02 intent lines in the ledger for 191152-191155 are garbled (a zsh loop did not split words); the 02:03
  lines replace them.

## Blockers / needs the owner
- None. The push ALERT of 01:17 was stale: after a fetch, origin/arc = HEAD = 5c2075f (0 unpushed); ALERT deleted,
  gripe +1 on 4442771e (the watchdog keeps a push ALERT after a later push heals it).
