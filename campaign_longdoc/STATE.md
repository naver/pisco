# STATE — campaign longdoc   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-17 01:30 (second wake). Question: where do long inputs lose accuracy (eval, training data, length,
`<MEM>` reading, positions, long-chunk training)? Diagnosis only: inference probes and CPU audits, no training.

## TL;DR — where this leaves the question
**Settled**
- The evaluation feeds the long inputs it claims (arm A, feasibility gate PASSED; job 191147, verdict closed): on
  1,500 questions per variant (14 variants; whole documents 500), the gold answer is never cut by truncation
  (0 / 1,500 on every gating variant: documents 4 / 10 / 20 pages, text 10 / 20 / 40 passages) and the compressed and
  uncompressed modes get the same documents in the same order on 1,500 / 1,500. Health: R's 4-page `<MEM>` total
  646,042 = the recorded 646,042. The check does fail when it should: with chunks cut at 256 tokens, 924 / 1,500
  (joined HotpotQA) and 275 / 1,500 (MRQA) lose the answer.
- Training never shows the decoder long inputs (a count, not yet a cause; job 191147): finetune rows reach at most
  488 `<MEM>`s and a position span of 4,017 (2,000 sampled rows). Out of that range: 10- and 20-page documents
  1,500 / 1,500, 40 text passages 1,498 / 1,500, 20 passages 35 / 1,500, whole documents 198 / 500; 4-page documents
  0 / 1,500 (max 485 `<MEM>`s, span 3,962). Stage 1: C256 max 66 `<MEM>`s (rows at the 512-token decoder cap on the
  median), L1024 max 258; stage 2 max 132.
- provisional: the length-effect calculator recovers the tail model's one-chunk minus ten-chunk HotpotQA difference
  (183538, 1,500 pairs): d = −.1687 [−.1960, −.1407], sign test +121 / −374 (target −.169 ± .005), self-pair 0
  (stdlib on recorded files, no job; ledger 2026-11-17 00:56). Arm J control 1 of 2.

**Open, and running**
- Arm T (why long-chunk training lost), first probe: gold page only (distractor pages removed) on documents, 1,500 Q,
  for the four finetunes C256 s42 / L1024 s42 / C256 s43 / L1024 s43 = jobs 191148 / 191149 / 191150 / 191151, all
  RUNNING on homer-017 since ~01:33, ~18 min generation + judge each (`--time 0:55`). Gate: L1024 − C256 stays
  ≤ −3 pt (CI below 0) → the deficit sits in the compressed page; shrinks inside ±2 pt → it sits in reading
  several pages. Plain pairs (all 4 pages) exist: L1024 − C256 −6.2 (s42) / −4.3 (s43) pt.

## Owned jobs
| job | arm | state | rows | eta | note |
|---|---|---|---|---|---|
| 191148 | T gold-only, C256 s42 | RUNNING homer-017 | 1,500 Q | ~02:15 | out `expQ/longdoc/t_c256_s42/eval/mm_docqa_pisco_px1003520_gold_mm_docqa_1500_1m.json` |
| 191149 | T gold-only, L1024 s42 | RUNNING homer-017 | 1,500 Q | ~02:15 | `expQ/longdoc/t_l1024_s42/eval/` |
| 191150 | T gold-only, C256 s43 | RUNNING homer-017 | 1,500 Q | ~02:15 | `expQ/longdoc/t_c256_s43/eval/` |
| 191151 | T gold-only, L1024 s43 | RUNNING homer-017 | 1,500 Q | ~02:15 | `expQ/longdoc/t_l1024_s43/eval/` |
| 191147 | A audit (full) | COMPLETED 7:10 | 20,000 eval + 8,000 training | | verdict closed (gate PASS) |
| 191146 | A audit smoke | COMPLETED 1:58 | 168 + 48 | | verdict closed (plumbing) |
| 191145 | A audit smoke | CANCELLED before start | | | verdict closed (16 CPUs never free) |

Cells used: 7 of 70 (3 audit jobs, 4 eval cells). GPU-h used: 0 so far (audits are CPU); the 4 evals will add ~2.

## Adopted jobs
(none — the only other jobs are the expmon daemons 188764 / 162472)

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| A audit | DECIDED: gate PASS | re-run once when the mixed set exists (arm L builds it) |
| T long-chunk training | gold-only cells running | pair L1024 vs C256 per seed with `scripts/paired_judge.py` on the `_gold` files; plain files `expQ/longchunk/ft_cml{256,1024}_s4{2,3}/eval/mm_docqa_pisco_px1003520_mm_docqa_1500_1m.json`. If the deficit stays ≤ −3 pt: launch `RAW=route1` on the 4 finetunes (`--time 1:34:00` = 31:09 of 188441 × 3). Then plain 10-chunk text (TOPK=10 CML=4096) on the 4 finetunes; then write `mem_recall.py` / `ae_nll.py` |
| J instrument | control 1 PASS (provisional) | after the first decided Δ_len: `rejudge.sbatch` (project prompt) on copies + EM, sign kept within 2 pt |
| L length on a fixed model | not started | DML per size = audit max + 256 (documents 10 pages pisco 1,578, base 9,983; 20 pages pisco 2,823, base 19,564); one `LIMIT=300` plumbing cell per new variant (XD on text rows unverified) |
| U reading many `<MEM>`s | not started | after L's long size |
| P positions | not started | needs the `--mem_position_offset` flag (patchable IF); offset N from the audit spans (documents 20 pages p50 18,027 − 4 pages p50 3,720 ≈ 14,300) |

## Health evidence this wake
- Code md5 (pairing key) unchanged: eval_mm_4b.sh 50abc167, eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py 02bc8748,
  pisco/model.py 994cff3d — identical to longchunk's plain cells, so the gold-only cells pair with them.
- The text gold-only variant shows a 16 / 1,500 "cut": on all 16 the answer is only in distractor passages, which
  gold-only removes by design (no truncation on any of them). Recorded in known_issues.md; not a gating variant.
- No longchunk cell is queued or running.

## Blockers / needs the owner
- ALERT (owner only): the tick's last git push failed and 91 commits sit unpushed on branch `arc` (remote
  github.com/naver/pisco). No error text is recorded anywhere, so the wake cannot diagnose it; the wake does not
  push. Gripe +1 on 43fe4ef5 (the tick drops git's stderr). Left for the owner.
