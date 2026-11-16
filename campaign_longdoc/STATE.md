# STATE — campaign longdoc   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-17 00:47 (first wake). Question: where do long inputs lose accuracy (eval, training data, length,
`<MEM>` reading, positions, long-chunk training)? Diagnosis only: inference probes and CPU audits, no training.

## TL;DR — where this leaves the question
**Settled**
- The length-effect calculator works: on the tail model (183538, HotpotQA, 1,500 pairs) one joined chunk minus ten
  chunks gives d = −.1687 [−.1960, −.1407], sign test +121 / −374 (target −.169 ± .005), and a file paired with itself
  gives exactly 0. `scripts/paired_judge.py` gives the same 374 / 121 split. Arm J control 1 of 2 passed (no job:
  stdlib on recorded files, ledger 2026-11-17).
- The input audit reproduces what the eval really feeds: on 12 rows per variant (smoke 191146), the predicted `<MEM>`
  count equals the recorded `mems_used` on 12 / 12 rows of R's 4-page documents, 183538's g10 and MRQA (CML 4096),
  and the decoder prompt length equals R's on 12 / 12. Plumbing only, decides nothing.

**Open, and running**
- Arm A, full input audit (the feasibility gate): job 191147, RUNNING on homer-017 since 00:55, 20,000 eval rows
  (14 variants, pisco + base mode) + 8,000 training rows (4 stages); expected ~1-1.5 h. Gate: cut rate ≤ 1% and
  parity 100% on every gating variant; health = 646,042 `<MEM>`s on R's 4-page documents.
- provisional (12 rows each, smoke 191146): the finetune decoder saw at most 488 `<MEM>`s and a position span of
  3,992; 10- and 20-page document evals carry 950 / 1,920 `<MEM>`s (span 7,730 / 15,508) on 12 / 12 rows, i.e. all
  beyond the finetune range; 40-passage text 11 / 12 beyond. Stage-1 C256 decoder sequences sit at the 512-token
  cap on the p50 row. An uncompressed 20-page prompt is ~15.5k tokens.

## Owned jobs
| job | arm | state | elapsed | rows | eta | note |
|---|---|---|---|---|---|---|
| 191147 | A audit (full) | RUNNING homer-017 | ~0:08 at 01:03 (2,000 of 20,000 eval rows at 1:42) | 20,000 eval + 8,000 training | ~02:30-04:00 (10- and 20-page rows decode 6-16 borrowed rows each) | 12 CPU, 64 GB, --time 4 h |
| 191146 | A audit smoke | COMPLETED 1:58 | | 168 + 48 | | verdict closed (plumbing) |
| 191145 | A audit smoke | CANCELLED before start | | | | asked 16 CPUs, none free on an allowed node; verdict closed |

## Adopted jobs
(none — the only other jobs are the expmon daemons 188764 / 162472)

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| A audit | full cell 191147 running | read `expQ/longdoc/audit_191147/audit.json`: health 646,042; cut rate + parity per gating variant → pass/stop the campaign; write the data-audit table in `docs/longdoc_diagnosis.md` |
| J instrument | control 1 PASS (−.1687, self-pair 0) | after the first decided Δ_len: `rejudge.sbatch` (project prompt) on copies + EM, sign kept within 2 pt |
| T long-chunk training | not started | first after A: `GOLD=1`, `RAW=route1` on the C256 / L1024 finetunes (4 seeds × 2 probes, documents), via `expQ/longdoc/<cell>/model` symlinks |
| L length on a fixed model | not started | after A: DML = audit max + 256 per size; one `LIMIT=300` plumbing cell per new variant (XD on text rows unverified) |
| U reading many `<MEM>`s | not started | after L's long size |
| P positions | not started | needs the `--mem_position_offset` flag (patchable IF); offset N from the audit spans |

## Health evidence this wake
- Code md5 (pairing key): eval_mm_4b.sh 50abc167, eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py 02bc8748,
  pisco/model.py 994cff3d — identical to longchunk's, so its recorded files pair with this campaign's cells.
- Smoke 191146: 0 errors, MaxRSS 2.9 GB at 4 workers, ~2 s per row per worker.
- No longchunk cell is queued or running (only the daemons are), so the patchable-IF eval files may be patched later.
- Law fix this wake: CAMPAIGN.md's stage-1 model paths now carry the `expQ/longchunk/` prefix (the dirs live there).

## Blockers / needs the owner
(none)
