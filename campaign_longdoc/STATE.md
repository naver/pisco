# STATE — campaign longdoc   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-17 02:30 (fourth wake). Question: where do long inputs lose accuracy (eval, training data, length,
`<MEM>` reading, positions, long-chunk training)? Diagnosis only: inference probes and CPU audits, no training.

## TL;DR — where this leaves the question
**Settled**
- Training never shows the decoder long inputs (a count, not yet a cause; job 191147, verifier: supported): finetune
  rows reach at most 488 `<MEM>`s and a position span of 4,017 (2,000 sampled rows). Out of that range: 10- and 20-page
  documents 1,500 / 1,500, 40 text passages 1,498 / 1,500, 20 passages 35 / 1,500, whole documents 198 / 500; 4-page
  documents 0 / 1,500 (max 485 `<MEM>`s, span 3,962).
- provisional: the evaluation feeds the long inputs it claims (arm A, gate passed; job 191147, verdict closed): truncation
  cut 0 / 1,500 and pisco/base parity 1,500 / 1,500 on every gating variant (documents 4 / 10 / 20 pages, text 10 / 20 /
  40 passages); health 646,042 = 646,042; the CML 256 controls do fail (924 / 1,500 and 275 / 1,500). Limits: on
  documents the cut check can only fail on prompt length (images are never cut); the audit's parity used one row
  builder for both modes. Re-check on recorded runs: pages per question equal in both modes on 1,500 / 1,500 (4 pages,
  184778 vs base), 1,500 / 1,500 (10 pages, 173550 vs base), 500 / 500 (whole documents, 174908 vs base) — a count.
- provisional: the length-effect calculator recovers the tail model's one-chunk minus ten-chunk HotpotQA difference
  (183538, 1,500 pairs): d = −.1687 [−.1960, −.1407], sign test +121 / −374 (target −.169 ± .005), self-pair 0
  (stdlib on recorded files; ledger 2026-11-17 00:56). Arm J control 1 of 2; control 2 (project prompt, EM) not run.
- provisional: arm T, the long-chunk loss is NOT in multi-page reading. Gold page only (distractor pages removed),
  documents, 1,003,520 px, 1,500 pairs per seed: L1024 − C256 = −6.7 [−8.9, −4.5], McNemar L1024-only 106 / C256-only
  207 (s42; 191153 .568 vs 191152 .635) and −4.9 [−7.1, −2.5], 113 / 186 (s43; 191155 .579 vs 191154 .628). Every
  source negative at both seeds. The all-4-pages deficit was −6.2 [−8.6, −3.9] (s42, 190271 vs 188708) and −4.3
  [−6.7, −1.8] (s43, 190274 vs 188711): removing distractors does not shrink it. Half of "in the compressed page" met.

**Open, and running**
- Arm T second half: the picked page raw (`RAW=route1`: the model picks the answer page from its `<MEM>`s, that page
  goes in raw, the other 3 stay compressed), documents, 1,500 Q, 1,003,520 px, on C256 s42 / L1024 s42 / C256 s43 /
  L1024 s43 = jobs 191165 / 191166 / 191167 / 191168, RUNNING since 02:31 (homer-017 ×3, homer-011), `--time 1:34`
  (≈31 min generation + judge each, from R's route1 cell 188441). Gate: the deficit shrinks inside ±2 pt at both seeds
  → it sits in the compressed page. Prediction written before the campaign: it sits in the compressed page.

## Owned jobs
| job | arm | state | rows | eta | note |
|---|---|---|---|---|---|
| 191165 | T picked page raw, C256 s42 | RUNNING | 1,500 Q | ~03:10 | out `expQ/longdoc/t_c256_s42/eval/mm_docqa_pisco_px1003520_rawroute1_mm_docqa_1500_1m.json` |
| 191166 | T picked page raw, L1024 s42 | RUNNING | 1,500 Q | ~03:10 | `expQ/longdoc/t_l1024_s42/eval/` same name |
| 191167 | T picked page raw, C256 s43 | RUNNING | 1,500 Q | ~03:10 | `expQ/longdoc/t_c256_s43/eval/` same name |
| 191168 | T picked page raw, L1024 s43 | RUNNING | 1,500 Q | ~03:10 | `expQ/longdoc/t_l1024_s43/eval/` same name |
| 191152-191155 | T gold page only, 1,003,520 px | COMPLETED 18:43-18:53, homer-017 | 1,500 Q each | | verdict closed (decided, above); judge .635 / .568 / .628 / .579 |
| 191148-191151 | T gold page only at 451,584 px | COMPLETED, homer-017 | 1,500 Q each | | verdict closed off-regime; −6.3 / −4.7, same gap |
| 191147 | A audit (full) | COMPLETED 7:10 | 20,000 eval + 8,000 training | | verdict closed (gate PASS, provisional) |
| 191145 / 191146 | A audit smokes | CANCELLED / COMPLETED | | | verdict closed |

Cells used: 15 of 70. GPU-h used: ~2.5 (eight 19-min cells); the 4 running add ~2.

## Adopted jobs
(none — the only other jobs are the expmon daemons 188764 / 162472)

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| A audit | DECIDED: gate PASS (provisional) | re-run once when the mixed set exists (arm L builds it) |
| T long-chunk training | gold half decided; picked-page-raw cells running | 1) pair `_px1003520_rawroute1` files per seed (`scripts/paired_judge.py`). Inside ±2 pt at both seeds → verdict "in the compressed page"; still ≤ −3 pt → the loss is also in how L1024's decoder reads a raw page (write it as found, amend the law's prediction). 2) transcription probe, tooling ready (below): next wake launch `mem_recall.sbatch v4kd 120` (validation, `--time 4:26:00` = 1:28:30 of legacy 174860 × 3, `--mem 16G` = 8.1 GB MaxRSS × 2), `ref 167` (`--time 6:10:00` = 1:28:30 × 501/360 × 3), `s2_c256 167 20` (plumbing, `--time 0:30:00`); the full `s2_c256 167` / `s2_l1024 167` only after v4kd reproduces 70.5% ± 2 pt (`recall_pair.py --ref expQ/archq/recall/ref.json --a expQ/archq/recall/ae_s2.json --b expQ/longdoc/recall/v4kd_n120.json`). 3) plain ten-chunk text on the 4 finetunes: `eval_mm_4b.sh expQ/longdoc/t_<cell>/model pisco "" eval_data/text_hotpotqa_1500` with `--env TOPK=10 --env CML=4096`, `--time 0:59:00` (19:43 of 190273 × 3); compare with the recorded g10 pairs 190270/190273 vs 188707/188710. 4) write `ae_nll.py` |
| J instrument | control 1 PASS (provisional) | after the first decided Δ_len: `rejudge.sbatch` (project prompt) on copies + EM, sign kept within 2 pt |
| L length on a fixed model | not started | DML per size = audit max + 256 (documents 10 pages pisco 1,578, base 9,983; 20 pages pisco 2,823, base 19,564); one `LIMIT=300` plumbing cell per new variant (XD on text rows unverified); every document cell carries `--env PX=1003520` |
| U reading many `<MEM>`s | not started | after L's long size |
| P positions | not started | needs the `--mem_position_offset` flag (patchable IF); offset N from the audit spans (documents 20 pages p50 18,027 − 4 pages p50 3,720 ≈ 14,300) |

## Tooling written this wake (campaign code, `scripts/longdoc/`, not run on a GPU yet)
- `mem_recall.py`: stride-aware port of `scripts/probe_mem_recall.py`. Imports the legacy `pick_rows` and `run_ref`
  unchanged; its AE arm passes `PISCO.decoder_position_ids` to `decoder.generate` (None on a stride-1 checkpoint, so
  the legacy path is unchanged). `mem_recall.sbatch ARM N LIMIT`, ARM = ref | v4kd | s2_c256 | s2_l1024, output
  `expQ/longdoc/recall/<ARM>_n<N>.json`, refuses to overwrite. Symlinks `expQ/longdoc/s2_c256|s2_l1024/model` → the
  longchunk stage-2 models (both stride 8).
- `recall_pair.py` (stdlib): per-page word recall vs a reference, paired bootstrap CI + sign test. Self-pair gives 0
  on 360 pages; the recorded legacy pair ae_s2 vs ae_ftcomp gives −.027 [−.053, −.001], so it can show a difference.
  Legacy recall recomputed = 70.5% (360 pages), the port's validation target.

## Health evidence this wake
- 191152-191155: n = 1,500, a judge on every row, `judge_unparseable` 0, `image_max_pixels` 1,003,520, homer-017
  (allowed); `mems_used` p50 119 / max 238 = arm A's gold-only prediction (119 / 238).
- Code md5 (pairing key) unchanged at the route1 launch: eval_mm_4b.sh 50abc167, eval_mm_docqa.py dfe32aa0,
  judge_mm_docqa.py 02bc8748, pisco/model.py 994cff3d — identical to longchunk's plain cells.
- No longchunk cell is queued or running.

## Blockers / needs the owner
- None.
