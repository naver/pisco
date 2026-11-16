# STATE — campaign longdoc   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-17 03:30 (fifth wake). NEXT: wake-when job:191169 (the port validation) max 5h. Question: where do long inputs lose accuracy (eval, training data, length,
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
- provisional: arm T LOCATE DECIDED — the long-chunk documents loss sits IN THE COMPRESSED PAGE (both halves of the
  gate, both seeds; the prediction written before the campaign holds). Documents `mm_docqa_1500_1m`, 1,003,520 px,
  1,500 pairs per seed, `scripts/paired_judge.py`:
  - gold page only (distractor pages removed): L1024 − C256 = −6.7 [−8.9, −4.5], McNemar 106 / 207 (s42; 191153 .568
    vs 191152 .635) and −4.9 [−7.1, −2.5], 113 / 186 (s43; 191155 .579 vs 191154 .628). Every source negative.
  - picked page raw (`RAW=route1`: the model picks the answer page from its `<MEM>`s, that page goes raw, the other 3
    stay compressed): +0.1 [−0.8, +1.1], McNemar 29 / 27, p .89 (s42; 191166 .827 vs 191165 .825) and −0.1
    [−1.1, +0.9], 29 / 31, p .90 (s43; 191168 .823 vs 191167 .824). Every source within ±0.6 pt.
  - both models pick the answer page on 98.5% of questions (1,477 / 1,478 / 1,476 / 1,477 of 1,500).
  - all-4-pages reference deficit: −6.2 [−8.6, −3.9] (s42, 190271 vs 188708), −4.3 [−6.7, −1.8] (s43, 190274 vs 188711).
  Reading: L1024's page `<MEM>`s still say where the answer is but hold it worse; the decoder is not the problem.

**Open, and running**
- Arm T, transcription probe ("are fewer of a page's words in L1024's stage-2 `<MEM>`s?"):
  - port validation on the legacy stride-1 checkpoint (`mem_recall.sbatch v4kd 120`, 360 pages): job 191169, RUNNING
    since 03:36, `--time 4:26` (≈1:30 expected). Gate: word recall 70.5% ± 2 pt vs `expQ/archq/recall/ref.json`.
  - reference transcriptions, uncompressed backbone, 501 pages (`ref 167`): job 191170, RUNNING since 03:36, `--time 6:10`.
  - stride-8 plumbing (`s2_c256 167 20`): job 191171 DONE, PASS (20 / 20 pages rebuilt as coherent text, 4:41 wall).
- Arm T, stage-1 autoencoding probe (new tool `scripts/longdoc/ae_nll.py`, below): plumbing on 20 documents, job
  191172, DONE, PASS (4:01 wall, MaxRSS 5.7 GB; the wrong-document control gives 2.58 / 2.70 nats per token vs 0.011
  with the document's own `<MEM>`s). The full 500-document cell is NOT launched: this wake used its 4 submissions.
  provisional hint, n = 20, decides nothing: both models rebuild a 256-token chunk almost perfectly (0.0111 vs 0.0113
  nats per token), so the "worse at 256" half of the gate may sit at a ceiling; L1024 is far better at 1,024 tokens
  (0.007 vs 0.106), as expected since C256 never trained on targets above 512 tokens.

## Owned jobs
| job | arm | state | rows | note |
|---|---|---|---|---|
| 191169 | T transcription port validation (v4kd, stride 1) | RUNNING | 360 pages | out `expQ/longdoc/recall/v4kd_n120.json` |
| 191170 | T transcription reference (uncompressed) | RUNNING | 501 pages | out `expQ/longdoc/recall/ref_n167.json` |
| 191171 | T transcription plumbing, stride 8 | COMPLETED 4:41, homer-017 | 20 pages | verdict closed (plumbing PASS) |
| 191172 | T stage-1 autoencoding plumbing | COMPLETED 4:01, homer-017 | 20 documents × 4 conditions × 2 models | verdict closed (plumbing PASS); `expQ/longdoc/ae/ae_nll_n20.json` |
| 191165-191168 | T picked page raw, 1,003,520 px | COMPLETED 31 min each, homer-017 ×3 / homer-011 | 1,500 Q each | verdict closed (locate decided); judge .825 / .827 / .824 / .823 |
| 191152-191155 | T gold page only, 1,003,520 px | COMPLETED, homer-017 | 1,500 Q each | verdict closed; judge .635 / .568 / .628 / .579 |
| 191148-191151 | T gold page only at 451,584 px | COMPLETED, homer-017 | 1,500 Q each | verdict closed off-regime; −6.3 / −4.7, same gap |
| 191147 | A audit (full) | COMPLETED 7:10 | 20,000 eval + 8,000 training | verdict closed (gate PASS, provisional) |
| 191145 / 191146 | A audit smokes | CANCELLED / COMPLETED | | verdict closed |

Cells used: 19 of 70 (11 before the last wake, 4 picked-page-raw, 4 this wake). GPU-h used: ~4.8 (eight 19-min + four 31-min cells, two 4-5-min plumbings); the 2 running add ~3.5.

## Adopted jobs
(none — the only other jobs are the expmon daemons 188764 / 162472)

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| A audit | DECIDED: gate PASS (provisional) | re-run once when the mixed set exists (arm L builds it) |
| T long-chunk training | LOCATE DECIDED (compressed page); transcription and AE probes under way | 1) launch the full AE cell: `expmon launch --exp longdoc --stage T --time 1:50:00 --mem 9G --partition gpu-be --exclude 'homer-[005-009]' scripts/longdoc/ae_nll.sbatch 500` (1:50 = ≈36 min (≈2 s per document per model × 1,000 + 2.5 min loads, from 191172) × 3; 9G = 5.7 GB × 1.5); pair with `python3 -I scripts/longdoc/ae_pair.py expQ/longdoc/ae/ae_nll_n500.json`. Gate: L1024 better at `l1024` AND worse at `s256`, both CIs excluding 0; if `s256` sits at the ceiling (both ≈ 0.01 nats, CI inside ±0.003), write "no difference measurable at 256 tokens" rather than "not worse", and lean on the transcription probe. 2) when 191169 ends: `recall_pair.py --ref expQ/archq/recall/ref.json --a expQ/archq/recall/ae_s2.json --b expQ/longdoc/recall/v4kd_n120.json`; B within ±2 pt of 70.5% → launch `mem_recall.sbatch s2_c256 167` and `s2_l1024 167` (`--time` from 191171: 4:41 for 20 pages → ≈ 501/20 × 4 min + load ≈ 1:45, × 3 = 5:15; `--mem 16G`), pair against `ref_n167.json` (gate: ≥ 3 pt fewer words, sign test p < .05). 3) plain ten-chunk text on the 4 finetunes: `eval_mm_4b.sh expQ/longdoc/t_<cell>/model pisco "" eval_data/text_hotpotqa_1500` with `--env TOPK=10 --env CML=4096`, `--time 0:59:00` (19:43 of 190273 × 3); compare with the recorded g10 pairs 190270/190273 vs 188707/188710 |
| J instrument | control 1 PASS (provisional) | after the first decided Δ_len: `rejudge.sbatch` (project prompt) on copies + EM, sign kept within 2 pt |
| L length on a fixed model | not started | DML per size = audit max + 256 (documents 10 pages pisco 1,578, base 9,983; 20 pages pisco 2,823, base 19,564); one `LIMIT=300` plumbing cell per new variant (XD on text rows unverified); every document cell carries `--env PX=1003520` |
| U reading many `<MEM>`s | not started | after L's long size |
| P positions | not started | needs the `--mem_position_offset` flag (patchable IF); offset N from the audit spans (documents 20 pages p50 18,027 − 4 pages p50 3,720 ≈ 14,300) |

## Tooling written this wake (campaign code, `scripts/longdoc/`)
- `ae_nll.py` + `ae_nll.sbatch [N_DOCS]`: the stage-1 autoencoding probe. Both stage-1 text models (C256 188422,
  L1024 188466) rebuild the same held-out MS MARCO documents (rows from 1,000,000 on — training used rows [0, 1M),
  `pisco/train.py` `select(range(samples))` — kept at ≥ 1,024 tokens) in the stage-1 format (`<AE> <MEM>*n <bos> text
  <eos>`, loss after the last `<MEM>`, `PISCO.forward` so stride-8 positions apply). Four conditions per document:
  `s256` (one 256-token chunk → those 256 tokens), `l1024` (one 1,024-token chunk), `l1024_4x256` (four 256-token
  chunks → the same 1,024 tokens: chunk length varied, target fixed), `wrong256` (the previous document's `<MEM>`s:
  control, must be far worse). Note: C256 trained with targets ≤ 512 tokens (DML 512), so `l1024` is out of its range.
- `ae_pair.py` (stdlib): per document, d = A − B in nats per token, paired bootstrap CI + sign test, self-pair.
  Tested on planted data: a +0.30 shift on one condition recovered (CI [+.24, +.46], p 6e-6), others null, self-pair 0.

## Health evidence this wake
- 191165-191168: n = 1,500, a judge on every row, `judge_unparseable` 0, `image_max_pixels` 1,003,520, `raw_policy`
  route1, `n_raw` = 1 on every row, homer-017 / homer-011 (allowed). `mems_used` sum 483.5k ≈ .748 of the plain
  646,042 (3 of 4 pages compressed, by design; p50 348, max 367).
- Code md5 (pairing key) unchanged: eval_mm_4b.sh 50abc167, eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py 02bc8748,
  pisco/model.py 994cff3d — identical to longchunk's plain cells.
- No longchunk cell is queued or running.

## Blockers / needs the owner
- None.
