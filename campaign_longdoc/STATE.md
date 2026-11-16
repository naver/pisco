# STATE — campaign longdoc   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-17 05:00 (sixth wake). NEXT: wake-when job:191170,191218,191219,191220,191221 max 6h. Question: where
do long inputs lose accuracy (eval, training data, length, `<MEM>` reading, positions, long-chunk training)? Diagnosis
only: inference probes and CPU audits, no training.

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
- The transcription-probe port is validated (job 191169, verdict closed): on the legacy stride-1 checkpoint
  `expQ/pt_mm_v4kd_4b` it rebuilt 360 / 360 gold pages byte-identical to the recorded legacy probe
  (`expQ/archq/recall/ae_s2.json`); word recall 70.5% vs 70.5% (gate ±2 pt; infovqa .578, mpdocvqa .833, slidevqa
  .704; answer found in the rebuild 192 / 360 in both). Command: `recall_pair.py --ref expQ/archq/recall/ref.json --a
  expQ/archq/recall/ae_s2.json --b expQ/longdoc/recall/v4kd_n120.json`.

**Open, and running** (all 5 on homer-017, submitted 05:01 except 191170)
- Arm T, transcription probe ("are fewer of a page's words in L1024's stage-2 `<MEM>`s?"), 501 gold pages each:
  - reference, uncompressed backbone (`ref 167`): job 191170, RUNNING, 420 / 501 pages at 5,163 s.
  - C256 stage-2 model (`s2_c256 167`): job 191219; L1024 stage-2 model (`s2_l1024 167`): job 191220. `--time 5:15`
    (≈1:45 expected). Gate: L1024 ≥ 3 pt fewer words than C256, sign test p < .05, paired per page.
- Arm T, stage-1 autoencoding probe, 500 held-out MS MARCO documents (`ae_nll.sbatch 500`): job 191218, `--time 1:50`
  (≈36 min expected). Gate: L1024 better at `l1024` AND worse at `s256`, both CIs excluding 0.
  provisional hint, n = 20, decides nothing (job 191172): both models rebuild a 256-token chunk almost perfectly
  (0.0111 vs 0.0113 nats per token, d −0.0001 [−0.0030, +0.0030]), so the "worse at 256" half may sit at a ceiling.
- Arm T, ten short text chunks (`text_hotpotqa_1500`, TOPK=10, CML=4096): C256 s42 = job 191221 (`--time 0:59`).
  L1024 s42 and C256 / L1024 s43 are NOT launched yet (submission cap of 4 reached).

## Owned jobs
| job | arm | state | rows | note |
|---|---|---|---|---|
| 191218 | T stage-1 autoencoding, full | RUNNING | 500 documents × 4 conditions × 2 models | out `expQ/longdoc/ae/ae_nll_n500.json` |
| 191219 / 191220 | T transcription, C256 / L1024 stage-2 | RUNNING | 501 pages each | out `expQ/longdoc/recall/s2_c256_n167.json`, `s2_l1024_n167.json` |
| 191221 | T ten-chunk text, C256 s42 | RUNNING | 1,500 Q | out `expQ/longdoc/t_c256_s42/eval/mm_docqa_pisco_k10_cml4096_text_hotpotqa_1500.json` |
| 191170 | T transcription reference (uncompressed) | RUNNING | 501 pages | out `expQ/longdoc/recall/ref_n167.json` |
| 191169 | T transcription port validation (v4kd, stride 1) | COMPLETED 1:26:11, MaxRSS 7.7 GB, homer-017 | 360 pages | verdict closed (PASS, identical) |
| 191171 / 191172 | T plumbing (stride-8 transcription; stage-1 AE) | COMPLETED 4:41 / 4:01, homer-017 | 20 pages / 20 docs | verdict closed (plumbing PASS) |
| 191165-191168 | T picked page raw, 1,003,520 px | COMPLETED 31 min each | 1,500 Q each | verdict closed (locate decided); judge .825 / .827 / .824 / .823 |
| 191152-191155 | T gold page only, 1,003,520 px | COMPLETED, homer-017 | 1,500 Q each | verdict closed; judge .635 / .568 / .628 / .579 |
| 191148-191151 | T gold page only at 451,584 px | COMPLETED, homer-017 | 1,500 Q each | verdict closed off-regime; −6.3 / −4.7, same gap |
| 191147 | A audit (full) | COMPLETED 7:10 | 20,000 eval + 8,000 training | verdict closed (gate PASS, provisional) |
| 191145 / 191146 | A audit smokes | CANCELLED / COMPLETED | | verdict closed |

Cells used: 23 of 70. GPU-h used: ~6.6 (tick sensor, 24 h); the 5 running add ~4.5.

## Adopted jobs
(none — the only other jobs are the expmon daemons 188764 / 162472)

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| A audit | DECIDED: gate PASS (provisional) | re-run once when the mixed set exists (arm L builds it) |
| T long-chunk training | LOCATE DECIDED (compressed page); 3 probes running | 1) AE: `python3 -I scripts/longdoc/ae_pair.py expQ/longdoc/ae/ae_nll_n500.json` (also `--a l1024 --b l1024` self-pair); if `s256` sits at the ceiling (both ≈ 0.01 nats, CI inside ±0.003) write "no difference measurable at 256 tokens", not "not worse". 2) transcription: `recall_pair.py --ref expQ/longdoc/recall/ref_n167.json --a …/s2_c256_n167.json --b …/s2_l1024_n167.json`. 3) text: launch the other 3 cells exactly like 191221 (`expmon launch --exp longdoc --stage T --time 0:59:00 --partition gpu-be --exclude 'homer-[005-009]' --env TOPK=10 --env CML=4096 eval_mm_4b.sh expQ/longdoc/t_<l1024_s42\|c256_s43\|l1024_s43>/model pisco "" eval_data/text_hotpotqa_1500`), then pair L1024 − C256 per seed and set it against the recorded one-chunk g10 pairs (190270 / 190273 vs 188707 / 188710: −1.1 / −3.3). Then close arm T with one verdict and the training arm it motivates |
| J instrument | control 1 PASS (provisional) | after the first decided Δ_len: `rejudge.sbatch` (project prompt) on copies + EM, sign kept within 2 pt |
| L length on a fixed model | not started (next arm after T) | DML per size = audit max + 256 (documents 10 pages pisco 1,578, base 9,983; 20 pages pisco 2,823, base 19,564); one `LIMIT=300` plumbing cell per new variant (XD on text rows unverified: read `main()` of `scripts/eval_mm_docqa.py` first); every document cell carries `--env PX=1003520` |
| U reading many `<MEM>`s | not started | after L's long size |
| P positions | not started | needs the `--mem_position_offset` flag (patchable IF); offset N from the audit spans (documents 20 pages p50 18,027 − 4 pages p50 3,720 ≈ 14,300) |

## Tooling and controls
- `scripts/longdoc/ae_nll.py` + `ae_nll.sbatch [N_DOCS]`: the stage-1 autoencoding probe. Both stage-1 text models (C256
  188422, L1024 188466) rebuild the same held-out MS MARCO documents (rows from 1,000,000 on — training used rows
  [0, 1M), `pisco/train.py` `select(range(samples))` — kept at ≥ 1,024 tokens) in the stage-1 format, through
  `PISCO.forward` so stride-8 positions apply. Conditions: `s256`, `l1024`, `l1024_4x256` (chunk length varied, target
  fixed), `wrong256` (the previous document's `<MEM>`s: control). C256 trained with targets ≤ 512 tokens (DML 512), so
  `l1024` is out of its range.
- `ae_pair.py` controls, now saved (`expQ/longdoc/ae/controls/ae_pair_controls_n20.txt`, this wake): +0.30 planted on
  L1024's `s256` rows of the n = 20 file is recovered as +0.2999 [+0.2970, +0.3030], sign 0 / 20, other conditions
  unchanged; a real self-pair (`--a l1024 --b l1024`) gives 0 on every condition. (The previous wake's "CI [+.24, +.46],
  p 6e-6" came from an unsaved synthetic file and is withdrawn.)
- `mem_recall.py` (stride-aware port of `scripts/probe_mem_recall.py`) + `mem_recall.sbatch ARM [N] [LIMIT]` +
  `recall_pair.py` (paired bootstrap + sign test; self-pair 0 on 360 pages). Validated: see Settled.
- n = 20 plumbing note: the C256 stage-2 rebuild averaged 71 words per page vs 198 for the legacy stride-1 model
  (191171 vs 191169); different models and pages, decides nothing.

## Health evidence this wake
- 191169: 360 / 360 records, exit 0, homer-017 (allowed), 1:26:11 wall, MaxRSS 7.7 GB.
- Code md5 (pairing key) unchanged at 05:01: eval_mm_4b.sh 50abc167, eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py
  02bc8748, pisco/model.py 994cff3d — identical to longchunk's plain cells.
- No longchunk cell is queued or running.

## Blockers / needs the owner
- None.
