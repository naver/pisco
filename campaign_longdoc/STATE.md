# STATE — campaign longdoc   (rewritten WHOLESALE by every LLM wake; never append)

Last wake: 2026-11-17 07:30 (seventh wake). NEXT: wake-when job:191305,191306,191307,191308 max 3h. Question: where
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
  (ledger 2026-11-17 00:56). Arm J control 1 of 2; control 2 (project prompt, EM) not run.
- provisional: arm T LOCATE DECIDED — the long-chunk documents loss sits IN THE COMPRESSED PAGE (both halves of the
  gate, both seeds). Documents `mm_docqa_1500_1m`, 1,003,520 px, 1,500 pairs per seed, `scripts/paired_judge.py`:
  - gold page only: L1024 − C256 = −6.7 [−8.9, −4.5], McNemar 106 / 207 (s42; 191153 .568 vs 191152 .635) and −4.9
    [−7.1, −2.5], 113 / 186 (s43; 191155 .579 vs 191154 .628). Every source negative.
  - picked page raw (`RAW=route1`): +0.1 [−0.8, +1.1], McNemar 29 / 27 (s42; 191166 .827 vs 191165 .825) and −0.1
    [−1.1, +0.9], 29 / 31 (s43; 191168 .823 vs 191167 .824). Both models pick the answer page on 98.5% of questions.
  - all-4-pages reference deficit: −6.2 [−8.6, −3.9] (s42, 190271 vs 188708), −4.3 [−6.7, −1.8] (s43, 190274 vs 188711).
- provisional: arm T TRANSCRIPTION gate MET (jobs 191219 / 191220 vs reference 191170; one stage-2 model per chain, so
  no seed repeat exists): rebuilding the gold page from its `<MEM>`s, L1024's stage-2 model recovers .566 of the page's
  words vs C256's .647 — −8.0 pt [−10.6, −5.5], sign 147 / 322, p 4e-16, 500 pages paired (167 / 166 / 167 per source;
  infovqa −8.9, MP-DocVQA −11.0, SlideVQA −4.3 [−9.0, +0.2]). Answer string in the rebuild 228 vs 253 of 500.
  Self-pair 0. Table: `expQ/longdoc/recall/pair_s2_c256_vs_l1024_n167.txt`.
- provisional: arm T STAGE-1 AUTOENCODING gate NOT MET on its second half (job 191218, 500 held-out MS MARCO documents,
  nats per token, lower better, d = L1024 − C256): one 256-token chunk .0126 vs .0144, d −.0018 [−.0031, −.0006], sign
  271 / 229, p .067 — L1024 is NOT worse at 256 (both near-perfect). One 1,024-token chunk d −.0934 [−.0988, −.0884],
  500 / 0 (expected: C256 never trained on it). Control (another document's slots) 2.71 / 2.58; self-pair 0. Table:
  `expQ/longdoc/ae/ae_pair_n500.txt`. Reading: the page loss enters at stage 2 or later, not in stage 1's short chunks.
- The transcription-probe port is validated (job 191169): 360 / 360 legacy pages byte-identical, word recall 70.5% =
  70.5% (gate ±2 pt).

**Open, and running** (all 4 submitted 07:3x, PENDING on Priority at submit)
- Arm T, ten short text chunks (`text_hotpotqa_1500`, TOPK=10, CML=4096), the other three finetunes: L1024 s42 = job
  191305, C256 s43 = job 191306, L1024 s43 = job 191307; `--time 0:59` (191221 took 22:06). C256 s42 done: job 191221,
  judge .5793, 1,500 rows, judge on every row, unparseable 0, homer-017. Then pair L1024 − C256 per seed and set it
  against the recorded one-chunk g10 pairs (190270 / 190273 vs 188707 / 188710: −1.1 / −3.3).
- Arm T, the same autoencoding probe on the two STAGE-2 models (`ae_nll.sbatch 500 s2`): job 191308, `--time 0:47`,
  `--mem 15G`. Gate (written in CAMPAIGN.md before the cell): L1024 − C256 at `s256` > 0 with CI above 0 and sign
  p < .05 → stage 2 erodes text detail too; otherwise the page loss is image-specific.

## Owned jobs
| job | arm | state | rows | note |
|---|---|---|---|---|
| 191305 / 191306 / 191307 | T ten-chunk text, L1024 s42 / C256 s43 / L1024 s43 | PENDING | 1,500 Q each | out `expQ/longdoc/t_<cell>/eval/mm_docqa_pisco_k10_cml4096_text_hotpotqa_1500.json` |
| 191308 | T autoencoding on stage-2 models | PENDING | 500 docs × 4 conditions × 2 models | out `expQ/longdoc/ae/ae_nll_s2_n500.json` |
| 191221 | T ten-chunk text, C256 s42 | COMPLETED 22:06, homer-017 | 1,500 Q | judge .5793; verdict after the pairs |
| 191218 | T stage-1 autoencoding, full | COMPLETED 15:27, MaxRSS 6.5 GB, homer-017 | 500 docs | verdict closed (second half not met) |
| 191219 / 191220 | T transcription, C256 / L1024 stage-2 | COMPLETED 1:56 / 2:10, MaxRSS 9.6 / 9.8 GB, homer-017 | 501 pages each | verdict closed (gate met) |
| 191170 | T transcription reference (uncompressed) | COMPLETED 1:45, MaxRSS 9.2 GB, homer-017 | 501 pages | verdict closed |
| 191169 | T transcription port validation (v4kd, stride 1) | COMPLETED 1:26 | 360 pages | verdict closed (PASS, identical) |
| 191171 / 191172 | T plumbing | COMPLETED | 20 pages / 20 docs | verdict closed |
| 191165-191168 | T picked page raw | COMPLETED | 1,500 Q each | verdict closed |
| 191152-191155 | T gold page only, 1,003,520 px | COMPLETED | 1,500 Q each | verdict closed |
| 191148-191151 | T gold page only at 451,584 px | COMPLETED | 1,500 Q each | verdict closed off-regime |
| 191147 / 191145 / 191146 | A audit, smokes | COMPLETED / CANCELLED / COMPLETED | | verdict closed |

Cells used: 27 of 70. GPU-h: the tick sensor read 12.4 for the last 24 h at 07:30 (the campaign's first cell ran
00:5x on 2026-11-17, so this is the whole campaign so far); the 4 running add ≈ 1.6.

## Adopted jobs
(none — the only other jobs are the expmon daemons 188764 / 162472)

## Open arms and the next decision for each
| arm | status | next |
|---|---|---|
| A audit | DECIDED: gate PASS (provisional) | re-run once when the mixed set exists (arm L builds it) |
| T long-chunk training | locate + transcription decided; 4 cells running | 1) text: when 191305-191307 end, check health (1,500 rows, judge on every row, unparseable 0), `scripts/paired_judge.py` L1024 − C256 per seed on the ten-chunk files, set against −1.1 / −3.3 (one-chunk g10). 2) stage-2 AE: `python3 -I scripts/longdoc/ae_pair.py expQ/longdoc/ae/ae_nll_s2_n500.json > expQ/longdoc/ae/ae_pair_s2_n500.txt`, apply the gate. 3) close arm T with one verdict line + the training arm it motivates; rewrite `docs/longdoc_diagnosis.md` §1 and the arm T tables |
| J instrument | control 1 PASS (provisional) | after the first decided Δ_len: `rejudge.sbatch` (project prompt) on copies + EM, sign kept within 2 pt |
| L length on a fixed model | not started (next after T) | DML per size = audit max + 256 (documents 10 pages pisco 1,578, base 9,983; 20 pages pisco 2,823, base 19,564); one `LIMIT=300` plumbing cell per new variant (XD on text rows unverified: read `main()` of `scripts/eval_mm_docqa.py` first); every document cell carries `--env PX=1003520` |
| U reading many `<MEM>`s | not started | after L's long size |
| P positions | not started | needs the `--mem_position_offset` flag (patchable IF); offset N from the audit spans (≈ 14,300) |

## Tooling and controls
- `scripts/longdoc/ae_nll.sbatch [N_DOCS] [SET]`: SET `s1` (default, unchanged: stage-1 models 188422 / 188466, out
  `ae_nll_n<N>.json`) or `s2` (this wake: stage-2 models `expQ/longchunk/pt_mm_cml256|cml1024/model`, out
  `ae_nll_s2_n<N>.json`). `ae_nll.py` itself unchanged. Stage-2's collator rehearses text in the same `<AE>` format
  (pisco/mm_collator.py:66).
- `ae_pair.py` controls saved (`expQ/longdoc/ae/controls/ae_pair_controls_n20.txt`): +0.30 planted is recovered, a real
  self-pair gives 0.
- `mem_recall.py` + `mem_recall.sbatch ARM [N] [LIMIT]` + `recall_pair.py` (self-pair 0 on 500 pages this wake).

## Health evidence this wake
- 191170 / 191218 / 191219 / 191220 / 191221: COMPLETED, exit 0, homer-017 (allowed). 191221: 1,500 rows, `judge` on
  1,500, unparseable 0, `mems_used` on every row (265,880 total).
- Code md5 (pairing key) at 07:3x: eval_mm_4b.sh 50abc167, eval_mm_docqa.py dfe32aa0, judge_mm_docqa.py 02bc8748,
  pisco/model.py 994cff3d — unchanged.
- No longchunk cell is queued or running.

## Blockers / needs the owner
- None.
