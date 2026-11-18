# CAMPAIGN longdoc — pisco   (the law; the agent re-reads this every wake; rewrite, never append)

created 2026-11-16 · kind **probe** · root `/beegfs/scratch/user/hdejean/pisco` · window until **2027-02-14** (launches allowed through that day 23:59) · owner herve.dejean@naverlabs.com · wakes run `claude-opus-5-5`

A probe campaign runs many short inference/evaluation cells (minutes each) and decides on counts or metrics
against a baseline cell — not on training curves. The binding constraint is submissions per wake and queue
time, not GPU-hours (navmem: 0.4 of 12 GPU-h used, cells of 1m42–3m17).

## Mission
Successor of campaign 'longchunk' (its report: /beegfs/scratch/user/hdejean/pisco/campaign_longchunk/REPORT.md). Open at hand-off: pt_text_4b_long.sh — job 189693 CANCELLED, no verdict.
(REPORT.md is not written yet; REPORT.json carries the same Settled list. 189693 belongs to longchunk and was stopped by
the owner on 2026-11-16: never verdict it or relaunch it here.)
**Question (owner, QUESTION.md):** every attempt at long inputs loses accuracy. WHERE is the bottleneck: training data,
training itself, evaluation, or something the probes point to (the decoder reading many `<MEM>`s, positions)? This is a
DIAGNOSIS: inference probes on trained models and CPU audits only. No training (hard rule 8).
Facts to explain:
- Long-chunk stage 1 lost: L1024 − C256 on g10 −1.1 / −3.3 pt, documents −6.2 / −4.3, photos +0.3 / 0.0 (paired, 1,500 Q,
  seeds 42 / 43; campaign_longchunk/STATE.md:17-25). Stage-2 and finetune held-out losses were higher at every point (:30),
  while stage-1 ownership stayed .969. The two chains differ only in stage-1 chunk length (:42).
- What training shows the decoder: stage 1 two chunks per document; stage 2 one image or 2-4 pages (`find`,
  `multi_max_docs` 4, pisco/mm_collator.py:110); finetune 4 pages, decoder 1,024 tokens, text cut to 256 tokens
  (pisco/configs/finetune_mm.yaml:39-41). The evals go up to 20 pages (docs/data_review.md:109).
- On a fixed model, the gap between compressed and uncompressed has looked flat in length so far: 4 / 8 / 10 pages
  .578 / .574 / .585 vs .809 / .807 / .801 (docs/architecture_review.md:31-33, model 174908). HotpotQA: ten chunks .579,
  one 1,270-token chunk .617, uncompressed .752 (docs/multimodal_results.md:105-107, EXPERIMENTS.md:407). Whole documents
  (500 Q): all-`<MEM>` .698 vs uncompressed .918 (EXPERIMENTS.md:394, 404). Route-then-expand loses −.115 at 11-20 pages,
  and the loss comes from the page pick (architecture_review.md:71-77).
- On multi-page documents the answer stays in the `<MEM>`s but the decoder does not use it (27 such failures vs 6 losses,
  architecture_review.md:100-110). OpenEQA: more compressed frames barely help (48-52 vs 57 for 32 raw frames, HANDOFF.md:57).
**Deciding numbers, one per candidate** (paired judge accuracy on the same questions unless stated):
- evaluation: arm A's cut rate and parity, arm J's judge agreement, and arm E (do the gating sets hold long documents at all);
- length on a fixed model: Δ_len per modality (arm L);
- reading many `<MEM>`s: dilution D (arm U);
- positions: the offset cost (arm P);
- training itself: where the L1024 − C256 deficit sits (arm T);
- training data: arm A's out-of-range share. This is a count; it names a cause only together with a paired effect from arm U or P.
The answer is the list of confirmed causes, ranked by how many long-input questions each one repairs.
**Regime:** judge accuracy in [0, 1], pt = .01, Qwen3-8B rubric judge (`scripts/judge_mm_docqa.py`), greedy decoding,
paired on query_id by `scripts/paired_judge.py` (bootstrap 95% CI + McNemar). Gating sets, 1,500 Q each:
- documents `eval_data/mm_docqa_1500_1m` (500 InfoVQA / MP-DocVQA val / SlideVQA, 4 pages, PX=1003520);
- text `eval_data/text_hotpotqa_1500` (HotpotQA distractor validation, 10 passages, TOPK=10, CML=4096);
- mixed `eval_data/longdoc_mixed_1500` (to build, arm L).
Reported only (n < 1,500): `mm_docqa_longdoc_1m` (500 Q, 1-20 pages) and the `longbench_multifieldqa_en` length buckets.
**The gating sets are not long documents (owner, 2026-11-17: "do we really have the proper dataset to evaluate?" →
"send it"; arm A's audit, job 191147):** g10 is 10 independent HotpotQA paragraphs glued together (p50 1,515 tokens,
answer in ~110-token gold paragraphs); the text passage sets are many ~140-token chunks; MRQA chunks p50 401; document
pages are separate images where text chunk length does not apply; only LongBench MultiFieldQA is long (0-4k / 4-8k /
8-12k / 12k+ tokens: 39 / 48 / 48 / 15 Q = 150, too few for a 3-pt bar). So no recorded number measures one long
coherent document. Arm E builds that yardstick; the long-chunk verdict on text is open until E's paired cells say.
Models:
- R = `expQ/ft_mm_4b_184778/model` (s42), repeated on `expQ/ft_mm_4b_184780/model` (s43);
- C256 / L1024 finetunes = `expQ/longchunk/ft_cml256_s42|s43/model` / `ft_cml1024_s42|s43/model`; their stage-2 models
  `expQ/longchunk/pt_mm_cml256|cml1024/model`; their stage-1 models `expQ/longchunk/pt_text_1M_cml256_188422/model` and `expQ/longchunk/pt_text_1M_cml1024_188466/model`;
- uncompressed = `eval_mm_4b.sh … base` on the same rows (existing files in `expQ/base_qwen35_4b/eval/` are reused).
The owner's standing rules: probe before training; one variable per cell; paired judge on 1,500 Q; text, images and mixed
documents for every capability; tables name the metric and its direction, and carry the uncompressed decoder and a compression-rate column.
If it replicates or extends a paper, `papers/<slug>.md` (`expmon paper`, skill `paper`) is the source of every number
quoted here — cite the card, not the PDF.

## Arms — what varies, what is held constant
| arm | cell launcher (committed) | factors (what varies) | held constant | baseline cell | gate (metric vs baseline) | cells |
|---|---|---|---|---|---|---|
| A. audit (feasibility gate) | `scripts/longdoc/audit.sbatch` → `scripts/longdoc/audit_inputs.py` (to write); CPU, `-p gpu-be --gres=gpu:0` | eval side: every variant this law names, plus the settings earlier long claims used (g10 CML=4096, MRQA CML=4096, longdoc k20), in both modes. Training side: stage 1 (the C256 / L1024 settings on `data/msmarco_docs`), stage 2 (188467's `PT_DATA` + collator kwargs), finetune (188468's `FT_DATA`). 2,000 sampled rows per stage | R's tokenizers and image processor, PX=1003520, rate 8, stride 8; the eval's own row builders (`example_chunks`, `with_distractors`, `gold_only`) are imported, never rewritten | — | feasibility gate below. Reports per variant: `<MEM>`s per question (p50 / p99 / max), tokens per chunk, decoder prompt tokens, decoder position span, and the out-of-range share (questions above the finetune sample's maximum `<MEM>` count and position span) | 1 (+1 re-run) |
| J. instrument | `scripts/longdoc/length_effect.py` (stdlib, to write: computes Δ_len and D from result jsons); `scripts/longdoc/rejudge.sbatch` (to write) = `judge_mm_docqa.py --prompt project` run on COPIES under `expQ/longdoc/` | the metric; the judge prompt | the result files | — | `length_effect.py` must recover the tail model's g10 − g1 = −.169 (multimodal_results.md:107, files of 183538) within ±.005, and give 0 for a file paired with itself. Every Δ_len that decides an arm must keep its sign and stay within 2 pt under the rubric judge, the project prompt and EM; otherwise that arm is "judge-sensitive" (an evaluation bottleneck) | 3 |
| L. length on a fixed model | `eval_mm_4b.sh expQ/longdoc/<cell>/model pisco\|base "" <set>` with `XD=n TOPK=… DML=<audit max + 256>` (plus `PX=1003520` on images, `CML` ≥ the chunk on text) | input size only. Sizes: documents 4 / 10 / 20 pages (`XD` 0 / 6 / 16); text 10 / 20 / 40 passages (`XD` 0 / 10 / 30); text joined into one chunk of ~1.3k / ~2.5k / ~5k tokens (g10 / g20 / g40, built with `scripts/probes/build_groups.py` or a new builder); mixed: 0 / 6 / 16 added items of the other modality (`scripts/longdoc/build_mixed.py`, to write) | model, questions, PX, rate 8, judge; the distractor draw is the same in both modes (arm A parity) | the same model and questions at the shortest size; `base` on the same rows at each size. Existing base files: 4 pages `mm_docqa_base_px1003520_mm_docqa_1500_1m.json`, 10 pages `mm_docqa_base_k10_px1003520_xd6_mm_docqa_1500_1m.json`, text `mm_docqa_base_k10_text_hotpotqa_1500.json` | Δ_len (below) | ~24 |
| U. reading many `<MEM>`s | as L at the longest size, with ONE probe env per cell: `KEEPF=0.125 KEEPM=attn_global`, `RAW=gold`, `RAW=route1` | slots kept / gold item raw / picked item raw | as L | R plain at the same size, and the same probe at the short size (exists for documents: eighth .613, job 189057; route1 .820, job 188441; gold-only .632, job 188440) | dilution D and picking (below) | ~12 |
| P. positions | eval flag `--mem_position_offset N` (to write, patchable IF; env `MPOS`): shifts the decoder position ids of the `<MEM>`s and of everything after them by N | position only: N = the 20-page (40-passage) span minus the short span, taken from arm A | R; 4-page documents and 10-passage text; slot count | R plain. `MPOS=0` must give byte-identical predictions | offset cost ≤ −3 pt with the CI below 0, at s42 and at s43 → positions past the trained range cost accuracy | ~5 |
| T. why long-chunk training lost | `eval_mm_4b.sh` on the C256 / L1024 finetunes (symlinked): `GOLD=1` and `RAW=route1` on documents (PX=1003520); plain `text_hotpotqa_1500` (ten chunks, TOPK=10, CML=4096). `scripts/longdoc/mem_recall.py` (to write: a stride-aware port of `scripts/probe_mem_recall.py`, which is legacy-only per HANDOFF.md:138) on the two stage-2 models, 500 gold pages. `scripts/longdoc/ae_nll.py` (to write) on the two stage-1 models, 500 MS MARCO documents from row 1,000,000 on (never trained on), chunks of 256 and 1,024 tokens | stage-1 chunk length (the only difference between the chains) | recipes, rows, seeds, code md5 | C256 of the same seed with the same probe. Plain pairs exist (190270-190275 vs 188707-188712) | locate (below) | ~16 |
| E. a proper long-document yardstick (owner, 2026-11-17) | builders `scripts/longdoc/build_long_{text,pages,mixed}.py` + `build_long.sbatch` (to write; CPU, `-p gpu-be --gres=gpu:0`; downloads run inside that job, the way `scripts/build_mm_eval.py` fetched its sources); cells = `scripts/longdoc/eval_mnt.sbatch` (= `eval_mm_4b.sh` as in arm L, answer cap `MNT=512`, see below) | the set (text / pages / mixed), the length bucket (reported per bucket from one cell), the model | rate 8, PX=1003520, judge; ONE CML per comparison, the same for both models compared | uncompressed on the same rows; closed-book (uncompressed, no document) on the same rows | set soundness and the long-document gaps (below) | build 3 + plumbing 3 + soundness 6 + model cells (costed first) |
Arm E sets, written to `eval_data/longdoc_text_1500`, `eval_data/longdoc_pages_1500`, `eval_data/longdoc_mixed_1500`
(1,500 Q each where sources allow; else every usable question, with the smallest difference that n can show):
- one coherent document per question (never independent passages glued together), answer needing the document;
- text in length buckets of about 4k / 8k / 16k / 32k tokens; pages in buckets of about 10 / 20 / 40 pages;
- candidate sources (sizes and licences to check before use; a licence that forbids research use excludes a source):
  text — QuALITY (multiple choice, exact scoring), Qasper (whole papers), LongBench MultiFieldQA kept as reported-only;
  pages — MMLongBench-Doc, LongDocURL, MP-DocVQA whole documents; mixed — the same PDFs with some pages given as their
  text layer and the others as images (or another construction the wake justifies in STATE.md);
- one synthetic needle check per modality: a fact placed at 10% / 50% / 90% depth of a real long document at each
  bucket length, exact scoring (it measures capacity, it never decides a cause alone);
- a long text document reaches the compressor whole: split into consecutive CML-token pieces (pre-split in the set, or
  the collator's `chunk_docs` with `n_max_chunks` large enough). Without that the eval keeps only the first CML tokens
  (`pisco/mm_collator.py:457-462`). Arm A's cut check is re-run on every E set and must give 0%.
- the answer cap (2026-11-18): the shared 32-token cap cut ~61-74% of the uncompressed decoder's page answers
  mid-sentence (191634-191636, 300 rows each), so uncompressed scored below R. At 256 tokens the uncompressed answers
  still reach the cap on 7 / 300 page rows (2.3%) but 25 / 300 text rows (8.3%) (191669 / 191670; heuristic ≥ 600
  characters, no final punctuation). So ONE cap serves every arm E cell, every set, model and mode, closed-book and
  needles included: `MNT=512` through `scripts/longdoc/eval_mnt.sbatch` (a runtime copy of `eval_mm_4b.sh` with only the
  cap changed; the shared launcher is never edited). Cells pair only at the same cap; the 32- and 256-token cells are
  plumbing only. If the uncompressed text answers still reach the cap on ≥ 5% of the 300 rows at 512 (191777), the cap
  becomes 1,024 for every arm E cell and the closed-book cell re-runs at it, before any other soundness cell.
  [2026-11-18 22:10, corrected 2026-11-19 02:00: "reach the cap" is counted with the tokenizer, never by characters,
  with its padding OFF (`tokenizer.json` pads to a multiple of 64 and truncates at 1,024: call `no_padding()` and
  `no_truncation()` first); answer ≥ cap − 2 tokens. 191777: 7 / 300 = 2.3% → the cap stays 512. Characters vs tokens:
  256 8.3% vs 8.0%; 512 6.3% vs 2.3%. This rule was written after 191777 was seen (verifier); a cap is a token
  count, so the token count is its definition, and it is kept.]
- closed-book (soundness check 1) runs once on `eval_data/longdoc_closedbook` (every question once, no document; base
  mode `TOPK=0`): without a document the prompt is the same in every set and bucket, so each bucket's closed-book score
  is that file restricted to the bucket's questions (paired on the question id before the last `__`).
Model cells, in this order: R s42 then s43 (does the gap grow with length on real documents?); then the long-chunk
question on text: the C256 and L1024 finetunes of the same seed, both at CML 1,024 (the length L1024 was trained on),
paired, s42 and s43. A tie at CML 1,024 opens CML 256 for both as a second round, never in the same round.
Start order (a hypothesis, not a work order): A → J controls → T (it answers the owner's first sentence) → L → U → P.
Owner, 2026-11-17: arm E goes next, before U and P; L's queued and running cells finish as planned.
Mixed cells go with L and U.

Rules: every arm is a committed launcher + a committed cell spec, never raw overrides on the launch line. A
comparison with a difference not listed here is not a result. A relaunch repeats the original command exactly.
- Every eval cell runs on a symlinked model dir `expQ/longdoc/<cell>/model → <model dir>`, so its result lands in
  `expQ/longdoc/<cell>/eval/` and never overwrites a recorded file. A `base` cell writes to `expQ/base_qwen35_4b/eval/` only
  under a file name that does not exist yet (check with `ls`). New launchers are `scripts/longdoc/*.sbatch`, because the
  repo `.gitignore` drops new `*.sh` files (campaign_probes/CAMPAIGN.md:56-57).
- Before a probe env is used for the first time on a mode, set or path, read `main()` of `scripts/eval_mm_docqa.py` and
  confirm the flag reaches that branch. Still unverified: `XD` on text rows, `GOLD` / `RAW` on mixed rows, mixed rows in
  `base` mode. `STYLE=expand` together with `RAW=*` is refused (campaign_probes/CAMPAIGN.md:49-51).
- Every intent note records the md5 of `eval_mm_4b.sh`, `scripts/eval_mm_docqa.py`, `scripts/judge_mm_docqa.py` and
  `pisco/model.py`. Longchunk's values were 50abc167 / dfe32aa0 / 02bc8748 / 994cff3d (campaign_longchunk/STATE.md:41).
  Two cells pair only when they ran on the same code.
- A new eval set becomes a ledger `resource:` note as soon as it exists (path, size, how it was built).

## Decision gates
Baseline: the same model on the same questions at the shortest input, with the probe off (for arm T: C256 of the same
seed with the same probe). Metric: paired judge accuracy in pt. Margin: a cause needs 3 pt with the 95% CI on the
far side of 0; "not this cause" needs the CI inside ±3 pt (the same bars as campaign_longchunk/CAMPAIGN.md:69-71). Noise band 1.2 (in the metric's units) vs the
baseline: below it nothing is a result; a tie triggers a repeat with another seed/split; above it once still needs
the repeat to be "adopted". Decide arms as soon as their cells complete — do not wait for the whole grid.
- **Δ_len (arm L, a NEW metric, n = 1,500 pairs per size):** for each question, d = (R − U at the long size) − (R − U at
  the short size). Report the mean, a paired bootstrap 95% CI, and a sign test on the non-zero d; its known-positive
  control is arm J's. A length loss is Δ_len ≤ −3 pt with the CI below 0, at s42 and at s43. No length loss is the CI
  inside ±3 pt at both seeds. Report each size's R, U and gap next to it.
- **Dilution D (arm U, a NEW metric):** D = (eighth − plain at the long size) − (eighth − plain at the short size), same
  statistics. D ≥ +3 pt with the CI above 0, at both seeds → the decoder reads many `<MEM>`s worse than few, and the
  most-attended eighth recovers that share. Picking: `RAW=route1` − `RAW=gold` at the long size ≤ −3 pt with the CI
  below 0 → the pick limits route-then-expand at length (already seen on whole documents: −.115 at 11-20 pages, 500 Q).
- **Locate (arm T, both seeds):** the documents deficit L1024 − C256 (−6.2 / −4.3 pt in the plain cells) sits:
  - IN THE COMPRESSED PAGE if it stays ≤ −3 pt (CI below 0) with `GOLD=1` and shrinks inside ±2 pt with `RAW=route1`;
  - IN MULTI-PAGE READING if it shrinks inside ±2 pt with `GOLD=1`.
  The transcription port then tells whether L1024's page `<MEM>`s hold fewer words (paired per page, ≥ 3 pt fewer, sign
  test p < .05). The AE probe tells whether stage 1 learned a length the later stages never use: L1024 better at 1,024
  tokens AND worse at 256, both CIs excluding 0, paired per document. [2026-11-17 07:30, written before the cell:
  the same AE probe on the two STAGE-2 models (`ae_nll.sbatch 500 s2`, text in the stage-2 rehearsal format) splits
  where the page loss enters: L1024 − C256 at `s256` > 0 with the CI above 0 and sign p < .05 → stage 2 erodes
  L1024's detail on text too (a general capacity loss in stage 2); CI touching 0 or below → the loss is specific to
  image pages. Controls: `wrong256` far worse than `s256` for both models, self-pair 0.] On text, L1024 − C256 on ten short chunks, set
  against the recorded g10 pairs, tells whether its text loss depends on the chunk length used at eval.
- **Arm E (owner, 2026-11-17), set soundness first, per set and per bucket:** (1) uncompressed with the document beats
  closed-book by ≥ 20 pt (else answers come from the model's memory: the bucket is dropped, the set rejected if most
  buckets go); (2) arm A's cut check 0% and parity 100%; (3) the needle check scores ≥ 90% uncompressed at that length
  (else the length is past what the base model reads, and the bucket is dropped). Multiple-choice rows are scored on
  the chosen option, and the judge still runs on them. Then: the gap R − uncompressed per bucket, and Δ_len vs the
  shortest bucket (arm L's statistics and bars). Long-chunk: L1024 − C256 at CML 1,024, paired per question, overall
  and per bucket: ≥ +3 pt with the CI above 0 at both seeds → long-chunk training helps on long documents (the g10 loss
  was a yardstick fault); CI inside ±3 at both → no effect; ≤ −3 with the CI below 0 at both → a real loss.
- Health per cell (a 0.0 that is a bug looks like a result):
  - n rows = the set size, a `judge` on every row, `judge_unparseable` 0, no "exceeds --decoder_max_length" error;
  - `mems_used` within ±1% of arm A's prediction for that variant (pruning, RAW and GOLD cells report the ratio instead, by design);
  - node outside homer-[005-009], recorded together with the attempt that wrote the file (known_issues.md row 21).
  A cell that fails is `closed "health"` and relaunched once. Arm A's health: its predicted `<MEM>` total for R's plain
  4-page document cell equals the recorded 646,042 (campaign_probes/STATE.md:42). The transcription port must reproduce the
  legacy probe on the checkpoint `probe_recall_4b.sh ae_s2` used (71% of words rebuilt, architecture_review.md:105) within
  ±2 pt before any use on a stride-8 model. The AE probe must give 0 for a model paired with itself.
- Small cells are plumbing: a `LIMIT=300` cell proves the path runs. It never decides an arm, even when it agrees.
- A NEW metric (one the wake designs itself) closes nothing until its row here states its n (cells or pairs) and
  a known-positive control it recovers (variety shipped a null built from one seed pair). Next to any threshold count, report the paired per-item test —
  navmem's count showed nothing where a paired sign test found the one real effect (98 closer / 59 farther, p=0.002; control p=0.22).
- Headline claims in STATE.md and the ledger are prefixed `provisional:` until the next critic pass, or until the
  next wake re-derives them from `expmon grid`.

**Feasibility gate (first, alone, 0 GPU-h):** arm A's audit. Its numbers, per gating variant and per setting an earlier
long claim used:
(1) cut rate = the share of questions whose gold answer appears in the raw documents but not in what the compressor
receives after the eval's truncation (text, mixed), or whose prompt exceeds the planned `DML` (all sets);
(2) parity = the share of questions where `pisco` and `base` receive the same documents in the same order.
Threshold: cut rate ≤ 1% and parity 100% on every gating variant, plus arm A's health number. If (1) or (2) fails
on a gating variant, the campaign STOPS with that as the finding: "the evaluation does not feed the long input it
claims" (an evaluation bottleneck). BLOCKED.md names the sets, the counts and the recorded claims it voids; NEXT
`waiting-owner`. The fix belongs to the pisco chats (the eval code is shared), never to the wake. A failure on an
earlier claim's setting is recorded as a finding (the claim it voids) and does not stop the campaign. A failed health
number is a bug in the audit: fix it and re-run; it is not a finding. Next, before any arm's full cell: one `LIMIT=300`
plumbing cell per new variant (the 20-page prompt fits on one A100, and the flag reaches its branch).
- **Diagnose on the whole set before choosing a fix (owner 12 Nov).** A fix target is a CAUSE shared by many items,
  not one failing item and not one check. Before a fix arm opens: (1) the set is sound: its size is stated against
  the effect it must show, every setting varies, each check both passes and fails, a fresh set is held back; (2) the
  failing items are grouped by shared condition across checks, and causes are ranked by items repaired, each lead
  confirmed on the fresh set; (3) the fix is judged on the large set, paired, with the compared count stated: nothing
  else worse, no item lost. A check that fails on (almost) every item, or on the accepted references, is a fault of
  the check: source its limit before any fix.
- **Promising, then ONE repeat (owner 12 Nov).** An arm whose paired gain clears the usual bar (sign test p < 0.05,
  positive in every seed, no guard subset losing beyond its floor) but not the noise floor is PROMISING, never closed
  as a fail. It gets exactly one repeat on NEW items (another split / episode set / query set, never more seeds of the
  same items), with its own baseline re-run on those items, size fixed before launch, judged once at the usual bar. It
  passes if the repeat clears that bar; otherwise it is closed with both reads. Never extend a read until it passes,
  never a second repeat.
  The new items here are, for text, `eval_data/text_mrqa_ood_1500` (other questions) and, for documents,
  `mm_docqa_longdoc_1m` (500 Q). Mixed uses a second build seed drawn from the other half of the source questions.
- **On acceptance:** a finding is delivered, nothing is switched on. Four of the five parts of a proper judge are missing:
  (1) judge frozen for the wake: only by prose ("never" in ## Patchable; no hook or CAMP_READ_DENY covers `scripts/judge_*.py`, `scripts/paired_judge.py`);
  (2) controls: the no-op is refused (a file paired with itself gives 0), but no tool plants a gain: MISSING;
  (3) a second program tested on planted cases: MISSING (`scripts/judge_agreement.py` measures agreement, not planted
  cases; the human sheet is unlabelled, HANDOFF.md:58);
  (4) paired on every item: yes (`paired_judge.py`), but every campaign has reused these eval sets: no untouched set;
  (5) automatic revert on a worse yardstick: MISSING.
  So an inference-time mitigation this campaign accepts (for example attention pruning on long inputs) stays opt-in (an
  `eval_mm_4b.sh` env, off by default) and reaches the owner as ONE question in BLOCKED.md. A corrected eval setting is
  a hand-off to the pisco chats (it moves every campaign's yardstick); the wake never applies it itself.

## Budget and caps
- cells for the whole campaign: 70 · GPU-hours: 60 (owner, 2026-11-16). The estimate uses measured walls: a judged 1,500-Q document
  cell took 28-98 min, photos 18 min, text 17 min (campaign_probes/CAMPAIGN.md:116-120). About 62 cells are planned
  (A 2, J 3, L ~24, U ~12, P ~5, T ~16), mostly ≤ 1.6 h each. Uncompressed 20-page cells are unmeasured: the first one
  measures, with `--time 4:54:00` until then. CPU jobs count 0 GPU-h. A cell = every `expmon launch --exp longdoc`
  submission, builds and CPU jobs included; an automatic requeue is not a new cell.
- Arm E (owner, 2026-11-17): its builds, `LIMIT=300` plumbing and soundness cells fit inside the caps above. Before
  E's model cells, cost the grid (cells, and GPU-h from the measured plumbing walls and memory). If it passes the cells
  or GPU-hours left, write ONE BLOCKED.md question with that cost and NEXT `waiting-owner`; never trim the grid to fit.
  Uncompressed 32k-token and 40-page prompts are unmeasured: their plumbing cell measures wall and memory first.
- LLM wakes: ≤ 50. Chain plumbing → full cell → paired test inside one wake; wait with `wake-when job:…`.
- at most CAMP_MAX_SUBMIT_PER_WAKE new submissions per wake (campaign.conf: 4)
  Set to 4 by the owner on 2026-11-16; one wake is also capped at $8 (CAMP_MAX_USD).
- every launch: `--time` = measured cell wall × 3 (from the first cell of that arm; never hours for a minutes-long
  cell: an oversized request loses backfill priority); `--mem` from the profile rung's MaxRSS × 1.5
- one GPU per cell, `-p gpu-be --exclude=homer-[005-009]` by default (homer-008 changes image answers;
  campaign_probes/CAMPAIGN.md:75-89). At most 2 cells on `gpu` at once (the owner's 8-GPU cap is shared).
- no launch below CAMP_DISK_FLOOR_GB free; nothing new after CAMP_UNTIL
- **Owner checkpoint:** none before cells (owner, 2026-11-16). Nothing trains, every cell is an inference cell of a few GPU-hours at
  most on existing models, and none is irreversible. The campaign ends at one checkpoint: the ranked causes reach the
  owner as ONE BLOCKED.md question, with the training arm each cause motivates (`waiting-owner`).
- **Live sibling `longchunk`** (campaign_longchunk/CAMPAIGN.md, campaign.conf) and what it uses:
  - GPUs: its stage 1 took 4 A100 on `gpu`, one run at a time. None run now: 189693 was cancelled on 2026-11-16 18:23
    (EXPERIMENTS.md:1100). Its 1-GPU stage 2, finetunes and evals ran on gpu-be with `--exclude homer-[005-009]`.
  - Machines: homer-019 (stage 1); homer-012, 013, 016, 019 and 020 (evals).
  - Engine trees / worktrees: none of its own. It shares this checkout (branch `arc`, with uncommitted
    `pisco/model.py`, `scripts/eval_mm_docqa.py`, `eval_mm_4b.sh`) and its pixi env.
  - Data: `data/msmarco_docs` and `expQ/longchunk/**`, plus the shared stage-2, finetune and eval sets.
  - Wake quota: the same account. Heartbeat 24 h, CAMP_MAX_USD 8, NEXT `wake-when job:189693 max 44h`, a REQUEST
    pending. CAMP_MAX_SUBMIT_PER_WAKE 6; commit paths `expQ/longchunk scripts/expmon_notify.sh`.
  This campaign therefore:
  - writes nothing under `expQ/longchunk/**` or `campaign_longchunk/**`; it reads that campaign's models and result
    files through `expQ/longdoc/` symlinks;
  - never kills, requeues, relaunches or verdicts its jobs;
  - patches no shared eval file while a longchunk cell is queued or running;
  - keeps at most 2 cells on `gpu`, and keeps its wakes rare.
  (The probes and freeze campaigns are in STAND-DOWN.)
- **Live sibling `qafreeze`** (armed 2026-11-17; campaign_qafreeze/CAMPAIGN.md, campaign.conf) and what it uses:
  - GPUs: one 1-GPU finetune at a time on `gpu` (A100) inside the owner's shared 8-GPU cap; its evals and CPU checks on
    `gpu-be --exclude=homer-[005-009]`. Machines: the shared homer A100 nodes, none reserved.
  - Engine trees / worktrees: none of its own; this checkout (branch `arc`) and its pixi env. Wake quota: the same account.
  - Dirs: `scripts/qafreeze/**`, `expQ/qafreeze/**`, `campaign_qafreeze/**`; it reads R's model dirs. Its evals pair
    only on the code md5 50abc167 / dfe32aa0 / 02bc8748 / 994cff3d (campaign_qafreeze/CAMPAIGN.md:34).
  This campaign therefore writes nothing under its dirs, never touches its jobs, counts its `gpu` finetune against the
  8-GPU cap before using `gpu`, and patches no shared eval file while a qafreeze eval is queued or running.

## Deliverable — how results reach the owner
`docs/longdoc_diagnosis.md`, rewritten (never appended) whenever an arm decides, holds four things:
(1) the bottleneck table: candidate | probe | n (pairs) | number (paired, 95% CI, McNemar / sign-test p) | gate | verdict | questions repaired.
(2) one length-curve table per modality: input size (pages / passages / tokens) | `<MEM>` slots per question |
compression rate (8 tokens per slot; slots per page) | compressed judge accuracy ↑ | uncompressed judge accuracy ↑ (same
questions) | gap ↑ (pt) | Δ_len vs the shortest size (95% CI).
(3) the data audit table: `<MEM>`s per decoder prompt (p50 / p99 / max) and position span, for each training stage, next to each eval variant.
(4) one figure, `expQ/longdoc/figures/length_curve.png` (`scripts/longdoc/plot_length.py`, to write): judge accuracy ↑
against `<MEM>` slots per question, compressed and uncompressed, one line per modality.
Every decided arm also gets one `expmon note` with its headline number. The wake produces it as soon as an arm decides; a result that exists only as a number in a log is not
delivered. Lacking tooling for it is a gripe (`expmon gripe`), not an excuse. The default page is
`expmon display --exp longdoc` (Settled / Open, the family's results table with live state, every clip and
image under the runs' result dirs); name here only what goes beyond it.

## Pre-authorised actions (the agent may do these without asking) and their loop guards
| situation | action | guard |
|---|---|---|
| a cell FAILED / NODE_FAIL / TIMEOUT for a transient cause (node failure, a wall-time misestimate) | relaunch the same cell, `--time` re-derived from measured cells | max 2 relaunches per cell, then BLOCKED.md |
| `expmon check` prints `node <n>: N fast deaths` (a poisoned node killing cells at model load) | relaunch the dead cells with `--exclude=<n>` before the launcher | 1 exclude-relaunch per cell |
| a cell's output is empty or unparsable while the job COMPLETED | verdict `closed "<why>"`, relaunch once with `--time` × 2 | never a third time |
| a grid cell of a listed arm is missing | `expmon launch --exp longdoc …` (or `sweep --dry-run`, then for real) | within CAMP_MAX_SUBMIT_PER_WAKE |
| a cell PENDING > 2 h for Resources with `--mem` above the profile rung's MaxRSS × 1.5 while cheaper twins run | `scontrol update JobId=… MinMemoryNode=…` (downward only) | never touch a job the campaign does not own |
| a cell PENDING > 6 h for Resources with an oversized `--time` | `scontrol update JobId=… TimeLimit=…` (downward only) | never touch a job the campaign does not own |
| the expmon daemon is gone | `expmon daemon` | — |
| a straggler FAILED in seconds with a benign signature (known_issues.md) | ignore, note once | — |
| a reproduced crash whose fix is ≤ CAMP_PATCH_LINES lines inside a "## Patchable" path | apply it, smoke on the cheapest cell, record `<file> — <traceback line>` in STATE.md; the tick commits it (its path must be in CAMP_COMMIT_PATHS) | 1 patch per symptom, then BLOCKED.md |
| a probe needs a new eval flag, eval set, builder or audit | write it under `scripts/longdoc/` (free), or add ONE flag to `scripts/eval_mm_docqa.py` + `eval_mm_4b.sh` (patchable IF, below); smoke with `LIMIT=300`, then run the full cell | 1 flag per probe |
| a cell PENDING > 6 h on gpu-be for Priority | `scontrol update JobId=… Partition=gpu,gpu-be` | at most 2 campaign cells on `gpu` at once; never a job the campaign does not own |
| a requeued eval overwrites a complete judged file | keep `<file>.kept_<job>_<node>_<hhmm>` as known_issues.md row 21 says | — |
Anything else: BLOCKED.md and stop.

## Patchable
campaign.conf sets CAMP_PATCH_LINES=40 and CAMP_COMMIT_PATHS='scripts/longdoc expQ/longdoc docs/longdoc_diagnosis.md'
(owner, 2026-11-16); the scripts this law marks "to write" need both, and the conf wins over this section. Three settings:
- freely patchable (the campaign's own code; this is where the wake iterates): `scripts/longdoc/**`, `expQ/longdoc/**`,
  `campaign_longdoc/**`, `docs/longdoc_diagnosis.md`, and new sets `eval_data/longdoc_*`.
- patchable IF the invariant holds, with its output pasted in STATE.md for that wake: `scripts/eval_mm_docqa.py` and
  `eval_mm_4b.sh`, to add ONE probe flag or env. Default behaviour must stay byte-identical. The invariant:
  `pixi run python test_positions.py` and `pixi run python test_mm_collators.py` pass through SLURM (`-p gpu-be --gres=gpu:0`),
  AND a `LIMIT=300` plain cell of 184778 on `mm_docqa_1500_1m` gives 300 predictions equal to the first 300 of
  `expQ/ft_mm_4b_184778/eval/mm_docqa_pisco_px1003520_mm_docqa_1500_1m.json`. Allowed only while no longchunk cell is
  queued or running (its pairs need unchanged md5).
- never: `pisco/**`; every training launcher (`qwen35_4b_pt.sh`, `pt_mm_4b.sh`, `ft_mm_4b.sh`,
  `expQ/longchunk/pt_text_4b_long.sh`); `scripts/judge_*.py`, `scripts/judge_prompts.py`, `scripts/paired_judge.py`;
  `scripts/probe_mem_recall.py` (port it, never edit it); existing `eval_data/` sets; `data/**`; recorded result jsons
  (re-judge copies only); `expQ/**` other than `expQ/longdoc/`; `campaign_longchunk/**` and `expQ/longchunk/**`;
  `EXPERIMENTS.md` except through `expmon note`.
The WRITE BOUNDARY is not prose: `CAMP_WRITE_ROOTS` in campaign.conf
(default: this repo) is enforced by the wake hook, which also refuses `~/.claude`, campaign.conf, the tick's state dir
and sibling campaign dirs whatever this section says.

## The wake owns the science
The arms table is a starting hypothesis, not a work order: read the evidence, launch the cells (the owner does not),
propose arms the table lacks (one knob off the control, gate written BEFORE the cell), close decided arms, and amend
this law when the evidence refutes it — a wrong prediction is kept and marked, never quietly deleted. A campaign seeded
deliberately unfinished names the properties the wake may not weaken while fixing (GUIDE.md).
Prefer the probe that would change a training decision. Each verdict names the training arm it motivates, or the idea
it closes. Training itself goes to the owner (hard rule 8). The prediction written before any cell, kept and marked
if it turns out wrong: on a fixed model, the image gap is flat from 4 to 20 pages; the L1024 deficit sits in the
compressed page. [2026-11-17, provisional: the second half CONFIRMED by arm T at both seeds — gold page only −6.7 /
−4.9 pt, picked page raw +0.1 / −0.1 pt, 1,500 pairs each; the first half not tested yet (arm L).]

## Hard rules
1. Submissions only via `expmon launch --exp longdoc` / `expmon sweep --exp longdoc`; never raw sbatch (enforced).
2. Cancel only campaign-owned jobs, only via `expmon kill`. Never `rm` an output dir — rename it `VOID_<name>` (enforced).
3. Never edit campaign.conf, crontab, hooks, `~/.claude`, the tick or the watchdog; never relax a rule here.
   Code edits only as "## Patchable" allows. A `REQUEST` file in this dir is the owner speaking (`expmon campaign
   resolve`): execute it before anything else, then delete it; nothing else may create one. Rules 1–3 are enforced by
   the wake hook: a refusal is a signal to write BLOCKED.md, never to find another spelling. The tick commits for you.
4. Hypothesis ≠ conclusion: write "on track to", "consistent with"; a conclusion needs its gate AND the repeat.
5. Report n (cells, pairs, queries) with every number; a count without its denominator is not a result.
6. End every wake with one line in NEXT (`done` | `wake-when job:<ids> [max <N>m]` | `wake-in <min>` | `waiting-owner`), or nothing while work is left;
   with minutes-long cells running, `wake-when job:<their ids> max 60m` wakes you as they end, not on a guess.
7. Settled by campaign 'longchunk' (hold constant what won, do not retry what closed): pre-campaign inference probe of 2026-11-06 (long-chunk grouping g1/g3/g10, no training); results in the ledger (spread g10 .617 vs g1 .579; tail .399); not a cell of the 2026-11-10 campaign — jobs 185499, 185500, 185501 … 185603 (16 cells), closed; data check, SmolLM2: 23.8% of docs >= 1,024 tokens, 8.3% >= 2,048 -> corpus rejected for long chunks (feasibility gate a) — job 188339, closed; data build + check, MS MARCO docs: 43.4% >= 1,024, 20.4% >= 2,048 -> corpus adopted; data/msmarco_docs built (feasibility gate a PASSED) — job 188358, closed; feasibility gate (b) PASS at L2048 (CML 2048, DML 4096, BS 3 x ACCUM 16 x 4 A100-80GB): 50/50 steps, exit 0, GPU memory peak 288 GB over 4 GPUs (~72 GB each, sacct gpumem), host RAM 108 of 160 GB, GPU util 99.5%; 58.6 s/step steady (steps 10-49, 39 steps in 2287 s); full cell 5,208 steps = 84.8 h -> --time 106 h, 339 GPU-h; eval_loss 2.026 at step 50 (9,600 docs) — job 188420, closed; fit + speed test PASS at L1024 (CML 1024, DML 2048, BS 6 x ACCUM 8): 50/50 steps, exit 0, GPU memory peak 285 GB over 4 GPUs (~71 GB each), host RAM 96 GB, util 99.7%; 42.1 s/step steady (steps 10-49, 1641 s); full cell 5,208 steps = 60.9 h -> --time 76 h, 244 GPU-h; eval_loss 2.096 at step 50 (9,600 docs) — job 188421, closed; C256 stage 1 COMPLETED 16:43 h, 1M docs (5,208 steps); held-out loss .801/.877/1.007 at steps 2000/4000/5208 (64 rows); SmolLM2 reference 180486 had the same end rise .924/.920/1.023, so not a collapse; train loss ~.94 — job 188422, closed
8. No training of any kind here (a finetune, a stage, an RL step). A cause this campaign confirms reaches the owner as a
   proposed training arm. Dead ends: do not re-run these as arms (their recorded files MAY be reused as short-size points):
   - long-chunk stage 1 at 1,024 / 2,048 tokens (campaign_longchunk/STATE.md:17-25; the owner's stop of 2026-11-16);
   - from campaign_probes (CAMPAIGN.md:39-45): evidence-then-answer on documents (188439 / 189058); route then densify at
     rate 4 (188590 / 188591); gold-only as a fix (188440 / 188459); OCR words for photo text (189059); passage order
     (188473 / 188513); the no-oracle top-3 grouping (188521 / 188546); per-head scales (185581); PPE as an inference
     probe; untrained slot dropping as a coarse level;
   - from HANDOFF.md and docs/: compression rate 16 or 32; the tail layout on long chunks (−.169); a linear or moving
     connector; a density-adaptive rate; the `<MEM>` mixer; a bidirectional compressor on Qwen3.5; decoder-side stride
     alone; the coarse-to-fine slot order;
   - from campaign_freeze (STATE.md:9-24): a decoder-only QA finetune (compressor frozen) costs ~3 pt. Settled.
9. Never touch the sibling `longchunk`'s jobs, dirs or outputs (Budget and caps). Never touch the expmon daemon job
   188764 except through `expmon daemon`.
10. Never compute on the login node: anything that runs longer than a few seconds goes through `expmon launch`. Stdlib
    python on small JSON is fine (`paired_judge.py`, `length_effect.py`).

## Wake checklist (expmon grid tooling)
1. `expmon grid longdoc` — where every cell stands (state, verdict, factors, RESULT).
2. `expmon verdict <jid> closed|best "<why>"` for decided cells/arms.
3. Fill missing cells: `expmon launch --exp longdoc …` (or `expmon sweep --exp longdoc --dry-run …`, then for real).
4. `expmon probe <jid> <metric>=<val>` when a cell's number is not picked up by the results script.
5. Rewrite STATE.md and `docs/longdoc_diagnosis.md`; write NEXT.

## Escalation
BLOCKED.md (what, why, options, recommendation) and stop. The watchdog mails herve.dejean@naverlabs.com and writes ALERT for what
the loop cannot fix itself: stale ticks, wakes failing fast (login), no GPU-hour for CAMP_IDLE_H, a rejected
push, unpushed work, low disk, missing cron or daemon.

## Close-out (after CAMP_UNTIL)
`expmon campaign report` (one screen; its Settled / Open block is the successor's brief: `expmon campaign init <label> --from <this dir>`) → `expmon campaign retro` → fill its Lessons → `expmon gripe -c retro "RETRO longdoc: WENT OK: … WENT NOT OK: … NET: …"`
(one line; the cross-cluster feedback digest scores campaigns from it) → the OWNER pastes this dir's SYNC.md (written by retro) into ~/.claude/SYNC.md — never the wake (hard rule 3) → `expmon campaign disarm`.
