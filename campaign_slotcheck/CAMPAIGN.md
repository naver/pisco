# CAMPAIGN slotcheck — pisco   (the law; the agent re-reads this every wake; rewrite, never append)

created 2026-11-18 · kind **probe** · root `/beegfs/scratch/user/hdejean/pisco` · window until **2027-02-16** (launches allowed through that day 23:59) · owner herve.dejean@naverlabs.com · wakes run `claude-opus-5`

A probe campaign runs many short inference/evaluation cells (minutes each) and decides on counts or metrics
against a baseline cell — not on training curves. The binding constraint is submissions per wake and queue
time, not GPU-hours (navmem: 0.4 of 12 GPU-h used, cells of 1m42–3m17).

## Mission
Successor of campaign 'longdoc' (its report: /beegfs/scratch/user/hdejean/pisco/campaign_longdoc/REPORT.md). Open at hand-off: rejudge.sbatch — job 191629 COMPLETED, no verdict; CML=1024 PX=1003520 TOPK=1000 — job 191634 COMPLETED, no verdict; CML=1024 PX=1003520 TOPK=1000 — job 191635 COMPLETED, no verdict; CML=1024 DML=5612 PX=1003520 TOPK=1000 — job 191636 COMPLETED, no verdict.
**Those four jobs belong to the LIVE sibling longdoc: never verdict, relaunch, kill or re-use them here.**
**Question (owner, QUESTION.md):** "no progress has been done after many tries to improve the models. double check
again training set, models architecture, evaluation dataset and protocol." The document loss — judge accuracy **.610 /
.615 compressed vs .809 uncompressed** on the same 1,500 questions (`eval_data/mm_docqa_1500_1m`;
campaign_qafreeze/CAMPAIGN.md:12) — sits in what ONE page's `<MEM>` slots hold, and no training arm so far aimed at it.
The owner approved four checks and nothing else ("yes go 1-4 with camp", QUESTION.md:4): diagnosis only — inference
probes, CPU audits and ONE small overfit training; no full training run.
**The number that decides it:** per check, one number with its regime (below); the campaign's answer is the four
verdicts ranked by how much of the 19.9-pt document gap each one explains, plus the one-line consequence of each for
the stage-2 rebuild that would follow.
**Regime of every number:** judge accuracy in [0, 1], pt = .01, Qwen3-8B rubric judge (`scripts/judge_mm_docqa.py`),
greedy decoding, paired per `query_id` by `scripts/paired_judge.py` (bootstrap 95% CI + exact McNemar). Documents
`eval_data/mm_docqa_1500_1m` (4 pages, PX=1003520, 430.7 slots per question = 646,042 / 1,500,
docs/probe_campaign_2026-11.md:6-7). Models: R = `expQ/ft_mm_4b_184778/model` (seed 42) and
`expQ/ft_mm_4b_184780/model` (seed 43); stage-2 start `expQ/archq3/pt_mm_5M_spread/model`; uncompressed =
`eval_mm_4b.sh … base` on the same rows (recorded files in `expQ/base_qwen35_4b/eval/`). Owner's standing rules: probe
before training; one variable per cell; text, images and mixed for every capability; every table names its metric and
direction, carries a compression-rate column (tokens per slot, slots per page) and the uncompressed decoder on the
same questions.
If it replicates or extends a paper, `papers/<slug>.md` (`expmon paper`, skill `paper`) is the source of every number
quoted here — cite the card, not the PDF.

## Arms — what varies, what is held constant
| arm | cell launcher (committed) | factors (what varies) | held constant | baseline cell | gate (metric vs baseline) | cells |
|---|---|---|---|---|---|---|
| C. finetune-set count (check 1) — **the feasibility gate** | `scripts/slotcheck/audit_ft_rows.sbatch` → `audit_ft_rows.py` (to write; CPU, `-p gpu-be --gres=gpu:0`), over every row of `data/mm_ft_docpages_20k_1m_teacher` (≈ 20k rows, campaign_qafreeze/CAMPAIGN.md:25) | nothing — recorded data | the committed collator (`MultimodalFineTuningCollator.example_chunks`, imported, never rewritten), `topk_docs=4` (pisco/mm_collator.py:489) | — | feasibility gate below, then "Check 1" | 1 (+1 re-run) |
| P. padding (check 2) | (a) `scripts/slotcheck/pad_probe.sbatch` → `pad_probe.py` (to write; 1 GPU): the same pages compressed with 0 / 17 / 63 left pad tokens; (b) `scripts/slotcheck/eval_pad64.sbatch` (to write): a wrapper that IMPORTS `scripts/eval_mm_docqa.py` and hands its collator `pad_to_multiple_of=64` (training's value, pisco/mm_collator.py:431; the eval passes None, scripts/eval_mm_docqa.py:90) — the shared file stays byte-identical | pad width (a); training padding on / off (b) | model, pages/questions, PX=1003520, rate 8, judge; same node, same code md5 | (a) the same input compressed twice (noise floor) + another page (planted difference); (b) the SAME wrapper with `pad_to_multiple_of=None`, re-run on this code and node, never the recorded 11-09 file | Check 2a / 2b | 2 + 4 |
| X. text slots beside image slots (check 3) | `scripts/slotcheck/transcribe_pages.{py,sbatch}` + `build_text_sets.py` (to write; they import the TRANSCRIBE prompt and `run_ref` of `scripts/probe_mem_recall.py` — run, never edit — and write `eval_data/slotcheck_docqa_{text,imgtext}_500`); cells = `eval_mm_4b.sh expQ/slotcheck/<cell>/model pisco "" <set>` with `TOPK`/`CML`/`DML` from the build's measured maxima | what is compressed: page image / page transcription as text / both | model R s42, the same 500 questions, rate 8, judge, PX=1003520; **no raw page ever enters the decoder prompt** | the same model and questions with image slots only, re-run on this code and node; uncompressed on the same rows (ceiling) | Check 3 | 4 build/shards + 6 |
| O. overfit capacity (check 4; the ONLY training) | `scripts/slotcheck/build_dense100.py` (100 dense pages from `data/mm_pretrain_pdfa_1m`, by page-text length) + `pt_overfit.sbatch` (written 2026-11-18: `pt_mm_4b.sh` with the recipe of `expQ/archq3/pt_mm_5M_spread/training_config.yaml`, task mix `p_image_ae=1, p_text=0, p_image_cont=0, p_desc=0, p_find=0`); recall by `scripts/longdoc/mem_recall.py` (exists, validated 360 / 360, EXPERIMENTS.md:1163 — run, never edit) through `scripts/slotcheck/recall.sbatch` | precision: bf16 / fp32 master weights (one variable) | start `expQ/archq3/pt_mm_5M_spread/model`, the same 100 pages, rate 8 (8 tokens per slot), 1,003,520 px | the SAME start model's recall on the SAME 100 pages before the overfit; uncompressed transcription of those pages (`--arm ref`) | Check 4 | 1 build + 1 smoke + 2 trainings + 4 recall |
Start order (a hypothesis, not a work order): C → P(a) → P(b) → O → X. X is last, not third, because its page
transcriptions are the campaign's biggest single cost (measured 12.6 s/page: 501 pages in 1:45:10, job 191170,
EXPERIMENTS.md:1177).

Rules: every arm is a committed launcher + a committed cell spec, never raw overrides on the launch line. A
comparison with a difference not listed here is not a result. A relaunch repeats the original command exactly.
- Every eval cell runs on a symlinked model dir `expQ/slotcheck/<cell>/model → <model dir>`, so its result lands in
  `expQ/slotcheck/<cell>/eval/` and never overwrites a recorded file. A `base` cell writes into
  `expQ/base_qwen35_4b/eval/` only under a name that does not exist yet (`ls`) and that `campaign_longdoc/STATE.md`
  does not list as planned. New launchers are `scripts/slotcheck/*.sbatch` (the repo `.gitignore` drops new `*.sh`).
- Every intent note records the md5 of `eval_mm_4b.sh`, `scripts/eval_mm_docqa.py`, `scripts/judge_mm_docqa.py`,
  `scripts/paired_judge.py` and `pisco/model.py` (2026-11-17: 50abc167 / dfe32aa0 / 02bc8748 / a3cc13b2 / 994cff3d;
  `eval_mm_4b.sh` b3d1cda1 since 17:11 — campaign_spread/CAMPAIGN.md:67-70). Two cells pair only on the same md5s and
  the same node image; image answers move on homer-008 and across code dates (campaign_probes/CAMPAIGN.md:75-89).
- Before using an eval env on a set or mode it was never used with, read `main()` of `scripts/eval_mm_docqa.py` and
  confirm the flag reaches that branch. This campaign adds NO flag to a shared file (hard rule 8).

## Decision gates
Every difference is PAIRED per question (or per page), same model, same node, same code. House bars, as
campaign_longdoc/CAMPAIGN.md:107: a cause needs 3 pt with the 95% CI on the far side of 0; "not this cause" needs the
CI inside ±3 pt. Noise band 1.2 (in the metric's units) vs the
baseline: below it nothing is a result; a tie triggers a repeat with another seed/split; above it once still needs
the repeat to be "adopted". Decide arms as soon as their cells complete — do not wait for the whole grid.
- **Check 1 (arm C; a count, n = every row, reported per source):** the share of rows with more than 4 documents, and
  of those the share whose first gold page sits at index ≥ 4 (dropped by `topk_docs=4`) while the teacher label was
  written from up to 8 pages (`scripts/relabel_mm_teacher.py:60`). ≥ 10% of image-source rows losing a gold
  page = a data fault worth fixing (hand-off: filter those rows or raise `topk_docs`); ≤ 2% = closed, not a cause;
  between = reported with its denominator, no cause. A count names a cause only beside a paired effect.
  Definition (fixed 2026-11-18 17:45, before the count was read): a row "loses its gold page" when EVERY gold position
  is ≥ 4 (= the first gold sits at index ≥ 4); "partial" (one gold kept, another dropped) is reported apart and does not
  gate. The gate reads the TRAINED rows (the first 20,000, `path:20000` in train.py); the whole set and
  `mm_ft_natural_8k_teacher` (the other 8,000 trained rows) are reported beside it.
- **Check 2a (arm P, a NEW metric, n = 120 pages × 3 pad widths):** per `<MEM>` vector, cosine and max absolute
  difference between the padded and unpadded compress of the same page. Controls, in the same cell: the same input
  repeated gives cosine 1.0000 / max abs 0 (no-op refused), another page gives cosine below .99 (planted difference) —
  without both the cell decides nothing. Inert = median cosine ≥ .9999 AND max abs ≤ 2× the repeat floor at every
  width → padding is not a cause and (b) is not run. Otherwise (b) says what it costs.
- **Check 2b (arm P, n = 1,500 pairs per seed, both seeds):** padded − unpadded judge accuracy on documents. ≥ +3 pt
  with the CI above 0 at both seeds → the eval's missing training padding costs accuracy (a hand-off to the pisco
  chats: it moves every campaign's yardstick, so the wake never applies it); CI inside ±3 pt at both → not a cause;
  1.2-3 pt with the CI above 0 → PROMISING, one repeat on photos `eval_data/mm_naturalqa_1500`.
  Before that repeat (tightened 2026-11-18, job 191781): a question whose compressor batch has ONE row takes the
  one-row pad leak when padded (median cos .953 even in fp32) — a path no training run uses. Count compressor rows
  per photo question first; one-row questions are reported apart and do not gate.
- **Check 3 (arm X, n = 500 pairs, seed 42; seed 43 only if the first read clears the bar):** text slots − image slots,
  and image+text slots − image slots, paired per question. ≥ +5 pt with the CI above 0 → the image slots lose
  content the page's own words keep → a text side channel beside the image slots is worth building (a stage-2
  proposal); CI inside ±5 pt → it buys nothing; ≤ −5 pt with the CI below 0 → the transcription path is worse. The bar
  is 5 pt, not 3, because one seed's 95% CI on 500 Q is about ±4 pt (scaled from .044-.053 at 1,500 Q,
  campaign_spread/CAMPAIGN.md:88). Report beside it: uncompressed on the same rows, slots per question per arm and the
  `mems_used` ratio (a different budget by design). Nearest prior read, a different thing: OCR tokens instead of pixels
  on PHOTOS lost −.434 (campaign_probes/CAMPAIGN.md:44) — that was OCR words on photos, not a transcription of
  document pages, and it does not predict this check.
- **Check 4 (arm O, word recall, n = 100 pages, paired per page, sign test):** stop the training when the mean loss over
  the last 50 steps is within 2% of the previous 50, or at `max_steps` 1,000, then measure the overfit model's
  word recall on those same 100 pages against their uncompressed transcription. ≥ 90% → slot capacity is fine
  and TRAINING is the limit (the rebuild is objective and data); ≤ 75% → a capacity or architecture limit (the rebuild
  must change slots per page or the read-out); between → reported, undecided. Controls: the start model's recall on the
  same pages before the overfit (the baseline; the recorded .647 / .566 belong to other chains' stage-2 models, jobs
  191219 / 191220 — never used as this baseline), the reference paired with itself = 100%, and 100 held-out
  dense pages whose recall must NOT rise, else the cell measured a general gain, not memorisation. fp32 masters are a
  second cell, one variable (precision), only if the budget line below still holds.
  Tightened 2026-11-18 22:00 (before any arm O cell ran): `max_steps` = the smaller of 1,000 and the steps that fit
  2 GPU-h at the 20-step smoke's measured s/step (minus the smoke's load time), fixed on the launch line; the plateau
  rule is read from the logged loss afterwards — not reached = the recall is reported as a LOWER bound of what training
  can store, never as a capacity verdict ≤ 75%. Ceiling control (added): the page's own text layer (the AE target)
  scored against the same uncompressed transcription (`textlayer_*.json` of the build); the 90% / 75% bars are read
  against this ceiling too, and a ceiling under 90% is reported beside the verdict. The training set repeats each page
  10 times (`pisco/train.py` moves `min(64, rows // 10)` random rows to its eval split; on 100 rows it would drop 10
  pages) — same 100 pages, same data path.
- **Health per arm** (a 0.0 that is a bug looks exactly like a negative result):
  - C: rows read = the set's row count; per-source counts sum to the total; `text_rag` rows (empty `gold_positions` by
    construction, scripts/build_mm_finetune.py:163) reported apart; the planted 6-doc row is counted as "gold dropped".
  - P: the `<MEM>` count per page identical across pad widths; eval cells 1,500 rows, a `judge` on every row,
    `judge_unparseable` 0, no "exceeds --decoder_max_length", `mems_used` within ±1% of the unpadded twin, node
    outside homer-[005-009] recorded with the file.
  - X: the set's row count, a `judge` on every row, 0 unparseable, no "exceeds" error, < 2% empty transcriptions, mean
    words per page reported; cut check 0% at the chosen `CML` (run `scripts/longdoc/audit_inputs.py` as a tool, md5
    recorded, never edited).
  - O: s/step and peak GPU memory from the smoke; the loss falls below its first value; `<MEM>`s per page constant;
    the recall port prints `mem_position_stride=8`.
  A cell that fails health is `closed "health"` and relaunched once.
- Small cells are plumbing: a `LIMIT=300` cell proves the path runs. It never decides an arm, even when it agrees.
- A NEW metric (one the wake designs itself) closes nothing until its row here states its n (cells or pairs) and
  a known-positive control it recovers (variety shipped a null built from one seed pair). Next to any threshold count, report the paired per-item test —
  navmem's count showed nothing where a paired sign test found the one real effect (98 closer / 59 farther, p=0.002; control p=0.22).
- Headline claims in STATE.md and the ledger are prefixed `provisional:` until the next critic pass, or until the
  next wake re-derives them from `expmon grid`.

**Feasibility gate (first, ALONE, 0 GPU-h): arm C — is the dropped-gold count askable from the record at all?**
(1) `data/mm_ft_docpages_20k_1m_teacher` loads and every image-source row carries `docs`, `doc_images`, `source` and a
non-empty `gold_positions`: ≥ 95% of image-source rows; (2) known-positive control: on 50 sampled rows
`example_chunks(..., topk_docs=4)` returns exactly `min(len(docs), 4)` "Document j:" blocks, and a PLANTED row of 6
documents with golds at indices 4 and 5 is counted as "gold dropped"; (3) the health numbers above. FAIL on (1) or (2)
→ the campaign STOPS with that as the finding: "the finetune set does not record where the answer page sits, so what
the collator drops cannot be measured from the record" — BLOCKED.md with the counts, NEXT `waiting-owner`. A bug in
the audit is not a finding: fix it and re-run.
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
  never a second repeat. The new items here: documents `eval_data/mm_docqa_retr10_1m` or `mm_docqa_longdoc_1m` (500 Q),
  photos `eval_data/mm_naturalqa_1500`, text `eval_data/text_mrqa_ood_1500`.
- **On acceptance:** nothing is switched on by the wake; this campaign delivers findings. Default-on is refused because
  four of the five parts of a proper judge are missing: (1) judge frozen for the wake — prose only ("never" below; no
  hook and `CAMP_READ_DENY=''` in campaign.conf covers `scripts/judge_*.py`, `scripts/judge_prompts.py`,
  `scripts/paired_judge.py`); (2) controls — the no-op is refused (a file paired with itself gives 0.000) but no tool
  plants a gain: MISSING; (3) a second program tested on planted cases: MISSING (`scripts/judge_agreement.py` measures
  agreement; the human sheet is unlabelled, HANDOFF.md:66); (4) paired on every item: yes
  (`scripts/paired_judge.py`), but every campaign has reused these sets: no untouched set; (5) automatic revert on a
  worse yardstick: MISSING. So: **opt-in, then ONE question to the owner.** A confirmed eval fault (check 2b) is a
  hand-off to the pisco chats; a confirmed data fault (check 1) is a hand-off; a stage-2 rebuild motivated by check 3
  or 4 is a proposed arm in BLOCKED.md, never launched here.

## Budget and caps
- cells for the whole campaign: **32** · GPU-hours: **30**. Owner-confirmed 2026-11-18, built from measured walls:
  a judged 1,500-Q document cell 28-98 min (campaign_probes/CAMPAIGN.md:118-120); uncompressed page transcription
  12.6 s/page (job 191170, EXPERIMENTS.md:1177); the AE recall probe 14 s/page (job 191169, 360 pages in 1:26);
  stage 2 ≈ 5,300 samples/h (19 h at 100k, campaign_spread/CAMPAIGN.md:155). Per check: C 0 (CPU) · P ≈ 6 · O ≈ 5 ·
  X ≈ 12 (2,000 pages of transcription ≈ 7 in 4 shards + 5 judged 500-Q cells) · slack ≈ 7. Over it: BLOCKED.md.
  A cell = every `expmon launch --exp slotcheck` submission, builds and CPU jobs included; an automatic requeue is not
  a new cell.
- LLM wakes ≤ 30. Chain plumbing → full cell → paired test inside one wake; wait with `wake-when job:<ids>`.
- Owner-confirmed 2026-11-18; the conf always wins over this law: the campaign needs `CAMP_PATCH_LINES=40`,
  `CAMP_COMMIT_PATHS='scripts/slotcheck docs/slot_checks.md'` and `CAMP_MAX_SUBMIT_PER_WAKE=4`. **As campaign.conf
  stands (`CAMP_PATCH_LINES=0`, `CAMP_COMMIT_PATHS=''`, `CAMP_MAX_SUBMIT_PER_WAKE=1`) the wake may write none of the
  scripts marked "to write" and the campaign cannot start.** The owner changes it by hand before arming.
- at most CAMP_MAX_SUBMIT_PER_WAKE new submissions per wake (campaign.conf; raise it — 2-minute cells do not need a 1-per-wake cap)
- every cell is 1 GPU (or `--gres=gpu:0` for CPU) on `-p gpu-be --exclude=homer-[005-009]`. The `gpu` partition is the
  owner's shared 8-GPU cap, held by the two live training siblings: use it only for a cell PENDING > 6 h for Priority,
  after `gpucap land a100 1`, at most 1 campaign cell at a time, never for the overfit training without re-deriving
  `--time`.
- every launch: `--time` = measured cell wall × 3 (from the first cell of that arm; never hours for a minutes-long
  cell: an oversized request loses backfill priority); `--mem` from the profile rung's MaxRSS × 1.5. A training's
  `--time` = measured s/step × steps × 1.25.
- no launch below CAMP_DISK_FLOOR_GB free; nothing new after CAMP_UNTIL
- **Owner checkpoint:** none before checks 1-4 — the owner approved exactly these four ("yes go 1-4 with camp",
  QUESTION.md:4), the only training is the 100-page overfit (≤ 2 GPU-h per cell, writes only under `expQ/slotcheck/`),
  nothing is irreversible and no shared file is touched; ONE checkpoint when the first transcription shard measures and
  the projected cost of arm X exceeds **12 GPU-h** — the wake writes that cost in ONE BLOCKED.md question with
  NEXT `waiting-owner` and never trims the set to fit; and ONE at the end: the four verdicts and the stage-2 rebuild
  each motivates reach the owner as ONE question.
- **Live sibling `longdoc`** (campaign_longdoc/CAMPAIGN.md, campaign.conf; 42 of 70 cells, ≈ 21 of 60 GPU-h used):
  1 GPU per inference cell on `gpu-be --exclude=homer-[005-009]`, at most 2 cells on `gpu`; no training (its hard rule
  8). Machines: the shared homer A100 nodes (homer-011 / 016 / 017 / 019 / 020 lately), none reserved. Engine trees /
  worktrees: none of its own — THIS checkout (branch `arc`) and its pixi env. Writes `scripts/longdoc/**`,
  `expQ/longdoc/**`, `docs/longdoc_diagnosis.md`, `eval_data/longdoc_*`, new names under `expQ/base_qwen35_4b/eval/`;
  MAY patch the shared `scripts/eval_mm_docqa.py` and `eval_mm_4b.sh` (invariant-gated). Wake quota: the same account;
  CAMP_MAX_SUBMIT_PER_WAKE 4, CAMP_MAX_USD 8, heartbeat 24 h; jobs 191629 / 191634 / 191635 / 191636 in flight.
- **Live sibling `qafreeze`** (campaign_qafreeze/CAMPAIGN.md, campaign.conf): 2 finetunes ≈ 10 GPU-h each, 1 GPU on
  `gpu` inside the shared 8-GPU cap, plus a 20-step smoke; 8 judged evals and CPU checks on
  `gpu-be --exclude=homer-[005-009]`; 55 GPU-h (80 if its arm H is released). No worktrees — THIS checkout and its
  pixi env. Writes `scripts/qafreeze/**`, `expQ/qafreeze/**`, `campaign_qafreeze/**`. Data: the shared 28k finetune
  rows — **`data/mm_ft_docpages_20k_1m_teacher`, the set arm C audits (read-only here)** — and the four gating eval
  sets. Wake quota: the same account; CAMP_MAX_SUBMIT_PER_WAKE 4, CAMP_PATCH_LINES 40, heartbeat 24 h.
- **Live sibling `spread`** (campaign_spread/CAMPAIGN.md, campaign.conf): inference and CPU cells on
  `gpu-be --exclude=homer-[005-009]`; its arm E is ONE training on `gpu` inside the same shared cap (30 GPU-h before
  its owner checkpoint, 95 in all). No worktrees — THIS checkout and its pixi env. Writes `scripts/spread/**`,
  `expQ/spread/**`, `campaign_spread/**`, `docs/spread_cases.md`, `eval_data/spread_*`; it adds no eval flag. Wake
  quota: the same account; CAMP_MAX_SUBMIT_PER_WAKE 4, CAMP_PATCH_LINES 40, heartbeat 24 h.
  **This campaign therefore:** writes nothing under `scripts/{longdoc,qafreeze,spread}/**`,
  `expQ/{longdoc,qafreeze,spread}/**`, `campaign_{longdoc,qafreeze,spread}/**`, `docs/longdoc_diagnosis.md`,
  `docs/spread_cases.md`, `eval_data/longdoc_*` or `eval_data/spread_*`; never kills, requeues, relaunches or verdicts
  one of their jobs; never edits `data/**` or the reference models (`expQ/ft_mm_4b_184778|184780/**`, `expQ/archq3/**`)
  — reads only; runs `scripts/longdoc/*.py` as tools with their md5 recorded; keeps every cell on `gpu-be`; checks
  `campaign_longdoc/STATE.md` before writing any new name under `expQ/base_qwen35_4b/eval/`; and patches no shared eval
  or judge file (longdoc may — record the md5s with EVERY cell and pair only cells that ran on the same ones).
  Campaigns `freeze` and `probes` are in STAND-DOWN and `longchunk` is closed: read their files, write nothing there.

## Deliverable — how results reach the owner
`docs/slot_checks.md` (to write), rewritten (never appended) whenever a check decides: ONE table per check, columns
naming the metric and its direction — check | cell | model | set | n (pairs) | the check's metric ↑ | baseline ↑ |
difference (95% CI, McNemar / sign p) | uncompressed judge accuracy ↑ on the same questions | compression rate
(8 tokens per slot; slots per page / per question) | health | verdict — and under each table ONE line on what it means
for the stage-2 rebuild that would follow. Arm C's table is per source. Every decided check also gets one `expmon note`
with its headline number. The wake produces it as soon as an arm decides; a result that exists only as a number in a log is not
delivered. Lacking tooling for it is a gripe (`expmon gripe`), not an excuse. The default page is
`expmon display --exp slotcheck` (Settled / Open, the family's results table with live state, every clip and
image under the runs' result dirs); name here only what goes beyond it.

## Pre-authorised actions (the agent may do these without asking) and their loop guards
| situation | action | guard |
|---|---|---|
| a cell FAILED / NODE_FAIL / TIMEOUT for a transient cause (node failure, a wall-time misestimate) | relaunch the same cell, `--time` re-derived from measured cells | max 2 relaunches per cell, then BLOCKED.md |
| `expmon check` prints `node <n>: N fast deaths` (a poisoned node killing cells at model load) | relaunch the dead cells with `--exclude=<n>` before the launcher | 1 exclude-relaunch per cell |
| a cell's output is empty or unparsable while the job COMPLETED | verdict `closed "<why>"`, relaunch once with `--time` × 2 | never a third time |
| a grid cell of a listed arm is missing | `expmon launch --exp slotcheck …` (or `sweep --dry-run`, then for real) | within CAMP_MAX_SUBMIT_PER_WAKE |
| a cell PENDING > 2 h for Resources with `--mem` above the profile rung's MaxRSS × 1.5 while cheaper twins run | `scontrol update JobId=… MinMemoryNode=…` (downward only) | never touch a job the campaign does not own |
| a cell PENDING > 6 h for Resources with an oversized `--time` | `scontrol update JobId=… TimeLimit=…` (downward only) | never touch a job the campaign does not own |
| the expmon daemon is gone | `expmon daemon` | — |
| a straggler FAILED in seconds with a benign signature (known_issues.md) | ignore, note once | — |
| a reproduced crash whose fix is ≤ CAMP_PATCH_LINES lines inside a "## Patchable" path | apply it, smoke on the cheapest cell, record `<file> — <traceback line>` in STATE.md; the tick commits it (its path must be in CAMP_COMMIT_PATHS) | 1 patch per symptom, then BLOCKED.md |
| a probe needs a new script, builder, set or audit | write it under `scripts/slotcheck/` (freely patchable); smoke with `LIMIT=300` or `--limit`, then run the full cell | never a shared eval / judge file (hard rule 8) |
| a cell PENDING > 6 h on `gpu-be` for Priority | `scontrol update JobId=… Partition=gpu,gpu-be` after `gpucap land a100 1` | at most 1 campaign cell on `gpu`; never a sibling's job, never the last free slot |
| the overfit training NaNs, or its loss sits at its first value for 2 probes | `expmon kill`, `expmon verdict <jid> closed "<why>"` | never relaunch the same cell |
| a requeued eval overwrites a complete judged file | keep `<file>.kept_<job>_<node>_<hhmm>` as known_issues.md row 21 says | — |
Anything else: BLOCKED.md and stop.

## Patchable
CAMP_PATCH_LINES in campaign.conf is what is enforced and it wins over this section (it is 0 today — see Budget). The
three settings (GUIDE.md):
- **freely patchable** (the campaign's own code, where the wake iterates): `scripts/slotcheck/**`, `expQ/slotcheck/**`,
  `campaign_slotcheck/**` (never campaign.conf), `docs/slot_checks.md`, new sets `eval_data/slotcheck_*` and
  `data/slotcheck_*`.
- **patchable IF an invariant holds**, with its output pasted in STATE.md for that wake: `scripts/slotcheck/*.sbatch` —
  header, paths, env plumbing, logging. Invariant: `bash -n` passes; for arm O the 20-step smoke's
  `training_config.yaml` differs from `expQ/archq3/pt_mm_5M_spread/training_config.yaml` only in the data path, the
  task mix, `max_steps`, the precision field of that cell and the output paths; for the padded-eval wrapper, the md5 of
  `scripts/eval_mm_docqa.py` and `eval_mm_4b.sh` is unchanged AND a `LIMIT=300` wrapper run with
  `pad_to_multiple_of=None` gives the first 300 predictions of
  `expQ/ft_mm_4b_184778/eval/mm_docqa_pisco_px1003520_mm_docqa_1500_1m.json` re-run on this code and node. A script a
  running or finished cell used is frozen for THAT cell.
- **never**: `pisco/**`, `test_*.py`, `scripts/eval_mm_docqa.py`, `eval_mm_4b.sh`, `pt_mm_4b.sh`, `ft_mm_4b.sh`,
  `scripts/judge_*.py`, `scripts/judge_prompts.py`, `scripts/paired_judge.py`, `scripts/probe_mem_recall.py`,
  `scripts/longdoc/**` and `scripts/probes/**` (run them, never edit), existing `eval_data/` sets, `data/**` (the
  audited finetune set is read-only), recorded result jsons (re-judge COPIES only), `expQ/**` other than
  `expQ/slotcheck/`, everything of the three live siblings, and `EXPERIMENTS.md` except through `expmon note`.
The WRITE BOUNDARY is not prose: `CAMP_WRITE_ROOTS` in campaign.conf
(default: this repo) is enforced by the wake hook, which also refuses `~/.claude`, campaign.conf, the tick's state dir
and sibling campaign dirs whatever this section says.

## The wake owns the science
The arms table is a starting hypothesis, not a work order: read the evidence, launch the cells (the owner does not),
propose arms the table lacks (one knob off the control, gate written BEFORE the cell), close decided arms, and amend
this law when the evidence refutes it — a wrong prediction is kept and marked, never quietly deleted. A campaign seeded
deliberately unfinished names the properties the wake may not weaken while fixing (GUIDE.md).
The prediction written before any cell, kept and marked if wrong: checks 1 and 2 are not causes (under 2% of rows lose
a gold page; padding is inert on the slots); check 4's overfit recall clears 90%, so the limit is training, not slot
capacity. Each verdict names the stage-2 training arm it motivates, or the idea it closes; training itself goes to the
owner (hard rule 9).

## Hard rules
1. Submissions only via `expmon launch --exp slotcheck` / `expmon sweep --exp slotcheck`; never raw sbatch (enforced).
2. Cancel only campaign-owned jobs, only via `expmon kill`. Never `rm` an output dir — rename it `VOID_<name>` (enforced).
3. Never edit campaign.conf, crontab, hooks, `~/.claude`, the tick or the watchdog; never relax a rule here.
   Code edits only as "## Patchable" allows. A `REQUEST` file in this dir is the owner speaking (`expmon campaign
   resolve`): execute it before anything else, then delete it; nothing else may create one. Rules 1–3 are enforced by
   the wake hook: a refusal is a signal to write BLOCKED.md, never to find another spelling. The tick commits for you.
4. Hypothesis ≠ conclusion: write "on track to", "consistent with"; a conclusion needs its gate AND the repeat.
5. Report n (cells, pairs, queries) with every number; a count without its denominator is not a result.
6. End every wake with one line in NEXT (`done` | `wake-when job:<ids> [max <N>m]` | `wake-in <min>` | `waiting-owner`), or nothing while work is left;
   with minutes-long cells running, `wake-when job:<their ids> max 60m` wakes you as they end, not on a guess.
7. Settled by campaign 'longdoc' (hold constant what won, do not retry what closed): launch fault, not a result: FAILED exit 2 in 0 s, log 'missing <8 paths joined>': the wake's zsh passed the 8 files as ONE argv element (unquoted $VAR is not word-split in zsh). Replaced by 191629 (12 files, argv checked = 13 elements) — job 191517, closed; arm E cut check PASS: cut 0 / parity 1.0000 on E_text_cml1024 (2,246), E_pages_cml1024 (4,221), E_mixed_cml1024 (4,221) and the 3 needle sets; the CML 256 control fails as it should (235 / 2,246 = 10.5% cut). Health null by design (--only E_ skips docs_p4; health passed in 191147) — job 191516, closed; arm E build OK: COMPLETED 0:0 31 min; LongDocURL 2,325 Q / 396 PDFs -> 1,407 usable Q on 387 PDFs; pages 4,221 rows (1,407 x p10/p20/p40), mixed 4,221 (m10/m20/m40), text 2,246 (t4k 637 / t8k 610 / t16k 588 / t32k 411), needles text 300 / pages 225 / mixed 225 — job 191515, closed; healthy: COMPLETED 0:0 homer-019 26:57, one attempt; 1,500 rows, judge on all, unparseable 0. Uncompressed text 40 passages: judge .707 (10 passages .752) — job 191393, closed; healthy: COMPLETED 0:0 attempt 4 on homer-016 (01:04-01:45; 3 preemptions), file written 01:45:58 by that attempt; 1,500 rows, judge on all, unparseable 0, mems_used 1,075,610 = audit 1,075,610. R s42 text 40 passages: judge .534 (10 passages .605) — job 191392, closed; healthy (COMPLETED 0:0, homer-017, 50:42, one attempt; 1,500 rows, judge on all, unparseable 0, mems_used 1,614,940 = audit; re-read from the result file 2026-11-17 22:00). Docs 10 vs 4 pages s42: R .607 vs .610, delta_len +0.5 [-1.8,+2.7] sign 144/138 p .77; EM +1.6 — job 191378, closed
8. No shared file is edited here: `scripts/eval_mm_docqa.py`, `eval_mm_4b.sh`, `scripts/judge_*.py`,
   `scripts/paired_judge.py`, `scripts/probe_mem_recall.py`, `scripts/longdoc/**` are RUN, never edited — three live
   siblings pair on their md5s. A probe that seems to need a new flag gets a wrapper under `scripts/slotcheck/`.
9. The only training is arm O (100 dense pages, image autoencoding, ≤ 2 GPU-h per cell, output under
   `expQ/slotcheck/`). Any other training is a proposal to the owner, never an arm here.
10. NO route-then-expand, no raw page or raw tile in the decoder prompt, no picked-page trick — as an arm or as a
    proposal (owner, 2026-11-18, QUESTION.md:7-8). Raw-page scores may be quoted as a ceiling only. Every arm is about
    what the slots hold.
11. Never run python, pixi, a dataset load or a model load on the login node: every script goes through
    `expmon launch` (CPU jobs on `gpu-be --gres=gpu:0`). Stdlib python on small JSON (`paired_judge.py`) is fine.
12. Dead ends — do not retry, hold constant what won: compression rate 16 or 32 (docs −.065 / −.074 and −.172 / −.198,
    campaign_spread/CAMPAIGN.md:287); the `<MEM>` mixer, a linear or moving connector, a density-adaptive rate, a
    bidirectional compressor on Qwen3.5, PPE as an inference probe, per-head scales, the coarse-to-fine slot order
    archq3b, untrained slot dropping or blind halving (campaign_spread/CAMPAIGN.md:287-298); evidence-then-answer
    labels, gold-only as a fix, route-then-densify at rate 4, OCR words for photo text (−.434), passage order
    (campaign_probes/CAMPAIGN.md:39-45); a decoder-only QA finetune on a compressor that never saw QA (−2.8 / −3.5 pt,
    campaign_freeze, HANDOFF.md:67); long-chunk stage 1 at 1,024 / 2,048 tokens (the owner's stop of 2026-11-16, and
    its loss sits in the compressed image page, campaign_longdoc/STATE.md:42-50); query-dependent compression, more
    finetune data, ViT LoRA, KL at finetune, RAFT, a rate changed at inference only. Never judge with EM / F1 alone:
    the judge accuracy decides.
13. Never touch the live siblings `longdoc`, `qafreeze`, `spread` — their jobs, dirs or outputs (Budget and caps) —
    and never the expmon daemon job 188764 except through `expmon daemon`.

## Wake checklist (expmon grid tooling)
1. `expmon grid slotcheck` — where every cell stands (state, verdict, factors, RESULT).
2. `expmon verdict <jid> closed|best "<why>"` for decided cells/arms.
3. Fill missing cells: `expmon launch --exp slotcheck …` (or `expmon sweep --exp slotcheck --dry-run …`, then for real).
4. `expmon probe <jid> <metric>=<val>` when a cell's number is not picked up by the results script.
5. Rewrite STATE.md and `docs/slot_checks.md`; write NEXT.

## Escalation
BLOCKED.md (what, why, options, recommendation) and stop. The watchdog mails herve.dejean@naverlabs.com and writes ALERT for what
the loop cannot fix itself: stale ticks, wakes failing fast (login), no GPU-hour for CAMP_IDLE_H, a rejected
push, unpushed work, low disk, missing cron or daemon.

## Close-out (after CAMP_UNTIL)
`expmon campaign report` (one screen; its Settled / Open block is the successor's brief: `expmon campaign init <label> --from <this dir>`) → `expmon campaign retro` → fill its Lessons → `expmon gripe -c retro "RETRO slotcheck: WENT OK: … WENT NOT OK: … NET: …"`
(one line; the cross-cluster feedback digest scores campaigns from it) → the OWNER pastes this dir's SYNC.md (written by retro) into ~/.claude/SYNC.md — never the wake (hard rule 3) → `expmon campaign disarm`.
