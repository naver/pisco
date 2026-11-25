# CAMPAIGN rebuild — pisco   (the law; the agent re-reads this every wake; rewrite, never append)

created 2026-11-25 · kind **train** · root `/beegfs/scratch/user/hdejean/pisco` · window until **2027-02-23** (launches allowed through that day 23:59) · owner herve.dejean@naverlabs.com · wakes run `claude-opus-5-5`

## Mission
Successor of campaign 'slotcheck' (report `campaign_slotcheck/REPORT.md`, brief `campaign_slotcheck/SUCCESSOR.md`,
findings `docs/slot_checks.md`). **Question (owner, 2026-11-25 18:40 / 18:54 / ~19:25, hand-offs 4c0b26 + d30825):**
rebuild stage 2 so the image slots keep the page's words. Owner's words: "A — a new campaign, stage 2 with a larger
share of page transcription on document pages (one change) plus a second seed of the current stage-2 recipe as
control; gate 1 = page word recall on the 100 untrained dense pages and on eval pages, stop at ~63 GPU-h if it fails;
gate 2 = finetune + paired judge on documents, photos, text"; "build BOTH stage-2 runs on the spread <MEM> layout";
"yes" to three more arms: transcription from anywhere on the page, fp32 training, a finetune with the wrong labels removed.
**Why (slotcheck):** text slots of the page's transcription beside the image slots +5.0 pt [+2.5, +7.6] (1,000 paired Q);
~120 slots per page hold a dense page's words once trained on it (recall .921 vs .702, 100 pages). The limit is what
stage 2 teaches, not the slot count.
**The numbers that decide it:** gate 1 = page word recall, change − control, paired per page; gate 2 = judge accuracy
on documents, change − control, paired per question, photos and text as guards.
**Regime of every number:** page word recall in [0, 1] = share of the reference transcription's words found in the
model's rebuild from the slots (`scripts/longdoc/mem_recall.py --arm ae` vs `--arm ref`, paired by
`scripts/longdoc/recall_pair.py`; plain `<AE>` prompt, 768 new tokens). Judge accuracy in [0, 1], pt = .01, Qwen3-8B rubric
judge (`scripts/judge_mm_docqa.py`), greedy, answer cap 32 (the shared `eval_mm_4b.sh` default, kept so files pair with
the recorded ones), paired per `query_id` by `scripts/paired_judge.py` (bootstrap 95% CI + exact McNemar). Sets:
documents `eval_data/mm_docqa_1500_1m`, photos `eval_data/mm_naturalqa_1500`, text `eval_data/text_mrqa_ood_1500`
(`CML=4096`), PX=1003520. Compression rate everywhere: 8 tokens per slot, ≈ 108 slots per document page (646,042 / 1,500
questions / 4 pages). Uncompressed decoder on the same questions: recorded files in `expQ/base_qwen35_4b/eval/`.
Models: R2 = the recorded stage 2 `expQ/archq3/pt_mm_5M_spread/model` (seed 42, code e936195, 2026-11-04); R = its
finetunes `expQ/ft_mm_4b_184778/model` (s42) / `expQ/ft_mm_4b_184780/model` (s43).

## Arms — what varies, what is held constant
Stage-2 recipe = the recorded overrides of `expQ/archq3/pt_mm_5M_spread/.hydra/overrides.yaml`, exactly: start
`expQ/archq3/pt_text_5M_spread/model`; data `mm_pretrain_pdfa_1m_desc:80000, mm_pretrain_wiki_1m_desc:30000,
mm_pretrain_natural_desc:50000`, 160k rows (the dense sets exclude exactly these pdfa rows: build 191891); task mix
image AE .2 / text .5 / image continuation .05 / desc .15 / find .1; bs 8 × accum 8, lr 3e-5, bf16, KD .5, teacher
inputs (6,144), merger LoRA, 1,003,520 px, `ae_target_max_tokens` 768. **Every stage-2 arm runs seed 43**
(`++hf_training.seed=43`, as 184780 did) on today's committed code, so change − K is one variable on one code date.
| arm | launcher (committed) | the one change | held constant | baseline | gate | GPU-h |
|---|---|---|---|---|---|---|
| K. control | `scripts/rebuild/pt_stage2.sbatch K` | none (seed 43) | the recipe | R2 (seed floor + code drift) | — | 30 |
| S. transcription share | `scripts/rebuild/pt_stage2.sbatch S` | document rows (`source` = `pdfa` or `wiki-ss`): image AE .45 / text .25 / cont .05 / desc .15 / find .1; photo rows (`cauldron/*`) keep today's mix. Collator kwargs `mix_sources` + `mix_override` (`pisco/mm_collator.py`) | everything else | K | gate 1, then gate 2 | 30 + 14 |
| W. window targets | `scripts/rebuild/pt_stage2.sbatch W` | on document rows whose page text exceeds 768 tokens, half of the image-AE rows (`ae_window_p` .5) target a 768-token window starting at a uniform random token ≥ 1 (a full window fits) instead of the top; its first 16 tokens (`ae_window_cue`) are given in clear: `<AE> cue <MEM>*n <bos> window[16:] <eos>`, the cue masked. Rows ≤ 768 tokens, the other half, continuation and find unchanged | task mix, everything else | K | gate 1 (W) | 30 |
| P. fp32 masters | `scripts/rebuild/pt_stage2.sbatch P` | `++model_config_overrides.torch_dtype=float32 ++model_config_overrides.frozen_base_dtype=bfloat16`: trainable weights in fp32, frozen base and bf16 autocast unchanged; lr 3e-5 held (owner 2026-11-25). Round 1 found fp32 at the bf16 rate too high for finetuning (loss .662 vs .600 at step 10), so a tie or loss closes ONLY "fp32 masters at lr 3e-5", never fp32 in stage 2; only a pass is clean | everything else | K | gate 1 | 30 (smoke measures) |
| L. cleaned finetune labels | `scripts/rebuild/ft.sbatch L <42\|43>` over `data/rebuild_ft_clean_{docpages,natural}` (to build) | finetune rows whose teacher label the Qwen3-8B rubric judge scores wrong against the dataset gold are DROPPED, not replaced (owner 2026-11-25; refusals count as wrong): `scripts/mmrl_teacher_split.py dump` → `judge_mm_docqa.py` → `build --out_pd` (run, never edited), on BOTH `mm_ft_docpages_20k_1m_teacher` (first 20,000 rows) and `mm_ft_natural_8k_teacher` (first 8,000). The old split (`expQ/mmrl_split/`) judged the 451k-px labels, not these: never reused. Report rows and optimizer steps beside R's (28,000 rows, 437 steps): a tie cannot separate cleaner labels from fewer rows; a pass reads as a label effect | start R2, the 184778 (s42) / 184780 (s43) finetune recipe, 1 epoch | R s42 / s43 | gate 2 (L) | 28 |
Gate-2 finetunes of a stage-2 arm: `scripts/rebuild/ft.sbatch <ARM> 42` = the 184778 argv exactly (28k rows, bs 1 × accum
64, lr 2e-5, bf16, connector_out_norm init .461) from `expQ/rebuild/<ARM>/model`; K is finetuned the same way.
Start order: the feasibility gate ALONE (Decision gates); then K, S, W, P side by side after each passes its 20-step
smoke; L at once after the gate (CPU dump + judge first). Arms run in parallel, never one after another.

**Code path held equal to R's (question review 2026-11-25):** a photo finetune row holds one image
(`scripts/build_mm_natural_qa.py:82`) and the finetune runs bs 1 padded to 64, so R's finetunes trained the 8,000 photo
rows through the one-row pad leak (docs/slot_checks.md, check 2a, corrected). The pad guard of hand-off 2de131 is now
opt-in (`PISCOConfig.one_row_pad_guard`, default False) and no launcher here turns it on, so L − R and K-ft − R differ
only in what the arm changes. Turning it on is its own arm, parked (rule 9). Pre-arm proof (the pisco chat, 2026-11-25):
`scripts/rebuild/preflight_cpu.sbatch` = the unit tests (`TestRebuildOptions`: off = the old collator's RNG stream draw
for draw, against commit 69869340) + every arm's resolved config through the committed launchers (Hydra `--cfg job`, no
training) passing `config_check.py`. The first wake re-reads its RESULT line and `git status pisco/` (clean); else BLOCKED.md.

Rules: every arm is a committed launcher. `pt_stage2.sbatch` / `ft.sbatch` check the run's resolved
`training_config.yaml` against the recorded one as soon as it is written (`scripts/rebuild/config_check.py`, an
allow-list per arm: seed, that arm's change, `max_steps` in a smoke; finetunes: the start model, or L's data) and kill
the run on any other difference.
A comparison with a difference not listed here is not a result. A relaunch repeats the original command exactly.
- Outputs: `expQ/rebuild/<arm>/` (stage 2), `expQ/rebuild/ft_<arm>_s<seed>/` (finetunes), eval cells on symlinked
  model dirs `expQ/rebuild/<cell>/model`, recall under `expQ/rebuild/recall/`. Never write into a recorded dir.
- Every intent note records the md5 of `eval_mm_4b.sh`, `scripts/eval_mm_docqa.py`, `scripts/judge_mm_docqa.py`,
  `scripts/paired_judge.py`, `pisco/model.py`, `pisco/mm_collator.py`, `scripts/longdoc/mem_recall.py` and the git SHA.
  Two cells pair only on the same md5s; image answers move on homer-[005-009] (exclude them for evals).
- R's recorded eval files are the gate-2 baseline of arm L only if a re-run of R s42 on documents with today's md5s
  gives ≥ 99.5% identical answers (the 188419 check); otherwise R's six evals are re-run first (+6 GPU-h).

## Decision gates
House bars (campaign_longdoc/CAMPAIGN.md:107): a gain needs 3 pt with the 95% CI above 0; "no effect" needs the CI
inside ±3 pt. Noise band 1.2 pt: below it nothing is a result. Decide each arm as soon as its cells end.
- **Feasibility gate (first, ALONE, ≈ 1.5 GPU-h): can gate 1 be read on today's code?** `scripts/rebuild/recall.sbatch`
  (to write: slotcheck's `recall.sbatch` with outputs under `expQ/rebuild/recall/` and any model dir as ARM; same
  `mem_recall.py`, md5 recorded) on `eval_data/slotcheck_dense100`: R2 must give .702 ± .01 (job 191893), the slotcheck
  overfit model `expQ/slotcheck/o_bf16/model` .921 ± .01 (known positive, job 192209), the reference paired with itself
  1.000. FAIL → the campaign STOPS before any training: BLOCKED.md "gate 1 cannot be read on today's code" with the
  numbers, NEXT `waiting-owner`. A bug in the new launcher is not a finding: fix it and re-run.
- **Seed floor (first read of gate 1):** K − R2 per page on every gate-1 set. A change arm's gain must exceed |K − R2|
  on that set, or it is reported as inside the floor, never as a gain.
- **Gate 1 (S, P; n = 100 + 100 + 360 pages, paired per page, sign test + bootstrap CI):** plain-AE page word recall on
  `eval_data/slotcheck_dense100_heldout`, `eval_data/slotcheck_dense100` (both untrained, R2 .670 / .702, text-layer
  ceiling .983 / .974, job 191893) and the gold pages of `mm_docqa_1500_1m` (`mem_recall.py --per_source 120`, 360 pages;
  reference = one `--arm ref` pass shared by every model). PASS = change − K ≥ +3 pt with the CI above 0 and above the
  seed floor on held-out dense pages AND on eval pages → its gate-2 finetune. FAIL on both → `closed`, no finetune.
  One of two → reported, ONE owner question. Report beside it: recall by fifth of the page (the .87 → .75 top-to-bottom
  slope) and the share of document AE targets cut at 768 tokens.
- **Gate 1 for W (n = 100 long pages + the sets above):** guard = W's plain-AE recall on the three sets above is not
  below K by more than the seed floor. Its own read: on 100 untrained pdfa pages over 768 tokens (`eval_data/rebuild_long100`,
  to build from the untrained tail like build 191891), the cued readout of each page's LAST 768 tokens (the W prompt,
  same cue length), for W and K, each with the page's own slots and with another page's slots. Content gain = own −
  other recall. K is read twice: in the W prompt and in the image-continuation prompt (the same 16 tokens in clear
  before the slots, no `<AE>` — a format K trained on); K's content gain is the LARGER of the two, so a W pass cannot
  come from W merely knowing the cue format. PASS = W's content gain − K's ≥ +3 pt with the CI above 0, guard held.
  The other-page read is the known-negative control (no content → near the floor). Known-positive control: the same
  cued readout of the TOP window of each long page in the continuation format, on K: own − other must be ≥ +10 pt
  with the CI above 0, else the readout registers no content and W's gate decides nothing (fix the readout, re-run).
- **Finetune floor:** K-ft (s42) is ALWAYS trained and evaluated, whatever gate 1 says. Floor = |K-ft − R s42| per set
  (stage-2 seed + code drift at finetune level).
- **Gate 2 (S, and W / P if the owner releases them; n = 1,500 paired Q per set, finetune seed 42):** change-ft − K-ft.
  PASS = documents ≥ +3 pt with the CI above 0 AND above the finetune floor, and neither photos nor text loses ≥ 3 pt
  with the CI below 0. Report beside it: the uncompressed decoder on the same questions, the compression-rate column,
  per-source rows (infovqa / mpdocvqa / slidevqa).
- **Gate 2 for L (n = 1,500 paired Q per set, both seeds):** L s42 − R s42 and L s43 − R s43. PASS = documents ≥ +3 pt
  with the CI above 0 at both seeds AND the s42 gain above the finetune floor (so a shift from today's training code
  cannot pass as a label effect), guards as above. Report rows dropped per source (n kept / n judged). Prior, a
  different model: dropping teacher-wrong rows was null at 500 Q, one seed, old judge (`mmrl_ft_20k_pd` .510 vs .510,
  docs/data_review.md:47).
- **Promising, then ONE repeat (owner 12 Nov).** A gain with McNemar / sign p < .05, positive everywhere it was read,
  no guard losing beyond its floor, but under the 3-pt bar is PROMISING, never closed as a fail. No untouched document
  set exists, so its repeat is the seed-43 finetune pair (change-ft s43 vs K-ft s43, ≈ 24 GPU-h): ONE owner question
  with that cost, judged once at the same bar. Never a second repeat.
- **Health per arm** (a 0.0 that is a bug looks exactly like a negative result): smoke = s/step, peak memory, the loss
  falling, `<MEM>`s per page constant, the config diff as the Rules say; S and W: the collator's printed task counts per
  source match the mix within 1 pt over the first 1,000 rows; stage 2 ends at 2,499 steps (160k rows, 1 epoch) with a
  final loss within 10% of R2's .90; recall cells print `mem_position_stride=8` and non-empty text on a 5-page
  plumbing pass; eval cells 1,500 rows, a `judge` on every row, 0 unparseable, no "exceeds" error. L: rows judged =
  rows dumped, a planted wrong label (gold swapped with another row's) is scored wrong, a planted exact copy of gold is
  scored right. A cell that fails health is `closed "health"` and relaunched once.
- Small cells are plumbing (`LIMIT`, `--limit`, the 20-step smoke): they prove the path runs and decide nothing.
- A NEW metric (one the wake designs itself) closes nothing until its row here states its n (cells or pairs) and
  a known-positive control it recovers. Next to any threshold count, report the paired per-item test.
- Headline claims in STATE.md and the ledger are prefixed `provisional:` until the next critic pass, or until the
  next wake re-derives them from `expmon grid`.
- **On acceptance:** nothing is switched on by the wake; this campaign delivers trained models and findings. Default-on
  is refused because the judge is not proper: frozen for the wake by prose only; no tool plants a gain; no second
  program tested on planted cases; paired on every item yes, but every set has been reused by past campaigns; no
  automatic revert. So: a passing arm reaches the pisco chats as a hand-off (adopt the stage-2 recipe) and the owner
  as ONE question.

## Budget and caps
- GPU-hours: **190** (owner's figure ≈ 175 = 90 + 85; the +15 is the label judge, arm L's six evals and recall on five
  models; owner-confirmed 2026-11-25, "yes as recommended"). Built from measured walls: stage 2 ≈ 5,300 rows/h → 160k ≈ 30 h on one A100
  (campaign_spread/CAMPAIGN.md:155); a 28k finetune 11:14 h (183814); a judged 1,500-Q cell 28-98 min; recall 14 s/page
  (job 191169). Split: feasibility 1.5 · smokes 2 · K/S/W/P stage 2 120 · gate-1 recall 12 · gate 2 S (S-ft, K-ft, 6 evals) 28 · arm L
  (judge, 2 finetunes, 6 evals, R re-run check) 28. Cells: ≤ 60. Over either: BLOCKED.md.
- **Owner checkpoints:** (1) W or P passes gate 1 → their gate-2 finetunes (≈ 14 GPU-h each) are outside the 190:
  ONE question. (2) A PROMISING gate 2 → the seed-43 repeat, ONE question. (3) P's smoke projects stage 2 over 40 GPU-h → ONE
  question before its full run. (4) The end: the verdicts as ONE question.
- at most CAMP_MAX_TRAIN (6) trainings in flight; at most CAMP_MAX_SUBMIT_PER_WAKE (4) new submissions per wake.
- Trainings: 1 A100, `-p gpu` (the owner's shared 8-GPU cap; `--requeue`, resumable), after `gpucap land a100 1`;
  `--time` = the smoke's s/step × steps × 1.25. Inference and CPU cells: `-p gpu-be --exclude=homer-[005-009]`,
  `--time` = measured cell wall × 3, CPU jobs `--gres=gpu:0`. `--mem` from the smoke's MaxRSS × 1.5 (160k rows load into RAM).
- no launch below CAMP_DISK_FLOOR_GB free; nothing new after CAMP_UNTIL.
- **Live sibling `longdoc`** (armed; campaign_longdoc/CAMPAIGN.md): inference cells on `gpu-be`, at most 2 on `gpu`, no
  training; writes `scripts/longdoc/**`, `expQ/longdoc/**`, `docs/longdoc_diagnosis.md`, `eval_data/longdoc_*`; may
  patch `scripts/eval_mm_docqa.py` and `eval_mm_4b.sh` (record md5s with every cell). This campaign never touches its
  jobs, dirs or outputs. Campaigns `slotcheck`, `freeze`, `probes`, `qafreeze`, `spread`, `longchunk` are stood down or
  closed: read their files, write nothing there.

## Deliverable — how results reach the owner
`docs/stage2_rebuild.md` (to write), rewritten (never appended) whenever a gate decides: ONE table per gate, columns
naming metric and direction — arm | model | set | n (pages or pairs) | metric ↑ | K (control) ↑ | difference (95% CI,
sign / McNemar p) | seed floor | uncompressed judge accuracy ↑ on the same questions (gate 2) | compression rate (8
tokens per slot; slots per page) | health | verdict — and under each table ONE line on what it means for the stage-2
recipe. Every decided gate also gets one `expmon note` with its headline number. The default page is
`expmon display --exp rebuild`.

## Pre-authorised actions (the agent may do these without asking) and their loop guards
| situation | action | guard |
|---|---|---|
| a job FAILED / NODE_FAIL / TIMEOUT for a transient cause (preemption without resume, node failure, wall-time misestimate) | relaunch the same command, `--time` re-derived; a training resumes from its checkpoint | max 2 relaunches per cell, then BLOCKED.md |
| `expmon check` prints `node <n>: N fast deaths` | relaunch the dead cells with `--exclude=<n>` | 1 exclude-relaunch per cell |
| collapse: NaN loss, or the loss at its first value for 2 probes | `expmon kill`, `expmon verdict <jid> closed "<why>"` | never relaunch the same cell |
| `starved util=NN%` on a training older than 1 h | `expmon occupancy`; record it; kill only if the GPU is idle (dataloader stall) | 1 kill-relaunch per arm |
| a cell of a listed arm is missing | `expmon launch --exp rebuild …` | within CAMP_MAX_SUBMIT_PER_WAKE |
| a job PENDING > 6 h with an oversized `--time` or `--mem` | `scontrol update JobId=… TimeLimit=…` / `MinMemoryNode=…` (downward only) | never touch a job the campaign does not own |
| an inference cell PENDING > 6 h on `gpu-be` for Priority | `scontrol update JobId=… Partition=gpu,gpu-be` after `gpucap land a100 1` | trainings + this ≤ 8 GPUs on `gpu` |
| the expmon daemon is gone | `expmon daemon` | — |
| a straggler FAILED in seconds with a benign signature (known_issues.md) | ignore, note once | — |
| a gate needs a launcher, builder, set or readout script | write it under `scripts/rebuild/` (freely patchable); smoke with a `LIMIT` / `--limit` run, then the full cell | never a shared file (hard rule 8) |
| a reproduced crash whose fix is ≤ CAMP_PATCH_LINES lines inside a "## Patchable" path | apply it, smoke on the cheapest cell, record `<file> — <traceback line>` in STATE.md | 1 patch per symptom, then BLOCKED.md |
Anything else: BLOCKED.md and stop.

## Patchable
CAMP_PATCH_LINES in campaign.conf is what is enforced and it wins over this section.
- **freely patchable:** `scripts/rebuild/**`, `expQ/rebuild/**`, `campaign_rebuild/**` (never campaign.conf),
  `docs/stage2_rebuild.md`, new sets `eval_data/rebuild_*` and `data/rebuild_*`.
- **patchable IF an invariant holds** (its output pasted in STATE.md that wake): `scripts/rebuild/pt_stage2.sbatch` and
  `ft.sbatch` — header, paths, env plumbing, logging. Invariant: `bash -n` passes and the printed config diff against the
  recorded `training_config.yaml` (stage 2: `expQ/archq3/pt_mm_5M_spread`; finetune: `expQ/ft_mm_4b_184778`) lists only
  seed, the arm's change, data path (L only) and output paths. A launcher a running or finished cell used is frozen for
  THAT cell.
- **never:** `pisco/**`, `test_*.py`, `pt_mm_4b.sh`, `ft_mm_4b.sh`, `eval_mm_4b.sh`, `scripts/eval_mm_docqa.py`,
  `scripts/judge_*.py`, `scripts/judge_prompts.py`, `scripts/paired_judge.py`, `scripts/probe_mem_recall.py`,
  `scripts/mmrl_teacher_split.py`, `scripts/longdoc/**`, `scripts/slotcheck/**` (run them, never edit), existing
  `eval_data/` sets, `data/**` other than `data/rebuild_*`, recorded models and result files (`expQ/**` other than
  `expQ/rebuild/`), everything of the live sibling, and `EXPERIMENTS.md` except through `expmon note`. A needed change
  in a "never" path is a hand-off to the pisco chats (`expmon handoff add --to pisco`).

## The wake owns the science
The arms are the owner's and fixed before launch (window rule, share, label rule): never tuned on gate 1. The wake
launches the cells, reads the gates, closes decided arms, writes readouts and sets the gates name, and amends this law
when evidence refutes it — a wrong prediction is kept and marked, never quietly deleted.
The prediction written before any cell, kept and marked if wrong: S raises held-out dense-page recall by ≥ 3 pt (the
overfit reached .921) and gains 1-3 pt on documents at gate 2; W leaves plain-AE recall inside the floor and raises
bottom-window content recall; P ties K inside the floor (fp32 tied bf16 at finetune, round 1); L ties R within ±2 pt
at both seeds (prior null).

## Hard rules
1. Submissions only via `expmon launch --exp rebuild` / `expmon sweep --exp rebuild`; never raw sbatch (enforced).
2. Cancel only campaign-owned jobs, only via `expmon kill`. Never `rm` a run dir — rename it `VOID_<name>` (enforced).
3. Never edit campaign.conf, crontab, hooks, `~/.claude`, the tick or the watchdog; never relax a rule here.
   Code edits only as "## Patchable" allows. A `REQUEST` file in this dir is the owner speaking (`expmon campaign
   resolve`): execute it before anything else, then delete it; nothing else may create one. Rules 1–3 are enforced by
   the wake hook: a refusal is a signal to write BLOCKED.md, never to find another spelling. The tick commits for you.
4. Hypothesis ≠ conclusion: write "on track to", "consistent with"; a conclusion needs its gate AND the repeat.
5. Report rows seen / epochs and n (pages, pairs) with every number, never bare step counts.
6. End every wake with one line in NEXT (`done` | `wake-when job:<ids> [max <N>m]` | `wake-in <min>` | `waiting-owner`), or nothing while work is left.
7. Settled by campaign 'slotcheck' (hold constant, do not retry): the finetune 4-document cap drops no answer page (0 of
   20,000 rows, job 191663); training-style padding costs nothing (+0.5 / +0.9 pt, 1,500 Q × 2 seeds); a one-row padded
   compressor batch leaks pads (latent, no training path takes it; hand-off 2de131); text slots of the transcription beside
   image slots +5.0 pt [+2.5, +7.6] (1,000 Q, jobs 192223 / 192239); slot capacity is not the limit (overfit recall .921 vs
   .702, job 192209).
8. No shared file is edited here (the "never" list): they are RUN; the live sibling pairs on their md5s.
9. The only trainings are arms K, S, W, P (stage 2) and their gate-2 finetunes, and arm L's two finetunes, under
   `expQ/rebuild/`. Any other training, or a second change in an arm, is a proposal to the owner, never an arm here.
10. NO route-then-expand, no raw page or raw tile in the decoder prompt, no picked-page trick — as an arm or as a
    proposal (owner, 2026-11-18). Raw-page scores may be quoted as a ceiling only.
11. Never run python, pixi, a dataset load or a model load on the login node: every script goes through
    `expmon launch` (CPU jobs on `gpu-be --gres=gpu:0`). Stdlib python on small JSON (`paired_judge.py`) is fine.
12. Dead ends — do not retry: compression rate 16 or 32; the `<MEM>` mixer, a linear or moving connector, a
    density-adaptive rate, a bidirectional compressor on Qwen3.5, PPE, per-head scales, archq3b slot order, untrained
    slot dropping (campaign_spread/CAMPAIGN.md:287-298); evidence-then-answer labels, gold-only pages as a fix,
    route-then-densify at rate 4, OCR words for photo text, passage order (campaign_probes/CAMPAIGN.md:39-45); a
    decoder-only QA finetune on a compressor that never saw QA (campaign_freeze); long-chunk stage 1 at 1,024 / 2,048
    tokens; query-dependent compression, more finetune data, ViT LoRA, KL at finetune, RAFT, a rate changed at
    inference only. Never judge with EM / F1 alone: the judge accuracy decides.
13. Never touch the live sibling `longdoc` (its jobs, dirs, outputs) and never the expmon daemon job 188764 except
    through `expmon daemon`.

## Wake checklist (expmon grid tooling)
1. `expmon grid rebuild` — where every cell stands (state, verdict, factors, RESULT / latest probe).
2. `expmon verdict <jid> closed|best "<why>"` for decided arms.
3. Fill missing cells: `expmon launch --exp rebuild --stage <S> --parent <jid> --dependency afterok:<jid> …`.
4. `expmon probe <jid> --step <rows seen> <metric>=<val>` for in-flight trainings.
5. Rewrite STATE.md and `docs/stage2_rebuild.md`; write NEXT.

## Escalation
BLOCKED.md (what, why, options, recommendation) and stop. The watchdog mails herve.dejean@naverlabs.com and writes ALERT for what
the loop cannot fix itself: stale ticks, wakes failing fast (login), no GPU-hour for CAMP_IDLE_H, a rejected
push, unpushed work, low disk, missing cron or daemon.

## Close-out (after CAMP_UNTIL)
`expmon campaign report` (one screen; its Settled / Open block is the successor's brief: `expmon campaign init <label> --from <this dir>`) → `expmon campaign retro` → fill its Lessons → `expmon gripe -c retro "RETRO rebuild: WENT OK: … WENT NOT OK: … NET: …"`
(one line; the cross-cluster feedback digest scores campaigns from it) → the OWNER pastes this dir's SYNC.md (written by retro) into ~/.claude/SYNC.md — never the wake (hard rule 3) → `expmon campaign disarm`.
