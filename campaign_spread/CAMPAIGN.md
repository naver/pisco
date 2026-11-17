# CAMPAIGN spread — pisco   (the law; the agent re-reads this every wake; rewrite, never append)

created 2026-11-17 · kind **train** · root `/beegfs/scratch/user/hdejean/pisco` · window until **2027-02-15** (launches allowed through that day 23:59) · owner herve.dejean@naverlabs.com · wakes run `claude-opus-5`

## Mission
**Question (owner, QUESTION.md):** why was the spread `<MEM>` layout not validated — in which cases is it better, equal or
worse than the tail layout, does the evaluation itself hide or inflate the difference, and can the gain be brought back?
Probes (inference only) first; a training arm only after a probe supports it, one variable per arm.
**The deciding number:** paired judge accuracy (Qwen3-8B rubric judge `scripts/judge_mm_docqa.py`; `scripts/paired_judge.py`
= paired bootstrap 95% CI + exact McNemar), **spread − tail on the SAME questions**, per case; and, for the vanished gain,
the one-variable full-budget difference of arm E. Unit [0, 1], pt = .01, greedy decoding, 1,500 Q per gating set
(500 per source), seeds 42 and 43. The answer is a CASE TABLE (better / equal / worse, with the evaluation's resolving
power next to each case) plus a ranked list of tested hypotheses.
**Facts to explain, each verified against the file named:**
- Reduced budget, one variable (1M-text stage 1 → one 100k stage-2 pass → 28k finetune): documents **+.093**
  [+.069, +.115] (s42, 183814 − 183538) / **+.075** [+.051, +.099] (s43), photos +.023 / +.021
  (docs/architecture_review.md:243-248).
- Full budget vs the PRODUCTION tail models: documents +.011 [−.014, +.035] / +.024 [−.002, +.048], photos −.009 / +.010
  (EXPERIMENTS.md:518). **Not one variable:** production ran three stage-2 passes (170810 + 172617 + v4kd,
  EXPERIMENTS.md:499). "Stage-2 data covers what the layout gave" is a hypothesis nobody tested.
- The missing one-variable control EXISTS as a buildable chain: `expQ/qwen35_4b_pt_5M_143228/training_config.yaml` and
  `expQ/archq3/pt_text_5M_spread/training_config.yaml` are identical except `compressor_mem_layout: spread` +
  `mem_position_stride: 8` (lines 46-47), and both model dirs hold the same four checkpoint files. Arm E.
- Where spread is known to win and tail to collapse: one 1,270-token chunk .617 vs ten chunks .579 (spread) against
  .399 vs .568 (tail) (docs/multimodal_results.md:103-107); MRQA one long passage spread .694 = split .695, tail .639 vs
  .689, production .631 vs .697 (EXPERIMENTS.md:521); image MCQ +.026 (multimodal_results.md:120-122); trained rate 16
  under spread = tail at rate 8 at half the tokens (EXPERIMENTS.md:508).
- Where spread costs: dropping half the slots blind costs spread −.121 on documents, tail −.043 (multimodal_results.md:92-94).
- Why the gating eval may show none of it: the 4-page document set sits exactly at the finetune's maximum (455 / 485
  slots vs 488; docs/longdoc_diagnosis.md:102-103), and the document loss is dominated by reading the compressed answer
  page — a ceiling both layouts share (picked page raw .820 / .822 vs plain .610 / .615,
  docs/probe_campaign_2026-11.md:38-40).
**Regime.** Gating sets, 1,500 Q: documents `eval_data/mm_docqa_1500_1m` (PX=1003520, 4 pages), photos
`eval_data/mm_naturalqa_1500`, text `eval_data/text_hotpotqa_1500` (TOPK=10, CML=4096) and `_g10` (the same ten passages
as ONE chunk). Reported, not gating: `eval_data/mm_mcq_1000`, `text_mrqa_ood_1500`, 10- and 20-page document cells,
`mm_docqa_longdoc_1m` (500 Q). Models, fixed (only arm E trains): spread reduced `expQ/ft_mm_4b_183814|183816/model`,
tail reduced `183538|183539`; spread full `184778|184780`; tail production `174908|176242`; stage-2 models
`expQ/archq3/pt_mm_spread`, `pt_mm_base_from1M`, `pt_mm_5M_spread`, `expQ/pt_mm_v4kd_4b`; uncompressed =
`eval_mm_4b.sh … base` on the same rows (recorded files in `expQ/base_qwen35_4b/eval/`).
Owner's standing rules: probe before training; one variable per arm; text, images and mixed documents for every claim;
paired judge on 1,500 Q with 2 seeds for a training claim under ~.03; every table names its metric and direction, the
compression rate (tokens per slot, slots per page) and the uncompressed decoder on the same questions.
If it replicates or extends a paper, `papers/<slug>.md` (`expmon paper`, skill `paper`) is the source of every number
quoted here — cite the card, not the PDF.

## Arms — what varies, what is held constant
| arm | recipe file (committed) | factors (what varies) | held constant | gate (metric ≥ X vs whom) | budget (GPU-h) |
|---|---|---|---|---|---|
| A. case audit + slice power — **the feasibility gate** | `scripts/spread/case_audit.sbatch` → `scripts/spread/slice_pairs.py` (to write; CPU, `-p gpu-be --gres=gpu:0`; reads recorded judged jsons only, slices by source, `n_images`, `mems_used`, question type, answer length) | nothing — recorded files | `scripts/paired_judge.py` (never edited); the recorded files | feasibility gate below | 0 |
| J. instrument | `scripts/spread/rejudge.sbatch` (to write) = `scripts/judge_mm_docqa.py --prompt project` on COPIES under `expQ/spread/`; EM / ANLS / match read from the same records | the metric, the judge prompt | the predictions | every case that decides an arm keeps its sign and stays within 2 pt under the rubric judge, the project prompt and EM; otherwise that case is "judge-sensitive" — an evaluation finding, not a layout finding | 2 |
| B. long chunks at full budget | `eval_mm_4b.sh expQ/spread/<cell>/model pisco "" <set>` with `TOPK=10 CML=4096` on `text_hotpotqa_1500` and `text_hotpotqa_1500_g10` | layout (184778 / 184780 vs 174908 / 176242) × grouping (ten chunks / one chunk) | questions, rate 8, judge, CML, TOPK | spread − tail ≥ +3 pt with the CI above 0 at one chunk **and** inside ±2 pt at ten chunks, at both seeds → spread wins where chunks are long and the gating eval never asks | 4 |
| C. many pages at full budget | the cells the sibling `longdoc` runs for the spread models, on the TAIL models: `XD=6 TOPK=10` (10 pages) and `XD=16 TOPK=20` (20 pages), `PX=1003520`, `DML` from longdoc's audit (docs/longdoc_diagnosis.md:103-105) | layout × input size (4 / 10 / 20 pages) | questions, the distractor draw, PX, judge | (spread − tail) at 20 pages minus (spread − tail) at 4 pages ≥ +3 pt with the CI above 0 at both seeds → the layout pays at length | 8 |
| D. what the slots hold (mechanism, no decoder question) | `expQ/archq2/probe_mixer.sh` (text ownership) and `probe_connector.sh` (paired answer NLL on held-out rows) — both exist: run, never edit; `scripts/longdoc/mem_recall.py` (the stride-aware transcription port, validated 360 / 360 vs the legacy probe, campaign_longdoc/STATE.md:12-13) on 500 gold pages per finetune — read-only, md5 recorded | layout × budget: the four stage-2 models, the four finetunes | pages, rows, rate 8, PX, seeds | the reduced-budget spread advantage (text ownership .977 vs .799; answer loss pages 1.138 vs 1.334, text 1.812 vs 1.927, photos 1.765 vs 1.802 — architecture_review.md:273-277) shrinks inside ±2 pt between `pt_mm_5M_spread` and `pt_mm_v4kd_4b` → "the extra stage-2 data buys what the layout bought"; it holds → the tie is not a representation tie | 8 |
| E. the missing control (**TRAINING**; only after the owner checkpoint, only on a REQUEST) | `scripts/spread/pt_mm_tail5m.sbatch` + `scripts/spread/ft_tail5m.sbatch` (to write: the recorded commands and env of 184777 and 184778 / 184780, `expmon lineage <id>`, with ONE change — `model_name_or_path = expQ/qwen35_4b_pt_5M_143228/model`) | the `<MEM>` layout at full budget with ONE stage-2 pass | stage-2 recipe = `expQ/archq3/pt_mm_5M_spread/training_config.yaml` (160k, bs 8 × accum 8, lr 3e-5, KD .5, merger, 1,003,520 px); finetune = 184778 / 184780's recipe (28k, bs 1 × accum 64, lr 2e-5, bf16), seeds 42 / 43. Listed difference that cannot be removed: the two stage-1 runs are months apart (code era), same config otherwise | spread − tail(one pass) on documents ≥ +3 pt with the CI above 0 at both seeds → the layout gain DOES carry at full budget and production's tie came from its two extra stage-2 passes. CI inside ±2 pt at both seeds → the ledger's reading is confirmed and the layout buys 2/3 of the stage-2 data, not accuracy | 60 |
| F. fixes (**TRAINING**, at most 2; proposed by the wake at the owner checkpoint, each launched only on its own REQUEST) | `scripts/spread/ft_fix<k>.sbatch` / `pt_fix<k>.sbatch` (to write: a recorded recipe — the reference 184778 / 184780 chain or arm E's — with ONE change) | ONE config or recipe field (e.g. stage-2 data mix or size, stride, lr), chosen from what arms A-D found; never new code in `pisco/` | everything else of the recipe it copies; questions, rate 8, judge | written in its BLOCKED.md question BEFORE launch: the case it targets, fix − its recipe ≥ +3 pt with the CI above 0 at both seeds on that case, and no gating set (documents, photos, MRQA) ≤ −2 pt with the CI below 0 | owner, per REQUEST |
Start order (a hypothesis, not a work order): A → J controls → B → D → C → **owner checkpoint** → E (and F, if the owner approves fixes).

Rules: every arm is a committed recipe file, never raw overrides on the launch line. Before launching two arms
that will be compared, diff their resolved configs and list EVERY difference in this table — a comparison with
an unlisted difference is not a result (splare lost both headline claims this way). A relaunch repeats the
original command exactly (a flag left at its default is not a factor, but a changed default is a confound).
- Every eval cell runs on a symlinked model dir `expQ/spread/<cell>/model → <model dir>`, so its result lands in
  `expQ/spread/<cell>/eval/` and never overwrites a recorded file. A `base` cell writes into
  `expQ/base_qwen35_4b/eval/` only under a name that does not exist yet (`ls`) and that `campaign_longdoc/STATE.md`
  does not list as planned. New launchers are `scripts/spread/*.sbatch` (the repo `.gitignore` drops new `*.sh`,
  campaign_probes/CAMPAIGN.md:56-57).
- Record in every intent note the md5 of `eval_mm_4b.sh`, `scripts/eval_mm_docqa.py`, `scripts/judge_mm_docqa.py`,
  `scripts/paired_judge.py`, `pisco/model.py` (2026-11-17: 50abc167 / dfe32aa0 / 02bc8748 / a3cc13b2 / 994cff3d —
  campaign_longdoc/STATE.md:103, campaign_freeze/STATE.md:29; `eval_mm_4b.sh` is b3d1cda1 since 17:11 = requeue markers
  + an off-by-default DENSE env, same python call). Two cells pair only when they ran on the same code.
  A pair across code eras or node images is REPORTED, never decided, until one side is re-run on today's code: the code
  date does not move text answers (188518 = 185601, 1,500 / 1,500) but homer-008 moves image answers
  (campaign_probes/CAMPAIGN.md:70-89); launch with `--exclude=homer-[005-009]`.
- Before using a probe env on a model, mode or set it was never used with, read `main()` of `scripts/eval_mm_docqa.py`
  and confirm the flag reaches that branch. This campaign adds NO eval flag (two live siblings depend on those md5s).

## Decision gates
Every difference is PAIRED per question, same seed. Noise band CAMP_NOISE_PT = 1.2 pt: below it nothing is a result;
a tie triggers a seed replicate; above it once still needs a replicate to be "adopted". Case verdicts (the same bars as
campaign_longdoc/CAMPAIGN.md:83): **better** = ≥ +3 pt with the 95% CI above 0 at both seeds; **equal** = CI inside
±2 pt at both seeds; **worse** = ≤ −3 pt with the CI below 0 at both seeds. Decide arms as cells complete.
- A slice decides nothing with fewer than 300 questions or fewer than 40 discordant pairs (McNemar b + c); it is
  reported with its discordant count and called "not resolvable here", never "equal".
- Measured by arm A (191454, 23 recorded pairs): one seed's 95% CI on 1,500 Q is .044-.050 wide on documents and text,
  .036-.041 on photos, so "equal" (CI inside ±2 pt) cannot be reached on documents or text. Until the owner answers
  BLOCKED.md, a case that would need "equal" is reported "not resolvable here (CI width w)", never "equal".
- Photo re-runs drift across code dates even on el9_8 nodes: 184778 photos 184783 (homer-011, 11-09) vs 188602
  (homer-016, 11-11) differ on 233/1,500 answers, 85 judge flips, .677 vs .673; 184780's homer-008 photo file equals its
  re-run 1,500/1,500; documents re-ran identical. A photo pair across code dates is reported, never decided.
- **Feasibility gate (first, ALONE, 0 GPU-h): arm A — can the layout difference be measured from what exists?**
  (1) Re-derivation: from the recorded judged files, documents spread − tail = +.093 (s42) / +.075 (s43) and photos
  +.023 / +.021 at reduced budget, and +.011 / +.024 at full budget, each within ±.003 (tolerance: the recorded
  values are rounded to 3 decimals).
  (2) No-op control: any file paired with itself gives 0.000.
  (3) Inventory per planned case: n, discordant pairs, the node and code era of every file, and which pairs the
  same-node rule blocks. Health: the recorded 4-page plain cell of 184778 has 1,500 records, a `judge` on every row,
  0 unparseable and `mems_used` 646,042 (docs/probe_campaign_2026-11.md:7).
  FAIL on (1) or (2) → the campaign STOPS with "the recorded layout comparison cannot be re-derived, so no slicing of it
  is evidence" as the finding: BLOCKED.md with the files and counts, NEXT `waiting-owner`. (3) failing on most planned
  slices is a FINDING about the evaluation (it hides the difference) and the campaign re-plans with fresh cells — not a
  stop. A bug in the audit is not a finding: fix it and re-run.
- **Health per cell** (a score without its health number is not reported): 1,500 rows, a `judge` verdict on every row,
  `judge_unparseable` 0, no "exceeds --decoder_max_length" error, `mems_used` within ±1% of its twin cell at the same
  input (a cell that changes what is compressed reports the ratio instead, by design), node outside homer-[005-009]
  recorded with the file. Arm D: the controls must pass (another document's / page's slots far worse; a model paired
  with itself 0; the transcription port within ±2 pt of the legacy 70.5%). Arm E: `eval_loss` every 100 steps
  (`expmon probe`) read against 184778's curve at the same epoch, plus s/step and peak GPU memory. A cell that fails
  health is `closed "health"` and relaunched once.
- Small cells are plumbing: a `LIMIT=300` cell proves the path runs. It never decides an arm, even when it agrees.
- A NEW metric (one the wake designs itself) closes nothing until its row here states its n (cells or pairs) and
  a known-positive control it recovers — here: the reduced-budget +.093 (s42) / +.075 (s43) on documents out of the same
  code path, and 0.000 on a self-pair. Next to any threshold count, report the paired per-item test —
  navmem's count showed nothing where a paired sign test found the one real effect (98 closer / 59 farther, p=0.002; control p=0.22).
- A paired-NLL probe on tens of rows bounds an effect at ~.02-.03 judge accuracy, not at zero (EXPERIMENTS.md:475):
  arm D's NLL numbers are mechanism evidence, never an accuracy claim.
- Headline claims in STATE.md and the ledger are prefixed `provisional:` until the next critic pass, or until the
  next wake re-derives them from `expmon grid`.
- **Diagnose on the whole set before choosing a fix (owner 12 Nov).** A fix target is a CAUSE shared by many items,
  not one failing item and not one check. Before a fix arm opens: (1) the set is sound: its size is stated against
  the effect it must show, every setting varies, each check both passes and fails, a fresh set is held back; (2) the
  failing items are grouped by shared condition across checks, and causes are ranked by items repaired, each lead
  confirmed on the fresh set; (3) the fix is judged on the large set, paired, with the compared count stated: nothing
  else worse, no item lost. A check that fails on (almost) every item, or on the accepted references, is a fault of
  the check: source its limit before any fix.
- **Promising, then ONE repeat (owner 12 Nov).** An arm whose paired gain clears the usual bar (sign test p < 0.05,
  positive in every seed, no guard subset losing beyond its floor) but not the noise band is PROMISING, never closed
  as a fail. It gets exactly one repeat on NEW evaluation data (another split or held-out set, never more seeds on the
  same data), with its own baseline evaluated on that data, size fixed before launch, judged once at the usual bar. It
  passes if the repeat clears that bar; otherwise it is closed with both reads. Never extend a read until it passes,
  never a second repeat. The new items are documents `eval_data/mm_docqa_retr10_1m` (or `mm_docqa_longdoc_1m`,
  500 Q), text `eval_data/text_mrqa_ood_1500`, images `eval_data/mm_mcq_1000`.
- **On acceptance:** nothing is switched on by the wake. Default-on is refused because four of the five parts of a
  proper judge are missing here: (1) judge frozen for the wake — prose only (`scripts/judge_*.py`,
  `scripts/judge_prompts.py`, `scripts/paired_judge.py` are "never" below, but no hook or `CAMP_READ_DENY` enforces it);
  (2) controls — the no-op is refused (self-pair 0.000) but no tool plants a gain: MISSING; (3) a second program tested
  on planted cases: MISSING (`scripts/judge_agreement.py` measures agreement; the human sheet is unlabelled,
  HANDOFF.md:59); (4) paired on every item: yes (`scripts/paired_judge.py`), but every campaign has reused these sets:
  no untouched set; (5) automatic revert on a worse yardstick: MISSING. So: **opt-in, then one question to the owner.**
  A serving case this campaign confirms (for example "compress long text as one chunk") stays an existing
  `eval_mm_4b.sh` env, off by default, and reaches the owner as ONE question in BLOCKED.md plus a hand-off; a winning
  arm E becomes a proposed default for the next chain (replacement text for `docs/architecture_review.md` and
  `HANDOFF.md` written in STATE.md, applied by the owner's chats), never applied by the wake.

## Budget and caps
- GPU-hours: **30 before the owner checkpoint, 95 for the whole campaign** (built from measured walls: a judged
  1,500-Q document cell 28-98 min, photos 18 min, text 17 min (campaign_probes/CAMPAIGN.md:116-120), a 10-page document
  cell 50:42 (campaign_longdoc/STATE.md:67); arm E = a 160k stage-2 pass ≈ 30 h scaled from 19 h at 100k
  (EXPERIMENTS.md:507) + finetunes 11.0 h and 13.8 h (campaign_qafreeze/CAMPAIGN.md:22) + ~6 judged evals + smokes).
  Over it: BLOCKED.md. CPU jobs count 0 GPU-h.
- cells ≈ 50 (A 2, J 4, B 10, C 8, D 14, E 12; F counted at the checkpoint); LLM wakes ≤ 40. A cell = every `expmon launch --exp spread`
  submission, builds and CPU jobs included; an automatic requeue is not a new cell. Chain plumbing → full cell → paired
  test in one wake; wait with `wake-when job:<ids>` or `wake-when file:<path>`, never a timer.
- conf set by the owner (2026-11-17; the conf always wins over this law): `CAMP_PATCH_LINES=40`,
  `CAMP_COMMIT_PATHS='scripts/spread docs/spread_cases.md'` (`expQ/` is gitignored: run outputs are never committed),
  `CAMP_MAX_SUBMIT_PER_WAKE=4`. The wake writes the scripts marked "to write" itself, under "## Patchable".
- **Arm F GPU-hours are not in the 95:** each fix the owner approves at the checkpoint gets the GPU-hours written in
  its BLOCKED.md question, released by that REQUEST only.
- at most CAMP_MAX_TRAIN trainings in flight ("in flight" = submitted and not ended, PENDING included; a smoke counts);
  at most CAMP_MAX_SUBMIT_PER_WAKE new submissions per wake (campaign.conf).
- Evals and CPU cells: 1 GPU, `-p gpu-be --exclude=homer-[005-009]`. Arm E: 1 GPU on `gpu`, ONE training at a time;
  run `gpucap land a100 1` first and never take the last free slot (the owner's 8-GPU cap on `gpu` is shared with the
  live sibling `qafreeze`). At most 2 cells of this campaign on `gpu` at once.
- every launch: `--time` = measured cell wall × 3 for inference cells (from the slowest measured run of that cell kind),
  and measured s/step × steps × 1.25 for a training; `--mem` / `--cpus` from the recipe or the profile rung's MaxRSS × 1.5
- no launch below CAMP_DISK_FLOOR_GB free; nothing new after CAMP_UNTIL. Arm E writes one stage-2 tree + two finetunes.
- **Owner checkpoint:** none before arms A, J, B, C and D (all inference or CPU on existing models, nothing
  irreversible, ≤ 30 GPU-h); ONE before arm E, the training control, which is the larger part of the budget and uses the
  shared `gpu` cap: the wake writes the case table, the config diff of the two stage-1 runs and the cost into ONE
  BLOCKED.md question, writes NEXT `waiting-owner`, and continues only on a REQUEST (`CAMP_AUTO_ACCEPT_GPU_H=5` cannot
  release it). The same question proposes up to 2 fixes (arm F), each with the hypothesis it tests, the evidence from
  arms A-D, its one change, its gate and its GPU-hours; the owner approves each fix separately.
- **Live sibling `longdoc`** (campaign_longdoc/CAMPAIGN.md, campaign.conf) and what it uses:
  - GPUs / partitions: 1 GPU per inference cell on `gpu-be --exclude=homer-[005-009]`, at most 2 cells on `gpu`; budget
    70 cells / 60 GPU-h (35 cells, 14.6 GPU-h used at 2026-11-17 16:00); no training of any kind (its hard rule 8).
  - Machines: the shared homer A100 nodes — homer-011 and homer-017 at this writing; none reserved.
  - Engine trees / worktrees: none of its own; it shares THIS checkout (branch `arc`) and its pixi env.
  - Dirs it writes: `scripts/longdoc/**`, `expQ/longdoc/**`, `docs/longdoc_diagnosis.md`, new sets `eval_data/longdoc_*`,
    and new file names under `expQ/base_qwen35_4b/eval/`. Shared files it MAY patch (invariant-gated):
    `scripts/eval_mm_docqa.py`, `eval_mm_4b.sh`. Jobs in flight: 191376, 191377, 191390, 191391, 191392, 191393.
  - Wake quota: the same account (a 5-hour usage limit stops both); CAMP_MAX_SUBMIT_PER_WAKE 4, CAMP_MAX_USD 8,
    heartbeat 24 h.
  This campaign therefore: writes nothing under `scripts/longdoc/**`, `expQ/longdoc/**`, `campaign_longdoc/**`,
  `docs/longdoc_diagnosis.md` or `eval_data/longdoc_*`; never kills, requeues, relaunches or verdicts a longdoc job;
  reuses its recorded spread cells and base files READ-ONLY (arm C waits for them with `wake-when file:<path>` instead
  of re-running them) and runs `scripts/longdoc/*.py` as tools without editing them, md5 recorded; checks
  `campaign_longdoc/STATE.md` before writing any new name into `expQ/base_qwen35_4b/eval/`; never patches the shared
  eval or judge scripts (longdoc may — a patch landing mid-campaign invalidates a pair, so record the md5s with EVERY cell).
- **Live sibling `qafreeze`** (campaign_qafreeze/CAMPAIGN.md, campaign.conf; scaffolded 2026-11-17, no wake yet):
  - GPUs / partitions: 2 finetunes of ≈ 10 GPU-h, 1 GPU each on `gpu`, plus a 20-step smoke; 8 judged evals and CPU
    checks on `gpu-be --exclude=homer-[005-009]`; budget 55 GPU-h (80 if its arm H is released).
  - Machines: the shared homer A100 nodes; none reserved. Engine trees / worktrees: none — THIS checkout and its pixi env.
  - Dirs it writes: `scripts/qafreeze/**`, `expQ/qafreeze/**`, `campaign_qafreeze/**`, and re-runs of the reference cells
    of 184778 / 184780. Data: the shared 28k finetune rows and the four gating eval sets.
  - Wake quota: the same account; CAMP_MAX_SUBMIT_PER_WAKE 1, CAMP_PATCH_LINES 0, CAMP_MAX_USD 8, heartbeat 24 h.
  This campaign therefore: writes nothing under `scripts/qafreeze/**`, `expQ/qafreeze/**` or `campaign_qafreeze/**`;
  never touches its jobs; keeps arm E to ONE GPU on `gpu` at a time behind `gpucap land a100 1`; never edits
  `expQ/freeze/**` or the reference models `expQ/ft_mm_4b_184778|184780/**` and `expQ/archq3/**` (read them, run
  `expQ/freeze/check_module_move.sh`, write nothing). Campaigns `freeze` and `probes` are in STAND-DOWN and `longchunk`
  is closed: read their files, write nothing in them.

## Deliverable — how results reach the owner
`docs/spread_cases.md` (to write), rewritten (never appended) whenever an arm decides, holds four things:
(1) **the case table**: case (set, input size, source, question type, modality) | n | discordant pairs | spread judge
accuracy ↑ | tail judge accuracy ↑ | difference (95% CI, McNemar p) | uncompressed judge accuracy ↑ on the same
questions | compression rate (8 tokens per slot; slots per page / per question) | verdict better / equal / worse /
not resolvable;
(2) **the evaluation table**: for each deciding case, the same difference under the rubric judge, the project prompt,
EM and ANLS, plus the share of questions both layouts get right and both get wrong (what the gating eval can still move);
(3) **the mechanism table**: ownership, answer NLL and page transcription for the four stage-2 models and the four
finetunes, with their controls;
(4) **the hypothesis table**: hypothesis | probe or arm | n | number | gate | verdict | what it would change.
Every decided arm also gets one `expmon note` with its headline number, and the wake proposes in STATE.md the
replacement text for `docs/architecture_review.md:252-256` and HANDOFF.md thread 1 — the owner's chats write those files.
The wake produces it as soon as an arm decides; a result that exists only as a number in a log is not
delivered. Lacking tooling for it is a gripe (`expmon gripe`), not an excuse. The default page is
`expmon display --exp spread` (Settled / Open, the family's results table with live state, every clip and
image under the runs' result dirs); name here only what goes beyond it.

## Pre-authorised actions (the agent may do these without asking) and their loop guards
| situation | action | guard |
|---|---|---|
| a campaign job FAILED / NODE_FAIL / TIMEOUT for a transient cause (preemption without resume, node failure, a wall-time misestimate) | relaunch the same recipe, `--time` re-derived | max 2 relaunches per cell, then BLOCKED.md |
| `expmon check` prints `node <n>: N fast deaths` (a poisoned node killing jobs at model load) | relaunch the dead cells with `--exclude=<n>` before the launcher | 1 exclude-relaunch per cell |
| collapse: grad→0 ∧ l0→0, or loss pinned at ln(#candidates) for 2 probes, or NaN | `expmon kill`, `expmon verdict <jid> closed "<why>"` | never relaunch the same cell |
| `starved util=NN%` on a run older than 1 h | `expmon occupancy`; if the recipe is the cause: kill, `expmon optimize` (profile rung), relaunch the best-fit | 1 optimize sweep per arm |
| a grid cell of a listed arm is missing | `expmon sweep --exp spread --dry-run …`, then the real sweep | within CAMP_MAX_SUBMIT_PER_WAKE |
| a job PENDING > 6 h for Resources with an oversized request | `scontrol update JobId=… TimeLimit=…` (downward only) | never touch a job the campaign does not own |
| a job PENDING > 2 h for Resources with `--mem` above the profile rung's MaxRSS × 1.5 while cheaper twins run | `scontrol update JobId=… MinMemoryNode=…` (downward only) | never touch a job the campaign does not own |
| a cell PENDING > 6 h on `gpu-be` for Priority | `scontrol update JobId=… Partition=gpu,gpu-be` | at most 2 campaign cells on `gpu` at once; never move an arm-E training to `gpu-be` without re-deriving `--time` |
| the expmon daemon is gone | `expmon daemon` | — |
| a probe/eval straggler FAILED in seconds with a benign signature (known_issues.md) | ignore, note once | — |
| a cell's output is empty or unparsable while the job COMPLETED | verdict `closed "<why>"`, relaunch once with `--time` × 2 | never a third time |
| a requeued eval overwrites a complete judged file | keep `<file>.kept_<job>_<node>_<hhmm>` and say so in STATE.md | — |
| an arm-E training has written its checkpoint at about a quarter of the run | PREVIEW: the documents eval with `LIMIT=300`, judged, paired against 184778's final on the same 300 ids; report as `provisional:` with n = 300 | one preview per cell; a preview never validates or fails a gate |
| a reproduced crash whose fix is ≤ CAMP_PATCH_LINES lines inside a "## Patchable" path | apply it, smoke on the cheapest cell, record `<file> — <traceback line>` in STATE.md; the tick commits it (its path must be in CAMP_COMMIT_PATHS) | 1 patch per symptom, then BLOCKED.md |
Anything else: BLOCKED.md and stop.

## Patchable
CAMP_PATCH_LINES in campaign.conf is what is enforced and it wins over this section; the owner set it to **40** and
`CAMP_COMMIT_PATHS='scripts/spread docs/spread_cases.md'` (2026-11-17), so the wake writes the files marked "to write"
itself (arm F's recipes included: ONE change vs the recipe each copies, shown by a 20-step smoke's
`training_config.yaml` diff pasted in STATE.md). The three settings:
- **freely patchable** (the campaign's own code, where the wake iterates): `scripts/spread/**`, `expQ/spread/**`,
  `campaign_spread/**` (never campaign.conf), `docs/spread_cases.md`, new sets `eval_data/spread_*`.
- **patchable IF an invariant holds**, with its output pasted in STATE.md for that wake: `scripts/spread/*.sbatch` —
  SLURM header, paths, env plumbing, logging. Invariant: `bash -n` passes AND, for arm E, a 20-step smoke's
  `training_config.yaml` differs from `expQ/archq3/pt_mm_5M_spread/training_config.yaml` (stage 2) or
  `expQ/ft_mm_4b_184778/training_config.yaml` (finetune) only in `model_name_or_path`, the output paths and `max_steps`.
  A recipe a running or finished cell used is frozen for THAT cell.
- **never**: `pisco/**`, `test_*.py`, `scripts/eval_mm_docqa.py`, `eval_mm_4b.sh`, `pt_mm_4b.sh`, `ft_mm_4b.sh`,
  `qwen35_4b_pt.sh`, `scripts/judge_*.py`, `scripts/judge_prompts.py`, `scripts/paired_judge.py`, `scripts/probes/**`,
  `expQ/archq2/*.sh` and `scripts/longdoc/**` (run them, never edit), existing `eval_data/` sets, `data/**`, recorded
  result jsons (re-judge COPIES only), `expQ/**` other than `expQ/spread/`, everything of the two live siblings listed
  in "Budget and caps", and `EXPERIMENTS.md` except through `expmon note`. A crash there is BLOCKED.md with the
  traceback; the owner's chats fix it.
The WRITE BOUNDARY is not prose: `CAMP_WRITE_ROOTS` in campaign.conf
(default: this repo) is enforced by the wake hook, which also refuses `~/.claude`, campaign.conf, the tick's state dir
and sibling campaign dirs whatever this section says.

## The wake owns the science
The arms table is a starting hypothesis, not a work order: read the evidence, launch the cells (the owner does not),
propose arms the table lacks (one knob off the control, gate written BEFORE the cell), close decided arms, and amend
this law when the evidence refutes it — a wrong prediction is kept and marked, never quietly deleted. A campaign seeded
deliberately unfinished names the properties the wake may not weaken while fixing (GUIDE.md).
The prediction written before any cell, kept and marked if wrong: spread beats tail where the input leaves the finetune's
regime (long chunks, many pages) and ties it inside that regime, because both layouts hit the same compressed-page
ceiling; arm E therefore lands inside ±2 pt on 4-page documents.

## Hard rules
1. Submissions only via `expmon launch --exp spread` / `expmon sweep --exp spread`; never raw sbatch (enforced).
2. Cancel only campaign-owned jobs, only via `expmon kill`. Never `rm` a run dir — rename it `VOID_<name>` (enforced).
3. Never edit campaign.conf, crontab, hooks, `~/.claude`, the tick or the watchdog; never relax a rule here.
   Code edits only as "## Patchable" allows. A `REQUEST` file in this dir is the owner speaking (`expmon campaign
   resolve`): execute it before anything else, then delete it; nothing else may create one. Rules 1–3 are enforced by
   the wake hook: a refusal is a signal to write BLOCKED.md, never to find another spelling. The tick commits for you.
4. Hypothesis ≠ conclusion: write "on track to", "consistent with"; a conclusion needs its gate AND a replicate.
5. Report epochs / rows seen, never bare step counts (steps are incomparable across effective batch).
6. End every wake with one line in NEXT (`done` | `wake-when job:<ids> [max <N>m]` | `wake-in <min>` | `waiting-owner`), or nothing while work is left.
7. Dead ends — do not retry, hold constant what won. Compression rate 16 or 32 on the spread layout (docs −.065 / −.074
   and −.172 / −.198, rate round closed, EXPERIMENTS.md:508,519); the coarse-to-fine slot order archq3b (a permutation,
   not a hierarchy, EXPERIMENTS.md:503); the `<MEM>` mixer; a linear connector, a moving connector, a density-adaptive
   rate; a bidirectional compressor on Qwen3.5; PPE as an inference probe; per-head scales (SEAL, null); untrained slot
   dropping or blind slot halving as a coarse level (−.105 / −.115); re-running the slot-pruning ladder on the reference
   models (campaign_probes arm E, closed at both seeds down to an eighth); evidence-then-answer labels (+.001 / −.009);
   gold-only as a fix (+.022 / +.025); route-then-densify at rate 4 (+.063 / +.049, below gate); OCR words for photo
   text (−.434); passage order inside a chunk; the no-oracle top-3 grouping as a serving mode; long-chunk stage 1 at
   1,024 / 2,048 tokens (campaign_longchunk, the owner's stop of 2026-11-16; its loss sits in the compressed image page,
   docs/longdoc_diagnosis.md:15); a decoder-only QA finetune on a compressor that never saw QA (−2.8 / −3.5 pt,
   campaign_freeze); query-dependent compression, more finetune data, ViT LoRA, KL at finetune, RAFT, a rate changed at
   inference only. Never re-run the reduced-budget layout verdict itself (183814 / 183816 vs 183538 / 183539: recorded).
   Never judge with EM / F1 alone: the judge accuracy decides.
8. No training of any kind before the owner checkpoint, and then only arm E and arm F (at most 2 fixes), each on its own
   REQUEST. A cause or fix this campaign finds that needs new code in `pisco/` is a hand-off, never an arm here.
9. Never touch the live siblings `longdoc` and `qafreeze` — their jobs, dirs or outputs (see "Budget and caps") — and
   never the expmon daemon job 188764 except through `expmon daemon`.
10. Never run python, pixi, a dataset load or a model load on the login node: every script goes through `expmon launch`
    (CPU jobs on `gpu-be` with `--gres=gpu:0`). Stdlib python on small JSON (`scripts/paired_judge.py`) is fine.

## Wake checklist (expmon grid tooling)
1. `expmon grid spread` — where every cell stands (state, verdict, factors, RESULT / latest probe).
2. `expmon verdict <jid> closed|best "<why>"` for decided arms.
3. Fill missing cells: `expmon sweep --exp spread --stage <S> --parent <jid> --dry-run <launcher> 'V=a,b'`, then for real.
4. `expmon probe <jid> --step <epoch> <metric>=<val>` for in-flight arms.
5. Rewrite STATE.md and `docs/spread_cases.md`; write NEXT.

## Escalation
BLOCKED.md (what, why, options, recommendation) and stop. The watchdog mails herve.dejean@naverlabs.com and writes ALERT for what
the loop cannot fix itself: stale ticks, wakes failing fast (login), no GPU-hour for CAMP_IDLE_H, a rejected
push, unpushed work, low disk, missing cron or daemon.

## Close-out (after CAMP_UNTIL)
`expmon campaign report` (one screen; its Settled / Open block is the successor's brief: `expmon campaign init <label> --from <this dir>`) → `expmon campaign retro` → fill its Lessons → `expmon gripe -c retro "RETRO spread: WENT OK: … WENT NOT OK: … NET: …"`
(one line; the cross-cluster feedback digest scores campaigns from it) → the OWNER pastes this dir's SYNC.md (written by retro) into ~/.claude/SYNC.md — never the wake (hard rule 3) → `expmon campaign disarm`.
