# CAMPAIGN qafreeze — pisco   (the law; the agent re-reads this every wake; rewrite, never append)

created 2026-11-17 · kind **train** · root `/beegfs/scratch/user/hdejean/pisco` · window until **2027-02-15** (launches allowed through that day 23:59) · owner herve.dejean@naverlabs.com · wakes run `claude-opus-5`

## Mission
Successor of campaign 'freeze' (its report: /beegfs/scratch/user/hdejean/pisco/campaign_freeze/REPORT.md — not written yet; REPORT.json carries the same Settled list). Open at hand-off: nothing — the previous campaign settled every arm.
**Question (owner, QUESTION.md; HANDOFF.md:66-70):** campaign freeze froze a compressor that had NEVER seen question-answering training. Can a compressor that HAS had one QA finetune then stay FROZEN while a fresh decoder LoRA is trained on the same rows? That is the case `<MEM>` caching needs: compress a collection once, train only the decoder afterwards.
**Arm Q (what is tested):** the reference finetune recipe of 184778 / 184780 with ONE change — it starts from a hybrid init: R's own `compressor/` + `connector.pt` (QA-trained) with R's `decoder_state.pt` replaced by the stage-2 decoder (`expQ/archq3/pt_mm_5M_spread/model/decoder_state.pt`, R's own starting point), run with `+model_config_overrides.freeze_compressor=true`, so only the decoder LoRA trains. The hybrid is assembled by copying R's model dir and swapping that one file — the only route that needs no new code, since `PISCO.from_pretrained` reads exactly `decoder_state.pt`, `compressor/compressor_state.pt`, `connector.pt` and `config.json` (pisco/model.py:1376-1405); the two configs differ only in `connector_out_norm` / its init / the image-processor path (`expQ/ft_mm_4b_184778/model/config.json:11-12,24`), and R's config travels with R's connector.
**The deciding number:** paired judge accuracy (Qwen3-8B rubric judge `scripts/judge_mm_docqa.py`; `scripts/paired_judge.py` = bootstrap 95% CI + McNemar), Q − R of the same seed, n = 1,500 questions per set, seeds 42 (vs 184778) and 43 (vs 184780). Seed 43 freezes 184780's compressor (the seed-43 reference), so each seed is one self-contained replicate; the owner's question names only 184778.
Gating sets: documents `eval_data/mm_docqa_1500_1m`, photos `eval_data/mm_naturalqa_1500`, MRQA text `eval_data/text_mrqa_ood_1500`. HotpotQA `eval_data/text_hotpotqa_1500` is reported, not gating (freeze's F − R on the whole HotpotQA set: −2.8 / −2.3 / −2.6, campaign_freeze/STATE.md:13; on its bridge questions alone −3.0 / −2.8 / −3.6, QUESTION.md:10).
PASS = every gating set ≥ −2 pt with no CI wholly below 0 at both seeds → the `<MEM>` cache is built (a hand-off; see On acceptance). FAIL = ≤ −3 pt with the CI wholly below 0 at both seeds.
**Reference numbers, same instrument** (campaign_freeze/STATE.md:66-85, n = 1,500 per set): R documents .610 / .615, photos .673 / .685, MRQA .718 / .724, HotpotQA .603 / .593; F (stage-2 compressor + connector frozen, decoder only) − R documents −.028 / −.035 / −.041; A1 (compressor only frozen) − R −.029 / −.039 / −.035; A2 (connector frozen) − R +.023 / −.006; uncompressed decoder on the same questions documents .809, photos .717, MRQA .807.
**Prediction, written before any cell (kept and marked if wrong):** Q − R on documents lands in [−1, +2] pt — A1 − F ≈ 0 says the connector cannot recover what the compressor's QA training gives (STATE.md:28), A2 − F ≈ +5 / +3 pt says that training is the carrier (STATE.md:83); once it is already inside the frozen compressor, a decoder trained against a fixed target should tie R.
**What a PASS does not show:** that the frozen compressor transfers to data it never saw. That is a later campaign (campaign_freeze/STATE.md:60), not an arm here.
Project invariants: CLAUDE.md "Multimodal PISCO"; history: docs/training_plan.md:65-112 ("Standing proposal", stage A / stage B), HANDOFF.md thread 7 and Decisions pending 1.
If it replicates or extends a paper, `papers/<slug>.md` (`expmon paper`, skill `paper`) is the source of every number
quoted here — cite the card, not the PDF.

## Arms — what varies, what is held constant
| arm | recipe file (committed) | factors (what varies) | held constant | gate (metric ≥ X vs whom) | budget (GPU-h) |
|---|---|---|---|---|---|
| R (reference, exists; 184778 s42 / 184780 s43) | `ft_mm_4b.sh` as recorded in campaign_freeze/CAMPAIGN.md:35 | — | — | the comparison point | 0 (done: 11.0 / 13.8 h) |
| B (init build + check) — **the feasibility gate** | `scripts/qafreeze/build_init.sbatch` (to write; CPU, `-p gpu-be --gres=gpu:0`) + `expQ/freeze/check_module_move.sh` (exists; run it, never edit it) | builds `expQ/qafreeze/init_s42` and `init_s43` | the four checkpoint files and R's config | feasibility gate below | 0 |
| Qs (20-step smoke, seed 42) | `scripts/qafreeze/ft_qafreeze.sbatch frozen 42 20` (to write: `expQ/freeze/ft_freeze.sh` with the init dir as an argument) | `max_steps=20` | as Q | plumbing check below | 0.6 |
| Q (QA-trained compressor frozen) seeds 42, 43 | `scripts/qafreeze/ft_qafreeze.sbatch frozen 42` / `… 43` | init = hybrid (R's compressor + connector, stage-2 decoder) AND `freeze_compressor=true` | 28k rows (docpages 20k + natural 8k, teacher labels), bs 1 × accum 64, lr 2e-5, bf16, 1,003,520 px, rate 8, spread layout stride 8, 1 epoch = 438 steps, seed | Decision gates | ≤ 10 each + evals |
| Q3 (third seed 44, only on a tie) | `… frozen 44`, init from the seed-44 reference 188652's model | seed | as Q | 2 of 3 | ≤ 10 + evals |
| H (decoder LoRA lr 1e-4) — only after the owner checkpoint | `scripts/qafreeze/ft_qafreeze.sbatch frozen_hi 42` / `… 43` (new `ARM)` case; patch rule below) | lr 2e-5 → 1e-4 (2e-5 is a full-finetune value, docs/training_review.md §4) | as Q | ≥ +2 pt over Q on documents with the CI above 0, and no gating set ≤ −2 pt vs R, at both seeds | ≤ 10 each + evals |
| (slot-mixing layer on cached `<MEM>`s, QUESTION.md:15) | needs new code in `pisco/` — **not an arm here**: a hand-off, and only if Q fails | | | | 0 |

Evals of each finetuned model, judge chained (`--dependency afterok`), on `gpu-be --exclude=homer-[005-009]` (homer-008
changes image answers, campaign_freeze/CAMPAIGN.md:42): four sets per model — documents, photos, MRQA text, HotpotQA.
Copy the exact command AND env of the recorded reference cells 188597 / 188602 / 188606 / 188615 (`expmon lineage <id>`);
do not retype the env from memory. The reference side of every pair is the judged file of 184778 / 184780 written by the
SAME code version: md5 `eval_mm_4b.sh` 50abc167, `scripts/eval_mm_docqa.py` dfe32aa0, `scripts/judge_mm_docqa.py`
02bc8748, `scripts/paired_judge.py` a3cc13b2 (campaign_freeze/STATE.md:29). Commit 5955890 touched the eval scripts, so
expect a mismatch: when any md5 differs, re-run the reference cell FIRST (`--stage eval-ref`) and pair only files that
ran on the same md5. Record all four md5s plus `pisco/model.py` in every intent note.

Rules: every arm is a committed recipe file, never raw overrides on the launch line. Before launching two arms
that will be compared, diff their resolved configs and list EVERY difference in this table — a comparison with
an unlisted difference is not a result (splare lost both headline claims this way). A relaunch repeats the
original command exactly (a flag left at its default is not a factor, but a changed default is a confound).
Before the first full cell, diff the smoke's `training_config.yaml` against `expQ/ft_mm_4b_184778/training_config.yaml`
and paste it in STATE.md: the ONLY allowed differences are `model_name_or_path` (the hybrid init),
`model_config_overrides.freeze_compressor`, `hydra.run.dir` / output paths and `max_steps` (smoke only).

## Decision gates
Metric unit: judge accuracy in [0, 1]; "pt" = .01. Every difference is PAIRED per question, same seed, n = 1,500 per set.
- **Feasibility gate (first, ALONE, 0 GPU-h): the hybrid init is what it claims to be.** Build `init_s42`, then two CPU
  runs of `check_module_move.sh`:
  (1) `… expQ/ft_mm_4b_184778/model expQ/qafreeze/init_s42` → every `compressor.*` and `connector.*` row **changed 0.0%**
  (R's QA-trained compressor side arrived intact) and `decoder.lora_B` changed > 60% (the decoder really was reset);
  (2) `… expQ/archq3/pt_mm_5M_spread/model expQ/qafreeze/init_s42` → every `decoder.*` row **changed 0.0%** (the decoder
  equals R's own starting point) and `compressor.lora_B` changed > 60% — the known-positive control: R moved it 74%
  (campaign_freeze/STATE.md:107), so the check sees movement when there is some;
  (3) the builder asserts the swapped `decoder_state.pt` matches the stage-2 file in key names and shapes.
  FAIL on any of the three = the init cannot be assembled from saved checkpoints, so the question cannot be asked without
  new code in `pisco/` (never patchable here): the campaign STOPS with that as the finding — BLOCKED.md with the rows,
  NEXT `waiting-owner`. A builder bug is not a finding: fix it (patch rule) and re-run.
- **Plumbing check before the full cells (never a result):** the 20-step smoke, then
  `check_module_move.sh expQ/qafreeze/init_s42 <smoke>/model`. PASS = the log prints `freeze_compressor=True: froze N
  compressor + connector tensors` and `Compressor trainable parameters: 0`; every `compressor.*` / `connector.*` row 0.0%;
  `decoder.lora_B` > 0%; the config diff above is clean. Record s/step and peak GPU memory; `--time` of a full cell =
  s/step × 438 × 1.25. Compare a 20-step loss only with a 20-step smoke (known_issues row 22): steps 10 and 20
  within .10 of the frozen smoke 188528 (.594 / .584); outside that band is BLOCKED.md, not a fail.
- **Mid-run health, per arm (a score without its health number is not reported):** `eval_loss` every 100 steps
  (`expmon probe`), read against F's curve at the same epoch (.455 / .444 / .438 / .434 / .435 at .23 / .46 / .69 / .92 /
  1.0 epoch, seed 42) and R's (.513 / .455 / .393 / .394 / .414) — campaign_freeze/STATE.md:98-99. NaN, or a loss above
  this run's first probe for 2 probes → `expmon kill`, closed. After each full run: `check_module_move.sh
  expQ/qafreeze/init_s<seed> <run>/model` → compressor and connector rows still 0.0%, decoder lora_B > 60% (F full runs
  66.3-66.7%, STATE.md:106).
- **Health per eval cell:** 1,500 records, a `judge` verdict on every row, 0 unparseable, `image_max_pixels` 1003520 on
  the image sets, no "exceeds --decoder_max_length" error, node outside homer-[005-009] recorded with the file. A cell
  that fails it is `closed "health"` and relaunched once.
- **Validated (a QA-trained compressor can stay frozen):** documents, photos AND MRQA text, at BOTH seeds, Q − R ≥ −2 pt
  with the CI not wholly below 0.
- **Failed (even a QA-trained compressor must keep training):** any gating set ≤ −3 pt with the CI wholly below 0 at both seeds.
- **Tie:** anything else → the third seed 44 against reference 188652 (its judged files exist), then decide 2 of 3,
  as campaign_freeze/CAMPAIGN.md:76-79 did; still a tie after three seeds → BLOCKED.md, no fourth seed.
- **Known-positive control of the instrument, in the wake that reads the gate:** (a) `paired_judge.py` on R's documents
  file against itself must give 0.000 (the no-op); (b) report Q − F of the same seed on documents next to the gate — the
  same instrument measured F − R = −2.8 / −3.5 pt and A2 − F = +5.1 / +2.9 pt at this n, so an effect of this size
  registers. If Q passes, Q − F on documents must be ≥ +2 pt with the CI above 0; if it is not, the pass and the
  freeze result disagree → BLOCKED.md before any claim. F's files must carry the current eval-code md5 (else re-run F
  documents at both seeds: 2 cells).
- Noise band: CAMP_NOISE_PT = 1.2 pt (campaign.conf) — below it nothing is a result; the owner's −2 / −3 pt bars above
  are not relaxed by it. Mid-run gate: compare each probe with F's and R's curve at the same epoch — "on track to tie"
  is a reason to redirect, not to finish.
- A NEW metric (one the wake designs itself) closes nothing until its row here states its n (cells or pairs) and
  a known-positive control it recovers (variety shipped a null built from one seed pair). Next to any threshold count, report the paired per-item test —
  navmem's count showed nothing where a paired sign test found the one real effect (98 closer / 59 farther, p=0.002; control p=0.22).
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
  never a second repeat.
- **On acceptance:** nothing is switched on by the wake. A PASS makes the `<MEM>` cache an ENGINEERING hand-off
  (`expmon handoff add --to pisco "build the <MEM> cache: compress once, store, train the decoder on stored vectors"`,
  as campaign_freeze/CAMPAIGN.md:126-127 reserved) plus ONE owner question in BLOCKED.md; a winning arm H is proposed as a
  recipe change, never applied. Default-on is refused because four of the five parts of a proper judge are missing here:
  (1) judge frozen for the wake — mechanical only while `CAMP_PATCH_LINES=0` (campaign.conf); no hook or `CAMP_READ_DENY`
  covers `scripts/judge_*.py`, `scripts/judge_prompts.py`, `scripts/paired_judge.py`, so the "never" list below is prose;
  (2) controls — the no-op is refused (a file paired with itself gives 0) but no tool plants a gain: MISSING;
  (3) a second program tested on planted cases: MISSING (`scripts/judge_agreement.py` measures agreement; the human sheet
  is unlabelled, HANDOFF.md:59); (4) paired on every item: yes (`paired_judge.py`), but every campaign has reused these
  four eval sets: no untouched set; (5) automatic revert on a worse yardstick: MISSING. So: opt-in, then one question to
  the owner.

## Budget and caps
- GPU-hours for the whole campaign: **55** (2.3 GPU-days), built from measured costs: 2 full cells ≤ 10 h each
  (F ran 8.7 / 9.3 / 8.7 h at 71.5-76.1 s/step × 438, campaign_freeze/STATE.md:117) = 20; smoke 0.6 (188528 = 0.5);
  8 judged evals of Q ≈ 4 (documents 28-98 min, photos 18, text 17 min, campaign_longdoc/CAMPAIGN.md:159-161); up to 8
  reference re-runs ≈ 4; 2 F re-runs ≈ 1; a third seed on a tie ≈ 14; CPU cells 0; slack ≈ 11. Over it: BLOCKED.md.
- **+25 GPU-h released only by the owner checkpoint** for arm H (2 trainings + 8 evals) → campaign cap 80.
- cells ≈ 30 for arm Q; LLM wakes ≤ 25 — a full cell runs ~10 h, so wait with `wake-when job:<ids>`, never a timer.
- at most CAMP_MAX_TRAIN trainings in flight; "in flight" = submitted and not ended, PENDING included, and a 20-step smoke
  counts as a training (campaign_freeze/CAMPAIGN.md:133-134); at most CAMP_MAX_SUBMIT_PER_WAKE new submissions per wake (campaign.conf)
- conf set by the owner (2026-11-17; the conf always wins over this law): `CAMP_PATCH_LINES=40`,
  `CAMP_COMMIT_PATHS='scripts/qafreeze'`, `CAMP_MAX_SUBMIT_PER_WAKE=4`. The wake writes `build_init.sbatch` and
  `ft_qafreeze.sbatch` itself, under "## Patchable".
- every finetune is 1 GPU on `gpu`; the owner's cap is 8 GPUs on `gpu`, shared: run `gpucap land a100 1` before each
  training launch, never take the last free slot, and read known_issues row 26 (gpucap counts GPUs, not free CPUs; the
  recipe asks 8). Evals and CPU checks go to `gpu-be` (preemptible) with `--exclude=homer-[005-009]`.
- every launch: `--time` from measured s/step × steps × 1.25; `--mem`/`--cpus` as the recipe states (96 G, 8 CPUs)
- no launch below CAMP_DISK_FLOOR_GB free; nothing new after CAMP_UNTIL. Each hybrid init copies one checkpoint tree per seed.
- **Owner checkpoint:** none before the feasibility gate, the smoke and arm Q's two full cells (and the third seed on a
  tie) — the owner approved exactly those at ~9 GPU-h per seed (QUESTION.md:4-6, HANDOFF.md:66-70), nothing is
  irreversible and no shared file is touched; one checkpoint BEFORE arm H (lr 1e-4) and before anything needing
  new code in `pisco/` — the wake writes Q's table, ONE question in BLOCKED.md and NEXT `waiting-owner`, and continues
  only on a REQUEST.
- **Live sibling `longdoc`** (campaign_longdoc/CAMPAIGN.md, campaign.conf) and what it uses:
  - GPUs / partitions: 1 GPU per inference cell, `-p gpu-be --exclude=homer-[005-009]` by default, at most 2 cells on
    `gpu` at once; budget 70 cells / 60 GPU-h; no training of any kind (its hard rule 8).
  - Machines: the shared homer A100 nodes (its predecessor ran on homer-012 / 013 / 016 / 019 / 020); none reserved.
  - Engine trees / worktrees: none of its own — it shares THIS checkout (branch `arc`) and its pixi env.
  - Dirs it writes: `scripts/longdoc/**`, `expQ/longdoc/**`, `docs/longdoc_diagnosis.md`, new sets `eval_data/longdoc_*`;
    it reads other campaigns' models through `expQ/longdoc/` symlinks. Shared files it MAY patch (invariant-gated):
    `scripts/eval_mm_docqa.py` and `eval_mm_4b.sh`.
  - Wake quota: the same account (a 5-hour usage limit stops both, known_issues row 27); CAMP_MAX_SUBMIT_PER_WAKE 4,
    CAMP_MAX_USD 8, heartbeat 24 h.
  This campaign therefore: writes nothing under `scripts/longdoc/**`, `expQ/longdoc/**`, `campaign_longdoc/**`,
  `docs/longdoc_diagnosis.md` or `eval_data/longdoc_*`; never kills, requeues, relaunches or verdicts a longdoc job;
  never patches the shared eval or judge scripts (longdoc may change them — record their md5 with EVERY cell and pair
  only cells that ran on the same md5: a longdoc patch landing mid-campaign invalidates a pair); keeps at most 2 GPUs on
  `gpu` and its evals on `gpu-be`.
- **Live sibling `spread`** (campaign_spread/CAMPAIGN.md, armed 2026-11-17 with this one; added by the first wake): one
  training at a time on `gpu` (its arm E) inside the same owner 8-GPU cap, evals on `gpu-be`; writes `scripts/spread/**`,
  `expQ/spread/**`. This campaign writes nothing there or in `campaign_spread/**` and never touches its jobs.
  Campaigns `freeze` and `probes` are in STAND-DOWN and `longchunk` is closed: read
  their files and run `expQ/freeze/check_module_move.sh`, write nothing in them.

## Deliverable — how results reach the owner
One table in STATE.md (and the decision in the ledger via `expmon note`): rows = R reference, F frozen (recorded), A2
connector-frozen (recorded), Q, then arm H if it runs, last row the uncompressed decoder; columns = what trains,
compression rate (8 tokens per slot; 123 slots per 1M-px page), then per set the judge accuracy ↑ at seed 42 / seed 43
and the paired difference to R with its 95% CI, McNemar p and n: documents, photos, MRQA text, HotpotQA. Every column
names its metric and direction. Under it: the module-movement table of every run (entries changed %, per module, against
that run's init) and the training cost (hours, s/step, peak GPU memory) of Q against R and F. The wake proposes in
STATE.md the replacement text for `docs/training_plan.md:103-106` ("Open: …") and HANDOFF.md Decisions pending 1 — the
owner's chats write those files. `expmon display --exp qafreeze` is the live page (Settled / Open, the family's results
table with live state); name here only what goes beyond it. A result that exists only as a number in a log is not
delivered; lacking tooling for it is a gripe (`expmon gripe`), not an excuse.

## Pre-authorised actions (the agent may do these without asking) and their loop guards
| situation | action | guard |
|---|---|---|
| a campaign job FAILED / NODE_FAIL / TIMEOUT for a transient cause (preemption without resume, node failure, a wall-time misestimate) | relaunch the same recipe, `--time` re-derived | max 2 relaunches per cell, then BLOCKED.md |
| `expmon check` prints `node <n>: N fast deaths` (a poisoned node killing jobs at model load) | relaunch the dead cells with `--exclude=<n>` before the launcher | 1 exclude-relaunch per cell |
| collapse: grad→0 ∧ l0→0, or loss pinned at ln(#candidates) for 2 probes, or NaN | `expmon kill`, `expmon verdict <jid> closed "<why>"` | never relaunch the same cell |
| `starved util=NN%` on a run older than 1 h | `expmon occupancy`; if the recipe is the cause: kill, `expmon optimize` (profile rung), relaunch the best-fit | 1 optimize sweep per arm |
| a grid cell of a listed arm is missing | `expmon sweep --exp qafreeze --dry-run …`, then the real sweep | within CAMP_MAX_SUBMIT_PER_WAKE |
| a job PENDING > 6 h for Resources with an oversized request | `scontrol update JobId=… TimeLimit=…` (downward only) | never touch a job the campaign does not own |
| a job PENDING > 2 h for Resources with `--mem` above the profile rung's MaxRSS × 1.5 while cheaper twins run | `scontrol update JobId=… MinMemoryNode=…` (downward only) | never touch a job the campaign does not own |
| the expmon daemon is gone | `expmon daemon` | — |
| a probe/eval straggler FAILED in seconds with a benign signature (known_issues.md) | ignore, note once | — |
| a reproduced crash whose fix is ≤ CAMP_PATCH_LINES lines inside a "## Patchable" path | apply it, smoke on the cheapest cell, record `<file> — <traceback line>` in STATE.md; the tick commits it (its path must be in CAMP_COMMIT_PATHS) | 1 patch per symptom, then BLOCKED.md |
| `launch: REFUSED — this cell … was closed as a dead arm by run <id>` on a cell never tried (known_issues row 30: this campaign's launchers are not in `scripts/expmon_factors.py`, so the signature is empty) | confirm with `expmon lineage <id>` that the clashing run has no factors and is another arm, then `expmon launch --force …` and say so in the intent note | once per cell |
| a training of a listed arm has written its checkpoint at about a quarter of the run (step ~100 of 438) | PREVIEW: the documents eval with `LIMIT=300`, judged, paired against R's final on the same 300 query ids; report as `provisional:` with n = 300 (campaign_freeze/CAMPAIGN.md:165) | one preview per cell; a preview never validates or fails the gate |
| a training PENDING > 6 h on `gpu` for Resources | resubmit with `--partition gpu-be` (preemptible; a preemption relaunch counts as transient) | 1 move per cell; never move an EVAL to `gpu` (that would relax a rule: owner only) |
| a requeued eval overwrites a complete judged file | keep `<file>.kept_<job>_<node>_<hhmm>` (known_issues rows 21 / 31) | — |
Anything else: BLOCKED.md and stop.

## Patchable
CAMP_PATCH_LINES in campaign.conf is what is enforced and it wins over this section; the owner set it to **40** and
`CAMP_COMMIT_PATHS='scripts/qafreeze'` (2026-11-17), so the wake writes `scripts/qafreeze/build_init.sbatch` and
`scripts/qafreeze/ft_qafreeze.sbatch` (both **to write**) itself. The three settings:
- **freely patchable** (the campaign's own code, where the wake iterates): `scripts/qafreeze/**`, `campaign_qafreeze/**`
  (never campaign.conf), run dirs under `expQ/qafreeze/**`.
- **patchable IF an invariant holds**, with its output pasted in STATE.md: `scripts/qafreeze/ft_qafreeze.sbatch` — its
  SLURM header, paths, env plumbing, logging, or ONE new `ARM)` case for an arm whose gate is already written in this law
  (arm H); never an existing arm's training argument. Invariant: `bash -n` passes AND a 20-step smoke's
  `training_config.yaml` differs from arm Q's only in that arm's documented field. A recipe a running or finished cell
  used is frozen for THAT cell.
- **never**: `pisco/**`, `test_*.py`, `scripts/eval_mm_docqa.py`, `eval_mm_4b.sh`, `ft_mm_4b.sh`, `scripts/judge_*.py`,
  `scripts/judge_prompts.py`, `scripts/paired_judge.py`, `data/**`, `eval_data/**`, the reference models
  `expQ/ft_mm_4b_184778|184780/**` and `expQ/archq3/**`, `expQ/freeze/**` (campaign freeze's recipes: run
  `check_module_move.sh`, never edit it), and everything of the live sibling listed in "Budget and caps". A crash there is
  BLOCKED.md with the traceback; the owner's chats fix it.
The repo `.gitignore` drops new `*.sh` files and all of `expQ/*` (.gitignore:11,21), so the launchers are `.sbatch` under
`scripts/qafreeze/` and that path must be in CAMP_COMMIT_PATHS or nothing is committed. The WRITE BOUNDARY is not prose:
`CAMP_WRITE_ROOTS` in campaign.conf (default: this repo) is enforced by the wake hook, which also refuses `~/.claude`,
campaign.conf, the tick's state dir and sibling campaign dirs whatever this section says.

## The wake owns the science
The arms table is a starting hypothesis, not a work order: read the evidence, launch the cells (the owner does not),
propose arms the table lacks (one knob off the control, gate written BEFORE the cell), close decided arms, and amend
this law when the evidence refutes it — a wrong prediction is kept and marked, never quietly deleted. A campaign seeded
deliberately unfinished names the properties the wake may not weaken while fixing (GUIDE.md).

## Hard rules
1. Submissions only via `expmon launch --exp qafreeze` / `expmon sweep --exp qafreeze`; never raw sbatch (enforced).
2. Cancel only campaign-owned jobs, only via `expmon kill`. Never `rm` a run dir — rename it `VOID_<name>` (enforced).
3. Never edit campaign.conf, crontab, hooks, `~/.claude`, the tick or the watchdog; never relax a rule here.
   Code edits only as "## Patchable" allows. A `REQUEST` file in this dir is the owner speaking (`expmon campaign
   resolve`): execute it before anything else, then delete it; nothing else may create one. Rules 1–3 are enforced by
   the wake hook: a refusal is a signal to write BLOCKED.md, never to find another spelling. The tick commits for you.
4. Hypothesis ≠ conclusion: write "on track to", "consistent with"; a conclusion needs its gate AND a replicate.
5. Report epochs / rows seen, never bare step counts (steps are incomparable across effective batch).
6. End every wake with one line in NEXT (`done` | `wake-when job:<ids> [max <N>m]` | `wake-in <min>` | `waiting-owner`), or nothing while work is left.
7. Settled by campaign 'freeze' (hold constant what won, do not retry what closed): A2 connector frozen VALIDATED at seeds 42/43 (190020): every gating set >= -2 pt vs reference, no CI wholly below 0; ties the reference recipe — jobs 189842, 189899 (2 cells), best; unit tests of freeze_compressor pass (test_freeze.py 5/5: only decoder trainable, no gradient to the frozen side, frozen side in eval mode, flag off = old behaviour), exit 0. Ran on the UNCOMMITTED copy 9dd687d+dirty; the flag + test were committed afterwards as d167725 (no committed run yet; re-run on HEAD planned) — job 188517, closed; recipe bug: max_steps needs +append; fixed, relaunched — job 188526, closed; never ran: dependency of the failed smoke 188526 (recipe bug); replaced by 188529 — job 188527, closed; feasibility gate PASS (frozen 20-step smoke, 1,280 rows = .046 epoch): log froze 506 compressor+connector tensors, compressor trainable 0, total trainable = decoder 155,978,240; module move (188578): every compressor/connector row incl. norm gain changed 0.0%, decoder lora_B 76.3%; train loss step 10 .594 vs unfrozen smoke .639 (-.045, within .05), step 20 .584 vs .580 (+.004); config diff vs 184778 = max_steps + freeze_compressor only; 85.1 s/step overall (72-76 steady) vs unfrozen 100.7 (88-90); sacct gpumem 49.7 GB vs unfrozen 31.9 GB — job 188528, closed; superseded by 188578: same rows 0.0%, but the check script skipped the connector norm gain (absent from the stage-2 init, retrofitted at load); patched — job 188529, closed
8. Never run python, pixi, a dataset load or a model load on the login node: every script goes through `expmon launch`
   (CPU jobs on `gpu-be` with `--gres=gpu:0`). Stdlib python on a small JSON (`paired_judge.py`) is fine.
9. Dead ends — do not retry; hold constant what won. A decoder-only finetune on a compressor that never saw QA training,
   frozen whole or compressor-only (documents −2.8 / −3.5 / −4.1 and −2.9 / −3.9 / −3.5 pt; campaign_freeze/STATE.md:9-18);
   training the connector on top of a frozen compressor (A1 − F −.001 / −.004 / +.006, STATE.md:28); freezing the
   connector as a speed fix (A2 ties R and saves nothing, STATE.md:53); a connector that MOVES in the finetune (own
   connector lr, linear connector: documents −.06, photos −.03) and a trainable norm gain under fp32 (documents −.17);
   fp32 masters at lr 2e-5 (only 5e-6 ties); query-dependent compression; more finetune data; ViT LoRA; KL at finetune;
   RAFT; a rate changed at inference only (campaign_freeze/CAMPAIGN.md:207-211; docs/training_plan.md:114-121);
   compression rate 16 or 32, the tail layout, the `<MEM>` mixer, a bidirectional compressor on Qwen3.5
   (campaign_longdoc/CAMPAIGN.md:273-275); long-chunk stage 1 at 1,024 / 2,048 tokens (the owner's stop, 2026-11-16).
   Never change the collator, the data, the rate, the layout or the stage-1 / stage-2 recipes: those are other campaigns.
   Never judge with EM / F1 alone: the judge accuracy decides.
10. Never touch the live sibling `longdoc`'s jobs, dirs or outputs, and never the expmon daemon job 188764 except through
    `expmon daemon`.

## Wake checklist (expmon grid tooling)
1. `expmon grid qafreeze` — where every cell stands (state, verdict, factors, RESULT / latest probe).
2. `expmon verdict <jid> closed|best "<why>"` for decided arms.
3. Fill missing cells: `expmon sweep --exp qafreeze --stage <S> --parent <jid> --dry-run <launcher> 'V=a,b'`, then for real.
4. `expmon probe <jid> --step <epoch> <metric>=<val>` for in-flight arms.
5. Rewrite STATE.md; write NEXT.

## Escalation
BLOCKED.md (what, why, options, recommendation) and stop. The watchdog mails herve.dejean@naverlabs.com and writes ALERT for what
the loop cannot fix itself: stale ticks, wakes failing fast (login), no GPU-hour for CAMP_IDLE_H, a rejected
push, unpushed work, low disk, missing cron or daemon.

## Close-out (after CAMP_UNTIL)
`expmon campaign report` (one screen; its Settled / Open block is the successor's brief: `expmon campaign init <label> --from <this dir>`) → `expmon campaign retro` → fill its Lessons → `expmon gripe -c retro "RETRO qafreeze: WENT OK: … WENT NOT OK: … NET: …"`
(one line; the cross-cluster feedback digest scores campaigns from it) → the OWNER pastes this dir's SYNC.md (written by retro) into ~/.claude/SYNC.md — never the wake (hard rule 3) → `expmon campaign disarm`.
