# CAMPAIGN freeze — pisco   (the law; the agent re-reads this every wake; rewrite, never append)

created 2026-11-11 · kind **train** · root `/beegfs/scratch/user/hdejean/pisco` · window until **2027-02-09** (launches allowed through that day 23:59) · owner herve.dejean@naverlabs.com · wakes run `claude-opus-5-5`

Audits on this campaign: the read-only verifier after every wake that changes a claim (VERIFY.md), the escalation check
on every new BLOCKED.md question, and `expmon review` every Monday. Requested by the owner on 2026-11-11 ("train the
compressor, then freeze it and train the decoder LoRA"; "unattended, with judges").

## Mission
Can the question-answering finetune train the DECODER LoRA ONLY, with the compressor and the connector frozen, and reach
the same judge accuracy as the reference finetune that trains everything? If yes, a collection is compressed ONCE (one
stored index of `<MEM>` vectors) and every later usage is a cheap decoder-only training on cached vectors: this is
"stage A / stage B" in `docs/training_plan.md` ("Standing proposal"), and this campaign is its gate 0, then its first
stage-B arms. "Frozen compressor" here = compressor adapter + `<MEM>` token row + merger + connector (norm gain
included), all `requires_grad=False` AND in eval mode during training (no LoRA dropout on the frozen side), so a
training `<MEM>` equals the one `PISCO.compress` would cache (`PISCOConfig.freeze_compressor`, `test_freeze.py`).
The deciding number is the paired judge accuracy (Qwen3-8B rubric judge, `scripts/paired_judge.py`) of the frozen
finetune minus the reference finetune of the same seed, on documents, photos and text (all three: every capability
must hold for text, images and mixed). Why it may tie: in the bf16 reference recipe the finetune already moves the
connector by .03% and a connector that moves more costs ~.03 on photos (training_plan.md, warning 2026-11-01).
(Refuted wake 10, 2026-11-12, kept: the gate FAILED on three seeds — documents −.028 / −.035 / −.041, CI below 0 at
all three, ≤ −3 pt at seeds 43 and 44; MRQA −.023 / −.027 / −.025; photos tied −.009 / −.013 / +.005; jobs 188650 /
188701, n = 1,500 per set. The compressor side's finetune matters for documents and text, not for photos.)
Project invariants: CLAUDE.md "Multimodal PISCO"; history: docs/training_plan.md, HANDOFF.md thread 7.

## Arms — what varies, what is held constant
Reference chain (exists, never retrained): stage 1 `expQ/archq3/pt_text_5M_spread` → stage 2
`expQ/archq3/pt_mm_5M_spread/model` (job 184777) → finetunes `expQ/ft_mm_4b_184778/model` (seed 42) and
`expQ/ft_mm_4b_184780/model` (seed 43). Known judge accuracy of the references (1,500 Q): documents .610 / .615,
photos .677 / .685, MRQA text .718 (seed 42). Uncompressed decoder on the same questions: documents .809, photos .717,
MRQA .807.

| arm | recipe file (committed) | factors (what varies) | held constant | gate (metric vs whom) | budget (GPU-h) |
|---|---|---|---|---|---|
| R (reference, exists) | `ft_mm_4b.sh <stage-2 model> 28000 1 64 hf_training.learning_rate=2e-5 +model_config_overrides.image_max_pixels=1003520` (+ `+hf_training.seed=43`), env `FT_DATA` as in the recipe file | — | — | the comparison point | 0 (done: 184778 11.0 h, 184780 13.8 h) |
| F (frozen) | `expQ/freeze/ft_freeze.sh frozen 42` and `… frozen 43` | `freeze_compressor=true` (compressor + connector frozen and in eval mode; decoder LoRA trains) | stage-2 model, 28k rows (docpages 20k + natural 8k, teacher labels), bs 1 × accum 64, lr 2e-5, bf16, 1,003,520 px, rate 8, spread layout stride 8, seed | see Decision gates | ≤ 14 each (expected less: no compressor backward) + evals |
| R3 (only on a tie) | `expQ/freeze/ft_freeze.sh unfrozen 44` with `… frozen 44` | seed 44, both sides | same | third seed | ~14 + ~14 |

Evals of each finetuned model, judge chained (`eval_mm_4b.sh <model> pisco "" <set>`, on `gpu-be`,
`--exclude=homer-[005-009]`: homer-008 changes image answers, see campaign_probes/CAMPAIGN.md "Same node rule"):
documents `eval_data/mm_docqa_1500_1m`, photos `eval_data/mm_naturalqa_1500`, text `CML=4096 … eval_data/text_mrqa_ood_1500`,
multi-passage text `TOPK=10 … eval_data/text_hotpotqa_1500`. The reference side of every pair is the judged file of
184778 / 184780 on the same set written by the SAME code version (eval answers drifted between the 11-07 and 11-11
code: 55 of 1,500); when that file is missing or older than the last change of `scripts/eval_mm_docqa.py`, re-run the
reference cell first (it is a campaign cell, `--stage eval-ref`).

Rules: every arm is a committed recipe file, never raw overrides on the launch line. Before the first full launch,
diff the frozen smoke's `training_config.yaml` against `expQ/ft_mm_4b_184778/training_config.yaml` and paste the diff
in STATE.md: the ONLY differences allowed are `model_config_overrides.freeze_compressor`, `hydra.run.dir` / output
paths, and `max_steps` for the smoke. A comparison with an unlisted difference is not a result. A relaunch repeats the
original command exactly.

## Decision gates
Metric unit: judge accuracy in [0, 1]; "pt" = .01. Every difference is PAIRED per question
(`scripts/paired_judge.py <frozen json> <reference json>`: bootstrap 95% CI + McNemar), same seed, n = 1,500 per set.
- **Feasibility gate (first, alone):** `expQ/freeze/ft_freeze.sh frozen 42 20` (20 steps), then
  `expQ/freeze/check_module_move.sh <stage-2 model> expQ/freeze/ft_frozen_s42_smoke/model` (CPU, `gpu-be`). PASS =
  the log says `freeze_compressor=True: froze N` and `Compressor trainable parameters: 0`; every `compressor.*` and
  `connector.*` row shows **changed 0.0%**; `decoder.lora_B` shows changed > 0%; the loss at steps 10 AND 20 is within .05 of
  the UNFROZEN 20-step smoke's (`ft_freeze.sh unfrozen 42 20`, same rows, same lr schedule); that smoke's own
  module-move check must show `compressor.*` / `connector.*` changed > 0% (positive control of the check); the config
  diff above is clean. (Amended wake 1, 2026-11-11: the law first compared with the full run 184778, whose lr schedule
  differs — `warmup_steps: .05` = 1 warm-up step of 20 vs 22 of 438, lr at step 10 1.16e-5 vs 8.2e-6. That reading,
  kept: frozen smoke .594 vs 184778 .680, gap .086, confounded, not a failure signal.) Record s/step and peak GPU
  memory; `--time` of the full cells = s/step × 438 steps × 1.25. FAIL on the frozen rows = BLOCKED.md with the output
  (the flag lives in `pisco/model.py`, never patchable here). The smoke may already have been run by the owner's chat:
  look for it in `expmon grid freeze` before relaunching.
- **Mid-run health:** the finetune writes `eval_loss` every 100 steps: `expmon probe` it and compare with the
  reference's curve at the same step (`expQ/ft_mm_4b_184778`). NaN or a loss above the reference's first value for 2
  probes → kill, closed. After each full run, `check_module_move.sh` again: frozen rows must still read 0.0%.
- **Validated (decoder-only is enough):** on documents, photos AND MRQA text, at BOTH seeds, frozen − reference ≥ −2 pt
  and the CI does not lie wholly below 0. Report HotpotQA too (reported, not gating).
- **Failed (the finetune needs the compressor):** on any gating set, ≤ −3 pt with the CI wholly below 0 at both seeds.
- **Tie:** anything else → the third-seed pair R3, then decide on three seeds (2 of 3).
  (Clarified wake 6, 2026-11-11, before the seed-44 result: "2 of 3" replaces "both seeds" in each rule above.
  Validated = every gating set passes at ≥ 2 of 3 seeds; Failed = one gating set fails at ≥ 2 of 3 seeds. Still a
  tie after three seeds → BLOCKED.md for the owner, no fourth seed.)
- Noise band 2 pt: below it nothing is a result. Headline claims in STATE.md and the ledger are prefixed `provisional:`
  until the next wake re-derives them from `expmon grid freeze` and `scripts/paired_judge.py`.
- A NEW metric closes nothing until its row here states its n and a known-positive control it recovers.

**After the gate (the wake proposes, one knob off arm F, gate written here BEFORE the cell, 2 seeds):**
- if validated: what the frozen side makes possible. Candidates: a higher decoder LoRA learning rate (the 2e-5 of the
  recipe is a full-finetune number, training_review.md §4; the compressor can no longer drift); the fp32-masters regime
  (`frozen_base_dtype=bfloat16 torch_dtype=float32 freeze_norm_gain=true`, lr 5e-6) on the decoder alone; a second pass
  over the 28k rows. Each compares with arm F of the same seed, gate ≥ +2 pt with the CI above 0 on documents, no set
  ≤ −2 pt. Each needs its own line in `expQ/freeze/ft_freeze.sh` (a new ARM value; patch rule below).
- if failed: do NOT retry variants of freezing. Write BLOCKED.md proposing the split test (freeze the connector only /
  the compressor only), which needs a new flag in `pisco/`: the owner's chats write it.
- Caching the `<MEM>` vectors on disk (the speed-up of stage B proper) is engineering in `pisco/`: hand it off
  (`expmon handoff add --to pisco "…"`) once the gate validates; it is not an arm.

## Budget and caps
- GPU-days for the whole campaign: **8** (gate 0: 2 finetunes ≤ 28 GPU-h + smoke 1 + evals 8 cells ≈ 8 + reference
  re-runs ≤ 8; a third-seed pair ≈ 30; two follow-up arms × 2 seeds ≈ 70; slack). Over it: BLOCKED.md.
- at most CAMP_MAX_TRAIN trainings in flight; at most CAMP_MAX_SUBMIT_PER_WAKE new submissions per wake (campaign.conf)
- every finetune is 1 GPU on `gpu`; the owner's cap is 8 GPUs on `gpu`, shared with the sibling campaign `longchunk`
  (4 GPUs for its stage 1, 1 per stage 2 / finetune): run `gpucap land a100 1` before each training launch and never
  take the last free slot of the cap. Evals and CPU checks go to `gpu-be` (preemptible).
- chain inside one wake with `--dependency afterok` (evals after each finetune, the module-move check after it too)
- every launch: `--time` from measured s/step × steps × 1.25; `--mem` / `--cpus` as the recipe file states
- no launch below CAMP_DISK_FLOOR_GB free; nothing new after CAMP_UNTIL

## Deliverable — how results reach the owner
One table in STATE.md (and the decision in the ledger via `expmon note`): rows = reference R, frozen F (then any later
arm), last row = the uncompressed decoder; columns = what trains, compression rate (8 tokens per slot; slots per page),
then per set the judge accuracy ↑ at seed 42 / seed 43 and the paired difference to R with its 95% CI and n:
documents, photos, MRQA text, HotpotQA. Every column names its metric and direction. Under it, the module-movement
table of each run (entries changed %, per module) and the training cost (hours, s/step, peak GPU memory) of F against R.
`expmon display --exp freeze` is the live page. The wake proposes in STATE.md the text for `docs/training_plan.md`
(gate 0 result) and HANDOFF.md thread 7; the owner's chats write those files.

## Pre-authorised actions (the agent may do these without asking) and their loop guards
| situation | action | guard |
|---|---|---|
| a campaign job FAILED / NODE_FAIL / TIMEOUT for a transient cause (preemption without resume, node failure, a wall-time misestimate) | relaunch the same recipe, `--time` re-derived | max 2 relaunches per cell, then BLOCKED.md |
| `expmon check` prints `node <n>: N fast deaths` (a poisoned node killing jobs at model load) | relaunch the dead cells with `--exclude=<n>` before the launcher | 1 exclude-relaunch per cell |
| collapse: NaN, or `eval_loss` above the reference's first value for 2 probes | `expmon kill`, `expmon verdict <jid> closed "<why>"` | never relaunch the same cell |
| `starved util=NN%` on a run older than 1 h | `expmon occupancy`; if the recipe is the cause: kill, `expmon optimize` (profile rung), relaunch the best-fit | 1 optimize sweep per arm |
| a grid cell of a listed arm is missing | `expmon launch --exp freeze --stage <S> --parent <jid> …` with the exact command of this law | within CAMP_MAX_SUBMIT_PER_WAKE |
| a job PENDING > 6 h for Resources with an oversized request | `scontrol update JobId=… TimeLimit=…` (downward only) | never touch a job the campaign does not own |
| a job PENDING > 2 h for Resources with `--mem` above the profile rung's MaxRSS × 1.5 while cheaper twins run | `scontrol update JobId=… MinMemoryNode=…` (downward only) | never touch a job the campaign does not own |
| the expmon daemon is gone | `expmon daemon` | — |
| a probe/eval straggler FAILED in seconds with a benign signature (known_issues.md) | ignore, note once | — |
| a reproduced crash in a campaign file whose fix is ≤ CAMP_PATCH_LINES lines inside a "## Patchable" path | apply it, smoke on the cheapest cell (≤ 20 steps), record `<file> — <traceback line> — <lines changed>` in STATE.md; the tick commits it (CAMP_COMMIT_PATHS) | 1 patch per symptom, then BLOCKED.md |
| a job PENDING > 6 h for Resources on `gpu` | resubmit with `--partition gpu-be` (preemptible; a preemption relaunch counts as transient) | 1 move per cell |
| a finetune of a listed arm has written its checkpoint at about a quarter of the run (step ~100 of 438) | PREVIEW: `eval_mm_4b.sh <checkpoint> pisco 300 eval_data/mm_docqa_1500_1m`, judge chained, paired against the reference's checkpoint at the same step and seed when it exists; report as `provisional:` with n = 300 | one preview per finetune cell; a preview never validates or fails the gate |
Anything else: BLOCKED.md and stop.

## Patchable
CAMP_PATCH_LINES=20 (campaign.conf; the conf wins).
- **freely patchable** (≤ 20 lines per symptom): `expQ/freeze/check_module_move.sh`.
- **patchable IF an invariant holds**: `expQ/freeze/ft_freeze.sh` — its SLURM header, paths, env plumbing, logging, or
  ONE new `ARM)` case for an arm whose gate is already written in this law; never an existing arm's training argument.
  Invariant: `bash -n` passes and a 20-step smoke's `training_config.yaml` differs from the frozen arm's only in the
  new arm's documented fields; paste the diff in STATE.md. A recipe a running or finished cell has used is frozen for
  THAT cell.
- **never**: `pisco/`, `test_*.py`, `scripts/eval_mm_docqa.py`, `scripts/judge_*.py`, `scripts/paired_judge.py`,
  `eval_mm_4b.sh`, `ft_mm_4b.sh`, data under `data/`, eval sets under `eval_data/`, anything of the sibling campaigns.
  A crash there is BLOCKED.md with the traceback; the owner's chats fix it.
The WRITE BOUNDARY is not prose: `CAMP_WRITE_ROOTS` in campaign.conf is enforced by the wake hook.

## The wake owns the science
The arms table is a starting hypothesis, not a work order: read the evidence, launch the cells (the owner does not),
propose arms the table lacks (one knob off the control, gate written BEFORE the cell), close decided arms, and amend
this law when the evidence refutes it — a wrong prediction is kept and marked, never quietly deleted. A campaign seeded
deliberately unfinished names the properties the wake may not weaken while fixing (GUIDE.md).

## Hard rules
1. Submissions only via `expmon launch --exp freeze` / `expmon sweep --exp freeze`; never raw sbatch (enforced).
2. Cancel only campaign-owned jobs, only via `expmon kill`. Never `rm` a run dir — rename it `VOID_<name>` (enforced).
3. Never edit campaign.conf, crontab, hooks, `~/.claude`, the tick or the watchdog; never relax a rule here.
   Code edits only as "## Patchable" allows. A `REQUEST` file in this dir is the owner speaking (`expmon campaign
   resolve`): execute it before anything else, then delete it; nothing else may create one. Rules 1–3 are enforced by
   the wake hook: a refusal is a signal to write BLOCKED.md, never to find another spelling. The tick commits for you.
4. Hypothesis ≠ conclusion: write "on track to", "consistent with"; a conclusion needs its gate AND a replicate.
5. Report epochs / rows seen, never bare step counts (steps are incomparable across effective batch).
6. End every wake with one line in NEXT (`done` | `wake-when job:<ids> [max <N>m]` | `wake-in <min>` | `waiting-owner`), or nothing while work is left.
7. Dead ends, do not retry (docs/training_plan.md, HANDOFF.md): a connector that MOVES in the finetune (own connector
   lr, linear connector: docs −.06, photos −.03); fp32 masters at lr 2e-5 (worse; only 5e-6 ties); a trainable norm
   gain under fp32 (docs −.17); query-dependent compression, more finetune data, ViT LoRA, KL at finetune, RAFT (all
   null); a rate changed at inference only. Never change the collator, the data, the rate, the layout or the stage-1 /
   stage-2 recipes: those are other campaigns. Never judge with EM / F1 alone: the judge accuracy decides.
8. Never run python, pixi or a dataset load on the login node: every script goes through `expmon launch` (CPU jobs on
   `gpu-be` with `--gres=gpu:0`, as `run_tests_cpu.sh` does).

## Wake checklist (expmon grid tooling)
1. `expmon grid freeze` — where every cell stands (state, verdict, factors, RESULT / latest probe).
2. `expmon verdict <jid> closed|best "<why>"` for decided arms.
3. Fill missing cells: `expmon sweep --exp freeze --stage <S> --parent <jid> --dry-run <launcher> 'V=a,b'`, then for real.
4. `expmon probe <jid> --step <epoch> <metric>=<val>` for in-flight arms.
5. Rewrite STATE.md; write NEXT.

## Escalation
BLOCKED.md (what, why, options, recommendation) and stop. The watchdog mails herve.dejean@naverlabs.com and writes ALERT for what
the loop cannot fix itself: stale ticks, wakes failing fast (login), no GPU-hour for CAMP_IDLE_H, a rejected
push, unpushed work, low disk, missing cron or daemon.

## Close-out (after CAMP_UNTIL)
`expmon campaign report` (one screen; its Settled / Open block is the successor's brief: `expmon campaign init <label> --from <this dir>`) → `expmon campaign retro` → fill its Lessons → `expmon gripe -c retro "RETRO freeze: WENT OK: … WENT NOT OK: … NET: …"`
(one line; the cross-cluster feedback digest scores campaigns from it) → the OWNER pastes this dir's SYNC.md (written by retro) into ~/.claude/SYNC.md — never the wake (hard rule 3) → `expmon campaign disarm`.
