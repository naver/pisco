# CAMPAIGN longchunk — pisco   (the law; the agent re-reads this every wake; rewrite, never append)

created 2026-11-10 · kind **train** · root `/beegfs/scratch/user/hdejean/pisco` · window until **2027-02-08** (launches allowed through that day 23:59) · owner herve.dejean@naverlabs.com · wakes run `claude-opus-5-5`

DRAFT written by Claude on 2026-11-10 from HANDOFF.md and docs/training_plan.md; the owner reviews every section before `arm`.

## Mission
Does training stage 1 (text pretraining of the compressor + connector + decoder LoRA) with LONG compressor chunks
(1,024 or 2,048 tokens instead of 256) make the compressed model better at reading long passages, without losing on
pages and photos? Background: under the spread `<MEM>` layout, chunk length is already free at inference — one
1,270-token chunk scored .617 against .579 for ten short chunks on the grouped HotpotQA set, with a model trained on
256-token chunks only (ledger 2026-11-06/10). A model TRAINED on long chunks should gain more and makes whole-document
compression possible. The deciding number is the paired judge accuracy (Qwen3-8B rubric judge, `scripts/paired_judge.py`)
of the arm's finetuned model minus the control's, same seed, on `eval_data/text_hotpotqa_1500_g10` (one long chunk per
question, 1,500 Q); the guard numbers are the same difference on documents (`mm_docqa_1500_1m`) and photos
(`mm_naturalqa_1500`). Project invariants: CLAUDE.md "Multimodal PISCO"; regime and history: docs/training_plan.md.

## Arms — what varies, what is held constant
Chain per arm (reduced budget, the same chain as the old reference): stage 1 → stage 2 → finetune (2 seeds) → evals + judge.
**Stage-1 data for every cell of this campaign: `data/msmarco_docs`** (the MS MARCO document collection, 3,213,596 web
pages as `text` = title + body, shuffled once with seed 42, built by job 188358; `data.samples=1000000` takes its first 1M
rows). The recipe's SmolLM2 corpus is too short for long chunks (see Decision gates), so the control is RETRAINED on
MS MARCO with 256-token chunks: the comparison is one variable (chunk length) on one corpus. The old SmolLM2 chain —
stage 1 `expQ/archq3/pt_text_1M_spread` (180486), stage 2 183537, finetunes 183814 / 183816, judged g10 .617 (s42),
g1 .579, documents .599 / .618, photos .671 / .671 — is a REFERENCE row only (corpus differs), never the comparison.

| arm | recipe file (committed) | factors (what varies) | held constant | gate (metric vs whom) | budget (GPU-h) |
|---|---|---|---|---|---|
| C256 (control, to train) | `expQ/longchunk/pt_text_4b_long.sh 1000000 256 512 24 2` | CML 256, DML 512 (the recipe's lengths) | everything below | — (the comparison point) | ~64 stage 1 (4 GPUs × 16 h, as 180486) + 19 stage 2 + 22 finetunes + ~20 evals |
| L1024 | `expQ/longchunk/pt_text_4b_long.sh 1000000 1024 2048 6 8` | CML 1024, DML 2048 (DML = 2 × CML as in the recipe; so each document contributes up to 4 × more tokens) | Qwen3.5-4B/4B, rate 8, spread layout, stride 8, lr 5e-5, MS MARCO 1M documents, effective batch 192 docs, LoRA r64, bf16 | see Decision gates | ~130 + 19 + 22 + 20 |
| L2048 | `expQ/longchunk/pt_text_4b_long.sh 1000000 2048 4096 3 16` | CML 2048, DML 4096 | same | see Decision gates | ~260 + 19 + 22 + 20 |
| reference (exists) | `qwen35_4b_pt.sh 1000000 24 2 ++…mem_layout=spread ++…mem_position_stride=8` on SmolLM2 | corpus | — | not compared; reported as a row | 0 (done) |

Downstream of stage 1, every arm uses EXACTLY the control's commands, only the init model changes:
- stage 2: `pt_mm_4b.sh 100000 8 8 merger <arm stage-1 model> +model_config_overrides.image_max_pixels=1003520 hf_training.learning_rate=3e-5 collator_kwargs.p_text=0.5 collator_kwargs.p_image_ae=0.2 collator_kwargs.p_image_cont=0.05 +collator_kwargs.p_desc=0.15 +collator_kwargs.p_find=0.1 +model_config_overrides.kd_weight=0.5 +collator_kwargs.teacher_inputs=true +collator_kwargs.teacher_max_length=6144` (= job 183537's argv; 256-token text chunks stay: chunk length is a stage-1 variable only in this campaign)
- finetune: `ft_mm_4b.sh <arm stage-2 model> 28000 1 64 hf_training.learning_rate=2e-5 +model_config_overrides.image_max_pixels=1003520` and the same with `+hf_training.seed=43` (= jobs 183814 / 183816; bf16 recipe, NOT the fp32 5e-6 regime, so the comparison to the control is clean)
- evals of each finetuned model, judge chained (`eval_mm_4b.sh <model> pisco …`): g10 `CML=4096 TOPK=10 … eval_data/text_hotpotqa_1500_g10`, g1 `TOPK=10 … eval_data/text_hotpotqa_1500`, documents `eval_data/mm_docqa_1500_1m`, photos `eval_data/mm_naturalqa_1500`, MRQA `CML=4096 … eval_data/text_mrqa_ood_1500` (ledger 2026-11-06 18:00 and 2026-11-10 give the exact lines used for the control; repeat them).

Rules: every arm is a committed recipe file, never raw overrides on the launch line. Before launching two arms
that will be compared, diff their resolved configs (`training_config.yaml` in the run dir) and list EVERY difference in
this table — a comparison with an unlisted difference is not a result. A relaunch repeats the original command exactly.
Known confound to report, not to fix: a longer DML means more text tokens per document, so L1024 / L2048 see more
tokens than the control at the same 1M documents and 5,208 steps. State tokens seen per arm in STATE.md.

## Decision gates
Metric unit: judge accuracy in [0, 1]; "pt" below = .01. Every difference is PAIRED per question
(`scripts/paired_judge.py <arm json> <control json>`: bootstrap 95% CI + McNemar), arm − C256 of the same seed (never
arm − the SmolLM2 reference: the corpus differs).
- **Feasibility gate (first, alone, before any GPU-day):** (a) DATA, DONE 2026-11-10/11: the corpus must have ≥ 30% of
  documents with ≥ 1,024 tokens (else L1024 trains on half-empty chunks) and ≥ 15% with ≥ 2,048 (else L2048 is closed).
  SmolLM2 (job 188339, `expQ/longchunk/data_check_188339.json`): 23.8% / 8.3%, a 1,024-chunk 56% full → FAILED.
  MS MARCO documents (job 188358, `data_check_msmarco_188358.json`, 107k of 3.2M sampled): median 881 tokens, p90 3,589;
  **43.4% ≥ 1,024 (83% of tokens), 20.4% ≥ 2,048 (63%)**; a 1,024-chunk 72% full, a 2,048-chunk 51% → PASSED; adopted
  by the owner (2026-11-10 "as planned"). Documents are truncated to DML by the collator (no packing). (b) a smoke
  of `pt_text_4b_long.sh` at L2048 settings (`expmon launch --exp longchunk --time 1:00:00 … ++hf_training.max_steps=50`,
  its own cell, verdict closed afterwards) must fit in 80 GB per GPU and give s/step; derive `--time` = s/step × 5,208 × 1.25
  for the full cell (C256 takes 16 h on 4 GPUs: 180486). If it does not fit at BS 3, try BS 2 / ACCUM 24 once; else L2048
  is closed with "does not fit" and the campaign runs L1024 only. Launch order: C256 first (it is also the smoke of the
  new corpus), then L1024, then L2048, one stage 1 at a time.
- **Mid-run health (stage 1):** held-out `eval_loss` every 2,000 steps (`expmon probe`) compared with C256's curve at the
  same step (and the SmolLM2 reference `expQ/archq3/pt_text_1M_spread/train.log` for scale); collapse (loss flat at the
  first value, NaN) → kill, closed. Loss values are NOT comparable across chunk lengths (different targets per step);
  only collapse and the final ownership probe are health signals across arms.
  After stage 1: slot ownership ≥ .95 on text (`expQ/archq2/probe_mixer.sh <stage-1 model> 40`, control .977): below
  .90 the arm's slots no longer own their segment — report it as the health number next to every later score.
- **Adopted (the arm is better):** g10 ≥ +3 pt with the CI above 0 at BOTH seeds, AND documents and photos each
  ≥ −2 pt (CI not excluding 0 on the negative side) at both seeds. Then also report g1 and MRQA.
- **Closed (no gain):** g10 CI excludes +2 pt at both seeds, or documents or photos ≤ −3 pt with CI below 0 at both seeds.
- **Tie:** one seed adopts and the other does not → a third seed (`+hf_training.seed=44`) for the arm AND the control
  finetune (the control's stage 2 exists), then decide on three.
- Noise band 2 pt: below it nothing is a result. Headline claims in STATE.md and the ledger are prefixed `provisional:`
  until the next wake re-derives them from `expmon grid longchunk` and `scripts/paired_judge.py`.
- A NEW metric closes nothing until its row here states its n and a known-positive control it recovers.

## Budget and caps
- GPU-days for the whole campaign: **35** (C256 ≈ 5, L1024 ≈ 8, L2048 ≈ 13 GPU-days at the estimates above; a third seed ≈ 3; slack for the smoke and relaunches)
- at most CAMP_MAX_TRAIN trainings in flight; at most CAMP_MAX_SUBMIT_PER_WAKE new submissions per wake (campaign.conf)
- every launch: `--time` from measured s/step × steps × 1.25; `--mem`/`--cpus` as the recipe files state (profiled on the control)
- stage 1 takes 4 GPUs: run ONE stage 1 at a time (the user cap is 8 GPUs on `gpu`); stage 2 / finetune / evals are 1 GPU each
- chain inside one wake with `--dependency afterok` (stage 2 after stage 1, finetunes after stage 2, evals after each finetune)
- no launch below CAMP_DISK_FLOOR_GB free; nothing new after CAMP_UNTIL

## Deliverable — how results reach the owner
One table in STATE.md (and the ledger via `expmon note`), rows = SmolLM2 reference, C256, L1024, L2048; columns = corpus, chunk length (tokens),
compression rate (tokens per slot: 8), slots per 256-token chunk (33) and per chunk at the arm's length, then per set the
judge accuracy ↑ with s42 / s43 and the paired difference to the control with its CI: g10, g1, MRQA, documents, photos. Each row
names its n (1,500 Q per set) and the health number (ownership). The uncompressed decoder's score on the same questions
is the last row (documents .809, photos .717, MRQA .807 — ledger). `expmon display --exp longchunk` is the live page;
`docs/multimodal_results.md` gets the final table (the owner's chats write it; the wake proposes the text in STATE.md).

## Pre-authorised actions (the agent may do these without asking) and their loop guards
| situation | action | guard |
|---|---|---|
| a campaign job FAILED / NODE_FAIL / TIMEOUT for a transient cause (preemption without resume, node failure, a wall-time misestimate) | relaunch the same recipe, `--time` re-derived | max 2 relaunches per cell, then BLOCKED.md |
| `expmon check` prints `node <n>: N fast deaths` | relaunch the dead cells with `--exclude=<n>` | 1 exclude-relaunch per cell |
| collapse: loss flat or NaN for 2 probes | `expmon kill`, `expmon verdict <jid> closed "<why>"` | never relaunch the same cell |
| `starved util=NN%` on a run older than 1 h | `expmon occupancy`; if the recipe is the cause: kill, re-profile, relaunch the best-fit | 1 per arm |
| a grid cell of a listed arm is missing | `expmon launch --exp longchunk --stage <S> --parent <jid> …` with the exact command of this law | within CAMP_MAX_SUBMIT_PER_WAKE |
| a job PENDING > 6 h for Resources on `gpu` | resubmit with `--partition gpu-be` (preemptible; if preempted, relaunch counts as transient) | 1 move per cell |
| the expmon daemon is gone | `expmon daemon` | — |
| an eval / judge straggler FAILED in seconds with a benign signature (known_issues.md) | relaunch once, note | 1 |
Anything else: BLOCKED.md and stop. In particular: no change to the collator, the chunking, the data, the rate, the
layout, the stage-2 or finetune recipes — those are other campaigns.

## Patchable
`none`. CAMP_PATCH_LINES=0: the wake edits no code. A crash in `pisco/` or a launcher is BLOCKED.md with the traceback;
`expQ/longchunk/*.sh` are the campaign's recipe files and are frozen once a cell has used them.

## The wake owns the science
The arms table is a starting hypothesis, not a work order: read the evidence, launch the cells (the owner does not),
close decided arms, and amend this law when the evidence refutes it — a wrong prediction is kept and marked, never
quietly deleted. Arms the wake may propose (gate written BEFORE the cell, within the budget): L512 (CML 512 / DML 1024)
if L1024 adopts and L2048 does not; a DML-only control (CML 256, DML 2048) if the owner asks whether the extra tokens,
not the chunk length, explain a gain. PARKED by the owner (2026-11-10), not for this campaign: a higher image pixel
budget (2M pixels, the image-side long chunk; needs rebuilt page sets and a stage-2 round) — do not propose or launch it.

## Hard rules
1. Submissions only via `expmon launch --exp longchunk` / `expmon sweep --exp longchunk`; never raw sbatch (enforced).
2. Cancel only campaign-owned jobs, only via `expmon kill`. Never `rm` a run dir — rename it `VOID_<name>` (enforced).
3. Never edit campaign.conf, crontab, hooks, `~/.claude`, the tick or the watchdog; never relax a rule here.
   Code edits only as "## Patchable" allows. A `REQUEST` file in this dir is the owner speaking (`expmon campaign
   resolve`): execute it before anything else, then delete it; nothing else may create one. Rules 1–3 are enforced by
   the wake hook: a refusal is a signal to write BLOCKED.md, never to find another spelling. The tick commits for you.
4. Hypothesis ≠ conclusion: write "on track to", "consistent with"; a conclusion needs its gate AND both seeds.
5. Report documents seen and tokens seen, never bare step counts; every accuracy carries n (questions) and the compression rate.
6. End every wake with one line in NEXT (`done` | `wake-when job:<ids> [max <N>m]` | `wake-in <min>` | `waiting-owner`), or nothing while work is left.
7. Dead ends — do not retry (settled, docs/multimodal_results.md and HANDOFF.md): compression rate 16 or 32 (−.07 / −.17
   on documents); a moving or linear connector (never positive); the coarse-to-fine slot order (a permutation, not a
   hierarchy); the tail `<MEM>` layout (collapses on long chunks); the fp32 5e-6 regime inside THIS comparison (it equals
   bf16 but the control is bf16); density-adaptive rate; decoder-side stride alone. Never compute on the login node:
   anything beyond seconds goes through `expmon launch` (stdlib python on small JSON is fine).

## Wake checklist (expmon grid tooling)
1. `expmon grid longchunk` — where every cell stands (state, verdict, factors, RESULT / latest probe).
2. `expmon verdict <jid> closed|best "<why>"` for decided arms.
3. Fill missing cells with the exact commands of this law (`--stage pt1|pt2|ft|eval --parent <jid>`), chained with `--dependency afterok:<jid>`.
4. `expmon probe <jid> --step <step> eval_loss=<val>` for stage 1 in flight (from `train.log`).
5. Rewrite STATE.md; write NEXT (`wake-when job:<ids>` while cells run).

## Escalation
BLOCKED.md (what, why, options, recommendation) and stop. The watchdog mails herve.dejean@naverlabs.com and writes ALERT for what
the loop cannot fix itself: stale ticks, wakes failing fast (login), no GPU-hour for CAMP_IDLE_H, low disk, missing cron or daemon.

## Close-out (after CAMP_UNTIL or when both arms are decided)
`expmon campaign report` → `expmon campaign retro` → fill its Lessons → `expmon gripe -c retro "RETRO longchunk: WENT OK: … WENT NOT OK: … NET: …"`
→ the OWNER pastes this dir's SYNC.md into ~/.claude/SYNC.md — never the wake (hard rule 3) → `expmon campaign disarm`.
