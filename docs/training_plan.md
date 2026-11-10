# Training plan: the next rounds, in order (written 2026-10-30 for the next session)

Restart point for the training work. Everything here was decided with the owner on 2026-10-29/30; the reasons are in
[architecture_review.md](architecture_review.md) → "Architecture levers", [connector_probe.md](connector_probe.md),
[mem_tokens_review.md](mem_tokens_review.md), [training_review.md](training_review.md) and `data_review.md` (data
agent, pending). The ledger (`EXPERIMENTS.md`, `expmon ledger`) has the dated decisions; `HANDOFF.md` is the thread
board. Launch everything with `expmon launch --exp <label> …`, never raw sbatch; never run python on the login node.

## Rules the owner set

1. **One variable per round.** A new idea goes to the "later" list, never into a running or planned arm, even when it
   is a one-line change (2026-10-30: the coarse-to-fine slot order was parked for this reason).
2. **Probe before training.** Design the inference-only version of the question first (edit or replace the component
   on the trained model, paired answer NLL on held-out rows, `expQ/archq2/probe_connector.py` is the pattern).
3. **Two seeds per training arm** (`+hf_training.seed=43`), paired judge on the 1,500-question sets against the
   same-seed control; a training effect under ~.03 on documents is noise otherwise.
4. **Every results table names its metric in the column headers**, next to the control on the same questions.
5. **Shake out before launching**: unit tests (`run_tests_cpu.sh`), `mm_shakeout.sh` with `--config_override`, a
   20-step finetune when the change touches training, and a weight-movement check (`expQ/archq2/check_module_move.sh
   <init model> <trained model>`): a run whose target weights did not move is a null test, not a result (arm 1 of
   archq2 was one).

## Where we stand (facts to rely on)

| fact | number | source |
|---|---|---|
| best model, documents / photos (judge, 1,500 Q each) | .599 / .685 (174908, norm on) vs uncompressed .809 / .717 | multimodal_results, ledger |
| route-then-expand at inference | .810 on documents (routing .98) | archq |
| the connector is a linear map; finetunes do not train it | refit linear = MLP within .006 NLL; connector moves 0.6% per finetune | connector_probe |
| each `<MEM>` encodes its place; late `<MEM>`s of a chunk repeat early ones | R² .92-.97; half the vectors cost +.17 docs / 0 photos | connector_probe |
| bf16 trainable weights: entries above ~.005 never move at lr 2e-5 | 80% connector / 84% LoRA-A / 35% LoRA-B never change | ledger 2026-10-23, training_review §4 |
| fp32 masters at the same lr 2e-5 are worse | docs −.043; with trainable norm gain −.167 | 176295 / 176296 |
| RL headroom | pass@8 .698 vs greedy .510 on 500 Q; RAFT (positive half only, one round) null | mmrl 173667 / 173710 |
| seed spread | .029 on documents between seeds 42 and 43 of the same recipe | 173550 vs 176243 |

## Round 0: state 2026-11-10 (check with `expmon status`, `expmon grid archq3`)

**archq3 verdict (2026-11-03): the spread `<MEM>` layout wins** (judge, 1,500 Q, paired vs the tail chain at equal
budget: documents +.093 s42 / +.075 s43, photos +.023 / +.021; table in `architecture_review.md`). **Full-budget
spread chain done (2026-11-10):** 5M text (184769) → one stage-2 pass 160k (184777) → finetunes 184778 (s42) / 184780
(s43) → documents .610 / .615, photos .677 / .685 = production 174908 / 176242 within noise (documents +.011 / +.024,
photos −.009 / +.010) with one stage-2 pass instead of three. **184778 / 184780 are the reference models from now on.**
The fp32 5e-6 regime (round 1) can now go into `finetune_mm.yaml` (nothing bf16 is pending).
**Rate round closed (2026-11-10):** trained rate 16 −.07 documents / −.03 photos, rate 32 −.17 to −.20 / −.09 vs rate 8
(table in `multimodal_results.md`); rate 32 slots no longer own a segment (.547). Rate 8 stays; 16 only where tokens
matter on photos. **Long chunks (probe 2026-11-10):** with the spread layout one 1,270-token chunk beats ten 130-token
chunks (+.037) at inference with no training, while the tail layout loses .17: the next stage-1 arm is **long chunks
(compressor_max_length 1,024-2,048)**. **archq3b (coarse order) parked** (round 4). **Literature-scan probes:**
attention-mass page picking beats the digit prompt on images (10 pages .885 vs .861) and loses on text (.873 vs .922);
per-head scales (SEAL) null; slot dropping is cheaper under tail than spread (no free coarse level either way).
**SLURM notes:** `expmon launch --parent` records lineage only; pass `--dependency afterany:<parent>` at submission.
4-GPU jobs on gpu-be with checkpoints 6 h apart do not survive preemption (rate-32 stage 1 lost a day); use
`--partition gpu,gpu-be` and keep the 8-GPU cap in mind.

Recipe of a reduced-budget chain (what every archq3 chain used):

| step | command |
|---|---|
| stage 2 (100k, v4kd recipe, 1M px, KD) | `PT_DATA="…pdfa_1m_desc:50000,…wiki_1m_desc:19000,…natural_desc:31000" PT_OUT=expQ/archq3/pt_mm_<arm> expmon launch --exp archq3 --stage pt --parent <stage1 job> pt_mm_4b.sh 100000 8 8 merger expQ/archq3/pt_text_1M_<arm>/model +model_config_overrides.image_max_pixels=1003520 hf_training.learning_rate=3e-5 collator_kwargs.p_text=0.5 collator_kwargs.p_image_ae=0.2 collator_kwargs.p_image_cont=0.05 +collator_kwargs.p_desc=0.15 +collator_kwargs.p_find=0.1 +model_config_overrides.kd_weight=0.5 +collator_kwargs.teacher_inputs=true +collator_kwargs.teacher_max_length=6144` |
| finetune, 2 seeds | `FT_DATA=<two-set spec> ft_mm_4b.sh <stage2 model> 28000 1 64 hf_training.learning_rate=2e-5 +model_config_overrides.image_max_pixels=1003520 [+hf_training.seed=43]` |
| evals | `PX=1003520 eval_mm_4b.sh <model> pisco "" eval_data/mm_docqa_1500_1m` and `eval_mm_4b.sh <model> pisco "" eval_data/mm_naturalqa_1500`; pair with `scripts/paired_judge.py <reference.json> <arm.json>` (prints reference − arm) |

The two-set finetune data spec: `/beegfs/scratch/user/hdejean/pisco/data/mm_ft_docpages_20k_1m_teacher:20000,/beegfs/scratch/user/hdejean/pisco/data/mm_ft_natural_8k_teacher:8000`.

## Standing proposal: two stages by what is frozen (stage A representation, stage B usage)

Discussed with the owner on 2026-10-28, recorded here on 2026-10-31. Not a round yet: it is the shape the rounds
below should converge to, and it has one cheap gate.

**Why the current split is drawn in the wrong place.** Stage 1 (text), stage 2 (multimodal) and the QA finetune all
train everything. The finetune changes little and cannot add information (facts above; training_review §1.6), yet
it moves the compressor, so every finetune implies its own index: a collection compressed for one use cannot be
reused for another.

| | **stage A, representation** | **stage B, usage** |
|---|---|---|
| trains | compressor, connector and decoder, once | decoder only (LoRA); compressor and connector frozen, `<MEM>`s computed once and cached |
| data | text, pages, photos, frames together; all tasks (AE, continuation, describe, find, QA, KD) | long contexts built from stored vectors: many documents, two-level hierarchies, mixed raw + `<MEM>`, timestamps, order, routing twice |
| gives | one **generic compressor** = one index per collection | one generic reader by default; **one small adapter per usage** (interval output for grounding, robot memory) on the same frozen index when the output format or domain differs |
| cost | one long chain (stage 1 + 2 + QA merged) | cheap: no compressor pass, so hour-long rows are affordable |

Known cost of "generic": domains compete at a fixed budget (pages + nav frames in one stage 2: docs −.03, nav −.055
against separate models). Pay for breadth with budget (more `<MEM>`s per item at the fine level), not with
specialisation; keep query-dependent compression only where recompression is cheap anyway (the few picked items).
In this repo "frozen compressor" means the compressor adapter **and** the connector (both define the stored vectors);
original PISCO had one backbone and no connector, so there it meant the compressor adapter alone.

**Gate 0 (documents, under one GPU-day, not run yet).** Freeze the compressor adapter and the connector of
`pt_mm_v4kd_4b`, run the 173550 finetune recipe with the decoder LoRA alone on the same 28k rows, 2 seeds, paired
judge against 174908 / 176242 on the 1,500-Q document and photo sets. Equal within noise = stage B as decoder-only is
validated and the stored index can be shared across finetunes; worse = the finetune still needs the compressor, and
the proposal waits for a stage A that already covers QA. Missing code: a `freeze_compressor` flag in `PISCOConfig`
(compressor adapter + connector `requires_grad=False`; `freeze_decoder` exists as the pattern), plus
`check_module_move.sh` on the result to confirm nothing but the decoder LoRA moved. Caching the `<MEM>`s is an
optimisation for stage B proper, not needed for the gate.

**Where the rounds stand against it.** Round 2 (decoder-only RL on cached `<MEM>`s) is the first stage-B training.
The video reader test (`scripts/video/train_reader.py`, cache 180520 done, training 180522 pending) is stage B on
video. archq3 and round 4 are stage-A changes. If gate 0 passes, the "find v2" routing rows and the long-context
rows of round 4 move from the finetune into stage B, on cached vectors.

**Warning (2026-11-01).** The linear connector + `connector_lr=5e-5` pair came out worse (docs −.063 at seed 42,
mostly multi-page documents −.126; −.012 at seed 43; photos −.02). The inference probe had shown linear = MLP, so the
suspect is the connector *moving* during the QA finetune, the same sign as fp32 masters at lr 2e-5. Disentangling
arms at seed 42 (judged, 1,500 Q, vs the norm run): linear connector alone docs −.013 / photos −.030; connector lr
alone docs +.002 / photos −.028; the combination had been −.063 / −.024, so the two parts do not add up on documents
(an interaction or an unlucky run; seed 43 arms 183565 / 183566 pending). Decision: keep the MLP, keep the connector
effectively frozen in finetunes (moving it buys nothing and costs ~.03 on photos); rounds 2 and 3 train the decoder
only. The precision finetunes at lr 5e-6 unfreeze the connector but move it little, and matched bf16.

## Round 1: the precision and learning-rate regime (label `precision`; training_review §4, rank 2)

**Why first.** In bf16 the finetune is a magnitude-gated sparse update: only small entries learn. Any arm that needs
the connector, LoRA-A or small RL updates (~1e-5) to land is a null test under it. The regime is the gate for rounds 2
to 4.

**Readout of the 20-step grid.** For each arm, per module: fraction of entries changed and relative move
(`check_module_move.sh`), and the 20-step loss from the job log (`grep "'loss'" ft_mm_4b_<jid>.out`). Accept the
fp32 learning rate that (a) keeps the 20-step loss within ~.03 of the bf16 control (.63 at step 10 for 174908) and
(b) moves connector and LoRA-A by 0.5-2% (bf16 moves them 0.03% and ~0). Expected winner 5e-6 or 1e-5.

**Grid result (2026-11-01, jobs 180560-180564; movement = relative Frobenius change over 19 steps):**

| regime | entries changed: connector / LoRA-A / LoRA-B | moved: connector / LoRA-A / LoRA-B | loss at step 10 |
|---|---|---|---|
| bf16, lr 2e-5 (control) | 29% / 30% / 74% | .0027 / .0028 / .0207 | .600 |
| fp32 masters, lr 2e-6 | 100% / 83% / 83% | .0007 / .0008 / .0034 | .639 |
| fp32 masters, lr 5e-6 | 100% / 83% / 83% | .0016 / .0016 / .0069 | .597 |
| fp32 masters, lr 1e-5 | 100% / 83% / 83% | .0028 / .0029 / .0123 | .628 |
| fp32 masters, lr 2e-5 | 100% / 83% / 83% | .0052 / .0055 / .0236 | .662 |

fp32 removes the magnitude gating (every entry moves) but the total movement is set by lr × steps, so no setting
reaches "0.5-2% in 19 steps"; 5e-6 matches the control's loss with all entries moving, 1e-5 matches the control's
total movement. Both go to full finetunes.

**Settled (2026-11-02):** fp32 masters + frozen gain at **lr 5e-6** equals bf16 at both seeds (docs −.009 / −.009,
photos −.005 / −.006 judge accuracy, 1,500 Q, CIs include 0) while every entry moves; lr 1e-5 is slightly worse
(docs −.019 at s42). Regime for rounds 2-4: `+model_config_overrides.frozen_base_dtype=bfloat16
+model_config_overrides.torch_dtype=float32 +model_config_overrides.freeze_norm_gain=true hf_training.learning_rate=5e-6`
(the connector still moves little at this lr; a connector that moves a lot hurts, see the warning above). Not yet the
`finetune_mm.yaml` default: the full-budget spread finetunes 184778 / 184780 must keep the bf16 recipe of 174908 (one
variable) and read the config when they start (~2026-11-08); write it in after that.

**(was) Then** two learning rates × two seeds as full finetunes (same `ft_mm_4b.sh` recipe with the three
`model_config_overrides` above), evals, paired against 174908 / 176242. Adopt the regime if it matches bf16 within
noise at both seeds; it is allowed to be merely equal, its job is to make later arms testable. Record the per-module
movement of the full runs too.

**Not available**: stochastic rounding (torchao is not in the pixi env). If fp32 masters fail at every lr, add
torchao (`bf16_stochastic_round=True` in its AdamW) as the next arm.

## Round 2: decoder-only RL on cached `<MEM>`s (training_review rank 1; label `mmrl`)

This is the first stage-B training of the standing proposal above (frozen compressor and connector, cached vectors).

**Why.** The decoder has the answer in the vectors and does not use it (P2); RAFT's positive-only update was null;
the papers that closed this gap used on-policy RL with a verifiable reward (ACL 2026 "Memorization-Utilization
Gap": SFT 75.6 → RL 82.1 vs 83.1 full context; DEX-Comp 2609.05152: GRPO on teacher-failed rows, 72.0 vs 69.5 full
context). In plain terms GRPO here is weighted maximum likelihood on the model's own samples with the group mean
as baseline: correct samples up, confident wrong samples down, redrawn every step.

**Design.**
- Policy = decoder LoRA only; compressor, connector and the `<MEM>` vectors are frozen and **cached once per row**
  (compute them with `PISCO.compress`, store per row). Regime = round 1's (fp32 masters), lr ~1e-5 for LoRA
  (DEX-Comp / ACL-682 use 1e-6 near-full finetuning).
- Rows: the 20k + 8k finetune rows (teacher labels kept as reference, dataset gold in `answers`); train on rows
  where greedy is wrong and at least one of G=8 samples is right (DAPO-style dynamic filtering drops all-right /
  all-wrong groups; `data/mmrl_sminus_*` already hold the 6.4k teacher-failed rows and probe outputs).
- Sampling: G=8, temperature 1.0, top_p .95, top_k 0 (the probe-A settings of `scripts/raft_rollout.py`); max 32
  new tokens for the brief format, 256 for evidence-then-answer.
- Reward: 1 if `pisco.reward.is_correct` (containment EM / F1 against gold **and** teacher label) else 0; refusals
  (`is_refusal`) get 0; optional judge reward (Qwen3-8B rubric) on a held-out check only, it is too slow in the loop.
- Loss: GRPO with group-mean baseline, PPO clip, KL penalty to the SFT checkpoint (β ~.04), no critic.
- Two formats as two arms: brief answer (current prompt) and evidence-then-answer (`EXPAND_SYSTEM_PROMPT` in
  `scripts/eval_mm_docqa.py`, the prompt alone gave +.025).
- **Controls**: the SFT checkpoint it starts from, and an SFT run on exactly the same rows at the same regime (tells
  whether the negative half is what was missing). Two seeds for the RL arm.
- Eval: the usual 1,500-Q sets with the judge, paired; also pass@8 / greedy gap after training (should shrink).

**What exists / what is missing.** Exists: rollout and reward code (`scripts/raft_rollout.py`, `pisco/reward.py`,
`RolloutCollator`), the teacher-failed splits, judge sampling in the eval (`--num_samples`). Missing: a GRPO trainer.
TRL is not in the pixi env; options are adding `trl` (its GRPOTrainer wants a generation path we do not have for
prompt embeddings) or a small in-repo loop (≈200 lines: sample with `PISCO.generate(num_return_sequences=G)`,
score, compute advantages, one PPO-clip step with cached vectors). The in-repo loop is the recommended route.
Shake-out: 20 steps on 256 rows, check the policy LoRA moves and the reward mean rises on the training rows.

**Closed by probe (2026-11-02):** a density-adaptive rate (more `<MEM>`s on dense pages, mem_tokens_review rank 5).
Rate 4 vs rate 8 by gold-page density on 360 questions, both seeds: dense third −.075 / −.058, sparse third +.008 /
−.017. Twice the vectors make dense pages worse, not better.

## Round 3: the `<MEM>` mixer — premise not supported (probes 183615 / 183616, 2026-11-02)

The late slots of a chunk are not linear copies of the early ones (held-out R² .01 to .05), slots already lean toward
their own segment under the tail layout (text ownership .754), and the connector must stay frozen in finetunes. What
the probes showed: the late slots are the least used (keeping the first 75% costs ~0 in NLL on tens of rows), the
first slot of a photo is a summary slot (+.085 when dropped), and the shared top direction is an attention sink.
**At judged scale the 75% cut is not free**: on 1,500 Q it costs −.023 documents / −.024 photos (183652/183653), so
no serving action; the NLL probe on tens of rows bounds an effect at about .02 accuracy, it does not prove zero. The
mixer is parked; if revived it is a stage-2 component.

### (parked) the `<MEM>` mixer (mem_tokens_review rank 1)

One chunk-local bidirectional attention block over a page's `<MEM>` vectors inside the connector, initialised from
the decoder's layer 0 with zero-initialised output projections (exact retrofit), trained with `connector_lr` under
round 1's regime. Targets the redundancy of the late `<MEM>`s. Prerequisite: 180488/180489 show the connector learning
rate is safe. Premise probes first (specified in mem_tokens_review §4): can late vectors be predicted from early
ones; the first-k curve. Two seeds vs 174908 / 176242. Expected +.01 to +.03 on documents, medium confidence.

## Round 4: stage-2 upgrades (gate passed 2026-11-03: the layout chain beats its baseline chain)

The spread layout is the stage-1 base from now on. The **coarse-to-fine slot order** (archq3b, `spread_coarse`) is
**parked 2026-11-04**: it only permutes equal-width segment centres, the first kept slot sits at the chunk start (not
the middle), and a position id cannot give a slot a wider span; its stage-1 ownership is weaker than spread (.927 vs
.977, late←early R² .109 vs .013). A real coarse level needs levels (1, 2, 4, 8 slots at span midpoints), nested-budget
training (lower levels hidden from the decoder at random) and probably a level token: a training round, not a layout.
Probe done (2026-11-10): keeping the first 50% of slots costs .12 documents under spread but .04 under tail (tail slots
are redundant), 25% costs .26 vs .08; so a trained hierarchy must beat the tail's cheap degradation. Next stage-1
variables, ranked: (a) **long chunks** (compressor_max_length 1,024-2,048; the inference probe already gives +.037
for one chunk vs ten, see round 0); (b) **multi-position slots** (PPE, literature scan rank 1; a training arm, since an
inference probe on one-id slots only measures the shift); (c) the **per-slot ownership loss** (mem_tokens_review rank 2). Group pooling is dropped (it was the fallback for a negative verdict). Data and labels ([data_review.md](data_review.md), ranked): (1) **routing rows on real questions** in the finetune
("find v2": 8-20 documents, same-document + ColQwen2 negatives, 1-2 gold, pages + text + frames, the eval's
`ROUTE_QUESTION` prompt; target pick@1 ≥ .85 at 11-20 pages vs .77-.81 now; control = the same number of extra QA
rows); (2) **evidence-then-answer teacher labels** as a second mode next to brief (cheap check first: does the
uncompressed teacher itself gain in evidence mode?); (3) **counterfactual rows** (gold page swapped; zero-cost probe
first: swap the gold page on the 1,500 Q and count unchanged answers); (4) dense synthetic pages + region tasks in
stage 2 (PixMo-Docs, CoSyn); (5) best-of-k judge-verified labels. Two findings to act on before any of these:
**three stage-2 collator cuts bias toward the top of the page** (AE target cut at 768 tokens, continuation reads
tokens 0-32, `find` snippets from the first 300 words), which matches the measured recall gradient (top fifth .87,
bottom .75). Size of the bias (job 180620): 26% of PDF pages exceed the 768-token AE target (17% of all PDF text is
never an AE target), 50% exceed the continuation reach, 56% exceed the 300-word `find` window; wiki pages: AE covers
99% but continuation / `find` miss 69% / 77%; photos unaffected. Fix = random-start windows (AE, continuation) and
`find` snippets from anywhere, one stage-2 variable, readout = transcription recall by page fifth; and a **dedup
check**: MP-DocVQA eval documents may share pages with the DocVQA train pages in our
finetune set, so hash all pages before the next data round (SlideVQA decks are evaluation-only by licence; Doc-750K
is to be avoided). Independently of the layout: **stage-2 distillation upgrade**
(KL on all tasks, teacher-reliability gating, reconstruction share 20-50%; training_review rank 3), and the data /
label changes from `data_review.md` once the data agent's review is in (labels: the text RAG set carries Mistral-7B
labels and nav carries synthetic gold; both would be relabelled by our decoder with `scripts/relabel_mm_teacher.py`
before any use in a finetune).

## Serving wins that need no training

- Drop every second `<MEM>` of a photo at inference: no measurable loss (connector_probe), 2× fewer vectors.
- Store `<MEM>`s in int8 (archq): free, half the storage.
- Route-then-expand for documents (archq): .810 vs .599, the document mode.

## Closed (do not reopen without new evidence)

Multi-layer read-out as a finetune arm (null test: connector untrained); decoder-side `<MEM>` positions (flat,
vectors already encode their place); deeper / wider connectors, resamplers, Q-Former; distinct `<MEM_i>` embeddings;
rate 4; changing the rate at inference; query-dependent compression at finetune; RAFT positive-only; KL at finetune;
more of the same finetune data; fp32 masters at lr 2e-5 (too high once updates land); a trainable norm gain in fp32.
