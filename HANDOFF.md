# HANDOFF — PISCO (multimodal)

_Updated 2026-11-02 evening (NLE), branch `arc`. Restart point: this file is the **one board** of every open
thread. Findings live in the docs, not here: accuracy **`docs/multimodal_results.md`**; design review, route-then-expand
and its hard test **`docs/architecture_review.md`**; connector probe **`docs/connector_probe.md`**; video
**`docs/hierarchical_video.md`**; speed **`docs/latency.md`**. Science record `EXPERIMENTS.md` (`expmon ledger`).
Invariants `CLAUDE.md` → "Multimodal PISCO"._

## Goal
Multimodal PISCO: compress images (document pages, photos, charts, video frames) and text into `<MEM>` vectors so a
decoder answers without seeing pixels, as close as possible to the uncompressed Qwen3.5-4B and much faster.

## Where we stand (one paragraph)
Best model `expQ/ft_mm_4b_174908/model` (173550 recipe + connector norm). All-`<MEM>` on 1,500 Q: documents .599 vs
uncompressed .809, photos .685 vs .717, nav .806 vs .792. **Route-then-expand** (the model picks the answer page from
the `<MEM>`s, that page goes raw) reaches the uncompressed score on 4-10 candidate pages and on text at 1/3 of the
tokens; it loses only where the pick misses (whole documents of 11-20 pages). The projector is settled: a linear
connector is not better and a connector that moves in the QA finetune costs photos; keep the MLP, keep it effectively
frozen. The training regime is settled: fp32 master weights + frozen norm gain at lr 5e-6 equals bf16 at both seeds
while every weight can move (the gate for RL and connector rounds). **The `<MEM>` layout verdict is in (2026-11-03): slots placed at the centre of the segment they compress (archq3,
trained from stage 1) beat the tail layout at equal budget by +.09 / +.08 on documents (seeds 42 / 43) and +.02 on
photos, and match the full-budget production model with 62% of its stage-2 data.** The spread layout is the stage-1
base from now on; the full-budget spread chain is training (184769 → 184785, ETA ~2026-11-09). The coarse-to-fine slot
order (archq3b) is parked: it is a permutation, not a hierarchy (first kept slot at the chunk start; weaker ownership). The standing
proposal for the pipeline is two stages by what is frozen: stage A (representation, generic compressor) and stage B
(decoder-only on cached vectors, adapters per usage); its gate 0 is not run yet (`docs/training_plan.md`).

## Thread board

| # | thread | state | result so far | next action (who) |
|---|---|---|---|---|
| 1 | **archq3: spread `<MEM>` layout** (`compressor_mem_layout=spread mem_position_stride=8`, trained from stage 1) vs a tail-layout chain at equal reduced budget (1M text → stage 2 v4kd 100k → finetune 28k); MLP connector in both, one variable | **WON (2026-11-03)**, complete at 2 seeds on both sets | judge 1,500 Q paired, spread − tail chain: docs s42 .599 vs .507 = **+.093** (CI +.069..+.115), s43 .618 vs .543 = **+.075** (CI +.051..+.099), every source gains; photos s42 .671 vs .648 = +.023 (CI +.002..+.043), s43 .671 vs .649 = +.021 (CI +.002..+.039). Vs the full-budget norm model: docs .000 / +.027, photos −.015 / −.004. Before the finetune it already had lower answer loss on every modality and every slot owned a segment (.977 vs .799) | table in `architecture_review.md`; **full-budget spread chain** (owner's go 2026-11-03; relaunched 2026-11-04 00:50 after the BeeGFS outage killed 184709-184717): stage 1 184769 (5M text, 4 GPUs, 143228 recipe + `compressor_mem_layout=spread mem_position_stride=8` → `expQ/archq3/pt_text_5M_spread`, ~79 h) → stage 2 184777 (pt_mm_v4kd_4b recipe, 160k, → `expQ/archq3/pt_mm_5M_spread`, ~30 h) → finetunes 184778 (s42) / 184780 (s43) (174908 recipe, bf16) → evals 184781 / 184783 (s42 docs / photos), 184784 / 184785 (s43); references 174908 / 176242; ETA ~2026-11-09 (Claude watches; dead dirs `pt_text_5M_spread_dead184709`, `pt_mm_5M_spread_dead184770`) |
| 1b | **archq3b: coarse-to-fine slot order** on the spread layout (`compressor_mem_layout=spread_coarse`; one variable vs 1) | **parked 2026-11-04** (owner's design question); stage 1 184175 done, stage 2 184706 died in the outage, not relaunched | ownership probe (184704 / 184705, stage-1 text): coarse .927 vs spread .977, late←early R² .109 vs .013. The bit-reversal order puts the first kept slot at the chunk START (n = 33: start, end, middle, quarters), not the middle as the docstring claims; a position id is a point, not a span, so no order yields a true coarse level | if a hierarchy is wanted: levels 1/2/4/8 at span midpoints + nested-budget training + level token (a training round); probes first: keep-first-k at inference and decoder-side slot permutation on the spread finetune (Claude, on go) |
| 1c | **connector form and movement** (archq3 label): linear connector; `connector_lr=5e-5` | **settled, 2 seeds** | judged 1,500 Q vs same-seed norm (arm − reference): linear + lr (180488/9) docs −.063 / −.012, photos −.024 / −.015; linear alone (183563/5) docs −.013 / +.013, photos −.030 / −.001; MLP + lr (183564/6) docs +.002 / **−.055**, photos −.028 / −.016. A moving connector is never positive and hurt badly at s43 | **keep the MLP, keep the connector effectively frozen in finetunes**; `training_plan.md` rounds 2-3 and `connector_probe.md` carry the caveat (a paired-NLL probe on tens of rows bounds an effect at ~.02-.03) |
| 1d | **precision regime** (`training_plan.md` round 1; label `precision`) | **settled, adopted**: fp32 masters + frozen norm gain, lr 5e-6 (183548 / 183549) = bf16 norm within noise at both seeds (docs −.009 / −.009, photos −.005 / −.006) while 100% of connector / 83% of LoRA-A entries move (29% / 30% in bf16); lr 1e-5 slightly worse on photos at both seeds (−.009 / −.022) | overrides `+model_config_overrides.frozen_base_dtype=bfloat16 +model_config_overrides.torch_dtype=float32 +model_config_overrides.freeze_norm_gain=true hf_training.learning_rate=5e-6` | write into `finetune_mm.yaml` after the full-budget finetunes 184778 / 184780 have started (~2026-11-08; they must keep the bf16 recipe of 174908) (Claude); regime for RL (round 2) |
| 1e | **rate round** (`compr_rate` 16 / 32 on the spread layout, `mem_position_stride` = rate; one variable vs the spread rate-8 chain 183814 / 183816 at equal reduced budget) | **rate 16 done (2026-11-06)**; rate 32: stage 1 184815 done, stage 2 184816 waiting on the 8-GPU cap (allowed on gpu,gpu-be), then 184817 / 184818 → evals 184820-184823, ~2026-11-08; outputs `expQ/rate/` | judge 1,500 Q paired, rate 16 − spread rate 8 (s42 / s43): docs .535 / .544 vs .599 / .618 = **−.065 / −.074** (CIs exclude 0), photos .643 / .642 vs .671 / .671 = −.027 / −.029; vs the tail rate-8 chain: docs +.028 / +.001, photos n.s. → spread at rate 16 = tail at rate 8 at half the tokens. Halving the budget costs ~.07 docs / ~.03 photos: the image rate curve is not flat. Untrained halving had cost .13 / .04-.09 | read rate 32 (~2026-11-08) then pick the operating point (owner): rate 8 for accuracy, 16 where tokens matter, 32 as a routing level if it holds |
| 2 | **inference probes on the slots** (no training; `mem_tokens_review.md`, `connector_probe.md`) | **done** | connector = linear map at inference on every backbone pair; each slot encodes its place; mixer premise: late slots are NOT copies of early ones (R² .01-.05), so the `<MEM>` mixer loses its premise; density: rate 4 does not help dense pages (worse at both seeds) → density-adaptive rate closed; **budget curve at judged scale (174908, 1,500 Q):** keep first 75% of slots docs −.023 / photos −.024; 50% −.128 / −.087 (every 2nd −.130 / −.043); 25% −.283 / −.233. Untrained 16× costs .13 on docs; 32× destructive. The "free 25% cut" from the NLL probe was **retracted** at judged scale | nothing; a trained-rate-16 image chain is the next step only if 16× is wanted |
| 3 | **route-then-expand / retrieve-first** | **done**, decision-ready (`architecture_review.md` → 1, 1b) | route2 = uncompressed on retrieved pages and text; top-2 raw beats raw at 1/5 tokens; long documents lose on picking; coarse untrained `<MEM>`s pick badly | training lever = routing rows on real questions ("find v2", `data_review.md` 5.1) or stage-B RL (round 2); owner's call after Tuesday |
| 4 | **video** (ExtremeWhenBench on LVBench, `scripts/video/`) | zero-shot round **done** (`hierarchical_video.md` → 1); stage-B reader test: `<MEM>`s of all 65,964 frames cached (180520), `train_reader.py` ran a 4-step smoke only (180522) | 10 windows: model ≈ CLIP; whole hour: coarse level < CLIP, CLIP top-3 + zoom .498 vs .446; MCQ: pick is the bottleneck (.437 vs .492 with gold); ASR null; blind = chance | owner decision: run the reader training on the cached frames (3 folds, `scripts/video/train_reader.py`) or park video until archq3b gives a trained coarse level. Small fix: `mcq_levels.py summarize` crashes on blind runs (result file is written) |
| 5 | **OpenEQA** (external nav test; regime "OpenEQA EM-EQA HM3D 557 Q, memory only") | **done** incl. whole tour (`multimodal_results.md` → "External navigation test") | LLM-Match (Qwen3-14B): generic PISCO 174908 at 16 / 32 / 64 / all frames **48.2 / 50.9 / 49.6 / 51.9** vs uncompressed 16 frames 50.0, 32 frames 57.0; retrieval (campaign, 15 frames) 54.0 (2B) / 59.3 (4B); nav PISCO `ft_nav_4b` 35-38 (templated finetune → 1-word answers). The whole tour compressed (≈ 4,700-11,600 `<MEM>`) edges 16 raw frames (+2.0, CI −1.7..+5.8) at fewer tokens, but the k curve is nearly flat: the decoder does not exploit a long compressed memory | next lever = stage B: decoder trained to read 32-all compressed frames, or route-then-expand over the tour; not before Tuesday (owner) |
| 6 | judge human check | waits on the user | 114 rows in `expQ/judge_study/human_sheet.md` | label, then `scripts/judge_agreement.py … --human` (owner) |
| 7 | **stage A / stage B proposal** (`training_plan.md` → "Standing proposal") | written 2026-10-31; gate 0 **not run** | stage A = stage 1 + 2 + QA merged, generic compressor; stage B = decoder-only on cached `<MEM>`s, adapters per usage | gate 0 needs a `freeze_compressor` flag (compressor adapter + connector), then a decoder-only finetune of `pt_mm_v4kd_4b` under the 1d regime, 2 seeds vs 174908 / 176242 (Claude, on go) |
| 8 | repo hygiene | open | 9 commits ahead of `origin/arc` (github.com/naver/pisco, public); 27 files uncommitted (model.py, eval, docs, video/nav scripts, 5 review docs); `*.out` logs pile at the repo root; scratch at 89% | commit the archq2/archq3 code + docs; push is the owner's call (public remote); archive logs older than a week to `expQ/logs/` (Claude, on request) |

Jobs by thread: `expmon status`; problems: `expmon check`. Rounds and their order: `docs/training_plan.md`.

## Decisions pending (owner)
1. (decided 2026-11-03: full-budget spread chain, relaunched 2026-11-04 as 184769 → 184785; archq3b parked
   2026-11-04.) Do we want a real slot hierarchy (levels 1/2/4/8 + nested-budget training, `training_plan.md` round 4)?
   Thread 2's budget curve says untrained slot dropping costs .28 at 25%, so only a trained level can pay; probes first.
2. **Which stage-B training first** (both need the 1d regime; `training_plan.md` round 2): decoder-only GRPO on cached
   `<MEM>`s with an evidence-then-answer format (the "use" failures, OpenEQA's flat k curve), or routing rows on real
   questions ("find v2", the long-document pick). Recommendation: gate 0 of the stage A/B proposal first (one GPU-day),
   because it tells whether a decoder-only stage can carry either.
3. **Video**: reader training on the cached frames now, or wait for the trained coarse level (archq3b).
4. **Push** the `arc` branch to the public remote or not; SlideVQA whole decks need the NTT licence; OpenDocVQA corpus
   is gated.

## Key files & commands
- Launch only via `expmon launch --exp <label> …` (labels `mmpisco`, `archq`, `archq2`, `openeqa`, `video`).
  Partition `gpu` caps 24 queued / 8 running per user; `gpu-be` is preemptible and can sit at zero priority — move a
  stuck job to `-p gpu --constraint gpu_a100`. `expmon launch --parent` records lineage only (NO SLURM dependency): add `scontrol update JobId=<child> Dependency=afterany:<parent>` after submitting a chain.
- **Never compute on the login node**: tests, `pixi run python`, dataset loads go through SLURM
  (`-p gpu-be --gres=gpu:0`). Stdlib `python3` on small JSON (`scripts/paired_judge.py`) is fine. Compute nodes do not
  see the login node's `/tmp`: job scripts live under the repo (`expQ/…`).
- Stage-2 pretrain: `pt_mm_4b.sh <samples> <bs> <accum> merger <init model> <hydra overrides>`, env
  `PT_DATA='path:N,…'`, `PT_OUT`. Task mix `collator_kwargs.p_text / p_image_ae / p_image_cont`,
  `+collator_kwargs.p_desc / p_find`; KL `+model_config_overrides.kd_weight=0.5 +collator_kwargs.teacher_inputs=true`.
- Finetune: `ft_mm_4b.sh <model> <samples> <bs> <accum> <overrides>`, env `FT_DATA`, `FT_OUT`; second seed
  `+hf_training.seed=43`; norm off = `model_config_overrides.connector_out_norm=false`; archq2 fields
  `model_config_overrides.compressor_extra_layers=[18]` / `mem_position_stride=8`.
- Eval: `eval_mm_4b.sh <model> <pisco|base> [limit] [data] [image_rate]`, env `TOPK`, `PX`, `ADAPTER` (fair
  ceiling), `MERGE=1`, `RAW=gold|route1|route2|all`, `RANK=<retriever json>`, `STYLE=expand`, `GOLD=1`, `MQ`, `XD`,
  `DML`, `CFG` (own `_cfg` suffix), `JPROMPT` / `JMODEL` (OpenEQA: `openeqa` / `Qwen/Qwen3-14B`). Judge chained.
  Tables: `scripts/show_archq.py`, `scripts/show_mm_results.py --ci`, `scripts/openeqa_results.py table`; paired
  stats `scripts/paired_judge.py`.
- Probes: `expQ/archq2/probe_connector.sh <ckpt> [N_ROWS] [REF]`, `expQ/archq2/probe_mem_position.sh`,
  `probe_recall_4b.sh ref|ae_s2|ae_ftcomp` → `scripts/analyze_mem_recall.py`; serving `bench_combo_4b.sh`,
  `bench_compress_4b.sh`.
- Video: `expQ/video/{prep_lvbench,route_ewb,zoom_ewb,mcq_levels,transcribe}.sh`; scripts in `scripts/video/`;
  videos `data/video/lvbench`, frames `data/video/lvbench_frames`.
- Integration checks after touching `pisco/model.py`, `pisco/mm_*.py`, `scripts/eval_mm_docqa.py`:
  `expQ/mm_fix_checks/mm_fix_checks.py` (GPU job), `run_tests_cpu.sh` (`test_positions.py`), `python test_mm_collators.py`.
- Judge tooling: `scripts/build_judge_pool.py` → `scripts/judge_study.py` → `scripts/judge_agreement.py`;
  re-score with `scripts/judge_mm_docqa.py [--prompt rubric|project|openeqa]`.
- Data builders: `build_stage2_v3.sh`, `teacher_desc_4b.sh`, `build_stage2_v4.sh`, `relabel_mm_4b.sh`; eval sets
  `scripts/build_mm_eval{,_retrieved,_longdoc,_openeqa}.py`, `scripts/build_text_eval_hotpot.py`, `scripts/rank_text_docs.py`.
- vLLM env `Reranking/vllm_env` (needs `C_INCLUDE_PATH=/usr/local/cuda/include`, `VLLM_DISABLE_COMPILE_CACHE=1`);
  Habitat env `/beegfs/scratch/user/hdejean/cosmos_nav_test/habitat_env/pixi.toml`.

## Models and data
| path | what |
|---|---|
| `expQ/ft_mm_4b_174908/model` (s42), `expQ/ft_mm_4b_176242/model` (s43) | best: 173550 recipe + connector norm; the references for every arm |
| `expQ/ft_mm_4b_173550/model` | same without the norm (numbers in `multimodal_results.md`) |
| `expQ/ft_mm_4b_17831{9,20,21,22}/model` | archq2 arms (closed; extra layer 18 / decoder stride 8) |
| `expQ/archq3/pt_text_1M_spread`, `pt_mm_spread`, `pt_mm_base_from1M`, `pt_text_1M_coarse`, `pt_text_5M_spread` (184769, running), `pt_mm_5M_spread` (184777, queued) | archq3 chains: spread stage 1 / stage 2, tail baseline stage 2, archq3b stage 1 (parked), full-budget spread stage 1 / 2 |
| `expQ/ft_mm_4b_18353{8,9}` (baseline chain), `18381{4,6}` (spread chain), `18354{8,9}` (fp32 5e-6 regime), `180488/9`, `18356{3,4,5,6}` (connector arms) | archq3 / precision finetunes |
| `expQ/ft_nav_4b/model`, `expQ/ft_nav_base_4b/model` | nav model (rate 16) and its fair-ceiling LoRA |
| `expQ/149907_nle`, `expQ/149908` | text-only 4B→9B / 4B→4B (connector probe) |
| `data/mm_pretrain_{pdfa_1m,wiki_1m,natural,charts}(_desc)`, `data/nav_pretrain(_desc)` | stage-2 sets |
| `data/mm_ft_docpages_20k_1m_teacher`, `data/mm_ft_natural_8k_teacher`, `data/nav_ft_{train,val}_v2` | finetune sets |
| `eval_data/mm_docqa_1500_1m`, `mm_naturalqa_1500` | main eval sets (500 per source; `*_500*` are subsets) |
| `eval_data/mm_docqa_retr{4,10}_1m`, `mm_docqa_longdoc_1m`, `text_hotpotqa_1500`, `nav_ft_val_v2_gold` | hard-test sets |
| `eval_data/openeqa_hm3d_k{16,32,64,all}`; frames `/beegfs/scratch/user/hdejean/open-eqa/data/frames/hm3d-v0` | OpenEQA |
| `expQ/judge_study/` | judge pool, 7 judge outputs, human sheet |

## Don't-break list
- Image spans are atomic (never chunk/truncate); pixel row order == image-chunk row order; `bidirectional` + images
  refused, and `bidirectional` + any hybrid (Qwen3.5) compressor refused.
- `<MEM>` count in compressor == decoder (`assert_consistent_n_mems`); chat-task rows mask by prompt length, others by
  `mask_before_mem` (`MultimodalPretrainingCollator.torch_call`). `mask_before_mem` is a no-op in the base
  `PretrainingCollator` (do NOT change).
- A stride > 1 checkpoint must not go through scripts that call `model.decoder.generate` without `position_ids`
  (`bench_mm_latency.py`, `probe_mem_recall.py`, `eval_arc_tasks.py` are legacy-only); always pass an attention mask.
- KD batches: teacher tensors hold only the KD rows (`teacher_kd_rows`); `PISCO.forward` slices the student side.
- `pisco/mm_tasks.py` prompts are shared by training and the vLLM teacher: change them → regenerate `data/teacher_desc`.
- `scripts/judge_prompts.py` is the single copy of the judge prompts (rubric, project, openeqa).
- Speed numbers need merged LoRA: `pisco.model.merge_lora_inplace` (`merge_and_unload` was a silent no-op).
- Any decoder forward with `input_ids` + images leaves `rope_deltas` cached on Qwen3.5; snapshot/restore it
  (`PISCO.kd_loss`, eval `_reset_rope_deltas`). Mixed raw + `<MEM>` prompts inject `<MEM>`s through a forward hook on
  the embedding layer and pass `input_ids`, so Qwen places vision features and M-RoPE itself.
- Qwen3.5-4B has no generation_config.json: keep `PISCO.decoder_eos_ids()` / `eos_token_ids()` in every generate call.
- The eval refuses prompts over `--decoder_max_length` instead of trimming; `--image_max_pixels` defaults to the
  checkpoint's. Every eval with `CFG` or `LIMIT` gets its own result file: never let a shake-out write a default name.
- Compare arms paired on ≥ 500 Q within one eval campaign (answers are GPU-state sensitive); a training change under
  ~.03 on documents needs 2 seeds (`train.py` hardcodes 42).
- SLURM: `afterok` on a job that finished minutes ago fails; depend only on queued jobs; rewire with
  `scontrol update JobId=… Dependency=…`. HF streaming builders must `os._exit`. gpu-be preemptions cost up to 100 steps.
- `*.sh` launchers are gitignored; durable logic lives in `scripts/`, `pisco/configs/` and `expQ/*/`.
- Agent-QA numbers before 2026-07-15 are on buggy labels (`my_qa_fixed.json`). Older open items: [raft] gpt-4.1 relabel
  not launched (needs `OPENAI_API_KEY`); [bidi_ablation] causal 4B→4B finetune 162668 never evaluated.
