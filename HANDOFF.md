# HANDOFF — PISCO (multimodal)

_Updated 2026-10-28 (NLE), branch `arc`. Restart point. Findings: **`docs/multimodal_results.md`** (what compresses
well, how the evaluation is done, how each component was settled). Design review + probes: **`docs/architecture_review.md`**.
Speed: **`docs/latency.md`**. Science record: `EXPERIMENTS.md` (`expmon ledger`). Design/invariants: `CLAUDE.md` → "Multimodal PISCO"._

## Goal
Multimodal PISCO: compress images (document pages, photos, charts, video frames) and text into `<MEM>` vectors so a
decoder answers without seeing pixels — as close as possible to the uncompressed Qwen3.5-4B, much faster.

## Status

2026-10-28: everything up to now is committed. Building the hard route-then-expand test (next step 1); training
waits for its result.

- **Connector norm and training precision (2026-10-21/23, `expQ/norm_probe/`):** `connector_out_norm=true` (RMSNorm on
  the connector output, gain init 0.461, retrofit onto a stage-2 checkpoint) gives photos +.021 at both seeds (174908
  s42, 176242 s43), docs nothing (+.021 / −.016), LongBench F1 +.012 (n.s.), route-then-expand .810. **On by default
  in `finetune_mm.yaml` since 2026-10-28** (the init is tied to pt_mm_v4kd_4b: re-measure it for another start
  checkpoint; off = `model_config_overrides.connector_out_norm=false`, no `+`). Trainable weights
  are stored in bf16, so updates below half a bf16 step vanish: in a finetune 46% of the 308M trainable entries ever
  change (mostly LoRA-B), in stage-2 58%. fp32 master weights at lr 2e-5 are worse (docs −.043; with the norm −.167).

- **archq campaign (2026-10-21): is the design right?** Literature review (~100 papers) + 25 inference-only jobs; written
  up in `docs/architecture_review.md`. Verdict: keep the architecture, extend it.
  - **Route-then-expand closes the document gap with no training.** The model picks the page that answers from the
    `<MEM>`s (`find`-style prompt), that page goes raw, the rest stay `<MEM>`: .801 vs uncompressed .809 (p = .35) on
    1,500 Q, routing 98%; holds at 8 and 10 pages (.803 / .796 vs .807 / .801), 22-36% of the raw tokens, 2-3× the raw
    speed under load. `eval_mm_4b.sh` env `RAW=route1` (also `gold`, `all`, `route2`), `XD`/`TOPK` for extra pages.
  - **Where answers are lost** (transcription probe): infographics lose content in compression (25 lost vs 17 unused),
    multi-page documents keep it but the decoder does not use it (27 unused vs 6 lost).
  - **Rate 4 (1.8× the tokens) does not pay on documents:** vs rate 8 at the same seed, docs +.034 (s42, 174693 vs
    173550) but −.022 (s43, 176425 vs 176243): mean +.006. Photos +.005 / +.029: mean +.017. Keep rate 8.
  - Small results: evidence-then-answer prompt +.025; int8 `<MEM>`s free (int4 −.018); rate is rigid at inference
    (r16 −.070); distractor pages are not the cause (gold page only +.014).
  - Serving: a full-attention decoder gains more from compression (Qwen3-4B 5.0× vs Qwen3.5 3.0×, 512-token answers,
    256 requests); float16 linear-attention state +9-14% on long answers; fp8 KV impossible on A100 (vLLM 0.19.1); MTP
    refused with prompt embeddings (on token prompts it gives 1.45× for one request, nothing at 64). Compression cost
    per page: 4B 65.6 ms, 0.8B 21.0 ms.
- **Result on 1,500 Q (rubric judge), best model `expQ/ft_mm_4b_173550/model`:** documents .578 vs uncompressed .809
  (slides 84%, multi-page 74%, infographics 56% kept); photos .665 vs .717 (93%); nav .806 vs fair ceiling .792.
  Compression beats fewer pixels at the same token budget (documents .578 vs .403).
- **Evaluation (2026-10-20):** 500 questions per source, rubric judge (`scripts/judge_prompts.py`), paired comparisons
  (`scripts/paired_judge.py`, `scripts/show_archq.py`); effects under ~.04 on 500 Q are unproven. Training noise: the
  same finetune at seed 42 vs 43 differs by .029 on docs, which the paired test does not see, so a training change under
  ~.03 needs 2 seeds per arm (`+hf_training.seed=43`; `train.py` hardcodes 42).

## Next step
1. **Route-then-expand, hard cases (in progress):** same-document distractors hold (379 Q already in the set: .818 vs
   uncompressed .805, routing 95%; `expQ/archq/hardneg_split.py`). Building: retrieved negatives (a visual retriever
   ranks the VisRAG corpora) and whole documents of more than 10 pages (score the whole "Document j" answer instead
   of one digit; needs the original SlideVQA / MP-DocVQA documents).
2. **Training, decided by step 1:** if route-then-expand drops on the hard cases, a finetune on mixed raw + `<MEM>`
   prompts (norm on, 2 seeds); if it holds, GRPO with an evidence-then-answer format for the all-`<MEM>` mode
   (~1-2 GPU-days), or stop training. Owner's call.
3. **Human check of the judge:** label the 114 rows of `expQ/judge_study/human_sheet.md` (last column, 1/0), then
   `python3 scripts/judge_agreement.py --pool expQ/judge_study/pool.json expQ/judge_study/q*.json expQ/judge_study/gemma*.json --human expQ/judge_study/human_sheet.md`.
4. **(Low priority) fp32 master weights with a learning-rate sweep:** 2e-6 / 5e-6 / 1e-5 with
   `+model_config_overrides.frozen_base_dtype=bfloat16 +model_config_overrides.torch_dtype=float32`, 2 seeds for the
   best; switch the recipe only if it beats bf16. At 2e-5 it is worse (176295/176296).

## Key files & commands
- Launch only via `expmon launch --exp mmpisco …`. Partition `gpu` caps 24 queued / 8 running jobs per user; `gpu-be` is
  preemptible and can sit at zero priority — move a stuck job to `-p gpu --constraint gpu_a100`.
- **Never compute on the login node** (memory `never-compute-on-frontend`): tests, `pixi run python`, dataset loads go through
  SLURM (`-p gpu-be --gres=gpu:0`). Stdlib `python3` on small JSON (e.g. `scripts/paired_judge.py`) is fine. Compute nodes
  do not see the login node's `/tmp`: job scripts live under the repo (`expQ/…`).
- Stage-2 pretrain: `pt_mm_4b.sh <samples> <bs> <accum> merger <init model> <hydra overrides>`, env `PT_DATA='path:N,path:N,…'`
  (mixed sets, quoted for Hydra) and `PT_OUT`. Task mix via `collator_kwargs.p_text / p_image_ae / p_image_cont` and
  `+collator_kwargs.p_desc / p_find`; KL via `+model_config_overrides.kd_weight=0.5 +collator_kwargs.teacher_inputs=true`.
- Finetune: `ft_mm_4b.sh <model> <samples> <bs> <accum> <overrides>`, env `FT_DATA` (same `path:N` spec), `FT_OUT`.
- Eval: `eval_mm_4b.sh <model> <pisco|base> [limit] [data] [image_rate]`, env `TOPK` (docs per Q), `PX` (pixel budget),
  `ADAPTER` (base mode + LoRA, fair ceiling), `MERGE=1` (merged LoRA); archq probes `RAW`, `STYLE=expand`, `GOLD=1`,
  `MQ`, `XD` (+ `TOPK`), `DML` (prompt limit). Judge (rubric) is chained. Tables: `scripts/show_archq.py`.
- archq launchers: `probe_recall_4b.sh ref|ae_s2|ae_ftcomp` (transcription probe → `scripts/analyze_mem_recall.py`),
  `bench_combo_4b.sh` (vLLM serving arms ref/tri/fp8/ssm16/mtp/mtptext), `bench_compress_4b.sh`; launch lists
  `expQ/archq/launch_wave{1,2}.sh`.
- Judge tooling: `scripts/build_judge_pool.py` → `scripts/judge_study.py` (vLLM env, one condition per call) →
  `scripts/judge_agreement.py`; re-score files with `scripts/judge_mm_docqa.py [--prompt rubric|project] <json…>`.
- Integration checks of the multimodal code path: `expQ/mm_fix_checks/mm_fix_checks.py` (A–F; GPU job; run after any
  change to `pisco/model.py`, `pisco/mm_*.py`, `scripts/eval_mm_docqa.py`). Unit tests: `python test_mm_collators.py`.
- Data: `build_stage2_v3.sh` (pages/photos/charts), `teacher_desc_4b.sh` (vLLM teacher descriptions),
  `build_stage2_v4.sh` (attach descriptions, natural-QA sets), `relabel_mm_4b.sh` (teacher labels; `SHARD=merge`).
- vLLM: env `Reranking/vllm_env` (`pixi run --manifest-path …/vllm_env/pixi.toml`); launchers need
  `C_INCLUDE_PATH=/usr/local/cuda/include` and `VLLM_DISABLE_COMPILE_CACHE=1` (see `docs/latency.md`).
- Habitat env (nav renders): `pixi run --manifest-path /beegfs/scratch/user/hdejean/cosmos_nav_test/habitat_env/pixi.toml`.

## Data (current dirs)
| dir | content |
|---|---|
| `data/mm_pretrain_{pdfa_1m,wiki_1m,natural,charts}` (+ `_desc` = with `teacher_text`) | stage-2 pages at 1M px (92.5K / 40K), photos 60K, charts 37K |
| `data/nav_pretrain` (+ `_desc`) | 44,950 HM3D frames + descriptions (18K described) |
| `data/teacher_desc/*.jsonl` | raw teacher descriptions (plain text ≤200 words) |
| `data/mm_ft_docpages_20k_1m_teacher`, `data/mm_ft_natural_8k_teacher` | the generic finetune set (doc 1M + photo QA, teacher labels) |
| `data/nav_ft_{train,val}_v2`, `data/nav_hm3d_*` | nav memory QA, raw renders |
| `data/mmrl_*` | RL-track splits |
| `eval_data/mm_docqa_1500_1m`, `eval_data/mm_naturalqa_1500` | the eval sets (500 per source); `*_500*` = their first 500 |
| `expQ/judge_study/` | judge pool, 7 judge outputs, human sheet |

## Open questions / decisions pending
- Backbone for serving: a full-attention VLM decoder (e.g. Qwen3-VL-4B) and/or a 0.8B compressor (each a stage-2 + finetune).
- Nav external evals not started (OpenEQA EM-EQA via HF mirror `the-Lin/openeqa`, VSI-Bench). OpenDocVQA corpus is gated.
- Disk: 2026-10-15 cleanup freed 412 GB; `Reranking/cache/hub` (465 GB, other projects) untouched by user decision.
- Older: [raft] gpt-4.1 relabel of the 453K text set not launched (needs `OPENAI_API_KEY`); [bidi_ablation] causal 4B→4B
  finetune 162668 never evaluated.

## Don't-break list
- Image spans are atomic (never chunk/truncate); pixel row order == image-chunk row order; `bidirectional` + images refused,
  and `bidirectional` + any hybrid (Qwen3.5) compressor refused.
- `<MEM>` count in compressor == decoder (`assert_consistent_n_mems`); chat-task rows mask by prompt length, others by
  `mask_before_mem` (`MultimodalPretrainingCollator.torch_call`).
- KD batches: teacher tensors hold only the KD rows (`teacher_kd_rows`); `PISCO.forward` slices the student side.
- `pisco/mm_tasks.py` prompts are shared by training and the vLLM teacher: change them → regenerate `data/teacher_desc`.
- `scripts/judge_prompts.py` is the single copy of both judge prompts (judge_teacher imports it).
- Speed numbers need merged LoRA: `pisco.model.merge_lora_inplace` (the old `merge_and_unload` path was a silent no-op).
- Any decoder forward with `input_ids` + images leaves `rope_deltas` cached on Qwen3.5; snapshot/restore it (`PISCO.kd_loss`).
- Qwen3.5-4B has no generation_config.json: keep `PISCO.decoder_eos_ids()` / `eos_token_ids()` in every generate call.
- `mask_before_mem` is a no-op in the base `PretrainingCollator` (do NOT change).
- The eval refuses prompts over `--decoder_max_length` instead of trimming; `--image_max_pixels` defaults to the checkpoint's.
- SLURM: `afterok` on a job that finished minutes ago fails ("Job dependency problem") — depend only on queued jobs;
  rewire pending jobs with `scontrol update JobId=… Dependency=…`. HF streaming builders must `os._exit` (they hang).
- `*.sh` launchers are gitignored; durable logic lives in `scripts/` and `pisco/configs/`.
- Agent-QA numbers before 2026-07-15 are on buggy labels (`my_qa_fixed.json`).
