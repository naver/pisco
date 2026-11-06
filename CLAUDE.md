# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PISCO (ACL 2025) and OSCAR (ICLR 2026) are **compress-then-generate** models for faster RAG inference. The same code base supports both; OSCAR is just PISCO with `query_dependent=True` in the fine-tuning collator. The full PISCO model is a single HuggingFace `PreTrainedModel` (`pisco.model.PISCO`) wrapping three pieces:

- **Compressor**: a small LM. Tokenized text contains `<MEM>` markers; the compressor's last-layer hidden states at those positions are extracted.
- **Connector**: a 2-layer MLP (`hidden_size -> compressor_mlp_hidden_dim -> decoder_hidden_size`). Always present — this is what enables using a *small* compressor with a *large* decoder, and means there is no "no-pretraining" path.
- **Decoder**: a (usually larger) LM, optionally LoRA-adapted. Its input embeddings are computed normally, then any `<MEM>` token's embedding is **replaced** with the corresponding compressed embedding from the connector before the forward pass (`PISCO.replace_embeddings`).

**Invariant the entire system depends on**: the number of `<MEM>` tokens in `compressor_input_ids` must equal the number of `<MEM>` tokens in `decoder_input_ids`. Collators enforce this with `assert_consistent_n_mems`; new collators must do the same.

The number of `<MEM>` tokens for a chunk is proportional to chunk length: `n_mems = len(chunk) // compr_rate + 1` (see `add_memory_tokens_to_inputs` in [collator_utils.py](pisco/collator_utils.py)). This differs from the original PISCO paper, which used a fixed count.

## Layout

- [pisco/model.py](pisco/model.py) — `PISCOConfig`, `PISCO`. `forward` returns `{loss, logits}`. Custom `save_pretrained`/`from_pretrained` because checkpoints may store **only** LoRA adapters + connector, not the base decoder/compressor weights. Decoder is loaded via `AutoModelForImageTextToText` when possible, falling back to `AutoModelForCausalLM` — this matters for adapter path consistency between training and loading.
- [pisco/collator.py](pisco/collator.py) — four collators sharing `BaseCollator`:
  - `PretrainingCollator` — autoencoding (full text compressed, decoder reproduces it after `<AE>`) mixed with text-continuation, gated by `ae_ratio`.
  - `FineTuningCollator` — RAG: expects `docs`, `query`, `mistral_label`. Set `query_dependent=True` for OSCAR.
  - `FineTuningCollatorA` — variant with `compressed_docs` + `uncompressed_docs` fields (hybrid context).
  - `AgentTrajCollator` — agent trajectories; compresses individual steps with probability `p_compress`.
- [pisco/collator_utils.py](pisco/collator_utils.py) — chunking, `<MEM>` insertion, label masking (`mask_before_mem` for pretraining, prefix-length masking for finetuning).
- [pisco/train.py](pisco/train.py) — Hydra entry point. Picks config via `--config-name=` and forwards `hf_training` kwargs to `TrainingArguments`. Sets `datasets.disable_caching()` and `IN_MEMORY_MAX_SIZE = 200 GB` (the whole arrow dataset is loaded into host RAM — size `--mem` accordingly). Passes `image_processor=model.image_processor` to any collator whose `__init__` declares that kwarg.
- [pisco/mm_utils.py](pisco/mm_utils.py) / [pisco/mm_collator.py](pisco/mm_collator.py) — multimodal (image) compression, see the section below.
- [pisco/configs/](pisco/configs/) — Hydra configs (`pretraining.yaml`, `finetune.yaml`, `agentic.yaml`, `arc.yaml`, `arc_lora_c.yaml`, `pt_lora.yaml`, `pt_lora_bidi.yaml`, `pt_lora_mm.yaml`, `finetune_mm.yaml`). All inherit from `defaults.yaml`.
- [scripts/eval_agent_qa.py](scripts/eval_agent_qa.py) — standalone eval for agentic QA and LongBench, with both `--mode pisco` and `--mode base` (uncompressed decoder baseline). [scripts/eval_mm_docqa.py](scripts/eval_mm_docqa.py) is the multimodal counterpart (page images; modes `pisco` / `base` / `pisco_text`).
- [docs/multimodal_results.md](docs/multimodal_results.md) — what compresses well (per image type, rates, ceilings), how each multimodal component was settled, open questions, best checkpoints.
- [docs/latency.md](docs/latency.md) — latency/throughput of multimodal PISCO (HF and vLLM), how to serve the PISCO decoder in vLLM with prompt embeddings, when compression stops paying. Benchmarks: `scripts/bench_mm_latency.py`, `scripts/bench_vllm.py` (also `--kv_cache_dtype`, `--mamba_ssm_cache_dtype`, `--spec_mtp`), `scripts/bench_vllm_sim.py` (`--model` for another backbone), `scripts/bench_compress_cost.py`.
- [docs/architecture_review.md](docs/architecture_review.md) — is the design right (literature review + probes, 2026-10-21): keep the architecture; **route-then-expand** (the model picks the answer page from its `<MEM>`s, that page goes raw) reaches the uncompressed score on documents; where answers are lost; serving and backbone trade-offs; ranked next changes.
- [docs/connector_probe.md](docs/connector_probe.md) — how to probe a component at inference before training (replace / perturb it on the trained model, paired answer NLL on held-out rows); the connector results: a refit linear map equals the MLP, `<MEM>` order inside a page is unused.
- [docs/mem_tokens_review.md](docs/mem_tokens_review.md) — literature + design review (2026-10-30) of the `<MEM>` tokens themselves: design space (embeddings, budget, placement, attention among slots, pooling, slot-level losses), what our probes rule out, ranked shortlist (chunk-local `<MEM>` mixer, per-slot ownership loss, nested budgets) with inference probes to run first.
- [docs/training_review.md](docs/training_review.md) — literature + design review (2026-10-30) of the training pipeline: objectives, distillation, RL (decoder-only GRPO on cached `<MEM>`s), data, stage design, and the bf16 lost-update problem (why fp32 masters were worse at the same lr); ranked shortlist with controls.
- [docs/training_plan.md](docs/training_plan.md) — **the ordered plan for the next training rounds** (2026-10-30): rules the owner set, what is running, round 1 precision/lr regime, round 2 decoder-only GRPO on cached `<MEM>`s (design, controls, what code is missing), round 3 `<MEM>` mixer, round 4 stage-2 upgrades gated on archq3, closed items. Start here in a new session.
- [docs/data_review.md](docs/data_review.md) — review (2026-10-30) of training data, stage-2 tasks and labels: routing rows on real questions, evidence-then-answer teacher labels, counterfactual rows, dense synthetic pages; three stage-2 collator cuts that bias toward the top of the page; dedup risks (MP-DocVQA val vs DocVQA train pages, SlideVQA licence).
- [docs/literature_scan_2026-11.md](docs/literature_scan_2026-11.md) — literature scan (2026-11-06) against the measured weaknesses after the position-layout win: ranked shortlist (multi-position slots / PPE, per-head attention scales / SEAL, answer-aligned slots / CMC, attention-mass routing, Matryoshka prefix training), per-domain notes, what the backbone blocks, open questions.
- [docs/hierarchical_video.md](docs/hierarchical_video.md) — literature review (2026-10-28) for hierarchical compression of hour-long video: prior work by family, whether temporal information survives compression, design rules for a trained coarse level.
- [pisco/example.py](pisco/example.py) — minimal forward+backward example, useful as a sanity check that a checkpoint loads.

## Common commands

This project uses **pixi** (not pip/poetry/conda). The Pixi-defined tasks in [pixi.toml](pixi.toml):

```bash
pixi run pretrain   # python pisco/train.py --config-name=pretraining
pixi run finetune   # python pisco/train.py --config-name=finetune
pixi run agentic    # python pisco/train.py --config-name=agentic
pixi run eval-qa --checkpoint_path <path>   # scripts/eval_agent_qa.py
```

Hydra-style overrides apply to all training commands, e.g.:

```bash
python pisco/train.py --config-name=pretraining \
  ++data.samples=1000 ++hf_training.logging_steps=1 ++hf_training.eval_steps=1 \
  ++model.init_args.config.decoder_model_name=Qwen/Qwen3-0.6B \
  ++model.init_args.config.compressor_model_name=Qwen/Qwen3-0.6B

python pisco/train.py --config-name=finetune \
  ++model_name_or_path=PRETRAINED_MODEL_OUT_PATH \
  ++data.samples=1000
```

Dev tooling (in the `dev` feature): `ty` and `ruff`. Activate via `pixi shell -e dev`.

There is no test suite — `example.py` is the closest thing to a smoke test.

## Things to know before changing code

- **Adding a collator**: subclass `BaseCollator`, return a dict with `compressor_input_ids`, `compressor_attention_mask`, `decoder_input_ids`, `decoder_attention_mask`, `labels`. Call `assert_consistent_n_mems` before returning. Wire it in via `collator_class:` in a config.
- **`<MEM>` and `<AE>` are added as special tokens** in both tokenizers (`PISCO.create_*_tokenizer`); model embeddings are resized accordingly. Don't strip special tokens in places that expect them.
- **Checkpoints are partial**: by default the decoder is LoRA, so `save_pretrained` writes only the decoder adapter + the compressor (or its adapter, under `compressor/`) + `connector.pt` + the PISCO config. `from_pretrained` reads the PISCO config to decide whether to load adapters or full weights, and rewrites `compressor_model_name` / adapter paths to point inside the checkpoint dir.
- **Decoder padding side is `left`, truncation side is `right`** (set in `create_decoder_tokenizer`). Generation depends on this.
- **`<MEM>` placement**: the system relies on `<MEM>` tokens being at the **end** of each compressor chunk (so trimming from the tail is safe — see `_adjust_compressor_chunks_to_target_mems` in `eval_agent_qa.py` for the trick used when decoder-side truncation drops `<MEM>`s).
- **Hardware**: CUDA 12; `flash_attention` is expected for performance. `transformers == 5.5` (per `pixi.toml`) — note this is a fairly new major; APIs like `AutoModelForImageTextToText` are used.
- **GPU: A100/H100 "just work"; V100 needs adaptation but is viable at scale.** The configs are built around **bf16** (`hf_training.bf16: True`, `PISCOConfig.torch_dtype: "bfloat16"`), native on Ampere+ (A100/H100). **V100 (Volta, sm70) has no native bf16.** When A100/H100 are scarce, V100 is practical — there are whole 4-V100 nodes free — once you apply the six points below. A working V100 launcher (`GPU_KIND=v100` path, 4-GPU DDP) is `phase2_cell.sbatch`.
  1. **Precision**: bf16 on V100 is software-emulated → ~20× slower (≈64 s/it vs a few). Switch to fp16: `hf_training.bf16=False +hf_training.fp16=True` (`fp16` isn't in the config struct, so it needs the `+` append form, not a plain override).
  2. **Master weights**: fp16 AMP requires fp32 master weights, else `ValueError: Attempting to unscale FP16 gradients`. Set `++model.init_args.config.torch_dtype=float32` (saved into the checkpoint, so a finetune started from it inherits fp32).
  3. **Memory (GPU)**: fp32 weights double memory → the `pt_lora` default `per_device_train_batch_size=24` OOMs on a 32 GB V100. Drop to bs≈2 with higher `gradient_accumulation_steps`.
  4. **Attention**: flash-attention-2 needs sm80+, so it is unavailable on V100; it falls back to sdpa.
  5. **Multi-GPU to recover throughput**: a single V100 is slow (~20 s/it at bs2 fp32). Use a whole node via `pixi run accelerate launch --multi_gpu --num_processes 4 …` — confirmed ~6.9 s/it on 4×V100 (data-parallel offsets Volta's per-GPU deficit). bf16/A100 stays single-GPU `python`.
  6. **Memory (host RAM)**: multi-process DDP loads the dataset **per process** → `--mem=64G` gets `slurmstepd oom_kill`. Request `--mem=256G` for a 4-GPU node (and bump `--cpus-per-task`).
  Note: `PISCO.replace_embeddings` casts the compressed embeddings to the decoder-embedding dtype before `index_copy`, so mixed bf16/fp16 autocast no longer crashes there (no-op when dtypes already match).

## Multimodal PISCO (compressing page images)

The Qwen3.5 backbones are **native VLMs** (`Qwen3_5ForConditionalGeneration`; both backbones are loaded with `AutoModelForImageTextToText`), so an image can go through the *compressor* with no new module: the compressor input gets an **image span** `<|vision_start|> <|image_pad|>×N <|vision_end|>` (N = visual tokens, one per 32×32 px after `smart_resize`; 451 584 px ≈ 440 tokens) followed by `<MEM>` tokens using the usual rule `n_mems = len(span)//compr_rate + 1`. The decoder only ever sees the `<MEM>` placeholders.

- `PISCO.compress/forward/generate` accept `pixel_values`, `image_grid_thw`, `mm_token_type_ids` (all three or none; `mm_token_type_ids` is required by Qwen's forward whenever `image_grid_thw` is given). Text-only batches follow the legacy code path byte for byte.
- **Image spans are atomic**: never chunk or truncate one (the HF model raises "image features and image tokens do not match"; `compress` asserts the counts first). `compressor_max_length` bounds text chunks only. Trimming `<MEM>`s from a chunk tail is still safe.
- Row order of image chunks == concatenation order of `pixel_values` == rows of `image_grid_thw` (`pisco.mm_utils.collate_compressor` enforces it). Don't re-sort compressor rows without re-sorting pixels.
- Image processor: `Qwen2VLImageProcessorPil` (numpy/PIL, no torchvision; `AutoProcessor` needs torchvision for the video processor and is not used). `PISCOConfig.image_min_pixels/image_max_pixels` set the budget; `save_pretrained` writes `preprocessor_config.json` under `compressor/`.
- `PISCOConfig.train_vision_lora ∈ {all, merger, none}`: `all-linear` LoRA also wraps the ViT; old checkpoints carry those adapters with `lora_B == 0`. mm runs use `merger` (ViT frozen, merger + LM trained — the Qwen-VL convention).
- `PISCOConfig.connector_out_norm` adds an RMSNorm with a per-dim gain after the connector MLP, registered as `connector.norm` (the legacy `0.*`/`2.*` connector keys stay; the Trainer's "norm" rule keeps the gain out of weight decay). `load_trainable_state` retrofits it onto a checkpoint saved without it (gain = `connector_out_norm_init`). The `PISCOConfig` default stays `False` (old checkpoints lack the key); `finetune_mm.yaml` turns it on with init 0.461 = mean `<MEM>` length / sqrt(hidden) of pt_mm_v4kd_4b, so re-measure the init for another start checkpoint. Never shrink `<MEM>`s to token scale: they are ~30x a word embedding by design.
- Read-out path options (archq2, 2026-10-29, both off by default): `PISCOConfig.compressor_extra_layers=[k, …]` concatenates the `<MEM>` hidden states of those compressor layers (unit-RMS, no gain) to the last layer's; the connector's first Linear widens and `load_trainable_state` zero-fills the new columns of an old checkpoint (exact retrofit). `PISCOConfig.mem_position_stride=S` gives every decoder `<MEM>` S position ids (`PISCO.decoder_position_ids`, applied in `forward`/`generate`); a script that calls `model.decoder.generate` or `model.decoder(...)` itself on a stride > 1 checkpoint must pass `position_ids` (see `_pos_kwargs` in `scripts/eval_mm_docqa.py`, which also handles prompts with raw image spans via `get_rope_index`) and always an attention mask (HF reads non-consecutive ids with no mask as packed sequences). Tests: `test_positions.py`; shake-out: `mm_shakeout.sh … "--config_override …"`.
- `bidirectional=True` is refused for any hybrid backbone (Qwen3.5: its linear-attention layers ignore the 4D mask and skip padding zeroing), and for images on any backbone (the 4D mask breaks Qwen's `get_rope_index`).
- Collators: `MultimodalPretrainingCollator` (rows `image`, `text`; tasks image-AE / image-continuation / text rehearsal / `desc` / `find`) and `MultimodalFineTuningCollator` (rows `docs`, `doc_images`, `query`, `mistral_label`). KD batches carry teacher tensors for the KD rows only (`teacher_kd_rows` marks them; `PISCO.forward` slices the student side to match). `clean_text` also defuses literal vision tokens in data. Tests: `python test_mm_collators.py` (real modules, mocked tokenizer/processor).
- `compress()` calls the compressor with `logits_to_keep=1, use_cache=False` (same hidden states, no full-vocab logits).
- Checkpoints: a non-LoRA decoder is saved whole in `decoder_state.pt`; a non-LoRA compressor is saved in HF format under `compressor/` and reloaded on resume by `load_trainable_state` (needed for requeued pretraining runs).
- Data builders: `scripts/build_mm_pretrain.py` (Tevatron/wiki-ss-corpus screenshots + text), `scripts/build_mm_finetune.py` (OpenDocVQA train + corpus, gold + distractor pages), `scripts/build_mm_eval.py` (OpenDocVQA infovqa/chartqa test + VisRAG SlideVQA). Pages are pre-resized to the pixel budget and stored as JPEG; `from_pretrained` loads the whole arrow into RAM.
- Eval: `scripts/eval_mm_docqa.py` (EM/F1/ANLS/match, `--rescore`) builds its chunks through `MultimodalFineTuningCollator.example_chunks`, so it truncates/cleans documents exactly like finetuning (`--query_dependent` for OSCAR checkpoints); the pixel budget defaults to the checkpoint's own (`--image_max_pixels` overrides); a prompt over `--decoder_max_length` is an error, never a silent trim. Then `scripts/judge_mm_docqa.py` (local Qwen3-8B; default the **rubric** prompt of `scripts/judge_prompts.py`, JSON verdict + reason kept as `judge_raw`; `--prompt project` = the earlier Yes/No prompt) adds `judge` scores in place; `eval_mm_4b.sh` chains both. Eval sets: `eval_data/mm_docqa_1500_1m` and `mm_naturalqa_1500` (500 questions per source; the older 500-question sets are exact subsets). Compare two arms with `scripts/paired_judge.py` (paired bootstrap CI + McNemar); `scripts/show_mm_results.py --ci` prints each model's paired difference to the baseline. Probe options of the eval (all off by default; `eval_mm_4b.sh` env in brackets): `--raw_policy gold|route1|route2|all` (`RAW`: some pages raw instead of `<MEM>`s, via a forward hook on the decoder embedding layer; routing reads one digit, so at most 10 pages), `--answer_style expand` (`STYLE`), `--gold_only` (`GOLD`), `--mem_quant int8|int4|fp8` (`MQ`), `--extra_distractors N` (`XD`, with `TOPK`); `scripts/show_archq.py` tabulates them paired with a reference file. Transcription probe ("is the answer in the `<MEM>`s?"): `scripts/probe_mem_recall.py` + `scripts/analyze_mem_recall.py`. Judge reliability: `scripts/build_judge_pool.py` / `judge_study.py` (vLLM) / `judge_agreement.py`.
- Shake-out before training: `scripts/mm_shakeout.py` (processor shapes, token ids, one-image compress, mixed-batch M-RoPE check, fwd+bwd memory probe, generate, datasets schema).

## Reproducing paper results

- PISCO: `--config-name=finetune` with `data.samples=500000` and appropriate backbones, starting from a pretrained checkpoint (`--config-name=pretraining` first).
- OSCAR: same as PISCO finetune but with `++collator_kwargs.query_dependent=True`. OSCAR's exact training data is not released; PISCO data with `query_dependent=True` gives a reasonable approximation.

The README notes results land within ~1% of the published numbers due to implementation differences from the original release (single `<MEM>` token instead of `<MEM1>`/`<MEM2>`/...; length-proportional `<MEM>` count instead of fixed; mandatory connector).
