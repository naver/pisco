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
- [pisco/train.py](pisco/train.py) — Hydra entry point. Picks config via `--config-name=` and forwards `hf_training` kwargs to `TrainingArguments`. Sets `datasets.disable_caching()` and `IN_MEMORY_MAX_SIZE = 200 GB`.
- [pisco/hydra_utils.py](pisco/hydra_utils.py) — registers two OmegaConf resolvers used in config interpolation: `sanitize_override_dirname` (turns long `model_name_or_path` overrides into short hashes for sweep subdir names) and `path_after_user_dir` (strips `user_dir` prefix).
- [pisco/configs/](pisco/configs/) — Hydra configs (`pretraining.yaml`, `finetune.yaml`, `agentic.yaml`, `arc.yaml`, `arc_lora_c.yaml`, `pt_lora.yaml`). All inherit from `defaults.yaml` and optionally a gitignored `local.yaml` that sets `user_dir`.
- [scripts/eval_agent_qa.py](scripts/eval_agent_qa.py) — standalone eval for agentic QA, with both `--mode pisco` and `--mode base` (uncompressed decoder baseline).
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

## Reproducing paper results

- PISCO: `--config-name=finetune` with `data.samples=500000` and appropriate backbones, starting from a pretrained checkpoint (`--config-name=pretraining` first).
- OSCAR: same as PISCO finetune but with `++collator_kwargs.query_dependent=True`. OSCAR's exact training data is not released; PISCO data with `query_dependent=True` gives a reasonable approximation.

The README notes results land within ~1% of the published numbers due to implementation differences from the original release (single `<MEM>` token instead of `<MEM1>`/`<MEM2>`/...; length-proportional `<MEM>` count instead of fixed; mandatory connector).
