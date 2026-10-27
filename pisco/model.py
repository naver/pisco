"""
PISCO
Copyright (c) 2026-present NAVER Corp.
All Rights Reserved.

This is the PISCO model
It contains:
- a compressor model, which maps texts to embeddings, by collecting the last layer hiddens states
in front of every '<MEM>' token in its inputs
- a decoder model: during forward, any '<MEM>' placeholder in its context is replaced with the
appropriate embeddings computed by the compressor.
"""

import contextlib
import os
import warnings
from typing import List, Optional, Dict, Tuple
from typing import cast, TYPE_CHECKING

import torch
from torch import nn
from peft import LoraConfig, PeftModel

from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
)

if TYPE_CHECKING:
    from transformers.generation import GenerationMixin
    from transformers import TokenizersBackend

    # class for better type annotations
    class _BaseModelWithGenerate(PreTrainedModel, GenerationMixin):
        pass



class PISCOConfig(PretrainedConfig):
    model_type = "PISCO"
    # decoder_model_name: str = "Qwen/Qwen3-8B"
    # freeze_decoder: bool = False
    # compressor_model_name: str = "Qwen/Qwen3-4B"
    # compr_rate: int = 16
    # compressor_mlp_hidden_dim: int = 4096
    # lora_decoder: bool = True
    # lora_compressor: bool = False
    # lora_r_compressor: int = 64
    # lora_r_decoder: int = 64
    # attn_implementation: Optional[str] = None
    # device_map: Optional[str] = None
    # load_decoder: bool = True
    # decoder_gradient_checkpointing: bool = False
    # compressor_gradient_checkpointing: bool = False
    # decoder_adapter_path: Optional[str] = None
    # compressor_adapter_path: Optional[str] = None

    def __init__(
        self,
        decoder_model_name: str = "Qwen/Qwen3-8B",
        freeze_decoder: bool = False,
        compressor_model_name: str = "Qwen/Qwen3-4B",
        compr_rate: int = 16,
        compressor_mlp_hidden_dim: int = 4096,
        lora_decoder: bool = True,  
        lora_compressor: bool = False,
        lora_r_compressor: int = 64,
        lora_r_decoder: int = 64,
        attn_implementation: Optional[str] = None,
        device_map: Optional[str] = None,
        load_decoder: bool = True,
        decoder_gradient_checkpointing: bool = False,
        decoder_adapter_path: Optional[str] = None,
        compressor_gradient_checkpointing: bool = False,
        compressor_adapter_path: Optional[str] = None,
        torch_dtype: str = "bfloat16",
        frozen_base_dtype: Optional[str] = None,
        bidirectional: bool = False,
        image_processor_source: Optional[str] = None,
        image_min_pixels: int = 65536,
        image_max_pixels: int = 451584,
        train_vision_lora: str = "all",
        kd_weight: float = 0.0,
        kd_temperature: float = 1.0,
        connector_out_norm: bool = False,
        connector_out_norm_init: float = 0.4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        #decoder settings
        self.decoder_model_name = decoder_model_name  # model name of decoder
        self.load_decoder = load_decoder
        self.decoder_gradient_checkpointing = decoder_gradient_checkpointing
        self.decoder_adapter_path = decoder_adapter_path
        self.lora_decoder = lora_decoder  # boolean type, whether to use lora training
        self.lora_r_decoder = lora_r_decoder  # lora_r for lora training
        self.freeze_decoder = freeze_decoder

        #compressor settings
        self.compressor_model_name = compressor_model_name  # model name of compressor
        self.compr_rate = compr_rate  # compression rate
        self.compressor_mlp_hidden_dim = compressor_mlp_hidden_dim
        self.compressor_gradient_checkpointing = compressor_gradient_checkpointing
        self.lora_compressor = lora_compressor  # boolean type, whether to use lora training
        self.lora_r_compressor = lora_r_compressor  # lora_r for lora training
        self.compressor_adapter_path = compressor_adapter_path
        
        #other settings
        self.attn_implementation = attn_implementation
        self.device_map = device_map
        # When True, the compressor runs with full (bidirectional) attention
        # instead of the default causal mask. Requires a 4D-mask-capable attn
        # implementation ('sdpa'/'eager'), since flash_attention_2 only exposes
        # causality via an is_causal flag. See build_bidirectional_4d_mask.
        self.bidirectional = bidirectional
        # Guard: a 4D bidirectional mask is silently ineffective under
        # flash_attention_2 (and unset usually resolves to flash). Force sdpa so
        # the bidirectional path can't be left misconfigured. Honour an explicit
        # 'eager' (also 4D-capable); only override None/'flash_attention_2'.
        if self.bidirectional and self.attn_implementation not in ("sdpa", "eager"):
            if self.attn_implementation not in (None, "flash_attention_2"):
                warnings.warn(
                    f"bidirectional=True is incompatible with attn_implementation="
                    f"{self.attn_implementation!r}; forcing 'sdpa'."
                )
            self.attn_implementation = "sdpa"
        # Stored as string so the config serializes to JSON cleanly.
        # Resolved to a torch.dtype via _resolve_torch_dtype() at load time.
        self.torch_dtype = torch_dtype
        # Option 2 (V100 memory): when set (e.g. "float16"), the FROZEN base
        # backbones load in this dtype while the trainable LoRA adapters + connector
        # stay in `torch_dtype` (fp32). Halves base-weight memory so 4B+ fits a 32GB
        # V100. None -> everything uses torch_dtype (current/default behaviour).
        self.frozen_base_dtype = frozen_base_dtype
        # Multimodal (image) compression. The compressor backbone must be a VLM (Qwen3.5 is).
        # image_processor_source: where to read preprocessor_config.json from; None -> the
        #   compressor model name (resolved in from_pretrained BEFORE the <ckpt>/compressor
        #   rewrite, since that dir has no preprocessor config). Pixel budget caps the number
        #   of visual tokens per image: 1 token per 32x32 px -> 451584 px ~ 440 tokens.
        # train_vision_lora: which vision-tower LoRA params stay trainable when the
        #   compressor is LoRA-adapted with target_modules="all-linear" (which also wraps
        #   the ViT): "all" (legacy behaviour), "merger" (only the vision->LM projection,
        #   the official Qwen-VL recipe: ViT frozen, merger + LLM trained), "none".
        self.image_processor_source = image_processor_source
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels
        if train_vision_lora not in ("all", "merger", "none"):
            raise ValueError(f"train_vision_lora={train_vision_lora!r} not in ('all','merger','none')")
        self.train_vision_lora = train_vision_lora
        # Online token-level distillation (finetuning): loss = CE + kd_weight * T^2 *
        #   KL(teacher || student) on the answer tokens, teacher = THIS decoder with its LoRA
        #   adapters disabled reading the UNCOMPRESSED documents (raw pages / text). Active only
        #   when kd_weight > 0 AND the batch carries `teacher_input_ids` (collator option
        #   `teacher_inputs=True`); eval / generation / text-only batches are untouched.
        self.kd_weight = float(kd_weight)
        self.kd_temperature = float(kd_temperature)
        # RMSNorm on the connector OUTPUT (the <MEM> vectors the decoder reads): every <MEM> gets the same
        #   length, a learned per-dim gain sets the scale. The gain starts at connector_out_norm_init
        #   (RMS per dim): 0.4 -> L2 ~20 at 2560 dims, the scale trained <MEM>s reach without the norm
        #   (norm probe 174711; word embeddings are ~0.67, and shrinking <MEM>s to that erases the
        #   documents). A checkpoint saved without the norm loads into a model with it (retrofit).
        self.connector_out_norm = bool(connector_out_norm)
        self.connector_out_norm_init = float(connector_out_norm_init)


class ConnectorRMSNorm(nn.Module):
    """RMSNorm with a per-dim gain, computed in fp32 like Qwen's. Registered as `connector.norm` so the
    Trainer's no-weight-decay name rule ("norm") covers the gain."""

    def __init__(self, dim: int, init: float, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.full((dim,), float(init)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return (h * self.weight.float()).to(x.dtype)



def build_bidirectional_4d_mask(attention_mask: torch.Tensor, num_query_tokens: Optional[int] = None) -> torch.Tensor:
    """Boolean 4D attention mask (B, 1, Q, K) where True means 'attend'.

    Every query position may attend to every non-pad key position with no
    causal triangle, i.e. full bidirectional attention over the real tokens.

    transformers>=5 forwards an already-4D ``attention_mask`` to the attention
    backend unchanged (see ``masking_utils.create_causal_mask``: "It can also be
    an already prepared 4D mask, in which case it is returned as-is"), so passing
    this in place of the usual 2D padding mask turns a causal decoder
    bidirectional. Requires attn_implementation 'sdpa' or 'eager' —
    flash_attention_2 rejects 4D masks. Never returns None, so the model's
    causal construction can't re-engage.
    """
    pad = attention_mask.bool()              # (B, K) True = real token
    B, K = pad.shape
    Q = K if num_query_tokens is None else num_query_tokens
    return pad[:, None, None, :].expand(B, 1, Q, K)


_TRAINABLE_KEY_HINTS = ("lora", "token_adapter", "modules_to_save")


def _load_filtered_state(module: nn.Module, path: str, what: str, full: bool = False) -> None:
    """Load a PISCO state dict into a live module. Filtered files (adapters + token_adapter + ...)
    need strict=False (base weights are not in the file), but a saved adapter key that fails to
    land, or an unexpected key, means the module tree drifted (e.g. different PEFT wrapping) and
    training would silently continue from fresh LoRA. Make that loud. `full` (a whole module, saved
    when it is trained without LoRA) tolerates no missing key at all."""
    res = module.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"), strict=False)
    missing = res.missing_keys if full else [k for k in res.missing_keys if any(h in k for h in _TRAINABLE_KEY_HINTS)]
    if res.unexpected_keys or missing:
        raise RuntimeError(
            f"{what}: checkpoint/module mismatch — unexpected {res.unexpected_keys[:5]}, "
            f"missing {'' if full else 'adapter '}keys {missing[:5]} (+{max(0, len(missing) - 5)} more)"
        )


def eos_token_ids(tokenizer) -> List[int]:
    """Stop tokens for generation with a chat decoder: the tokenizer's eos plus the chat
    end-of-turn token when it exists. Some snapshots (Qwen3.5-4B) ship no generation_config.json,
    so the model-level eos is only <|endoftext|> and a chat-finetuned decoder keeps writing a fake
    next turn after <|im_end|> unless it is registered as a stop token here."""
    ids = []
    for t in (tokenizer.eos_token_id,
              tokenizer.convert_tokens_to_ids("<|im_end|>"),
              tokenizer.convert_tokens_to_ids("<|endoftext|>")):
        if isinstance(t, int) and t >= 0 and t != tokenizer.unk_token_id and t not in ids:
            ids.append(t)
    return ids


def has_linear_attention(config) -> bool:
    """True for hybrid backbones (Qwen3.5: GatedDeltaNet layers next to full attention)."""
    for cfg in (config, getattr(config, "text_config", None)):
        if cfg is not None and "linear_attention" in (getattr(cfg, "layer_types", None) or []):
            return True
    return False


_HF_WEIGHT_FILES = ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin")


def _load_full_compressor(model: "PISCO", comp_dir: str) -> bool:
    """A non-LoRA compressor is saved in HF format under compressor/ (save_pretrained). Copy those
    weights into the live module, keeping its wrapping (gradient checkpointing, hooks). Returns False
    when the dir holds no HF weights. from_pretrained does not need this: it builds the compressor
    from that dir directly."""
    if not any(os.path.exists(os.path.join(comp_dir, f)) for f in _HF_WEIGHT_FILES):
        return False
    live = model.compressor
    # Same class as the live module, so the state-dict keys match whatever renaming HF applies on save.
    src = type(live).from_pretrained(comp_dir, torch_dtype=next(live.parameters()).dtype)
    res = live.load_state_dict(src.state_dict(), strict=False)
    del src
    if res.missing_keys or res.unexpected_keys:
        raise RuntimeError(f"compressor: checkpoint/module mismatch — missing {res.missing_keys[:5]}, "
                           f"unexpected {res.unexpected_keys[:5]}")
    return True


def load_trainable_state(model: "PISCO", ckpt_dir: str, *, decoder: bool = True, compressor: bool = True) -> None:
    """Load the trainable parts saved by PISCO.save_pretrained (decoder_state.pt, the compressor:
    compressor/compressor_state.pt for LoRA or HF weights under compressor/ otherwise, connector.pt)
    into an already-built model. Used by from_pretrained and by the Trainer's resume-from-checkpoint
    path (checkpoint-N dirs hold exactly these files)."""
    dec_path = os.path.join(ckpt_dir, "decoder_state.pt")
    comp_dir = os.path.join(ckpt_dir, "compressor")
    comp_path = os.path.join(comp_dir, "compressor_state.pt")
    if decoder and os.path.exists(dec_path):
        _load_filtered_state(model.decoder, dec_path, "decoder", full=not getattr(model.config, "lora_decoder", True))
    comp_loaded = "no"
    if compressor and os.path.exists(comp_path):
        _load_filtered_state(model.compressor, comp_path, "compressor")
        comp_loaded = "lora"
    elif compressor and not getattr(model.config, "lora_compressor", False):
        # Full compressor training: a resumed run must not silently restart from the base weights.
        if not _load_full_compressor(model, comp_dir):
            raise FileNotFoundError(f"no compressor weights under {comp_dir} (full compressor, resume)")
        comp_loaded = "full"
    conn_state = torch.load(os.path.join(ckpt_dir, "connector.pt"), weights_only=True, map_location="cpu")
    if hasattr(model.connector, "norm") and "norm.weight" not in conn_state:
        # Retrofit: a checkpoint trained without the output norm; the gain keeps its init.
        conn_state["norm.weight"] = model.connector.norm.weight.detach().cpu()
        print(f"[pisco] connector.norm absent from {ckpt_dir}: gain initialised to "
              f"{model.config.connector_out_norm_init}")
    conn = model.connector.load_state_dict(conn_state)
    print(f"[pisco] loaded trainable state from {ckpt_dir} (decoder={decoder and os.path.exists(dec_path)}, "
          f"compressor={comp_loaded}, connector={conn})")


def _resolve_torch_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    """Map a string like 'bfloat16' or 'float32' to torch.bfloat16 / torch.float32.
    Accepts None or 'auto' to mean 'let HF decide'."""
    if name is None or name == "auto":
        return None
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"torch_dtype={name!r} is not a valid torch dtype")
    return dtype


def kd_loss_on_tails(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    ignore_ids: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Token-level KL(teacher || student) over the supervised positions of right-aligned
    sequences. All three tensors cover the SAME trailing window of K positions:
    student_logits / teacher_logits (B, K, V) and labels (B, K); position k's logits predict
    labels[:, k+1] (HF shift), so the loss runs on logits[:, :-1] vs labels[:, 1:] != -100.
    Returns (mean KL per supervised token, scaled by temperature^2; number of such tokens).
    `ignore_ids` (e.g. <MEM>, <AE>) are removed from both distributions: the frozen base
    decoder never trained their rows, so the teacher's mass there is noise.
    """
    assert student_logits.shape == teacher_logits.shape, (student_logits.shape, teacher_logits.shape)
    assert labels.shape == student_logits.shape[:2], (labels.shape, student_logits.shape)
    mask = labels[:, 1:] != -100                      # (B, K-1)
    n = int(mask.sum())
    if n == 0:
        return student_logits.sum() * 0.0, 0
    s = student_logits[:, :-1][mask].float()          # (n, V)
    t = teacher_logits[:, :-1][mask].float()
    if ignore_ids:
        idx = torch.tensor([i for i in ignore_ids if 0 <= i < s.shape[-1]], device=s.device, dtype=torch.long)
        if idx.numel():
            s = s.index_fill(-1, idx, float("-inf"))
            t = t.index_fill(-1, idx, float("-inf"))
    T = float(temperature)
    log_p_t = torch.log_softmax(t / T, dim=-1)
    log_p_s = torch.log_softmax(s / T, dim=-1)
    p_t = log_p_t.exp()
    # Ignored ids: log_p_t = log_p_s = -inf there, and 0 * (-inf - -inf) is NaN, so zero those
    # terms explicitly (p_t == 0 anyway). Everything else is finite.
    term = torch.where(torch.isfinite(log_p_t), p_t * (log_p_t - log_p_s), torch.zeros_like(p_t))
    kl = term.sum(-1)                                  # (n,)
    return kl.mean() * (T * T), n


def merge_lora_inplace(model: nn.Module) -> nn.Module:
    """
    Fold LoRA adapters into the base weights for INFERENCE (same outputs up to bf16 rounding, base-model speed).
    PISCO's decoder/compressor carry transformers-integrated PEFT adapters (`PreTrainedModel.add_adapter`): there is
    no `merge_and_unload` / `merge_adapter` on them, so each `LoraLayer` is merged (`layer.merge()`) and then replaced
    by its base layer. Left unmerged, the rank-64 adapter path runs on every linear at every decode step (+60-80 %
    per-token latency on Qwen3.5-4B, probe 2026-10-16).
    TrainableTokens wrappers (the <AE>/<MEM> rows of embed_tokens and the tied lm_head) are left as they are: they
    share one weight, so merging both would add the delta twice, and they cost ~1 ms/token.
    A real `PeftModel` is handled with its own `merge_and_unload`.
    """
    if hasattr(model, "merge_and_unload"):
        return model.merge_and_unload()
    from peft.tuners.lora import LoraLayer
    n = 0
    for module in list(model.modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoraLayer):
                if not child.merged:
                    child.merge()
                setattr(module, child_name, child.get_base_layer())
                n += 1
    print(f"[merge_lora_inplace] merged + unwrapped {n} LoRA layers")
    return model


def _snapshot_rope_deltas(module: nn.Module) -> List[Tuple[nn.Module, object]]:
    """[(submodule, its current `rope_deltas`)] for every submodule carrying that cache."""
    return [(m, m.rope_deltas) for m in module.modules() if hasattr(m, "rope_deltas")]


def _restore_rope_deltas(saved: List[Tuple[nn.Module, object]]) -> None:
    for m, v in saved:
        m.rope_deltas = v


class PISCO(PreTrainedModel):
    config_class = PISCOConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: PISCOConfig):
        super().__init__(config)
        self.decoder_tokenizer = self.create_decoder_tokenizer(config)

        if config.load_decoder:
            self.decoder = self.create_decoder(config)
            print("Base decoder nb parameters", self.decoder.num_parameters())
            print(
                f"Decoder trainable parameters: {self.decoder.num_parameters(only_trainable=True)}"
            )

        decoder_config = AutoConfig.from_pretrained(config.decoder_model_name)
        hidden_size = decoder_config.hidden_size if hasattr(decoder_config, "hidden_size") else decoder_config.text_config.hidden_size

        self.compressor_tokenizer = self.create_compressor_tokenizer(config)
        self.compressor, self.connector = self.create_compressor_and_connector(
            config, decoder_hidden_dim=hidden_size
        )

        print("Base compressor nb parameters", self.compressor.num_parameters())

        if config.bidirectional and has_linear_attention(self.compressor.config):
            # The 4D mask only reaches the full-attention layers. Qwen3.5's GatedDeltaNet layers
            # stay causal and, given a 4D mask, skip their padding zeroing
            # (apply_mask_to_padding_states expects a 2D mask), so <MEM>s would depend on the
            # batch's padding. Bidirectional compression needs a pure-attention backbone.
            raise NotImplementedError(
                f"bidirectional=True is not supported for the hybrid backbone {config.compressor_model_name} "
                "(linear-attention layers): use a full-attention compressor (e.g. Qwen2.5 / Qwen3)."
            )

        # Image processor, only when the compressor backbone has a vision tower. Failure to
        # build one must never break a text-only run.
        self.image_processor = None
        self.vision_token_ids = None
        if any(name.split(".")[-1] == "visual" for name, _ in self.compressor.named_modules()):
            try:
                from pisco.mm_utils import load_image_processor, get_vision_token_ids
                self.image_processor = load_image_processor(
                    config.image_processor_source or config.compressor_model_name,
                    min_pixels=config.image_min_pixels,
                    max_pixels=config.image_max_pixels,
                )
                self.vision_token_ids = get_vision_token_ids(self.compressor_tokenizer)
            except Exception as e:
                warnings.warn(f"compressor has a vision tower but no image processor could be built: {e}")
                self.image_processor = None


        print(
            f"Compressor trainable parameters: {self.compressor.num_parameters(only_trainable=True)}"
        )
        print(f"Total trainable parameters: {self.num_parameters(only_trainable=True)}")

        # Option 2: base loaded in frozen_base_dtype (e.g. fp16), but the trainable
        # LoRA adapters inherited that dtype too. fp16 AMP's grad-scaler rejects fp16
        # trainable params ("Attempting to unscale FP16 gradients"), so promote every
        # trainable param to the fp32 master dtype. Connector is already torch_dtype.
        if config.frozen_base_dtype is not None:
            master_dtype = _resolve_torch_dtype(config.torch_dtype)
            n_cast = 0
            for p in self.parameters():
                if p.requires_grad and p.dtype != master_dtype:
                    p.data = p.data.to(master_dtype)
                    n_cast += 1
            print(f"[option2] base in {config.frozen_base_dtype}; promoted "
                  f"{n_cast} trainable tensors to {config.torch_dtype}")

        # other settings
        self.generation_top_k = 1
        self.compr_rate = config.compr_rate


    def create_decoder(self, config: PISCOConfig) -> PreTrainedModel:
        """
        Loads the base decoder and optionally loads a PEFT adapter from decoder_adapter_path.
        """
        adapter_source = config.decoder_adapter_path
        has_adapter_checkpoint = False
        if adapter_source is not None:
            adapter_config_path = os.path.join(adapter_source, "adapter_config.json")
            has_adapter_checkpoint = os.path.isdir(adapter_source) and os.path.exists(
                adapter_config_path
            )
            if not has_adapter_checkpoint:
                raise FileNotFoundError(
                    f"decoder_adapter_path is set to {adapter_source}, but adapter_config.json was not found."
                )
            print(f"Loading decoder adapter from {adapter_source}")

        ### Load using AutoModelForImageTextToText if possible, otherwise fallback to AutoModelForCausalLM
        ### It's important that when loading the pisco adapter, we use the same path as during piso training.
        # Frozen base loads in frozen_base_dtype when set (option 2), else torch_dtype.
        dtype = _resolve_torch_dtype(config.frozen_base_dtype or config.torch_dtype)
        try:
            decoder = cast(
                PreTrainedModel,
                AutoModelForImageTextToText.from_pretrained(
                    config.decoder_model_name,
                    attn_implementation=config.attn_implementation,
                    torch_dtype=dtype,
                ),
            )
        except Exception as e:
            print(f"Error loading decoder: {e}")
            decoder = cast(
                PreTrainedModel,
                AutoModelForCausalLM.from_pretrained(
                    config.decoder_model_name,
                    attn_implementation=config.attn_implementation,
                    torch_dtype=dtype,
                ),
            )
        decoder.resize_token_embeddings(len(self.decoder_tokenizer))


        if config.freeze_decoder:
            print("Freezing decoder")
            for param in decoder.parameters():
                param.requires_grad = False

        elif has_adapter_checkpoint and adapter_source is not None:
            decoder = PeftModel.from_pretrained(
                decoder, 
                adapter_source, 
                is_trainable=config.lora_decoder
            )
        elif config.lora_decoder:
            print("Creating fresh decoder LoRA adapter")
            peft_config = self.get_peft_config(lora_r=config.lora_r_decoder)
            decoder.add_adapter(peft_config)

        if config.decoder_gradient_checkpointing:
            print("Activating gradient checkpointing on decoder")
            decoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}  # optional
            )

        return decoder

    def create_decoder_tokenizer(self, config: PISCOConfig) -> "TokenizersBackend":

        decoder_tokenizer = cast("TokenizersBackend", AutoTokenizer.from_pretrained(
            config.decoder_model_name, padding_side="left", truncation_side="right"
        ))

        decoder_tokenizer.mem_token = "<MEM>"
        decoder_tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<MEM>", "<AE>"]}
        )

        decoder_tokenizer.mem_token_id = decoder_tokenizer.convert_tokens_to_ids(
            decoder_tokenizer.mem_token
        )

        decoder_tokenizer.ae_token = "<AE>"  # token for autoencoding on decoder side
        decoder_tokenizer.ae_token_id = decoder_tokenizer.convert_tokens_to_ids(
            decoder_tokenizer.ae_token
        )

        decoder_tokenizer.bos_token = decoder_tokenizer.bos_token or ""

        assert decoder_tokenizer.eos_token is not None

        # if pad token exists then use pad token, othrwise bos token
        if decoder_tokenizer.pad_token_id is None:
            decoder_tokenizer.pad_token_id = decoder_tokenizer.bos_token_id

        print("Decoder Pad token", decoder_tokenizer.pad_token)

        return decoder_tokenizer

    def create_compressor_and_connector(
            self, 
            config: PISCOConfig, 
            decoder_hidden_dim: int
        ) -> PreTrainedModel:

        # if not adapter path  then load compressor_model_name +'compressor'
        # else load  compressor_model_name   then load adapter

        has_adapter_checkpoint = False
        if  config.compressor_adapter_path is not None:
            adapter_config_path = os.path.join( config.compressor_adapter_path, "adapter_config.json")
            has_adapter_checkpoint = os.path.isdir( config.compressor_adapter_path) and os.path.exists(
                adapter_config_path
            )
            if not has_adapter_checkpoint:
                raise FileNotFoundError(
                    f"compressor_adapter_path is set to {config.compressor_adapter_path}, but adapter_config.json was not found."
                )
            print(f"Loading compressor adapter from {config.compressor_adapter_path}")



        # load model
        # if lora: point to base model
        # else path/compressor
        # Frozen base loads in frozen_base_dtype when set (option 2), else torch_dtype.
        dtype = _resolve_torch_dtype(config.frozen_base_dtype or config.torch_dtype)
        try:
            compressor = AutoModelForImageTextToText.from_pretrained(config.compressor_model_name, attn_implementation=config.attn_implementation, torch_dtype=dtype)

        except Exception:
            compressor = AutoModelForCausalLM.from_pretrained(config.compressor_model_name, attn_implementation=config.attn_implementation, torch_dtype=dtype)

        compressor.resize_token_embeddings(len(self.compressor_tokenizer))
        
        # load adapter 
        if has_adapter_checkpoint: # and adapter_source is not None:
            print(f"Loading compressor adapter from {config.compressor_adapter_path}")
            compressor = PeftModel.from_pretrained(
                compressor, 
                config.compressor_adapter_path, 
                is_trainable=config.lora_compressor
            )
        # create new adapter
        elif config.lora_compressor:
            print("Creating fresh compressor LoRA adapter")
            # peft_config = self.get_peft_config(lora_r=config.lora_r_compressor)
            peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=config.lora_r_compressor,
            lora_alpha=2 * config.lora_r_compressor,
            target_modules="all-linear",
            lora_dropout=0.1,
            trainable_token_indices=[self.compressor_tokenizer.mem_token_id]
            )
            compressor.add_adapter(peft_config)



        if config.compressor_gradient_checkpointing:
            print("Activating gradient checkpointing on compressor")
            compressor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}  # optional
            )

        # Vision-tower LoRA policy. target_modules="all-linear" also wraps the ViT blocks and
        # the merger of a VLM backbone; existing checkpoints carry those adapters with
        # lora_B == 0 (never trained), so freezing is a pure requires_grad decision.
        if config.train_vision_lora != "all":
            n_frozen, n_kept = 0, 0
            for name, p in compressor.named_parameters():
                if ".visual." not in name or not p.requires_grad:
                    continue
                keep = config.train_vision_lora == "merger" and ".visual.merger." in name
                if keep:
                    n_kept += 1
                else:
                    p.requires_grad = False
                    n_frozen += 1
            print(f"[mm] train_vision_lora={config.train_vision_lora}: "
                  f"froze {n_frozen} vision tensors, kept {n_kept} trainable")

        hidden_size = compressor.config.hidden_size if hasattr(compressor.config, "hidden_size") else compressor.config.text_config.hidden_size

        connector = nn.Sequential(
            nn.Linear(hidden_size, config.compressor_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.compressor_mlp_hidden_dim, decoder_hidden_dim),
        )
        if getattr(config, "connector_out_norm", False):
            # Named (not index 3): keeps the legacy 0.*/2.* keys and keeps the gain out of weight decay.
            connector.add_module("norm", ConnectorRMSNorm(decoder_hidden_dim, config.connector_out_norm_init))

        # Cast connector to match the model dtype so compressor hidden states pass through without fp32 promotion.
        connector_dtype = _resolve_torch_dtype(config.torch_dtype)
        if connector_dtype is not None:
            connector.to(connector_dtype)

        return compressor, connector

    def create_compressor_tokenizer(self, config: PISCOConfig) -> "TokenizersBackend":
        compressor_tokenizer = cast(
            "TokenizersBackend",
            AutoTokenizer.from_pretrained(
                config.compressor_model_name, padding_side="left"
            ),
        )

        compressor_tokenizer.mem_token = "<MEM>"
        compressor_tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<MEM>"]}
        )

        compressor_tokenizer.mem_token_id = compressor_tokenizer.convert_tokens_to_ids(
            compressor_tokenizer.mem_token
        )

        # if pad token exists then use pad token, othrwise bos token
        if compressor_tokenizer.pad_token_id is None:
            compressor_tokenizer.pad_token_id = compressor_tokenizer.bos_token_id

        print("Compressor Pad token", compressor_tokenizer.pad_token)

        return compressor_tokenizer

    def compress(
        self,
        input_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.LongTensor],
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Compresses.
        It returns a list of embeddings, one for each input_ids,
        they may have different lengths

        Multimodal: when `pixel_values` is given (with `image_grid_thw` and
        `mm_token_type_ids`, see pisco.mm_utils.collate_compressor), the VLM compressor
        scatters the vision-tower output into the `<|image_pad|>` slots of `input_ids`
        before running the language model; the `<MEM>` read-out is unchanged. Text-only
        calls follow exactly the legacy code path.
        """
        if input_ids is None:
            raise ValueError("`input_ids` must not be None for compression.")

        extra = {}
        if pixel_values is not None:
            if image_grid_thw is None or mm_token_type_ids is None:
                raise ValueError("pixel_values requires image_grid_thw and mm_token_type_ids")
            if getattr(self.config, "bidirectional", False):
                raise NotImplementedError(
                    "bidirectional compressor + images: the 4D mask breaks Qwen's get_rope_index"
                )
            image_token_id = int(self.compressor.config.image_token_id)
            n_pad = int((input_ids == image_token_id).sum())
            merge = int(self.compressor.config.vision_config.spatial_merge_size)
            n_feat = int((image_grid_thw.prod(-1) // (merge ** 2)).sum())
            assert n_pad == n_feat, (
                f"{n_pad} <|image_pad|> tokens but image_grid_thw implies {n_feat} visual tokens "
                f"(an image span was probably chunked or truncated)"
            )
            # PeftModelForCausalLM.forward forwards unknown kwargs to the wrapped
            # Qwen3_5ForConditionalGeneration.forward, which accepts exactly these names.
            # If a future PEFT stops doing so, call self.compressor.base_model.model(...).
            # No dtype cast here: Qwen's get_image_features does pixel_values.type(visual.dtype).
            extra = dict(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
            )

        # Bidirectional: replace the 2D padding mask with a full 4D mask so the
        # compressor attends across the whole document (text tokens included,
        # not just the trailing <MEM> tokens). attn_implementation must be
        # 'sdpa'/'eager' for this to be honored.
        compr_attention_mask = attention_mask
        if getattr(self.config, "bidirectional", False) and attention_mask is not None:
            compr_attention_mask = build_bidirectional_4d_mask(attention_mask)

        # Only hidden_states[-1] is read: skip the full-vocabulary logits (B x T x ~250k) and
        # the KV / recurrent cache, which the LM head would otherwise materialise for nothing.
        last_hidden_states: torch.Tensor = self.compressor(
            input_ids=input_ids,
            attention_mask=compr_attention_mask,
            output_hidden_states=True,
            logits_to_keep=1,
            use_cache=False,
            **extra,
        ).hidden_states[
            -1
        ]  # shape is B, T, hc

        mask = input_ids == self.compressor_tokenizer.mem_token_id

        hidden_states = [
            last_hidden_states[i, mask[i], :] for i in range(last_hidden_states.shape[0])
        ]  # B-length list of (l_i, hc) shapes

        all_hidden_states = torch.cat(hidden_states, 0)  # (sum l_i, hc) shape
        embeddings = self.connector(all_hidden_states)

        return embeddings  # (sum l_i, hd) shape

    def get_peft_config(self, lora_r: int) -> LoraConfig:
        """
        Builds the peft config
        """
        return LoraConfig(
            task_type="CAUSAL_LM",
            r=lora_r,
            lora_alpha=2 * lora_r,
            target_modules="all-linear",
            lora_dropout=0.1,
            trainable_token_indices=[self.decoder_tokenizer.ae_token_id], # For pre-training
        )

    def replace_embeddings(self, compressed_embs: torch.Tensor, dec_input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Replace memory tokens in the decoder input with the compressed embeddings
        This assumes (and checks) that there are as many elements compressed_embs as there are mem tokens in dec_input_ids
        """
        dec_embeds = self.decoder.get_input_embeddings()(dec_input_ids)
        B, L, H = dec_embeds.shape

        # Locate mem_token positions
        mem_mask = (dec_input_ids == self.decoder_tokenizer.mem_token_id)

        num_mem_tokens = mem_mask.sum().item()
        assert num_mem_tokens == compressed_embs.shape[0], (
            f"Mismatch: {num_mem_tokens} mem tokens but "
            f"{compressed_embs.shape[0]} compressed embeddings"
        )

        # Replace embeddings in order
        mem_indices = mem_mask.view(-1).nonzero(as_tuple=False).squeeze(1)

        # Flatten embeddings for easy indexing (view is safe: get_input_embeddings returns contiguous).
        dec_embeds_flat = dec_embeds.view(-1, H)
        # Match dtype: under fp16/bf16 autocast the connector output and the embedding
        # lookup can differ in dtype; index_copy requires them equal. No-op when matched.
        compressed_embs = compressed_embs.to(dec_embeds_flat.dtype)
        # Use out-of-place index_copy to avoid in-place ops on autograd views.
        dec_embeds_flat = dec_embeds_flat.index_copy(0, mem_indices, compressed_embs)

        # Restore shape
        dec_embeds = dec_embeds_flat.view(B, L, H)

        return dec_embeds

    def forward(
        self,
        compressor_input_ids: torch.LongTensor,
        compressor_attention_mask: torch.LongTensor,
        decoder_input_ids: torch.LongTensor,
        decoder_attention_mask: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.LongTensor] = None,
        teacher_input_ids: Optional[torch.LongTensor] = None,
        teacher_attention_mask: Optional[torch.LongTensor] = None,
        teacher_pixel_values: Optional[torch.Tensor] = None,
        teacher_image_grid_thw: Optional[torch.LongTensor] = None,
        teacher_mm_token_type_ids: Optional[torch.LongTensor] = None,
        teacher_kd_rows: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # NOTE: named params on purpose (no **kwargs): HF Trainer injects num_items_in_batch
        # as soon as forward accepts **kwargs.
        # teacher_*: the UNCOMPRESSED prompt (same chat template, raw pages / text instead of
        # <MEM>s, same answer suffix) for online distillation; see PISCOConfig.kd_weight.
        # Compression
        embeddings = self.compress(
            compressor_input_ids,
            compressor_attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )

        # Inserting compressed reps into the decoder inputs:
        dec_inputs_embeds = self.replace_embeddings(embeddings, decoder_input_ids)

        decoder_outputs = self.decoder(
            inputs_embeds=dec_inputs_embeds,
            attention_mask=decoder_attention_mask,
            labels=labels,
        )
        out = {"loss": decoder_outputs.loss, "logits": decoder_outputs.logits}

        kd_w = float(getattr(self.config, "kd_weight", 0.0) or 0.0)
        if kd_w > 0 and self.training and teacher_input_ids is not None and labels is not None:
            # Pretraining batches mix chat rows (with a teacher prompt) and AE / text rows without one:
            # `teacher_kd_rows` (B,) marks the former, and the teacher tensors hold ONLY those rows
            # (the student logits / labels are sliced to match). A legacy batch whose teacher tensors
            # hold all B rows (dummy prompts on the others) is still accepted: the KL is masked instead.
            s_logits, kd_labels = decoder_outputs.logits, labels
            if teacher_kd_rows is not None:
                rows = teacher_kd_rows.to(labels.device).bool()
                if teacher_input_ids.shape[0] == labels.shape[0]:
                    kd_labels = labels.masked_fill(~rows.unsqueeze(1), -100)
                else:
                    s_logits, kd_labels = decoder_outputs.logits[rows], labels[rows]
            if kd_labels.shape[0] == 0:
                kd = decoder_outputs.logits.sum() * 0.0
            else:
                kd, _ = self.kd_loss(
                    s_logits, kd_labels,
                    teacher_input_ids, teacher_attention_mask,
                    pixel_values=teacher_pixel_values, image_grid_thw=teacher_image_grid_thw,
                    mm_token_type_ids=teacher_mm_token_type_ids,
                )
            out["ce_loss"] = decoder_outputs.loss.detach()
            out["kd_loss"] = kd.detach()
            out["loss"] = decoder_outputs.loss + kd_w * kd
        return out

    @contextlib.contextmanager
    def adapters_disabled(self, module: nn.Module):
        """Run `module` as its frozen base model (LoRA + trainable-token layers off).
        Handles both a PeftModel (legacy adapter-dir checkpoints) and transformers'
        integrated PEFT (`add_adapter`, the new-format path)."""
        if isinstance(module, PeftModel):
            with module.disable_adapter():
                yield
            return
        if getattr(module, "_hf_peft_config_loaded", False):
            module.disable_adapters()
            try:
                yield
            finally:
                module.enable_adapters()
            return
        yield   # no adapters (full finetune / frozen decoder): base == current weights

    def kd_loss(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_input_ids: torch.LongTensor,
        teacher_attention_mask: torch.LongTensor,
        *,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Online distillation term. Teacher = the decoder backbone with adapters disabled
        reading the uncompressed prompt (no grad). Both prompts are LEFT padded and end with
        the identical `assistant: <answer><|im_end|>\n` suffix, so the answer occupies the
        last K columns of both; only those K logits are computed on the teacher side
        (`logits_to_keep`) to avoid materialising a (B, ~2000, 250k) tensor.
        """
        n_ans = int((labels != -100).sum(1).max())
        K = min(n_ans + 1, student_logits.shape[1], teacher_input_ids.shape[1])
        if K < 2:
            return student_logits.sum() * 0.0, 0
        # Guard the alignment assumption cheaply: supervised ids must coincide on the tails.
        lab_tail = labels[:, -K:]
        sup = lab_tail != -100
        assert torch.equal(teacher_input_ids[:, -K:][sup], lab_tail[sup]), (
            "teacher/student answer tails differ: the teacher prompt must end with the same "
            "answer suffix as the decoder prompt (same tokenizer, same chat template)")
        extra = {}
        if pixel_values is not None:
            if image_grid_thw is None or mm_token_type_ids is None:
                raise ValueError("teacher_pixel_values requires teacher_image_grid_thw and teacher_mm_token_type_ids")
            extra = dict(pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                         mm_token_type_ids=mm_token_type_ids)
        # Qwen3.5 caches `rope_deltas` on its inner model when it computes M-RoPE from
        # input_ids + image_grid_thw, and REUSES the cache in any later forward that gets only
        # inputs_embeds (our student path). Snapshot and restore it around the teacher pass,
        # otherwise every student step after the first runs with the teacher's position offsets.
        saved = _snapshot_rope_deltas(self.decoder)
        try:
            with torch.no_grad(), self.adapters_disabled(self.decoder):
                t_logits = self.decoder(
                    input_ids=teacher_input_ids,
                    attention_mask=teacher_attention_mask,
                    logits_to_keep=K,
                    use_cache=False,
                    **extra,
                ).logits
        finally:
            _restore_rope_deltas(saved)
        ignore = [self.decoder_tokenizer.mem_token_id, self.decoder_tokenizer.ae_token_id]
        return kd_loss_on_tails(
            student_logits[:, -K:], t_logits, lab_tail,
            temperature=float(getattr(self.config, "kd_temperature", 1.0) or 1.0),
            ignore_ids=ignore,
        )

    def generate(
        self,
        model_input: Dict[str, torch.LongTensor],
        max_new_tokens: int = 128,
        return_ids: bool = False,
        **gen_kwargs,
    ) -> List[str]:
        """
        Greedy by default. Pass HF generation kwargs (do_sample / temperature / top_p /
        num_return_sequences) to sample instead -- needed for rejection sampling and RL.

        Two properties of the inputs_embeds path that callers rely on:
          * with `inputs_embeds` and no `input_ids`, generate() returns ONLY the newly
            generated tokens, so there is no prompt to slice off;
          * with num_return_sequences=G the rows come back grouped per prompt, i.e.
            [prompt0 x G, prompt1 x G, ...], so a .view(B, G) recovers the groups.
        """

        (
            compressor_input_ids,
            compressor_attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
        ) = (
            model_input["compressor_input_ids"],
            model_input["compressor_attention_mask"],
            model_input["decoder_input_ids"],
            model_input["decoder_attention_mask"],
        )

        embeddings = self.compress(
            compressor_input_ids,
            compressor_attention_mask,
            pixel_values=model_input.get("pixel_values"),
            image_grid_thw=model_input.get("image_grid_thw"),
            mm_token_type_ids=model_input.get("mm_token_type_ids"),
        )

        decoder_inputs_embeds = self.replace_embeddings(embeddings, decoder_input_ids)

        gen_kwargs.setdefault("do_sample", False)
        if not gen_kwargs["do_sample"]:
            # HF warns when top_p is set but unused under greedy decoding.
            gen_kwargs.setdefault("top_p", None)
        gen_kwargs.setdefault("eos_token_id", self.decoder_eos_ids())

        output_ids = self.decoder.generate(
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )

        decoded = self.decoder_tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )

        if return_ids:
            # Callers that must distinguish a naturally-finished sample from one cut
            # off at max_new_tokens need the raw ids (look for eos), not the text.
            return decoded, output_ids

        return decoded

    def decoder_eos_ids(self) -> List[int]:
        """Stop tokens for decoder generation (see eos_token_ids)."""
        return eos_token_ids(self.decoder_tokenizer)

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save the trainable parts of the model.

        Decoder (and LoRA-adapted compressor) are saved as a filtered state_dict
        via torch.save rather than PEFT.save_pretrained. Reason: transformers
        unconditionally runs remove_tied_weights_from_state_dict, which raises on
        the shared token_adapter tensors that PEFT inserts when lm_head and
        embed_tokens are tied. torch.save preserves shared tensors via pickle
        reference tracking and skips that check entirely.
        """
        os.makedirs(save_directory, exist_ok=True)

        if not self.config.freeze_decoder:
            # LoRA decoder: adapters + token_adapter only. Full finetune: every tensor, else the
            # checkpoint would hold an empty file and reload as the base decoder.
            decoder_state = {
                k: v for k, v in self.decoder.state_dict().items()
                if not self.config.lora_decoder or any(hint in k for hint in _TRAINABLE_KEY_HINTS)
            }
            torch.save(decoder_state, os.path.join(save_directory, "decoder_state.pt"))

        compressor_dir = os.path.join(save_directory, "compressor")
        if self.config.lora_compressor:
            os.makedirs(compressor_dir, exist_ok=True)
            compressor_state = {
                k: v for k, v in self.compressor.state_dict().items()
                if any(hint in k for hint in _TRAINABLE_KEY_HINTS)
            }
            torch.save(compressor_state, os.path.join(compressor_dir, "compressor_state.pt"))
        else:
            # Plain HF compressor: tied embeddings are handled by HF's own save.
            self.compressor.save_pretrained(compressor_dir)

        torch.save(
            self.connector.state_dict(), os.path.join(save_directory, "connector.pt")
        )
        self.compressor_tokenizer.save_pretrained(compressor_dir)
        if getattr(self, "image_processor", None) is not None:
            # Makes the checkpoint self-contained for images (preprocessor_config.json
            # next to the compressor tokenizer); from_pretrained picks it up from there.
            os.makedirs(compressor_dir, exist_ok=True)
            self.image_processor.save_pretrained(compressor_dir)
        self.decoder_tokenizer.save_pretrained(save_directory)
        self.config.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ) -> "PISCO":
        """
        Loading: handles two checkpoint formats.
        - New (filtered state_dict): `decoder_state.pt` at the root, optional
          `compressor/compressor_state.pt`. Decoder/compressor are built fresh
          (with new LoRA adapters) and the saved state is layered in via
          load_state_dict(strict=False).
        - Legacy (PEFT adapter dir): `adapter_config.json` + `adapter_model.safetensors`
          at the root and/or under `compressor/`. Loaded via PeftModel.from_pretrained
          inside create_decoder / create_compressor_and_connector.

        freeze_decoder / load_decoder are read from the saved PISCOConfig, not from kwargs.

        Any kwarg that names a PISCOConfig field overrides the saved value BEFORE the model is
        built — needed e.g. to finetune an A100/bf16 checkpoint on a V100 by passing
        frozen_base_dtype="float16" + torch_dtype="float32" (base dtype is decided in __init__).
        """
        config = PISCOConfig.from_pretrained(pretrained_model_name_or_path)
        for _k in list(kwargs):
            if hasattr(config, _k):
                setattr(config, _k, kwargs.pop(_k))

        decoder_state_path = os.path.join(pretrained_model_name_or_path, "decoder_state.pt")
        compressor_dir = os.path.join(pretrained_model_name_or_path, "compressor")
        compressor_state_path = os.path.join(compressor_dir, "compressor_state.pt")

        new_format_decoder = os.path.exists(decoder_state_path)
        new_format_compressor = os.path.exists(compressor_state_path)

        if not config.freeze_decoder and config.lora_decoder and not new_format_decoder:
            # Legacy path: PEFT adapter dir at the checkpoint root.
            config.decoder_adapter_path = pretrained_model_name_or_path

        # Image processor source: a checkpoint saved by a multimodal-aware PISCO carries
        # preprocessor_config.json under compressor/; older ones don't, so fall back to the
        # backbone name. Must be decided BEFORE compressor_model_name is rewritten below.
        if os.path.exists(os.path.join(compressor_dir, "preprocessor_config.json")):
            config.image_processor_source = compressor_dir      # self-contained checkpoint wins
        elif getattr(config, "image_processor_source", None) is None:
            config.image_processor_source = config.compressor_model_name

        if config.lora_compressor and not new_format_compressor:
            # Legacy path: PEFT adapter dir under compressor/.
            config.compressor_adapter_path = compressor_dir
        elif not config.lora_compressor:
            config.compressor_model_name = compressor_dir

        model = cls(config)

        load_trainable_state(model, pretrained_model_name_or_path,
                             decoder=new_format_decoder and not config.freeze_decoder,
                             compressor=new_format_compressor and config.lora_compressor)
        connector_dtype = _resolve_torch_dtype(config.torch_dtype)
        if connector_dtype is not None:
            model.connector.to(connector_dtype)
        return model
