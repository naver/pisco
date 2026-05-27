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
from typing import List, Optional, Dict
from typing import cast, TYPE_CHECKING

import torch
from torch import nn
from peft import LoraConfig, PeftModel


@contextlib.contextmanager
def _untied_for_save(module):
    """Break PEFT shared tensors so save_pretrained doesn't crash on tied
    Qwen3/Qwen2.5 embeddings, then restore the aliasing on exit.

    When lm_head and embed_tokens are tied and PEFT's trainable_token_indices
    is set, both get a TrainableTokensWrapper whose token_adapter shares
    `base_layer.weight` and `trainable_tokens_delta.default`. transformers'
    remove_tied_weights_from_state_dict raises on these undeclared aliases.
    """
    lm_head = getattr(module, "lm_head", None)
    inner = getattr(module, "model", None)
    embed_tokens = getattr(inner, "embed_tokens", None) if inner is not None else None
    restore = []
    if lm_head is None or embed_tokens is None:
        yield
        return

    lh_ta = getattr(lm_head, "token_adapter", None)
    et_ta = getattr(embed_tokens, "token_adapter", None)
    has_token_adapter = lh_ta is not None and et_ta is not None

    # Outer .weight — only meaningful when there's no token_adapter (which
    # otherwise covers the same tensor at base_layer.weight). On wrappers like
    # PEFT's TrainableTokensWrapper, `weight` is a property and setattr raises.
    if not has_token_adapter:
        lh_w = getattr(lm_head, "weight", None)
        et_w = getattr(embed_tokens, "weight", None)
        if (
            lh_w is not None
            and et_w is not None
            and isinstance(lh_w, torch.nn.Parameter)
            and lh_w.data_ptr() == et_w.data_ptr()
        ):
            lm_head.weight = torch.nn.Parameter(
                lh_w.detach().clone(), requires_grad=lh_w.requires_grad
            )
            restore.append((lm_head, "weight", lh_w))

    if has_token_adapter:
        # token_adapter.base_layer.weight
        lh_base = getattr(lh_ta, "base_layer", None)
        et_base = getattr(et_ta, "base_layer", None)
        if (
            lh_base is not None
            and et_base is not None
            and lh_base.weight.data_ptr() == et_base.weight.data_ptr()
        ):
            original = lh_base.weight
            lh_base.weight = torch.nn.Parameter(
                original.detach().clone(), requires_grad=original.requires_grad
            )
            restore.append((lh_base, "weight", original))

        # trainable_tokens_delta is a ParameterDict; swap in an independent copy.
        lh_delta = getattr(lh_ta, "trainable_tokens_delta", None)
        et_delta = getattr(et_ta, "trainable_tokens_delta", None)
        if isinstance(lh_delta, torch.nn.ParameterDict) and isinstance(
            et_delta, torch.nn.ParameterDict
        ):
            shares_any = any(
                k in et_delta and lh_delta[k].data_ptr() == et_delta[k].data_ptr()
                for k in lh_delta.keys()
            )
            if shares_any:
                new_dict = torch.nn.ParameterDict({
                    k: torch.nn.Parameter(
                        lh_delta[k].detach().clone(),
                        requires_grad=lh_delta[k].requires_grad,
                    )
                    for k in lh_delta.keys()
                })
                lh_ta._modules["trainable_tokens_delta"] = new_dict
                restore.append((lh_ta, "trainable_tokens_delta", lh_delta))

    try:
        yield
    finally:
        for parent, key, original in restore:
            if key == "trainable_tokens_delta":
                parent._modules[key] = original
            else:
                setattr(parent, key, original)

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
        # Stored as string so the config serializes to JSON cleanly.
        # Resolved to a torch.dtype via _resolve_torch_dtype() at load time.
        self.torch_dtype = torch_dtype




def _resolve_torch_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    """Map a string like 'bfloat16' or 'float32' to torch.bfloat16 / torch.float32.
    Accepts None or 'auto' to mean 'let HF decide'."""
    if name is None or name == "auto":
        return None
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"torch_dtype={name!r} is not a valid torch dtype")
    return dtype


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


        print(
            f"Compressor trainable parameters: {self.compressor.num_parameters(only_trainable=True)}"
        )
        print(f"Total trainable parameters: {self.num_parameters(only_trainable=True)}")

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
        dtype = _resolve_torch_dtype(config.torch_dtype)
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
        dtype = _resolve_torch_dtype(config.torch_dtype)
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

        hidden_size = compressor.config.hidden_size if hasattr(compressor.config, "hidden_size") else compressor.config.text_config.hidden_size

        connector = nn.Sequential(
            nn.Linear(hidden_size, config.compressor_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.compressor_mlp_hidden_dim, decoder_hidden_dim),
        )

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
    ) -> torch.Tensor:
        """
        Compresses.
        It returns a list of embeddings, one for each input_ids,
        they may have different lengths
        """
        if input_ids is None:
            raise ValueError("`input_ids` must not be None for compression.")

        last_hidden_states: torch.Tensor = self.compressor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
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
    ) -> Dict[str, torch.Tensor]:
        # Compression
        embeddings = self.compress(compressor_input_ids, compressor_attention_mask)

        # Inserting compressed reps into the decoder inputs:
        dec_inputs_embeds = self.replace_embeddings(embeddings, decoder_input_ids)

        decoder_outputs = self.decoder(
            inputs_embeds=dec_inputs_embeds,
            attention_mask=decoder_attention_mask,
            labels=labels,
        )

        return {"loss": decoder_outputs.loss, "logits": decoder_outputs.logits}

    def generate(self, model_input: Dict[str, torch.LongTensor], max_new_tokens: int = 128) -> List[str]:

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

        embeddings = self.compress(compressor_input_ids, compressor_attention_mask)

        decoder_inputs_embeds = self.replace_embeddings(embeddings, decoder_input_ids)

        output_ids = self.decoder.generate(
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            do_sample=False,
            top_p=None,
            max_new_tokens=max_new_tokens,
        )

        decoded = self.decoder_tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )

        return decoded

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save only the LoRA adapters and their configurations.
        Use PEFT standard artifacts for the decoder adapter.
        """
        if not self.config.freeze_decoder:
            with _untied_for_save(self.decoder):
                self.decoder.save_pretrained(save_directory, safe_serialization=True)
        with _untied_for_save(self.compressor):
            self.compressor.save_pretrained(os.path.join(save_directory, "compressor"))
        torch.save(
            self.connector.state_dict(), os.path.join(save_directory, "connector.pt")
        )
        self.compressor_tokenizer.save_pretrained(
            os.path.join(save_directory, "compressor")
        )
        # useless??  tokenizer are always re-created from the model name
        self.decoder_tokenizer.save_pretrained(save_directory
            #os.path.join(save_directory, "decoder_tokenizer")
        )
        # real pisco config!!!
        self.config.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ) -> "PISCO":
        """
        Loading: to take care of checkpoints containing only lora and not base model.
        freeze_decoder / load_decoder are read from the saved PISCOConfig, not from kwargs.
        """
        # Load the configuration
        config = PISCOConfig.from_pretrained(pretrained_model_name_or_path)

        if not config.freeze_decoder:
            if config.lora_decoder:
                config.decoder_adapter_path = pretrained_model_name_or_path
            #config.decoder_model_name = pretrained_model_name_or_path

        #config.compressor_model_name = os.path.join(pretrained_model_name_or_path, "compressor")
        if config.lora_compressor:
            config.compressor_adapter_path =  os.path.join(pretrained_model_name_or_path, "compressor")
        else:
            config.compressor_model_name = os.path.join(pretrained_model_name_or_path, "compressor")
            
        print(config)
        model = cls(config)


        model.connector.load_state_dict(
            torch.load(os.path.join(pretrained_model_name_or_path, "connector.pt"), weights_only=True)
        )
        connector_dtype = _resolve_torch_dtype(config.torch_dtype)
        if connector_dtype is not None:
            model.connector.to(connector_dtype)
        print(model)
        return model
