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

import os
from typing import List, Optional, Dict
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
        Save the trainable parts of the model.

        Decoder (and LoRA-adapted compressor) are saved as a filtered state_dict
        via torch.save rather than PEFT.save_pretrained. Reason: transformers
        unconditionally runs remove_tied_weights_from_state_dict, which raises on
        the shared token_adapter tensors that PEFT inserts when lm_head and
        embed_tokens are tied. torch.save preserves shared tensors via pickle
        reference tracking and skips that check entirely.
        """
        _TRAINABLE_KEY_HINTS = ("lora", "token_adapter", "modules_to_save")
        os.makedirs(save_directory, exist_ok=True)

        if not self.config.freeze_decoder:
            decoder_state = {
                k: v for k, v in self.decoder.state_dict().items()
                if any(hint in k for hint in _TRAINABLE_KEY_HINTS)
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
        """
        config = PISCOConfig.from_pretrained(pretrained_model_name_or_path)

        decoder_state_path = os.path.join(pretrained_model_name_or_path, "decoder_state.pt")
        compressor_dir = os.path.join(pretrained_model_name_or_path, "compressor")
        compressor_state_path = os.path.join(compressor_dir, "compressor_state.pt")

        new_format_decoder = os.path.exists(decoder_state_path)
        new_format_compressor = os.path.exists(compressor_state_path)

        if not config.freeze_decoder and config.lora_decoder and not new_format_decoder:
            # Legacy path: PEFT adapter dir at the checkpoint root.
            config.decoder_adapter_path = pretrained_model_name_or_path

        if config.lora_compressor and not new_format_compressor:
            # Legacy path: PEFT adapter dir under compressor/.
            config.compressor_adapter_path = compressor_dir
        elif not config.lora_compressor:
            config.compressor_model_name = compressor_dir

        print(config)
        model = cls(config)

        if new_format_decoder and not config.freeze_decoder:
            model.decoder.load_state_dict(
                torch.load(decoder_state_path, weights_only=True), strict=False
            )
        if new_format_compressor and config.lora_compressor:
            model.compressor.load_state_dict(
                torch.load(compressor_state_path, weights_only=True), strict=False
            )

        model.connector.load_state_dict(
            torch.load(os.path.join(pretrained_model_name_or_path, "connector.pt"), weights_only=True)
        )
        connector_dtype = _resolve_torch_dtype(config.torch_dtype)
        if connector_dtype is not None:
            model.connector.to(connector_dtype)
        return model
