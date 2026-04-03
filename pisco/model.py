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
from peft import LoraConfig

from transformers import (
    AutoModelForCausalLM,
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
    decoder_model_name: str = "Qwen/Qwen3-8B"
    compressor_model_name: str = "Qwen/Qwen3-0.6B"
    compr_rate: int = 16
    compressor_mlp_hidden_dim: int = 4096
    lora_decoder: bool = True
    lora_r_decoder: int = 64
    attn_implementation: Optional[str] = None
    device_map: Optional[str] = None
    load_decoder: bool = True
    decoder_gradient_checkpointing: bool = False

    def __init__(
        self,
        decoder_model_name: str = "Qwen/Qwen3-8B",
        compressor_model_name: str = "Qwen/Qwen3-0.6B",
        compr_rate: int = 16,
        compressor_mlp_hidden_dim: int = 4096,
        lora_decoder: bool = True,  # TODO: this is mandatory right now
        lora_r_decoder: int = 64,
        attn_implementation: Optional[str] = None,
        device_map: Optional[str] = None,
        load_decoder: bool = True,
        decoder_gradient_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.decoder_model_name = decoder_model_name  # model name of decoder

        self.compressor_model_name = compressor_model_name  # model name of compressor
        self.compr_rate = compr_rate  # compression rate
        self.compressor_mlp_hidden_dim = compressor_mlp_hidden_dim

        self.lora_decoder = lora_decoder  # boolean type, whether to use lora training
        self.lora_r_decoder = lora_r_decoder  # lora_r for lora training, we use 16 throughout the experiment.

        self.attn_implementation = attn_implementation
        self.device_map = device_map

        self.load_decoder = load_decoder
        self.decoder_gradient_checkpointing = decoder_gradient_checkpointing


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

        # other settings
        self.generation_top_k = 1
        self.compr_rate = config.compr_rate

        print(
            f"Compressor trainable parameters: {self.compressor.num_parameters(only_trainable=True)}"
        )
        print(f"Total trainable parameters: {self.num_parameters(only_trainable=True)}")

    @staticmethod
    def _model_load_kwargs(config: PISCOConfig) -> Dict[str, object]:
        """Common kwargs for AutoModelForCausalLM.from_pretrained."""
        kw: Dict[str, object] = {"dtype": torch.bfloat16, "device_map": config.device_map}
        if config.attn_implementation is not None:
            kw["attn_implementation"] = config.attn_implementation
        return kw

    def create_decoder(self, config: PISCOConfig) -> PreTrainedModel:
        """
        Loads the base decoder.
        """
        decoder = cast(PreTrainedModel, AutoModelForCausalLM.from_pretrained(
            config.decoder_model_name,
            **self._model_load_kwargs(config),
        ))
        decoder.resize_token_embeddings(len(self.decoder_tokenizer))

        if config.lora_decoder:
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
        ) -> tuple["_BaseModelWithGenerate", nn.Sequential]:
        compressor = cast("_BaseModelWithGenerate", AutoModelForCausalLM.from_pretrained(
            config.compressor_model_name,
            **self._model_load_kwargs(config),
        ))
        compressor.resize_token_embeddings(len(self.compressor_tokenizer))

        hidden_size = getattr(
            compressor.config, 
            "hidden_size",
            compressor.config.text_config.hidden_size
        )

        connector = nn.Sequential(
            nn.Linear(hidden_size, config.compressor_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.compressor_mlp_hidden_dim, decoder_hidden_dim),
        )

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

        # Flatten embeddings for easy indexing.
        dec_embeds_flat = dec_embeds.reshape(-1, H)
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
            inputs_embeds=decoder_inputs_embeds.to("cuda"),
            attention_mask=decoder_attention_mask.to("cuda"),
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
        self.config.save_pretrained(save_directory)
        self.decoder.save_pretrained(save_directory, safe_serialization=True)
        self.compressor.save_pretrained(os.path.join(save_directory, "compressor"))
        torch.save(
            self.connector.state_dict(), os.path.join(save_directory, "connector.pt")
        )
        self.compressor_tokenizer.save_pretrained(
            os.path.join(save_directory, "compressor_tokenizer")
        )
        self.decoder_tokenizer.save_pretrained(
            os.path.join(save_directory, "decoder_tokenizer")
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        load_decoder=True,
        *args,
        **kwargs,
    ) -> "PISCO":
        """
        Loading: to take care of checkpoints containing only lora and not base model.
        """
        # Load the configuration
        config = PISCOConfig.from_pretrained(pretrained_model_name_or_path)
        config.load_decoder = load_decoder

        model = cls(config)

        if load_decoder:
            adapter_config_path = os.path.join(
                pretrained_model_name_or_path, "adapter_config.json"
            )
            decoder_state_path = os.path.join(
                pretrained_model_name_or_path, "decoder_state.pt"
            )

            if os.path.exists(adapter_config_path):
                model.decoder.load_adapter(
                    pretrained_model_name_or_path, adapter_name="default"
                )
            elif os.path.exists(decoder_state_path):
                # Backward compatibility with old checkpoints.
                model.decoder.load_state_dict(
                    torch.load(decoder_state_path),
                    strict=False,
                )
            else:
                raise FileNotFoundError(
                    f"Could not find decoder adapter files in "
                    f"{pretrained_model_name_or_path}. Expected either "
                    f"'adapter_config.json' (new format) or "
                    f"'decoder_state.pt' (legacy format)."
                )

        model.compressor = AutoModelForCausalLM.from_pretrained(
            os.path.join(pretrained_model_name_or_path, "compressor"),
            **cls._model_load_kwargs(config),
        )
        model.connector.load_state_dict(
            torch.load(os.path.join(pretrained_model_name_or_path, "connector.pt"))
        )
        model.connector.to(torch.bfloat16)

        return model
