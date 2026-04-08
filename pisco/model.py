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
    AutoModelForMultimodalLM,
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
    decoder_adapter_path: Optional[str] = None

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
        decoder_adapter_path: Optional[str] = None,
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
        self.decoder_adapter_path = decoder_adapter_path


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

        self.compressor_tokenizer = self.create_compressor_tokenizer(config)
        self.compressor, self.connector = self.create_compressor_and_connector(
            config, decoder_hidden_dim=self._get_connector_target_dim(config)
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
    def _get_connector_target_dim(config: PISCOConfig) -> int:
        """Dimension the connector should project into (decoder embed_tokens hidden_size)."""
        decoder_config = AutoConfig.from_pretrained(config.decoder_model_name)
        if hasattr(decoder_config, "hidden_size"):
            return decoder_config.hidden_size
        return decoder_config.text_config.hidden_size

    def _get_target_embedding(self) -> nn.Module:
        """Return the embedding module where compressed representations are injected."""
        return self.decoder.get_input_embeddings()

    @staticmethod
    def _model_load_kwargs(config: PISCOConfig) -> Dict[str, object]:
        """Common kwargs for AutoModelForCausalLM.from_pretrained."""
        kw: Dict[str, object] = {"dtype": torch.bfloat16, "device_map": config.device_map}
        if config.attn_implementation is not None:
            kw["attn_implementation"] = config.attn_implementation
        return kw

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

        load_kwargs = self._model_load_kwargs(config)
        try:
            decoder = cast(
                PreTrainedModel,
                AutoModelForMultimodalLM.from_pretrained(config.decoder_model_name, **load_kwargs),
            )
        except Exception as e:
            print(f"AutoModelForMultimodalLM failed for {config.decoder_model_name}: {e}")
            decoder = cast(
                PreTrainedModel,
                AutoModelForCausalLM.from_pretrained(config.decoder_model_name, **load_kwargs),
            )
        self._resize_token_embeddings(decoder, len(self.decoder_tokenizer))

        if has_adapter_checkpoint and adapter_source is not None:
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
        ) -> tuple["_BaseModelWithGenerate", nn.Sequential]:
        compressor = cast("_BaseModelWithGenerate", AutoModelForCausalLM.from_pretrained(
            config.compressor_model_name,
            **self._model_load_kwargs(config),
        ))
        compressor.resize_token_embeddings(len(self.compressor_tokenizer))

        hidden_size = compressor.config.hidden_size if hasattr(compressor.config, "hidden_size") else compressor.config.text_config.hidden_size

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

    @staticmethod
    def _resize_token_embeddings(model: PreTrainedModel, new_num_tokens: int):
        """Resize all embedding tables to accommodate new tokens.

        Handles the standard embed_tokens / lm_head via the HF API, then
        resizes auxiliary tables like Gemma4's embed_tokens_per_layer that
        resize_token_embeddings does not cover.
        """
        model.resize_token_embeddings(new_num_tokens)

        for module in model.modules():
            if not hasattr(module, 'embed_tokens_per_layer'):
                continue
            old_emb: nn.Embedding = cast(nn.Embedding, module.embed_tokens_per_layer)
            if old_emb.num_embeddings >= new_num_tokens:
                return
            kwargs: Dict[str, object] = {
                "num_embeddings": new_num_tokens,
                "embedding_dim": old_emb.embedding_dim,
                "padding_idx": old_emb.padding_idx,
            }
            if hasattr(old_emb, 'scalar_embed_scale'):
                kwargs["embed_scale"] = old_emb.scalar_embed_scale
            new_emb: nn.Embedding = type(old_emb)(**kwargs)  # type: ignore[arg-type]
            new_emb.weight.data[:old_emb.num_embeddings] = old_emb.weight.data
            new_emb = new_emb.to(device=old_emb.weight.device, dtype=old_emb.weight.dtype)  # type: ignore[assignment]
            module.embed_tokens_per_layer = new_emb
            return

    def _embed_replace_hook(
        self,
        compressed_embs: torch.Tensor,
        dec_input_ids: torch.LongTensor,
        one_shot: bool = False,
    ):
        """
        Register a forward hook on the target embedding layer that replaces
        MEM-token embeddings with compressed representations.

        Returns the hook handle (caller must remove it).

        When *one_shot* is True the hook only fires once (for the prefill step
        during ``generate``); subsequent calls with a different sequence length
        are passed through unchanged.
        """
        mem_mask = (dec_input_ids == self.decoder_tokenizer.mem_token_id)
        num_mem = mem_mask.sum().item()
        assert num_mem == compressed_embs.shape[0], (
            f"Mismatch: {num_mem} mem tokens but "
            f"{compressed_embs.shape[0]} compressed embeddings"
        )
        mem_indices = mem_mask.view(-1).nonzero(as_tuple=False).squeeze(1)
        prefill_seq_len = dec_input_ids.shape[1]
        fired = [False]

        def hook_fn(module, args, output):
            if one_shot and (fired[0] or output.shape[1] != prefill_seq_len):
                return output
            fired[0] = True
            flat = output.reshape(-1, output.shape[-1])
            flat = flat.index_copy(0, mem_indices, compressed_embs)
            return flat.view(output.shape)

        return self._get_target_embedding().register_forward_hook(hook_fn)

    def forward(
        self,
        compressor_input_ids: torch.LongTensor,
        compressor_attention_mask: torch.LongTensor,
        decoder_input_ids: torch.LongTensor,
        decoder_attention_mask: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        embeddings = self.compress(compressor_input_ids, compressor_attention_mask)

        handle = self._embed_replace_hook(embeddings, decoder_input_ids)
        try:
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                labels=labels,
            )
        finally:
            handle.remove()

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

        handle = self._embed_replace_hook(embeddings, decoder_input_ids, one_shot=True)
        try:
            output_ids = self.decoder.generate(
                input_ids=decoder_input_ids.to("cuda"),
                attention_mask=decoder_attention_mask.to("cuda"),
                do_sample=False,
                top_p=None,
                max_new_tokens=max_new_tokens,
            )
        finally:
            handle.remove()

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
        if load_decoder:
            config.decoder_adapter_path = pretrained_model_name_or_path

        model = cls(config)

        model.compressor = AutoModelForCausalLM.from_pretrained(
            os.path.join(pretrained_model_name_or_path, "compressor"),
            **cls._model_load_kwargs(config),
        )
        model.connector.load_state_dict(
            torch.load(os.path.join(pretrained_model_name_or_path, "connector.pt"))
        )
        model.connector.to(torch.bfloat16)

        return model


class PISCOPLE(PISCO):
    """PISCO variant for Gemma4 that injects compressed representations into
    the per-layer embedding (PLE) stream instead of the main token embeddings.

    Because the main embed_tokens is left untouched, the MEM and AE token
    embeddings are learned normally through LoRA — only the PLE signal at
    MEM positions carries the compressed context.
    """

    @staticmethod
    def _get_connector_target_dim(config: PISCOConfig) -> int:
        """Connector targets the PLE embedding dim (num_layers * ple_dim)."""
        decoder_config = AutoConfig.from_pretrained(config.decoder_model_name)
        text_config = (
            decoder_config.text_config
            if hasattr(decoder_config, "text_config")
            else decoder_config
        )
        ple_dim = getattr(text_config, "hidden_size_per_layer_input", 0)
        if not ple_dim:
            raise ValueError(
                f"Decoder {config.decoder_model_name} has no per-layer embeddings. "
                "Use PISCO instead of PISCOPLE."
            )
        return text_config.num_hidden_layers * ple_dim

    def _get_target_embedding(self) -> nn.Module:
        """Return the embed_tokens_per_layer module from the decoder."""
        for module in self.decoder.modules():
            if hasattr(module, "embed_tokens_per_layer"):
                return module.embed_tokens_per_layer
        raise RuntimeError(
            "No embed_tokens_per_layer found in decoder. "
            "PISCOPLE requires a Gemma4-family decoder."
        )
