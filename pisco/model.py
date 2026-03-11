import os
import torch
from torch import nn
from peft import LoraConfig
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
)

from pisco.collator import add_memory_tokens_to_inputs
from pisco.colbert_utils import colbert_score_single


# Fake class for sentence transformers compliance:
class ModelCardData:
    def set_evaluation_metrics(self, model, metrics, epoch, step):
        return


class PISCOConfig(PretrainedConfig):

    model_type = "PISCO"

    def __init__(
        self,
        decoder_model_name: str = "Qwen/Qwen3-8B",
        compressor_model_name: str = "Qwen/Qwen3-0.6B",
        compr_rate: int = 16,
        compressor_mlp_hidden_dim: int = 4096,
        lora_r_decoder: int = 64,
        lora_compressor: bool = False,
        lora_r_compressor: int = 64,
        attn_implementation: str = "flash_attention_2",
        device_map=None,
        load_decoder=True,
        retrieval_pooling: str = "mean",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.decoder_model_name = decoder_model_name  # model name of decoder

        self.compressor_model_name = compressor_model_name  # model name of compressor
        self.compr_rate = compr_rate  # compression rate
        self.compressor_mlp_hidden_dim = compressor_mlp_hidden_dim

        self.lora_r_decoder = lora_r_decoder  # lora_r for lora training, we use 16 throughout the experiment.

        self.lora_r_compressor = lora_r_compressor
        self.lora_compressor = lora_compressor

        self.device_map = device_map

        self.attn_implementation = attn_implementation

        self.load_decoder = load_decoder
        self.retrieval_pooling = retrieval_pooling
        assert self.retrieval_pooling in ["last", "first", "mean", "extra", "colbert"]


class PISCO(PreTrainedModel):
    config_class = PISCOConfig

    # sentence transformers compliance:
    similarity_fn_name = "cosine"
    model_card_data = ModelCardData()

    def __init__(self, config):
        super().__init__(config)
        self.decoder_tokenizer = self.create_decoder_tokenizer(config)

        if config.load_decoder:
            self.decoder = self.create_decoder(config)
            print("Base decoder nb parameters", self.decoder.num_parameters())
            print(
                f"Decoder trainable parameters: {self.decoder.num_parameters(only_trainable=True)}"
            )

        decoder_config = AutoConfig.from_pretrained(config.decoder_model_name)

        self.compressor_tokenizer = self.create_compressor_tokenizer(config)
        self.compressor, self.connector = self.create_compressor_and_connector(
            config, decoder_hidden_dim=decoder_config.hidden_size
        )

        print("Base compressor nb parameters", self.compressor.num_parameters())

        # other settings
        self.generation_top_k = 1
        self.compr_rate = config.compr_rate

        print(
            f"Compressor trainable parameters: {self.compressor.num_parameters(only_trainable=True)}"
        )
        print(f"Total trainable parameters: {self.num_parameters(only_trainable=True)}")

    def create_decoder(self, config):
        """
        Loads the base decoder.
        """
        decoder = AutoModelForCausalLM.from_pretrained(
            config.decoder_model_name,
            attn_implementation=self.config.attn_implementation,
            dtype=torch.bfloat16,
            device_map=config.device_map,
        )
        decoder.resize_token_embeddings(len(self.decoder_tokenizer))

        peft_config = self.get_peft_config(lora_r=config.lora_r_decoder)
        decoder.add_adapter(peft_config)

        return decoder

    def create_decoder_tokenizer(self, config: PISCOConfig):
        decoder_tokenizer = AutoTokenizer.from_pretrained(
            config.decoder_model_name, padding_side="left", truncation_side="right"
        )

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

    def create_compressor_and_connector(self, config, decoder_hidden_dim):
        compressor = AutoModelForCausalLM.from_pretrained(
            config.compressor_model_name,
            attn_implementation=self.config.attn_implementation,
            dtype=torch.bfloat16,
            device_map=config.device_map,
        )
        compressor.resize_token_embeddings(len(self.compressor_tokenizer))

        if config.lora_compressor:
            print("Lora on compressor is on.")
            peft_config = self.get_peft_config(lora_r=config.lora_r_compressor)
            compressor.add_adapter(peft_config)

        connector = nn.Sequential(
            nn.Linear(compressor.config.hidden_size, config.compressor_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.compressor_mlp_hidden_dim, decoder_hidden_dim),
        )

        return compressor, connector

    def create_compressor_tokenizer(self, config):
        compressor_tokenizer = AutoTokenizer.from_pretrained(
            config.compressor_model_name, padding_side="left"
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

    def add_retrieval_token(self):
        print("Adding retrieval token")
        assert self.config.retrieval_pooling == "extra"

        self.compressor_tokenizer.retrieval_token = "<RET>"
        self.compressor_tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<RET>"]}
        )
        self.compressor_tokenizer.retrieval_token_id = (
            self.compressor_tokenizer.convert_tokens_to_ids(
                self.compressor_tokenizer.retrieval_token
            )
        )
        self.compressor.resize_token_embeddings(len(self.compressor_tokenizer))

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
            modules_to_save=["embed_tokens", "lm_head"],  # important
        )

    def prepare(
        self,
        retrieval_pooling: str = None,
        decoder_gradient_checkpointing: bool = False,
        compressor_gradient_checkpointing: bool = False,
    ):
        if retrieval_pooling is None:
            retrieval_pooling = self.config.retrieval_pooling
        self.config.retrieval_pooling = retrieval_pooling
        if self.config.retrieval_pooling != retrieval_pooling:
            print(
                f"Changing pooling from {self.config.retrieval_pooling} to {retrieval_pooling}"
            )
        if retrieval_pooling == "extra":
            self.add_retrieval_token()

        if decoder_gradient_checkpointing:
            print("Activating gradient checkpointing on decoder")
            self.decoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}  # optional
            )

        if compressor_gradient_checkpointing:
            print("Activating gradient checkpointing on compressor")
            self.compressor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}  # optional
            )

    def compressor_forward_and_gather(self, input_ids, attention_mask):
        """
        forward call of compressor + getting the hidden states in front of each mem
        returns a list of tensors of variable lengths (n_mems)
        """
        last_hidden_states = self.compressor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        ).hidden_states[
            -1
        ]  # shape is B, T, hc

        token_ids_list = [self.compressor_tokenizer.mem_token_id]
        retrieval_token_id = getattr(
            self.compressor_tokenizer, "retrieval_token_id", None
        )
        if retrieval_token_id is not None:
            token_ids_list.append(retrieval_token_id)

        mask = torch.isin(
            input_ids, torch.tensor(token_ids_list, device=input_ids.device)
        )

        hidden_states = [
            last_hidden_states[i, mask[i], :] for i in range(len(last_hidden_states))
        ]

        return hidden_states

    def compute_retrieval_embeddings_from_hidden_states(self, hidden_states):
        # For now simple: the list of mem tokens for each doc is averaged into a single retrieval embedding:
        hidden_states = [F.normalize(elt, p=2, dim=-1) for elt in hidden_states]
        if self.config.retrieval_pooling == "mean":
            emb = torch.stack([torch.mean(h, dim=0) for h in hidden_states])
        elif self.config.retrieval_pooling in ["last", "extra"]:
            emb = torch.stack([h[-1] for h in hidden_states])
        elif self.config.retrieval_pooling == "first":
            emb = torch.stack([h[0] for h in hidden_states])
        elif self.config.retrieval_pooling == "colbert":
            emb = hidden_states
            return emb
        else:
            raise ValueError(
                f"Unrecognized retrieval pooling mode {self.config.retrieval_pooling}"
            )
        return emb

    def compute_retrieval_embeddings(self, input_ids, attention_mask):
        hidden_states = self.compressor_forward_and_gather(input_ids, attention_mask)
        return self.compute_retrieval_embeddings_from_hidden_states(hidden_states)

    def compress(self, input_ids, attention_mask) -> list[torch.Tensor]:
        """
        Forward call + collection of last hidden states at each mem_token_id position
        It concatenates all obtained hidden states into a single tensor.
        """
        hidden_states = self.compressor_forward_and_gather(input_ids, attention_mask)

        # NB: we use pre-mlp stuff as retrieval embeddings TODO: try post mlp ?
        retrieval_embeddings = self.compute_retrieval_embeddings_from_hidden_states(
            hidden_states
        )

        # In that case we added an extra token: we remove now for the generation part.
        if self.config.retrieval_pooling == "extra":
            hidden_states = [elt[:-1] for elt in hidden_states]

        all_hidden_states = torch.cat(hidden_states, 0)  # (sum l_i, hc) shape
        embeddings = self.connector(all_hidden_states)

        return embeddings, retrieval_embeddings

    def replace_embeddings(self, compressed_embs, dec_input_ids):
        """
        Replace memory tokens in the decoder input with the compressed embeddings
        This assumes (and checks) that there are as many elements compressed_embs as there are mem tokens in dec_input_ids
        """
        dec_embeds = self.decoder.get_input_embeddings()(dec_input_ids)
        B, L, H = dec_embeds.shape

        # Locate mem_token positions
        mem_mask = dec_input_ids == self.decoder_tokenizer.mem_token_id

        num_mem_tokens = mem_mask.sum().item()
        assert num_mem_tokens == compressed_embs.shape[0], (
            f"Mismatch: {num_mem_tokens} mem tokens but "
            f"{compressed_embs.shape[0]} compressed embeddings"
        )

        # Replace embeddings in order
        mem_indices = mem_mask.view(-1).nonzero(as_tuple=False).squeeze(1)

        # Flatten embeddings for easy indexing
        dec_embeds_flat = dec_embeds.view(-1, H)

        # Replace
        dec_embeds_flat[mem_indices] = compressed_embs

        # Restore shape
        dec_embeds = dec_embeds_flat.view(B, L, H)

        return dec_embeds

    def forward(
        self,
        compressor_input_ids: torch.LongTensor = None,
        compressor_attention_mask: torch.LongTensor = None,
        decoder_input_ids: torch.LongTensor = None,
        decoder_attention_mask: torch.LongTensor = None,
        labels: torch.LongTensor = None,
        query_input_ids: torch.LongTensor = None,
        query_attention_mask: torch.LongTensor = None,
        doc_input_ids: torch.LongTensor = None,
        doc_attention_mask: torch.LongTensor = None,
    ):
        output = {}

        # Compression
        embeddings, d_embeddings = self.compress(
            compressor_input_ids, compressor_attention_mask
        )

        output["d_embedding"] = d_embeddings

        # Generation
        if decoder_input_ids is not None:
            assert decoder_attention_mask is not None
            # Inserting compressed reps into the decoder inputs:
            dec_inputs_embeds = self.replace_embeddings(embeddings, decoder_input_ids)

            decoder_outputs = self.decoder(
                inputs_embeds=dec_inputs_embeds,
                attention_mask=decoder_attention_mask,
                labels=labels,
            )

            output["loss"] = decoder_outputs.loss
            output["logits"] = decoder_outputs.logits

        # Query embedding
        if query_input_ids is not None:
            assert query_attention_mask is not None
            output["q_embedding"] = self.compute_retrieval_embeddings(
                query_input_ids, query_attention_mask
            )

        # Document embedding: overwriting.
        if doc_input_ids is not None:
            assert doc_attention_mask is not None
            output["d_embedding"] = self.compute_retrieval_embeddings(
                doc_input_ids, doc_attention_mask
            )

        return output

    def generate(self, model_input, max_new_tokens=128):

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

        embeddings, _ = self.compress(compressor_input_ids, compressor_attention_mask)

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
        """
        self.config.save_pretrained(save_directory)
        torch.save(
            {
                k: v
                for k, v in self.decoder.state_dict().items()
                if any(x in k for x in ["embed_tokens", "lm_head", "lora", "adapter"])
            },
            os.path.join(save_directory, "decoder_state.pt"),
        )

        if self.config.lora_compressor:
            # We load embedding + head + lora stuff "only".
            torch.save(
                {
                    k: v
                    for k, v in self.compressor.state_dict().items()
                    if any(
                        x in k for x in ["embed_tokens", "lm_head", "lora", "adapter"]
                    )
                },
                os.path.join(save_directory, "compressor_state.pt"),
            )
        else:
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
    ):
        """
        Loading: to take care of checkpoints containing only lora and not base model.
        """
        # Load the configuration
        config = PISCOConfig.from_pretrained(pretrained_model_name_or_path)
        config.attn_implementation = kwargs.get(
            "attn_implementation", config.attn_implementation
        )

        config.load_decoder = load_decoder

        model = cls(config)

        if load_decoder:
            model.decoder.load_state_dict(
                torch.load(
                    os.path.join(pretrained_model_name_or_path, "decoder_state.pt")
                ),
                strict=False,
            )

        if config.lora_compressor:
            model.compressor.load_state_dict(
                torch.load(
                    os.path.join(pretrained_model_name_or_path, "compressor_state.pt")
                ),
                strict=False,
            )
        else:
            model.compressor = AutoModelForCausalLM.from_pretrained(
                os.path.join(pretrained_model_name_or_path, "compressor"),
                dtype=torch.bfloat16,
                attn_implementation=config.attn_implementation,
            )

        model.connector.load_state_dict(
            torch.load(os.path.join(pretrained_model_name_or_path, "connector.pt"))
        )
        model.connector.to(torch.bfloat16)

        return model

    def encode_batch(self, texts, max_length):
        input_ids = self.compressor_tokenizer(
            texts,
            padding="do_not_pad",
            return_tensors=None,
            truncation=True,
            max_length=max_length,
        )["input_ids"]

        input_ids, _ = add_memory_tokens_to_inputs(
            input_ids, self.compressor_tokenizer, self.compr_rate
        )
        batch_dict = self.compressor_tokenizer.pad(
            {"input_ids": input_ids},
            padding="longest",
            return_tensors="pt",
        )
        batch_dict = {
            key: value.to(self.compressor.device) for key, value in batch_dict.items()
        }
        with torch.no_grad():
            embeddings = self.compute_retrieval_embeddings(
                **batch_dict
            )  # TODO: colbert ?
            if self.config.retrieval_pooling == "colbert":
                return [emb.cpu().float().numpy() for emb in embeddings]
            else:
                return embeddings.cpu().float().numpy()

    def encode(
        self,
        sentences: list[str],
        task_name: str = None,
        max_length: int = None,
        prompt_type=None,  # PromptType object
        batch_size: int = None,
        **kwargs,
    ) -> np.ndarray:
        """
        method for sentence transformers/MTEB compliance
        """
        all_embeddings = []

        # for MTEB:
        if isinstance(sentences, DataLoader):
            for batch in sentences:
                batch_texts = batch.get("text")
                if batch_texts is None:
                    batch_texts = batch["query"]
                embeddings = self.encode_batch(batch_texts, max_length=max_length)
                if self.config.retrieval_pooling == "colbert":
                    all_embeddings.extend(embeddings)
                else:
                    all_embeddings.append(embeddings)

        # for ST during evals:
        else:
            for i in range(0, len(sentences), batch_size):
                batch_texts = sentences[i : i + batch_size]
                embeddings = self.encode_batch(batch_texts, max_length=max_length)
                if self.config.retrieval_pooling == "colbert":
                    all_embeddings.extend(embeddings)
                else:
                    all_embeddings.append(embeddings)

        if self.config.retrieval_pooling == "colbert":
            return all_embeddings
        else:
            return np.concatenate(all_embeddings, axis=0)

    encode_query = encode
    encode_document = encode

    def similarity(self, a, b):
        """
        SentenceTransformers expects this.
        Cosine similarity = dot product of normalized vectors.
        for colbert, a is a list of (Ni, dim)
        b idem.
        """
        if self.config.retrieval_pooling == "colbert":
            scores = torch.zeros(len(a), len(b))
            for i, q in enumerate(a):
                q = torch.from_numpy(q)
                for j, d in enumerate(b):
                    scores[i, j] = colbert_score_single(q, torch.from_numpy(d))

            return scores

        if isinstance(a, np.ndarray):
            a = torch.from_numpy(a)
        if isinstance(b, np.ndarray):
            b = torch.from_numpy(b)

        return a @ b.T
