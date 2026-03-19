"""
PISCO
Copyright (c) 2026-present NAVER Corp.
All Rights Reserved.

Implementations of collators for PISCO/OSCAR. These are crucial components as they define which
parts of the inputs are to be compressed, where they are placed in a the decoder inputs and what
the decoder is trained to generate from the compressed.
"""

import torch
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from transformers import DefaultDataCollator

from pisco.collator_utils import (
    mask_before_mem,
    chunk_list,
    chunk_random_no_tiny_tail,
    add_memory_tokens_to_inputs,
)


class BaseCollator(DefaultDataCollator):
    def __init__(
        self,
        compressor_tokenizer,
        decoder_tokenizer,
        compr_rate,
        compressor_max_length=512,
        decoder_max_length=1024,
        *args,
        **kwargs,
    ):
        super(BaseCollator, self).__init__(*args, **kwargs)
        self.compressor_tokenizer = compressor_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.compr_rate = compr_rate
        self.compressor_max_length = compressor_max_length
        self.decoder_max_length = decoder_max_length
        print(f"Compressor max length in collator {compressor_max_length}")
        print(f"Decoder max length in collator {decoder_max_length}")

    def assert_consistent_n_mems(self, compressor_inputs, decoder_inputs):
        n_mem_in_decoder = (
            decoder_inputs["input_ids"] == self.decoder_tokenizer.mem_token_id
        ).sum()
        n_mem_in_compressor = 0
        if len(compressor_inputs["input_ids"]) > 0:
            n_mem_in_compressor = (
                compressor_inputs["input_ids"] == self.compressor_tokenizer.mem_token_id
            ).sum()

        # Check we formed as many chunks to compress as placeholders in decoder inputs to put them at:
        assert (
            n_mem_in_decoder == n_mem_in_compressor
        ), f"{n_mem_in_decoder} != {n_mem_in_compressor}"

    def clean_text(self, text: str) -> str:
        """Prevent accidental special token injection."""
        return text.replace("<RET>", "< RET >").replace("<MEM>", "< MEM >")

    def mask_special_tokens(self, labels):
        labels[labels == self.decoder_tokenizer.ae_token_id] = -100
        labels[labels == self.decoder_tokenizer.pad_token_id] = -100
        labels[labels == self.decoder_tokenizer.bos_token_id] = -100
        labels[labels == self.decoder_tokenizer.mem_token_id] = -100
        return labels

    def compressor_pad(self, input_ids):
        return self.compressor_tokenizer.pad(
            {"input_ids": input_ids},
            padding="longest",
            return_tensors="pt",
        )


class PretrainingCollator(BaseCollator):
    """
    Collator to use for pretraining.
    It expects inputs containing a 'text' field
    from any pretraining dataset like EleutherAI/SmolLM2-135M-10B)
    It allows to produce inputs for two types of tasks:
    - auto-encoding: the texts is fully compressed, and the decoder is asked to reproduce it
    given its embeddings and a task-specific <AE> token
    - text-continuation: text is split into t1 + t2 + t3, t2 is compressed and decoder has loss on
    t3 given t1 and compressed(t2).
    """

    def __init__(
        self,
        compressor_tokenizer,
        decoder_tokenizer,
        compr_rate,
        compressor_max_length=512,
        decoder_max_length=1024,
        *,
        ae_ratio=0.5,
        **kwargs,
    ):
        super(PretrainingCollator, self).__init__(
            compressor_tokenizer=compressor_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            compr_rate=compr_rate,
            compressor_max_length=compressor_max_length,
            decoder_max_length=decoder_max_length,
            **kwargs,
        )
        self.ae_ratio = ae_ratio

    def prepare_for_autoencoding(self, text: str, text_ids: List[int]):
        """
        prepares for autoencoding:
        text_ids is chunked into a random number of chunks of at most compressor_max_length tokens
        """
        text_ids = text_ids[: self.decoder_max_length]  # truncation
        text_length = len(text_ids)

        if text_length <= 64:
            chunks = [text_ids]
        else:
            # chunking:
            # number of chunks approx equal to decoder_max_length // compressor_max_length
            # No chunk larger that compressor_max_length
            chunks = chunk_random_no_tiny_tail(
                text_ids=text_ids, compressor_max_length=self.compressor_max_length
            )

        # Adding the mem tokens for compression:
        compressor_input_ids, n_mems = add_memory_tokens_to_inputs(
            chunks, self.compressor_tokenizer, self.compr_rate
        )

        # Forming the autoencoding decoder sequence, with mem placeholders:
        decoder_text = (
            self.decoder_tokenizer.ae_token
            + self.decoder_tokenizer.mem_token
            * sum(n_mems)  # as many mem tokens here as in all compressed chunks
            + self.decoder_tokenizer.bos_token
            + text
            + self.decoder_tokenizer.eos_token
        )

        return compressor_input_ids, decoder_text

    def prepare_for_text_continuation(self, text_ids):
        """
        Here, each text is chunked into text1 + text2 + text3.
        text2 is compressed and text3 is the target.
        """
        text_ids = text_ids[
            : self.compressor_max_length + self.decoder_max_length
        ]  # truncation

        # Splitting:
        s = np.random.randint(0, 32)
        idx = min(s + self.compressor_max_length, len(text_ids) // 2)
        past_text, future_text = text_ids[:idx], text_ids[idx:]

        # A small number of tokens (at most 64 or half the size) is kept in clear
        # this is to teach the decoder to handle hybrid context
        split_idx = min(s, len(past_text) // 2)
        past_text_clear, past_text_compressed = (
            past_text[:split_idx],
            past_text[split_idx:],
        )

        past_text_compressed, n_mems = add_memory_tokens_to_inputs(
            [past_text_compressed], self.compressor_tokenizer, self.compr_rate
        )

        decoder_text = (
            self.decoder_tokenizer.bos_token
            + self.compressor_tokenizer.decode(past_text_clear)
            + self.decoder_tokenizer.mem_token * sum(n_mems)
            + self.compressor_tokenizer.decode(future_text)
            + self.decoder_tokenizer.eos_token
        )

        return past_text_compressed, decoder_text

    def torch_call(self, examples):
        texts = [self.clean_text(x["text"]) for x in examples]

        # Global tokenization:
        compressor_pre_inputs = self.compressor_tokenizer(
            texts,
            padding="do_not_pad",
            return_tensors=None,
            truncation=False,  # we truncate later
        )

        all_compressor_input_ids = []
        all_decoder_texts = []

        for i, ids in enumerate(compressor_pre_inputs["input_ids"]):
            # Autoencoding
            if np.random.uniform() < self.ae_ratio:
                compressor_input_ids, decoder_text = self.prepare_for_autoencoding(
                    text=texts[i], text_ids=ids
                )

            # Text continuation
            else:
                compressor_input_ids, decoder_text = self.prepare_for_text_continuation(
                    text_ids=ids
                )

            all_compressor_input_ids.extend(compressor_input_ids)
            all_decoder_texts.append(decoder_text)

        # Padding
        compressor_inputs = self.compressor_pad(all_compressor_input_ids)

        decoder_inputs = self.decoder_tokenizer(
            all_decoder_texts,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
            max_length=self.decoder_max_length,
            truncation=True,
        )

        # Masking special tokens:
        labels = decoder_inputs["input_ids"].clone()
        labels = self.mask_special_tokens(labels)

        # Masking anything before last mem:
        labels = mask_before_mem(
            labels, mem_token_id=self.decoder_tokenizer.mem_token_id
        )

        self.assert_consistent_n_mems(compressor_inputs, decoder_inputs)

        return {
            "compressor_input_ids": compressor_inputs["input_ids"],
            "compressor_attention_mask": compressor_inputs["attention_mask"],
            "decoder_input_ids": decoder_inputs["input_ids"],
            "decoder_attention_mask": decoder_inputs["attention_mask"],
            "labels": labels,
        }


class AgentTrajCollator(BaseCollator):
    def __init__(
        self,
        compressor_tokenizer,
        decoder_tokenizer,
        compr_rate,
        compressor_max_length=512,
        decoder_max_length=1024,
        *,
        p_compress=0.5,
        **kwargs,
    ):
        super(AgentTrajCollator, self).__init__(
            compressor_tokenizer=compressor_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            compr_rate=compr_rate,
            compressor_max_length=compressor_max_length,
            decoder_max_length=decoder_max_length,
            **kwargs,
        )
        self.p_compress = p_compress

    def trajectory_compression(
        self, tok_steps, dec_tok_steps, to_compress, decoder_line_return_id
    ):
        """
        Given the list of steps (tokenized and clear), as well as the to_compress
        array indicating which steps to compress, this method forms the associated
        compressor and decoder inputs.
        """
        compressor_input_ids = []
        decoder_ids = []
        for tok_step, dec_tok_step, is_compressed in zip(
            tok_steps, dec_tok_steps, to_compress
        ):
            decoder_ids.append(decoder_line_return_id)
            if is_compressed:
                chunks = chunk_list(
                    tok_step,
                    chunk_length=self.compressor_max_length,
                    chunk_overlap=0,
                )
                chunks, n_mems = add_memory_tokens_to_inputs(
                    chunks, self.compressor_tokenizer, self.compr_rate
                )

                compressor_input_ids.extend(chunks)
                decoder_ids.extend([self.decoder_tokenizer.mem_token_id] * sum(n_mems))

                # early stopping to respect max length (only approximately though):
                if len(decoder_ids) > self.decoder_max_length:
                    break

            else:
                # early stopping to respect max length:
                if len(decoder_ids) + len(dec_tok_step) > self.decoder_max_length:
                    # We would overflow here. Instead, we add what we can from this step:
                    decoder_ids.extend(
                        dec_tok_step[: self.decoder_max_length - len(decoder_ids)]
                    )
                    break

                decoder_ids.extend(dec_tok_step)

        return compressor_input_ids, decoder_ids

    def torch_call(self, x):
        """
        Here the data is in the form of an agent interacting, much like alfworld.
        What we do is compress, with p_compress, the individual steps.
        We then use text continuation loss (no autoencoding) on the non-compressed steps
        """
        trajectories = []
        for elt in x:
            traj = [self.clean_text(step) for step in elt["trajectory"]]
            trajectories.append(traj)

        all_compressor_input_ids = []
        all_decoder_ids = []

        decoder_line_return_id = self.decoder_tokenizer.encode(
            "\n", add_special_tokens=False
        )[0]

        for steps in trajectories:
            tok_steps = self.compressor_tokenizer(
                steps,
                padding="do_not_pad",
                return_tensors=None,
                truncation=False,  # we truncate later
            )["input_ids"]

            dec_tok_steps = self.decoder_tokenizer(
                steps,
                padding="do_not_pad",
                return_tensors=None,
                truncation=False,  # we truncate later
            )["input_ids"]

            to_compress = np.random.uniform(size=len(steps)) < self.p_compress

            compressor_input_ids, decoder_ids = self.trajectory_compression(
                tok_steps, dec_tok_steps, to_compress, decoder_line_return_id
            )

            # Sometimes due to sizes conditions, there are no compressed steps
            # In that case, we force the first one:
            if len(compressor_input_ids) == 0:
                to_compress[0] = True
                compressor_input_ids, decoder_ids = self.trajectory_compression(
                    tok_steps, dec_tok_steps, to_compress, decoder_line_return_id
                )

            assert len(compressor_input_ids) > 0
            all_compressor_input_ids.extend(compressor_input_ids)
            all_decoder_ids.append(decoder_ids)

        decoder_inputs = self.decoder_tokenizer.pad(
            {"input_ids": all_decoder_ids},
            padding="longest",
            return_tensors="pt",
        )

        labels = decoder_inputs["input_ids"].clone()
        # Use text loss in between every compressed turn, so let's mask everything else:
        labels = self.mask_special_tokens(labels)

        compressor_inputs = self.compressor_pad(all_compressor_input_ids)

        assert (
            compressor_inputs["input_ids"].size(-1)
            <= self.compressor_max_length
            + self.compressor_max_length // self.compr_rate
            + 2
        ), f'{compressor_inputs["input_ids"].size(-1)} vs {self.compressor_max_length}'

        self.assert_consistent_n_mems(compressor_inputs, decoder_inputs)

        return {
            "compressor_input_ids": compressor_inputs["input_ids"],
            "compressor_attention_mask": compressor_inputs["attention_mask"],
            "decoder_input_ids": decoder_inputs["input_ids"],
            "decoder_attention_mask": decoder_inputs["attention_mask"],
            "labels": labels,
        }


class FineTuningCollator(BaseCollator):
    """
    Given a query and docs, forms the compressor inputs (potentially chunking each doc)
    and the decoder inputs, with a RAG prompt.
    """

    def __init__(
        self,
        compressor_tokenizer,
        decoder_tokenizer,
        compr_rate,
        compressor_max_length=512,
        decoder_max_length=1024,
        *,
        query_dependent: bool = False,
        chunk_docs: bool = False,  # If True, then docs exceed compressor_max_lengths are chunked
        chunk_overlap: int = 0,  # how much (number of tokens) the chunks should overlap with chunk_docs=True
        n_max_chunks: Optional[int] = None,  # in case of chunking, upper bound on chunk number.
        topk_docs: int = 5,  # how many docs to keep per query.
        system_prompt: str = (
            "You are a helpful assistant. Your task is to extract relevant information from "
            "provided documents and to answer to questions as briefly as possible."
        ),
        user_prompt: str = "\n\nBackground:[documents]\n Question: [question]",
        **kwargs,
    ):
        super(FineTuningCollator, self).__init__(
            compressor_tokenizer=compressor_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            compr_rate=compr_rate,
            compressor_max_length=compressor_max_length,
            decoder_max_length=decoder_max_length,
            **kwargs,
        )
        self.query_dependent = query_dependent

        self.chunk_docs = chunk_docs
        self.chunk_overlap = chunk_overlap
        self.n_max_chunks = n_max_chunks

        self.topk_docs = topk_docs

        if (
            self.chunk_docs
            and self.n_max_chunks is not None
            and (self.compressor_max_length + 1) * self.n_max_chunks
            > 0.1 * self.decoder_max_length
        ):
            print(
                "WARNING: with your chunking params you may exceed the decoder max length"
            )

        assert not (
            self.chunk_docs and self.query_dependent
        ), "Incompatible options for now"

        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

        if query_dependent:
            print("You are using a query-dependent collator.")

    def preprend_query_to_docs(self, queries: List[str], documents: List[List[str]]) -> List[List[str]]:
        query_documents: List[List[str]] = []
        for query, docs in zip(queries, documents):
            query_documents.append(
                ["Query: " + query + "\n Document: " + d for d in docs]
            )
        return query_documents

    def compute_prompt_and_prefix_length(
        self, docs: str, query: str, label: Optional[str]
    ) -> Tuple[str, int]:
        """
        Forms the templated prompt given docs and queries.
        Also returns the length of the prompt WITHOUT the label, to mask in the loss.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self.user_prompt.replace("[documents]", docs).replace(
                    "[question]", query
                ),
            },
        ]
        if label is not None:
            messages.append({"role": "assistant", "content": label})

        prompt = self.decoder_tokenizer.apply_chat_template(messages, tokenize=False)

        # To compute the labels mask
        prefix_length = len(
            self.decoder_tokenizer.apply_chat_template(
                messages[:-1] + [{"role": "assistant", "content": ""}],
                tokenize=True,
            )
        )

        return prompt, prefix_length

    def mask_labels_before_prefix(self, labels: torch.Tensor, prefix_lengths) -> torch.Tensor:
        # Masking anything before the response thanks to prefix lengths:
        n_pad = (labels == self.decoder_tokenizer.pad_token_id).sum(1).unsqueeze(
            1
        ) - 4  # some margin for safety... TODO make this perfect but tedious...
        prefix_lengths = torch.LongTensor(
            prefix_lengths, device=labels.device
        ).unsqueeze(
            1
        )  # (B, 1)
        positions = (
            torch.arange(labels.size(1), device=labels.device)
            .unsqueeze(0)
            .expand(labels.size())
        )  # (B, T)

        prefix_mask = positions < (n_pad + prefix_lengths)
        labels = labels.masked_fill_(prefix_mask, -100)
        return labels

    def preprocess_for_compressor(self, texts: List[str]) -> Dict[str, Any]:
        input_ids = self.compressor_tokenizer(
            texts,
            padding="do_not_pad",
            return_tensors=None,
            truncation=True,
            max_length=self.compressor_max_length,
        )["input_ids"]

        input_ids, _ = add_memory_tokens_to_inputs(
            input_ids, self.compressor_tokenizer, self.compr_rate
        )
        return self.compressor_pad(input_ids)

    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        documents = [elt["docs"] for elt in examples]
        queries = [elt["query"] for elt in examples]
        labels = [elt["mistral_label"] for elt in examples]

        # These are special tokens: we don't want them to appear accidentally in data !
        queries = [self.clean_text(q) for q in queries]
        labels = [self.clean_text(label) for label in labels]

        # In this case, we just prepend the query to the documents:
        if self.query_dependent:
            documents = self.preprend_query_to_docs(queries, documents)

        all_compressor_input_ids = []
        all_decoder_texts = []
        prefix_lengths = []

        for i in range(len(documents)):
            docs = documents[i]
            docs = [self.clean_text(doc) for doc in docs]

            if self.topk_docs is not None:
                docs = docs[: self.topk_docs]

            docs_input_ids = self.compressor_tokenizer(
                docs,
                padding="do_not_pad",
                return_tensors=None,
                truncation=not self.chunk_docs,  # we truncate only when not chunking
                max_length=self.compressor_max_length,
            )["input_ids"]

            if self.chunk_docs:
                chunked_docs_input_ids = []
                doc_text = ""
                for k, elt in enumerate(docs_input_ids):
                    # chunking
                    chunks = chunk_list(
                        elt,
                        chunk_length=self.compressor_max_length,
                        chunk_overlap=self.chunk_overlap,
                    )[: self.n_max_chunks]

                    # adding the mem tokens
                    chunks, n_mems = add_memory_tokens_to_inputs(
                        chunks, self.compressor_tokenizer, self.compr_rate
                    )
                    chunked_docs_input_ids.extend(chunks)

                    # Building the doc prompt, which numbers docs and their chunks
                    doc_text += (
                        f"Document {k}:"
                        + self.decoder_tokenizer.mem_token * sum(n_mems)
                    )

                all_compressor_input_ids.extend(chunked_docs_input_ids)

            else:
                docs_input_ids, n_mems = add_memory_tokens_to_inputs(
                    docs_input_ids, self.compressor_tokenizer, self.compr_rate
                )

                all_compressor_input_ids.extend(docs_input_ids)

                doc_text = "".join(
                    [
                        f"Document {j}:" + self.decoder_tokenizer.mem_token * n_mems[j]
                        for j in range(len(docs))
                    ]
                )

            prompt, prefix_length = self.compute_prompt_and_prefix_length(
                doc_text, queries[i], labels[i]
            )
            all_decoder_texts.append(prompt)
            prefix_lengths.append(prefix_length)

        # Padding
        compressor_inputs = self.compressor_pad(all_compressor_input_ids)

        decoder_inputs = self.decoder_tokenizer(
            all_decoder_texts,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
            max_length=self.decoder_max_length,
            truncation=True,
        )

        labels = decoder_inputs["input_ids"].clone()
        labels = self.mask_labels_before_prefix(labels, prefix_lengths)
        labels = self.mask_special_tokens(labels)

        # Check we formed as many chunks to compress as placeholders in decoder inputs to put them at:
        self.assert_consistent_n_mems(compressor_inputs, decoder_inputs)

        output = {
            "compressor_input_ids": compressor_inputs["input_ids"],
            "compressor_attention_mask": compressor_inputs["attention_mask"],
            "decoder_input_ids": decoder_inputs["input_ids"],
            "decoder_attention_mask": decoder_inputs["attention_mask"],
            "labels": labels,
        }

        return output
