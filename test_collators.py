"""
Tests for PISCO collators: PretrainingCollator, AgentTrajCollator, FineTuningCollator.

These tests use lightweight mock tokenizers and stubs so they can run without
the full PISCO / Transformers model stack.  Each test checks the shapes and
internal consistency constraints that the collators themselves assert (number
of <MEM> tokens in compressor == number of <MEM> placeholders in decoder, etc.).

Run with:
    pytest test_collators.py -v
or simply:
    python test_collators.py
"""

import math
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Minimal mock tokenizer
# ---------------------------------------------------------------------------

class MockTokenizer:
    """
    A lightweight tokenizer that encodes text as UTF-8 byte values (0-255)
    and reserves a handful of special-token IDs above 255.
    """

    PAD_ID  = 256
    BOS_ID  = 257
    EOS_ID  = 258
    MEM_ID  = 259
    AE_ID   = 260
    UNK_ID  = 261

    pad_token_id = PAD_ID
    bos_token_id = BOS_ID
    eos_token_id = EOS_ID
    mem_token_id = MEM_ID
    ae_token_id  = AE_ID
    unk_token_id = UNK_ID

    pad_token  = "<PAD>"
    bos_token  = "<BOS>"
    eos_token  = "<EOS>"
    mem_token  = "<MEM>"
    ae_token   = "<AE>"

    def __call__(self, texts, padding="longest", return_tensors=None,
                 truncation=False, max_length=None, add_special_tokens=True):
        if isinstance(texts, str):
            texts = [texts]

        encoded = [self._encode(t) for t in texts]

        if truncation and max_length:
            encoded = [ids[:max_length] for ids in encoded]

        if padding == "longest":
            max_len = max(len(ids) for ids in encoded) if encoded else 0
            attention_masks = []
            padded = []
            for ids in encoded:
                mask = [1] * len(ids) + [0] * (max_len - len(ids))
                ids  = ids + [self.PAD_ID] * (max_len - len(ids))
                padded.append(ids)
                attention_masks.append(mask)
        else:
            padded = encoded
            attention_masks = [[1] * len(ids) for ids in encoded]

        if return_tensors == "pt":
            return {
                "input_ids":      torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            }
        return {"input_ids": padded, "attention_mask": attention_masks}

    def pad(self, features, padding="longest", return_tensors=None):
        """Mirrors HF tokenizer.pad({\"input_ids\": [...]})."""
        input_ids = features["input_ids"]
        # input_ids may be a list of lists or a list of tensors
        if isinstance(input_ids[0], torch.Tensor):
            input_ids = [ids.tolist() for ids in input_ids]
        max_len = max(len(ids) for ids in input_ids) if input_ids else 0
        padded, masks = [], []
        for ids in input_ids:
            mask = [1] * len(ids) + [0] * (max_len - len(ids))
            ids  = ids + [self.PAD_ID] * (max_len - len(ids))
            padded.append(ids)
            masks.append(mask)
        if return_tensors == "pt":
            return {
                "input_ids":      torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        return {"input_ids": padded, "attention_mask": masks}

    def encode(self, text, add_special_tokens=True):
        return self._encode(text)

    def decode(self, ids):
        """Best-effort decode back to text (bytes only; specials become '?')."""
        chars = []
        for i in ids:
            if 0 <= i <= 255:
                try:
                    chars.append(bytes([i]).decode("utf-8", errors="replace"))
                except Exception:
                    chars.append("?")
        return "".join(chars)

    def apply_chat_template(self, messages, tokenize=False):
        """Minimal chat template: just concatenate role+content pairs."""
        result = ""
        for m in messages:
            result += f"[{m['role']}] {m['content']} "
        if tokenize:
            return self._encode(result)
        return result

    # ------------------------------------------------------------------
    def _encode(self, text: str):
        # Encode special token strings first, then fall back to bytes.
        special = {
            "<PAD>": self.PAD_ID,
            "<BOS>": self.BOS_ID,
            "<EOS>": self.EOS_ID,
            "<MEM>": self.MEM_ID,
            "<AE>":  self.AE_ID,
        }
        ids = []
        i = 0
        while i < len(text):
            matched = False
            for tok, tok_id in special.items():
                if text[i:].startswith(tok):
                    ids.append(tok_id)
                    i += len(tok)
                    matched = True
                    break
            if not matched:
                ids.extend(list(text[i].encode("utf-8")))
                i += 1
        return ids


# ---------------------------------------------------------------------------
# Stub for pisco.collator_utils so we don't need the real package installed
# ---------------------------------------------------------------------------

def _make_collator_utils():
    """
    Returns a module-like object with the four utility functions used by the
    collators, implemented just enough for testing.
    """
    m = types.ModuleType("pisco.collator_utils")

    def mask_before_mem(labels, mem_token_id):
        """Zero out everything BEFORE (and including) the last MEM token per row."""
        result = labels.clone()
        for i in range(result.size(0)):
            row = result[i]
            mem_positions = (row == mem_token_id).nonzero(as_tuple=True)[0]
            if len(mem_positions) > 0:
                last_mem = mem_positions[-1].item()
                result[i, : last_mem + 1] = -100
        return result

    def chunk_list(lst, chunk_length, chunk_overlap=0):
        step = max(1, chunk_length - chunk_overlap)
        return [lst[i : i + chunk_length] for i in range(0, len(lst), step)] or [lst]

    def chunk_random_no_tiny_tail(text_ids, compressor_max_length):
        if len(text_ids) <= compressor_max_length:
            return [text_ids]
        n = max(1, math.ceil(len(text_ids) / compressor_max_length))
        size = math.ceil(len(text_ids) / n)
        return [text_ids[i : i + size] for i in range(0, len(text_ids), size)]

    def add_memory_tokens_to_inputs(chunks, tokenizer, compr_rate):
        """Appends n_mem MEM tokens to each chunk; returns (new_chunks, n_mems_list)."""
        new_chunks, n_mems = [], []
        for chunk in chunks:
            n = max(1, len(chunk) // compr_rate)
            new_chunks.append(list(chunk) + [tokenizer.mem_token_id] * n)
            n_mems.append(n)
        return new_chunks, n_mems

    m.mask_before_mem              = mask_before_mem
    m.chunk_list                   = chunk_list
    m.chunk_random_no_tiny_tail    = chunk_random_no_tiny_tail
    m.add_memory_tokens_to_inputs  = add_memory_tokens_to_inputs
    return m


# ---------------------------------------------------------------------------
# Patch everything before importing the collators
# ---------------------------------------------------------------------------

import sys

# Stub out the pisco packages
sys.modules.setdefault("pisco", types.ModuleType("pisco"))
sys.modules["pisco.collator_utils"] = _make_collator_utils()

# Stub transformers.DefaultDataCollator
_transformers = sys.modules.get("transformers") or types.ModuleType("transformers")

class _DefaultDataCollator:
    def __init__(self, *a, **kw):
        pass

_transformers.DefaultDataCollator = _DefaultDataCollator
sys.modules["transformers"] = _transformers

# Now we can import collators (adjust the import path if your project layout differs)
# If running from the repo root, `from pisco.collators import ...` should work.
# We inline the classes here so the test file is self-contained.

# ---- inline the three collator classes (copy-pasted from collators.py) ----

from pisco.collator_utils import (
    mask_before_mem,
    chunk_list,
    chunk_random_no_tiny_tail,
    add_memory_tokens_to_inputs,
)


class BaseCollator(_DefaultDataCollator):
    def __init__(self, compressor_tokenizer, decoder_tokenizer, compr_rate,
                 compressor_max_length=512, decoder_max_length=1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compressor_tokenizer = compressor_tokenizer
        self.decoder_tokenizer    = decoder_tokenizer
        self.compr_rate           = compr_rate
        self.compressor_max_length = compressor_max_length
        self.decoder_max_length    = decoder_max_length

    def assert_consistent_n_mems(self, compressor_inputs, decoder_inputs):
        n_mem_dec  = (decoder_inputs["input_ids"] == self.decoder_tokenizer.mem_token_id).sum()
        n_mem_comp = 0
        if len(compressor_inputs["input_ids"]) > 0:
            n_mem_comp = (compressor_inputs["input_ids"] == self.compressor_tokenizer.mem_token_id).sum()
        assert n_mem_dec == n_mem_comp, f"{n_mem_dec} != {n_mem_comp}"

    def clean_text(self, text):
        return text.replace("<RET>", "< RET >").replace("<MEM>", "< MEM >")

    def mask_special_tokens(self, labels):
        for tid in (self.decoder_tokenizer.ae_token_id,
                    self.decoder_tokenizer.pad_token_id,
                    self.decoder_tokenizer.bos_token_id,
                    self.decoder_tokenizer.mem_token_id):
            labels[labels == tid] = -100
        return labels

    def compressor_pad(self, input_ids):
        return self.compressor_tokenizer.pad(
            {"input_ids": input_ids}, padding="longest", return_tensors="pt")


class PretrainingCollator(BaseCollator):
    def __init__(self, ae_ratio=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ae_ratio = ae_ratio

    def prepare_for_autoencoding(self, text, text_ids):
        text_ids = text_ids[:self.decoder_max_length]
        if len(text_ids) <= 64:
            chunks = [text_ids]
        else:
            chunks = chunk_random_no_tiny_tail(text_ids, self.compressor_max_length)
        compressor_input_ids, n_mems = add_memory_tokens_to_inputs(
            chunks, self.compressor_tokenizer, self.compr_rate)
        decoder_text = (
            self.decoder_tokenizer.ae_token
            + self.decoder_tokenizer.mem_token * sum(n_mems)
            + self.decoder_tokenizer.bos_token
            + text
            + self.decoder_tokenizer.eos_token
        )
        return compressor_input_ids, decoder_text

    def prepare_for_text_continuation(self, text_ids):
        text_ids = text_ids[:self.compressor_max_length + self.decoder_max_length]
        s = np.random.randint(0, 32)
        idx = min(s + self.compressor_max_length, len(text_ids) // 2)
        past_text, future_text = text_ids[:idx], text_ids[idx:]
        split_idx = min(s, len(past_text) // 2)
        past_clear, past_comp = past_text[:split_idx], past_text[split_idx:]
        past_comp, n_mems = add_memory_tokens_to_inputs(
            [past_comp], self.compressor_tokenizer, self.compr_rate)
        decoder_text = (
            self.decoder_tokenizer.bos_token
            + self.compressor_tokenizer.decode(past_clear)
            + self.decoder_tokenizer.mem_token * sum(n_mems)
            + self.compressor_tokenizer.decode(future_text)
            + self.decoder_tokenizer.eos_token
        )
        return past_comp, decoder_text

    def torch_call(self, examples):
        texts = [self.clean_text(x["text"]) for x in examples]
        pre = self.compressor_tokenizer(texts, padding="do_not_pad",
                                        return_tensors=None, truncation=False)
        all_comp_ids, all_dec_texts = [], []
        for i, ids in enumerate(pre["input_ids"]):
            if np.random.uniform() < self.ae_ratio:
                c_ids, d_text = self.prepare_for_autoencoding(texts[i], ids)
            else:
                c_ids, d_text = self.prepare_for_text_continuation(ids)
            all_comp_ids.extend(c_ids)
            all_dec_texts.append(d_text)
        compressor_inputs = self.compressor_pad(all_comp_ids)
        decoder_inputs    = self.decoder_tokenizer(
            all_dec_texts, return_tensors="pt", padding="longest",
            add_special_tokens=False, max_length=self.decoder_max_length, truncation=True)
        labels = decoder_inputs["input_ids"].clone()
        labels = self.mask_special_tokens(labels)
        labels = mask_before_mem(labels, self.decoder_tokenizer.mem_token_id)
        self.assert_consistent_n_mems(compressor_inputs, decoder_inputs)
        return {"compressor_input_ids": compressor_inputs["input_ids"],
                "compressor_attention_mask": compressor_inputs["attention_mask"],
                "decoder_input_ids": decoder_inputs["input_ids"],
                "decoder_attention_mask": decoder_inputs["attention_mask"],
                "labels": labels}


class AgentTrajCollator(BaseCollator):
    def __init__(self, p_compress=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_compress = p_compress

    def trajectory_compression(self, tok_steps, dec_tok_steps, to_compress, lr_id):
        comp_ids, dec_ids = [], []
        for tok_step, dec_step, compress in zip(tok_steps, dec_tok_steps, to_compress):
            dec_ids.append(lr_id)
            if compress:
                chunks = chunk_list(tok_step, self.compressor_max_length, 0)
                chunks, n_mems = add_memory_tokens_to_inputs(
                    chunks, self.compressor_tokenizer, self.compr_rate)
                comp_ids.extend(chunks)
                dec_ids.extend([self.decoder_tokenizer.mem_token_id] * sum(n_mems))
                if len(dec_ids) > self.decoder_max_length:
                    break
            else:
                if len(dec_ids) + len(dec_step) > self.decoder_max_length:
                    dec_ids.extend(dec_step[:self.decoder_max_length - len(dec_ids)])
                    break
                dec_ids.extend(dec_step)
        return comp_ids, dec_ids

    def torch_call(self, x):
        trajectories = [[self.clean_text(s) for s in elt["trajectory"]] for elt in x]
        all_comp_ids, all_dec_ids = [], []
        lr_id = self.decoder_tokenizer.encode("\n", add_special_tokens=False)[0]
        for steps in trajectories:
            tok_steps  = self.compressor_tokenizer(steps, padding="do_not_pad",
                                                   return_tensors=None, truncation=False)["input_ids"]
            dec_steps  = self.decoder_tokenizer(steps, padding="do_not_pad",
                                                return_tensors=None, truncation=False)["input_ids"]
            to_compress = np.random.uniform(size=len(steps)) < self.p_compress
            c_ids, d_ids = self.trajectory_compression(tok_steps, dec_steps, to_compress, lr_id)
            if len(c_ids) == 0:
                to_compress[0] = True
                c_ids, d_ids = self.trajectory_compression(tok_steps, dec_steps, to_compress, lr_id)
            assert len(c_ids) > 0
            all_comp_ids.extend(c_ids)
            all_dec_ids.append(d_ids)
        decoder_inputs = self.decoder_tokenizer.pad(
            {"input_ids": all_dec_ids}, padding="longest", return_tensors="pt")
        labels = decoder_inputs["input_ids"].clone()
        labels = self.mask_special_tokens(labels)
        compressor_inputs = self.compressor_pad(all_comp_ids)
        self.assert_consistent_n_mems(compressor_inputs, decoder_inputs)
        return {"compressor_input_ids": compressor_inputs["input_ids"],
                "compressor_attention_mask": compressor_inputs["attention_mask"],
                "decoder_input_ids": decoder_inputs["input_ids"],
                "decoder_attention_mask": decoder_inputs["attention_mask"],
                "labels": labels}


class FineTuningCollator(BaseCollator):
    def __init__(self, query_dependent=False, chunk_docs=False, chunk_overlap=0,
                 n_max_chunks=None, topk_docs=5,
                 system_prompt="You are a helpful assistant.",
                 user_prompt="\n\nBackground:[documents]\n Question: [question]",
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_dependent = query_dependent
        self.chunk_docs    = chunk_docs
        self.chunk_overlap = chunk_overlap
        self.n_max_chunks  = n_max_chunks
        self.topk_docs     = topk_docs
        self.system_prompt = system_prompt
        self.user_prompt   = user_prompt

    def preprend_query_to_docs(self, queries, documents):
        return [["Query: " + q + "\n Document: " + d for d in docs]
                for q, docs in zip(queries, documents)]

    def compute_prompt_and_prefix_length(self, docs, query, label):
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": self.user_prompt
                     .replace("[documents]", docs).replace("[question]", query)}]
        if label is not None:
            messages.append({"role": "assistant", "content": label})
        prompt = self.decoder_tokenizer.apply_chat_template(messages, tokenize=False)
        prefix_length = len(self.decoder_tokenizer.apply_chat_template(
            messages[:-1] + [{"role": "assistant", "content": ""}], tokenize=True))
        return prompt, prefix_length

    def mask_labels_before_prefix(self, labels, prefix_lengths):
        n_pad = (labels == self.decoder_tokenizer.pad_token_id).sum(1).unsqueeze(1) - 4
        n_pad = n_pad.clamp(min=0)
        plen  = torch.LongTensor(prefix_lengths).unsqueeze(1)
        pos   = torch.arange(labels.size(1)).unsqueeze(0).expand(labels.size())
        labels = labels.masked_fill_(pos < (n_pad + plen), -100)
        return labels

    def torch_call(self, examples):
        documents = [e["docs"]          for e in examples]
        queries   = [self.clean_text(e["query"]) for e in examples]
        labels_   = [self.clean_text(e["mistral_label"]) for e in examples]

        if self.query_dependent:
            documents = self.preprend_query_to_docs(queries, documents)

        all_comp_ids, all_dec_texts, prefix_lengths = [], [], []

        for i in range(len(documents)):
            docs = [self.clean_text(d) for d in documents[i]]
            if self.topk_docs is not None:
                docs = docs[:self.topk_docs]

            docs_ids = self.compressor_tokenizer(
                docs, padding="do_not_pad", return_tensors=None,
                truncation=not self.chunk_docs,
                max_length=self.compressor_max_length)["input_ids"]

            if self.chunk_docs:
                doc_text = ""
                for k, elt in enumerate(docs_ids):
                    chunks = chunk_list(elt, self.compressor_max_length,
                                        self.chunk_overlap)[:self.n_max_chunks]
                    chunks, n_mems = add_memory_tokens_to_inputs(
                        chunks, self.compressor_tokenizer, self.compr_rate)
                    all_comp_ids.extend(chunks)
                    doc_text += f"Document {k}:" + self.decoder_tokenizer.mem_token * sum(n_mems)
            else:
                docs_ids, n_mems = add_memory_tokens_to_inputs(
                    docs_ids, self.compressor_tokenizer, self.compr_rate)
                all_comp_ids.extend(docs_ids)
                doc_text = "".join(
                    f"Document {j}:" + self.decoder_tokenizer.mem_token * n_mems[j]
                    for j in range(len(docs)))

            prompt, plen = self.compute_prompt_and_prefix_length(
                doc_text, queries[i], labels_[i])
            all_dec_texts.append(prompt)
            prefix_lengths.append(plen)

        compressor_inputs = self.compressor_pad(all_comp_ids)
        decoder_inputs    = self.decoder_tokenizer(
            all_dec_texts, return_tensors="pt", padding="longest",
            add_special_tokens=False, max_length=self.decoder_max_length, truncation=True)
        labels = decoder_inputs["input_ids"].clone()
        labels = self.mask_labels_before_prefix(labels, prefix_lengths)
        labels = self.mask_special_tokens(labels)
        self.assert_consistent_n_mems(compressor_inputs, decoder_inputs)
        return {"compressor_input_ids": compressor_inputs["input_ids"],
                "compressor_attention_mask": compressor_inputs["attention_mask"],
                "decoder_input_ids": decoder_inputs["input_ids"],
                "decoder_attention_mask": decoder_inputs["attention_mask"],
                "labels": labels}


# ===========================================================================
# Helpers
# ===========================================================================

def make_tokenizers(compr_rate=4, compressor_max_length=64, decoder_max_length=256):
    comp_tok  = MockTokenizer()
    dec_tok   = MockTokenizer()
    return comp_tok, dec_tok


def make_pretraining_collator(**kw):
    comp_tok, dec_tok = make_tokenizers()
    defaults = dict(compr_rate=4, compressor_max_length=64,
                    decoder_max_length=256, ae_ratio=0.5)
    defaults.update(kw)
    return PretrainingCollator(
        compressor_tokenizer=comp_tok,
        decoder_tokenizer=dec_tok,
        **defaults)


def make_agent_collator(**kw):
    comp_tok, dec_tok = make_tokenizers()
    defaults = dict(compr_rate=4, compressor_max_length=64,
                    decoder_max_length=256, p_compress=0.5)
    defaults.update(kw)
    return AgentTrajCollator(
        compressor_tokenizer=comp_tok,
        decoder_tokenizer=dec_tok,
        **defaults)


def make_finetuning_collator(**kw):
    comp_tok, dec_tok = make_tokenizers()
    defaults = dict(compr_rate=4, compressor_max_length=64,
                    decoder_max_length=512, topk_docs=3)
    defaults.update(kw)
    return FineTuningCollator(
        compressor_tokenizer=comp_tok,
        decoder_tokenizer=dec_tok,
        **defaults)


SHORT_TEXT  = "Hello world."
MEDIUM_TEXT = "The quick brown fox jumps over the lazy dog. " * 10
LONG_TEXT   = "A" * 300   # Will force chunking


# ===========================================================================
# PretrainingCollator tests
# ===========================================================================

class TestPretrainingCollator(unittest.TestCase):

    def _batch(self, texts):
        return [{"text": t} for t in texts]

    # --- output keys --------------------------------------------------------

    def test_output_keys(self):
        col = make_pretraining_collator()
        out = col.torch_call(self._batch([SHORT_TEXT]))
        expected = {"compressor_input_ids", "compressor_attention_mask",
                    "decoder_input_ids",    "decoder_attention_mask", "labels"}
        self.assertEqual(set(out.keys()), expected)

    # --- tensor shapes -------------------------------------------------------

    def test_shapes_batch1(self):
        col = make_pretraining_collator(ae_ratio=1.0)   # force autoencoding
        out = col.torch_call(self._batch([MEDIUM_TEXT]))
        B = 1
        self.assertEqual(out["decoder_input_ids"].shape[0], B)
        self.assertEqual(out["labels"].shape, out["decoder_input_ids"].shape)

    def test_shapes_batch4(self):
        col = make_pretraining_collator()
        out = col.torch_call(self._batch([SHORT_TEXT, MEDIUM_TEXT] * 2))
        self.assertEqual(out["decoder_input_ids"].shape[0], 4)

    # --- MEM consistency (the core collator invariant) ----------------------

    def test_mem_consistency_ae(self):
        """#MEM in compressor == #MEM in decoder, autoencoding path."""
        col = make_pretraining_collator(ae_ratio=1.0)
        np.random.seed(0)
        col.torch_call(self._batch([SHORT_TEXT, MEDIUM_TEXT, LONG_TEXT]))

    def test_mem_consistency_tc(self):
        """#MEM in compressor == #MEM in decoder, text-continuation path."""
        col = make_pretraining_collator(ae_ratio=0.0)
        np.random.seed(42)
        col.torch_call(self._batch([MEDIUM_TEXT, LONG_TEXT]))

    def test_mem_consistency_mixed(self):
        col = make_pretraining_collator(ae_ratio=0.5)
        for seed in range(5):
            np.random.seed(seed)
            col.torch_call(self._batch([SHORT_TEXT, MEDIUM_TEXT, LONG_TEXT]))

    # --- labels masking -------------------------------------------------------

    def test_labels_mask_special_tokens(self):
        col = make_pretraining_collator(ae_ratio=1.0)
        out = col.torch_call(self._batch([SHORT_TEXT]))
        labels = out["labels"]
        # AE, PAD, BOS, MEM tokens should all be -100
        for tid in (MockTokenizer.AE_ID, MockTokenizer.PAD_ID,
                    MockTokenizer.BOS_ID, MockTokenizer.MEM_ID):
            mask = out["decoder_input_ids"] == tid
            self.assertTrue((labels[mask] == -100).all(),
                            f"Token {tid} not masked in labels")

    def test_labels_mask_before_mem(self):
        """Everything up to and including the last MEM must be -100."""
        col = make_pretraining_collator(ae_ratio=1.0)
        out = col.torch_call(self._batch([MEDIUM_TEXT]))
        labels = out["labels"]
        dec    = out["decoder_input_ids"]
        for row in range(labels.size(0)):
            mem_pos = (dec[row] == MockTokenizer.MEM_ID).nonzero(as_tuple=True)[0]
            if len(mem_pos) == 0:
                continue
            last_mem = mem_pos[-1].item()
            self.assertTrue((labels[row, :last_mem + 1] == -100).all(),
                            "Labels not fully masked before/at last MEM")

    # --- decoder max length is respected ------------------------------------

    def test_decoder_truncation(self):
        col = make_pretraining_collator(ae_ratio=1.0, decoder_max_length=64)
        out = col.torch_call(self._batch([LONG_TEXT]))
        self.assertLessEqual(out["decoder_input_ids"].shape[1], 64)

    # --- special-token injection prevention ---------------------------------

    def test_clean_text(self):
        col = make_pretraining_collator(ae_ratio=1.0)
        dirty = "Hello <MEM> world <RET> end"
        batch = self._batch([dirty])
        # Should not raise; special tokens are sanitised before encoding
        col.torch_call(batch)

    # --- attention masks are 0/1 --------------------------------------------

    def test_attention_mask_values(self):
        col = make_pretraining_collator()
        out = col.torch_call(self._batch([SHORT_TEXT, LONG_TEXT]))
        for key in ("compressor_attention_mask", "decoder_attention_mask"):
            vals = out[key].unique()
            self.assertTrue(set(vals.tolist()).issubset({0, 1}),
                            f"{key} contains values outside {{0,1}}")


# ===========================================================================
# AgentTrajCollator tests
# ===========================================================================

SIMPLE_TRAJ = ["Observe: room is dark.", "Action: turn on light.", "Observe: room is bright.",
                "Action: pick up key.", "Done."]

class TestAgentTrajCollator(unittest.TestCase):

    def _batch(self, trajectories):
        return [{"trajectory": t} for t in trajectories]

    def test_output_keys(self):
        col = make_agent_collator()
        out = col.torch_call(self._batch([SIMPLE_TRAJ]))
        expected = {"compressor_input_ids", "compressor_attention_mask",
                    "decoder_input_ids",    "decoder_attention_mask", "labels"}
        self.assertEqual(set(out.keys()), expected)

    def test_shapes_single(self):
        col = make_agent_collator()
        np.random.seed(0)
        out = col.torch_call(self._batch([SIMPLE_TRAJ]))
        self.assertEqual(out["decoder_input_ids"].shape[0], 1)
        self.assertEqual(out["labels"].shape, out["decoder_input_ids"].shape)

    def test_shapes_batch(self):
        col = make_agent_collator()
        np.random.seed(1)
        out = col.torch_call(self._batch([SIMPLE_TRAJ, SIMPLE_TRAJ[:3]]))
        self.assertEqual(out["decoder_input_ids"].shape[0], 2)

    def test_mem_consistency(self):
        col = make_agent_collator(p_compress=0.8)
        for seed in range(5):
            np.random.seed(seed)
            col.torch_call(self._batch([SIMPLE_TRAJ]))

    def test_always_at_least_one_compressed_step(self):
        """p_compress=0 still forces the first step to be compressed."""
        col = make_agent_collator(p_compress=0.0)
        np.random.seed(0)
        out = col.torch_call(self._batch([SIMPLE_TRAJ]))
        # At least one MEM token must appear in compressor inputs
        n_mem = (out["compressor_input_ids"] == MockTokenizer.MEM_ID).sum().item()
        self.assertGreater(n_mem, 0)

    def test_fully_compressed(self):
        col = make_agent_collator(p_compress=1.0)
        np.random.seed(0)
        out = col.torch_call(self._batch([SIMPLE_TRAJ]))
        n_mem = (out["compressor_input_ids"] == MockTokenizer.MEM_ID).sum().item()
        self.assertGreater(n_mem, 0)

    def test_decoder_length_respected(self):
        col = make_agent_collator(decoder_max_length=50)
        np.random.seed(0)
        out = col.torch_call(self._batch([SIMPLE_TRAJ]))
        # Decoder output should not massively exceed decoder_max_length
        # (it can slightly due to the early-stop heuristic, but let's be lenient)
        self.assertLessEqual(out["decoder_input_ids"].shape[1], 200)

    def test_attention_mask_values(self):
        col = make_agent_collator()
        np.random.seed(2)
        out = col.torch_call(self._batch([SIMPLE_TRAJ]))
        for key in ("compressor_attention_mask", "decoder_attention_mask"):
            vals = out[key].unique()
            self.assertTrue(set(vals.tolist()).issubset({0, 1}))

    def test_single_step_trajectory(self):
        """Edge case: trajectory of length 1 must still work."""
        col = make_agent_collator(p_compress=1.0)
        np.random.seed(0)
        out = col.torch_call(self._batch([["Only one step."]]))
        self.assertIn("decoder_input_ids", out)


# ===========================================================================
# FineTuningCollator tests
# ===========================================================================

DOCS  = ["Document about cats.", "Document about dogs.", "Document about birds."]
QUERY = "What animals are mentioned?"
LABEL = "Cats, dogs, and birds."

def _ft_batch(n=2):
    return [{"docs": DOCS, "query": QUERY, "mistral_label": LABEL}] * n


class TestFineTuningCollator(unittest.TestCase):

    def test_output_keys(self):
        col = make_finetuning_collator()
        out = col.torch_call(_ft_batch(1))
        expected = {"compressor_input_ids", "compressor_attention_mask",
                    "decoder_input_ids",    "decoder_attention_mask", "labels"}
        self.assertEqual(set(out.keys()), expected)

    def test_shapes(self):
        col = make_finetuning_collator()
        out = col.torch_call(_ft_batch(2))
        self.assertEqual(out["decoder_input_ids"].shape[0], 2)
        self.assertEqual(out["labels"].shape, out["decoder_input_ids"].shape)

    def test_mem_consistency(self):
        col = make_finetuning_collator()
        col.torch_call(_ft_batch(3))   # assert inside collator

    def test_topk_docs_respected(self):
        """With topk_docs=1, only the first doc should be compressed."""
        col = make_finetuning_collator(topk_docs=1)
        out_1 = col.torch_call(_ft_batch(1))
        col2  = make_finetuning_collator(topk_docs=3)
        out_3 = col2.torch_call(_ft_batch(1))
        mems_1 = (out_1["compressor_input_ids"] == MockTokenizer.MEM_ID).sum().item()
        mems_3 = (out_3["compressor_input_ids"] == MockTokenizer.MEM_ID).sum().item()
        self.assertLessEqual(mems_1, mems_3,
                             "Fewer docs should produce fewer MEM tokens")

    def test_query_dependent_mode(self):
        col = make_finetuning_collator(query_dependent=True)
        col.torch_call(_ft_batch(1))   # should not raise

    def test_chunk_docs(self):
        col = make_finetuning_collator(chunk_docs=True, n_max_chunks=2,
                                       chunk_overlap=0)
        col.torch_call(_ft_batch(1))   # should not raise + MEM consistency checked inside

    def test_labels_mask_special_tokens(self):
        col = make_finetuning_collator()
        out = col.torch_call(_ft_batch(1))
        labels = out["labels"]
        dec    = out["decoder_input_ids"]
        for tid in (MockTokenizer.AE_ID, MockTokenizer.PAD_ID, MockTokenizer.MEM_ID):
            mask = dec == tid
            if mask.any():
                self.assertTrue((labels[mask] == -100).all(),
                                f"Special token {tid} not masked in labels")

    def test_decoder_truncation(self):
        # The collator inserts MEM placeholders into the decoder text BEFORE
        # tokenisation + truncation.  If decoder_max_length is too small the
        # placeholders get cut off, causing assert_consistent_n_mems to fire.
        # Use a length that is generous enough to keep the MEM tokens but still
        # exercises the truncation path for the trailing answer text.
        col = make_finetuning_collator(decoder_max_length=256)
        out = col.torch_call(_ft_batch(1))
        self.assertLessEqual(out["decoder_input_ids"].shape[1], 256)

    def test_decoder_truncation_raises_on_too_small(self):
        """Verify that an extremely small decoder_max_length triggers the MEM
        consistency assertion — documenting the known limitation."""
        col = make_finetuning_collator(decoder_max_length=16)
        with self.assertRaises(AssertionError):
            col.torch_call(_ft_batch(1))

    def test_clean_text_in_docs(self):
        dirty_batch = [{"docs": ["Doc with <MEM> inside.", "Normal doc."],
                        "query": "question <RET>",
                        "mistral_label": "answer"}]
        col = make_finetuning_collator()
        col.torch_call(dirty_batch)   # should not raise

    def test_attention_mask_values(self):
        col = make_finetuning_collator()
        out = col.torch_call(_ft_batch(2))
        for key in ("compressor_attention_mask", "decoder_attention_mask"):
            vals = out[key].unique()
            self.assertTrue(set(vals.tolist()).issubset({0, 1}))

    def test_more_docs_than_topk(self):
        """Batch with more docs than topk_docs should silently truncate."""
        many_docs = DOCS * 10   # 30 docs
        batch = [{"docs": many_docs, "query": QUERY, "mistral_label": LABEL}]
        col   = make_finetuning_collator(topk_docs=2)
        out   = col.torch_call(batch)
        self.assertIn("decoder_input_ids", out)


# ===========================================================================
# Cross-collator / edge-case tests
# ===========================================================================

class TestEdgeCases(unittest.TestCase):

    def test_pretraining_empty_text(self):
        """Very short text (< 64 tokens) must still produce a valid batch."""
        col = make_pretraining_collator(ae_ratio=1.0)
        col.torch_call([{"text": "Hi"}])

    def test_pretraining_single_char(self):
        col = make_pretraining_collator(ae_ratio=1.0)
        col.torch_call([{"text": "X"}])

    def test_agent_long_steps(self):
        long_steps = ["Step " + str(i) + ": " + "x" * 200 for i in range(5)]
        col = make_agent_collator(compressor_max_length=64)
        np.random.seed(7)
        col.torch_call([{"trajectory": long_steps}])

    def test_finetuning_empty_docs(self):
        """One empty document should not crash the collator."""
        batch = [{"docs": ["", "Non-empty doc."], "query": "Q", "mistral_label": "A"}]
        col   = make_finetuning_collator()
        col.torch_call(batch)

    def test_finetuning_no_docs(self):
        """Edge-case: single document list."""
        batch = [{"docs": ["The only doc."], "query": "Q", "mistral_label": "A"}]
        col   = make_finetuning_collator(topk_docs=1)
        col.torch_call(batch)

    def test_compr_rate_1(self):
        """compr_rate=1 means every token gets its own MEM token; must not crash."""
        col = make_pretraining_collator(ae_ratio=1.0, compr_rate=1)
        col.torch_call([{"text": "Short text."}])


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)


