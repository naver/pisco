"""
PISCO
Copyright (c) 2026-present NAVER Corp.
All Rights Reserved.

Multimodal collators: document-page IMAGES and text both become compressor chunks that end
in `<MEM>` tokens; the decoder only ever sees the `<MEM>` placeholders (never pixels).

Kept in their own module so pisco.collator stays free of image dependencies. Wire them in a
config via `collator_class: pisco.mm_collator.MultimodalPretrainingCollator` etc.; train.py
passes `image_processor=model.image_processor` to any collator whose __init__ accepts it.

Batch contract (superset of BaseCollator's): the five standard keys, plus — only when the
batch holds at least one image — `pixel_values`, `image_grid_thw`, `mm_token_type_ids`
(see pisco.mm_utils.collate_compressor). The `<MEM>` invariant is asserted as everywhere.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from pisco.collator import PretrainingCollator, FineTuningCollator
from pisco.collator_utils import (
    chunk_list,
    chunk_random_no_tiny_tail,
    mask_before_mem,
)
from pisco import mm_tasks
from pisco.mm_utils import (
    MM_TOKENS,
    MMChunkBatch,
    collate_compressor,
    get_vision_token_ids,
    merge_size_of,
    n_image_tokens,
    segments_to_chunks,
)


def _as_pil(x):
    """datasets.Image() decodes lazily to PIL; also accept {'bytes':...} rows and None."""
    if x is None:
        return None
    from pisco.mm_utils import to_rgb
    return to_rgb(x)


class MultimodalPretrainingCollator(PretrainingCollator):
    """
    Stage-2 pretraining on (page image, page text) pairs. Rows: {"image": PIL|None, "text": str}.

    Per example one task is drawn (weights renormalised over the available ones; rows with
    image=None always get a text task):
      - image autoencoding  (p_image_ae):   compress the IMAGE, decoder reproduces the page
        TEXT after <AE>:   <AE> <MEM>*n <bos> text[:ae_target_max_tokens] <eos>
      - image continuation (p_image_cont): compress the IMAGE, decoder gets a short clear
        prefix of the text and continues it:  <bos> text[:s] <MEM>*n text[s:s+L] <eos>
      - text task          (p_text):       the parent's text autoencoding / continuation on
        the page text (ratio `ae_ratio`), i.e. the forgetting guard for text compression.
      CHAT tasks (finetuning's RAG template, prompts in pisco.mm_tasks; 0 by default):
      - description        (p_desc): 1..multi_max_docs images ("Document j:" + <MEM>s; the other images are
        the batch's other rows), question "Describe Document j in detail.", answer = row["teacher_text"]
        (the uncompressed backbone's own description, scripts/teacher_describe_vllm.py). Rows without a
        teacher_text fall back to image autoencoding.
      - find               (p_find): 2..multi_max_docs images, question 'Which document contains this text:
        "<snippet>"?' with a 6-12 word span of the target row's text, answer "Document j". Teaches the decoder
        to address one <MEM> block among several (multi-page QA, frame memory).
      Chat rows are masked by prompt length (only the answer is a target), the others by mask_before_mem.

    teacher_inputs=True (KL to the uncompressed backbone, PISCOConfig.kd_weight): chat rows also get their
    uncompressed prompt (raw image spans instead of <MEM>s, same answer suffix) as teacher_* tensors, and
    `teacher_kd_rows` (B,) marks them; the other rows get a one-token dummy prompt and no KL.

    The image is one atomic compressor chunk; `compressor_max_length` only bounds text chunks.
    """

    def __init__(
        self,
        compressor_tokenizer,
        decoder_tokenizer,
        compr_rate,
        compressor_max_length=256,
        decoder_max_length=1024,
        *,
        image_processor=None,
        ae_ratio: float = 0.5,
        p_image_ae: float = 0.5,
        p_text: float = 0.3,
        p_image_cont: float = 0.2,
        ae_target_max_tokens: int = 768,
        cont_target_max_tokens: int = 512,
        image_compr_rate: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = 64,
        p_desc: float = 0.0,
        p_find: float = 0.0,
        multi_max_docs: int = 4,
        desc_target_max_tokens: int = 512,
        find_snippet_words: tuple = (6, 12),
        teacher_inputs: bool = False,
        teacher_max_length: int = 6144,
        **kwargs,
    ):
        super().__init__(
            compressor_tokenizer=compressor_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            compr_rate=compr_rate,
            compressor_max_length=compressor_max_length,
            decoder_max_length=decoder_max_length,
            ae_ratio=ae_ratio,
            **kwargs,
        )
        if image_processor is None:
            raise ValueError("MultimodalPretrainingCollator needs image_processor= (model.image_processor)")
        self.multi_max_docs = int(multi_max_docs)
        self.desc_target_max_tokens = int(desc_target_max_tokens)
        self.find_snippet_words = tuple(int(x) for x in find_snippet_words)
        self.teacher_inputs = bool(teacher_inputs)
        self.teacher_max_length = int(teacher_max_length)
        if self.teacher_inputs:
            self.decoder_vision_ids = get_vision_token_ids(decoder_tokenizer)
        self.image_processor = image_processor
        self.vision_ids = get_vision_token_ids(compressor_tokenizer)
        self.image_compr_rate = image_compr_rate
        self.ae_target_max_tokens = ae_target_max_tokens
        self.cont_target_max_tokens = cont_target_max_tokens
        # Qwen3.5's Triton (GatedDeltaNet) kernels re-tune for every new (batch, length)
        # shape (30-90 s each, measured); padding to a multiple keeps the shape set small.
        self.pad_to_multiple_of = pad_to_multiple_of
        total = p_image_ae + p_text + p_image_cont + p_desc + p_find
        assert total > 0, "all task probabilities are zero"
        self.task_probs = {
            "image_ae": p_image_ae / total,
            "text": p_text / total,
            "image_cont": p_image_cont / total,
        }
        # chat tasks only enter the draw when enabled: an old config keeps its exact RNG stream
        if p_desc > 0:
            self.task_probs["desc"] = p_desc / total
        if p_find > 0:
            self.task_probs["find"] = p_find / total
        print(f"[mm-pretrain] task mix {self.task_probs}, image_compr_rate={image_compr_rate}")

    # ------------------------------------------------------------------ tasks
    def _image_chunks(self, image) -> MMChunkBatch:
        return segments_to_chunks(
            [("image", image)],
            compressor_tokenizer=self.compressor_tokenizer,
            image_processor=self.image_processor,
            compr_rate=self.compr_rate,
            image_compr_rate=self.image_compr_rate,
            text_chunker=lambda ids: [ids],
            vision_ids=self.vision_ids,
        )

    def prepare_for_image_autoencoding(self, image, text_ids: List[int]):
        chunks = self._image_chunks(image)
        target = self.compressor_tokenizer.decode(text_ids[: self.ae_target_max_tokens])
        decoder_text = (
            self.decoder_tokenizer.ae_token
            + self.decoder_tokenizer.mem_token * chunks.total_mems
            + self.decoder_tokenizer.bos_token
            + target
            + self.decoder_tokenizer.eos_token
        )
        return chunks, decoder_text

    def prepare_for_image_continuation(self, image, text_ids: List[int]):
        chunks = self._image_chunks(image)
        s = int(np.random.randint(0, 32))
        s = min(s, len(text_ids) // 2)
        clear, future = text_ids[:s], text_ids[s : s + self.cont_target_max_tokens]
        decoder_text = (
            self.decoder_tokenizer.bos_token
            + self.compressor_tokenizer.decode(clear)
            + self.decoder_tokenizer.mem_token * chunks.total_mems
            + self.compressor_tokenizer.decode(future)
            + self.decoder_tokenizer.eos_token
        )
        return chunks, decoder_text

    def _text_task(self, text: str, text_ids: List[int]) -> MMChunkBatch:
        if np.random.uniform() < self.ae_ratio:
            ids_list, decoder_text = self.prepare_for_autoencoding(text=text, text_ids=text_ids)
        else:
            ids_list, decoder_text = self.prepare_for_text_continuation(text_ids=text_ids)
        b = MMChunkBatch()
        for ids in ids_list:
            b.input_ids.append(ids)
            b.n_mems.append(int(sum(1 for t in ids if t == self.compressor_tokenizer.mem_token_id)))
            b.is_image.append(False)
        return b, decoder_text

    # ------------------------------------------------------------------ chat tasks
    def _chat_texts(self, system: str, documents: str, question: str, answer: str):
        """(full chat with the answer, number of tokens before the answer)."""
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": mm_tasks.user_content(documents, question)},
                {"role": "assistant", "content": answer}]
        full = self.decoder_tokenizer.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
        prefix = self.decoder_tokenizer.apply_chat_template(
            msgs[:-1], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        prefix_ids = prefix if isinstance(prefix, list) else prefix["input_ids"]
        return full, len(prefix_ids)

    def _find_snippet(self, text: str) -> Optional[str]:
        words = text.split()
        lo, hi = self.find_snippet_words
        if len(words) < hi + 2:
            return None
        n = int(np.random.randint(lo, hi + 1))
        start = int(np.random.randint(0, min(len(words) - n, 300) + 1))
        return " ".join(words[start : start + n])

    def _chat_example(self, i: int, task: str, texts: List[str], images: List[Any], teacher: List[str]):
        """One desc/find example for row i, the other rows' images as distractors. Returns
        (chunks, decoder_text, prefix_len, teacher_text or None), or None when the row cannot host the
        task (no teacher description, text too short, no distractor) -> the caller falls back to image_ae."""
        others = [k for k, im in enumerate(images) if k != i and im is not None]
        if task == "desc":
            answer = teacher[i]
            if not answer:
                return None
            a_ids = self.decoder_tokenizer(answer, add_special_tokens=False)["input_ids"]
            if len(a_ids) > self.desc_target_max_tokens:
                answer = self.decoder_tokenizer.decode(a_ids[: self.desc_target_max_tokens])
            n_docs = 1 if np.random.uniform() < 0.5 else int(np.random.randint(2, self.multi_max_docs + 1))
            system = mm_tasks.DESC_SYSTEM_PROMPT
            snippet = None
        else:
            snippet = self._find_snippet(teacher[i] or texts[i])
            if snippet is None:
                return None
            others = [k for k in others if snippet not in texts[k] and snippet not in teacher[k]]
            if not others:
                return None
            n_docs = int(np.random.randint(2, self.multi_max_docs + 1))
            system = mm_tasks.FIND_SYSTEM_PROMPT
            answer = None
        n_docs = min(n_docs, 1 + len(others))
        distract = [int(k) for k in np.random.choice(others, size=n_docs - 1, replace=False)] if n_docs > 1 else []
        pos = int(np.random.randint(0, n_docs))
        chunks_of = {k: self._image_chunks(images[k]) for k in [i] + distract}
        merge = merge_size_of(self.image_processor)
        while True:
            pos = min(pos, len(distract))
            order = distract[:pos] + [i] + distract[pos:]
            docs, t_docs = "", ""
            for d, k in enumerate(order):
                c = chunks_of[k]
                docs += f"Document {d}:" + self.decoder_tokenizer.mem_token * c.total_mems
                if self.teacher_inputs:
                    n_vis = n_image_tokens(c.image_grid_thw[0], merge)
                    t_docs += (f"Document {d}:" + MM_TOKENS["vision_start"] + MM_TOKENS["image_pad"] * n_vis
                               + MM_TOKENS["vision_end"])
            question = mm_tasks.desc_query(pos) if task == "desc" else mm_tasks.find_query(snippet)
            ans = answer if task == "desc" else mm_tasks.find_answer(pos)
            full, plen = self._chat_texts(system, docs, question, ans)
            n_tok = len(self.decoder_tokenizer(full, add_special_tokens=False)["input_ids"])
            if n_tok <= self.decoder_max_length or not distract:
                break
            distract = distract[:-1]          # over the decoder budget: drop a distractor, keep the target
            if task == "find" and not distract:
                return None
        b = MMChunkBatch()
        for k in order:
            b.extend(chunks_of[k])
        t_full = self._chat_texts(system, t_docs, question, ans)[0] if self.teacher_inputs else None
        return b, full, plen, t_full

    # ------------------------------------------------------------------ batch
    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [self.clean_text(x["text"]) for x in examples]
        images = [_as_pil(x.get("image")) for x in examples]
        teacher = [self.clean_text((x.get("teacher_text") or "").strip()) for x in examples]

        pre = self.compressor_tokenizer(
            texts, padding="do_not_pad", return_tensors=None, truncation=False
        )["input_ids"]

        names = list(self.task_probs)
        probs = np.array([self.task_probs[n] for n in names])
        # A degenerate mix (one task at p=1) must not consume randomness, so that
        # p_text=1 reproduces PretrainingCollator's RNG stream exactly.
        fixed_task = names[int(probs.argmax())] if probs.max() >= 1.0 else None

        batches: List[MMChunkBatch] = []
        decoder_texts: List[str] = []
        prefix_lengths: List[Optional[int]] = []      # None = not a chat row (mask_before_mem)
        teacher_texts: List[Optional[str]] = []       # None = no KL for this row
        for idx, (text, ids, image) in enumerate(zip(texts, pre, images)):
            if image is None:
                task = "text"
            elif fixed_task is not None:
                task = fixed_task
            else:
                task = str(np.random.choice(names, p=probs))
            chat = self._chat_example(idx, task, texts, images, teacher) if task in ("desc", "find") else None
            if task in ("desc", "find") and chat is None:
                task = "image_ae"
            plen, t_text = None, None
            if chat is not None:
                b, dec, plen, t_text = chat
            elif task == "image_ae":
                b, dec = self.prepare_for_image_autoencoding(image, ids)
            elif task == "image_cont":
                b, dec = self.prepare_for_image_continuation(image, ids)
            else:
                b, dec = self._text_task(text, ids)
            batches.append(b)
            decoder_texts.append(dec)
            prefix_lengths.append(plen)
            teacher_texts.append(t_text)

        compressor_inputs = collate_compressor(batches, self, self.vision_ids)

        decoder_inputs = self.decoder_tokenizer(
            decoder_texts,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
            max_length=self.decoder_max_length,
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        labels = decoder_inputs["input_ids"].clone()
        # Order matters: mask_before_mem first (it looks for <MEM> ids, which
        # mask_special_tokens erases). The base PretrainingCollator has the reverse order
        # and therefore trains on the clear prefix of continuation rows; the MultiTask
        # collator and this one use the corrected order (context is not a target).
        labels = mask_before_mem(labels, mem_token_id=self.decoder_tokenizer.mem_token_id)
        for i, plen in enumerate(prefix_lengths):     # chat rows: only the answer is a target (left padding)
            if plen is not None:
                n_pad = int((decoder_inputs["attention_mask"][i] == 0).sum())
                row = decoder_inputs["input_ids"][i].clone()
                row[: n_pad + plen] = -100
                labels[i] = row
        labels = self.mask_special_tokens(labels)

        self.assert_consistent_n_mems(
            {"input_ids": compressor_inputs["compressor_input_ids"]}, decoder_inputs
        )

        out = {
            "compressor_input_ids": compressor_inputs["compressor_input_ids"],
            "compressor_attention_mask": compressor_inputs["compressor_attention_mask"],
            "decoder_input_ids": decoder_inputs["input_ids"],
            "decoder_attention_mask": decoder_inputs["attention_mask"],
            "labels": labels,
        }
        for k in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
            if k in compressor_inputs:
                out[k] = compressor_inputs[k]
        if self.teacher_inputs:
            out.update(self._teacher_batch(teacher_texts, labels, batches))
        return out

    def _teacher_batch(self, teacher_texts: List[Optional[str]], labels: torch.Tensor,
                       batches: List[MMChunkBatch]) -> Dict[str, Any]:
        """Uncompressed prompts of the chat rows (a one-token dummy for the others), their pixels in
        prompt order, and `teacher_kd_rows`. Checks: no image span truncated, KL rows' answer tails
        coincide with the decoder's at the same offsets from the end (both left padded)."""
        kd_rows = torch.tensor([t is not None for t in teacher_texts], dtype=torch.bool)
        t_inputs = self.decoder_tokenizer(
            [t if t is not None else self.decoder_tokenizer.eos_token for t in teacher_texts],
            return_tensors="pt", padding="longest", add_special_tokens=False,
            max_length=self.teacher_max_length, truncation=True, pad_to_multiple_of=self.pad_to_multiple_of,
        )
        t_ids = t_inputs["input_ids"]
        out = {"teacher_input_ids": t_ids, "teacher_attention_mask": t_inputs["attention_mask"],
               "teacher_kd_rows": kd_rows}
        pvs, grids = [], []
        for i, t in enumerate(teacher_texts):
            if t is not None:
                pvs.extend(batches[i].pixel_values)
                grids.extend(batches[i].image_grid_thw)
        _, ip, _ = self.decoder_vision_ids
        if pvs:
            merge = merge_size_of(self.image_processor)
            expected = sum(n_image_tokens(g, merge) for g in grids)
            got = int((t_ids == ip).sum())
            assert got == expected, (
                f"teacher prompts hold {got} <|image_pad|> tokens but their images need {expected}: "
                f"raise teacher_max_length ({self.teacher_max_length}) so no image span is truncated")
            out["teacher_pixel_values"] = torch.from_numpy(np.concatenate(pvs, axis=0))
            out["teacher_image_grid_thw"] = torch.tensor(grids, dtype=torch.long)
            out["teacher_mm_token_type_ids"] = (t_ids == ip).long()
        lab_kd = labels.masked_fill(~kd_rows.unsqueeze(1), -100)
        K = min(lab_kd.shape[1], t_ids.shape[1])
        tail = lab_kd[:, -K:]
        sup = tail != -100
        if bool(sup.any()):
            assert torch.equal(t_ids[:, -K:][sup], tail[sup]), (
                "teacher/decoder answer tails differ (truncated teacher prompt or tokenizer drift)")
        return out


class MultimodalFineTuningCollator(FineTuningCollator):
    """
    RAG finetuning where each retrieved document is a page IMAGE or a text passage.
    Rows: docs: List[str] (text, "" when the doc is an image), doc_images: List[PIL|None]
    (optional column; missing -> all-text, identical to FineTuningCollator), query,
    mistral_label, optional system_prompt.

    Decoder prompt is the parent's: "Document j:" + <MEM>*n_j per doc inside the RAG chat
    template; label masking (prefix length) is inherited unchanged.

    teacher_inputs=True (online distillation, PISCOConfig.kd_weight): the batch also carries
    the UNCOMPRESSED prompt for the same rows -- identical chat template and answer suffix,
    but each page is its raw image span `<|vision_start|> <|image_pad|>*N <|vision_end|>`
    (text docs stay raw text) -- as `teacher_input_ids` / `teacher_attention_mask` and, when
    images are present, `teacher_pixel_values` / `teacher_image_grid_thw` /
    `teacher_mm_token_type_ids` (pixels are the same tensors as the compressor's: same
    images, same order). Left padded like the decoder prompt, so the answer occupies the
    last columns of both; the collator asserts those tails coincide token for token.
    `teacher_max_length` must hold the raw pages (4 x ~440 visual tokens + prompt + answer);
    truncating an image span is refused.
    """

    def __init__(
        self,
        compressor_tokenizer,
        decoder_tokenizer,
        compr_rate,
        compressor_max_length=256,
        decoder_max_length=1024,
        *,
        image_processor=None,
        image_compr_rate: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = 64,
        teacher_inputs: bool = False,
        teacher_max_length: int = 4096,
        **kwargs,
    ):
        super().__init__(
            compressor_tokenizer=compressor_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            compr_rate=compr_rate,
            compressor_max_length=compressor_max_length,
            decoder_max_length=decoder_max_length,
            **kwargs,
        )
        if image_processor is None:
            raise ValueError("MultimodalFineTuningCollator needs image_processor= (model.image_processor)")
        self.image_processor = image_processor
        self.vision_ids = get_vision_token_ids(compressor_tokenizer)
        self.image_compr_rate = image_compr_rate
        self.pad_to_multiple_of = pad_to_multiple_of  # see MultimodalPretrainingCollator
        self.teacher_inputs = teacher_inputs
        self.teacher_max_length = teacher_max_length
        if teacher_inputs:
            # the decoder tokenizer must know the vision tokens (it does for a VLM backbone)
            self.decoder_vision_ids = get_vision_token_ids(decoder_tokenizer)
            print("Teacher (uncompressed) inputs enabled; teacher max length", teacher_max_length)

    def _text_chunker(self, ids: List[int]) -> List[List[int]]:
        if self.chunk_docs:
            return chunk_list(
                ids, chunk_length=self.compressor_max_length, chunk_overlap=self.chunk_overlap
            )[: self.n_max_chunks]
        return [ids[: self.compressor_max_length]]

    def doc_segments(self, docs: List[str], doc_images: List[Any], query: Optional[str]):
        """One (segments, image_prefix_ids) pair per kept doc."""
        segs = []
        for j, doc in enumerate(docs):
            img = doc_images[j] if j < len(doc_images) else None
            if img is not None:
                segs.append(("image", img))
            else:
                segs.append(("text", doc))
        prefix_ids = None
        if self.query_dependent and query is not None:
            # OSCAR-style: image spans are preceded by the query (text docs already got it
            # via preprend_query_to_docs in the parent).
            prefix_ids = self.compressor_tokenizer(
                "Query: " + query + "\n Document: ", add_special_tokens=False
            )["input_ids"]
        return segs, prefix_ids

    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        documents = [list(elt["docs"]) for elt in examples]
        doc_images = [
            [_as_pil(x) for x in (elt.get("doc_images") or [])] for elt in examples
        ]
        queries = [self.clean_text(elt["query"]) for elt in examples]
        labels_txt = [self.clean_text(elt["mistral_label"]) for elt in examples]
        sys_prompts = [elt.get("system_prompt") for elt in examples]

        raw_documents = documents   # the teacher reads the docs as-is (its prompt has the query)
        if self.query_dependent:
            documents = self.preprend_query_to_docs(queries, documents)

        batches: List[MMChunkBatch] = []
        decoder_texts: List[str] = []
        prefix_lengths: List[int] = []
        teacher_texts: List[str] = []
        merge = merge_size_of(self.image_processor)

        for i, docs in enumerate(documents):
            if self.topk_docs is not None:
                docs = docs[: self.topk_docs]
            imgs = doc_images[i][: len(docs)]
            segs, prefix_ids = self.doc_segments(docs, imgs, queries[i] if self.query_dependent else None)

            example_batch = MMChunkBatch()
            doc_text = ""
            teacher_text = ""
            for j, seg in enumerate(segs):
                b = segments_to_chunks(
                    [seg],
                    compressor_tokenizer=self.compressor_tokenizer,
                    image_processor=self.image_processor,
                    compr_rate=self.compr_rate,
                    image_compr_rate=self.image_compr_rate,
                    text_chunker=self._text_chunker,
                    vision_ids=self.vision_ids,
                    clean_text=self.clean_text,
                    image_prefix_ids=prefix_ids,
                )
                example_batch.extend(b)
                doc_text += f"Document {j}:" + self.decoder_tokenizer.mem_token * b.total_mems
                if self.teacher_inputs:
                    if seg[0] == "image":
                        n_vis = n_image_tokens(b.image_grid_thw[0], merge)
                        teacher_text += (f"Document {j}:" + MM_TOKENS["vision_start"]
                                         + MM_TOKENS["image_pad"] * n_vis + MM_TOKENS["vision_end"])
                    else:
                        teacher_text += f"Document {j}:" + self.clean_text(raw_documents[i][j])
            batches.append(example_batch)

            prompt, prefix_length = self.compute_prompt_and_prefix_length(
                doc_text, queries[i], labels_txt[i], system_prompt=sys_prompts[i]
            )
            decoder_texts.append(prompt)
            prefix_lengths.append(prefix_length)
            if self.teacher_inputs:
                t_prompt, _ = self.compute_prompt_and_prefix_length(
                    teacher_text, queries[i], labels_txt[i], system_prompt=sys_prompts[i]
                )
                teacher_texts.append(t_prompt)

        compressor_inputs = collate_compressor(batches, self, self.vision_ids)

        decoder_inputs = self.decoder_tokenizer(
            decoder_texts,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
            max_length=self.decoder_max_length,
            truncation=True,
            enable_thinking=False,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        labels = decoder_inputs["input_ids"].clone()
        labels = self.mask_labels_before_prefix(labels, prefix_lengths)
        labels = self.mask_special_tokens(labels)

        self.assert_consistent_n_mems(
            {"input_ids": compressor_inputs["compressor_input_ids"]}, decoder_inputs
        )

        out = {
            "compressor_input_ids": compressor_inputs["compressor_input_ids"],
            "compressor_attention_mask": compressor_inputs["compressor_attention_mask"],
            "decoder_input_ids": decoder_inputs["input_ids"],
            "decoder_attention_mask": decoder_inputs["attention_mask"],
            "labels": labels,
        }
        for k in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
            if k in compressor_inputs:
                out[k] = compressor_inputs[k]
        if self.teacher_inputs:
            out.update(self._teacher_batch(teacher_texts, labels, compressor_inputs))
        return out

    def _teacher_batch(self, teacher_texts: List[str], labels: torch.Tensor, compressor_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize the uncompressed prompts and check the two contracts the KD loss relies on:
        every image span is intact (no truncation) and the answer tails match the decoder's."""
        t_inputs = self.decoder_tokenizer(
            teacher_texts,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
            max_length=self.teacher_max_length,
            truncation=True,
            enable_thinking=False,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        t_ids = t_inputs["input_ids"]
        _, ip, _ = self.decoder_vision_ids
        out = {"teacher_input_ids": t_ids, "teacher_attention_mask": t_inputs["attention_mask"]}
        if "pixel_values" in compressor_inputs:
            grid = compressor_inputs["image_grid_thw"]
            expected = int((grid.prod(-1) // (merge_size_of(self.image_processor) ** 2)).sum())
            got = int((t_ids == ip).sum())
            assert got == expected, (
                f"teacher prompt holds {got} <|image_pad|> tokens but the pages need {expected}: "
                f"raise teacher_max_length ({self.teacher_max_length}) so no image span is truncated")
            out["teacher_pixel_values"] = compressor_inputs["pixel_values"]
            out["teacher_image_grid_thw"] = grid
            out["teacher_mm_token_type_ids"] = (t_ids == ip).long()
        # answer tails: same tokens at the same offsets from the end (both left padded)
        K = min(labels.shape[1], t_ids.shape[1])
        lab_tail = labels[:, -K:]
        sup = lab_tail != -100
        assert bool(sup.any()), "no supervised answer tokens"
        assert torch.equal(t_ids[:, -K:][sup], lab_tail[sup]), (
            "teacher/decoder answer tails differ (truncated teacher prompt or tokenizer drift)")
        return out
