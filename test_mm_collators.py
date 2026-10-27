"""
Tests for the multimodal PISCO collators (pisco.mm_collator) and helpers (pisco.mm_utils).

Unlike test_collators.py (which inlines stub copies), these import the REAL modules and
only mock the tokenizers and the image processor, so they exercise the actual chunking,
<MEM> bookkeeping, pixel/grid ordering and label masking.

Run:  python test_mm_collators.py          (or pytest test_mm_collators.py -v)
"""

import unittest
from typing import List

import numpy as np
import torch
from PIL import Image

from pisco.collator import PretrainingCollator
from pisco.mm_collator import MultimodalFineTuningCollator, MultimodalPretrainingCollator
from pisco.mm_utils import (
    MM_TOKENS,
    collate_compressor,
    image_to_span,
    n_image_tokens,
    segments_to_chunks,
)

# ----------------------------------------------------------------------------- mocks
PAD, BOS, EOS, MEM, AE, VS, IP, VE, UNK = 0, 1, 2, 3, 4, 5, 6, 7, 8
FIRST_WORD = 100
MERGE = 2


class MockTokenizer:
    """Whitespace tokenizer. Special-token strings map to fixed ids; everything else to a
    stable id >= 100. Mimics the transformers API surface the collators use."""

    specials = {
        "<pad>": PAD, "<bos>": BOS, "<eos>": EOS, "<MEM>": MEM, "<AE>": AE,
        MM_TOKENS["vision_start"]: VS, MM_TOKENS["image_pad"]: IP, MM_TOKENS["vision_end"]: VE,
    }

    def __init__(self):
        self.pad_token_id, self.bos_token_id, self.eos_token_id = PAD, BOS, EOS
        self.pad_token, self.bos_token, self.eos_token = "<pad>", "<bos>", "<eos>"
        self.mem_token, self.mem_token_id = "<MEM>", MEM
        self.ae_token, self.ae_token_id = "<AE>", AE
        self.unk_token_id = UNK
        self.padding_side = "left"
        self.vocab = {}
        self.inv = {v: k for k, v in self.specials.items()}
        self.chat_template = "x"

    def _split(self, text: str) -> List[str]:
        out, buf = [], ""
        i = 0
        while i < len(text):
            if text[i] == "<":
                j = text.find(">", i)
                tok = text[i : j + 1]
                if tok in self.specials:
                    if buf.strip():
                        out.extend(buf.split())
                    buf = ""
                    out.append(tok)
                    i = j + 1
                    continue
            buf += text[i]
            i += 1
        if buf.strip():
            out.extend(buf.split())
        return out

    def _encode(self, text: str) -> List[int]:
        ids = []
        for w in self._split(text):
            if w in self.specials:
                ids.append(self.specials[w])
            else:
                if w not in self.vocab:
                    self.vocab[w] = FIRST_WORD + len(self.vocab)
                    self.inv[self.vocab[w]] = w
                ids.append(self.vocab[w])
        return ids

    def __call__(self, texts, padding="longest", return_tensors=None, truncation=False,
                 max_length=None, add_special_tokens=True, pad_to_multiple_of=None, **kw):
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        ids = [self._encode(t) for t in batch]
        if truncation and max_length is not None:
            ids = [x[:max_length] for x in ids]
        if return_tensors == "pt":
            return self.pad({"input_ids": ids}, return_tensors="pt", pad_to_multiple_of=pad_to_multiple_of)
        return {"input_ids": ids[0] if single else ids}

    def pad(self, features, padding="longest", return_tensors=None, pad_to_multiple_of=None):
        ids = features["input_ids"]
        L = max(len(x) for x in ids)
        if pad_to_multiple_of:
            L = ((L + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        inp = torch.full((len(ids), L), PAD, dtype=torch.long)
        att = torch.zeros((len(ids), L), dtype=torch.long)
        for i, x in enumerate(ids):           # left padding
            inp[i, L - len(x):] = torch.tensor(x)
            att[i, L - len(x):] = 1
        return {"input_ids": inp, "attention_mask": att}

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(self.inv.get(int(i), "?") for i in ids)

    def convert_tokens_to_ids(self, toks):
        return [self.specials.get(t, UNK) for t in toks]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
        s = ""
        for m in messages:
            s += f"<{m['role']}> {m['content']} <eot> "
        if add_generation_prompt:
            s += "<assistant> "
        if tokenize:
            return {"input_ids": self._encode(s)}
        return s


class FakeImageProcessor:
    """Image (w, h) -> grid (1, 2h/32*?): we define one visual token per 32x32 px so that
    a 64x64 image is 4 tokens, 96x64 is 6 tokens, etc. pixel_values rows = tokens * MERGE^2,
    filled with the image's mean value so ordering can be checked."""

    merge_size = MERGE
    patch_size = 16

    def __init__(self):
        self.calls = 0

    def __call__(self, images, return_tensors="np"):
        self.calls += len(images)
        pvs, grids = [], []
        for im in images:
            w, h = im.size
            gh, gw = (h // 32) * MERGE, (w // 32) * MERGE
            n_patches = gh * gw
            fill = float(np.asarray(im).mean())
            pvs.append(np.full((n_patches, 1536), fill, dtype=np.float32))
            grids.append([1, gh, gw])
        return {"pixel_values": np.concatenate(pvs, 0), "image_grid_thw": np.array(grids)}


def make_image(w=64, h=64, value=128):
    return Image.fromarray(np.full((h, w, 3), value, dtype=np.uint8))


def make_pretrain_collator(**kw):
    kw.setdefault("pad_to_multiple_of", None)
    tok = MockTokenizer()
    return MultimodalPretrainingCollator(
        compressor_tokenizer=tok, decoder_tokenizer=MockTokenizer(), compr_rate=4,
        compressor_max_length=16, decoder_max_length=256, image_processor=FakeImageProcessor(),
        **kw,
    )


def make_ft_collator(**kw):
    kw.setdefault("pad_to_multiple_of", None)
    return MultimodalFineTuningCollator(
        compressor_tokenizer=MockTokenizer(), decoder_tokenizer=MockTokenizer(), compr_rate=4,
        compressor_max_length=16, decoder_max_length=256, image_processor=FakeImageProcessor(),
        **kw,
    )


STANDARD_KEYS = {"compressor_input_ids", "compressor_attention_mask", "decoder_input_ids",
                 "decoder_attention_mask", "labels"}
MM_KEYS = {"pixel_values", "image_grid_thw", "mm_token_type_ids"}
TEACHER_KEYS = {"teacher_input_ids", "teacher_attention_mask", "teacher_pixel_values",
                "teacher_image_grid_thw", "teacher_mm_token_type_ids"}


# ----------------------------------------------------------------------------- tests
class TestMMUtils(unittest.TestCase):
    def test_image_to_span_counts(self):
        tok = MockTokenizer()
        span, pv, grid = image_to_span(make_image(96, 64), FakeImageProcessor(), (VS, IP, VE))
        n = n_image_tokens(grid, MERGE)
        self.assertEqual(n, 6)
        self.assertEqual(span, [VS] + [IP] * 6 + [VE])
        self.assertEqual(pv.shape, (6 * MERGE ** 2, 1536))

    def test_segments_mixed_and_mem_rule(self):
        tok = MockTokenizer()
        b = segments_to_chunks(
            [("text", "a b c d e f g h i j"), ("image", make_image(64, 64))],
            compressor_tokenizer=tok, image_processor=FakeImageProcessor(), compr_rate=4,
            text_chunker=lambda ids: [ids], vision_ids=(VS, IP, VE),
        )
        self.assertEqual(b.is_image, [False, True])
        self.assertEqual(b.n_mems[0], 10 // 4 + 1)
        self.assertEqual(b.n_mems[1], (4 + 2) // 4 + 1)       # span len 6 -> 2 MEM
        self.assertTrue(b.input_ids[1][-b.n_mems[1]:] == [MEM] * b.n_mems[1])  # MEMs at the tail

    def test_image_compr_rate_override(self):
        tok = MockTokenizer()
        img = make_image(128, 128)                              # 16 tokens, span 18
        b4 = segments_to_chunks([("image", img)], compressor_tokenizer=tok,
                                image_processor=FakeImageProcessor(), compr_rate=4,
                                text_chunker=lambda i: [i], vision_ids=(VS, IP, VE))
        b9 = segments_to_chunks([("image", img)], compressor_tokenizer=tok,
                                image_processor=FakeImageProcessor(), compr_rate=4,
                                image_compr_rate=9, text_chunker=lambda i: [i], vision_ids=(VS, IP, VE))
        self.assertEqual(b4.n_mems, [18 // 4 + 1])
        self.assertEqual(b9.n_mems, [18 // 9 + 1])

    def test_collate_alignment_and_order(self):
        coll = make_pretrain_collator()
        tok = coll.compressor_tokenizer
        imgs = [make_image(64, 64, 10), make_image(96, 64, 200)]
        batches = [
            segments_to_chunks([("image", imgs[0]), ("text", "x y z")], compressor_tokenizer=tok,
                               image_processor=coll.image_processor, compr_rate=4,
                               text_chunker=lambda i: [i], vision_ids=(VS, IP, VE)),
            segments_to_chunks([("image", imgs[1])], compressor_tokenizer=tok,
                               image_processor=coll.image_processor, compr_rate=4,
                               text_chunker=lambda i: [i], vision_ids=(VS, IP, VE)),
        ]
        out = collate_compressor(batches, coll, (VS, IP, VE))
        ids, tt = out["compressor_input_ids"], out["mm_token_type_ids"]
        self.assertEqual(ids.shape, tt.shape)
        # per-row image_pad counts: row0 image (4), row1 text (0), row2 image (6)
        self.assertEqual(tt.sum(1).tolist(), [4, 0, 6])
        self.assertEqual(out["image_grid_thw"].tolist(), [[1, 4, 4], [1, 4, 6]])
        # pixel rows: first 16 rows belong to image 0 (fill 10), next 24 to image 1 (fill 200)
        pv = out["pixel_values"]
        self.assertEqual(pv.shape[0], 16 + 24)
        self.assertTrue(torch.all(pv[:16] == 10) and torch.all(pv[16:] == 200))
        # mm_token_type_ids is 1 exactly where ids == image_pad (after left padding)
        self.assertTrue(torch.equal(tt, (ids == IP).long()))

    def test_collate_text_only_has_no_mm_keys(self):
        coll = make_pretrain_collator()
        b = segments_to_chunks([("text", "p q r s")], compressor_tokenizer=coll.compressor_tokenizer,
                               image_processor=None, compr_rate=4, text_chunker=lambda i: [i],
                               vision_ids=(VS, IP, VE))
        out = collate_compressor([b], coll, (VS, IP, VE))
        self.assertEqual(set(out), {"compressor_input_ids", "compressor_attention_mask"})


class TestMultimodalPretrainingCollator(unittest.TestCase):
    def _mem_invariant(self, batch):
        n_c = int((batch["compressor_input_ids"] == MEM).sum())
        n_d = int((batch["decoder_input_ids"] == MEM).sum())
        self.assertEqual(n_c, n_d)
        return n_c

    def test_image_ae_batch(self):
        coll = make_pretrain_collator(p_image_ae=1.0, p_text=0.0, p_image_cont=0.0)
        rows = [{"image": make_image(64, 64), "text": "the quick brown fox jumps"},
                {"image": make_image(96, 96), "text": "lorem ipsum dolor sit amet consectetur"}]
        b = coll.torch_call(rows)
        self.assertEqual(set(b), STANDARD_KEYS | MM_KEYS)
        n = self._mem_invariant(b)
        self.assertEqual(n, (6 // 4 + 1) + (11 // 4 + 1))        # 64px: span 6; 96px: 9 tok -> span 11
        # labels: <AE>, MEMs, bos masked; the text target is NOT masked; eos kept
        lab = b["labels"]
        self.assertTrue(((lab == -100) | (lab >= FIRST_WORD) | (lab == EOS)).all())
        self.assertGreater(int((lab >= FIRST_WORD).sum()), 0)
        for row in b["decoder_input_ids"]:
            real = row[row != PAD].tolist()
            self.assertEqual(real[0], AE)

    def test_image_cont_batch(self):
        coll = make_pretrain_collator(p_image_ae=0.0, p_text=0.0, p_image_cont=1.0)
        rows = [{"image": make_image(64, 64), "text": " ".join(f"w{i}" for i in range(40))}]
        b = coll.torch_call(rows)
        self._mem_invariant(b)
        self.assertIn("pixel_values", b)
        # everything up to and including the last MEM is masked (mask_before_mem)
        row, lab = b["decoder_input_ids"][0], b["labels"][0]
        last_mem = (row == MEM).nonzero().max().item()
        self.assertTrue((lab[: last_mem + 1] == -100).all())
        self.assertGreater(int((lab[last_mem + 1:] != -100).sum()), 0)

    def test_rows_without_image_fall_back_to_text(self):
        coll = make_pretrain_collator(p_image_ae=1.0, p_text=0.0, p_image_cont=0.0)
        rows = [{"image": None, "text": " ".join(f"w{i}" for i in range(30))}]
        b = coll.torch_call(rows)
        self.assertEqual(set(b), STANDARD_KEYS)                 # no images -> legacy layout
        self._mem_invariant(b)

    def test_text_only_matches_parent_collator(self):
        """p_text=1 must reproduce PretrainingCollator's inputs exactly under the same seed.
        Labels may only differ by the corrected masking order: ours masks a superset
        (the clear prefix before the last <MEM> in continuation rows)."""
        rows = [{"image": make_image(), "text": " ".join(f"w{i}" for i in range(50))},
                {"image": None, "text": " ".join(f"v{i}" for i in range(70))}]
        mm = make_pretrain_collator(p_image_ae=0.0, p_text=1.0, p_image_cont=0.0)
        ref = PretrainingCollator(compressor_tokenizer=MockTokenizer(), decoder_tokenizer=MockTokenizer(),
                                  compr_rate=4, compressor_max_length=16, decoder_max_length=256)
        np.random.seed(0); a = mm.torch_call(rows)
        np.random.seed(0); b = ref.torch_call(rows)
        self.assertEqual(set(a), STANDARD_KEYS)
        for k in STANDARD_KEYS - {"labels"}:
            self.assertTrue(torch.equal(a[k], b[k]), k)
        ours_masked, ref_masked = a["labels"] == -100, b["labels"] == -100
        self.assertTrue((ours_masked | ~ref_masked).all())          # ref masked => ours masked
        self.assertTrue(torch.equal(a["labels"][~ours_masked], b["labels"][~ours_masked]))

    def test_pad_to_multiple_of(self):
        np.random.seed(2)
        coll = make_pretrain_collator(pad_to_multiple_of=32)
        rows = [{"image": make_image(64, 64), "text": " ".join(f"w{i}" for i in range(40))} for _ in range(4)]
        b = coll.torch_call(rows)
        self.assertEqual(b["compressor_input_ids"].shape[1] % 32, 0)
        self.assertEqual(b["decoder_input_ids"].shape[1] % 32, 0)
        self._mem_invariant(b)
        if "mm_token_type_ids" in b:
            self.assertEqual(b["mm_token_type_ids"].shape, b["compressor_input_ids"].shape)
            self.assertTrue(torch.equal(b["mm_token_type_ids"], (b["compressor_input_ids"] == IP).long()))

    def test_mixed_batch(self):
        np.random.seed(1)
        coll = make_pretrain_collator()
        rows = [{"image": make_image(64, 64), "text": " ".join(f"w{i}" for i in range(40))} for _ in range(6)]
        b = coll.torch_call(rows)
        self._mem_invariant(b)
        if "mm_token_type_ids" in b:
            self.assertTrue(torch.equal(b["mm_token_type_ids"], (b["compressor_input_ids"] == IP).long()))


class TestPretrainingChatTasks(unittest.TestCase):
    """desc / find tasks (finetuning chat template inside stage-2) and their KL teacher inputs."""

    def _rows(self, n=3, teacher=True):
        rows = []
        for r in range(n):
            words = " ".join(f"r{r}w{i}" for i in range(40))           # distinct vocab per row
            row = {"image": make_image(64, 64, 30 + 60 * r), "text": words}
            if teacher:
                row["teacher_text"] = " ".join(f"d{r}t{i}" for i in range(12))
            rows.append(row)
        return rows

    def _check_chat_labels(self, b, i):
        lab, ids = b["labels"][i], b["decoder_input_ids"][i]
        sup = (lab != -100).nonzero().flatten().tolist()
        self.assertGreater(len(sup), 0)
        self.assertEqual(sup[-1], lab.numel() - 1)                       # the answer ends the row
        self.assertEqual(sup, list(range(sup[0], lab.numel())))          # one contiguous answer span
        self.assertTrue(all(int(ids[k]) not in (MEM, IP) for k in sup))  # no <MEM> / image in the target
        return [int(x) for x in ids[sup[0]:]]

    def test_desc_task(self):
        np.random.seed(3)
        coll = make_pretrain_collator(p_image_ae=0.0, p_text=0.0, p_image_cont=0.0, p_desc=1.0)
        rows = self._rows()
        b = coll.torch_call(rows)
        self.assertEqual(int((b["compressor_input_ids"] == MEM).sum()), int((b["decoder_input_ids"] == MEM).sum()))
        tok = coll.decoder_tokenizer
        for i in range(len(rows)):
            ans = tok.decode(self._check_chat_labels(b, i))
            self.assertIn(f"d{i}t0", ans)                                  # the row's own teacher description
            prompt = tok.decode(b["decoder_input_ids"][i][b["decoder_input_ids"][i] != PAD])
            self.assertIn("Describe Document", prompt)
        # one image chunk per document, at least one per row
        self.assertGreaterEqual(b["image_grid_thw"].shape[0], len(rows))
        self.assertTrue(torch.equal(b["mm_token_type_ids"], (b["compressor_input_ids"] == IP).long()))

    def test_find_task_answer_points_at_the_right_image(self):
        coll = make_pretrain_collator(p_image_ae=0.0, p_text=0.0, p_image_cont=0.0, p_find=1.0, multi_max_docs=3)
        tok = coll.decoder_tokenizer
        for seed in range(6):
            np.random.seed(seed)
            rows = self._rows(3, teacher=bool(seed % 2))     # with or without a teacher description
            b = coll.torch_call(rows)
            # pixel rows of each image chunk carry that image's fill value -> recover each example's doc order
            fills = [int(round(float(b["pixel_values"][k].mean()))) for k in np.cumsum(
                [0] + [int(g.prod()) for g in b["image_grid_thw"]])[:-1]]
            n_mem_rows = (b["compressor_input_ids"] == MEM).sum(1).tolist()
            c = 0
            for i, row in enumerate(rows):
                ans = tok.decode(self._check_chat_labels(b, i))
                j = int(ans.split("Document")[1].split()[0])
                prompt = tok.decode(b["decoder_input_ids"][i][b["decoder_input_ids"][i] != PAD])
                n_docs = prompt.count("Document") - 1                   # the answer "Document j" adds one
                self.assertGreaterEqual(n_docs, 2)
                own = 30 + 60 * i
                self.assertEqual(fills[c + j], own)                     # Document j IS the row's own image
                snippet = prompt.split('text: "')[1].split('"')[0]
                self.assertIn(snippet, row["text"])                     # the PAGE text, even with a teacher description
                c += n_docs
            self.assertEqual(c, len(fills))

    def test_find_snippet_falls_back_to_the_description_for_photos(self):
        coll = make_pretrain_collator(p_image_ae=0.0, p_text=0.0, p_image_cont=0.0, p_find=1.0, multi_max_docs=3)
        tok = coll.decoder_tokenizer
        np.random.seed(0)
        rows = self._rows(3)
        for r in rows:
            r["text"] = "short caption"                                   # a photo: no page text to quote
            r["teacher_text"] = " ".join(f"{r['teacher_text'].split()[0]}x{i}" for i in range(20))   # long enough to quote
        b = coll.torch_call(rows)
        for i, row in enumerate(rows):
            prompt = tok.decode(b["decoder_input_ids"][i][b["decoder_input_ids"][i] != PAD])
            snippet = prompt.split('text: "')[1].split('"')[0]
            self.assertIn(snippet, row["teacher_text"])

    def test_images_are_preprocessed_once_per_batch(self):
        coll = make_pretrain_collator(p_image_ae=0.0, p_text=0.0, p_image_cont=0.0, p_desc=1.0, multi_max_docs=4)
        np.random.seed(1)
        rows = self._rows(4)
        coll.torch_call(rows)
        self.assertEqual(coll.image_processor.calls, 4)                  # not once per (example, distractor)
        self.assertIsNone(coll._chunk_cache)                              # nothing kept across batches

    def test_desc_without_teacher_text_falls_back_to_ae(self):
        coll = make_pretrain_collator(p_image_ae=0.0, p_text=0.0, p_image_cont=0.0, p_desc=1.0)
        b = coll.torch_call(self._rows(2, teacher=False))
        for row in b["decoder_input_ids"]:
            self.assertEqual(row[row != PAD].tolist()[0], AE)

    def test_teacher_inputs_mixed_batch(self):
        coll = make_pretrain_collator(p_image_ae=0.5, p_text=0.0, p_image_cont=0.0, p_desc=0.5, teacher_inputs=True)
        for seed in range(6):
            np.random.seed(seed)
            b = coll.torch_call(self._rows(4))
            self.assertTrue(TEACHER_KEYS <= set(b) | {"teacher_pixel_values", "teacher_image_grid_thw",
                                                     "teacher_mm_token_type_ids"})
            kd = b["teacher_kd_rows"]
            t = b["teacher_input_ids"]
            self.assertEqual(int((t == MEM).sum()), 0)
            self.assertEqual(t.shape[0], int(kd.sum()))                   # teacher rows = the KD rows only
            self.assertEqual(b["teacher_attention_mask"].shape, t.shape)
            for n_t, i in enumerate(kd.nonzero().flatten().tolist()):     # in batch order
                n = int((b["labels"][i] != -100).sum())
                self.assertEqual(t[n_t, -n:].tolist(), b["labels"][i, -n:].tolist())
            if kd.any():
                self.assertEqual(int((t == IP).sum()), int(sum(int(g.prod()) // MERGE ** 2 for g in b["teacher_image_grid_thw"])))
                self.assertEqual(b["teacher_pixel_values"].shape[0], int(b["teacher_image_grid_thw"].prod(-1).sum()))

    def test_no_kd_row_gives_empty_teacher_tensors(self):
        coll = make_pretrain_collator(p_image_ae=1.0, p_text=0.0, p_image_cont=0.0, teacher_inputs=True)
        b = coll.torch_call(self._rows(2, teacher=False))
        self.assertFalse(bool(b["teacher_kd_rows"].any()))
        self.assertEqual(tuple(b["teacher_input_ids"].shape), (0, 1))
        self.assertNotIn("teacher_pixel_values", b)

    def test_shared_prompts_match_finetuning_defaults(self):
        import inspect
        from pisco import mm_tasks
        from pisco.collator import FineTuningCollator
        d = {k: v.default for k, v in inspect.signature(FineTuningCollator.__init__).parameters.items()}
        self.assertEqual(mm_tasks.USER_PROMPT, d["user_prompt"])
        self.assertEqual(mm_tasks.FIND_SYSTEM_PROMPT, d["system_prompt"])


class TestMultimodalFineTuningCollator(unittest.TestCase):
    def _row(self, with_images=True):
        return {
            "docs": ["", "alpha beta gamma delta epsilon zeta", ""],
            "doc_images": [make_image(64, 64, 20), None, make_image(96, 64, 220)] if with_images else [None] * 3,
            "query": "what is it",
            "mistral_label": "it is alpha",
        }

    def test_mixed_docs(self):
        coll = make_ft_collator(topk_docs=3)
        b = coll.torch_call([self._row()])
        self.assertEqual(set(b), STANDARD_KEYS | MM_KEYS)
        n_c = int((b["compressor_input_ids"] == MEM).sum()); n_d = int((b["decoder_input_ids"] == MEM).sum())
        self.assertEqual(n_c, n_d)
        self.assertEqual(n_d, (6 // 4 + 1) + (6 // 4 + 1) + (8 // 4 + 1))   # img4 span6, 6 words, img6 span8
        self.assertEqual(b["image_grid_thw"].tolist(), [[1, 4, 4], [1, 4, 6]])
        self.assertEqual(b["mm_token_type_ids"].sum(1).tolist(), [4, 0, 6])
        # decoder prompt names three documents
        dec = coll.decoder_tokenizer.decode(b["decoder_input_ids"][0][b["decoder_input_ids"][0] != PAD])
        self.assertIn("Document", dec)
        # labels: only the answer (3 words + whatever template suffix) is unmasked and sits at the end
        lab = b["labels"][0]
        unmasked = (lab != -100).nonzero().flatten().tolist()
        self.assertGreater(len(unmasked), 0)
        self.assertEqual(unmasked[-1], lab.numel() - 1)
        self.assertTrue(all(lab[i] not in (MEM, PAD, BOS, AE) for i in unmasked))
        self.assertTrue(all(lab[i] >= FIRST_WORD for i in unmasked))

    def test_text_only_row_is_legacy_layout(self):
        coll = make_ft_collator(topk_docs=3)
        b = coll.torch_call([self._row(with_images=False)])
        self.assertEqual(set(b), STANDARD_KEYS)

    def test_topk_truncates_images_too(self):
        coll = make_ft_collator(topk_docs=1)
        b = coll.torch_call([self._row()])
        self.assertEqual(b["image_grid_thw"].tolist(), [[1, 4, 4]])
        self.assertEqual(int((b["decoder_input_ids"] == MEM).sum()), 6 // 4 + 1)

    def test_missing_doc_images_column(self):
        coll = make_ft_collator(topk_docs=2)
        row = {"docs": ["one two three", "four five"], "query": "q", "mistral_label": "a"}
        b = coll.torch_call([row])
        self.assertEqual(set(b), STANDARD_KEYS)

    def test_teacher_inputs(self):
        coll = make_ft_collator(topk_docs=3, teacher_inputs=True)
        rows = [self._row(), {**self._row(), "mistral_label": "it is alpha beta gamma"}]
        b = coll.torch_call(rows)
        self.assertEqual(set(b), STANDARD_KEYS | MM_KEYS | TEACHER_KEYS)
        t = b["teacher_input_ids"]
        # raw pages: one <|image_pad|> per visual token, same pixels / grids as the compressor
        self.assertEqual(int((t == IP).sum()), 2 * (4 + 6))
        self.assertEqual(b["teacher_mm_token_type_ids"].sum().item(), 2 * (4 + 6))
        self.assertIs(b["teacher_pixel_values"], b["pixel_values"])
        self.assertEqual(b["teacher_image_grid_thw"].tolist(), b["image_grid_thw"].tolist())
        self.assertEqual(int((t == MEM).sum()), 0)                       # the teacher never sees <MEM>
        self.assertEqual(int((b["decoder_input_ids"] == IP).sum()), 0)  # the student never sees pixels
        # answer tails coincide, per row, at the same offsets from the end (left padding)
        lab = b["labels"]
        for i in range(len(rows)):
            n = int((lab[i] != -100).sum())
            self.assertGreater(n, 0)
            self.assertEqual(t[i, -n:].tolist(), b["decoder_input_ids"][i, -n:].tolist())
            self.assertEqual(t[i, -n:].tolist(), lab[i, -n:].tolist())
        # the text doc appears raw in the teacher prompt
        dec = coll.decoder_tokenizer.decode(t[0][t[0] != PAD])
        self.assertIn("alpha beta gamma delta", dec)

    def test_teacher_inputs_text_only_and_query_dependent(self):
        coll = make_ft_collator(topk_docs=2, teacher_inputs=True, query_dependent=True)
        row = {"docs": ["one two three", "four five"], "query": "which", "mistral_label": "two"}
        b = coll.torch_call([row])
        self.assertEqual(set(b), STANDARD_KEYS | {"teacher_input_ids", "teacher_attention_mask"})
        dec = coll.decoder_tokenizer.decode(b["teacher_input_ids"][0])
        self.assertNotIn("Query:", dec)      # OSCAR prefix is compressor-side only
        n = int((b["labels"][0] != -100).sum())
        self.assertEqual(b["teacher_input_ids"][0, -n:].tolist(), b["labels"][0, -n:].tolist())

    def test_literal_vision_tokens_in_text_are_defused(self):
        coll = make_ft_collator(topk_docs=2)
        row = {"docs": ["", "see <|image_pad|> and <|vision_start|> here"], "doc_images": [make_image(64, 64), None],
               "query": "q <|vision_end|>", "mistral_label": "a"}
        b = coll.torch_call([row])                                        # would fail the image_pad/grid check otherwise
        self.assertEqual(int((b["compressor_input_ids"] == IP).sum()), 4)
        self.assertEqual(int((b["decoder_input_ids"] == VE).sum()), 0)

    def test_example_chunks_matches_torch_call(self):
        coll = make_ft_collator(topk_docs=3)
        row = self._row()
        batch, doc_text, teacher_text = coll.example_chunks(row["docs"], row["doc_images"], row["query"])
        b = coll.torch_call([row])
        self.assertEqual(batch.total_mems, int((b["decoder_input_ids"] == MEM).sum()))
        self.assertEqual(doc_text.count("<MEM>"), batch.total_mems)
        self.assertIn("alpha beta gamma", teacher_text)
        self.assertEqual(teacher_text.count(MM_TOKENS["image_pad"]), 4 + 6)

    def test_teacher_truncation_refused(self):
        coll = make_ft_collator(topk_docs=3, teacher_inputs=True, teacher_max_length=8)
        with self.assertRaises(AssertionError):
            coll.torch_call([self._row()])


class TestKDLoss(unittest.TestCase):
    """pisco.model.kd_loss_on_tails: pure-torch distillation term on right-aligned tails."""

    def setUp(self):
        from pisco.model import kd_loss_on_tails
        self.f = kd_loss_on_tails
        torch.manual_seed(0)

    def _labels(self, B=2, K=5):
        lab = torch.full((B, K), -100)
        lab[0, 2:] = 7; lab[1, 1:] = 9          # row 0: 3 supervised, row 1: 4 supervised
        return lab

    def test_zero_when_identical(self):
        s = torch.randn(2, 5, 11); lab = self._labels()
        kl, n = self.f(s, s.clone(), lab)
        self.assertEqual(n, 3 + 4)               # shift: labels[:, 1:] keeps 3 + 4 supervised ids
        self.assertAlmostEqual(float(kl), 0.0, places=5)

    def test_positive_and_masked(self):
        s = torch.randn(2, 5, 11); t = torch.randn(2, 5, 11); lab = self._labels()
        kl, _ = self.f(s, t, lab)
        self.assertGreater(float(kl), 0.0)
        # changing logits at a masked position must not change the loss
        t2 = t.clone(); t2[0, 0] += 100.0; s2 = s.clone(); s2[1, 0] -= 50.0
        kl2, _ = self.f(s2, t2, lab)
        self.assertAlmostEqual(float(kl), float(kl2), places=5)
        # the last position never contributes (it predicts nothing)
        t3 = t.clone(); t3[:, -1] += 100.0
        self.assertAlmostEqual(float(kl), float(self.f(s, t3, lab)[0]), places=5)

    def test_ignore_ids_and_temperature(self):
        s = torch.randn(2, 5, 11); t = torch.randn(2, 5, 11); lab = self._labels()
        t_mem = t.clone(); t_mem[..., 3] = 50.0          # teacher dumps mass on an ignored id
        kl_ign, _ = self.f(s, t_mem, lab, ignore_ids=[3])
        kl_ref, _ = self.f(s, t, lab, ignore_ids=[3])
        self.assertAlmostEqual(float(kl_ign), float(kl_ref), places=4)
        self.assertTrue(torch.isfinite(kl_ign))
        self.assertGreater(float(kl_ref), 0.0)
        # without the ignore list the dumped mass changes the loss (sanity of the fixture)
        self.assertNotAlmostEqual(float(self.f(s, t_mem, lab)[0]), float(kl_ref), places=2)
        # T -> large flattens both distributions: T^2 * KL stays finite and shrinks the gap
        kl_T, _ = self.f(s, t, lab, temperature=4.0)
        self.assertTrue(torch.isfinite(kl_T))
        self.assertLess(float(kl_T), float(kl_ref) * 16)

    def test_rope_deltas_snapshot_restore(self):
        from pisco.model import _snapshot_rope_deltas, _restore_rope_deltas
        inner = torch.nn.Linear(2, 2); inner.rope_deltas = None
        outer = torch.nn.Sequential(inner, torch.nn.Linear(2, 2))
        saved = _snapshot_rope_deltas(outer)
        self.assertEqual(len(saved), 1)
        inner.rope_deltas = torch.tensor([-1700, -1650])   # what a teacher pass leaves behind
        _restore_rope_deltas(saved)
        self.assertIsNone(inner.rope_deltas)

    def test_all_masked(self):
        s = torch.randn(1, 4, 6); t = torch.randn(1, 4, 6)
        kl, n = self.f(s, t, torch.full((1, 4), -100))
        self.assertEqual(n, 0); self.assertEqual(float(kl), 0.0)
        self.assertTrue(kl.requires_grad is False or kl.grad_fn is not None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
