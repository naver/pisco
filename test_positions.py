"""
Tests for the spread <MEM> position layout (pisco.model.mem_spread_position_ids, PISCOConfig.mem_position_stride).
Pure tensor logic, no model.  Run:  python test_positions.py   (or pytest test_positions.py -v)
"""

import unittest

import torch

from pisco.model import coarse_to_fine_order, compressor_spread_position_ids, mem_spread_position_ids

MEM, PAD, T = 3, 0, 100


class TestSpreadPositions(unittest.TestCase):
    def test_stride_one_is_consecutive(self):
        ids = torch.tensor([[PAD, T, MEM, MEM, T, T]])
        mask = torch.tensor([[0, 1, 1, 1, 1, 1]])
        pos = mem_spread_position_ids(ids, mask, MEM, 1)
        self.assertEqual(pos.tolist(), [[0, 0, 1, 2, 3, 4]])

    def test_footprint_and_centres(self):
        # text, 3 <MEM>s (stride 4), text: the <MEM>s cover ids 1..12, centred at 3, 7, 11; the text resumes at 13
        ids = torch.tensor([[T, MEM, MEM, MEM, T, T]])
        mask = torch.ones_like(ids)
        pos = mem_spread_position_ids(ids, mask, MEM, 4)
        self.assertEqual(pos.tolist(), [[0, 3, 7, 11, 13, 14]])

    def test_two_runs_and_left_padding(self):
        ids = torch.tensor([[PAD, PAD, T, MEM, MEM, T, MEM, T],
                            [T, MEM, T, T, T, T, T, T]])
        mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1]])
        pos = mem_spread_position_ids(ids, mask, MEM, 8)
        # row 0: T=0, MEM slots [1,8] -> 5, [9,16] -> 13, T=17, MEM slot [18,25] -> 22, T=26
        self.assertEqual(pos[0].tolist(), [0, 0, 0, 5, 13, 17, 22, 26])
        # row 1: T=0, MEM slot [1,8] -> 5, then 9, 10, ...
        self.assertEqual(pos[1].tolist(), [0, 5, 9, 10, 11, 12, 13, 14])
        # the next generated token continues after the last footprint: last + 1 per HF generate
        self.assertEqual(int(pos[0, -1]) + 1, 27)

    def test_mrope_base_shift(self):
        # 3-axis base (as Qwen computes for raw image spans): every axis gets the same shift
        ids = torch.tensor([[T, MEM, MEM, T, T]])
        mask = torch.ones_like(ids)
        base = torch.arange(5).view(1, 1, 5).expand(3, 1, 5).clone()
        base[1, 0, 3] = 7  # pretend an image token with its own height id
        pos = mem_spread_position_ids(ids, mask, MEM, 4, base=base)
        self.assertEqual(pos.shape, (3, 1, 5))
        shift = [0, 2, 5, 6, 6]  # (stride-1)*k before me + centre offset 2 for a <MEM>
        for ax in range(3):
            self.assertEqual(pos[ax, 0].tolist(), [int(base[ax, 0, i]) + shift[i] for i in range(5)])

    def test_no_mem_is_identity(self):
        ids = torch.tensor([[T, T, T]])
        mask = torch.ones_like(ids)
        self.assertEqual(mem_spread_position_ids(ids, mask, MEM, 8).tolist(), [[0, 1, 2]])


IMG = 9


class TestCompressorSpread(unittest.TestCase):
    def test_text_chunk_segment_centres(self):
        # left pad, 8 content tokens at ids 0..7, 4 <MEM>s -> segment middles 1, 3, 5, 7
        ids = torch.tensor([[PAD, T, T, T, T, T, T, T, T, MEM, MEM, MEM, MEM]])
        mask = torch.tensor([[0] + [1] * 12])
        base = (torch.cumsum(mask, -1) - 1).masked_fill(mask == 0, 0).unsqueeze(0).expand(3, -1, -1)
        pos = compressor_spread_position_ids(ids, mask, MEM, base)
        self.assertEqual(pos[0, 0, 9:].tolist(), [1, 3, 5, 7])
        self.assertEqual(pos[1, 0, 9:].tolist(), [1, 3, 5, 7])
        self.assertEqual(pos[0, 0, :9].tolist(), base[0, 0, :9].tolist())   # content and pad untouched

    def test_image_chunk_tiles(self):
        # an image of grid 4x6 (after merge), 6 <MEM>s -> 2 rows x 3 cols of tiles; start position 1
        h, w, n = 4, 6, 6
        ids = torch.tensor([[T] + [IMG] * (h * w) + [MEM] * n])
        mask = torch.ones_like(ids)
        base = torch.zeros(3, 1, ids.shape[1], dtype=torch.long)
        base[:, 0, 0] = 0
        st = 1
        for i in range(h * w):   # Qwen layout: (t, h, w) = (st, st + row, st + col)
            base[0, 0, 1 + i], base[1, 0, 1 + i], base[2, 0, 1 + i] = st, st + i // w, st + i % w
        base[:, 0, 1 + h * w:] = st + max(h, w) + torch.arange(n)
        grid = torch.tensor([[1, 2 * h, 2 * w]])
        pos = compressor_spread_position_ids(ids, mask, MEM, base, image_token_id=IMG, image_grid_thw=grid, merge_size=2)
        m = slice(1 + h * w, None)
        self.assertEqual(pos[0, 0, m].tolist(), [st] * n)                       # temporal = image start
        self.assertEqual(pos[1, 0, m].tolist(), [st + 1] * 3 + [st + 3] * 3)    # row centres of 2 tile rows in 4
        self.assertEqual(pos[2, 0, m].tolist(), [st + 1, st + 3, st + 5] * 2)   # column centres of 3 tile cols in 6
        self.assertEqual(pos[:, 0, :1 + h * w].tolist(), base[:, 0, :1 + h * w].tolist())

    def test_mixed_batch_image_rows_matched_in_order(self):
        ids = torch.tensor([[T, T, T, T, MEM, MEM], [T, IMG, IMG, IMG, IMG, MEM]])
        mask = torch.ones_like(ids)
        base = torch.arange(6).view(1, 1, 6).expand(3, 2, 6).clone()
        grid = torch.tensor([[1, 4, 4]])   # 2x2 after merge, 1 <MEM> -> centre (1, 1) + start
        pos = compressor_spread_position_ids(ids, mask, MEM, base, image_token_id=IMG, image_grid_thw=grid, merge_size=2)
        self.assertEqual(pos[0, 0, 4:].tolist(), [1, 3])
        self.assertEqual(pos[:, 1, 5].tolist(), [1, 2, 2])


class TestCoarseToFine(unittest.TestCase):
    def test_order_power_of_two(self):
        self.assertEqual(coarse_to_fine_order(8), [0, 4, 2, 6, 1, 5, 3, 7])

    def test_order_is_a_permutation_and_prefixes_spread(self):
        for n in (1, 2, 5, 9, 33, 123):
            o = coarse_to_fine_order(n)
            self.assertEqual(sorted(o), list(range(n)))
            if n >= 16:   # with at least 4 slots in the first quarter, that prefix covers all four quarters of the chunk
                q = {i * 4 // n for i in o[: n // 4]}
                self.assertEqual(q, {0, 1, 2, 3}, (n, o[: n // 4]))

    def test_text_chunk_coarse_centres(self):
        ids = torch.tensor([[PAD, T, T, T, T, T, T, T, T, MEM, MEM, MEM, MEM]])
        mask = torch.tensor([[0] + [1] * 12])
        base = (torch.cumsum(mask, -1) - 1).masked_fill(mask == 0, 0).unsqueeze(0).expand(3, -1, -1)
        pos = compressor_spread_position_ids(ids, mask, MEM, base, coarse_to_fine=True)
        # segment centres 1, 3, 5, 7 taken in order [0, 2, 1, 3] -> slots at 1, 5, 3, 7
        self.assertEqual(pos[0, 0, 9:].tolist(), [1, 5, 3, 7])

    def test_image_chunk_coarse_tiles(self):
        h, w, n = 4, 6, 6
        ids = torch.tensor([[T] + [IMG] * (h * w) + [MEM] * n])
        mask = torch.ones_like(ids)
        base = torch.zeros(3, 1, ids.shape[1], dtype=torch.long)
        st = 1
        for i in range(h * w):
            base[0, 0, 1 + i], base[1, 0, 1 + i], base[2, 0, 1 + i] = st, st + i // w, st + i % w
        grid = torch.tensor([[1, 2 * h, 2 * w]])
        plain = compressor_spread_position_ids(ids, mask, MEM, base, image_token_id=IMG, image_grid_thw=grid, merge_size=2)
        coarse = compressor_spread_position_ids(ids, mask, MEM, base, image_token_id=IMG, image_grid_thw=grid, merge_size=2, coarse_to_fine=True)
        m = slice(1 + h * w, None)
        # same set of tile centres, permuted by coarse_to_fine_order(6) = [0, 4, 2, 1, 5, 3]
        order = coarse_to_fine_order(6)
        for ax in (1, 2):
            self.assertEqual(coarse[ax, 0, m].tolist(), [int(plain[ax, 0, 1 + h * w + j]) for j in order])


if __name__ == "__main__":
    unittest.main()
