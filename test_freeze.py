"""
Tests for PISCOConfig.freeze_compressor (stage B: frozen compressor + connector, decoder-only training).
No backbone is loaded: a stand-in PISCO with tiny modules.  Run:  python test_freeze.py
"""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from pisco.model import PISCO, PISCOConfig, freeze_compressor_side


class _TinyPISCO(PISCO):
    """Same class (so PISCO.train is the real one), no backbones."""

    def __init__(self, freeze: bool):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(freeze_compressor=freeze)
        self.compressor = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))
        self.connector = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
        self.decoder = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))


class TestFreezeCompressor(unittest.TestCase):
    def test_config_default_off_and_roundtrip(self):
        self.assertFalse(PISCOConfig().freeze_compressor)
        cfg = PISCOConfig(freeze_compressor=True)
        self.assertTrue(PISCOConfig.from_dict(cfg.to_dict()).freeze_compressor)

    def test_only_decoder_trainable(self):
        m = _TinyPISCO(True)
        n = freeze_compressor_side(m)
        self.assertEqual(n, 6)  # 2 compressor + 4 connector tensors
        trainable = {k.split(".")[0] for k, p in m.named_parameters() if p.requires_grad}
        self.assertEqual(trainable, {"decoder"})

    def test_train_keeps_frozen_side_in_eval(self):
        m = _TinyPISCO(True)
        freeze_compressor_side(m)
        m.train()
        self.assertFalse(m.compressor.training)
        self.assertFalse(m.connector.training)
        self.assertTrue(m.decoder.training)
        x = torch.ones(2, 4)
        self.assertTrue(torch.equal(m.compressor(x), m.compressor(x)))  # dropout off: deterministic <MEM>s
        m.eval()
        self.assertFalse(m.decoder.training)

    def test_flag_off_is_the_old_behaviour(self):
        m = _TinyPISCO(False)
        m.train()
        self.assertTrue(m.compressor.training and m.connector.training and m.decoder.training)
        self.assertTrue(all(p.requires_grad for p in m.parameters()))

    def test_no_gradient_reaches_frozen_side(self):
        m = _TinyPISCO(True)
        freeze_compressor_side(m)
        m.train()
        before = [p.detach().clone() for p in list(m.compressor.parameters()) + list(m.connector.parameters())]
        opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1)
        loss = m.decoder(m.connector(m.compressor(torch.randn(3, 4)))).pow(2).sum()
        loss.backward()
        opt.step()
        after = list(m.compressor.parameters()) + list(m.connector.parameters())
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(after, before)))
        self.assertTrue(all(p.grad is None for p in after))


if __name__ == "__main__":
    unittest.main(verbosity=2)
