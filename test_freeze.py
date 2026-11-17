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


# ---- split test (PISCOConfig.freeze_part, campaign freeze 2026-11-12): one part frozen, the other trains ----
from pisco.model import FREEZE_PARTS, frozen_parts  # noqa: E402


class _TinyPISCOPart(_TinyPISCO):
    def __init__(self, part):
        super().__init__(False)
        self.config = SimpleNamespace(freeze_compressor=False, freeze_part=part)


def _sgd_step(m):
    torch.manual_seed(0)  # the compressor's Dropout(0.5) is live when it trains: fix the mask
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1)
    m.decoder(m.connector(m.compressor(torch.randn(3, 4)))).pow(2).sum().backward()
    opt.step()


class TestFreezePart(unittest.TestCase):
    def test_config_default_unset_and_roundtrip(self):
        self.assertIsNone(PISCOConfig().freeze_part)
        self.assertEqual(frozen_parts(PISCOConfig()), ())
        for part in FREEZE_PARTS:
            self.assertEqual(PISCOConfig.from_dict(PISCOConfig(freeze_part=part).to_dict()).freeze_part, part)

    def test_frozen_parts_resolution(self):
        self.assertEqual(frozen_parts(PISCOConfig(freeze_compressor=True)), ("compressor", "connector"))
        self.assertEqual(frozen_parts(PISCOConfig(freeze_part="compressor")), ("compressor",))
        self.assertEqual(frozen_parts(PISCOConfig(freeze_part="connector")), ("connector",))
        with self.assertRaises(ValueError):
            frozen_parts(PISCOConfig(freeze_part="decoder"))
        with self.assertRaises(ValueError):
            frozen_parts(PISCOConfig(freeze_compressor=True, freeze_part="connector"))

    def test_compressor_only(self):
        m = _TinyPISCOPart("compressor")
        self.assertEqual(freeze_compressor_side(m, frozen_parts(m.config)), 2)
        self.assertEqual({k.split(".")[0] for k, p in m.named_parameters() if p.requires_grad}, {"connector", "decoder"})
        m.train()
        self.assertFalse(m.compressor.training)
        self.assertTrue(m.connector.training and m.decoder.training)
        comp, conn = [p.detach().clone() for p in m.compressor.parameters()], [p.detach().clone() for p in m.connector.parameters()]
        _sgd_step(m)
        self.assertTrue(all(torch.equal(a, b) and a.grad is None for a, b in zip(m.compressor.parameters(), comp)))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(m.connector.parameters(), conn)))

    def test_connector_only_gradient_flows_through(self):
        m = _TinyPISCOPart("connector")
        self.assertEqual(freeze_compressor_side(m, frozen_parts(m.config)), 4)
        self.assertEqual({k.split(".")[0] for k, p in m.named_parameters() if p.requires_grad}, {"compressor", "decoder"})
        m.train()
        self.assertFalse(m.connector.training)
        self.assertTrue(m.compressor.training and m.decoder.training)
        comp, conn = [p.detach().clone() for p in m.compressor.parameters()], [p.detach().clone() for p in m.connector.parameters()]
        _sgd_step(m)
        self.assertTrue(all(torch.equal(a, b) and a.grad is None for a, b in zip(m.connector.parameters(), conn)))
        self.assertTrue(all(p.grad is not None for p in m.compressor.parameters()))  # backward crosses the frozen connector
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(m.compressor.parameters(), comp)))

    def test_unset_part_trains_everything(self):
        m = _TinyPISCOPart(None)
        m.train()
        self.assertTrue(m.compressor.training and m.connector.training and m.decoder.training)
        self.assertTrue(all(p.requires_grad for p in m.parameters()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
