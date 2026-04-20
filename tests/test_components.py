"""
tests/test_components.py
========================

Lightweight shape / sanity tests for every module in :mod:`src`.

Run:
    python -m tests.test_components
or with pytest if you have it:
    pytest tests/test_components.py -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # noqa: E402

from src.config import ModelConfig  # noqa: E402
from src.positional_encoding import PositionalEncoding  # noqa: E402
from src.embeddings import TokenEmbedding, OutputProjection  # noqa: E402
from src.scaled_dot_product_attention import scaled_dot_product_attention  # noqa: E402
from src.multi_head_attention import MultiHeadAttention  # noqa: E402
from src.feed_forward import PositionwiseFeedForward  # noqa: E402
from src.masking import padding_mask, subsequent_mask, make_tgt_mask  # noqa: E402
from src.encoder import Encoder  # noqa: E402
from src.decoder import Decoder  # noqa: E402
from src.transformer import Transformer  # noqa: E402
from src.label_smoothing import LabelSmoothingLoss  # noqa: E402
from src.optimizer import NoamScheduler  # noqa: E402


class TestPositionalEncoding(unittest.TestCase):
    def test_shape_and_determinism(self):
        pe = PositionalEncoding(d_model=16, max_len=20, dropout=0.0)
        x = torch.zeros(2, 10, 16)
        out1 = pe(x)
        out2 = pe(x)
        # With zero dropout the encoding is deterministic.
        self.assertTrue(torch.equal(out1, out2))
        self.assertEqual(out1.shape, (2, 10, 16))

    def test_max_value_is_1(self):
        pe = PositionalEncoding(d_model=16, max_len=50, dropout=0.0)
        # sine / cosine are bounded in [-1, 1].
        self.assertLessEqual(pe.pe.abs().max().item(), 1.0 + 1e-6)


class TestTokenEmbedding(unittest.TestCase):
    def test_scaling_by_sqrt_dmodel(self):
        torch.manual_seed(0)
        emb = TokenEmbedding(vocab_size=7, d_model=4)
        ids = torch.tensor([[1, 2, 3]])
        out = emb(ids)
        raw = emb.lut(ids)
        self.assertTrue(torch.allclose(out, raw * 2.0))   # √4 = 2


class TestScaledDotProductAttention(unittest.TestCase):
    def test_rows_sum_to_one(self):
        torch.manual_seed(0)
        Q = torch.randn(2, 5, 8)
        K = torch.randn(2, 5, 8)
        V = torch.randn(2, 5, 8)
        out, attn = scaled_dot_product_attention(Q, K, V)
        row_sums = attn.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))
        self.assertEqual(out.shape, Q.shape)


class TestMultiHeadAttention(unittest.TestCase):
    def test_shape(self):
        mha = MultiHeadAttention(d_model=16, num_heads=4, dropout=0.0)
        x = torch.randn(3, 7, 16)
        out, attn = mha(x, x, x)
        self.assertEqual(out.shape, (3, 7, 16))
        self.assertEqual(attn.shape, (3, 4, 7, 7))


class TestFFN(unittest.TestCase):
    def test_shape(self):
        ffn = PositionwiseFeedForward(d_model=8, d_ff=32, dropout=0.0)
        x = torch.randn(2, 5, 8)
        out = ffn(x)
        self.assertEqual(out.shape, x.shape)


class TestMasks(unittest.TestCase):
    def test_padding(self):
        src = torch.tensor([[2, 3, 0, 0], [4, 5, 6, 0]])
        mask = padding_mask(src, pad_idx=0)
        self.assertEqual(mask.shape, (2, 1, 1, 4))
        self.assertEqual(mask[0, 0, 0].tolist(), [True, True, False, False])

    def test_subsequent(self):
        m = subsequent_mask(4)
        expected = torch.tensor([[1, 0, 0, 0],
                                 [1, 1, 0, 0],
                                 [1, 1, 1, 0],
                                 [1, 1, 1, 1]], dtype=torch.bool)
        self.assertTrue(torch.equal(m[0, 0], expected))

    def test_combined(self):
        tgt = torch.tensor([[1, 2, 0]])
        m = make_tgt_mask(tgt, pad_idx=0)
        # Position 2 is <pad>, so column 2 should be False throughout.
        self.assertFalse(m[0, 0, :, 2].any())


class TestEncoderDecoder(unittest.TestCase):
    def test_shapes(self):
        enc = Encoder(num_layers=2, d_model=16, num_heads=4, d_ff=32, dropout=0.0)
        dec = Decoder(num_layers=2, d_model=16, num_heads=4, d_ff=32, dropout=0.0)
        src = torch.randn(2, 6, 16)
        tgt = torch.randn(2, 4, 16)
        memory = enc(src)
        out = dec(tgt, memory)
        self.assertEqual(memory.shape, src.shape)
        self.assertEqual(out.shape, tgt.shape)


class TestTransformer(unittest.TestCase):
    def test_forward_shape(self):
        cfg = ModelConfig(src_vocab_size=20, tgt_vocab_size=20,
                          d_model=16, num_heads=4, num_layers=2,
                          d_ff=32, max_seq_len=32, dropout=0.0)
        model = Transformer(cfg)
        src = torch.randint(1, 20, (2, 9))
        tgt = torch.randint(1, 20, (2, 7))
        out = model(src, tgt)
        self.assertEqual(out.shape, (2, 7, 20))

    def test_greedy_decode(self):
        cfg = ModelConfig(src_vocab_size=20, tgt_vocab_size=20,
                          d_model=16, num_heads=4, num_layers=2,
                          d_ff=32, max_seq_len=32, dropout=0.0)
        model = Transformer(cfg)
        src = torch.randint(1, 20, (1, 5))
        out = model.greedy_decode(src, max_len=6, start_symbol=1)
        self.assertEqual(out.shape, (1, 6))


class TestLabelSmoothing(unittest.TestCase):
    def test_non_negative_and_differentiable(self):
        loss_fn = LabelSmoothingLoss(vocab_size=10, padding_idx=0, smoothing=0.1)
        logits = torch.randn(2, 3, 10, requires_grad=True)
        target = torch.randint(1, 10, (2, 3))
        loss = loss_fn(logits, target)
        loss.backward()
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertIsNotNone(logits.grad)


class TestNoam(unittest.TestCase):
    def test_peak_at_warmup(self):
        s = NoamScheduler(d_model=512, warmup_steps=4000)
        # Eq. (3) peaks exactly at step == warmup_steps.
        peak = max([s(x) for x in range(1, 10_000)])
        self.assertAlmostEqual(peak, s(4000), places=8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
