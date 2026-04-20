"""
examples/07_full_transformer_demo.py
====================================

End-to-end forward pass through the :class:`Transformer` of §3 / Fig. 1.

We use a *small* configuration — d_model = 64, 2 layers, 4 heads,
vocab = 50 — so the script runs instantly on a CPU.  The *base*
configuration of Table 3 (d_model = 512, N = 6, h = 8, d_ff = 2048)
works the same way but is too big for a demo.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.config import ModelConfig
from src.transformer import Transformer


def main():
    torch.manual_seed(0)

    cfg = ModelConfig(
        src_vocab_size=50,
        tgt_vocab_size=50,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        max_seq_len=32,
        dropout=0.1,
    )
    model = Transformer(cfg)
    model.eval()

    # Fake batch of 2 source sequences and 2 target sequences.
    src = torch.randint(1, cfg.src_vocab_size, (2, 9))       # avoid pad=0
    tgt = torch.randint(1, cfg.tgt_vocab_size, (2, 7))

    logits = model(src, tgt)
    print(f"src       shape: {tuple(src.shape)}")
    print(f"tgt       shape: {tuple(tgt.shape)}")
    print(f"logits    shape: {tuple(logits.shape)}    "
          f"(batch, tgt_len, vocab_size)")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nThis tiny Transformer has {n_params:,} parameters.")
    print("(The paper's base model has ≈ 65M parameters — Table 3.)")

    # Demo greedy decoding.
    decoded = model.greedy_decode(src, max_len=10, start_symbol=1)
    print("\nGreedy decoded output ids:\n", decoded)


if __name__ == "__main__":
    main()
