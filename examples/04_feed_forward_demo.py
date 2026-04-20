"""
examples/04_feed_forward_demo.py
================================

Position-wise FFN demo — §3.3 Eq. (2):

    FFN(x) = max(0, x W_1 + b_1) W_2 + b_2.

Shows:

1. Basic shape behaviour (d_model -> d_ff -> d_model).
2. That the FFN is indeed "position-wise" — applying it token-by-token
   gives the same answer as applying it to the whole sequence at once.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.feed_forward import PositionwiseFeedForward


def main():
    torch.manual_seed(0)
    d_model, d_ff = 8, 32
    ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=0.0)
    ffn.eval()                                    # disable any dropout randomness

    # Batched input
    x = torch.randn(1, 5, d_model)
    y_batched = ffn(x)
    print(f"Input  shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(y_batched.shape)}")

    # Token-by-token: same output, proving §3.3's "applied to each
    # position separately and identically".
    y_per_token = torch.stack([ffn(x[:, i:i+1, :]) for i in range(x.size(1))], dim=1).squeeze(2)
    print("\nPosition-wise property holds:",
          torch.allclose(y_batched, y_per_token, atol=1e-6))


if __name__ == "__main__":
    main()
