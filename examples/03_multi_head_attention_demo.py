"""
examples/03_multi_head_attention_demo.py
========================================

Run one forward pass through :class:`MultiHeadAttention` and inspect the
per-head attention weights.  Relevant paper section: §3.2.2.

The script:

1. Instantiates multi-head attention with d_model = 16, h = 4
   (so d_k = d_v = 4, matching §3.2.2's `d_model / h` rule).
2. Feeds it a random (batch, seq, d_model) tensor.
3. Prints the output shape and the per-head attention distribution to
   confirm each head attends independently — §3.2.2 last sentence:

        > "Multi-head attention allows the model to jointly attend to
        >  information from different representation subspaces at
        >  different positions."
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.multi_head_attention import MultiHeadAttention


def main():
    torch.manual_seed(0)

    d_model, num_heads = 16, 4
    batch, seq = 2, 6

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    x = torch.randn(batch, seq, d_model)

    # Self-attention: Q = K = V = x  (§3.2.3 use #1).
    out, attn = mha(x, x, x)
    print(f"input   shape: {tuple(x.shape)}    (batch, seq, d_model)")
    print(f"output  shape: {tuple(out.shape)}    (batch, seq, d_model)")
    print(f"attn    shape: {tuple(attn.shape)}    (batch, heads, q_len, k_len)")

    # Each head's attention matrix should be row-stochastic (rows sum to 1).
    row_sums = attn.sum(dim=-1)
    print(f"\nall row sums ≈ 1?  {torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)}")

    # Show the attention distribution of batch 0, head 0:
    print("\nAttention (batch 0, head 0):")
    print(attn[0, 0])

    # Parameter count sanity check — §3.2.2 argues that multi-head with
    # reduced per-head dimension has cost comparable to single-head at full d.
    n_params = sum(p.numel() for p in mha.parameters())
    print(f"\nTotal params in this multi-head attention block: {n_params}")
    print(f"Expected = 4 * d_model² = {4 * d_model**2}")


if __name__ == "__main__":
    main()
