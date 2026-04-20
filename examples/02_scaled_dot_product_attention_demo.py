"""
examples/02_scaled_dot_product_attention_demo.py
================================================

Walk through §3.2.1 of the paper — Equation (1):

    Attention(Q, K, V) = softmax( QKᵀ / √d_k ) V.

This script:

1. Builds a tiny, hand-crafted Q, K, V so you can read the numbers.
2. Computes `scaled_dot_product_attention` and prints every intermediate
   matrix (QKᵀ, scaled, softmax, output).
3. Demonstrates the look-ahead mask from §3.1 / §3.2.3 by re-running the
   attention with the subsequent mask and showing that positions cannot
   see the future.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import torch

from src.scaled_dot_product_attention import scaled_dot_product_attention
from src.masking import subsequent_mask


def main():
    torch.manual_seed(0)

    # --- 1. craft a tiny input -------------------------------------------------
    # Imagine 4 tokens, each embedded into d_k = 2.
    # We use small integers so you can verify the computation by hand.
    Q = torch.tensor([[[1., 0.],
                       [0., 1.],
                       [1., 1.],
                       [1., -1.]]])                 # shape (batch=1, seq=4, d_k=2)
    K = Q.clone()                                    # same as Q → "self-attention"
    V = torch.tensor([[[10., 0.],
                       [ 0., 10.],
                       [ 5.,  5.],
                       [ 5., -5.]]])

    print("Q =\n", Q[0])
    print("K =\n", K[0])
    print("V =\n", V[0])

    # --- 2. do the attention ---------------------------------------------------
    out, attn = scaled_dot_product_attention(Q, K, V)
    print("\n--- §3.2.1 intermediates -----------------------------------------")
    d_k = Q.size(-1)
    qkT = Q @ K.transpose(-2, -1)
    print("QKᵀ =\n", qkT[0])
    print(f"Scaling factor 1/√d_k with d_k={d_k} is {1/math.sqrt(d_k):.4f}")
    print("QKᵀ / √d_k =\n", (qkT / math.sqrt(d_k))[0])
    print("softmax(·) =\n", attn[0])                # shape (seq, seq) for batch=0
    print("output = softmax(·) V =\n", out[0])

    # --- 3. apply the subsequent (look-ahead) mask from §3.1 -------------------
    print("\n--- with subsequent mask (§3.1 / §3.2.3) --------------------------")
    mask = subsequent_mask(4)                        # (1, 1, 4, 4)
    # Broadcast the mask to the shape expected by `scaled_dot_product_attention`.
    out_m, attn_m = scaled_dot_product_attention(Q, K, V, mask=mask)
    print("Attention weights (should be lower-triangular, row-stochastic):\n",
          attn_m[0, 0])
    print("output =\n", out_m[0])
    print("Observe: row i only has non-zero weights in columns 0..i.")


if __name__ == "__main__":
    main()
