"""
examples/09_label_smoothing_demo.py
===================================

Inspect the smoothed target distribution produced by
:class:`src.label_smoothing.LabelSmoothingLoss` — the one promised by
§5.4:

    > "During training, we employed label smoothing of value ε_ls = 0.1."
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.label_smoothing import LabelSmoothingLoss


def main():
    vocab_size, padding_idx = 6, 0
    loss_fn = LabelSmoothingLoss(vocab_size=vocab_size, padding_idx=padding_idx,
                                 smoothing=0.1)

    # Pretend the model outputs perfectly-confident logits for token 3.
    # Check how the smoothed target distribution looks.
    logits = torch.zeros(1, 1, vocab_size)
    logits[..., 3] = 10.0
    target = torch.tensor([[3]])

    loss = loss_fn(logits, target)

    # Re-build the smoothed distribution for visualisation.
    smooth_value = 0.1 / (vocab_size - 2)
    q = torch.full((vocab_size,), smooth_value)
    q[3] = 0.9
    q[padding_idx] = 0.0
    print("Smoothed target distribution q(k) for y = 3, <pad>=0:")
    for k, v in enumerate(q.tolist()):
        marker = "   ← true token" if k == 3 else "   ← <pad>, forced 0" if k == padding_idx else ""
        print(f"  q({k}) = {v:.4f}{marker}")
    print(f"\nLoss for a very-confident correct prediction: {loss.item():.4f}")
    print("(§5.4 notes this 'hurts perplexity' — the model can never")
    print(" reach zero KL because the target is never one-hot.)")


if __name__ == "__main__":
    main()
