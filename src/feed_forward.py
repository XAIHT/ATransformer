"""
src/feed_forward.py
===================

Position-wise Feed-Forward Network from §3.3 of the paper — Equation (2).

Paper references
----------------
* §3.3 "Position-wise Feed-Forward Networks":

      > "In addition to attention sub-layers, each of the layers in our
      >  encoder and decoder contains a fully connected feed-forward
      >  network, which is applied to each position separately and
      >  identically.  This consists of two linear transformations with
      >  a ReLU activation in between."

      > "FFN(x) = max(0, x W_1 + b_1) W_2 + b_2.        (2)"

      > "While the linear transformations are the same across different
      >  positions, they use different parameters from layer to layer.
      >  Another way of describing this is as two convolutions with
      >  kernel size 1.  The dimensionality of input and output is
      >  d_model = 512, and the inner-layer has dimensionality
      >  d_ff = 2048."

* §5.4 "Regularization" (residual dropout):
      > "We apply dropout to the output of each sub-layer, before it is
      >  added to the sub-layer input and normalized."
  → dropout on the *output of the FFN* is handled by the residual
  wrapper in :mod:`src.encoder` / :mod:`src.decoder`.  The dropout
  below is the one commonly applied *inside* the FFN, between the ReLU
  and the second linear — this follows the reference implementation
  (tensor2tensor) that accompanies the paper.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    """Two-layer MLP applied position-wise (§3.3, Eq. (2)).

    Parameters
    ----------
    d_model : int
        Input / output dimensionality.  Paper base: 512.
    d_ff : int
        Inner hidden size.  Paper base: 2048.
    dropout : float
        Dropout applied after the ReLU.  Paper §5.4: P_drop = 0.1.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.W_1 = nn.Linear(d_model, d_ff)            # §3.3: x W_1 + b_1
        self.W_2 = nn.Linear(d_ff, d_model)            # §3.3: · W_2 + b_2
        self.dropout = nn.Dropout(dropout)             # §5.4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``FFN(x) = max(0, x W_1 + b_1) W_2 + b_2`` (§3.3 Eq. 2).

        Parameters
        ----------
        x : FloatTensor ``(batch, seq, d_model)``.
        Returns
        -------
        FloatTensor ``(batch, seq, d_model)``.
        """
        # Step 1:  x W_1 + b_1                       (§3.3, inside Eq. (2))
        h = self.W_1(x)
        # Step 2:  max(0, ·)  — ReLU activation       (§3.3 "ReLU activation")
        h = F.relu(h)
        # Step 3:  dropout (tensor2tensor convention matching §5.4)
        h = self.dropout(h)
        # Step 4:  · W_2 + b_2                        (§3.3, Eq. (2))
        return self.W_2(h)
