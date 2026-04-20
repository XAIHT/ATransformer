"""
src/encoder.py
==============

Encoder stack from §3.1 of the paper (left-hand side of Fig. 1).

Paper references
----------------
* §3.1 "Encoder and Decoder Stacks" — encoder paragraph:

      > "Encoder: The encoder is composed of a stack of N = 6 identical
      >  layers. Each layer has two sub-layers.  The first is a
      >  multi-head self-attention mechanism, and the second is a simple,
      >  position-wise fully connected feed-forward network.  We employ
      >  a residual connection around each of the two sub-layers,
      >  followed by layer normalization.  That is, the output of each
      >  sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is
      >  the function implemented by the sub-layer itself.  To
      >  facilitate these residual connections, all sub-layers in the
      >  model, as well as the embedding layers, produce outputs of
      >  dimension d_model = 512."

* §5.4 "Regularization" — Residual Dropout:
      > "We apply dropout to the output of each sub-layer, before it is
      >  added to the sub-layer input and normalized."
  → this is what :class:`SublayerConnection` does.

* §3.2.3 "Applications of Attention in our Model", first bullet:
      > "In ‘encoder-decoder attention’ layers, the queries come from
      >  the previous decoder layer, and the memory keys and values come
      >  from the output of the encoder."
  → i.e. the encoder produces ``memory``; the decoder in
  :mod:`src.decoder` uses it for its second attention sub-layer.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .multi_head_attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class SublayerConnection(nn.Module):
    """Residual connection + LayerNorm as described in §3.1.

    Implements the exact sentence
        > "the output of each sub-layer is LayerNorm(x + Sublayer(x))"

    but with the §5.4 detail
        > "We apply dropout to the output of each sub-layer, before it
        >  is added to the sub-layer input and normalized."

    So the computation is:
        ``LayerNorm(x + Dropout(Sublayer(x)))``.
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)        # §3.1 "followed by layer normalization"
        self.dropout = nn.Dropout(dropout)       # §5.4 residual dropout

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        """``x + Dropout(Sublayer(x))`` then LayerNorm."""
        # `sublayer(x)` is either a multi-head attention or FFN call.
        return self.norm(x + self.dropout(sublayer(x)))


# ---------------------------------------------------------------- single layer
class EncoderLayer(nn.Module):
    """One of the N = 6 identical encoder layers (§3.1).

    Contains the two sub-layers described in the encoder paragraph of
    §3.1:

    1. Multi-head **self-attention** (§3.2.3, first of three use cases,
       applied with Q = K = V = the layer's input).
    2. Position-wise feed-forward network (§3.3, Eq. 2).

    Both are wrapped in a residual connection + LayerNorm.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.res1      = SublayerConnection(d_model, dropout)    # around self-attn
        self.res2      = SublayerConnection(d_model, dropout)    # around FFN

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the two sub-layers of §3.1 on ``x``.

        Parameters
        ----------
        x    : FloatTensor ``(batch, src_len, d_model)``.
        mask : Encoder self-attention mask (only hides <pad>).

        Returns
        -------
        FloatTensor ``(batch, src_len, d_model)`` — same shape as input
        (essential for the residual connection to be well-defined).
        """
        # Sub-layer 1: self-attention with Q = K = V = x (§3.2.3 use #1).
        x = self.res1(x, lambda t: self.self_attn(t, t, t, mask=mask)[0])
        # Sub-layer 2: position-wise FFN (§3.3).
        x = self.res2(x, self.ffn)
        return x


# ---------------------------------------------------------------- stack of N
class Encoder(nn.Module):
    """Stack of ``N`` :class:`EncoderLayer` s (§3.1, ``N = 6`` in the paper)."""

    def __init__(self, num_layers: int, d_model: int, num_heads: int,
                 d_ff: int, dropout: float):
        super().__init__()
        # ModuleList of N distinct layers — each with its own parameters
        # (§3.3: "they use different parameters from layer to layer").
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        # Final LayerNorm: The pre-norm reference implementations and
        # the original tensor2tensor code both apply a final LayerNorm
        # after the stack; we keep it for numerical stability.
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply the N encoder layers in sequence.

        Parameters
        ----------
        x    : FloatTensor ``(batch, src_len, d_model)`` — output of
               the embedding + positional-encoding sum.
        mask : Encoder self-attention mask.

        Returns
        -------
        ``memory`` : FloatTensor ``(batch, src_len, d_model)``
            The encoder output used as keys/values by the decoder's
            cross-attention (§3.2.3 first bullet).
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
