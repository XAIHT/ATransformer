"""
src/positional_encoding.py
==========================

Sinusoidal positional encoding from §3.5 of the paper.

Paper references
----------------
* §3.5 "Positional Encoding":
      > "Since our model contains no recurrence and no convolution, in
      >  order for the model to make use of the order of the sequence,
      >  we must inject some information about the relative or absolute
      >  position of the tokens in the sequence."
  → this module produces a fixed ``(max_len, d_model)`` matrix PE that
    we add to the token embeddings at the input of the encoder and
    decoder.

* §3.5 continues:
      > "In this work, we use sine and cosine functions of different
      >  frequencies:
      >     PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
      >     PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
      >  … We chose this function because we hypothesized it would
      >  allow the model to easily learn to attend by relative
      >  positions, since for any fixed offset k, PE(pos+k) can be
      >  represented as a linear function of PE(pos)."

* §3.5 last sentence:
      > "We also experimented with using learned positional embeddings
      >  instead … the two versions produced nearly identical results."
  We implement the sinusoidal version because it is what Fig. 1 of the
  paper uses and because it generalizes to sequences longer than any
  seen during training (also mentioned in §3.5).
"""

from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Deterministic sine / cosine positional encoding of §3.5.

    Parameters
    ----------
    d_model : int
        Embedding dimensionality (512 for the "base" model; §3.2.2).
    max_len : int
        Largest sequence length for which we pre-compute PE.
    dropout : float
        §5.4 Residual Dropout:
            > "In addition, we apply dropout to the sums of the
            >  embeddings and the positional encodings in both the
            >  encoder and decoder stacks."
        Hence the :class:`nn.Dropout` applied to the *sum* in
        :meth:`forward`.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)       # §5.4 residual dropout on embeddings+PE

        # --------------------------------------------------------------
        # Build the (max_len, d_model) matrix of §3.5.
        # --------------------------------------------------------------
        pe = torch.zeros(max_len, d_model)                  # the output matrix PE

        # "pos" column vector of positions 0, 1, …, max_len-1
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

        # We want the denominator 10000^(2i/d_model) for i = 0 … d_model/2-1.
        # Working in log-space is numerically safer:
        #     10000^(2i/d_model) = exp(2i * (-log 10000) / d_model) ^ -1
        # so the *reciprocal* (which is what we actually multiply by) is
        #     exp(2i * (-log 10000) / d_model).
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)      # 2i  for i = 0,1,…
            * (-math.log(10000.0) / d_model)                    # paper constant 10000
        )                                                       # shape (d_model/2,)

        # Even columns: sin(pos / 10000^(2i/d_model))    — §3.5, Eq. for PE(pos,2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Odd columns : cos(pos / 10000^(2i/d_model))    — §3.5, Eq. for PE(pos,2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Store as a non-trainable buffer so it moves with the module
        # (to GPU, to half-precision, …) but is NOT updated by the
        # optimizer — the paper defines PE as a *fixed* function of pos.
        self.register_buffer("pe", pe.unsqueeze(0))            # (1, max_len, d_model)

    # --------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to ``x``.

        Parameters
        ----------
        x : FloatTensor of shape ``(batch, seq_len, d_model)``
            Token embeddings already scaled by √d_model (see §3.4 and
            :class:`src.embeddings.TokenEmbedding`).

        Returns
        -------
        FloatTensor of shape ``(batch, seq_len, d_model)``.

        Paper
        -----
        §3.4 last sentence:
            > "In the embedding layers, we multiply those weights by √d_model."
        §3.5 Fig. 1 shows that the *sum* embedding + PE is then passed
        through dropout before entering the first encoder / decoder layer
        — this matches :class:`nn.Dropout` below.
        """
        seq_len = x.size(1)
        # Slicing .pe lets us accept any length up to ``max_len`` without
        # recomputing sines and cosines.
        x = x + self.pe[:, :seq_len, :]                       # §3.5 sum
        return self.dropout(x)                                # §5.4 dropout(embed + PE)
