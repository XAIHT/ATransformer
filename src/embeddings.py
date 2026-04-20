"""
src/embeddings.py
=================

Token embedding layer from §3.4 of the paper.

Paper references
----------------
* §3.4 "Embeddings and Softmax":
      > "Similarly to other sequence transduction models, we use learned
      >  embeddings to convert the input tokens and output tokens to
      >  vectors of dimension d_model. We also use the usual learned
      >  linear transformation and softmax function to convert the decoder
      >  output to predicted next-token probabilities. In our model, we
      >  share the same weight matrix between the two embedding layers
      >  and the pre-softmax linear transformation, similar to
      >  [Press & Wolf, 2017]. In the embedding layers, we multiply those
      >  weights by √d_model."
"""

from __future__ import annotations

import math

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Word embedding of dimension ``d_model``, scaled by √d_model (§3.4).

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the vocabulary.
    d_model : int
        Embedding dimensionality.
    padding_idx : int
        Token id whose embedding will be held at zero and never receive
        gradient — this is the standard way of implementing "<pad> does
        not participate in the loss or attention".
    """

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0):
        super().__init__()
        # Learned (vocab_size × d_model) embedding matrix E.
        self.lut = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.d_model = d_model                            # kept for the √d_model scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Look up embeddings and scale by √d_model.

        Parameters
        ----------
        x : LongTensor ``(batch, seq_len)``
            Token ids.

        Returns
        -------
        FloatTensor ``(batch, seq_len, d_model)``.

        Paper
        -----
        §3.4: "In the embedding layers, we multiply those weights by √d_model."
        The paper does not motivate the scaling explicitly, but the
        widely-accepted intuition is that it keeps the variance of the
        pre-softmax logits and of the positional-encoding sum comparable.
        """
        return self.lut(x) * math.sqrt(self.d_model)       # §3.4 √d_model scaling


class OutputProjection(nn.Module):
    """Pre-softmax linear layer that turns decoder states into logits.

    §3.4 asks us to **share** this matrix with the two token-embedding
    tables.  When ``shared_embedding`` is supplied, we simply keep a
    reference to it and compute logits as ``x @ E^T``.
    """

    def __init__(self, d_model: int, vocab_size: int,
                 shared_embedding: "TokenEmbedding | None" = None):
        super().__init__()
        if shared_embedding is None:
            # Un-tied version (kept for clarity; not used by default).
            self.proj = nn.Linear(d_model, vocab_size, bias=False)
            self.shared_embedding = None
        else:
            # Tied version, as in §3.4.
            self.proj = None
            self.shared_embedding = shared_embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return un-normalised logits over the vocabulary.

        The final softmax itself is applied by the loss function
        (:class:`src.label_smoothing.LabelSmoothingLoss` — §5.4).
        """
        if self.shared_embedding is not None:
            # Tied:  logits = x · E^T
            return x @ self.shared_embedding.lut.weight.T
        return self.proj(x)                               # (batch, seq_len, vocab_size)
