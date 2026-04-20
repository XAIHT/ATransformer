"""
src/masking.py
==============

Helpers to build the two kinds of masks used by the Transformer.

Paper references
----------------
* §3.1 "Encoder and Decoder Stacks" — last sentence about the decoder:
      > "We also modify the self-attention sub-layer in the decoder stack
      >  to prevent positions from attending to subsequent positions.
      >  This masking, combined with the fact that the output embeddings
      >  are offset by one position, ensures that the predictions for
      >  position i can depend only on the known outputs at positions
      >  less than i."
  → implemented by :func:`subsequent_mask` ("look-ahead mask").

* §3.2.3 "Applications of Attention in our Model" — third bullet:
      > "We need to prevent leftward information flow in the decoder to
      >  preserve the auto-regressive property.  We implement this inside
      >  of scaled dot-product attention by masking out (setting to −∞)
      >  all values in the input of the softmax which correspond to
      >  illegal connections."
  → implemented by the way we pass the mask to
    :func:`scaled_dot_product_attention`.

* §5.4 "Regularization" indirectly — padding tokens never contribute to
  the loss and must never be attended to; hence :func:`padding_mask`.
"""

from __future__ import annotations

import torch


def padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """Return ``True`` for *real* tokens and ``False`` for ``<pad>`` tokens.

    Parameters
    ----------
    seq : LongTensor of shape ``(batch, seq_len)``
        Token ids.
    pad_idx : int
        Id of the <pad> token.  Default 0 (see :class:`ModelConfig`).

    Returns
    -------
    BoolTensor of shape ``(batch, 1, 1, seq_len)``
        Broadcastable to the attention-score tensor
        ``(batch, heads, query_len, key_len)``.  ``True`` means
        "this key position is a real token and can be attended to".

    Paper
    -----
    Padding is not discussed explicitly (the paper assumes batches of
    equal length after bucketing, §5.1), but every practical
    implementation must mask <pad> out of the softmax exactly as
    described in §3.2.3 for the decoder self-attention.
    """
    # seq == pad_idx  -> True where padding, invert to get "valid" positions
    mask = (seq != pad_idx)                        # (batch, seq_len)
    return mask.unsqueeze(1).unsqueeze(2)          # (batch, 1, 1, seq_len)


def subsequent_mask(size: int) -> torch.Tensor:
    """Return a lower-triangular Boolean mask of shape ``(1, 1, size, size)``.

    Cell ``[i, j] = True`` iff ``j <= i``, i.e. position ``i`` is allowed
    to attend to position ``j``.  This is the *look-ahead* / *causal*
    mask described at the end of §3.1:

        > "We also modify the self-attention sub-layer in the decoder
        >  stack to prevent positions from attending to subsequent
        >  positions."
    """
    # torch.tril builds a lower-triangular matrix of ones; ``.bool()``
    # turns them into True / False so downstream code can do
    # ``scores.masked_fill(~mask, -inf)``.
    m = torch.tril(torch.ones(size, size, dtype=torch.bool))
    return m.unsqueeze(0).unsqueeze(0)             # (1, 1, size, size)


def make_src_mask(src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """Mask used by the encoder self-attention (only hides padding)."""
    return padding_mask(src, pad_idx)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """Mask used by the decoder self-attention.

    Combines :func:`padding_mask` (hide <pad> in the target) with
    :func:`subsequent_mask` (hide future positions) via a logical AND
    — this is exactly the construction described in §3.1 and §3.2.3.
    """
    pad = padding_mask(tgt, pad_idx)               # (batch, 1, 1, tgt_len)
    sub = subsequent_mask(tgt.size(1)).to(pad.device)  # (1, 1, tgt_len, tgt_len)
    return pad & sub                               # broadcast to
                                                    # (batch, 1, tgt_len, tgt_len)
