"""
src/scaled_dot_product_attention.py
===================================

Scaled Dot-Product Attention from §3.2.1 of the paper — Equation (1).

Paper references
----------------
* §3.2.1 "Scaled Dot-Product Attention":

      > "The input consists of queries and keys of dimension d_k, and
      >  values of dimension d_v.  We compute the dot products of the
      >  query with all keys, divide each by √d_k, and apply a softmax
      >  function to obtain the weights on the values."

      > "In practice, we compute the attention function on a set of
      >  queries simultaneously, packed together into a matrix Q.
      >  The keys and values are also packed together into matrices
      >  K and V.  We compute the matrix of outputs as:
      >
      >     Attention(Q, K, V) = softmax( Q K^T / √d_k ) V.     (1)"

      > "The two most commonly used attention functions are additive
      >  attention, and dot-product (multiplicative) attention.
      >  Dot-product attention is identical to our algorithm, except
      >  for the scaling factor of 1/√d_k.  …  While for small values
      >  of d_k the two mechanisms perform similarly, additive
      >  attention outperforms dot product attention without scaling
      >  for larger values of d_k.  We suspect that for large values
      >  of d_k, the dot products grow large in magnitude, pushing
      >  the softmax function into regions where it has extremely
      >  small gradients.  To counteract this effect, we scale the
      >  dot products by 1/√d_k."

* §3.2.3 third bullet:
      > "…masking out (setting to −∞) all values in the input of the
      >  softmax which correspond to illegal connections."
  → the ``mask`` argument to :func:`scaled_dot_product_attention`.

* §5.4 "Regularization":
      > "Attention dropout" (Table 3 footnote) — some settings in the
      >  paper apply dropout to the *attention weights*; we expose a
      >  ``dropout`` argument for that purpose.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key:   torch.Tensor,
    value: torch.Tensor,
    mask:  Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Equation (1) of the paper: ``softmax(QKᵀ / √d_k) V``.

    This function is deliberately head-agnostic: ``query``, ``key``,
    ``value`` can be shaped either ``(batch, seq_len, d_k)`` for a
    single-head call or ``(batch, heads, seq_len, d_k)`` for the
    multi-head call in :class:`src.multi_head_attention.MultiHeadAttention`.

    Parameters
    ----------
    query : FloatTensor ``(..., q_len, d_k)``
        The queries Q.  Shape of the leading dimensions must match
        ``key`` and ``value``.
    key : FloatTensor ``(..., k_len, d_k)``
        The keys K.  Must share ``d_k`` with ``query``.
    value : FloatTensor ``(..., k_len, d_v)``
        The values V.  ``d_v`` may differ from ``d_k``; §3.2.2 uses
        ``d_k = d_v = d_model / h`` but the function does not require it.
    mask : BoolTensor broadcastable to ``(..., q_len, k_len)`` or ``None``
        ``True`` = keep, ``False`` = mask out (set pre-softmax to −∞).
        Used to hide <pad> (§3.2.3) and future positions (§3.1).
    dropout : nn.Dropout or ``None``
        Dropout applied to the attention weights *after* the softmax
        (as in the Table 3 "attention dropout" and several follow-up
        implementations of the paper).

    Returns
    -------
    output : FloatTensor ``(..., q_len, d_v)``
        Weighted sum of the values — the output of the attention.
    attn_weights : FloatTensor ``(..., q_len, k_len)``
        The softmax probabilities, returned for inspection (see
        §4 "Why Self-Attention" where the paper visualises them).

    Notes
    -----
    The scalar ``1/√d_k`` exists *only* because §3.2.1 argues that
    un-scaled dot-product attention has a softmax-saturation problem
    when ``d_k`` is large.  This is arguably the single most important
    equation in the paper — everything else in the architecture feeds
    into, or post-processes, this operation.
    """
    d_k = query.size(-1)                                      # §3.2.1: last dim is d_k

    # -- Step 1 : Q K^T  -------------------------------------------------------
    # `key.transpose(-2, -1)` swaps the last two dimensions so that a batched
    # matrix multiply gives the pairwise dot products between every query and
    # every key — i.e. the matrix "QK^T" of Eq. (1).
    scores = torch.matmul(query, key.transpose(-2, -1))       # shape (..., q_len, k_len)

    # -- Step 2 : scale by 1/√d_k  ---------------------------------------------
    # §3.2.1: "We scale the dot products by 1/√d_k."
    scores = scores / math.sqrt(d_k)

    # -- Step 3 : apply mask (§3.2.3) ------------------------------------------
    if mask is not None:
        # "Masking out (setting to −∞) all values in the input of the softmax
        #  which correspond to illegal connections." — §3.2.3.
        # `-inf` makes those entries have softmax-probability exactly zero.
        scores = scores.masked_fill(~mask, float("-inf"))

    # -- Step 4 : softmax over the last axis (keys) -----------------------------
    # "…and apply a softmax function to obtain the weights on the values." — §3.2.1.
    attn_weights = F.softmax(scores, dim=-1)                  # shape (..., q_len, k_len)

    # -- Step 5 : optional dropout on attention weights ------------------------
    # (Table 3 footnote and §5.4)
    if dropout is not None:
        attn_weights = dropout(attn_weights)

    # -- Step 6 : weighted sum of the values V ---------------------------------
    # This is the right-hand side of Eq. (1): `softmax(...) V`.
    output = torch.matmul(attn_weights, value)                # shape (..., q_len, d_v)

    return output, attn_weights
