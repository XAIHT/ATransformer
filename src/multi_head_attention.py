"""
src/multi_head_attention.py
===========================

Multi-Head Attention from §3.2.2 of the paper.

Paper references
----------------
* §3.2.2 "Multi-Head Attention":

      > "Instead of performing a single attention function with
      >  d_model-dimensional keys, values and queries, we found it
      >  beneficial to linearly project the queries, keys and values
      >  h times with different, learned linear projections to d_k,
      >  d_k and d_v dimensions, respectively.  On each of these
      >  projected versions of queries, keys and values we then
      >  perform the attention function in parallel, yielding
      >  d_v-dimensional output values.  These are concatenated and
      >  once again projected, resulting in the final values, as
      >  depicted in Figure 2."

      > "MultiHead(Q, K, V) = Concat(head_1, …, head_h) W^O
      >  where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)."

      > "Where the projections are parameter matrices
      >     W_i^Q ∈ ℝ^{d_model × d_k},
      >     W_i^K ∈ ℝ^{d_model × d_k},
      >     W_i^V ∈ ℝ^{d_model × d_v},
      >     W^O   ∈ ℝ^{h·d_v × d_model}."

      > "In this work we employ h = 8 parallel attention layers, or
      >  heads. For each of these we use d_k = d_v = d_model/h = 64.
      >  Due to the reduced dimension of each head, the total
      >  computational cost is similar to that of single-head attention
      >  with full dimensionality."

* §3.2.3 "Applications of Attention in our Model" — the three call
  sites (encoder self-attention, decoder self-attention with look-ahead
  mask, and encoder-decoder cross-attention).  This module supports all
  three: the *caller* just supplies the right ``query``/``key``/``value``
  tensors and mask.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """h parallel scaled-dot-product attention heads (§3.2.2).

    Efficient implementation trick
    ------------------------------
    Rather than keeping ``h`` separate linear layers per Q / K / V, we
    pack them into a single ``(d_model → d_model)`` linear projection
    and then ``view`` the output as ``(batch, seq, h, d_k)``.  This is
    mathematically identical to the paper's formulation:

        W^Q = [ W_1^Q | W_2^Q | … | W_h^Q ]   (concatenated columns)

    and gives the same parameter count: ``h · d_model · d_k = d_model²``.
    """

    def __init__(self,
                 d_model:  int,
                 num_heads: int,
                 dropout:  float = 0.1):
        super().__init__()

        # §3.2.2: "d_k = d_v = d_model / h" — requires divisibility.
        assert d_model % num_heads == 0, \
            "d_model must be divisible by num_heads (§3.2.2: d_k = d_model / h)."

        self.d_model   = d_model                 # §3.2.2
        self.num_heads = num_heads               # §3.2.2: h
        self.d_k       = d_model // num_heads    # §3.2.2: d_k = d_model / h
        self.d_v       = self.d_k                # §3.2.2: d_v = d_k in the paper

        # W^Q, W^K, W^V  — each is (d_model × d_model) but is logically
        # a stack of h projections of shape (d_model × d_k).
        self.W_Q = nn.Linear(d_model, d_model, bias=False)    # §3.2.2 W_i^Q
        self.W_K = nn.Linear(d_model, d_model, bias=False)    # §3.2.2 W_i^K
        self.W_V = nn.Linear(d_model, d_model, bias=False)    # §3.2.2 W_i^V

        # W^O : (h·d_v × d_model) = (d_model × d_model).
        self.W_O = nn.Linear(d_model, d_model, bias=False)    # §3.2.2 W^O

        # Dropout on attention weights (Table 3 footnote / §5.4).
        self.attn_dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------ helpers
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape ``(batch, seq, d_model)`` → ``(batch, heads, seq, d_k)``.

        Corresponds to §3.2.2:
            > "…linearly project the queries, keys and values h times
            >  with different, learned linear projections to d_k, d_k
            >  and d_v dimensions, respectively."
        We do the h projections *together* (efficient trick) and then
        split the last dimension into (heads, d_k).
        """
        batch, seq, _ = x.shape
        # view splits (d_model) into (num_heads, d_k); transpose puts
        # the head axis before the sequence axis so the batched
        # matmul in SDPA works per-head.
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`_split_heads`: ``(batch, heads, seq, d_v)`` → ``(batch, seq, d_model)``.

        Implements the ``Concat`` in
            > "MultiHead(Q, K, V) = Concat(head_1, …, head_h) W^O" — §3.2.2.
        """
        batch, _, seq, _ = x.shape
        # transpose back, then merge (heads, d_v) into (h·d_v) = d_model.
        # .contiguous() is necessary because transpose returns a non-
        # contiguous view and .view() requires contiguous storage.
        return x.transpose(1, 2).contiguous().view(batch, seq, self.d_model)

    # ------------------------------------------------------------------ forward
    def forward(self,
                query: torch.Tensor,
                key:   torch.Tensor,
                value: torch.Tensor,
                mask:  Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute MultiHead(Q, K, V) of §3.2.2.

        Parameters
        ----------
        query, key, value : FloatTensor ``(batch, seq, d_model)``
            The three inputs to the attention.  Who supplies them differs
            by call site (§3.2.3):
                * encoder self-attention      : Q = K = V = enc input
                * decoder self-attention       : Q = K = V = dec input
                  (with subsequent mask)
                * encoder–decoder attention    : Q = dec state,
                                                  K = V = enc output
        mask : BoolTensor broadcastable to ``(batch, heads, q_len, k_len)``
            ``True`` = keep, ``False`` = mask out.

        Returns
        -------
        output : FloatTensor ``(batch, q_len, d_model)``.
        attn   : FloatTensor ``(batch, heads, q_len, k_len)``
            The attention weights per head (useful for Fig. 3–5 style
            visualisations in §4).
        """
        # 1) project into Q, K, V in d_model dimensions
        Q = self.W_Q(query)          # §3.2.2: Q W^Q
        K = self.W_K(key)            # §3.2.2: K W^K
        V = self.W_V(value)          # §3.2.2: V W^V

        # 2) split into h heads of width d_k (/ d_v)
        Q = self._split_heads(Q)     # §3.2.2 "h parallel attention layers"
        K = self._split_heads(K)
        V = self._split_heads(V)

        # 3) run scaled dot-product attention (§3.2.1 / Eq. 1) in parallel
        #    across the h heads.
        head_outputs, attn = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.attn_dropout
        )   # head_outputs: (batch, heads, q_len, d_v)

        # 4) Concat all heads back into d_model, then apply W^O.
        concat = self._merge_heads(head_outputs)     # §3.2.2 Concat
        output = self.W_O(concat)                    # §3.2.2 · W^O

        return output, attn
