"""
src/decoder.py
==============

Decoder stack from §3.1 of the paper (right-hand side of Fig. 1).

Paper references
----------------
* §3.1 "Encoder and Decoder Stacks" — decoder paragraph:

      > "Decoder: The decoder is also composed of a stack of N = 6
      >  identical layers.  In addition to the two sub-layers in each
      >  encoder layer, the decoder inserts a third sub-layer, which
      >  performs multi-head attention over the output of the encoder
      >  stack.  Similar to the encoder, we employ residual connections
      >  around each of the sub-layers, followed by layer normalization.
      >  We also modify the self-attention sub-layer in the decoder stack
      >  to prevent positions from attending to subsequent positions.
      >  This masking, combined with the fact that the output embeddings
      >  are offset by one position, ensures that the predictions for
      >  position i can depend only on the known outputs at positions
      >  less than i."

* §3.2.3 "Applications of Attention in our Model":
      > "In ‘encoder-decoder attention’ layers, the queries come from
      >  the previous decoder layer, and the memory keys and values
      >  come from the output of the encoder.  This allows every
      >  position in the decoder to attend over all positions in the
      >  input sequence.  This mimics the typical encoder-decoder
      >  attention mechanisms in sequence-to-sequence models such as
      >  [38, 2, 9]."
      > "…The encoder contains self-attention layers…"
      > "Similarly, self-attention layers in the decoder allow each
      >  position in the decoder to attend to all positions in the
      >  decoder up to and including that position.  We need to
      >  prevent leftward information flow in the decoder to preserve
      >  the auto-regressive property."
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .multi_head_attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward
from .encoder import SublayerConnection


class DecoderLayer(nn.Module):
    """One of the N = 6 identical decoder layers (§3.1).

    Three sub-layers, in order:

    1. Masked multi-head **self-attention** over the decoder input
       (§3.2.3 use #2 + look-ahead mask from §3.1).
    2. Multi-head **encoder-decoder attention**: queries from the
       decoder, keys/values from the encoder's ``memory``
       (§3.2.3 use #3).
    3. Position-wise feed-forward network (§3.3).
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        # Sub-layer 1 — masked self-attention.
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        # Sub-layer 2 — encoder-decoder (cross) attention.
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        # Sub-layer 3 — position-wise FFN.
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)

        # One residual+LayerNorm wrapper per sub-layer (§3.1).
        self.res1 = SublayerConnection(d_model, dropout)
        self.res2 = SublayerConnection(d_model, dropout)
        self.res3 = SublayerConnection(d_model, dropout)

    def forward(self,
                x:         torch.Tensor,
                memory:    torch.Tensor,
                src_mask:  Optional[torch.Tensor] = None,
                tgt_mask:  Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        """Run the three sub-layers on the decoder input ``x``.

        Parameters
        ----------
        x        : ``(batch, tgt_len, d_model)`` — decoder input
                   (embedding + PE + possibly previous decoder layers).
        memory   : ``(batch, src_len, d_model)`` — encoder output, used
                   as keys/values in the cross-attention (§3.2.3 #3).
        src_mask : mask for the cross-attention (hide <pad> in the source).
        tgt_mask : mask for the self-attention: hide <pad> in the target
                   **and** hide future positions (§3.1 look-ahead mask).
        """
        # Sub-layer 1: MASKED self-attention, Q = K = V = x.
        x = self.res1(x, lambda t: self.self_attn(t, t, t, mask=tgt_mask)[0])

        # Sub-layer 2: encoder-decoder attention.
        # Q from the decoder (x), K and V from the encoder memory.
        x = self.res2(x, lambda t: self.cross_attn(t, memory, memory, mask=src_mask)[0])

        # Sub-layer 3: position-wise FFN.
        x = self.res3(x, self.ffn)
        return x


class Decoder(nn.Module):
    """Stack of N decoder layers (§3.1, N = 6 in the paper)."""

    def __init__(self, num_layers: int, d_model: int, num_heads: int,
                 d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)    # final LayerNorm (tensor2tensor convention)

    def forward(self,
                x:         torch.Tensor,
                memory:    torch.Tensor,
                src_mask:  Optional[torch.Tensor] = None,
                tgt_mask:  Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.norm(x)
