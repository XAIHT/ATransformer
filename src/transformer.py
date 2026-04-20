"""
src/transformer.py
==================

Complete Transformer model (§3 and Fig. 1 of the paper).

Paper references
----------------
* §3 "Model Architecture":

      > "Most competitive neural sequence transduction models have an
      >  encoder-decoder structure.  Here, the encoder maps an input
      >  sequence of symbol representations (x_1, …, x_n) to a sequence
      >  of continuous representations z = (z_1, …, z_n).  Given z, the
      >  decoder then generates an output sequence (y_1, …, y_m) of
      >  symbols one element at a time."

      > "The Transformer follows this overall architecture using stacked
      >  self-attention and point-wise, fully connected layers for both
      >  the encoder and decoder, shown in the left and right halves of
      >  Figure 1, respectively."

* §3.4 "Embeddings and Softmax":
      > "In our model, we share the same weight matrix between the two
      >  embedding layers and the pre-softmax linear transformation."

* §3.5 "Positional Encoding" — the sum ``embedding + PE`` that enters
  both the encoder and decoder (see :mod:`src.positional_encoding`).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .config import ModelConfig
from .embeddings import TokenEmbedding, OutputProjection
from .positional_encoding import PositionalEncoding
from .encoder import Encoder
from .decoder import Decoder
from .masking import make_src_mask, make_tgt_mask


class Transformer(nn.Module):
    """Full encoder-decoder Transformer of Fig. 1 of the paper.

    Parameters
    ----------
    cfg : :class:`ModelConfig`
        Hyper-parameter bundle.  Defaults match the "base" row of
        Table 3 of the paper.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # --------------------------------------------------------------
        # Embeddings (§3.4) — shared weight matrix between the two
        # embedding tables **and** the pre-softmax linear layer.
        # --------------------------------------------------------------
        if cfg.share_embeddings and cfg.src_vocab_size == cfg.tgt_vocab_size:
            # §3.4: "we share the same weight matrix between the two
            #        embedding layers and the pre-softmax linear
            #        transformation".
            self.src_embed = TokenEmbedding(cfg.src_vocab_size, cfg.d_model, cfg.padding_idx)
            self.tgt_embed = self.src_embed
            self.generator = OutputProjection(cfg.d_model, cfg.tgt_vocab_size,
                                              shared_embedding=self.tgt_embed)
        else:
            self.src_embed = TokenEmbedding(cfg.src_vocab_size, cfg.d_model, cfg.padding_idx)
            self.tgt_embed = TokenEmbedding(cfg.tgt_vocab_size, cfg.d_model, cfg.padding_idx)
            self.generator = OutputProjection(cfg.d_model, cfg.tgt_vocab_size)

        # --------------------------------------------------------------
        # Positional encoding (§3.5).  One instance is enough because it
        # is stateless — but we instantiate two so each side can have an
        # independent dropout draw (matches the reference impl.).
        # --------------------------------------------------------------
        self.src_pe = PositionalEncoding(cfg.d_model, cfg.max_seq_len, cfg.dropout)
        self.tgt_pe = PositionalEncoding(cfg.d_model, cfg.max_seq_len, cfg.dropout)

        # --------------------------------------------------------------
        # Encoder and decoder stacks (§3.1).
        # --------------------------------------------------------------
        self.encoder = Encoder(cfg.num_layers, cfg.d_model, cfg.num_heads,
                               cfg.d_ff, cfg.dropout)
        self.decoder = Decoder(cfg.num_layers, cfg.d_model, cfg.num_heads,
                               cfg.d_ff, cfg.dropout)

        # Xavier-uniform initialisation of every linear weight.  The
        # paper does not name an initialisation scheme, but the
        # reference tensor2tensor code uses Xavier/Glorot, which keeps
        # activations and gradients at roughly unit variance.
        self._init_parameters()

    # ------------------------------------------------------------------ helpers
    def _init_parameters(self) -> None:
        """Xavier-uniform for every parameter of dimension > 1."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------ encoder
    def encode(self,
               src:       torch.Tensor,
               src_mask:  Optional[torch.Tensor] = None
               ) -> torch.Tensor:
        """Encoder pathway (left half of Fig. 1).

        Parameters
        ----------
        src      : LongTensor ``(batch, src_len)`` — source token ids.
        src_mask : BoolTensor — hides <pad> in the source.

        Returns
        -------
        memory : FloatTensor ``(batch, src_len, d_model)``.
        """
        # Input embeddings (§3.4) + positional encoding (§3.5).
        x = self.src_embed(src)                   # §3.4 token embeddings × √d_model
        x = self.src_pe(x)                        # §3.5 add PE, then dropout (§5.4)
        return self.encoder(x, src_mask)          # §3.1 encoder stack

    # ------------------------------------------------------------------ decoder
    def decode(self,
               tgt:        torch.Tensor,
               memory:     torch.Tensor,
               src_mask:   Optional[torch.Tensor] = None,
               tgt_mask:   Optional[torch.Tensor] = None
               ) -> torch.Tensor:
        """Decoder pathway (right half of Fig. 1)."""
        x = self.tgt_embed(tgt)                   # §3.4
        x = self.tgt_pe(x)                        # §3.5
        return self.decoder(x, memory, src_mask=src_mask, tgt_mask=tgt_mask)

    # ------------------------------------------------------------------ forward
    def forward(self,
                src:       torch.Tensor,
                tgt:       torch.Tensor,
                src_mask:  Optional[torch.Tensor] = None,
                tgt_mask:  Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """End-to-end forward pass, returning logits over ``tgt_vocab``.

        The final softmax is **not** applied here — it is folded into
        the loss (see §3.4 and :class:`src.label_smoothing.LabelSmoothingLoss`).
        """
        # Auto-build masks if the caller did not supply them.
        if src_mask is None:
            src_mask = make_src_mask(src, self.cfg.padding_idx)
        if tgt_mask is None:
            tgt_mask = make_tgt_mask(tgt, self.cfg.padding_idx)

        memory = self.encode(src, src_mask)
        dec_out = self.decode(tgt, memory, src_mask=src_mask, tgt_mask=tgt_mask)

        # §3.4: linear + softmax.  Return raw logits; the caller applies
        # softmax (in inference) or cross-entropy / label-smoothing KL
        # (in training).
        return self.generator(dec_out)

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def greedy_decode(self,
                      src:            torch.Tensor,
                      max_len:        int,
                      start_symbol:   int,
                      end_symbol:     Optional[int] = None
                      ) -> torch.Tensor:
        """Auto-regressive greedy decoding (for demo / toy training).

        Paper
        -----
        §6.1 "Machine Translation" notes that the paper uses beam search
        with beam size 4 and length penalty α = 0.6.  Beam search is out
        of scope for this pedagogical repository; greedy decoding is
        good enough to observe the model working on the toy copy task.
        """
        self.eval()
        src_mask = make_src_mask(src, self.cfg.padding_idx)
        memory = self.encode(src, src_mask)

        ys = torch.full((src.size(0), 1), start_symbol,
                        dtype=torch.long, device=src.device)
        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, self.cfg.padding_idx)
            out    = self.decode(ys, memory, src_mask=src_mask, tgt_mask=tgt_mask)
            logits = self.generator(out[:, -1, :])     # only the last step
            next_tok = logits.argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            if end_symbol is not None and (next_tok == end_symbol).all():
                break
        return ys
