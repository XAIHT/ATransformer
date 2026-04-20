"""
src/config.py
=============

Hyper-parameters for the Transformer, taken verbatim from the paper.

Paper references
----------------
* §3  "Model Architecture" — introduces the symbols ``d_model``, ``h``, ``N``.
* §3.2.2 "Multi-Head Attention"
      > "In this work we employ h = 8 parallel attention layers, or heads.
      >  For each of these we use d_k = d_v = d_model / h = 64."
* §3.3 "Position-wise Feed-Forward Networks"
      > "The dimensionality of input and output is d_model = 512,
      >  and the inner-layer has dimensionality d_ff = 2048."
* §5.4 "Regularization"
      > "We apply dropout … P_drop = 0.1."
      > "During training, we employed label smoothing of value ε_ls = 0.1."
* Table 3 "Variations on the Transformer architecture" — gives the
  "base" row (N=6, d_model=512, d_ff=2048, h=8, d_k=d_v=64, P_drop=0.1, ε_ls=0.1)
  and the "big" row (N=6, d_model=1024, d_ff=4096, h=16, P_drop=0.3, ε_ls=0.1).
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Container for every Transformer hyper-parameter mentioned in the paper.

    Attributes
    ----------
    src_vocab_size, tgt_vocab_size:
        Sizes of the source and target vocabularies.  In §5.1 the paper
        uses a 37 000-token shared byte-pair-encoding vocabulary for
        English↔German and a 32 000-token word-piece vocabulary for
        English↔French.  Here we leave them configurable so the toy
        training example can use a tiny vocabulary.
    d_model:
        §3.2.2 / Table 3: dimensionality of the model.  Base = 512.
    num_heads:
        §3.2.2 / Table 3: number of attention heads ``h``.  Base = 8.
    num_layers:
        §3.1 / Table 3: number of stacked encoder / decoder layers ``N``.
        Base = 6.
    d_ff:
        §3.3 / Table 3: inner feed-forward dimensionality.  Base = 2048.
    max_seq_len:
        Maximum sequence length we pre-compute positional encodings for.
        The paper trains on sequences up to length ≈ a few hundred tokens;
        5 000 is a safe educational default.
    dropout:
        §5.4 "Residual Dropout": P_drop.  Base = 0.1.
    label_smoothing:
        §5.4 "Label Smoothing": ε_ls = 0.1 for every configuration in Table 3.
    warmup_steps:
        §5.3 "Optimizer", Eq. (3): warmup_steps = 4000 in the paper.
    padding_idx:
        Conventional token id used for <pad>; masked out in attention.

    Notes
    -----
    The two instantiations :data:`BASE_CONFIG` and :data:`BIG_CONFIG` match
    the "base" and "big" rows of Table 3 of the paper.
    """

    # --- vocabulary --------------------------------------------------------
    src_vocab_size: int = 37000          # §5.1: "shared source-target vocabulary of about 37000 tokens"
    tgt_vocab_size: int = 37000          # §5.1: same BPE vocabulary for EN-DE

    # --- architecture ------------------------------------------------------
    d_model:       int = 512             # §3.2.2 / Table 3, base
    num_heads:     int = 8               # §3.2.2: h = 8
    num_layers:    int = 6               # §3.1  : N = 6
    d_ff:          int = 2048            # §3.3  : d_ff = 2048
    max_seq_len:   int = 5000            # §3.5  : max context for PE matrix

    # --- regularization ----------------------------------------------------
    dropout:         float = 0.1         # §5.4  : P_drop = 0.1
    label_smoothing: float = 0.1         # §5.4  : ε_ls = 0.1

    # --- optimizer ---------------------------------------------------------
    warmup_steps: int = 4000             # §5.3  : warmup_steps = 4000

    # --- misc --------------------------------------------------------------
    padding_idx:      int = 0            # convention for <pad> token
    share_embeddings: bool = True        # §3.4  : "we share the weight matrix
                                         # between the two embedding layers
                                         # and the pre-softmax linear transformation"

    # ------------------------------------------------------------------ derived
    @property
    def d_k(self) -> int:
        """§3.2.2: d_k = d_model / h (per-head key dimension)."""
        return self.d_model // self.num_heads

    @property
    def d_v(self) -> int:
        """§3.2.2: d_v = d_model / h (per-head value dimension)."""
        return self.d_model // self.num_heads


# ------------------------------------------------------------------ Table 3
# "base" row of Table 3 in the paper.
BASE_CONFIG = ModelConfig(
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048,
    dropout=0.1,
    label_smoothing=0.1,
    warmup_steps=4000,
)

# "big"  row of Table 3 in the paper.
BIG_CONFIG = ModelConfig(
    d_model=1024,
    num_heads=16,
    num_layers=6,
    d_ff=4096,
    dropout=0.3,          # §5.4: "For the big model … P_drop = 0.3 for EN-DE."
    label_smoothing=0.1,
    warmup_steps=4000,
)
