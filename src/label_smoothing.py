"""
src/label_smoothing.py
======================

Label-smoothing loss from §5.4 of the paper.

Paper references
----------------
* §5.4 "Regularization" — Label Smoothing:

      > "During training, we employed label smoothing of value
      >  ε_ls = 0.1 [36].  This hurts perplexity, as the model learns
      >  to be more unsure, but improves accuracy and BLEU score."

* Reference [36] = "Rethinking the Inception Architecture for Computer
  Vision" (Szegedy et al., 2016), which introduces label smoothing.

Formulation
-----------
For a vocabulary of size ``V``, the standard one-hot target
distribution ``δ_y`` puts probability 1 on the correct token and 0
elsewhere.  Label smoothing replaces it with::

    q(k) = (1 − ε_ls)           if  k = y
         = ε_ls / (V − 1)       otherwise

and the loss becomes the KL divergence between the model's softmax
output and ``q``.  We mask out the padding token from both the sum
and from the smoothed distribution.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """KL-divergence loss with label smoothing (§5.4, ε_ls = 0.1).

    Parameters
    ----------
    vocab_size : int
        Size of the target vocabulary (V).
    padding_idx : int
        Token id to exclude from the loss and from the smoothed
        distribution (so probability mass is never placed on <pad>).
    smoothing : float
        The value of ε_ls.  Paper default: 0.1.
    """

    def __init__(self, vocab_size: int, padding_idx: int = 0, smoothing: float = 0.1):
        super().__init__()
        assert 0.0 <= smoothing < 1.0, "smoothing must be in [0, 1)"
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing                   # ε_ls in the paper
        self.confidence = 1.0 - smoothing            # 1 - ε_ls = mass on true token
        # The KL-divergence of two distributions is a natural fit here;
        # `reduction="sum"` matches the §5.4 formulation (the denominator
        # is the number of non-padding tokens, computed below).
        self.criterion = nn.KLDivLoss(reduction="sum")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the smoothed KL loss.

        Parameters
        ----------
        logits : FloatTensor ``(batch, tgt_len, vocab_size)``
            Raw (un-softmaxed) output of :class:`src.embeddings.OutputProjection`.
        target : LongTensor ``(batch, tgt_len)``
            Ground-truth token ids.

        Returns
        -------
        FloatTensor scalar
            The per-(non-pad)-token KL divergence, i.e.
            ``Σ_{t : target_t ≠ pad} KL(q_t || p_t)  /  #{t : target_t ≠ pad}``.
        """
        # Flatten to (N, V) / (N,) so we can index into the smoothed
        # distribution with scatter_.
        logits = logits.reshape(-1, self.vocab_size)          # (N, V)
        target = target.reshape(-1)                           # (N,)

        # ------------------------------------------------------ build q
        # Start with a uniform ε_ls / (V - 2) over every "other" token.
        # We subtract 2 because the true token and the padding id each
        # get their own special treatment below.
        smooth_value = self.smoothing / (self.vocab_size - 2)
        q = torch.full_like(logits, smooth_value)             # (N, V)
        # Place (1 - ε_ls) on the true class — this is the "confidence".
        q.scatter_(1, target.unsqueeze(1), self.confidence)
        # Never place any probability on <pad>.
        q[:, self.padding_idx] = 0.0

        # Rows where the TARGET is <pad> should be ignored entirely.
        pad_mask = (target == self.padding_idx)
        if pad_mask.any():
            q.masked_fill_(pad_mask.unsqueeze(1), 0.0)

        # ------------------------------------------------------ KL loss
        # KLDivLoss expects log-probabilities in the first argument.
        log_p = F.log_softmax(logits, dim=-1)                 # (N, V)

        # Sum KL over all rows and divide by the number of *real* tokens
        # — matches what the paper does when reporting per-token losses.
        n_tokens = (~pad_mask).sum().clamp(min=1)
        return self.criterion(log_p, q) / n_tokens
