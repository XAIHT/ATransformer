"""
ATransformer — a line-by-line re-implementation of

    Vaswani et al., "Attention Is All You Need", NIPS 2017 (arXiv:1706.03762).

Each submodule is named after the section of the paper it implements:

* ``config``                       — Table 3 hyper-parameters.
* ``masking``                      — §3.2.3 padding + look-ahead masks.
* ``positional_encoding``          — §3.5 sinusoidal positional encoding.
* ``embeddings``                   — §3.4 shared input / output embeddings.
* ``scaled_dot_product_attention`` — §3.2.1, Eq. (1).
* ``multi_head_attention``         — §3.2.2.
* ``feed_forward``                 — §3.3, Eq. (2).
* ``encoder``                      — §3.1 encoder stack.
* ``decoder``                      — §3.1 decoder stack.
* ``transformer``                  — full model (Fig. 1).
* ``label_smoothing``              — §5.4 label-smoothing loss (ε_ls = 0.1).
* ``optimizer``                    — §5.3 Adam + Noam warm-up schedule (Eq. 3).
"""

from .config import ModelConfig, BASE_CONFIG, BIG_CONFIG  # noqa: F401
