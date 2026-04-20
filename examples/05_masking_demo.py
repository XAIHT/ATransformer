"""
examples/05_masking_demo.py
===========================

Visualise the two masks used in the paper:

* Padding mask (covered implicitly whenever §3.2.3 talks about
  "illegal connections").
* Subsequent / look-ahead mask from §3.1:

      > "We also modify the self-attention sub-layer in the decoder
      >  stack to prevent positions from attending to subsequent
      >  positions."
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.masking import padding_mask, subsequent_mask, make_tgt_mask


def main():
    # --- padding mask ------------------------------------------------------
    # Suppose pad_idx = 0 and a batch of 2 sequences, padded to length 5.
    src = torch.tensor([[7, 3, 5, 0, 0],       # real length 3
                        [2, 4, 1, 6, 0]])      # real length 4
    p_mask = padding_mask(src, pad_idx=0)
    print("Padding mask shape:", tuple(p_mask.shape))
    print("p_mask[0] (should be True, True, True, False, False):")
    print(p_mask[0, 0, 0])
    print("p_mask[1] (should be True, True, True, True, False):")
    print(p_mask[1, 0, 0])

    # --- subsequent (look-ahead) mask -------------------------------------
    s_mask = subsequent_mask(5)
    print("\nSubsequent mask (5x5, lower-triangular):")
    print(s_mask[0, 0].int())

    # --- combined target mask ---------------------------------------------
    tgt = torch.tensor([[1, 4, 5, 0, 0]])     # real length 3
    t_mask = make_tgt_mask(tgt, pad_idx=0)
    print("\nCombined target mask (padding AND no-look-ahead), shape:",
          tuple(t_mask.shape))
    print(t_mask[0, 0].int())


if __name__ == "__main__":
    main()
