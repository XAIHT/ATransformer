"""
examples/01_positional_encoding_demo.py
=======================================

Visualise the sinusoidal positional encoding of §3.5.

Run
---
    python examples/01_positional_encoding_demo.py

Outputs
-------
* A printout of PE's shape and the first few rows.
* Saves `positional_encoding.png` next to this file — a heatmap of
  PE exactly like the one often reproduced from the paper.

Paper
-----
§3.5 "Positional Encoding":
    > PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    > PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

The heatmap should show:
* Low-dimension columns (small i) varying quickly along `pos`.
* High-dimension columns (large i) varying slowly — almost constant.
This "frequency ladder" is the geometric progression the paper builds
so the model can attend by *relative* offsets.
"""

import os
import sys

# --- make `import src.*` work when running as a script ---------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # noqa: E402
from src.positional_encoding import PositionalEncoding  # noqa: E402


def main():
    d_model = 128
    max_len = 100
    pe = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)
    # Extract the raw (unscaled) PE matrix from the buffer.
    PE = pe.pe.squeeze(0).numpy()                     # (max_len, d_model)

    print(f"PE shape: {PE.shape}  (max_len, d_model)")
    print("PE[0, :8]  =", PE[0, :8])                  # first token, first 8 dims
    print("PE[1, :8]  =", PE[1, :8])                  # second token
    print("PE[50, :8] =", PE[50, :8])                 # middle of the sequence
    print()
    print("Every *even*-indexed column is a sine; every *odd*-indexed column")
    print("is a cosine; column j uses frequency 1 / 10000^(2⌊j/2⌋ / d_model).")

    # Try plotting if matplotlib is available.
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(PE, aspect="auto", cmap="RdBu")
        ax.set_xlabel("embedding dimension (i)")
        ax.set_ylabel("position (pos)")
        ax.set_title("Sinusoidal positional encoding (§3.5 of the paper)")
        fig.colorbar(im, ax=ax)
        out = os.path.join(os.path.dirname(__file__), "positional_encoding.png")
        plt.tight_layout()
        plt.savefig(out, dpi=120)
        print(f"\nSaved heatmap to {out}")
    except Exception as exc:
        print(f"(matplotlib unavailable: {exc} — skipping plot)")

    # Show what happens when we *add* PE to an embedding (§3.5 sum).
    fake_embeddings = torch.zeros(1, 10, d_model)       # (batch, seq, d_model)
    out = pe(fake_embeddings)
    print(f"\nAfter adding PE to a zero 'embedding' of shape (1, 10, {d_model}):")
    print(f"   output shape = {tuple(out.shape)}")
    print("   This is exactly the input that enters the first encoder layer.")


if __name__ == "__main__":
    main()
