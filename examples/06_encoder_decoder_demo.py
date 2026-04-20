"""
examples/06_encoder_decoder_demo.py
===================================

Forward-pass a random batch through the encoder and decoder stacks
(§3.1) independently, printing every shape so a student can verify
that the dimensions line up with Fig. 1 of the paper.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.encoder import Encoder
from src.decoder import Decoder
from src.masking import subsequent_mask


def main():
    torch.manual_seed(0)

    d_model, num_heads, d_ff = 32, 4, 64
    num_layers = 2       # keep it tiny for the demo
    batch, src_len, tgt_len = 3, 7, 5

    encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout=0.0)
    decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout=0.0)

    # Fake "embedding + PE" inputs
    src_in = torch.randn(batch, src_len, d_model)
    tgt_in = torch.randn(batch, tgt_len, d_model)

    # Masks
    src_mask = torch.ones(batch, 1, 1, src_len, dtype=torch.bool)
    tgt_mask = subsequent_mask(tgt_len).expand(batch, 1, tgt_len, tgt_len)

    memory = encoder(src_in, mask=src_mask)
    dec_out = decoder(tgt_in, memory, src_mask=src_mask, tgt_mask=tgt_mask)

    print(f"src_in        : {tuple(src_in.shape)}")
    print(f"memory (enc)  : {tuple(memory.shape)}    (= src_in shape, §3.1)")
    print(f"tgt_in        : {tuple(tgt_in.shape)}")
    print(f"dec_out       : {tuple(dec_out.shape)}    (= tgt_in shape, §3.1)")

    # Sanity: every encoder / decoder layer preserves d_model (§3.1).
    assert memory.shape == src_in.shape
    assert dec_out.shape == tgt_in.shape
    print("\nShapes match — §3.1: 'all sub-layers in the model produce outputs of dimension d_model = 512'.")


if __name__ == "__main__":
    main()
