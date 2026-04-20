"""
examples/10_toy_copy_training.py
================================

Train a tiny Transformer to *copy* its input — the classic sanity-check
task that uses every component of the paper (embeddings, PE, both
stacks, all three attention call-sites of §3.2.3, the FFN, label
smoothing, and the Noam schedule) in a pipeline that runs on a laptop
CPU in a minute or two.

Data
----
Sequences of length 10 drawn from a vocabulary of 11 tokens:

    0 : <pad>
    1 : <bos>
    2 : <eos>
    3..10 : "real" tokens

The target is the source shifted right so that the decoder has to
predict the next source token at every step — i.e. *copy*.

What to look for
----------------
* Loss should fall from ~2.3 to well below 0.5 within a few hundred
  steps.
* After training, greedy decoding should perfectly reproduce any
  test sequence you give it — confirming that §3.2.3's
  encoder-decoder attention is correctly wired and that the
  look-ahead mask of §3.1 prevents cheating.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.config import ModelConfig
from src.transformer import Transformer
from src.label_smoothing import LabelSmoothingLoss
from src.optimizer import make_noam_optimizer


VOCAB = 11
PAD, BOS, EOS = 0, 1, 2
SEQ_LEN = 10


def random_batch(batch_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Random "copy this sequence" batch.

    Returns
    -------
    src : (batch, SEQ_LEN)                           — source sequence
    tgt : (batch, SEQ_LEN)                           — shifted target with BOS prefix
    """
    # Real content tokens are in [3, VOCAB).
    body = torch.randint(3, VOCAB, (batch_size, SEQ_LEN - 1))
    src  = torch.cat([body, torch.full((batch_size, 1), EOS)], dim=1)
    # The decoder target is BOS followed by the source — "teacher forcing".
    tgt  = torch.cat([torch.full((batch_size, 1), BOS), body], dim=1)
    return src, tgt


def main():
    torch.manual_seed(0)

    cfg = ModelConfig(
        src_vocab_size=VOCAB,
        tgt_vocab_size=VOCAB,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        max_seq_len=32,
        dropout=0.1,
        label_smoothing=0.1,
        warmup_steps=400,        # short warmup for the tiny problem
        padding_idx=PAD,
    )
    model = Transformer(cfg)
    loss_fn = LabelSmoothingLoss(cfg.tgt_vocab_size, padding_idx=PAD,
                                 smoothing=cfg.label_smoothing)
    optimizer, scheduler = make_noam_optimizer(
        model, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps)

    model.train()
    NUM_STEPS = 600
    for step in range(1, NUM_STEPS + 1):
        src, tgt = random_batch(batch_size=32)
        # The standard teacher-forced target: feed tgt[:, :-1] to the
        # decoder and compare its output to tgt[:, 1:].
        dec_in  = tgt[:, :-1]
        dec_out = tgt[:, 1:]
        logits = model(src, dec_in)
        loss = loss_fn(logits, dec_out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 50 == 0 or step == 1:
            lr = scheduler.get_last_lr()[0]
            print(f"step {step:4d}   loss = {loss.item():.4f}   lr = {lr:.2e}")

    # --- inference ---------------------------------------------------------
    print("\n--- Greedy-decoding a held-out example ---")
    src, _ = random_batch(batch_size=1)
    out = model.greedy_decode(src, max_len=SEQ_LEN + 1,
                              start_symbol=BOS, end_symbol=EOS)
    print("src:", src[0].tolist())
    print("out:", out[0].tolist())
    match = src[0, :-1].tolist() == out[0, 1:SEQ_LEN].tolist()
    print("Exact copy:", match)


if __name__ == "__main__":
    main()
