# Paper → Code Mapping

This document quotes the relevant sentences of Vaswani et al. (2017),
*"Attention Is All You Need"* (arXiv:1706.03762) and points to the exact
file / symbol in this repository that implements them.  Read it side-by-side
with the paper.

---

## Abstract — "dispensing with recurrence and convolutions entirely"

The main claim of the paper is that the Transformer uses **only**
attention (no RNN, no CNN).  Our entire `src/` tree contains no
recurrent or convolutional layers — the only stateful operation is the
LayerNorm inside the residual wrappers.

---

## §3 "Model Architecture"

> *"Most competitive neural sequence transduction models have an
> encoder-decoder structure … The Transformer follows this overall
> architecture using stacked self-attention and point-wise, fully
> connected layers for both the encoder and decoder, shown in the left
> and right halves of Figure 1, respectively."*

Implemented by `src/transformer.py::Transformer`, which wires
`src.encoder.Encoder` (left half of Fig. 1) and `src.decoder.Decoder`
(right half) through the embedding / positional-encoding layers.

---

## §3.1 "Encoder and Decoder Stacks"

> *"Encoder: The encoder is composed of a stack of N = 6 identical
> layers.  Each layer has two sub-layers.  The first is a multi-head
> self-attention mechanism, and the second is a simple, position-wise
> fully connected feed-forward network."*

Implemented by `src/encoder.py::EncoderLayer` (single layer) and
`src/encoder.py::Encoder` (stack of `N`).

> *"We employ a residual connection around each of the two sub-layers,
> followed by layer normalization.  That is, the output of each
> sub-layer is LayerNorm(x + Sublayer(x))."*

Implemented by `src/encoder.py::SublayerConnection`, reused from the
decoder.

> *"Decoder: The decoder is also composed of a stack of N = 6 identical
> layers.  In addition to the two sub-layers in each encoder layer, the
> decoder inserts a third sub-layer, which performs multi-head attention
> over the output of the encoder stack."*

Implemented by `src/decoder.py::DecoderLayer` (three sub-layers) and
`src/decoder.py::Decoder` (stack).

> *"We also modify the self-attention sub-layer in the decoder stack to
> prevent positions from attending to subsequent positions."*

Implemented by `src/masking.py::subsequent_mask` and used inside
`Transformer.forward → make_tgt_mask` in `src/masking.py`.

---

## §3.2.1 "Scaled Dot-Product Attention", Eq. (1)

> *"Attention(Q, K, V) = softmax(QKᵀ / √d_k) V"*

Implemented by `src/scaled_dot_product_attention.py::scaled_dot_product_attention`.

> *"We scale the dot products by 1/√d_k."*

The `scores = scores / math.sqrt(d_k)` line in the same function.

> *"…masking out (setting to −∞) all values in the input of the softmax
> which correspond to illegal connections."*  (§3.2.3 but implemented in
> §3.2.1)

`scores.masked_fill(~mask, float("-inf"))` inside
`scaled_dot_product_attention`.

---

## §3.2.2 "Multi-Head Attention"

> *"MultiHead(Q, K, V) = Concat(head_1, …, head_h) W^O
> where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)."*

Implemented by `src/multi_head_attention.py::MultiHeadAttention`.
The packed `W_Q / W_K / W_V` linear layers correspond to stacking all
`h` projection matrices together.

> *"We employ h = 8 parallel attention layers, or heads.  For each of
> these we use d_k = d_v = d_model / h = 64."*

Captured by `ModelConfig.num_heads = 8` and the computed
properties `ModelConfig.d_k`, `ModelConfig.d_v`.

---

## §3.2.3 "Applications of Attention in our Model"

Three call-sites, all routed through the same `MultiHeadAttention`:

1. **Encoder self-attention** — `EncoderLayer.forward` passes `Q = K = V = x`.
2. **Decoder self-attention** (masked) — `DecoderLayer.forward` passes
   `Q = K = V = x` with `tgt_mask`.
3. **Encoder–decoder attention** — `DecoderLayer.forward` passes
   `Q = x`, `K = V = memory`.

---

## §3.3 "Position-wise Feed-Forward Networks", Eq. (2)

> *"FFN(x) = max(0, x W_1 + b_1) W_2 + b_2"*

Implemented by `src/feed_forward.py::PositionwiseFeedForward`.

> *"The dimensionality of input and output is d_model = 512, and the
> inner-layer has dimensionality d_ff = 2048."*

`ModelConfig.d_ff = 2048` (base) / `4096` (big).

---

## §3.4 "Embeddings and Softmax"

> *"We use learned embeddings to convert the input tokens and output
> tokens to vectors of dimension d_model … we share the same weight
> matrix between the two embedding layers and the pre-softmax linear
> transformation … In the embedding layers, we multiply those weights
> by √d_model."*

Implemented by `src/embeddings.py::TokenEmbedding` (with `* √d_model`)
and `src/embeddings.py::OutputProjection(shared_embedding=…)` (tied
weights).  The Transformer constructor wires them together when
`cfg.share_embeddings` is True.

---

## §3.5 "Positional Encoding"

> *"PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
>  PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))"*

Implemented by `src/positional_encoding.py::PositionalEncoding`.  The
matrix is computed once, registered as a buffer (not a parameter), and
added to the embeddings in `forward()`.

---

## §4 "Why Self-Attention"

This section argues for self-attention on three axes (complexity per
layer, sequential operations, path length).  No code change — our
implementation *is* the answer.  For per-head attention matrices that
you can inspect to confirm different heads attend to different
subspaces (as §4 illustrates with Figures 3-5), use the `attn` tensor
returned by `MultiHeadAttention.forward`.

---

## §5.3 "Optimizer", Eq. (3)

> *"lrate = d_model^{-0.5} · min(step_num^{-0.5},
>                                  step_num · warmup_steps^{-1.5})"*

Implemented by `src/optimizer.py::NoamScheduler`.

> *"Adam with β_1 = 0.9, β_2 = 0.98 and ε = 10^{-9}."*

`src/optimizer.py::make_noam_optimizer`.

> *"warmup_steps = 4000."*

`ModelConfig.warmup_steps = 4000`.

---

## §5.4 "Regularization"

> *"We apply dropout to the output of each sub-layer, before it is added
> to the sub-layer input and normalized.  In addition, we apply dropout
> to the sums of the embeddings and the positional encodings in both the
> encoder and decoder stacks.  For the base model, we use a rate of
> P_drop = 0.1."*

* Residual dropout — `SublayerConnection.dropout` in `src/encoder.py`.
* Embedding + PE dropout — `PositionalEncoding.dropout`.
* Value: `ModelConfig.dropout = 0.1`.

> *"During training, we employed label smoothing of value ε_ls = 0.1."*

Implemented by `src/label_smoothing.py::LabelSmoothingLoss`.

---

## Table 3 "Variations on the Transformer architecture"

* "base" row  → `src.config.BASE_CONFIG`.
* "big"  row  → `src.config.BIG_CONFIG` (d_model = 1024, d_ff = 4096,
  h = 16, P_drop = 0.3 for EN-DE).

---

## What is NOT implemented (scope of this pedagogical project)

* **Beam search** (§6.1 talks about beam = 4, α = 0.6).  We provide
  `Transformer.greedy_decode` instead, which is enough for the toy
  copy task.
* **Byte-pair / word-piece tokenisation** (§5.1).  The toy example uses
  raw integer ids; plugging in a real tokenizer is a one-line change.
* **Checkpoint averaging** (§5.2) and model parallelism (§5.2).  Out of
  scope for a single-file training run.

Every other aspect — every equation, every table entry, every
regularisation detail — has a corresponding module in `src/` and at
least one example in `examples/`.
