# BTC chord recognition — blueprint notes

Read from `jayg996/BTC-ISMIR19` (MIT) before writing any C++, per HARD RULE #1.
Five details here would each have been a silent parity bug if assumed from
"it's a transformer".

## Hyperparameters (`run_config.yaml`)

| | |
|---|---|
| feature (embedding) size | **144** — matches `core/cqt.h` n_bins |
| hidden size | **128** |
| layers | **8** |
| attention heads | **4** |
| FFN filter size | **128** |
| sequence length (timestep) | **108** |
| chord classes | **25** (`btc_model.pt`) / **170** (`btc_model_large_voca.pt`) |
| dropout (input/layer/attn/relu) | 0.2 — inference-time no-op |

Front end: CQT `n_bins=144`, `bins_per_octave=24`, `hop_length=2048`,
`sr=22050`. This is exactly what `core/cqt.h` was built and librosa-validated
for, so the front end is already done.

## The five details that are NOT the obvious default

1. **It is NOT full bidirectional attention.** (Confirmed at `btc_model.py:73`:
   the backward block is constructed with
   `torch.transpose(_gen_bias_mask(max_length), dim0=2, dim1=3)`, and at line 89
   it is fed the SAME `x` — bidirectionality comes from the transposed mask, not
   from reversing the sequence.) Each of the 8 layers runs TWO
   self-attention blocks: a forward one masked with
   `np.triu(np.full([L,L], -inf), 1)` (upper triangle above the diagonal is
   -inf, i.e. **causal — attend to past + self**) and a backward one using the
   **transposed** mask (attend to future + self). Their outputs are
   **concatenated and projected back to hidden_size**. Passing `mask=nullptr`
   would silently give full attention and be wrong — the exact trap the dev
   guide calls out for `soft_max_ext`.

2. **Positional encoding is CONCATENATED halves, not interleaved.**
   `signal = concat([sin(scaled_time), cos(scaled_time)], axis=1)` — so the
   first `hidden/2` channels are all sin and the second half all cos. (htdemucs'
   CrossTransformer interleaves; do not copy that layout.)
   `inv_timescales = min_timescale * exp(arange(n) * -log_timescale_increment)`,
   `log_timescale_increment = log(max_timescale/min_timescale) / (n - 1)`.

3. **The FFN is CONVOLUTIONAL with kernel 3, not a 1x1 linear.**
   `PositionwiseFeedForward(layer_config='cc')` = Conv(k=3) -> ReLU -> Conv(k=3).
   A linear FFN would be the wrong op AND the wrong receptive field.

   **Padding is SYMMETRIC, not causal.** `transformer_modules.py` documents a
   `pad_type='left'` option — `padding = (k-1, 0)` — but `btc_model.py:14`
   passes **`padding='both'`**, giving `(k//2, (k-1)//2)` = **(1, 1)** for k=3.
   Reading only the module docstring gives causal padding and a one-frame
   output shift. Read the CALL SITE, not the option list.

4. **Attention scaling is applied to Q, not to the scores**, as
   `queries *= (total_key_depth // num_heads) ** -0.5`. Numerically equivalent
   to scaling the logits, but worth matching for exact parity.

5. **LayerNorm eps = 1e-6**, not the 1e-5 used elsewhere in this repo.

Ordering is pre-norm: LayerNorm -> attention -> dropout -> residual ->
LayerNorm -> FFN -> dropout -> residual.

## Three MORE details that only the CHECKPOINT reveals

The architecture docs do not mention any of these. Inspecting
`test/btc_model.pt` directly was necessary:

6. **Scalar input normalisation ships WITH the checkpoint.** The `.pt` has
   top-level `mean = -2.2279364` and `std = 1.7191066` alongside `model`. The
   log-CQT features are normalised `(x - mean) / std` before the embedding
   projection. Nothing in the README or config says so.

7. **The output layer contains a BIDIRECTIONAL LSTM.** The docs call it merely
   a "SoftmaxOutputLayer classifying over num_chords classes", but the tensors
   are `output_layer.lstm.weight_ih_l0 (256,128)`, `weight_hh_l0 (256,64)` plus
   `_reverse` twins, then `output_layer.output_projection (25|170, 128)`. So
   hidden is 64 per direction (256 = 4 gates x 64), concatenated to 128 before
   the classifier. `core/lstm.h` already has bidirectional LSTM.

8. **Attention linears are bias-free** — q/k/v/output each have `.weight` only.

### Actual end-to-end shape

    CQT(144) -> (x - mean)/std -> embedding_proj(144->128) -> + posenc
      -> 8 x [ fwd attn block || bwd attn block -> concat(256) -> linear(256->128) ]
      -> biLSTM(128 -> 64x2) -> output_projection(128 -> 25 or 170)

221 tensors: 8 layers x 26, plus embedding_proj, the 8 LSTM tensors and the
2 projection tensors.

## Licence

Code MIT; the shipped `test/btc_model{,_large_voca}.pt` checkpoints were trained
on Isophonics / Robbie Williams / UsPop2002 annotations, which are
**CC BY-NC-SA**. Ships behind `--accept-license cc-by-nc-sa-4.0` (the gate
landed in 90d1a0c9e). See the licence scoping in `PLAN.md`.
