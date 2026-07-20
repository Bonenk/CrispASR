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

1. **It is NOT full bidirectional attention.** Each of the 8 layers runs TWO
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
   `PositionwiseFeedForward(layer_config='cc')` = Conv(k=3) -> ReLU -> Conv(k=3),
   with **left padding** (`pad_type='left'`), i.e. causal padding. A linear FFN
   would be the wrong op AND the wrong receptive field.

4. **Attention scaling is applied to Q, not to the scores**, as
   `queries *= (total_key_depth // num_heads) ** -0.5`. Numerically equivalent
   to scaling the logits, but worth matching for exact parity.

5. **LayerNorm eps = 1e-6**, not the 1e-5 used elsewhere in this repo.

Ordering is pre-norm: LayerNorm -> attention -> dropout -> residual ->
LayerNorm -> FFN -> dropout -> residual.

## Licence

Code MIT; the shipped `test/btc_model{,_large_voca}.pt` checkpoints were trained
on Isophonics / Robbie Williams / UsPop2002 annotations, which are
**CC BY-NC-SA**. Ships behind `--accept-license cc-by-nc-sa-4.0` (the gate
landed in 90d1a0c9e). See the licence scoping in `PLAN.md`.
