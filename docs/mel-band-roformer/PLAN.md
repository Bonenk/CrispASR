# §248 — Mel-Band RoFormer (source separation) port

**Claimed:** M1/Metal session, 2026-07-19. New category: `--task separate`.

## NOW — active work

- [x] License due diligence (below) — **cleared, with a caveat on which code to read**
- [x] HARD RULE #1: Python blueprint read line-by-line (below)
- [ ] `tools/reference_backends/mel_band_roformer.py` per-stage dumper
- [ ] `models/convert-mel-band-roformer-to-gguf.py`
- [ ] `src/mel_band_roformer.{h,cpp}` + per-stage diff
- [ ] Roundtrip acceptance (SDR / ASR on the vocal stem) — the ONLY gate that counts

## Licensing (verified 2026-07-19, do not re-derive)

| Artifact | License | Source of truth |
|---|---|---|
| Architecture code | **MIT** | `lucidrains/BS-RoFormer` LICENSE ("MIT License, Copyright (c) 2023 Phil Wang") |
| Weights (Kim vocals) | **MIT** | `KimberleyJSN/melbandroformer` HF card `license: mit` |
| Kim's inference repo | **NONE DETECTED** | `KimberleyJensen/Mel-Band-Roformer-Vocal-Model` has no LICENSE file |

⚠ **Read the lucidrains (MIT) implementation as the blueprint, NOT Kim's
inference repo** — that repo ships no license, so transcribing it would be
unlicensed copying. Kim's contribution we rely on is the *weights* (MIT) and
the *config YAML* (facts/hyperparameters, not creative code). Same shape as the
Transcoda clean-room note in LEARNINGS.

## Target checkpoint + config (Kim vocals)

`config_melband_roformer_vocals_kim.yaml` — note these differ from lucidrains
defaults, which is exactly why the blueprint gets read instead of assumed:

```
sample_rate 44100   chunk_size 352800   num_stems 1  (vocals; other = residual)
dim 384             depth 6             stereo true
time_transformer_depth 1                freq_transformer_depth 1
heads 8             dim_head 64         num_bands 60
stft_n_fft 2048     stft_hop_length 441 (NOT the lucidrains default 512)
dim_freqs_in 1024   mask_estimator_depth 2
attn_dropout 0      ff_dropout 0        flash_attn true
```

## Blueprint — exact forward (lucidrains `mel_band_roformer.py`)

1. `raw_audio (b,s,t)` → pack `(b*s, t)` → `torch.stft(..., return_complex=True)`
   → `view_as_real` → `(b,s,f,t,2)`.
2. **Layout:** `rearrange('b s f t c -> b (f s) t c')` — stereo folded into the
   frequency axis, **frequency-major, channel fastest** ⇒ packed index
   `= f*channels + s`.
3. **Mel band membership (binary, not weighted):**
   `mel = librosa.filters.mel(sr, n_fft, n_mels=num_bands)` → then **two
   hand-tweaks that shift DC/Nyquist membership and are trivial to miss**:
   `mel[0,0] = mel[0,1]*0.25`, `mel[-1,-1] = mel[-1,-2]*0.25`.
   Then `freqs_per_band = mel > 0` — the filterbank is used **only** to decide
   *which* bins belong to a band; the weights are discarded ("binary as in
   paper, then masks are averaged over overlapping regions").
4. `freq_indices` = per-band bin indices concatenated (a gather index). Stereo:
   `freq_indices = freq_indices*2 + arange(2)`, flattened — matches (2).
5. Gather `x = stft_repr[batch_arange, freq_indices]` → `'b f t c -> b t (f c)'`
   (complex folded into the freq axis). Per-band input width
   `= 2 * num_freqs_in_band * audio_channels` (bands have **different** widths).
6. `BandSplit`: per band `RMSNorm(dim_in) → Linear(dim_in, dim)`; stack →
   `(b, t, n_bands, dim)`.
7. Per layer (`depth` times): optional `linear_transformer` → **time**
   transformer (over `t`) → **freq** transformer (over `f`), RoPE inside.
   ⚠ **Value residuals**: the first layer's attention values are carried
   forward as `value_residual` into every later layer, separately for the
   linear/time/freq streams. Silently dropping this changes the output.
8. `MaskEstimator` per stem: per band
   `MLP(dim → dim_in*2, hidden=dim*4, depth=mask_estimator_depth, act=Tanh)`
   → `GLU(-1)` (halves back to `dim_in`) → concat over bands.
   ⚠ activation is **Tanh**, not GELU/SiLU.
9. Masks → `'b n t (f c) -> b n f t c'` → `view_as_complex`. Same for stft.
10. **Overlap handling:** `scatter_add_` masks into zeros at `freq_indices`
    (dim=2), then divide by `num_bands_per_freq.clamp(min=1e-8)` — i.e.
    overlapping bands are **averaged, not summed**.
11. `stft_repr = stft_repr * masks_averaged` — a **true complex multiply**, not
    magnitude masking.
12. Optional `zero_dc`: `index_fill(freq=0, 0.)`.
13. `torch.istft(..., length=raw_audio_length)` → `'(b n s) t -> b n s t'`.

## Port gotchas (carry into the C++ port)

- Band membership depends on librosa's mel edges **and** the two hand-tweaks —
  an off-by-one there shifts `freq_indices` and silently corrupts every band.
  Diff `freq_indices` / `num_bands_per_freq` as stage 0 before any tensor math.
- hop 441 (not 512) — assert the frame count against the reference.
- Averaging denominator is per-frequency band-count, not per-band.
- Complex multiply, Tanh MLP, GLU halving, value residuals — each is a silent
  divergence if missed.

## Reuse (DRY — do not write new DSP)

`src/core/fft.h` (STFT), `src/core/istft.h` (iSTFT), `core_attn` (attention),
`ffn.h`, RMSNorm. Only the band gather/scatter + per-band linears are new.

## Coordination

HTDemucs (other session, converter `a6a447587` + dumper `60ada0a06`) is the
same new category and shares the `--task separate` surface (CLI flag, stem WAV
output, capability bit). **Whoever lands that scaffolding first owns it; the
other builds on it.** Per-backend files are additive and conflict-free.
