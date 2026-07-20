# RVC voice conversion — port blueprint (§CB1)

Traced from `RVC-Project/Retrieval-based-Voice-Conversion-WebUI`
(`infer/module/models.py`, `infer/vc/pipeline.py`), shallow clone 2026-07-20.
Every claim below carries a file:line citation and was read from source, not
recalled. The wire contract lives in `SVC_RECORD_SHAPES.md` (CONFIRMED); this
document is about what we have to BUILD.

**Read this before writing any C++.** Two findings change the shape of the job.

---

## 0. RESOLUTIONS (CometBeat, 2026-07-20)

Both findings below are answered; recorded here so the doc reads as settled.

- **Finding 1 — CrispASR owns all three** (`enc_p` + `flow` + `dec`). The frozen
  contract stands: they send ContentVec features + F0 (Hz, 100 Hz) + speaker id,
  **not `z`**. Their reasoning is performance and it is sound: their pure-Dart
  `rvc.dart` already runs all three at **152x slower than real time**, with the
  transformer `enc_p` and the flow a large share of that — so a vocoder-only
  split would leave the transformer as a Dart bottleneck and still miss
  real-time. The right Dart↔native boundary is the ContentVec output, the one
  checkpoint-agnostic reusable piece; `enc_p`/`flow`/`dec` are all
  checkpoint-specific and belong together. Sending `z` would also leak a 192-ch
  model-internal latent into the wire format.
- **Finding 2 — replay the noise**, agreed. Both live RNG sites must be
  injectable in our ggml graph. **They have offered a cross-impl oracle**:
  `rvc.dart` runs the full generator deterministically from an injected noise
  buffer (`rvcSeededNoise` → the ONNX graph's `rnd [1,192,T]` input, cos
  0.99994), enabling a three-way deterministic harness — Python reference → their
  Dart-offline → our ggml graph, all fed identical noise. Production
  `convert()` stays random by design.
- Their offline impl already pins `F.interpolate` to 2x-**nearest**
  (`rvcAlignFeatures`) and the coarse-pitch mel map 50–1100 Hz
  (`rvcCoarsePitch`) — independent agreement with §3 and SVC_RECORD_SHAPES §4.

### Answer to their SineGen phase question — it is ZERO, by construction

They asked whether to zero or replay the SineGen initial phase, having noticed
their ONNX export appears to fix it. **There was never any randomness there to
fold away, for this architecture:**

`GeneratorNSF` hardcodes `SourceModuleHnNSF(sampling_rate=sr, harmonic_num=0)`
(`models.py:439-441`). With `harmonic_num = 0`:

- `SineGen.dim = harmonic_num + 1 = 1` (`models.py:294`) — fundamental only, no
  overtones.
- `rand_ini` is therefore shape `(batch, 1)` (`:325`), and the very next line is
  `rand_ini[:, 0] = 0` (`:328`).

So `rand_ini` is **identically zero**. The random-phase code is live only for
`harmonic_num > 0`, which RVC never uses. Both implementations should use zero
phase, and they will match bit-for-bit without any agreement being needed —
this is determined by the source, not a convention.

**That leaves exactly TWO live RNG sites, not three:** the `z_p` latent sample,
and SineGen's ADDITIVE noise (`:358`), which is ungated and always runs:

```python
noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
noise     = noise_amp * torch.randn_like(sine_waves)
sine_waves = sine_waves * uv + noise
```

Note it is voicing-dependent: voiced frames get sine + noise at `noise_std`
(0.003), unvoiced get pure noise at `sine_amp/3` (0.0333). Their ONNX export
exposes only `rnd` (the z_p noise), so **this second site is the one their path
may differ on** — worth confirming whether their export also folded the
additive noise away, because that IS a real randomness source, unlike the phase.

---

## 1. FINDING: the ask is ~3x bigger than "the NSF-HiFi-GAN generator"

CometBeat asked us to port "the RVC NSF-HiFi-GAN generator". But the seam they
need — ContentVec features + F0 + speaker id → converted audio — is
`SynthesizerTrnMs768NSFsid.infer()` (`models.py:664`), and that runs **three**
substantial components:

| # | component | what it is | `models.py` |
|---|---|---|---|
| 1 | `enc_p` — TextEncoder | pitch `nn.Embedding(256, hidden)` + **a real multi-layer transformer** (`attentions.Encoder`) + Conv1d proj | 16-50 |
| 2 | `flow` — ResidualCouplingBlock | **normalizing flow**, 4x (ResidualCouplingLayer + Flip), run in REVERSE | 79-113 |
| 3 | `dec` — GeneratorNSF | the NSF vocoder: SourceModuleHnNSF/SineGen + ConvTranspose1d upsample stack + ResBlocks | 420-547 |

```
phone(768) ─┐
pitch(1..255)┼─> enc_p ─> m_p, logs_p ─> [SAMPLE z_p] ─> flow(reverse) ─> z ─┐
             │                                                                ├─> dec ─> audio
nsff0(Hz) ───┴────────────────────────────────────> SineGen source ──────────┘
sid ─────────> emb_g ─────────────> g (conditions flow AND dec)
```

Only #3 is a vocoder. #1 is a transformer encoder and #2 is a flow — neither is
covered by "HiFi-GAN". **RESOLVED (§0): we own all three.** "Vocoder" was loose
wording; they always meant the whole `infer()`.

Note `emb_pitch = nn.Embedding(256, hidden_channels)` (`models.py:40`) — this is exactly why the coarse 1..255 quantisation must be
bit-right: it indexes an embedding table, so an off-by-one is a different
learned vector, not a small numeric error.

---

## 2. FINDING: inference is STOCHASTIC — there are TWO independent RNG sites

This is the single most important fact for validation, and it must be settled
before any graph is written.

**Site A — the latent sample** (`models.py:684`, and 691 for the non-chunked path):

```python
z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
```

**Site B — the sine source** (`models.py`, SineGen.forward ~325 and ~358).
**CORRECTION, see §0:** the `rand_ini` phase term below is DEAD for RVC —
`harmonic_num=0` makes it a single element that the next line zeroes. Only the
additive `noise` is a live RNG site.

```python
rand_ini = torch.rand(f0_buf.shape[0], f0_buf.shape[2], ...)   # random INITIAL PHASE per harmonic
rand_ini[:, 0] = 0
rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
...
noise = noise_amp * torch.randn_like(sine_waves)                # additive, std 0.003 by default
```

Consequences:

- **Output audio is not reproducible run to run**, even with identical inputs.
- **Waveform correlation against a reference run is an invalid acceptance
  test.** We already have this lesson recorded from melotts
  (`--seed` isn't deterministic); here it is structural, not incidental.
- A per-stage diff CANNOT compare anything downstream of `z_p` unless the
  reference's noise is replayed.

**Required harness design** (decide before building):

1. Reference dumper captures `m_p`, `logs_p`, **and the drawn `z_p`**, plus
   SineGen's `rand_ini` and its noise tensor.
2. The runtime REPLAYS those instead of sampling — exactly the pattern
   `btc_chords_diff` uses for `input_feat` and `mel_band_roformer_diff` uses
   for `input_audio`.
3. Stages up to `m_p`/`logs_p` are deterministic and diff normally; everything
   after is input-aligned on the replayed noise.
4. Ship a documented deterministic mode (replay/zero noise) so the acceptance
   test is reproducible, with the stochastic path as the default at runtime.

Without this the port is unverifiable, and "it sounds fine" becomes the only
check — which is precisely the failure mode that let piano-transcription ship
with every 2-D BatchNorm at the wrong epsilon.

---

## 3. Numerical hazards spotted in the trace

- **Phase accumulation by `cumsum`** (SineGen): phase is accumulated over the
  whole utterance at the OUTPUT rate (post-`upp` upsampling), then reduced mod
  1. In float32 over minutes of audio this drifts — the running sum grows while
  the meaningful part is the fraction. Accumulate in double, or reduce mod 1
  incrementally. Upstream's own comment on that line notes the `%1` placement
  blocks a later optimisation, so they were aware the ordering is load-bearing.
- **`F.interpolate` on the phase** — mode matters and must be read off the call,
  not assumed (the ContentVec 2x upsample in the same repo defaults to
  `nearest`, which surprised this port once already; see SVC_RECORD_SHAPES §2).
- **`upp` = `math.prod(upsample_rates)`** (`models.py:438`) ties the F0 rate to
  the vocoder's total upsampling — it is derived, not configured, so a mismatch
  between our hop and the checkpoint's upsample_rates is silent.
- **`noise_convs` stride** (`models.py:463-465`): `stride_f0 = prod(upsample_rates[i+1:])`
  — a per-stage downsample of the source signal that must match the upsample
  schedule exactly.
- SourceModuleHnNSF defaults: `sine_amp=0.1`, `add_noise_std=0.003`,
  `harmonic_num` from config — all read from the checkpoint's config, never
  assumed.

---

## 3b. Checkpoint structure (measured on the official v2 `f0G40k.pth`)

560 tensors: `enc_p` 113, `dec` 243, `enc_q` 103, `flow` 100, `emb_g` 1.

- **`enc_q` (103) is the PosteriorEncoder — TRAINING ONLY**, never called by
  `infer()`. Dead weight, like BTC's `output_layer.lstm.*`.
- **`dec` and `flow` are weight-normalised**: they store `weight_g`/`weight_v`,
  not `weight` (`dec.ups.0.weight` does not exist). Fuse at convert time,
  `w = g * v / ||v||` with the norm over every dim except 0. Some layers in the
  SAME module (`conv_pre`, `noise_convs`, `conv_post`) carry a plain `.weight`,
  so both forms must be handled. 104 pairs in this checkpoint.
- **`enc_p` uses RELATIVE positional attention**, not absolute PE:
  `emb_rel_k/v` are `(1, 2w+1, d)` with **w = 10**.
- Geometry read off the weights: `emb_phone` (192, 768) → content dim 768,
  hidden 192; `emb_pitch` (256, 192) → the coarse pitch is an EMBEDDING INDEX;
  `emb_g` (109, 256) → 109 speakers, gin 256.

### Upsample rates are NOT recoverable from the checkpoint

The 40k ConvTranspose1d kernels are `16,16,4,4`, while the rates are
`10,10,2,2`. Assuming `kernel == 2*rate` gives `8,8,2,2` — a silently wrong
model. The rates come from the config JSON (note v2 ships only 32k/48k; 40k
lives under `configs/v1/`).

They CAN be verified, though: `noise_convs[i]` has kernel `2*prod(rates[i+1:])`
(1 for the last stage). Checked against the real checkpoint:

| stage | kernel | 2·prod(rates[i+1:]) |
|---|---|---|
| 0 | 80 | 2·40 = 80 |
| 1 | 8 | 2·4 = 8 |
| 2 | 4 | 2·2 = 4 |
| 3 | 1 | 1 |

The converter asserts this, so a mismatched config fails loudly. Independently,
`sr / prod(rates)` is **exactly 100.0** for every shipped config (32k/40k/48k) —
a second, independent confirmation of the 100 Hz wire rate that
SVC_RECORD_SHAPES derives from `pipeline.window = 160`.

## 4. Proposed order of work

Standard order, unchanged by the above:

1. **Converter** (`models/convert-rvc-to-gguf.py`) — emit all three components
   plus config (`upsample_rates`, `harmonic_num`, `sine_amp`, `add_noise_std`,
   `tgt_sr`, `spk_embed_dim`, and the **content dim** for the
   `convert_content_dim()` guard CometBeat asked for).
2. **Executable spec** (`tools/rvc_torch_parity.py`) — numpy/torch
   reimplementation scored against the real model, with the noise replayed.
   This is where §2's design gets proven before any C++ exists.
3. **ggml graph** — `src/rvc_svc.{h,cpp}`.
4. **Per-stage diff** — `rvc_svc_diff()` + registration in
   `crispasr_diff_main.cpp` (registering it is part of the job, not a
   follow-up: mel-band-roformer shipped with a written-but-unregistered diff).
5. **Surfaces together** — CLI verb + session C ABI (`crispasr_session_convert*`)
   + wasm, and the arch in **both** detect paths. See
   `docs/contributing.md` section 7.

**Blocking question for CometBeat before step 1:** §1 — do they want all three
components, or only `dec`? That changes the wire format (features vs `z`) and
therefore the contract we just froze.

## 5. Licensing

RVC's code is MIT. Circulating checkpoints are NOT uniformly so — scope each
one before any registry entry, and give non-commercial ones their own licence
tag rather than reusing `cc-by-nc-sa-4.0` (the gate matches on the tag).
Beatrice v2 (§CB2) is custom/non-commercial and needs the same treatment.
