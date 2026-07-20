# RVC voice conversion — port blueprint (§CB1)

Traced from `RVC-Project/Retrieval-based-Voice-Conversion-WebUI`
(`infer/module/models.py`, `infer/vc/pipeline.py`), shallow clone 2026-07-20.
Every claim below carries a file:line citation and was read from source, not
recalled. The wire contract lives in `SVC_RECORD_SHAPES.md` (CONFIRMED); this
document is about what we have to BUILD.

**Read this before writing any C++.** Two findings change the shape of the job.

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
covered by "HiFi-GAN". **This is worth confirming with CometBeat**: if they
expected us to own only the vocoder and to run enc_p/flow in Dart, the split is
different from what the note implies (and their side would then need to send
`z`, not ContentVec features).

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

**Site B — the sine source** (`models.py`, SineGen.forward ~325 and ~358):

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
