# Issue #260 — qwen3-tts "tinny noise" / characteristic spectrogram pattern

## NOW — active work

- **Root cause FOUND & reproduced (model-free): the built-in spread-spectrum
  watermark, not the qwen3-tts port.** Every TTS output from the CLI is
  unconditionally watermarked (`crispasr_run.cpp:2374`,
  `crispasr_wm_dispatch::embed`). With no AudioSeal `--watermark-model`, this
  falls back to `crispasr_watermark_embed_impl` (`examples/cli/crispasr_watermark.h`).
- **The audible regression**: commit `8b81c0fc0` (2026-06-07) raised the default
  `alpha` from `0.005` → `0.08` (16×) "for robust detection." At 0.08 the
  watermark paints a **fixed comb of 32 key-derived frequency bins** onto every
  1024-sample frame. ~20 of those 32 bins sit between **4.4 kHz and 11.8 kHz**,
  where clean TTS speech has almost no energy — so the comb stands out as tinny
  high-frequency noise and shows as steady horizontal lines on a spectrogram
  (exactly the reporter's description).
- **Why aggregate SNR hid it**: the commit claimed "~38 dB SNR, imperceptible."
  Measured on a clean speech-like signal: aggregate 41 dB, **in-band (<4 kHz)
  47.5 dB — but 4–11.9 kHz band only 3.3 dB SNR** (noise ≈ as loud as signal
  there). The energy is dumped where the ear hears it against near-silence.
  The code deliberately injects into empty bins ("For bins with no energy,
  inject at 0-phase so detection can still see the magnitude") — that line is
  what makes it audible.
- **Backend-agnostic**: the watermark is applied at a generic call site for ALL
  TTS backends, so kokoro/f5/etc. get the identical comb. It is NOT a qwen3
  bug; qwen3's clean HF-matched output just makes the comb obvious vs the HF
  reference the reporter A/B'd against.
- **Instant mitigation for the reporter**: set env `CRISPASR_NO_WATERMARK=1`
  (only escape today — there is no `--no-watermark` CLI flag).

### Reproduction artifacts (scratchpad)
- `wm_probe.cpp` — feeds a clean speech-like signal through the exact header;
  prints the 32 fixed comb frequencies + band SNRs.
- `spectrogram_compare.png` — clean vs alpha=0.08: comb lines at 4.4–11.8 kHz.
- Real-model A/B (in progress): qwen3-tts-1.7b-customvoice-q8_0, `--seed 42`,
  `CRISPASR_NO_WATERMARK=1` vs default → diff = watermark only.

## Fix options (product decision — watermark/provenance policy)
1. **Perceptual masking**: only nudge bins already carrying energy; never inject
   into empty bins; cap per-bin boost to a fraction of the local masking
   threshold. Keeps provenance, kills the audible comb.
2. **Restrict comb to the speech band** (e.g. lo..~hi where energy exists) and
   lower alpha; trade some detection robustness for inaudibility.
3. **Add a proper `--no-watermark` CLI flag** (today only the env var exists).
4. **Make watermark opt-in** for TTS (default off), keep AudioSeal/C2PA path.

Likely: (1)+(3) — inaudible-but-detectable watermark + a first-class opt-out.

## Measured (both confirm the comb)
- Synthetic clean speech-like signal: broadband 41 dB, in-band(<4k) 47.5 dB,
  **above-band(4–11.9k) 3.3 dB** — comb clearly visible on spectrogram.
- REAL qwen3-tts-1.7b-customvoice-q8_0 clip ("The quick brown fox…", seed 42,
  2.48 s): broadband **38.8 dB** (matches the commit's claim), in-band 44.9 dB,
  **above-band(4–11.9k) 17.5 dB** — comb visible in the silence gaps where the
  clean clip is black. This is the tinny tone audible during pauses.

## Status
- [x] Root cause identified (built-in spread-spectrum watermark, alpha bump 8b81c0fc0)
- [x] Model-free reproduction + spectrogram (scratchpad wm_probe.cpp)
- [x] Real qwen3-tts A/B clip + spectrogram (single model run, watermark applied offline)
- [x] Confirmed backend-agnostic (generic call site crispasr_run.cpp:2374; key-derived comb)
- [ ] Fix direction confirmed with maintainer (product/provenance policy)
- [ ] Implement + A/B: detection confidence retained AND comb inaudible
