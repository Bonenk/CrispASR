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

## Fix — chosen direction: "2 then 1" + evaluate a permissive SOTA tool

### DONE — approach 2 (band-limit + lower alpha)  [commit on this branch]
- `wm_params()` in `crispasr_watermark.h`: comb `hi_bin` `n_fft/2-1` → `n_fft/5`
  (~4.8 kHz), default `alpha` `0.08` → `0.05`. Both embed + detect read
  `wm_params()` so they agree. `CRISPASR_WATERMARK_LEGACY=1` restores the old
  wideband/loud path (A/B + re-detect old marks). `embed_impl` alpha default
  `-1`=auto; `alpha==0` stays a true no-op.
- Real qwen3 clip: **above-5 kHz SNR 17.7 → 51.7 dB** (tinny region gone),
  detection 0.94 → 0.81 (>0.65 threshold), clean 0.44. All watermark unit
  tests pass (+ new speech-like #260 guard).

### NEXT — approach 1 (psychoacoustic masking), then neural upgrade
- Shape the nudge under a masking threshold; only modulate bins with a local
  masker; skip empty bins (needs the unit-test signal to be broadband, not a
  pure tone — sine watermarking relies on empty-bin injection today).
- **Permissive SOTA survey (agent):** **AudioSeal (Meta) is the answer** — MIT
  *code AND weights* since v0.1.2 (relicensed from CC-BY-NC), SOTA robustness,
  small SEANet convnet that ports to ggml; we already have `src/audioseal.cpp`
  partially wired → finish it as the optional `--watermark-model` default.
  Reject **audiowmark** + **Timbre WM** (GPLv3, copyleft — clean-room the DSP
  ideas only). WavMark/SilentCipher/Perth: code MIT but weights license
  unclear — verify before shipping. Full survey in agent output.

### Not done (optional, low-risk)
- `--no-watermark` CLI flag (today only `CRISPASR_NO_WATERMARK=1` env). Trivial
  additive; hold pending maintainer nod.

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
