# CrispASR v0.8.13

Stability + correctness release. Every GitHub Actions workflow is green,
including the nightly TTS→ASR regression suite where three backends
(parakeet-ja, bark-small, pocket-tts) are now **hard gates** after real
port/config bugs were found and fixed. Plus a batch of server/CLI features
(offset/duration windows, VAD segment reuse, opt-in voice cloning) and the
`CRISPASR_` env-var standardization.

## Fixes

- **#261 — server no longer prints a false "cpu-isa: OK" before a SIGILL.**
  The self-check inspected only CrispASR's own TUs (baseline) and missed the
  ISA baked into `libggml-cpu` (BMI2/AVX2/…), so a mismatched host got "OK"
  then crashed. It now reads `libggml-cpu`'s real ISA via `ggml_cpu_has_*()`
  and, on the server, refuses CPU-only `vad`/`diarize` requests with a clear
  HTTP 400 instead of crashing mid-request. All `.devops/*.Dockerfile` now
  spell out the ISA baseline explicitly (the `GGML_NATIVE=OFF` → `INS_ENB=ON`
  trap that silently defaults BMI2/SSE42/AVX ON is documented inline).
  For CPUs older than Haswell (no BMI2), the stock image still SIGILLs at
  startup (`ggml_cpu_init()` itself uses BMI2, hit during model load even for
  pure GPU ASR). New **`main-cuda-portable`** image: same CUDA GPU build with a
  generic SSE2 CPU baseline, so the server starts and runs VAD/diarize on
  pre-Haswell CPUs (GPU carries the ASR; CPU kernels are slower but correct).
  Or build locally with `-DGGML_BMI2=OFF -DGGML_AVX2=OFF …`.
- **bark-small TTS — two real port bugs fixed.** (1) `tokenize_text()`
  prepended `[CLS]`/appended `[SEP]`, but bark tokenizes with
  `add_special_tokens=False` — the extra token shifted the 256-token prompt
  and truncated the sentence. (2) The semantic history took the *first* 256
  prompt tokens right-aligned; the reference takes the *last* 256 left-aligned.
  Verified against the `transformers` blueprint (round-trips at WER 0.00);
  bark is a hard regression gate again.
- **parakeet auto-VAD (#89)** now fires on the backend's own safe window
  (`vad_slice_cap_seconds()`) instead of the unrelated 30 s constant, fixing
  degraded/hallucinated transcription on 12–30 s clips (surfaced as the
  parakeet-ja nightly failure; CER 0.68 → 0.00).
- **pocket-tts regression** promoted from advisory to a real voice-clone hard
  gate (the port was always correct — the red was a no-voice-clone model run
  unconditioned). The harness now strips the mandatory spoken AI-disclosure
  prefix before scoring, a general fix for every voice-clone roundtrip.
- **Build/CI:** missing `<string>` include in `crispasr_vad.h` (broke Linux
  Docker Smoke); Go bindings cgo/path-filter fixes; Windows ctest filter
  excludes placeholder/live tests; Examples WASM Pages deploy auto-enablement.
- **Python:** `SpeakerDB` now exported; `__del__` no longer crashes when
  construction was refused (#266).

## Features

- **#91 — time windows everywhere.** `--offset-t` / `--duration` are honored by
  all backends and by the server's `/v1/audio/transcriptions`
  (`offset_t_ms` / `duration_ms`). `--print-confidence` now works for
  non-whisper backends too.
- **#227 — VAD segment reuse.** Export/import VAD boundaries for cross-backend
  reuse, including server `vad_export` / `vad_import`, so a second pass doesn't
  re-run VAD.
- **#201 — opt-in voice cloning** on the TADA path via the server and the
  session C-ABI (in-memory / on-the-fly).
- **#265 — env-var standardization.** All env vars use the
  `CRISPASR_<BACKEND>_<FEATURE>` convention; old bare names still work as
  **deprecated aliases** (one-time warning; silence with
  `CRISPASR_SUPPRESS_ENV_DEPRECATION`). Reference-voice cache unified across
  backends. See `docs/environment-variables.md`.
- **#266 — cluster-level closed-roster speaker identification** (consent-gated,
  post-only, designed to stay outside EU AI Act Annex III 1(a)).
- Confirmed shipped/verified: Piper TTS (#128), qwen3-asr ChatML language
  prompt (§169).

## Performance

- Fused persistent step graphs and reference-voice caches across the TTS
  perf-sweep: dots-tts, omnivoice, openvoice2, vibevoice (byte-identical
  CPU+Metal where noted).

## Upgrade notes

- No API breaks. Env-var renames are backward-compatible via aliases; migrate
  at your convenience to silence the deprecation warnings.
- Server operators on pre-Haswell CPUs: see the #261 note above.

---
*Full commit log: `git log v0.8.12..v0.8.13`.*
