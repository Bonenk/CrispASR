# CrispASR v0.8.23

A fix-up release. Its main job is to **restore the Windows CPU binary** that
v0.8.22 shipped without — the mel-band-roformer speedup in v0.8.22 used a
construct MSVC rejects, so the Windows CPU release build failed and that asset
was missing. This release carries the fix, plus a VibeVoice voice-pack
correctness fix, native speaker labels in streaming, and a clearer submodule
build error. Drop-in from v0.8.22 — no behavior changes to existing flags.

## Fixed — Windows CPU release build (#296)

The v0.8.22 mel-band-roformer optimization used `__attribute__((weak))`, which
GCC/Clang accept but **MSVC does not** — so the Windows CPU release build broke
and v0.8.22 went out without its Windows CPU binary. Removed the weak-symbol
construct; the Windows CPU build links OpenBLAS and ships again with the fast
`--separate` path intact (macOS/Linux were unaffected and are unchanged).

## Fixed — VibeVoice base models + voice packs (#299)

Passing a realtime voice pack to a **VibeVoice base model** (1.5B / 7B) produced
noise. The realtime voice packs are KV-only artifacts baked for the
realtime-0.5B head (head_dim 64) and are structurally incompatible with the base
models (head_dim 128, no `tts_lm`, different conditioning mechanism). The base
models now **error clearly** instead of emitting garbage, and clone a voice from
a reference WAV via `--voice ref.wav` (the mechanism they actually support).

## New — structured speaker labels while streaming (#300)

Backends that produce a **structured** per-segment speaker label — `moss-diarize`
(MOSS-Transcribe-Diarize-0.9B), and `granite` in speaker-aware `--diarize` mode —
had that label dropped by the streaming emit path, which built text from
`seg.text` only. Now `--stream` / `--mic` / `--live` surface it:

- **Plain `--stream`** prefixes the label inline (like file-mode `text`/`srt`/`vtt`).
- **`--stream-json`** adds an optional `"speaker"` field to `final` events when
  the utterance is single-speaker; `text` stays clean.

Labels are window/utterance-local (no cross-utterance clustering runs live);
recording-stable labels via `--diarize-speakers` and named `--speaker-db`
identification remain recorded-file features by design. Documented in
[`docs/streaming.md`](docs/streaming.md).

Note: `vibevoice` (VibeVoice-ASR) also asked about in #300 emits speaker info
**inline in its transcript text**, not as a structured field — that already flows
through streaming as text, so this change is a no-op for it.

```bash
ffmpeg -i meeting.wav -f s16le -ar 16000 -ac 1 - 2>/dev/null \
  | crispasr --stream -m auto --backend moss-diarize
# (Speaker 1) welcome everyone
# (Speaker 2) thanks, glad to be here
```

## Fixed — clearer build error for an uninitialized ggml submodule (#298)

A fresh clone without `--recursive` (or a worktree missing
`git submodule update --init --recursive`) failed deep in CMake with a confusing
missing-`ggml/CMakeLists.txt` error. The build now detects the uninitialized
submodule up front and prints the exact command to fix it, and defines
`GNUInstallDirs` before adding `src`.

## Docs

- README backend counts corrected to match the binary and the repo's own
  `tts`-cap convention: **104 compiled backends, 53 ASR, 51 TTS, 27 caps** (the
  prose had drifted to 91/43/48/21 while the generated
  [`docs/feature-matrix.md`](docs/feature-matrix.md) was already current).

## PyPI

The `crispasr` Python wrapper was already published at 0.8.23; this binary
release realigns the tagged binaries with it.

**Full changelog:** https://github.com/CrispStrobe/CrispASR/compare/v0.8.22...v0.8.23
