# CrispASR v0.8.23

A fix-up release. Its main job is to **restore the Windows CPU binary** that
v0.8.22 shipped without — the mel-band-roformer speedup in v0.8.22 used a
construct MSVC rejects, so the Windows CPU release build failed and that asset
was missing. This release carries the fix, plus a VibeVoice voice-pack
correctness fix, native speaker labels in streaming, a clearer submodule
build error, **F5-TTS Chinese synthesis**, a **concurrency/scaling guide** with
a new `--server-workers` flag, **broader NVIDIA GPU coverage** (older
Pascal/Volta/P100 cards), and several backend fixes — CosyVoice3 on Vulkan,
whisper-VAD GPU acceleration, and cleaner whisper punctuation. Drop-in from
v0.8.22 — existing flags are unchanged.

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

## New — F5-TTS Chinese synthesis, and English no longer truncates (#294)

F5-TTS now speaks **Chinese**. It converts Han text to pinyin the way the model
expects — a built-in g2p (jieba-min segmentation + `pypinyin` TONE3 syllables with
不/一/third-tone sandhi); previously every Chinese character hit the unknown token
and produced silence. Mixed English+Chinese in one sentence works too. Use a
**Chinese reference clip** with an accurate `--ref-text` for best results (cloning
an English voice onto Chinese sounds rough regardless).

Separately, long **English** sentences no longer lose their tail: the output-length
estimator had a rate clamp with no upstream equivalent that shortened a slow or
expressive reference. It's now loose; `CRISPASR_F5_DURATION_CLAMP=0` restores the
exact upstream formula. Both fixes were audio-confirmed via TTS→ASR roundtrip.

## New — concurrency & scaling guide + `--server-workers` (#301)

New [`docs/concurrency.md`](docs/concurrency.md) documents how CrispASR scales:
intra-request ggml threads, the mutex-serialized server, the in-process worker
pool, replicas behind a load balancer, and CLI bulk fan-out — plus honest answers
on why PagedAttention isn't the fit and why there's no batched multi-stream
inference. The existing `CRISPASR_SERVER_WORKERS` pool is now also a discoverable
`--server-workers N` CLI flag (shown in `--help`; the env var still overrides).

Two crashes fixed while testing it: **moonshine** SIGSEGV'd on the 2nd server
request on a GPU (a cached encoder graph reused across scheduler cycles pointed at
freed GPU memory — now gated to CPU, rebuild-each-call on GPU); and an explicit
`-m <path>` that exists but can't be opened (a dangling symlink or permission
error) now prints a clear error instead of silently loading a downloadable default.

## Fixed — CUDA support for older NVIDIA GPUs (#302, #307)

CUDA 13 dropped the Pascal and Volta architectures, so the standard `main-cuda`
image can't run on those cards. The CUDA releases now split correctly:

- **`main-cuda-12`** (CUDA 12.x) ships **Pascal (sm_61)**, **Volta (sm_70)**, and
  **P100 (sm_60)** SASS — use it on GTX 10-series, Quadro P-series, Tesla P100/V100.
- **`main-cuda`** (CUDA 13) floors at **sm_75 (Turing)**, covering RTX 20-series
  and newer including RTX 50-series (Blackwell).

If `nvidia-smi` shows a pre-Turing card, pull the `-12` tag.

## Fixed — TTS backends blank/garbled/hung on Vulkan (#304)

Several TTS backends miscomputed under the **Vulkan** backend — the path
SubtitleEdit uses for every Windows user (it ships the Vulkan build and drives
synthesis through the crispasr server `/v1/audio/speech`). A full audit of every
SubtitleEdit-exposed CrispASR TTS engine on a real NVIDIA Vulkan device found
three broken; each now runs the affected stage(s) on **CPU** under a Vulkan
backend, restoring correct audio. **Metal, CUDA, and CPU were never affected**,
and each fix has an env override to force the native Vulkan path:

- **CosyVoice3** — blank/garbled: the flow-matching / HiFT / LLM stages
  miscompute (the AR decode collapses to a few near-silent tokens). Override:
  `CRISPASR_COSYVOICE3_VULKAN_NATIVE=1`.
- **Qwen3-TTS** — the talker LM runs away into a long clip of clipping noise; its
  codec was already CPU-pinned, so only the talker moves to CPU. Override:
  `CRISPASR_QWEN3_TTS_VULKAN_NATIVE=1`.
- **Zonos** — a hard hang/timeout (the AR transformer + DAC-44 kHz vocoder graph
  never completes). Override: `CRISPASR_ZONOS_VULKAN_NATIVE=1`.

The other audited backends — **F5-TTS, IndexTTS, MOSS-TTS, VibeVoice, VoxCPM2** —
render correctly on Vulkan and are unchanged.

## Improved — whisper-VAD runs on the GPU, faster (#305)

The `whisper-vad-asmr` voice-activity model had a CPU-hardcoded encoder; it now
runs on the **GPU** (Metal / CUDA / Vulkan) for ~**3.3×** faster VAD, plus a
parallelized mel front-end. The requantized GGUF keeps the positional embeddings
unquantized (halves the quant error on the short VAD windows).

## Fixed — whisper subtitles: double capitalization + spurious full stops (#308)

`--backend whisper` re-ran punctuation restoration on whisper's already-punctuated
text, producing double capitals (`Hello` → `HEllo`) and extra full stops. Whisper
now declares native punctuation so the redundant pass is skipped, and the
restorer's capitalizer no longer mangles an already-capitalized first letter.

## Docs

- README backend counts corrected to match the binary and the repo's own
  `tts`-cap convention: **104 compiled backends, 53 ASR, 51 TTS, 27 caps** (the
  prose had drifted to 91/43/48/21 while the generated
  [`docs/feature-matrix.md`](docs/feature-matrix.md) was already current).

## PyPI

The `crispasr` Python wrapper was already published at 0.8.23; this binary
release realigns the tagged binaries with it.

**Full changelog:** https://github.com/CrispStrobe/CrispASR/compare/v0.8.22...v0.8.23
