# CrispASR v0.8.14

Correctness + packaging release. Two long-running community issues close
(**#267** diarization ordering, **#273** CJK mojibake), **#227**'s VAD export
becomes a proper standalone verb, and a new **model-free WebRTC VAD** backend
lands. The `crispasr-sys` build is now portable enough to cross-compile and to
build under Rosetta, and a measured Kaggle A/B closes a long-standing
"unshipped CPU win" in the performance backlog — as a loss.

## Fixes

- **#267 — diarization now runs AFTER external CTC alignment.** When an ASR
  backend has no native word timestamps and you supply an aligner (`-am`),
  diarization used to run first and could only stamp one dominant speaker over
  a whole segment — even when the aligner subsequently produced the word
  timings needed to split it at the speaker turn. Alignment now runs first in
  **all three** dispatcher paths (per-slice, stitched, unified), so the
  pyannote/sherpa word-level turn splitter actually has `words[]` to work with.
  The stitched path additionally gained the diarize + global-speaker stages it
  was missing entirely. 14 unit tests + a live integration script.
- **#273 — CJK mojibake in glm-asr and mimo-asr.** Both detokenizers only
  handled `Ġ`→space and `Ċ`→newline from the GPT-2 byte table and passed every
  other encoded codepoint through raw, so Chinese text (e.g. `你`, stored as
  `ä½ł`) came out as Latin mojibake. All decode sites — `glm_asr.cpp`, the CLI
  adapter, and the session C API (which reimplements dispatch separately) — now
  use the full 256-entry `core_bpe::token_bytes_to_utf8()`, matching moss/tada.
- **#227 — `--vad-export` is now a standalone verb.** It loads audio, runs
  Silero VAD, writes the boundary JSON and exits — **no ASR model required**,
  and the short-circuit happens before backend creation so there is no model
  download/load overhead. Multi-file runs get per-file exports. `--vad-export`
  and `--vad-import` now also **imply `--vad`**, so you no longer silently get
  continuous fixed chunks instead of real VAD segments.
- **firered-asr session API:** `crispasr_session_open_with_params(use_gpu=0)`
  was silently ignored on the firered-asr branch (the context params never
  received `use_gpu`), so CPU-forcing had no effect; and
  `crispasr_detect_backend_from_gguf` lacked the `firered-asr`
  `general.architecture` mapping, so plain `crispasr_session_open` failed on
  FireRed GGUFs while the explicit backend worked. Both fixed.
- **Go bindings:** cgo `LDFLAGS` re-synced with the CMake link graph
  (`-ltada-encoder` drift, plus the new `webrtc-vad`), keeping the
  `cgo-ldflags-drift` CI gate green.
- **Lint:** three cppcheck findings resolved.

## Features

- **Source separation — a new task (`--separate`).** Splits a mix into stems,
  written as `<input>_<stem>.wav`. Two backends, architecture auto-detected
  from the GGUF:
  - **Mel-Band RoFormer** (§248, MIT weights + MIT code) — vocal/instrumental
    separation, ported from scratch (no PyTorch at runtime). The C++ forward
    was validated **stage by stage against the reference at cos = 1.000000**
    (STFT → mel-binary band split → 6 axial time/freq RoFormer layers with
    adjacent-pair RoPE + per-head gating → mask MLP → scatter-average complex
    mask → iSTFT), with the reconstructed waveform bit-exact (max_abs 2.4e-7).
    On a speech clip the vocal stem carries ~40× the energy of the residual.
  - **htdemucs** (4-stem: drums/bass/other/vocals).

  `--stems vocals,drums` selects a subset; `--sep-output-dir DIR` sets the
  output location. Separation runs as its own task (audio out, not a
  transcript) — a single dispatcher resolves the model, resamples to the
  model's rate, runs it, and writes the stems (no AI-provenance tag: the audio
  is the user's own, just split). See `docs/source-separation-surface.md`.
- **New WebRTC VAD backend — no model file at all.** Vendors Google's WebRTC
  VAD C source (BSD-3) into `third_party/webrtc/` and wraps it as a VAD
  backend. Unlike Silero / FireRed / MarbleNet this is purely algorithmic (a
  6-band GMM): **zero weights, zero download**, ~2K LOC of C. Select it with a
  VAD model path containing `webrtc` (e.g. `--vad -vm webrtc`); it uses 30 ms
  frames at 16 kHz and emits the same segment struct as every other VAD
  backend. Aggressiveness 0–3 via `CRISPASR_WEBRTC_VAD_MODE` (default 1,
  moderate). Useful for air-gapped/offline deployments and as a zero-cost
  fallback when no VAD GGUF is available.
- **Configurable shipped library name.** New `CRISPASR_LIB_OUTPUT_NAME` cmake
  cache variable (default `crispasr`) feeds the target `OUTPUT_NAME`, and
  `crispasr-sys` forwards its existing `CRISPASR_LIB_NAME` env override to it —
  so downstream bundlers can rebrand the `dylib`/`so`/`dll` without touching
  the C API.

## Build & packaging

- **`crispasr-sys` cross-compiles and builds under Rosetta.** ggml defaults to
  `-march=native`, which hard-fails under a Rosetta 2 toolchain: clang's
  host-CPU probe reports the Apple Silicon die (`apple-m2`/`m4`) while
  targeting `x86_64`, and cc rejects it with `unknown target CPU`. It is also
  simply wrong for distributed Intel-Mac artifacts, which must not be tuned to
  the build machine. `GGML_NATIVE=OFF` is now passed for host≠target cross
  builds **and** for all `x86_64` macOS builds (a Rosetta toolchain
  masquerades as a native x86_64 host, so the cross check alone can't catch
  it). ggml's per-ISA defaults (AVX2/FMA/F16C) still apply. Opt back in with
  `CRISPASR_FORCE_GGML_NATIVE=1`.
- **`crispasr-sys` no longer links Homebrew-only extras.** The opus/amr
  *file-decode* extras linked `libopusfile`/`libopencore-amr` by absolute path,
  breaking `libcrispasr` on machines without those Homebrew packages. Callers
  of the sys crate feed raw PCM, so the extras were dead weight.
- **ggml ccache auto-detection disabled in the sys build.** sccache + Ninja
  fails on the `GGML_METAL_EMBED_LIBRARY` assembly object (ninja emits a
  depfile rule for the `.s` compile; sccache can't produce the `.d` and
  aborts). The explicit CrispASR ccache launcher is unaffected.
- **CI:** the pub.dev publish / wrapper-release jobs are gated on the protected
  environment; JS binding version synced to `VERSION`.

## Performance

- **The "batched TDT decode is an unshipped CPU win" backlog item is closed —
  as a measured loss.** `CRISPASR_TDT_BATCH` batches the joint head into one
  large sgemm instead of many small ones, which predicted a CPU win. An A/B on
  a clean Kaggle CPU box (issue **#81**, transcripts byte-identical in both
  arms) measured the opposite on `parakeet-tdt-0.6b`:

  | clip | default decode | `TDT_BATCH=1` | ratio |
  |------|---------------|---------------|-------|
  | jfk 11 s   | **4.0×** RT | 3.2× RT | 0.80× (20 % slower) |
  | long 134 s | **3.0×** RT | 2.2× RT | 0.73× (27 % slower) |

  Batching the joint over *all* encoder frames does strictly more work than
  TDT's duration-skipping greedy decode, which visits far fewer frames — one
  big sgemm doesn't pay for the extra frames. **The default stays OFF**; the
  batched path remains opt-in and is still used by the long-form streamed path.
  (`parakeet-ctc` is unaffected at 0.99× — it never enters the TDT decode.)
- The real remaining #81 gap is now documented explicitly: crispasr
  `parakeet-tdt` on **CPU** is ~2.1× slower than onnx-asr int8 (4.0×/3.0× vs
  8.6×/5.8×). It needs a CPU BLAS/kernel lever — batched decode is ruled out.
  GPU is unaffected and strong (36.9×/50.9× on a P100).

## Internal / tooling

- **Every Kaggle build kernel was broken by the `c2pa-audio` submodule.**
  `src/CMakeLists.txt` builds `crispasr_c2pa_native` from that submodule, but
  kernels clone `--depth 1` and init only `ggml`, so cmake generate died with
  *"Cannot find source file: …/c2pa_native.cpp"*. All 87 kernels audited:
  those cloning `--recursive` were fine, 33 were not. C2PA signing is
  irrelevant to a benchmark/convert kernel, so `kaggle_harness` gained
  `crispasr_cmake_flags()` (→ `-DCRISPASR_NO_C2PA_NATIVE=ON`), folded into
  `cache_and_link_flags()` so 21 kernels were fixed with no per-kernel edit;
  11 hand-rolled cmake lines were patched directly.
- Kaggle harness regime hardening: `resolve_hf_token()` added to kernels that
  pulled from HF unauthenticated, `init_progress()` where missing, and
  `HF_HUB_ENABLE_HF_TRANSFER=0` (hf_transfer wedges multi-GB Kaggle downloads).

## Upgrade notes

- **Diarization output may change shape when `-am` is used.** With an external
  aligner, a segment that spans a speaker turn is now split into multiple
  single-speaker segments instead of being emitted as one segment with a
  dominant-speaker label. This is the #267 fix and is the intended behaviour;
  consumers that assumed a 1:1 ASR-segment → output-segment mapping should
  re-check. Segment-level dominant-speaker attribution is still the fallback
  when no word timestamps exist.
- **`--vad-export` no longer transcribes.** It is now a standalone verb that
  writes the boundary JSON and exits. Scripts that relied on getting a
  transcript *and* an export from one invocation must now run two passes
  (export, then `--vad-import`) — which is the intended workflow and skips the
  ASR model load entirely on the export pass.
- `--vad-export` / `--vad-import` now imply `--vad`; pass `--chunk-seconds N`
  explicitly if you actually wanted fixed chunk boundaries.
