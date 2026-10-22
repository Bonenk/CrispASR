# v0.8.44 — Python metadata, long-audio alignment and native runtime fixes

This release improves Python/native result parity, long-input forced alignment,
Whisper callback timelines, Breeze synthesis, Dolphin decoding and native
packaging. Existing model and precision defaults are preserved.

## Python metadata and long-audio alignment

- `SessionSegment.tokens` exposes owned token text, lossless UTF-8 bytes,
  timestamps and confidence through additive C ABI getters. Parakeet Japanese
  slice/gap-fill and output-splitting paths retain the selected words' token
  blocks without duplicating tied-frame tokens.
- MOSS-Transcribe-Diarize sessions parse native segment timestamps and speaker
  labels. Unavailable word/token timestamps are `None`, with `has_timestamps`
  indicating availability; VibeVoice word timing still requires alignment.
- Python exposes `crispasr.free_aligner_cache()` and
  `session.free_aligner_cache()` for the process-wide aligner cache.
- Canary/FastConformer alignment bounds encoder windows while preserving one
  global transcript alignment. CLI align-only honors `--chunk-seconds`; bindings
  use the bounded default. Host logits/alignment memory still scales with input
  duration, and align-only does not apply VAD.
- CTC alignment rejects incomplete subword paths and invalid ABI inputs.
  Experimental compact/streaming workspaces remain opt-in.

Real Japanese Parakeet Metal metadata reuse and a 2,703-second T4 repeated-speech
alignment run pass independent audits. The retained CUDA trace shows sampled
device use dropping from 2,735 to 125 MiB after cache release, followed by fresh
CLI model residency. These are bounded metadata, ownership and memory controls,
not general acoustic-boundary accuracy or per-process allocator accounting.
See [metadata usage](docs/python-session-metadata.md) and
[complete evidence and limits](docs/issue-followups-2026-10-20.md).

## Native inference and bindings

- Dolphin honors checkpoint causal/noncausal convolution and normalization
  behavior; automatic C ABI and Python sessions recognize its GGUFs.
- Breeze forwards instructions and reference audio together, initializes dynamic
  GPU backends before device selection, and handles server voice-cache changes
  and unknown-voice recovery.
- TTS sessions preserve reference-buffer ownership and use the correct input and
  output sample rates. Speaker APIs reject invalid indices, malformed buffers
  and inconsistent native results.
- Whisper callbacks preserve segment/token/DTW timelines within the supplied
  decoder window. Parallel decoding propagates worker failures, respects VAD
  timing and retains owned token snapshots. VAD does not throttle the decode
  that opens an utterance.
- Go high-level models guard overlapping native operations and close, including
  callback re-entry, and snapshot options for decoding. Raw low-level contexts
  still require caller serialization.
- Python, Go, Rust, Dart, Java, C# and Ruby/WASM adapters gain scoped operation,
  ownership, callback and lifetime fixes. Validation covers the documented
  contracts; it does not establish complete feature or platform parity.
- Dia corrects EOS/output slicing and top-k selection, forwards top-k/top-p
  controls through CLI and Session, and restores server sampling defaults after
  per-request overrides. OmniVoice handles output-allocation failures safely.

## CANN and distribution

- CANN exposes explicit backend/device routing, rejects contradictory device
  requests, and fixes rotary-cache capacity tracking. Scoped physical operator
  controls and both SDK architecture builds pass; these do not certify complete
  model inference or performance.
- Explicit model-weight usage hints support optional CANN NZ caching while
  retaining ordinary weight storage. The NZ cache remains **off by default**.
  Existing GGUF tensor names and published model files remain valid.
- All native library packagers ship `crispasr_session.h` with `crispasr.h`.
  Cross-platform installed-wheel checks verify relocated loading and exported
  Session APIs. Optional helper diagnostics describe native by-reference
  alternatives correctly.
- Clean-build CI discovers Python before registering its tests and checks that
  the cache/loader/launcher/metadata validators are selected. Model-free metadata
  validation is included in the ordinary unit tier.

## Model-license policy and known limits

Ordinary NC models remain eligible behind explicit license acceptance and clear
labeling. Managed OmniVoice bundle downloads require the distinct
`omnivoice-terms` acknowledgment; ordinary NC acceptance does not cover its
additional tokenizer terms. OpenRAIL also requires explicit acknowledgment.
These switches grant no additional rights or project approval of custom terms.
See [the license policy and source audit](docs/model-license-policy.md).

OmniVoice PR #521 and shared-ggml Conv1D PR #2 remain unmerged. Fish Drama3 and
Nyra integration are excluded under the current license policy. No new Echo or
MiMo Q4 recipe, broad Dia speech-quality claim, OpenVoice2 pretrained/GPU quality
claim, Radeon acceleration result or Ascend model-performance claim is promoted.
Separate reporter-specific Canary, Intel Arc and legacy NVIDIA cases still need
the affected inputs or hardware. Detailed accepted and rejected investigations
remain in [HISTORY.md](HISTORY.md) and the linked validation receipts.
