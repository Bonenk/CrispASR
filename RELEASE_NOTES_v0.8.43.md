# v0.8.43 — audio correctness, frontend parity and measured opt-in optimizations

This release covers merged PRs #519, #520, #522, #523, #524, #525, #526, #527, #528, #534, #535, #537, #538 and #539. Experimental quantization candidates and unfinished hardware
validation are excluded from the released behavior described below.

## Breeze TTS 2: honor session sampler controls

The existing Breeze session path now forwards temperature, seed and token
limits to the native context. The TTS seed setter also forwards the full 64-bit
seed. Clearing a token limit restores the model's GGUF metadata limit. Breeze
opening no longer depends on the unrelated Confucius build guard, and the CLI
accepts `--tts-instruct` as an alias for its existing `--instruct` option.

PR #539 passed all 56 applicable checks, with one intentional skip. Eight
source-compiled controls cover dispatch, native setter arguments, limit reset,
standalone Breeze builds and unsupported/null contexts; the previous source
failed six controls. This change does not include the pending instruction,
reference-ownership or server voice-cache features from #529/#531/#536.

Evidence: [independent sampler control proof](docs/breeze-independent-sampler-2026-10-11.json).

## Transcript cleanup: preserve UTF-8 continuation bytes

The post-decode repetition cleanup now recognizes only the six ASCII whitespace
bytes. Locale-dependent `std::isspace` could classify raw UTF-8 continuation
bytes 0xA0 and 0x85 as whitespace on macOS, stripping bytes from Chinese
characters such as 造 and 加 and breaking strict binding decoding.

PR #538 passed all 45 applicable checks, with one intentional skip. Actual
macOS libc reproduces the old corruption and the corrected cleanup preserves
both reported contexts byte-for-byte. Both supplied WAVs also pass actual Q4_K
C ABI decoding on hosted macOS and Linux: all raw-token 造/加 occurrences survive
segment/word publication, with strict UTF-8 and no replacement characters.
The hosted Mac has no Metal device; physical M4/macOS26 and GPU numerical parity
remain untested. Linux is a regression control: its libc does not reproduce
the locale classification bug.

Evidence: [macOS root-cause and clip proof](docs/qwen3-locale-utf8-2026-10-11.json),
[Linux reporter-clip proof](docs/qwen3-locale-utf8-linux-2026-10-11.json).

## HTDemucs: Q8 source separation on Metal

Cross-attention now uses its already-contiguous Q/K/V weight-row views directly,
removing redundant Q8 copies that Metal cannot execute. The previous graph
aborted with `unsupported op 'CONT'` on Q8 models.

PR #537 passed all 35 applicable CI checks, with two intentional skips. An
independent hosted macOS test forces the changed fused graph on CPU: all four
F16 and Q8 stems are byte-identical before/after. The contributor's pinned M4
receipt reports the Q8 abort fixed, byte-identical F16 Metal stems, and Q8
Metal/CPU per-stem cosine 0.999495–1.000000 with norm ratio 0.99953–1.00000.
Those physical Metal results are attributed to the contributor; the hosted Mac
has no Metal device. The fixture is speech-only, so this is a focused regression
check rather than a music-corpus quality or performance claim.

Evidence: [source, models, output hashes and attributed Metal receipt](docs/htdemucs-metal-views-2026-10-11.json).

## Session results: valid UTF-8 before publication

Malformed assembled result bytes now become U+FFFD before segment callbacks
and result getters publish them. Python's strict decoder stays enabled; a bad
byte no longer discards an otherwise usable transcription chunk. Valid UTF-8,
including characters assembled from multiple byte-level tokens, is unchanged.
Segment/word times, confidence and counts are preserved.

PR #534 passed all 54 applicable checks, with one intentional skip. The hosted
native build passed 2,008 unit tests and four strict ctypes regression tests.
This protects result publication; PR #538 separately fixes the locale-dependent
continuation-byte loss described above.

Evidence: [session UTF-8 controls](docs/session-utf8-2026-10-11.json).

## C ABI: complete canonical backend discovery

The generated capability table now includes Hikari and the German Dattazigzag
Moonshine backend, which the actual CLI listed but C ABI lookup omitted.
CI compares the built CLI and shared library across all 136 canonical names,
checks list sizing/error contracts and verifies generator output using pinned
clang-format 18. Aliases remain outside the canonical enumeration.

PR #535 passed all 42 applicable checks, with one intentional skip; its hosted
full native build passed the actual CLI/C ABI comparison and 2,008 unit tests.

Evidence: [canonical discovery proof](docs/backend-caps-2026-10-11.json).

## VoxCPM2: correct session audio rate and configurable CFM steps

VoxCPM2 session synthesis now returns native 48 kHz mono PCM, matching the
sample-rate getter and CLI. The previous session path halved the sample count
while advertising 48 kHz, causing clients that trusted the getter to play or
transcribe the result at double speed. Binding documentation now directs callers
to the backend output-rate getter when saving or playing PCM.

The existing session `set_tts_steps` setter and explicit CLI `--tts-steps`
override now reach VoxCPM2's CFM solver and apply to the next synthesis call.
Ten steps remain the default; lower counts can reduce speech quality. There is
no new eight-step quality claim or Intel B390 performance claim.

PR #524 passed all 45 applicable CI checks, including its model regression,
with one intentional skip. Integrated native files and VoxCPM2 C ABI changes
match the validated CUDA/Vulkan source. Original NVIDIA/F16 recognition reads
all 22 fixed VoxCPM2 controls correctly. This is decoded-output evidence,
not a claim of full Python/native waveform equality.

Evidence: [session contract and proof pins](docs/voxcpm2-session-contract-2026-10-09.json).

## Nemotron 3.5: correct terminal frontend masking and boundary diagnostics

The frontend now zeroes frames beyond floor(samples/hop), matching the
pinned original NVIDIA attention mask while retaining the original STFT
shape. The old terminal frame could carry log-mel values into preencode.
The diff tool now checks actual mel, preencode and encoder shapes and stages;
an additive C diagnostic API exposes the preencode output.

PR #525 passed all 44 applicable checks, with one intentional skip. Exact
source passed all three unchanged F16 stage gates on CPU and actual dual T4s.
On T4, encoder cosine minimum is 0.99999088 and relative L2 is 0.00028004;
all 26 original transcript controls agree in fresh and reverse-reused
sessions. Long-turn streaming presets 0/2/3 pass packet-size token equality,
byte-exact reset/repeat confidence and idempotent final flush. These streaming
checks do not claim parity with the original Python streaming implementation.

Evidence: [CUDA stages and streaming proof](docs/nemotron-frontend-gpu-2026-10-09.json),
[CPU original transcript controls](docs/nemotron-frontend-controls-2026-10-09.json).

## Python: band-limited array resampling and contiguous input buffers

Five Python transcription entry points now use the existing native Kaiser
polyphase resampler for array inputs at rates other than 16 kHz. This addresses
aliasing from the former linear interpolation path. An additive C helper
exposes in-memory resampling; older native libraries use a bounded NumPy
compatibility implementation without requiring SciPy.

Inputs become contiguous float32 arrays, including strided inputs. Exact
16 kHz input values are preserved. The independent file-loading path still
uses miniaudio; this change concerns Python array inputs.

PR #522 passed all 50 applicable CI checks, with one intentional skip. Eight
checks against the actual shared library cover alias rejection, agreement
between native and fallback output within 1e-7, input/rate/error contracts,
contiguity and independent allocation/free ownership.

Details: [Python audio resampling](docs/python-audio-resampling.md).

## Python: discover packaged native helpers

Automatic native-library discovery and `CRISPASR_LIB` selection now load the
adjacent platform-specific helpers library. Previously only an explicit
constructor library path reliably found that helper. Explicit helper overrides
and legacy build/helper-free fallbacks remain supported.

PR #526 passed all 41 applicable CI checks, with one intentional skip. All 18
constructor routes pass; eight resampling tests pass against the actual staged
Linux native and helper libraries with the candidate binding overlaid. This
proves package loading and resampling, without claiming native Windows/macOS
execution or model inference.

Evidence: [actual package helper proof](docs/python-package-helpers-2026-10-09.json).

## Release packages: immutable source and WASM output-rate guidance

All 33 release workflow checkouts now select the explicit requested tag or the
immutable workflow event commit. Queued dry-run jobs previously selected a
moving branch, allowing one matrix to contain packages from different commits.
A hosted package probe verified the fix while the branch advanced after dispatch.

The generic WASM archive README now uses `Module.sessionOutputSampleRate()`
after opening a model, rather than assuming every backend returns 24 kHz.
PR #527 passed all 32 applicable checks, with one intentional skip, including
Windows CUDA 13 verification. Actual staged WASM model-free API checks pass;
these do not establish synthesis or browser threading acceptance.

Evidence: [immutable checkout probe](docs/release-checkout-pin-hosted-2026-10-09.json).

## Bindings: discover bundled ggml CPU and GPU plugins

C ABI sessions now discover ggml plugins beside the loaded native library,
including CPU-only callers and direct Nemotron initialization. Previously the
loader searched the process executable directory and current directory, so a
Python process outside the CUDA package directory could find no backend.
Module-path discovery preserves UTF-8 under C++17 and C++20.

PR #528 passed all 50 applicable checks, with one intentional skip. The actual
rebuilt CUDA archive passes dual-T4 validation with the installed CUDA 12.8
runtime: its C ABI loads bundled CPU/CUDA plugins and allocates 1,330 MiB on its
own PID. All three unchanged F16 stage gates pass; 26 original word controls with exact fresh/reused
output, and native streaming packet/reset/flush checks pass. These results
do not establish Windows GPU execution, original streaming parity or speedups.

Evidence: [actual package CUDA proof](docs/release-cuda-package-gpu-2026-10-09.json).

## MiMo-ASR: opt-in cached GPU decode

`CRISPASR_MIMO_ASR_GPU_STEP_GRAPH=1` enables a cached text-only step graph while
retaining the existing prefill path. The cache is invalidated before every
transcription. The optimization remains OFF by default.

Actual T4 ABBA measurements improve warm full-call medians by 2.4–2.7%, with
sampled GPU peak memory increasing by 18 MiB. All 64 EN/ZH speech calls,
eight CLI/session pairs and 114 full-vocabulary comparisons remain byte-exact:
cosine 1, relative L2 zero, matching norms and argmax. This does not establish
CANN, P100 or new quantization acceptance. PR #519 passed applicable CI.

## Opt-in parallel mel projection

`CRISPASR_MEL_PROJECTION_PARALLEL=1` enables parallel scalar mel projection;
it remains OFF by default. The implementation retains the serial path below
64 frames or without OpenMP and leaves the BLAS path unchanged.

All 144 hosted ABBA cases are bit-exact. Four-thread projection medians improve
3.30–3.58× at 64 frames and 3.05–3.50× at 3000 frames on the hosted runner.
A contended local VPS regressed at 64 frames, so these component timings are
not an end-to-end ASR speed claim. Qwen3 stage gates and CLI transcripts pass.
PR #520 retains KokerZhou's independently authored mel implementation from
PR #492 and passed applicable CI and selected model regressions.

## Index-Echo: decoder activation calibration support

Opt-in activation collection now composes with decoder stage capture.
Without calibration environment variables, the existing callback remains in
use. A native CPU unit verifies actual GGUF activation sums and row counts.

Actual dual-T4 collection OFF/ON passes three clips × 76 stage comparisons
in each arm, complete file translations, CLI/anonymous-model C ABI calls and
TTS-ASR roundtrips. Decoded receipts match. Collection covers 249 decoder
matrices, including valid widths and counts for all 96 FFN matrices. PR #523
passed all 42 applicable CI checks, with one intentional skip.

The held-out smoke statistics are excluded from quantization. A separately
prepared 48-clip CC0 EN/ZH corpus (259.176 seconds) is PCM-disjoint from the
acceptance inputs; fresh dual-T4 calibration also collected valid statistics
for all 96 FFN matrices. This establishes calibration coverage, not Q4 model
acceptance.

Evidence: [callback smoke proof](docs/index-echo-calibration-smoke-2026-10-09.json),
[independent corpus and GPU calibration](docs/index-echo-calibration-corpus-2026-10-09.json).

## Model and validation status

No production model files, registry quantization pins or quality thresholds
change with these merged PRs. Actual dual-T4 calibrated Echo Q4 testing passed
its full F16 control, but all three candidates failed stage/magnitude and
complete translation gates. No candidate is promoted; the terminal proof is
pinned in `docs/index-echo-q4-imatrix-validation-2026-10-09.json`.
Nemotron RNNT/prompt guards recover all 26 fixed transcripts, but Q4 encoder
parity remains rejected. Further FFN projection isolation also rejects both
926 MB mixed-precision candidates (encoder cosine 0.977366 / 0.882196);
no quantization default is changed. The merged frontend length-mask correction passes
all three F16 stage gates and all 26 original transcript controls with no
fresh/reused differences. Actual dual-T4 validation also passes all three stage gates, the 26 original
transcripts, fresh/reused equality and native streaming packet/reset/flush
checks for presets 0/2/3. PR #525 is merged; further Q4 encoder precision work is unfinished. Full proof:
`docs/nemotron-frontend-gpu-2026-10-09.json`.
OmniVoice PR #521 remains a draft because clone acceptance is rejected.
The pinned original F32 model also fails the same fixed clone control at 2/9
word errors with and without cleanup. This upstream result does not waive
native acceptance; no cleanup default is promoted.
The remaining CANN and hardware-specific work in PR #492 is not included.

The final tagged release workflow passed every build/validation job and published
50 assets. Actual published CPU CLI/library/Python archives match GitHub digests
and pass dependency closure, relocated loading, all 136 capability rows and
automatic binding discovery. Actual tagged Windows CUDA 12.8/12.6 CLI/runtime/
library pairing passes, including wrong-runtime rejection.

Both actual tagged Linux CUDA archives pass on dual T4s in separate fresh
processes, each loading its own hash-verified CUDA library: three original
stage/magnitude comparisons, 26 original transcript controls in fresh and
reused sessions, and nine streaming packet/reset/flush controls per archive.
Independent verification recomputed all 12 saved native/reference arrays and
checked all 52 original word cases and 18 streaming controls. Encoder cosine
is 0.99999088 for each archive. This validates the published bytes separately
from the earlier candidate build; it does not certify every backend or GPU.

Immutable evidence: [tagged CPU/Python package proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/blob/e5d57833d50bc7e74aa61679e8a3842822b91a24/release/38199002944/cpu-python-package-proof.zip),
[tagged CUDA hardware and Windows runtime proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/blob/a5c5642fb48f1895f7b0bc2263abd89b80a6fd5a/release/38199002944/tagged-cuda-windows-package-proof.zip).

The subsequent Nemotron attention isolation also rejects all three source-guarded
Q4 arms. FFN+QKV source precision is closest at encoder cosine 0.998536 versus
the unchanged 0.999 gate, with a 1,143,237,536-byte model and only 28,311,552
bytes of actual Q4 tensor payload. FFN+attention-out and FFN+position-source
arms fail at 0.996296 and 0.987063. All seven baseline arrays match the prior
run exactly, F16 passes all three stages, and all 24 original frame-layer
controls pass for F16. These mostly-source diagnostic hybrids are excluded
from production quantization and release performance claims. See [nemotron-encoder-attention-2026-10-09.json](docs/nemotron-encoder-attention-2026-10-09.json).

A mostly-source Nemotron hybrid, retaining FFN/QKV/attention-out precision
and quantizing position projections to Q4_K, passes all three unchanged CPU
stage gates (encoder cosine 0.999842) and all 24 original layer comparisons.
Final relative L2 is 0.00309294. It also matches 26/26 original word controls,
22/22 VoxCPM2 controls and exact fresh/reverse-reused outputs. Its model hash
and protected/unprotected tensor ledger were independently verified.

This 1,179,413,408-byte artifact contains only 14,155,776 bytes of Q4 tensors;
its 8.3% saving from F16 is not a compact production Q4 or speed claim. Actual dual-T4 GPU validation now passes all three unchanged stage gates
(encoder cosine 0.999734, relative L2 0.003356), 26 original word controls,
exact fresh/reused state and native streaming presets 0/2/3 in separate F16
and candidate processes. Full arrays and independently verified public proof:
`docs/nemotron-position-q4-gpu-2026-10-10.json`. It remains excluded from
release quant pins; no speed, original streaming parity or full-layer GPU
capture claim.
See [stage/layer proof](docs/nemotron-encoder-late-layer-2026-10-09.json)
and [original decoded controls](docs/nemotron-q4-original-controls-2026-10-10.json).
