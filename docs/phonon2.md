# Phonon-2

[Fermion Research's Phonon-2](https://huggingface.co/FermionResearch/Phonon-2)
is an English speech recognizer derived from NVIDIA Parakeet TDT 0.6B v3.
CrispASR uses its existing Parakeet backend for this model.

```sh
crispasr -m phonon2 -f recording.wav                      # Q8_0 default, ~674 MB
crispasr -m phonon2 --model-quant f16 -f recording.wav     # reference fidelity, ~1,255 MB
crispasr -m phonon2 --model-quant q4_k -f recording.wav    # smaller, ~402 MB
```

## Convert the upstream weights

Install the Parakeet converter's dependencies, plus `zstandard`:

```sh
pip install torch numpy gguf sentencepiece librosa pyyaml safetensors huggingface_hub zstandard
python models/convert-parakeet-to-gguf.py \
  --hf FermionResearch/Phonon-2 --output phonon2-f16.gguf
./build/bin/crispasr-quantize phonon2-f16.gguf phonon2-q8_0.gguf q8_0
./build/bin/crispasr-quantize phonon2-f16.gguf phonon2-q4_k.gguf q4_k
./build/bin/crispasr -m phonon2-q8_0.gguf -f samples/jfk.wav
```

`--hf` also accepts a downloaded Hugging Face snapshot or the extracted
directory containing `config.json` and `model.fermion`. Conversion checks the
upstream archive/container SHA-256, expands the five-value encoder weights
exactly (`0`, `±lo`, `±hi`, with two magnitudes per output row), and expands the
remaining integer tables with their stored row scales. It reads the model
configuration and vocabulary from the archive; no teacher weights or downloaded
Python code are loaded. The resulting GGUF retains `general.architecture =
parakeet`, so bindings auto-detect it through the C ABI even if the file is
renamed.

## Size, quality, and speed

The upstream **164 MB** figure describes its compressed transport file, not
an F16 or ordinarily quantized GGUF and not runtime RAM. Its custom five-value
packing differs from ggml's quantization formats. CrispASR expands that format
before GGUF conversion; Q8_0 and Q4_K apply another quantization step. Those
exports should not be assigned the upstream model's benchmark scores without
their own evaluations.

Local CPU validation on 21 clips (JFK, one upstream LibriSpeech sample, and
19 clips from the Hugging Face LibriSpeech test subset):

| Export | Download | Exact reference transcripts | Word edits versus reference* |
|---|---:|---:|---:|
| F16 | 1,255 MB | 21/21 | 0/298 |
| Q8_0 (default) | 674 MB | 19/21 | 1/298 |
| Q4_K (optional) | 402 MB | 15/21 | 7/298 |

\* Lowercase, ignore punctuation, retain apostrophes inside words; edit distance
against the independent model reference, **not WER against human ground truth**.
Q8's differences were one capitalization change and one proper-name spelling.
This is a small conversion/quantization check, not a reproduction of the seven
benchmark sets. F16 passed all 28 compared stages on both JFK and the upstream
LibriSpeech sample: minimum cosine 0.999996 / 0.999796, with tensor norm-ratio
error bounded by 0.065% / 0.259%. C ABI sessions auto-detected the backend and
produced these transcripts with an arbitrary model filename.

Fermion reports 5.21% average WER across seven English benchmark sets and 174×
realtime on an M5 MacBook Air using its MLX engine, excluding loading. These
figures are upstream measurements, not CrispASR performance claims. Phonon-2
is advertised and evaluated for English; its inherited multilingual vocabulary
does not establish multilingual accuracy after retraining.

## Reference validation

```sh
python tools/dump_reference.py --backend phonon2 \
  --model-dir FermionResearch/Phonon-2 --audio samples/jfk.wav \
  --output phonon2-jfk-ref.gguf
./build/bin/crispasr-diff phonon2 phonon2-f16.gguf phonon2-jfk-ref.gguf samples/jfk.wav
```

The reference uses the unmodified upstream container reader and stock
Transformers `ParakeetForTDT`, independently of the converter's unpacker.
Install a recent `transformers` release providing `ParakeetForTDT` to run it.
Only the teacher's small JSON configuration/tokenizer files are fetched.
`PHONON2_BASE_DIR` can point to a local copy of those files for offline runs.

Weights: **CC-BY-4.0**, derived from NVIDIA's Parakeet TDT v3 and retrained by
Fermion Research. Retain attribution and describe conversion/quantization
changes when redistributing. The vendored reference reader is Apache-2.0;
its license is in `tools/reference_backends/phonon2/LICENSE`. The upstream
[NOTICE](https://huggingface.co/FermionResearch/Phonon-2/blob/main/NOTICE)
documents the original model, changes, and training data.

## Integration checklist

Phonon-2 is a model variant of the existing Parakeet runtime. The contributing
checklist applies through that shared engine:

| Checklist point | Wiring |
|---|---|
| C runtime and stage timers | `src/parakeet.{h,cpp}`; `CRISPASR_PARAKEET_BENCH=1` |
| CLI adapter and factory | Shared Parakeet adapter; explicit `--backend phonon2` and model filename routing |
| CLI/library CMake linkage | Existing `parakeet` library in both CLI and shared C ABI; no duplicate runtime library |
| C ABI dispatch, lifecycle, setters | Explicit `phonon2` alias normalizes to `parakeet`; existing transcription, hotwords, beam, temperature, attention-context and cleanup paths |
| Architecture auto-detection | `general.architecture=parakeet`, shared `arch_backend_map.h`; works after renaming |
| Registry | `phonon2`, Q8 default, F16/Q4 alternatives and weight license |
| Quantization | Existing Parakeet rules, including pointwise matmuls; published exports validated |
| Reference and diff | Independent upstream reader + Transformers; existing Parakeet stage APIs and strict pinned nightly gate |
| Bindings | Generic C ABI reaches all bindings; Python/Go/Dart document the variant |
| Go static linkage | Existing Parakeet library; generator checks the unchanged cgo library list |
| Architecture/capability docs | `docs/architecture.md#phonon2`; generated feature matrix and library capability table |
| Tests and live environment | Container/registry tests, pinned regression, `test_phonon2_live.py`, `CRISPASR_MODEL_PHONON2` |

`Session(..., backend="phonon2")` explicitly opens this variant; `Session.backend`
reports the shared runtime name, `parakeet`. The CLI uses model metadata to select
English without automatically loading Whisper for LID. Non-English language
requests produce a warning; the model cannot honour them. Explicit language
identification remains available. Native punctuation is advertised. `--no-flash-attn` and the C ABI open flag
now reach the graph builder; the live guard checks the resulting node trace.
The raw Parakeet context default now reports flash enabled, matching the
previous graph behaviour; an explicit false selects manual attention.

## Reproducible profiling

Build both `crispasr-lib` and `crispasr-cli`. Set the model and library paths:

```sh
python tools/profile_phonon2.py --engine runtime \
  --model phonon2-q8_0.gguf --lib build/src/libcrispasr.so \
  --output runtime-q8.json
CRISPASR_SCHED_PROFILE=1 CRISPASR_PARAKEET_BENCH=1 \
CRISPASR_PARAKEET_ENC_PROBE=1 CRISPASR_PARAKEET_DECODE_TIMING=1 \
python tools/profile_phonon2.py --engine runtime --trace \
  --model phonon2-q8_0.gguf --lib build/src/libcrispasr.so \
  --output trace-q8.json
python tools/profile_phonon2.py --engine reference \
  --model /path/to/upstream/snapshot --output reference-cpu.json
```

The benchmark loads once, warms each 11/55-second shape, checks nonempty stable
transcripts and proportional word counts, then records three inference times
and their median. The trace is a separate run: the scheduler callback forces
node materialization and changes dispatch overhead. Compare timings only on
the same host, device, thread count and load. The Python reference is stock
Transformers on CPU, not the upstream MLX engine.

The manual **Phonon-2 integration and profile** GitHub workflow runs Linux CPU
and macOS Metal, checks wiring/live transcripts and F16 stage parity first,
then records F16/Q8/Q4 timings plus Q8 stage/node traces. Linux also measures
the independent Python reference on the same runner. Artifacts retain all raw
samples, transcripts, model checksums, dependency versions and host details.

## Runtime optimization coverage

The encoder, including subsampling and all 24 FastConformer blocks, is a ggml
scheduler graph, with CPU fallback for operations unsupported by the selected
backend. The scheduler currently registers GPU/CPU backends, not the separate
ggml BLAS backend. Linux encoder matmuls use ggml CPU kernels even when the
build enables OpenBLAS; OpenBLAS accelerates the shared mel filter projection.
Mel extraction and TDT token selection are CPU code. Decoder
execution depends on the device:

| Path | Predictor and joint |
|---|---|
| Linux CPU | Scalar C++ loops over cached F32 weights; the OpenBLAS build option does not accelerate these loops |
| Apple CPU / Metal | Apple Accelerate; Metal encoder with CPU decoder is the existing default |
| CUDA / Vulkan | ggml predictor/joint graphs, built and allocated once per decode call and reused across token steps |

There is no transformer KV cache in this TDT decoder: it retains the two LSTM
hidden/cell states and reuses the predictor output across blanks. Encoder
projections are computed ahead of the token loop; CUDA uses the measured GPU
projection path by default, Apple uses batched SGEMM, and Linux CPU currently
uses scalar per-frame projections. The CPU predictor/joint weight conversions
are initialized lazily and retained by the model context.

The encoder folds batch normalization into depthwise convolution weights and
fuses Q/K/V projections at load time. Quantized models use the shared pointwise
weight repacking rules. Attention uses ggml flash attention by default, with
relative-position scores still computed separately; `--no-flash-attn` selects
manual attention. CUDA also retains its backend-specific manual-attention
policy. These are shared Parakeet optimizations, not a new Phonon-specific
packed five-value kernel.

The scheduler/context persist, but the encoder graph is rebuilt for each call
and its buffers are allocated through the scheduler. The experimental
`CRISPASR_PARAKEET_ENC_CACHE` remains off: the existing implementation can reuse
stale tensor pointers and corrupt repeated-call output. The trace separates
build, allocation and compute costs; graph caching should only be reconsidered
if those first two costs are material. A scheduler trace materializes each
node and perturbs execution, so its absolute timings are diagnostic only.

## Measured CPU profile (2026-09-30)

[CI run 36784469150](https://github.com/CrispStrobe/CrispASR/actions/runs/36784469150)
measured runtime commit `9e9816631` and the independent Python reference on the
same Linux runner: AMD EPYC 9V74, 4 vCPUs, 4 inference threads, Release build,
OpenBLAS. The reference is stock Transformers `ParakeetForTDT` in F32 with
PyTorch 2.7.0 CPU, loading the original Fermion container through the upstream
reader. It is not the separate MLX implementation.

One model is loaded per process; each 11/55-second shape gets a warmup followed
by three timed calls. Loading, downloads, warmup and diagnostic callbacks are
excluded. The 55-second clip repeats JFK five times and stays on the ordinary
single-pass path. Every repeat produces stable nonempty output: 22/110 words.
F16 and Q8 transcripts match the Python reference exactly at both lengths;
Q4 matches normalized words, with a punctuation difference on the longer clip.
This is a throughput check, not an accuracy benchmark on natural long audio.

| Engine/export | 11 s audio: median / realtime | 55 s audio: median / realtime | Peak process RSS* |
|---|---:|---:|---:|
| Python reference F32 | 1.580 s / 6.96× | 8.042 s / 6.84× | 3,414 MiB |
| CrispASR F16 | 3.731 s / 2.95× | 19.222 s / 2.86× | 2,249 MiB |
| CrispASR Q8_0 | 1.760 s / 6.25× | 9.301 s / 5.91× | 1,614 MiB |
| CrispASR Q4_K | 2.043 s / 5.38× | 10.878 s / 5.06× | 1,320 MiB |

\* Peak RSS covers model load, warmup and inference in the isolated process;
it is not file size, tensor allocation size or GPU VRAM. Q8 takes 11–16% more
inference time than this Python reference and uses 53% less peak process RAM.
Q4 is smaller but slower than Q8 on this host; the default remains Q8.

All 28 compared frontend/encoder stages pass on this CPU runner: minimum cosine 0.999994 and tensor
norm-ratio error bounded by 0.066% (RMS error divided by reference RMS).
The archive covers mel, subsampling, 24 encoder layers and two encoder-output
checks. Predictor/joint numerical stage parity is not captured here; decoded
transcripts provide their end-to-end validation. The local shared-library run
also passes all 28 stages (minimum cosine 0.999996,
bound 0.065%) and repeated CTest/CLI/C ABI checks. The live guard includes an
explicit flash-off node trace.

A separate warmed Q8 diagnostic call reports mel 13.5 ms, encoder 1170.7 ms
and decoder 587.4 ms. Encoder graph build/allocation are only 0.51/0.47 ms.
The two FFN matmul shape groups account for 50.3% of traced encoder time;
fused Q/K/V accounts for 9.3%, flash attention 3.2%, and relative-position
matmul 2.2%. The Linux decoder remains scalar: its encoder projection alone
is 70.6 ms. The old trace called that path "cblas" incorrectly; the diagnostic
label is now corrected to "scalar" ("accelerate" on Apple builds).

These instrumented timings are for finding hotspots, not benchmark numbers.
The useful next experiments are optimized CPU predictor/joint matvecs and
encoder FFN kernels (including an explicit BLAS scheduler A/B), each with transcript and stage-parity A/B. Encoder graph
caching would save less than a millisecond in this trace and retains its known
correctness problem; it stays off. No new performance default was selected
from these measurements.

### macOS Metal CI coverage

The same successful run validates the Metal build, F16 stage parity and live
surfaces on an Apple M1 **virtual machine** (3 vCPUs, 7 GB). Its Apple Paravirtual
Metal device reports SIMD-group matrix multiplication unavailable. This is
useful Metal-path validation, not physical Apple GPU performance evidence.
With 4 inference threads, its medians are:

| Export | 11 s median / realtime | 55 s median / realtime | Peak process RSS |
|---|---:|---:|---:|
| F16 | 13.205 s / 0.83× | 35.467 s / 1.55× | 2,578 MiB |
| Q8_0 | 8.396 s / 1.31× | 15.167 s / 3.63× | 1,958 MiB |
| Q4_K | 8.404 s / 1.31× | 15.955 s / 3.45× | 1,653 MiB |

All 28 F16 stages pass (minimum cosine 0.999992, magnitude error bounded by
0.069%). Q8's warmed diagnostic uses Metal for the encoder and Accelerate for
the CPU decoder: encoder graph build/allocation 0.27/0.97 ms, versus 15.56 s
of instrumented encoder compute and 140 ms of decode. Callback overhead and
the virtual GPU prevent extrapolating these numbers to physical M1/M5 hardware
or comparing them against Fermion's 174× MLX result.

[The checked-in receipt](phonon2-profile-2026-09-30.json) retains every raw
benchmark time, transcript, per-stage cosine/magnitude bound, model checksum
and pinned revision. The linked CI run additionally retains full wiring/live
logs, per-node traces, host details and Python dependency versions. Both Linux
and macOS jobs passed; the earlier run's missing Python `sentencepiece`
dependency was corrected before this measurement.

## CPU optimization controls

`CRISPASR_PARAKEET_CPU_BLAS=1` selects OpenBLAS for the cached F32 predictor
LSTM and joint matrices, and batches the invariant encoder-to-joint projection.
It is available when OpenBLAS development files are present at configure time
and `CRISPASR_MEL_BLAS` is on. `=0` keeps the original Linux scalar decoder.
Apple continues to use Accelerate. `CRISPASR_PARAKEET_FORCE_SCALAR` preserves a
scalar fallback on either BLAS implementation, including the bulk projections.
Set BLAS threading before starting the process; the runtime does not change the
process-wide OpenBLAS thread count just to accelerate small decoder matvecs.

`CRISPASR_PARAKEET_ENCODER_BLAS=1` registers the ggml BLAS backend before CPU
in the encoder scheduler when running on CPU. Unsupported operations keep their
CPU kernels. This experiment requires a built/loaded ggml BLAS backend and stays
off by default; quantized matmuls can pay extra dequantization costs. The explicit
public thread count is now applied to both CPU backend instances and to the
optional encoder BLAS backend. The earlier four-thread receipt already matched
ggml's default of four; this wiring fix makes other requested counts effective.
Encoder caching remains off. The ggml BLAS backend sets its BLAS thread count
to the public thread count; with OpenBLAS that setting is process-wide and also
affects the decoder. The profiling receipt queries the actual thread count after
inference as well as recording the startup environment.

The CPU A/B workflow validates **31** frontend/encoder/transducer rows against
an independent Transformers F32 dump: the original 28 rows plus all encoder
projections, the raw predictor output after the production one-blank SOS, and
joint logits at frame zero using reference encoder activations. It checks cosine
and relative RMS error (which bounds global norm-ratio error). The legacy NeMo
two-zero predictor capture remains available separately. These probes do not
capture every autoregressive state; decoded-output checks remain required.

`.github/workflows/phonon2-cpu-ab.yml` compares scalar, OpenBLAS with one/four
BLAS threads, persistent ggml CPU decode, encoder BLAS and both BLAS paths.
Each runs in a separate process with warmed 11/55-second shapes and three timed
repeats; diagnostic traces run separately. No new default is justified until
same-runner timing and transcript receipts pass.
