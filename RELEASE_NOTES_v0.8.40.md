# CrispASR v0.8.40

Intel macOS CPU performance, Phonon-2 speech recognition, long-audio AudioSeal
watermarking, and faster VoxCPM2 CPU prompt processing.

## Intel macOS CPU builds

The standard `crispasr-macos-x86_64.tar.gz` now enables AVX2, FMA and F16C
explicitly, with Apple Accelerate and Metal disabled. v0.8.39's portable build
turned these kernels off, causing the slowdown reported in
[#484](https://github.com/CrispStrobe/CrispASR/issues/484). Builds still disable
`-march=native`, so their instruction requirements do not depend on the runner.

For older Intel Macs without AVX2/FMA/F16C, use the new
`crispasr-macos-x86_64-cpu-legacy.tar.gz`. Apple Silicon downloads keep their
existing filenames and Metal support. Intel archives include the CLI,
quantizer, matching C2PA sidecar and license notices; release checks extract
and start both executables with the build-tree libraries hidden.

Intel validation compares identical source and model weights under both ISA
settings. F16/Q8/Q4 short transcripts match their normalized golden words,
and the Q8 long fixture preserves the same words in both builds. On a hosted
Intel i7-8700B, alternating warmed calls showed 1.50-1.93x speedups for a
66-second repetition fixture. Host timings varied; this is not a prediction
for the reporter's 455-second Russian audio. See
[measurements and limitations](https://github.com/CrispStrobe/CrispASR/blob/v0.8.40/PERFORMANCE.md#intel-macos-release-isa--2026-10-01-484).

## Phonon-2 English speech recognition

Phonon-2 now works through the CLI, C ABI and language bindings using the shared
Parakeet engine, with model conversion, automatic model selection, metadata
recognition and documented controls
([#481](https://github.com/CrispStrobe/CrispASR/issues/481)). Q8_0 is the
recommended model. Phonon-2 is English-only; its metadata preserves this
restriction even when a model file is renamed.

Native AVX2/F16C CPU builds use persistent predictor/joint graphs and a bulk
encoder projection. On a four-vCPU AMD EPYC runner, Q8 warmed inference fell
from 2.439 to 1.782 seconds for 11 seconds of speech, and from 13.000 to 9.856
seconds for a 55-second repetition fixture. F16/Q8/Q4 corpus outputs were
preserved across the tested decoder paths. Apple retains Accelerate and other
Parakeet models retain their existing decoder defaults.

FFN repacking and cached BLAS experiments remain opt-in. Q4 stage differences
and short-clip regressions prevent promoting those experiments. See
[Phonon-2 validation and controls](https://github.com/CrispStrobe/CrispASR/blob/v0.8.40/docs/phonon2.md)
for the measured scope and receipts.

## AudioSeal long-audio watermarking

Embedding or detecting watermarks on audio longer than approximately two
seconds could abort at ggml's graph-node limit. Graph and scheduler capacity
now scale with audio length
([#482](https://github.com/CrispStrobe/CrispASR/issues/482)). CPU tests and a
physical NVIDIA T4 CUDA test cover 1/4/10/1-second repeated-call sequences;
short-clip watermark output remains unchanged.

## VoxCPM2 native CPU RALM prefill

RALM prompt processing now batches the causal prefix for native AVX2/F16C CPU
models whose RALM matrices use F16 or Q8_0, complementing the TSLM batching in
v0.8.39 ([#478](https://github.com/CrispStrobe/CrispASR/issues/478)). The batch
populates the existing host and backend KV caches and preserves the eager RMS
and activation arithmetic.

At 249 positions on a four-vCPU Intel runner, warmed Q8 RALM prefill improved
from 4116.5 to 899.2 ms (4.58×), and F16 from 6159.5 to 3549.8 ms (1.74×).
These are prefill-stage measurements, not whole-synthesis speedups. State,
KV, decode-continuation, causal isolation, context reuse and synthesized
speech checks pass; default speech PCM matches the selected explicit path.

`CRISPASR_VOXCPM2_RALM_PREFILL_BATCH=0` restores eager prefill. Q4 batching
remains opt-in with `1`: it is faster but fails strict state parity and produced
extra trailing words in one speech check. Other CPU instruction sets and GPU
prefill keep their existing defaults.
