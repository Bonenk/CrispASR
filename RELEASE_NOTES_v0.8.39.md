# CrispASR v0.8.39

Separate Intel and Apple Silicon macOS downloads, faster VoxCPM2 prompt
processing and Vulkan VAE decoding, safer tensor-type overrides, and stronger
credential and CI checks. This release follows v0.8.38 with fixes for users
blocked by architecture errors or quantizer crashes.

## macOS downloads for Intel and Apple Silicon

Intel Macs now have an official CLI package, addressing the "Bad CPU type in
executable" error reported by Subtitle Edit users in
[#479](https://github.com/CrispStrobe/CrispASR/issues/479).

| Machine | Asset | Acceleration |
|---|---|---|
| Apple Silicon | `crispasr-macos-arm64.tar.gz` | Metal + Apple Accelerate |
| Intel Mac | `crispasr-macos-x86_64.tar.gz` | CPU + Apple Accelerate; Metal disabled |

Each archive contains `crispasr`, `crispasr-quantize`, a matching
`libc2pa_c.dylib`, and license notices. The Intel build uses the portable CPU
baseline rather than tuning instructions to the build machine. Both jobs set
their target architecture explicitly.

The existing `crispasr-macos.tar.gz` remains an **arm64 compatibility alias**,
including its original `crispasr-macos/` directory. Downloaders that support
Intel Macs must select `crispasr-macos-x86_64.tar.gz`; the legacy filename is
not universal. The architecture-specific archives measured approximately
17.2 MB for arm64 and 18.0 MB for Intel in release qualification.

Also fixed an existing macOS quantizer packaging defect: `crispasr-quantize`
could find C2PA inside the build checkout but failed after extraction elsewhere.
It now receives the same bundled-library search path as the CLI. The release
gate checks all three Mach-O architectures and starts both executables from an
extracted archive with the original build libraries hidden.

## VoxCPM2: faster prompt processing and correct thread selection

Graph-mode synthesis now processes the TSLM prompt once, in one batched causal
graph, instead of running a legacy prefill and replaying the prompt through
individual graph steps. The resulting hidden states and backend KV cache feed
the existing synthesis path. Eager matmuls now honor `-t` and
`voxcpm2_set_n_threads()` rather than remaining fixed at four threads
([#478](https://github.com/CrispStrobe/CrispASR/issues/478)).

A four-core CPU runner measured the following for a 62-position voice-clone
prompt, with the requested speech recovered by ASR in every arm:

| Measurement | Legacy path | Batched graph path |
|---|---:|---:|
| TSLM prefill | 2,496.5 ms | 1,321.1 ms |
| Main synthesis call | 39.54 s | 35.99 s |

These are synthesis-stage measurements, excluding the separate spoken AI
disclaimer generated for cloning. They are not a prediction for every CPU or
prompt length. RALM prefill still runs per position.

Fallbacks remain available: `CRISPASR_VOXCPM2_PREFILL_SERIAL=1` uses individual
graph calls, and `CRISPASR_VOXCPM2_LEGACY_PREFILL=1` restores legacy prefill plus
replay. Failed batched allocation or computation falls back to the serial path.

## VoxCPM2: less VAE setup work and faster Vulkan decoding

The VAE startup path now uses a tiled ConvTranspose1d weight permutation,
reconstructs independent weight-normalization rows in parallel where OpenMP is
available, and reuses host weights instead of downloading them back from the
GPU just to permute them. The shared transpose helper is checked byte for byte
against the previous implementation for F32, F16, and uneven tile dimensions.

On Vulkan, causal depthwise VAE convolutions now default to shifted
multiply-adds instead of the im2col path. On the reporter's Intel Arc B390,
the VAE stage, including setup, fell from about 1.69 seconds to 0.30–0.32
seconds. This is a **VAE-stage improvement**, not a fivefold improvement to the
whole synthesis call. The diffusion head remains a substantial part of total
runtime ([#461](https://github.com/CrispStrobe/CrispASR/issues/461)).

`CRISPASR_VOXCPM2_VAE_DW_SHIFT=0` restores the previous convolution path.
The new path defaults on for Vulkan only; CUDA and Metal keep their existing
default. VAE benchmark output now separates weight setup, graph allocation,
and computation so startup costs are visible.

Mixed Q8_0/F16 LocDiT weights were also evaluated for Vulkan. At the same
eight diffusion steps on the B390, the reported real-time factor improved from
1.10 with Q8_0 to 1.01 with F16 LocDiT weights. This is an optional model-weight
choice; it does not change the default diffusion-step count. Model producers
can use `--tensor-type '^locdit\.=f16'` with the corrected quantizer below.

## Quantizer: preserve scalar and one-dimensional tensor precision

`--tensor-type` now skips overrides below F32 for tensors with fewer than two
dimensions. A broad override such as `^locdit\.=f16` previously also lowered
biases and normalization weights to F16, creating a model that aborted during
CPU synthesis when ggml added F16 data to F32 activations. Vulkan accepting
that file had hidden the defect.

Eligible matrices still take the requested override, and explicit F32
overrides remain supported. A small, model-free GGUF regression test covers
matrix conversion, bias/norm preservation, and explicit F32 overrides.

## Security, CI, and maintenance

- Removed embedded Kaggle API credentials from tracked tooling; the owner
  reported revoking the exposed credentials. Tooling now uses local credential
  configuration or CI secrets. Kaggle account and token configuration is
  supplied through GitHub secrets instead of hardcoded account metadata.
- Added a credential scan on every push and pull request, covering supported
  token formats and private-key blocks, with positive controls for its patterns.
- Heavy verification workflows now check a branch's own changes against its
  merge-base with main. Rebasing onto unrelated main changes no longer launches
  duplicate expensive checks. An unavailable or failed scope check runs the
  verification rather than silently skipping it.
- Added a manually dispatched heavy CPU workflow for reference dumps,
  conversion, quantization, parity checks, and ASR roundtrips on GitHub runners,
  plus VoxCPM2 mixed-quantization and prefill A/B scripts.
- Updated platform download guidance, performance records, and the generated
  learnings index. Completed work moved from the active plan to history.

## Validation

The release's runtime changes passed the
[main CI matrix](https://github.com/CrispStrobe/CrispASR/actions/runs/36715787407),
including Linux unit tests, Windows, macOS, Android, iOS, Vulkan, audio ASan,
and fuzz smoke checks. Both macOS CLI jobs passed their native architecture
and relocated-package startup checks in the
[release dry run](https://github.com/CrispStrobe/CrispASR/actions/runs/36713861682).
The old macOS alias was also checked for identical payload hashes and file
permissions. VoxCPM2 CPU zero-shot and cloned speech passed ASR checks across
legacy, serial graph, and batched graph paths in the
[CPU A/B run](https://github.com/CrispStrobe/CrispASR/actions/runs/36642038819).
The mixed Q8_0/F16 model also passed an exact CPU ASR roundtrip in the
[mixed-precision check](https://github.com/CrispStrobe/CrispASR/actions/runs/36605906347).

[All changes since v0.8.38](https://github.com/CrispStrobe/CrispASR/compare/v0.8.38...v0.8.39).
