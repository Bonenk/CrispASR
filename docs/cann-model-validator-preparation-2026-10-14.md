# Full-model CANN validator preparation — 2026-10-14

Candidate `168384d47ba462ae72858042ffc13188b143cca1`, branch
`diag/cann-model-validator-20261014`, adds `tools/validate_cann_model.py`.
It is awaiting full hosted interface validation and is not integrated.

The runner consumes the immutable accepted108,495,214-byte Moonshine fixture
archive (SHA256 `7d63d2a5b90eb77843aff25107b1d1f70bafcfc1c8f73075e2f5a9a3f87d5e84`)
and a supplied Linux build. It checks archive identity and extraction paths,
original PCM/reference shape and arrays, snapshots actual native binaries and
libraries by SHA, retains CMake cache/compilation commands and ldd output, then
runs a real explicit CPU control before requested CANN execution. Each arm
requires encoder diff, repeated automatic/explicit C ABI Sessions, repeated
encoder arrays and two CLI transcriptions. All encoder values must pass the
existing cosine/row-norm/relative-L2 gates; CANN also compares directly to the
paired CPU output. Transcripts must reproduce the original words, with raw
text and byte-equality results retained. Output directories must be fresh;
partial receipts, complete logs and snapshots survive failure.

CANN acceptance requires successful exact-device initialization and allocated
encoder matmuls on that same device, followed by successful computation. Legitimate
unsupported-operation CPU fallback remains visible. Missing/invalid devices,
zero or partial execution evidence, mismatched devices, bad output or uncompiled
CANN cannot pass. Device metadata from npu-smi is retained when available;
checkout identity is explicitly the validator identity and is not asserted to
prove that supplied ELF bytes were compiled from it. Build logs/source provenance
remain a separate requirement when auditing returned hardware results.

The original native dumper forced CPU for its public encoder call even when
Sessions used GPU. New explicit GPU and encoder-only flags let the public
stage call reach the selected NPU; defaults preserve the previous CPU behavior.
Flags requiring native mode and incomplete native argument sets now fail at
argument parsing. Eight weight-free false-success/archive/argument guards pass,
and are wired into ordinary Linux CI.

The bounded actual unchanged Moonshine backend shared probe passes both CPU
GPU-flag settings, returning complete repeated457×288 arrays byte identical to
the accepted CPU fixture. With a CANN requirement, GPU=false rejects the
GPU-disabled contradiction; GPU=true reaches actual device lookup and rejects
the unavailable CANN registry. This proves argument forwarding, not NPU work.
It does not exercise the complete shared C ABI Session/CLI runner.

[Complete30-payload native flag proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/8b16f924c00fc285001f12060b7a24945708d9b8/cann-20261014-model-validator/complete-native-gpu-flag-controls.zip) includes actual source,
shared probe, full build/library-closure logs, arrays, guard results and method
ledgers. Anonymous full download verifies 2,208,646 bytes, SHA256
`de8965041d6e8cf88659c75ff676b243f09581fe31adc698b7e03defdaa0a7ea`. [Structured preparation receipt](cann-model-validator-preparation-2026-10-14.json).

[Hosted full interface run38602252235](https://github.com/CrispStrobe/CrispASR/actions/runs/38602252235)
will exercise the exact command through genuine CPU diff/CLI/auto+explicit
Sessions, require CANN rejection after CPU controls, and independently exercise
GPU=true public encoder initialization on the CPU host.
[General CI38602256898](https://github.com/CrispStrobe/CrispASR/actions/runs/38602256898)
is also submitted. Both remain pending.
[Official SDK matrix38598516003](https://github.com/CrispStrobe/CrispASR/actions/runs/38598516003)
is now in progress. No hardware procedure is published before interface proof;
physical NPU, quantized speech quality, ACL capture/replay and performance remain
unvalidated. All heavy local files stay under
`/mnt/storage/crispasr/cann-model-validator-20261014`.
