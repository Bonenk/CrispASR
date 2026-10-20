# Issue follow-ups — 2026-10-20

## License disposition

[Policy](model-license-policy.md): ordinary NC models remain eligible behind
explicit acceptance. Research-only and additional use/output/revenue/hosting
conditions require separate review. Do not silently treat code and weight
licenses as interchangeable. [Source hashes and correction receipt](license-source-audit-2026-10-20.json).

The exact pinned OmniVoice source already states NC main weights and includes
the Boson tokenizer terms. The public converted-model card is corrected; no
weights were modified. The registry now gates its bundle separately, and
OpenRAIL requires explicit acceptance. Registry controls fail in three cases
before the fix and pass 6,224 assertions in 63 cases afterward. Full bundle
promotion and PR521 remain pending license review and unchanged speech gates.

## Issue550

Version2 reached two T4 GPUs but failed before inference: no CPU backend was
registered by the frozen package from a foreign Python cwd. Complete terminal
logs and partial output were retained. Version3 explicitly loads the verified
bundle's plugins, checks CPU and CUDA0 before model download, and keeps identical
runtime/model/input pins and acceptance gates. It completed and passes an independent audit: all 147/8,925/147 units are
retained over 44.52/2,703/44.52 seconds, returning-short output is exact after
offset adjustment, CLI matches public output, and all 95 encoder splits are
on CUDA0. The long-phase sampled device total peaks at 2,735 MiB. Three forged
controls with recomputed hashes (missing unit, CPU placement, changed return
offset) are rejected. [Audit](alignment-cuda-live-2026-10-20.json).
This is not natural conversational alignment accuracy, constant host memory or
RTX3090 certification. Three loader protocol
controls pass. The local Kaggle CLI launcher was also repaired and tested
against a package exposing the CLI without a package `__main__`.

## Shared ggml PR2 review

Exact proposal head: `2ef4ee11c73e3a4186c4e2b1e8b522472de787ed`. The lowering
was ported onto current shared source `e2ae679a`; only CMake insertion points
needed adjustment. A fresh Metal-enabled build with the option ON links.
The actual graph toggles between CONV_2D and the legacy path; 100 CPU and
100 Metal cases pass a scalar convolution oracle, including F16/F32, batch1/3,
kernel1/3/5, stride/dilation1/2 and large activations. Full build and per-case
logs are retained; four additional non-contiguous CPU fallback controls pass
with the option both ON and OFF. [Receipt](ggml-conv1d-pr2-review-2026-10-20.json).
These synthetic controls establish no AMD Vulkan execution,
Chatterbox speech quality or performance result. Incompatible-type and non-CPU layout
fallback, current Vulkan compilation, actual Radeon finite/magnitude/cancellation
controls and decoded audio remain acceptance gates. No ggml change is merged.
The reusable source is `tools/ci-heavy/conv1d_pr2_probe.cpp`; it deliberately
rejects a build where the option does not actually alter the graph.

## Issue549

Final token-boundary source `ebf69c437` passes CI39117921894 and
lint39117923785. The complete Linux unit log was independently inspected:
2,033 CTest cases and both real exported-getter metadata suites pass, including
Japanese slice/gap-fill and tied-frame ownership. The actual Japanese Metal Session short/long/short passes with 17/123/17
tokens and exact returned-short metadata. The independent direct native short
route matches every token byte, time and probability. The initial strict time
ordering guard incorrectly rejected a native separator-only token retaining an
earlier frame; it now preserves those times while rejecting content inversions
and missing owned tokens, with three controls. [Live receipt](ja-session-metadata-live-2026-10-20.json).
Generic CLI postprocessing, published wheels and downstream reporter acceptance
remain separate. For CPU use, the worker requires a CPU-only build: the direct
Parakeet wrapper follows its native default device.

## Reporter actions

- [Issue485](https://github.com/CrispStrobe/CrispASR/issues/485#issuecomment-6291292498)
  closed completed: original Index-Echo request shipped in v0.8.41. Rejected
  quantization and unmeasured hardware are not promoted by that closure.
- [Issue545](https://github.com/CrispStrobe/CrispASR/issues/545#issuecomment-6291304859)
  and [issue546](https://github.com/CrispStrobe/CrispASR/issues/546#issuecomment-6291305573)
  closed not planned under the current Fish revenue/hosting conditions.
- [Nyra456](https://github.com/CrispStrobe/CrispASR/issues/456#issuecomment-6291353434)
  remains deferred for separate research/output-license review.
- [PR521](https://github.com/CrispStrobe/CrispASR/pull/521#issuecomment-6291353763),
  [issue550](https://github.com/CrispStrobe/CrispASR/issues/550#issuecomment-6291354113)
  and [ggml PR2](https://github.com/CrispStrobe/ggml/pull/2#issuecomment-6291304497)
  received scoped evidence and remaining-gate updates.
- Issue483 still needs GTX1660/MX150 evidence; issue532 needs the reporter's
  failing recording/assertion/device details. No speculative fixes or duplicate
  requests were posted.

[Issue550 acceptance/closure](https://github.com/CrispStrobe/CrispASR/issues/550#issuecomment-6291711315)
and [issue549 live follow-up](https://github.com/CrispStrobe/CrispASR/issues/549#issuecomment-6291711992)
are posted. A hosted CPU/software-Vulkan worker for PR2 and bounded native
package dry runs are prepared for the frozen follow-up source.

No new release is published by this checkpoint. Release/package acceptance and
remaining model/hardware tests must be completed before advertising them.

## Complete public proof

[Immutable archive](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/ad638b4cd525c5633f831da9eb763864e9827ce9/issue-followups-20261020/complete-proof.zip)
contains 55 files, 420,650 bytes; SHA256
`e446e1a4b2a3c925033598677d3555558f35470d5957d058d0c4c37e9731c70e`.
The full anonymous download is byte-verified. Numerical/word outputs, complete
logs, hashes, negative controls and licensed fixture attribution are retained.
No model weights, runtime binaries or credentials are republished. Text path
prefixes/account identifiers use portable aliases; the manifest records both
original and public-file hashes. The successful worker also appends its terminal
progress event after hashing that file; its original hash covers the verified
prefix, and the public manifest covers the complete retained file.

[Three bounded library dry runs](followup-package-dry-runs-2026-10-20.json) are
dispatched on frozen native source `f0199f54f`: Linux39429480076,
Mac39429485161 and Windows39429489355. Their tag inputs are empty, so they
upload workflow artifacts and cannot publish a release. They were queued at
this checkpoint. The later receipt/tooling changes do not modify native library
or wheel runtime sources; package claims still require reading actual results
and inspecting staged artifacts.
