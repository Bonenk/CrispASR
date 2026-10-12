# Draft release notes — changes after v0.8.43

The published release is documented in [v0.8.43 release notes](../RELEASE_NOTES_v0.8.43.md)
and [the tagged validation receipt](release-tag-v0.8.43-2026-10-12.json). No later
tag has been cut.

## Dolphin non-streaming decoding — PR #541 merged

Honor the checkpoint causal flag in both depthwise convolution sites and preserve
the original causal/noncausal LayerNorm/padding order. Register Dolphin in the
shared architecture table so automatic binding sessions can open its GGUFs.

Both non-streaming and streaming models pass Chinese/JFK original references:
64 stage comparisons, exact CTC frame argmax, exact original raw transcripts
and matching automatic C ABI/Python sessions. Independent checks cover all
128 complete native/reference arrays. The 12 original-module/native controls
and causal-only/normalization-order negative controls are also retained.
See [the full proof receipt](dolphin-noncausal-2026-10-12.json). All 45 applicable general CI checks pass, one intentional skip; the fix is
merged on main. Reporter Kylin/GCC9/private-clip and GPU validation are not
established.

## Breeze direct GPU initialization — PR #542 merged

Register dynamic backend plugins before selecting the Breeze compute device.
Direct library initialization now finds CUDA; verbose initialization reports the
actual compute backend. All 36 applicable PR checks pass, with two intentional
skips. The composition package exercises the identical initialization prefix on
Tesla T4: Breeze uses 6482/6924 MiB in stage/guided dumps, 107 original stage
comparisons pass, and all 20 speech/interface outputs remain byte-identical to
the prior real-CUDA run. The later branch-cache code differs between composition
and standalone builds; this is not a complete standalone-binary GPU proof.
See [the CUDA dispatch receipt](breeze-cuda-dispatch-2026-10-12.json).

## TTS session ownership, conversion rates and speaker boundaries — PR #531 merged

Rejected Breeze reference updates preserve the previous reference; replaced Moss
reference resources are released. Melo/OpenVoice conversions use the actual
input rate and return owned PCM at the converter model's output rate. Supertonic
direct speaker getters now safely reject null contexts and invalid indices.
Contributor history from #530 is preserved. All 48 applicable CI checks pass,
one intentional skip; the hosted native tier passes all 2,009 unit tests and
all 140 ONNX-enabled CLI/C ABI capability rows. Separate model-quality reports
in #530 remain unresolved.

## Pending integration — PRs #543 and #544

The TTS ownership/rate/speaker repairs now pass the complete ONNX capability
catalogue and native C ABI checks, plus all 2009 weight-free native tests.
Breeze's refreshed package additionally passes all 107 original stage/magnitude
comparisons and all 20 actual-T4 speech/interface roundtrips, with byte-identical
speech against the previous GPU run. Maximum ASR WER is 0.111111; speaker clone
similarity is 0.894997 versus the 0.527019 baseline. See [the refreshed package
receipt](breeze-cuda-integration-package-2026-10-12.json). Canonical complete
BF16 code equality remains 0/8 and is retained as a separate diagnostic;
fixtures, recipes, defaults and its comparator are unchanged. General CI and
integration are pending, so these features are not presented as merged.

The CANN follow-up exposes CLI backend/device selection and adds a separate
opt-in x86_64/910B and arm64/310P compile matrix for CLI, shared library, diff
and quantizer. Ten production-source routing controls pass, with four failing
against the previous helper. The x86_64/910B build passes all four targets and ggml-cann; ARM64/310P and general CI remain pending. Hosted compilation cannot establish Ascend inference or speed.

## Work excluded from release acceptance

PRs #492, #521, #529 and #536 remain pending their documented model, quantization, generation or hardware gates. The additional model-quality reports recorded in merged contributor PR #530 remain unresolved. Breeze's actual-CUDA strict
guided generation remains 0/8 against canonical BF16. Original-only arithmetic
controls reproduce some complete native grids: joint F32 matches 1/8 CPU and
3/8 CUDA; F16 KV rounding matches 2/8 CPU and 3/8 CUDA. These diagnostics do not
change fixtures or acceptance gates. The completed full-inference F32 original
control matches 5/8 CPU and 5/8 CUDA grids; canonical BF16 remains 0/8.
See [the full-inference arithmetic receipt](breeze-guided-full-inference-arithmetic-2026-10-12.json).
See [the KV arithmetic receipt](breeze-guided-kv-arithmetic-2026-10-12.json).
Canary CUDA allocation/reuse controls do not reproduce #532 and do not establish
original Q4 parity. No new default, quantization format or performance claim
follows from these diagnostics.
