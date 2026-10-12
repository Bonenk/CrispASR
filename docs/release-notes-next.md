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
against the previous helper. Current-source x86_64/910B (ACL graph on) and
ARM64/310P (ACL graph off) builds both pass all four targets and ggml-cann.
See [the complete compile receipt](cann-pr544-compile-2026-10-12.json). General
PR CI and integration remain pending. Hosted compilation cannot establish
Ascend inference or speed.

## Dia reference and short-speech fixes — validation in progress

The active Dia branch preserves aligned audio frames during delay reversal,
zeroing invalid individual codebook entries to match the pinned original
method. Eight complete code-grid regressions pass; five fail before the fix.
The reference/diff integration observes original generation and real DAC calls,
checks exact input boundaries plus stage cosine and magnitude, and rejects
missing or uniformly scaled reference stages. Input validation rejects
unaligned whitespace/NUL before model loading; four tiny-original capture
controls pass. These small tests establish boundary and capture behavior.

[Actual-weight numerical/codec validation](https://github.com/CrispStrobe/CrispASR/actions/runs/38278884937)
and [complete native F16/Q8/Q4 speech validation](https://github.com/CrispStrobe/CrispASR/actions/runs/38272642504)
are still in progress. The reported short-prompt silence is not yet declared
resolved, and the runtime changes are not merged or included in release
acceptance. Decoder-history and DAC-code replay do not establish free-running
sampler equality.

## Work excluded from release acceptance

PRs #492 and #521 remain pending their documented model, quantization or
hardware gates. Contributor PRs #529 and #536 are incorporated in the validated
Breeze composition awaiting #543 integration. The additional model-quality
reports recorded in merged contributor PR #530 remain unresolved. Breeze's actual-CUDA strict
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
