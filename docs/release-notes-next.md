# Draft release notes — changes after v0.8.43

The published release is documented in [v0.8.43 release notes](../RELEASE_NOTES_v0.8.43.md)
and [the tagged validation receipt](release-tag-v0.8.43-2026-10-12.json). No later
tag has been cut.

## Dolphin non-streaming decoding — PR #541 pending final CI

Honor the checkpoint causal flag in both depthwise convolution sites and preserve
the original causal/noncausal LayerNorm/padding order. Register Dolphin in the
shared architecture table so automatic binding sessions can open its GGUFs.

Both non-streaming and streaming models pass Chinese/JFK original references:
64 stage comparisons, exact CTC frame argmax, exact original raw transcripts
and matching automatic C ABI/Python sessions. Independent checks cover all
128 complete native/reference arrays. The 12 original-module/native controls
and causal-only/normalization-order negative controls are also retained.
See [the full proof receipt](dolphin-noncausal-2026-10-12.json). General PR CI
must finish before merge; reporter Kylin/GCC9/private-clip and GPU validation
are not established.

## Work excluded from release acceptance

PRs #492, #521, #529, #530, #531 and #536 remain pending their documented
model, quantization, generation or hardware gates. Breeze CUDA speech/stage
validation is in progress; successful speech alone does not waive strict
generation failures. Canary CUDA allocation/reuse controls do not reproduce
#532 and do not establish original Q4 parity. No new default, quantization
format or performance claim follows from these diagnostics.
