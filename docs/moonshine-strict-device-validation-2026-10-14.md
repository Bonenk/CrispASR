# Moonshine strict device validation — 2026-10-14

Candidate `9e2dd0116dd9f9d8b008b22b4d14e9a91757f61d` is on
`diag/cann-model-strict-20261014`, in the separate fixture worktree. Production
still uses the previously accepted CPU fixture; this candidate is not integrated.

The opt-in `CRISPASR_MOONSHINE_REQUIRE_BACKEND` requests an exact ggml registry
(case-insensitive, with metal→MTL alias). `CRISPASR_MOONSHINE_REQUIRE_DEVICE`
selects a nonnegative device ordinal within that registry; default0. Missing,
invalid or unavailable devices and non-CPU requirements with GPU disabled fail
instead of choosing auto/CPU. Shared Moonshine init covers CLI, C ABI Sessions
and public stage APIs without changing ABI structure sizes. Default automatic
selection stays available. The candidate also uses backend type rather than
pointer inequality to identify CPU instances.

In strict mode, the encoder reads scheduler assignments after graph allocation,
requires matmuls on the selected backend and no unassigned nodes, then records
per-backend operation counts and compute success. It adds no eval callbacks,
intermediate output marks or graph changes. CPU fallback for unsupported
individual GPU operations remains legitimate and visible; selecting a GPU with
no encoder matmuls assigned to it cannot pass validation.

The Moonshine diff dispatch activates the requested GPU path, checks exact
reference PCM, refuses missing/empty encoder references, checks full encoder
shape/count and enforces row norm and relative L2 error limits of1%, alongside
the unchanged cosine threshold0.999. Magnitudes are printed for passing stages
too. These are floating-point fixture gates, not accepted quantized-model limits.

Bounded actual native CPU replay passes normal and strict CPU arms. Complete
encoder arrays and both transcripts per arm are byte identical to the accepted
backend proof. Strict CPU logs39 encoder matmuls, zero unassigned nodes and
successful computation. Contradictory GPU-disabled, invalid-index and unavailable
CPU-index model opens fail. Thirteen actual selector/scheduler helper checks pass,
including rejecting an allocated ADD-only graph as model work. The complete diff
translation unit also passes syntax checking with its actual hosted build flags.
This bounded proof does not replace full CLI/C ABI validation.

Hosted full model controls: [38575701613](https://github.com/CrispStrobe/CrispASR/actions/runs/38575701613).
The worker reruns original conversion/dumper/generation and normal CPU checks,
then strict diff/CLI/auto+explicit Sessions and deliberately scaled, nonfinite,
wrong-shape, missing-stage and changed-PCM references. Missing-CANN and invalid
or unavailable device cases must fail through the real entry points.
General CI: [38575706459](https://github.com/CrispStrobe/CrispASR/actions/runs/38575706459).
Lint: [38575711575](https://github.com/CrispStrobe/CrispASR/actions/runs/38575711575).
Their acceptance is pending. No physical NPU, ACL capture/replay, quantized
quality or performance result is claimed. No unexecuted hardware procedure is
published. All build/run/evidence files remain under
`/mnt/storage/crispasr/cann-model-strict-20261014` or hosted scratch.

The bounded probe also compiles the verbatim actual Moonshine dispatch and
Ref implementation. Good reference passes; doubled scale, NaN and wrong shape
return6; changed PCM and missing encoder reference return4. This uses a raw PCM
loader, so it does not prove the full application audio reader or Session path.
It caught an initially misplaced magnitude edit before the queued model run
executed. The edit is corrected; superseded hosted submissions38574829573,
38574834564 and38574839175 were cancelled, with the observed local failure
retained. No measured model failure was discarded or acceptance gate loosened.

[Complete67-payload bounded proof](https://huggingface.co/datasets/cstr/crispasr-regression-fixtures/resolve/1caca74803027ac1b249250b213617970612b061/cann-20261014-model-strict/complete-local-strict-validation.zip) contains actual probes, source, executables, arrays, build/run logs and failed-before/corrected controls. Anonymous full download verifies 6,572,420 bytes, SHA256 `686cc4333f2bd3671846a384c79d0aa2d46b9f4551df15a86b38801e996ff0f0`. CPU libraries/model remain in their already published immutable bundles. [Result receipt](moonshine-strict-device-validation-2026-10-14.json).
