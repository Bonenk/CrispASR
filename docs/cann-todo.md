# CANN remaining work

Updated2026-10-14. PR544 is merged; its requested integration is complete.
Two subsequent device-reported defects have narrow repairs with retained
before/after evidence. The 310P BF16 capability repair is integrated. The
910B rotary-capacity repair has physical and SDK proof; final general CI
and production integration are still pending.

The maintainer owns these follow-ups. Contributor hardware help is optional;
there is no outstanding rerun request for the now-passing rotary candidate.
A final integration result will be posted on PR544. Contributors can
unsubscribe from the merged PR; this document is the durable follow-up list.

| Order | Task | Current state / acceptance | Owner / hardware need |
|---:|---|---|---|
| 0 | Integrate rotary byte-capacity repair | Physical910B3/SDK9.1:1691/1691 broad plus452/452 repeat. Both SDK9.2 builds pass. General CI38541851377 active; merge exact tested pin only after all required jobs pass. | Maintainer; no further device rerun currently requested |
| 1 | Prepare a small full-model ASR validation bundle | Pin supported model, original Python reference, fixture and expected transcript. Prove CPU pipeline and C ABI/CLI controls on local/GH before publishing executable hardware commands. | Maintainer prepares; optional310P3/910B3 execution afterward |
| 2 | Prove NPU full-model parity and decoded output | Same model/input/config on CPU and CANN; crispasr-diff stages include magnitude/error as well as cosine. Actual CLI and Python Session transcripts must pass declared quality gates. Preserve scheduler/device traces to prove real NPU work. Not yet validated. | Maintainer audits; physical NPU needed |
| 3 | Verify actual scheduler fallback for rejected310P BF16 cases | Current predicate/operator proof excludes unsupported cases; demonstrate supported CANN plus rejected CPU operations in a composed graph with correct output. Not yet validated. | Maintainer prepares CPU controls; physical310P needed |
| 4 | Verify ACL graph execution and reuse | Current910B build enables ACL graphs; operator logs do not establish executed capture/replay. Fixed-output graph-on/off comparison, repeated calls and growing/changing shapes; retain capture/replay evidence. Not yet validated. | Maintainer prepares; physical910B needed |
| 5 | Validate shipped quantized inference | After floating-point model acceptance, test relevant F16/Q8/Q4_K recipes against reference stages and decoded speech. Keep sensitive tensors where measured necessary; no tolerance waivers. Not yet validated. | Maintainer prepares; NPU model runs needed |
| 6 | Profile and optimize | Only after output acceptance: prove offload/fallback, time load/encoder/decode separately, warm up and use repeated medians under identical load. Gate changes; compare quality and speed before changing defaults. No NPU performance claim yet. | Maintainer analyses; physical NPU timing needed |
| 7 | Maintain coverage and release documentation | Keep lightweight regressions, manual official SDK matrix and complete device receipts. Add model/device evidence when available; release only after applicable CI and acceptance gates pass. | Maintainer; hosted CI covers compilation only |

Evidence: [310P BF16 integration](cann-310p-bf16-integration-2026-10-13.json),
[910B rotary physical proof](cann-910b3-rope-physical-2026-10-14.md),
[rotary SDK proof](cann-rope-sdk-2026-10-14.md),
[hardware protocol](cann-hardware-validation-2026-10-12.md).
Physical tests preserve unsupported cases, prior failures and unchanged
upstream tolerances. Passing operator suites alone cannot accept a model.
