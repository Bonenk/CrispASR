Checkpoint update: build38874191736 passes; independent audit verifies57
payloads and12 fresh graph replays using retained actual static libraries.
Lint/general CI remain queued. Pinned full-generation recovery is now accepted
within [its explicit scope](dia-serial-head-results-2026-10-16.md); current-main
speech remains a separate pending test. Earlier submission status below is
retained as preparation history.

# Dia current-main serial head — preparation

Candidate [d198d7a4](https://github.com/CrispStrobe/CrispASR/commit/d198d7a48742c8424444d9b67dde9eb5d26f590f)
adds an opt-in CPU F32 head graph node to current main, independently of the
still-running pinned-runtime full-generation experiment. Production defaults
are unchanged. [Local receipt](dia-current-main-head-preparation-2026-10-16.json).

`CRISPASR_DIA_SERIAL_HEAD` accepts `off`, `identity` and `serial`; absent/empty
means OFF. OFF returns the original graph tensor. Identity copies native logits
through the same custom-node topology; serial uses actual native weights and
current hidden, separately rounded F32 products and sequential F32 addition.
No numerical branch depends on a token or generation position. Compute/apply
functions remain byte-exact to the tested immutable causal candidate3c2fbb5a.
The ordinary matmul remains a dependency, so this intentionally pays both paths
and makes no performance claim. Opt-in modes reject GPU/non-F32/invalid head
shapes before the encoder. No model or external reference is injected.

Local proof builds actual ggml0a862076 static libraries. The30-check registered
CMake test exercises real tiny-tensor graphs, two batches, one/four threads,
identity, an exact fusion witness and admission rules. A separately compiled
copy with its GPU guard removed must fail at that guard's test, and does.
Captured-input replay matches all18,504 serial outputs on each of two hidden
states to the published original-compatible reference, exactly at both thread
counts. Native/OFF/identity/thread comparisons pass. The complete current Dia
runtime passes syntax compilation. Source/ELF/library/array hashes, commands
and logs are retained under:
/mnt/storage/crispasr/dia-main-head-20261016/local-v2/preflight-v2.
The preceding local-v1 shared-library build failed at CIFS symlink creation;
that failure remains intact. Static libraries avoid that filesystem constraint.
Clang-format18, Python compilation and all4 unskipped no-secrets checks pass.

Hosted [build/replay38874191736](https://github.com/CrispStrobe/CrispASR/actions/runs/38874191736),
[lint38874302965](https://github.com/CrispStrobe/CrispASR/actions/runs/38874302965)
and [general CI38874305718](https://github.com/CrispStrobe/CrispASR/actions/runs/38874305718)
are submitted at d198d7a4 and queued at this checkpoint. The build worker compiles
the actual current `dia-tts` library and CMake test target, runs ctest and the
captured-input replay, and retains static libraries/sources. Its single raw
collector session10831 preserves official complete metadata/logs/artifact and
verifies the download digest. Complete artifact auditing remains pending.
No full neural model runs in this validator.

Current-main migration must keep these differences visible:

| Boundary | Pinned causal experiment | Current candidate |
|---|---|---|
| ggml | c36dab89 | 0a862076; changes between these pins are CANN-only |
| Cross K/V | Explicit F32 diagnostic cache | Existing F16 cache |
| Sampling randomness | Verified portable exponential batch | Existing std::discrete_distribution |
| Top-k | Historical probability-threshold implementation | Current exact selected logit indices |
| Stop/output bookkeeping | Historical runtime | Current repaired capacity/countdown/crop protocol |
| RoPE | Source alternative disabled | Existing ggml RoPE |
| Speech proof | Full-generation run38870499727 pending | No new full-generation result |

The candidate corrects stale CFG comments to match both the implemented
`cond + scale*(cond-uncond)` expression and the pinned original; the expression
itself is unchanged. CLI and Session/C-ABI call the same synthesis library, so
there are no signature/registry/converter/quantizer changes. Actual end-to-end
CLI/Session checks remain required; shared dispatch alone proves no speech.

After the pinned causal result, run current-main baseline versus head-only at
fixed actual recipe, with pristine/OFF/identity controls, native/original DAC
and same-recognizer positives. Preserve current top-k/output repairs. Add cache
or reference-RNG diagnostic paths only if the observed results justify them.
Then freeze bounded additional prompts/seeds and their original positives
before quality comparison. Actual GPU/Q4 and profiling follow floating-point
speech acceptance. Serial rounding can restore one fragile decision without
establishing general robustness; no production promotion follows from this
preparation.
