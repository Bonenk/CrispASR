# OmniVoice runtime and acceptance audit — 2026-10-22

Audit inputs: remote main `22e1b6ada4cc8eced45f30bf967f614f19f360fe`
and codec branch `f76d4b190a109ba17b6854a71bcd0e3ae6dc4b22`.
PR #521 was already merged on main; the four codec optimization commits were
still on `perf/omnivoice-codec-decomp`. Older green branch checks do not certify
its latest source.

## Findings and disposition

- **Split-off changed ordinary cloning placement.** The voice-prompt cache-miss
  path unconditionally rebound the tokenizer to CPU. Only explicit split mode
  now changes placement; CUDA retains its existing ordinary GPU-codec policy.
- **Reference cache could hide backend differences.** Its key did not distinguish
  CPU/GPU encoder policy. The cache namespace is versioned and includes encoder
  backend placement, so old unclassified entries cannot certify a CPU/GPU A/B.
  Diagnostic comparisons must also disable the cache to exercise fresh encoding.
- **Repeated and failed loads leaked or retained invalid state.** Replacement
  loads now release previous allocations, failed/partial loads release their
  resources, and cleanup clears all borrowed encoder/decoder tensor pointers.
  A failed replacement leaves the tokenizer unloaded; it does not preserve the
  previous tokenizer. The same cleanup handles destruction and experimental
  placement changes. GPU-request policy is recorded separately from actual
  backend identity so CPU fallback does not cause repeated reloads.
- **Optional pre-permutation had an unsafe allocation-failure fallback.** The
  shared helper published tensors before buffer allocation and freed their
  context on failure. It now publishes only after success and refuses unsupported
  input dtypes. The fallback cannot consume dangling permutation pointers.
- **A CPU measurement promoted every GPU backend.** Decomposition now defaults
  to CPU only. CUDA/Metal/Vulkan and other GPU paths retain crop unless explicitly
  opted in with `CRISPASR_OMNIVOICE_CODEC_DECOMP=1`. Split remains off by default.
- **The cloning campaign selected a passing reference after observing output.**
  `station` met the declared `min(reference WER) <= 0.2` rule but was excluded
  because its clone failed. The corrected harness tests every eligible reference
  against both cross-sentence targets. Historical `window` results remain valid
  measurements of that subset, not acceptance of the complete campaign.
- **Several conclusions exceeded the evidence.** Six matching printed digits
  do not prove array identity. A bisect whose CPU intermediates disagree with
  the reference does not isolate a Metal kernel. ASR agreement raises a fixture
  concern without proving pronunciation causality. Two cold/warm timing pairs
  do not establish a fixed break-even count. Documentation now states these
  limits. ASR intelligibility also does not measure speaker similarity.

## Verification scope

The fixture pin resolves in the **model repository**
`cstr/crispasr-regression-fixtures`, revision
`d0a7d7a8be318a5841dfdbe6ad37d3acf75523e3`; the dataset repository with the same
name is different. Local fixture bytes match the historical manifest hashes.
Local model bytes match the public metadata at OmniVoice revision
`92266739c4496b04a497e164cf8c2801a792551e`:

| File | SHA256 |
| --- | --- |
| `omnivoice-q8_0.gguf` | `9d8835c80d50cce6ba66bed60c31e94836a7f990c40d2e22496c0b6710c98bb5` |
| `omnivoice-tokenizer-f16.gguf` | `710ef610e1f2845c6b7333d5432376b24f2d20d2c54a8cec9bc118d183ecea63` |

The native regression includes the actual runtime and uses tiny synthetic GGUF
weights. It covers replacement, missing/truncated input, recovery, the split-off
placement guard, CPU decomposition controls and a forced backend allocation
failure. An independent run with AddressSanitizer and UBSan passed these paths.
The existing output-allocation test passed five normal/OOM/gain cases and its
four historical/mutated failure controls. No full-model quality result follows
from these unit checks. Fresh CTest and live A/B results are recorded below when
completed.
