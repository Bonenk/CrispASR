# Hosted Dia terminal hidden-state intervention

This scratch-only worker extends the accepted one-step suffix replay. It pins
runtime8d0ad886/C36 and the public independently audited37-arm archive at
revision049d93d4d97ce7a542a030b4a20366df8b372699. It imports native self-cache
positions0..977 and executes position978 with the original fixed token row.
No production runtime, model, sampler or default changes.

Run on the Linux x86 GH heavy CPU runner:

```sh
gh workflow run heavy-cpu.yml --ref <candidate-branch> \
  -f script=tools/ci-heavy/dia_terminal_replay.py \
  -f pip="numpy==1.26.4 torch==2.6.0 huggingface_hub==0.36.0 safetensors==0.5.3 gguf"
```

Five arms run in this order:

1. Baseline: all127 stages and final logits must reproduce accepted native bytes.
2. Native layer17: all baseline stages/logits must remain exact.
3. Native final hidden: all baseline stages/logits must remain exact.
4. Original layer17: every stage and logit must reproduce the accepted37-arm
   original-layer17 anchor exactly, using its lossless capture references.
5. Original final hidden: substitute the actual original full-model final hidden
   immediately after final RMSNorm, before all9 head matrix multiplies. Every
   earlier stage, encoder and self/cross cache must remain native byte-exact.

The final-hidden leaf is retained by pointer after naming. Its discarded norm
producer remains an explicit graph root; both computed and consumed hidden are
captured and checked exactly. Native/original final-hidden inputs are4096 finite
F32 values. Missing prefixes, invalid-size/nonfinite files and simultaneous
layer/terminal replacements reject. Existing prefix/finite loader controls and
actual ggml ownership guard remain applicable; complete patched source receives
local syntax checking against exact pinned headers and is compiled on GH.

Each arm retains actual logits, all18 layer metrics with norms/errors, final
hidden metrics, computed/consumed files, all18 cross-cache identity checks and
actual original Torch2.6 filter margin/ranks/support. Split conditional and
unconditional captures must equal their raw logit branches. Captures use the
accepted lossless baseline/current-row encoding; all raw hashes and reconstruction
recipes remain. Worker/method/ELF/library/log provenance is retained in the output.
A full independent artifact audit is still required before interpreting a result.

This separates terminal paths in the actual hosted graph. Earlier local weighted
norm/head probes do not guarantee full-runtime/hosted byte reproducibility. A
sequential intervention difference is conditional, not an additive global causal
percentage. Skipping the prefix does not reproduce its RNG progression. No
sampled trajectory, speech repair, Q4, GPU, speed or release acceptance follows.

## Independent audit

`audit.py` verifies official terminal run/artifact metadata, archive SHA/size/CRC,
unique safe paths, root receipt and every payload, then immutable method git
objects, the exact composed scratch patch, ELF/library identities and a fresh
actual ggml guard. It uses the separately calibrated suffix auditor's byte
reconstruction and numerical helpers, never worker comparison results.

Run in the original Torch2.6 CPU reference environment:

```sh
python audit.py --bundle /storage/hosted-terminal-run \
  --repo /path/to/CrispASR --prior-out /storage/audited-suffix/out \
  --original-repo /path/to/original-dia \
  --output /storage/hosted-terminal-run/independent-audit-v1.json
```

The prior37-arm bundle must already have its own independent audit. All five
arms and both native controls are mandatory. All upstream stages remain exact;
original layer17 must reconstruct the complete accepted prior anchor. Computed
and consumed activations, logical full captures and all raw logits are checked
independently, including both conditional/unconditional branches. Original
final-hidden consumption must be byte-identical to the actual original capture.
The audit recalculates all90 layer metric rows, hidden/logit errors and actual
original-filter margin/ranks/support, and retains the four raw CFG operands.

Both the actual F32 CFG margins and F64 CFG evaluations from the four raw
logits receive separate ledgers, so propagated CFG roundoff is visible.
The margin ledger records baseline→original post17→original final hidden→
original logits. Its signed differences must telescope exactly; this is
conditional accounting along those interventions, not global additive causal
percentages. No unknown hosted margin is filled from a local reconstruction.
`audit_selftest.py` retains the calibrated24 reconstruction/numerical controls
and adds12 terminal corruption/nonfinite controls plus hand-derived signed
binary-fraction accounting. Actual accepted native/original layer17 activation
captures also calibrate the ownership checks. These preparations are not a new
terminal hidden-state result or speech acceptance.
