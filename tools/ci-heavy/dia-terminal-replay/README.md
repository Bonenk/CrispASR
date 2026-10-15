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
