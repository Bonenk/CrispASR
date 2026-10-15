# Dia one-step activation suffix replay

Scratch diagnostic only; production runtime/defaults are unchanged. The worker
is `../dia_suffix_replay.py`. Run on the Linux x86 GH heavy CPU runner with:

```sh
gh workflow run heavy-cpu.yml --ref <tested-source-branch> \
  -f script=tools/ci-heavy/dia_suffix_replay.py \
  -f pip="numpy==1.26.4 torch==2.6.0 huggingface_hub==0.36.0 safetensors==0.5.3 gguf"
```

The worker pins runtime `8d0ad8862d2cad5d305a48a57f2a99b5024bc39a`, ggml
`c36dab89b662838f0f5d4826c399198c0b90bbfc`, original weights, exported F32
model SHA, and the complete retained query/key artifact. It checks ZIP SHA,
CRC, unique safe paths and every extracted payload against that run's receipt.
The pinned original sampler is retained inside the verified archive. No full
original Python model is instantiated. All full-model execution belongs on GH.

Each native captured self K/V is `[batch=2,time=979,channels=512]`. Import only
positions0..977, transpose into the runtime's time-major host cache, then run
position978 using the same nine forced tokens, encoder and F32 cross producer.
The tiny actual ggml guard checks current-position exclusion, batch/time order,
replacement input ownership after renaming, and retention of the discarded
producer through an extra graph root. The finite exact-size C++ loader rejects
truncated, extended, missing and nonfinite activation files. The complete
patched runtime also receives a local syntax check against exact pinned headers.

The baseline must reproduce all127 previously captured stages and final logits
byte-for-byte. Then all18 native post-layer replacement controls must reproduce
the same bytes before any original replacement executes. All18 original arms
replace exactly one current post-layer activation with the actual original
full-model capture; all earlier layer outputs and self Q/K/V through that layer
stay exact. Every arm retains both computed and consumed activations; computed
must equal the unchanged native producer and consumed must equal the input file.
All18 cross K/V pairs are additionally captured in fixed layer/K/V order and
must stay exact in every arm (shape18×2×2×39×2048, F32).

Each arm reports original-sampler CFG margin, ranks and retained support,
all18 layer cosines, norms, scale errors, relative L2/max error, final hidden
and all-head logits. These are conditional suffix interventions with native
past caches, not a replacement of the historical original self caches. A rank
flip identifies sensitivity under these held conditions; it does not prove a
unique root cause or additive attribution.

Large captures use a lossless encoding. The baseline contains all full files.
For another arm, `captures.json` records every complete capture's byte count
and SHA: `full` keeps the raw file, `baseline` references identical baseline
bytes, and `current-row` references the baseline cache plus the retained
`*.current.f32` row at position978 (shape2×512). Every past prefix is checked
exact, and every removed full file is reconstructed and SHA-verified first.
Cross caches likewise retain one baseline file plus exact hashes per arm.
`receipt.json` hashes all final payloads and retains exact source/build/ELF/libs
and run logs. Retain partial artifacts on any failure; do not weaken exact gates.

The sampler is observed before multinomial drawing. Skipping978 steps does not
advance the original RNG state; no sampled trajectory, speech, Q4, speed or GPU
acceptance is claimed. Full original self-Q capture remains a separate follow-up.
