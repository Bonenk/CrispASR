# Actual original Dia trace and single sampled-decision fork

This CPU diagnostic runs the unchanged original neural model, sampler and DAC.
It pins original source4a9e29b1, checkpoint257bc72f and reference tools8d0ad886,
with physical source/weight/DAC hashes checked. Four separate processes use
Torch2.6 CPU, four threads, seed42,3072 tokens, temperature1.2, top_p.95,
top_k45 and CFG3 for `[S1] We are committed to making it secure.`

Run on the Linux x86 GitHub heavy CPU runner:

```sh
gh workflow run heavy-cpu.yml --ref <candidate-branch> \
  -f script=tools/ci-heavy/dia_original_trace.py \
  -f timeout_minutes=300 \
  -f pip="numpy==1.26.4 torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 pydantic>=2.11.3 huggingface_hub==0.36.0 safetensors==0.5.3 descript-audio-codec==1.0.0 gguf"
```

1. OFF: actual original reference with a passive sampled-token/RNG witness.
2. ON: simultaneous actual self Q/K/V/attention and16 decoder boundaries in
   all18 layers at position978, plus actual pre-RoPE Q/K/V projections and
   actual returned cross-cache valid39-position prefixes. Padded tails are
   deliberately excluded from the observation, without changing inference.
3. OFF repeat: all common arrays, complete logits/codes/PCM and RNG witnesses
   must be byte-exact across these three arms. Original inputs/codes must also
   reproduce the pinned canonical proof. Original TTS→ASR must have WER0.
4. Fork: call the actual sampler, then clone and replace only sampled
   step978/codebook2 value877 with904. No additional random draw or changed
   recipe. Inputs/logits through step978 and the safe963-frame undelayed
   prefix must be exact. Input979 must differ only in that entry. Keep complete
   ensuing codes/audio and report TTS→ASR even if it fails.

The tiny selftest uses actual pinned Attention/KVCache/sampler definitions and
the pinned reference test's tiny original neural model. The tiny fixture's DAC
is a stub and establishes observation transparency only. The hosted full run
uses the real pinned DAC; its original-positive ASR gate is mandatory before
the fork. A wiring test exercises the reference's function-local hook import.

Each arm retains full array shape/type/byte/hash manifests. After all gates,
byte-identical arrays are losslessly deduplicated against OFF. Resolve each
`array-storage.json` entry's `storage` path relative to the artifact root and
verify its full NPY byte count/SHA before loading. Every unique array, source
observer, original blueprint source, log and ASR library is retained; the root
receipt hashes every payload. Failed runs retain their available logs/arrays.
An independent complete artifact audit is required before accepting findings.

The fork measures sensitivity to one sampled decision in the original. It
does not establish a unique native defect, native speech quality, GPU/Q4 parity,
performance or release readiness. No production arithmetic/default is changed.

## Independent audit

Keep official `raw.zip`, `run.json`, `artifacts.json` and `lint.json` together
in a fresh bundle directory. The auditor verifies the complete ZIP/CRC/hash and
extracts it itself. Supply the pinned canonical original reference and the
separately accepted terminal bundle; no full model executes locally.

```sh
python tools/ci-heavy/dia-original-trace/audit_selftest.py
python tools/ci-heavy/dia-original-trace/audit.py --bundle /path/to/bundle \
  --repo /path/to/CrispASR --original-repo /path/to/original-dia \
  --canonical /path/to/canonical/ref.gguf \
  --terminal-bundle /path/to/accepted-terminal-bundle --output /path/to/fresh-audit.json
```

Use the existing reference environment with NumPy and gguf. Checks cover every
raw payload and retained method against immutable git objects, exact lossless
array reconstruction, complete OFF/ON/OFF equality, canonical grids, actual
attention/boundary/cache identities, fork prefix and RNG witnesses, original/
fork ASR reports and the retained recognizer ELF. The auditor also compares
new actual full-model intermediates to accepted native raw captures with cosine,
norms and relative error, and requires all18 original layer outputs plus final
hidden to remain exact to the prior accepted original captures. The selftest
rejects15 corrupt-prefix/path/nonfinite controls and checks scale sensitivity.
