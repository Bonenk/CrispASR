# CrispASR v0.8.38

Canary 180M Flash support, lower-latency VoxCPM2 synthesis, iOS 15 frameworks,
and fixes for long-form Whisper, Unicode JSON, speech-free chunking, Voxtral,
Korean alignment, and several model-parity defects found by the expanded
regression suite.

---

## Canary 180M Flash

The existing Canary backend now loads handy-computer's transcribe.cpp GGUFs
while retaining compatibility with the legacy Canary 1B-v2 layout. The runtime
reads checkpoint dimensions and frontend metadata, applies the trained
projection, builds Canary's aggregate prompt, validates language and translation
pairs, and uses the model's long-form path above 40 seconds. Q4 English ASR,
Q5 English-to-German translation, C-ABI auto-detection, legacy F16 loading,
CPU/Vulkan parity, Android arm64 packaging, and a real Android device run were
verified (#470).

## Performance and platforms

- VoxCPM2 runs the complete CFM Euler solve as one cached graph per audio patch
  (#461): CFM time fell 4% on Vulkan and 8% on CPU with matching output; CUDA
  was neutral. `CRISPASR_VOXCPM2_CFM_FUSED=0` restores the old path, and
  `CRISPASR_VOXCPM2_INFERENCE_STEPS` is documented as the main quality/speed
  control.
- Cohere Transcribe's published Q4 model now stores its 96 large pointwise
  matrices as Q8_0 instead of hidden F16. The transcript stayed identical while
  measured encoder time fell from 10.24 to 9.45 seconds.
- Apple XCFramework iOS slices now target iOS 15.0 instead of 16.4. The release
  dry run checks the binary and plist minimums for both slices.
- The scheduler node profiler is shared by Canary CTC, Cohere, FireRed-ASR,
  Granite Speech, Moonshine, Moonshine Streaming, and Paraformer through
  `CRISPASR_SCHED_PROFILE=1`.

## Fixes

- Explicit `--backend whisper` now passes long recordings to Whisper's native
  seek/window loop, matching the legacy no-VAD path instead of applying the
  generic energy chunker first (#463).
- JSON full-output remains valid UTF-8 when a BPE token divides a multibyte
  character (#475).
- Speech-free slices from the generic energy chunker are skipped rather than
  decoded into hallucinated text (#471).
- Voxtral 3B bounds its Tekken merge map to the embedding table (#472).
- Korean word alignment preserves spaces with punctuation stripping and
  mixed-language mode (#465).
- OmniVoice target duration survives per-request reset, unusable voice
  references fail closed, and reference-cache writes are atomic.
- Hojo-ASR V1 has its registry entry, length-aware token cap, and
  transformers-compatible KV-cached beam search.
- Wav2Vec2, XLS-R, and HuBERT even-kernel positional convolution padding now
  matches transformers. Parakeet v3 no longer assumes encoder input x-scaling
  when old GGUF metadata omits it. OmniASR CTC stitches chunk logits across a
  joint blank run rather than concatenating chunk text.
- Granite Speech prompting matches `apply_chat_template`; FastConformer CTC
  parity comparison correctly normalizes raw logits.

## Regression coverage

The push selector now compares with the last successful run, so a cancelled run
cannot hide changed backends. Gated stage diffs were added or tightened for
Wav2Vec2, HuBERT, Data2Vec, OmniASR CTC, FastConformer CTC, Granite Speech,
SenseVoice, Qwen3-ASR, Parakeet v3, Mini-Omni2, Nemotron, and Cohere. VoxCPM2's
speech roundtrip is included in the nightly matrix.
