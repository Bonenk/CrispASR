# Phonon-2

[Fermion Research's Phonon-2](https://huggingface.co/FermionResearch/Phonon-2)
is an English speech recognizer derived from NVIDIA Parakeet TDT 0.6B v3.
CrispASR uses its existing Parakeet backend for this model.

```sh
crispasr -m phonon2 -f recording.wav                      # Q8_0 default, ~674 MB
crispasr -m phonon2 --model-quant f16 -f recording.wav     # reference fidelity, ~1,255 MB
crispasr -m phonon2 --model-quant q4_k -f recording.wav    # smaller, ~402 MB
```

## Convert the upstream weights

Install the Parakeet converter's dependencies, plus `zstandard`:

```sh
pip install torch numpy gguf sentencepiece librosa pyyaml safetensors huggingface_hub zstandard
python models/convert-parakeet-to-gguf.py \
  --hf FermionResearch/Phonon-2 --output phonon2-f16.gguf
./build/bin/crispasr-quantize phonon2-f16.gguf phonon2-q8_0.gguf q8_0
./build/bin/crispasr-quantize phonon2-f16.gguf phonon2-q4_k.gguf q4_k
./build/bin/crispasr -m phonon2-q8_0.gguf -f samples/jfk.wav
```

`--hf` also accepts a downloaded Hugging Face snapshot or the extracted
directory containing `config.json` and `model.fermion`. Conversion checks the
upstream archive/container SHA-256, expands the five-value encoder weights
exactly (`0`, `±lo`, `±hi`, with two magnitudes per output row), and expands the
remaining integer tables with their stored row scales. It reads the model
configuration and vocabulary from the archive; no teacher weights or downloaded
Python code are loaded. The resulting GGUF retains `general.architecture =
parakeet`, so bindings auto-detect it through the C ABI even if the file is
renamed.

## Size, quality, and speed

The upstream **164 MB** figure describes its compressed transport file, not
an F16 or ordinarily quantized GGUF and not runtime RAM. Its custom five-value
packing differs from ggml's quantization formats. CrispASR expands that format
before GGUF conversion; Q8_0 and Q4_K apply another quantization step. Those
exports should not be assigned the upstream model's benchmark scores without
their own evaluations.

Local CPU validation on 21 clips (JFK, one upstream LibriSpeech sample, and
19 clips from the Hugging Face LibriSpeech test subset):

| Export | Download | Exact reference transcripts | Word edits versus reference* |
|---|---:|---:|---:|
| F16 | 1,255 MB | 21/21 | 0/298 |
| Q8_0 (default) | 674 MB | 19/21 | 1/298 |
| Q4_K (optional) | 402 MB | 15/21 | 7/298 |

\* Lowercase, ignore punctuation, retain apostrophes inside words; edit distance
against the independent model reference, **not WER against human ground truth**.
Q8's differences were one capitalization change and one proper-name spelling.
This is a small conversion/quantization check, not a reproduction of the seven
benchmark sets. F16 passed all 28 compared stages on both JFK and the upstream
LibriSpeech sample: minimum cosine 0.999996 / 0.999796, with tensor norm-ratio
error bounded by 0.065% / 0.259%. C ABI sessions auto-detected the backend and
produced these transcripts with an arbitrary model filename.

Fermion reports 5.21% average WER across seven English benchmark sets and 174×
realtime on an M5 MacBook Air using its MLX engine, excluding loading. These
figures are upstream measurements, not CrispASR performance claims. Phonon-2
is advertised and evaluated for English; its inherited multilingual vocabulary
does not establish multilingual accuracy after retraining.

## Reference validation

```sh
python tools/dump_reference.py --backend phonon2 \
  --model-dir FermionResearch/Phonon-2 --audio samples/jfk.wav \
  --output phonon2-jfk-ref.gguf
./build/bin/crispasr-diff phonon2 phonon2-f16.gguf phonon2-jfk-ref.gguf samples/jfk.wav
```

The reference uses the unmodified upstream container reader and stock
Transformers `ParakeetForTDT`, independently of the converter's unpacker.
Install a recent `transformers` release providing `ParakeetForTDT` to run it.
Only the teacher's small JSON configuration/tokenizer files are fetched.
`PHONON2_BASE_DIR` can point to a local copy of those files for offline runs.

Weights: **CC-BY-4.0**, derived from NVIDIA's Parakeet TDT v3 and retrained by
Fermion Research. Retain attribution and describe conversion/quantization
changes when redistributing. The vendored reference reader is Apache-2.0;
its license is in `tools/reference_backends/phonon2/LICENSE`. The upstream
[NOTICE](https://huggingface.co/FermionResearch/Phonon-2/blob/main/NOTICE)
documents the original model, changes, and training data.

## Integration checklist

Phonon-2 is a model variant of the existing Parakeet runtime. The contributing
checklist applies through that shared engine:

| Checklist point | Wiring |
|---|---|
| C runtime and stage timers | `src/parakeet.{h,cpp}`; `CRISPASR_PARAKEET_BENCH=1` |
| CLI adapter and factory | Shared Parakeet adapter; explicit `--backend phonon2` and model filename routing |
| CLI/library CMake linkage | Existing `parakeet` library in both CLI and shared C ABI; no duplicate runtime library |
| C ABI dispatch, lifecycle, setters | Explicit `phonon2` alias normalizes to `parakeet`; existing transcription, hotwords, beam, temperature, attention-context and cleanup paths |
| Architecture auto-detection | `general.architecture=parakeet`, shared `arch_backend_map.h`; works after renaming |
| Registry | `phonon2`, Q8 default, F16/Q4 alternatives and weight license |
| Quantization | Existing Parakeet rules, including pointwise matmuls; published exports validated |
| Reference and diff | Independent upstream reader + Transformers; existing Parakeet stage APIs and strict pinned nightly gate |
| Bindings | Generic C ABI reaches all bindings; Python/Go/Dart document the variant |
| Go static linkage | Existing Parakeet library; generator checks the unchanged cgo library list |
| Architecture/capability docs | `docs/architecture.md#phonon2`; generated feature matrix and library capability table |
| Tests and live environment | Container/registry tests, pinned regression, `test_phonon2_live.py`, `CRISPASR_MODEL_PHONON2` |

`Session(..., backend="phonon2")` explicitly opens this variant; `Session.backend`
reports the shared runtime name, `parakeet`. The CLI uses model metadata to select
English without automatically loading Whisper for LID. Non-English language
requests produce a warning; the model cannot honour them. Explicit language
identification remains available. Native punctuation is advertised. `--no-flash-attn` and the C ABI open flag
now reach the graph builder; the live guard checks the resulting node trace.
The raw Parakeet context default now reports flash enabled, matching the
previous graph behaviour; an explicit false selects manual attention.

## Reproducible profiling

Build both `crispasr-lib` and `crispasr-cli`. Set the model and library paths:

```sh
python tools/profile_phonon2.py --engine runtime \
  --model phonon2-q8_0.gguf --lib build/src/libcrispasr.so \
  --output runtime-q8.json
CRISPASR_SCHED_PROFILE=1 CRISPASR_PARAKEET_BENCH=1 \
CRISPASR_PARAKEET_ENC_PROBE=1 CRISPASR_PARAKEET_DECODE_TIMING=1 \
python tools/profile_phonon2.py --engine runtime --trace \
  --model phonon2-q8_0.gguf --lib build/src/libcrispasr.so \
  --output trace-q8.json
python tools/profile_phonon2.py --engine reference \
  --model /path/to/upstream/snapshot --output reference-cpu.json
```

The benchmark loads once, warms each 11/55-second shape, checks nonempty stable
transcripts and proportional word counts, then records three inference times
and their median. The trace is a separate run: the scheduler callback forces
node materialization and changes dispatch overhead. Compare timings only on
the same host, device, thread count and load. The Python reference is stock
Transformers on CPU, not the upstream MLX engine.

The manual **Phonon-2 integration and profile** GitHub workflow runs Linux CPU
and macOS Metal, checks wiring/live transcripts and F16 stage parity first,
then records F16/Q8/Q4 timings plus Q8 stage/node traces. Linux also measures
the independent Python reference on the same runner. Artifacts retain all raw
samples, transcripts, model checksums, dependency versions and host details.
