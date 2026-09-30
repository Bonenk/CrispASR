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
./build/bin/crispasr-diff parakeet phonon2-f16.gguf phonon2-jfk-ref.gguf samples/jfk.wav
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
