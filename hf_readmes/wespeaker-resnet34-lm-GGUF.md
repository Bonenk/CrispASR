---
license: cc-by-4.0
language:
- en
pipeline_tag: audio-classification
tags:
- speaker-recognition
- speaker-embedding
- speaker-diarization
- wespeaker
- resnet
- ggml
- gguf
library_name: ggml
base_model: Wespeaker/wespeaker-voxceleb-resnet34-LM
---

# WeSpeaker ResNet34-LM — GGUF (ggml conversion)

GGUF conversion of
[`Wespeaker/wespeaker-voxceleb-resnet34-LM`](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM),
a 256-dimensional speaker-embedding model, for the `--diarize-method foxnose`
diarizer in **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

## ⚠ Licence — attribution is required

These weights are **CC-BY-4.0**, inherited from the upstream model. Several
downstream projects describe them as Apache-2.0; that is incorrect — the
wenet-e2e/wespeaker *code* is Apache-2.0, the published *weights* are
CC-BY-4.0 and carry an attribution requirement. If you redistribute these
files, keep the attribution.

  Source: `Wespeaker/wespeaker-voxceleb-resnet34-LM`
  Upstream: https://github.com/wenet-e2e/wespeaker
  Licence: https://creativecommons.org/licenses/by/4.0/

## Files

| File | Size | Notes |
|---|---|---|
| `wespeaker-resnet34-lm-f32.gguf` | 26.5 MB | reference precision |
| `wespeaker-resnet34-lm.gguf` | 23.9 MB | conv kernels F32, linear F16 — **recommended** |

Conv kernels stay F32 in both: ggml's CPU `conv_2d_direct` trips
`GGML_ASSERT(src1->type == GGML_TYPE_F32)` with an F16 kernel. F16 on the 2-D
linear is fine and yields an identical embedding (cosine 0.99999744 vs
0.99999747).

## Architecture

ResNet34 `[3,4,6,3]` over 80-bin Kaldi fbank, TSTP pooling, `Linear(5120→256)`.
BatchNorm is folded into every convolution at conversion time (219 → 74
tensors) and the ArcMargin `projection` head is training-only and dropped
(11.25 M → 6.6 M params).

Three details that decide correctness, traced to `wespeaker/cli/speaker.py`
rather than assumed:

- the waveform is **int16-scale** (`torchaudio.load(normalize=False)`, since
  `wavform_norm` defaults to False), window is **hamming**, then per-utterance
  CMN;
- the 2-D map is **height=freq, width=time** — TSTP reduces over time and
  flattens (channel, freq) with freq fastest, which is the order `seg_1`'s
  5120 columns are in;
- TSTP's std uses torch's **unbiased (n−1)** variance, `+1e-7` inside the sqrt;
- the output is `seg_1(stats)` raw — no ReLU, no BatchNorm, **no L2
  normalisation**.

## Verification

Per-stage against the upstream PyTorch model run as an oracle
(`crispasr-diff wespeaker`), on an 11 s clip:

| stage | cos_mean |
|---|---|
| fbank | 0.999999 |
| stem / layer1–4 | 0.99997 – 0.999995 |
| stats | 0.999999 |
| **embedding** | **0.999997**, cosine(emb, ref) **0.99999747** |

Discriminative check on real audio: two windows of the same speaker score
cosine **0.595**, against **0.100** for a different speaker.

End-to-end, the CrispASR diarizer built on this model scores **DER 3.93%**
against the upstream Python pipeline's own output (same pinned speaker count,
0.25 s collar) with **zero speaker confusion** — the residual is entirely
false alarm from a different speech-segmentation source.

## Usage

```bash
crispasr -m <asr-model.gguf> -f audio.wav \
    --diarize --diarize-method foxnose \
    --diarize-embedder wespeaker-resnet34-lm.gguf
```

## Conversion

```bash
python models/convert-wespeaker-to-gguf.py \
    --model Wespeaker/wespeaker-voxceleb-resnet34-LM \
    --output wespeaker-resnet34-lm.gguf
```

The CrispASR runtime is an independent implementation written from the
published architecture; no upstream source is incorporated.
