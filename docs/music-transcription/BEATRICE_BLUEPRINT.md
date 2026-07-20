# Beatrice v2 — port blueprint (§CB2)

Reference: `fierce-cats/beatrice-trainer`, file `beatrice_trainer/__main__.py`
(4519 lines; `__init__.py` is only `from .__main__ import *`). **MIT**, source
and trained models alike.

This is the HARD RULE #1 read: every claim below was checked against that file
or measured, not summarised. Where a detail is a silent-bug risk it says so.

Status: **blueprint read in progress.** No converter, no graph, no harness yet.
`PitchEstimator` and the shared conv primitives are covered; `PhoneExtractor`,
`VectorQuantizer`, `ConverterNetwork` and `Vocoder` are not yet read in full.

---

## Why the earlier assessment was wrong

PLAN previously recorded §CB2 as *"custom/NON-COMMERCIAL"* and a WebFetch
summary of this repo claimed it held *"training scripts only, not inference code
or the model architecture"*. **Both are false.** `LICENSE` is plain MIT and the
README states the source **and the trained models** are MIT; the architecture is
fully defined in the file above. The genuine non-MIT signals — `beatrice.lib`
(the closed VST inference engine, "used under permission") and the "JVS Corpus
Edition" — describe *other artifacts* and do not constrain a port from this
source. See `PLAN.md` §CB2.

---

## Shape of the thing

Unlike RVC (§CB1), Beatrice is **end-to-end from waveform**:

```
ConverterNetwork.forward(x, target_speaker_id, formant_shift_semitone,
                         pitch_shift_semitone=None, ...)
    x: [batch, 1, wav_length]   ->   24 kHz audio
```

It runs phone extraction and pitch estimation *itself*, so a caller supplies no
ContentVec — CometBeat's side gets simpler than RVC's, at the cost of three
networks to port instead of one.

`out_sample_rate = 24000` and `hop_length = 24000 // 100` are hardcoded in
`ConverterNetwork.__init__`, so the **100 Hz frame rate is the same as §CB1's**
and `SVC_RECORD_SHAPES.md` carries over unchanged.

Weights are split across three files (see PLAN §CB2 table) — `phone_extractor`,
`pitch_estimator` and `net_g` each live in a *different* checkpoint, and
`net_g`'s file also carries a `net_d` discriminator that is training-only.

---

## Details that are NOT the obvious default

Each of these produces plausible-but-wrong output if assumed rather than read.

### 1. Weight standardisation uses UNBIASED variance

`WSConv1d.standardized_weight()`:

```python
var, mean = torch.var_mean(self.weight, [1, 2], keepdim=True)
scale = self.gain * (in_channels * kernel_size // groups * var + 1e-8).rsqrt()
return scale * (self.weight - mean)
```

`torch.var_mean` defaults to **correction=1 (unbiased)**. `np.var` defaults to
`ddof=0`. Measured on a 2x2x3 tensor: torch `3.5`, numpy-default `2.9167` — a
~20 % scale error on **every** weight, uniformly, which reads as "the model is
just a bit off" rather than as a bug. Use `ddof=1`.

`WSLinear` does the same over dim 1 with `in_features * var`.

**Fuse at convert time.** `merge_weights()` writes `standardized_weight()` back
into `.weight` and sets `gain = 1`, exactly as `convert-rvc-to-gguf.py` fuses
weight_norm. Do that in the converter so the runtime graph has no special case.

### 2. GELU is the tanh approximation, everywhere

Every activation on the pitch path is `F.gelu(..., approximate="tanh")`, not the
default exact/erf GELU. ggml's `ggml_gelu` is the tanh approximation and
`ggml_gelu_erf` is exact — so `ggml_gelu` is correct here, but the choice must
be deliberate, and the numpy spec must use the tanh form.

### 3. Causal convs carry a delay/trim, and trim is not the padding

`CausalConv1d.__init__`:

```python
padding = (kernel_size - 1) * dilation - delay
self.trim = (kernel_size - 1) * dilation - 2 * delay      # NOT equal to padding
```

`forward` right-trims by `self.trim` when nonzero. With `delay > 0` the layer is
*not* strictly causal — it looks ahead by `delay` frames, which is how the
stated latency budget is spent (`PitchEstimator` comment:
`delay=1  # 10ms, 特徴抽出と合わせると 22.5ms` — 10 ms here, 22.5 ms including
feature extraction). Getting `trim` wrong shifts the output in time, which
cosine similarity punishes hard but which is easy to misread as a broken graph.

### 4. There is an RNG site in the vocoder

`overlap_add()` sets a **random initial phase**:

```python
normalized_freq[:, 0] = torch.rand(batch_size, device=pitch.device)
```

So Beatrice, like RVC, is **stochastic and cannot be validated by waveform
comparison against an unpinned reference run**. §CB1's discipline applies
directly: make the draw injectable, replay it in the harness, and keep the
production default random. Note this is `torch.rand` (uniform), where RVC's live
site was `torch.randn_like` — a noise injector patching only `randn_like` would
miss it and the harness would silently compare against a *different* phase.

The phase itself is accumulated in **float64** (`normalized_freq.double()
.cumsum_(1) % 1.0`) before being cast back to float32. A float32 cumsum over a
long signal drifts; this is deliberate and must be reproduced in double.

### 5. `sample_pitch` is a banded argmax, not an argmax

`PitchEstimator.sample_pitch` does: softmax over bins -> **force bin 0 to
-100.0** (it is the unvoiced class, excluded from the pitch argmax) -> a
box-filter `conv1d` of width `band_width=4` -> argmax over the *filtered*
sequence -> a masked argmax within the winning band. Plain `argmax(logits)` is a
different function and will disagree on ambiguous frames only — i.e. it will
look almost right.

`return_features=True` additionally emits unvoiced / half-pitch / double-pitch
probabilities, using `pitch_bins_per_octave = 96` as the octave stride, and
`ConverterNetwork` consumes 4 pitch feature channels (`embed_pitch_features =
nn.Conv1d(4, hidden, 1)`).

### 6. The `.beatrice` dump format pre-folds the attention scale

`dump_layer()` is the trainer's export path to the closed inference engine. For
`nn.MultiheadAttention` it bakes the attention scale into the weights:

```python
in_proj_weight[: 2 * embed_dim] *= 1.0 / math.sqrt(math.sqrt(embed_dim // num_heads))
```

— i.e. `1/sqrt(d_head)` split as `sqrt` on **both** q and k, and it also
reorders to `[num_heads, 3, head_dim, embed_dim]`.

**Convert from the `.pt` state_dict, not from a `.beatrice` dump.** Reading a
dump means inheriting both transformations; applying the usual `1/sqrt(d)` on
top of pre-scaled weights would scale attention logits by `1/d`. Recorded here
because dumps are what circulate.

### 7. `ConverterNetwork` initialisation traps

* `embed_quantized_pitch` is a **fixed sinusoidal table** built in `__init__`
  and `requires_grad_(False)` — verify whether it is present in the checkpoint;
  if not, the converter must rebuild it (including the `* sqrt(4/5)` scale).
* `key_value_speaker_embedding` initialises **every speaker row as a copy of row
  0**. An A/B across speakers that shows little difference is therefore not
  automatically a port bug.
* `self.melspectrograms` is **loss-only** and not on the inference path.
* `VectorQuantizer` is injected into `phone_extractor.head` via a **forward
  hook** (`enable_hook`). This session lost time twice to hooks that never
  fired (RVC's `flow` calls `.forward()` directly, so `register_forward_hook`
  was silently comparing nothing). The spec must *prove* the hook is live by
  asserting the quantised path changes the output — never by assuming it ran.

---

## Order of work

Per component, smallest and most frozen first:

1. **`PitchEstimator`** — 7.1 MB, fully frozen, and its front end
   (`extract_pitch_features`: instantaneous frequency + autocorrelation, hop 160
   / win 560 / `max_corr_period` 256 / `corr_win_length` 304) is pure DSP that
   can be validated on its own before any learned layer is involved.
2. **`PhoneExtractor`** — 14.7 MB, also frozen.
3. **`ConverterNetwork` + `Vocoder`** — the LibriTTS-R `net_g`, 177 tensors.

For each: read -> numpy/torch executable spec -> ggml graph -> per-stage diff
harness, with the reference dump carrying the injected RNG draw so the
comparison is deterministic (`tools/rvc_torch_parity.py` is the template).

Do not skip the end-to-end stage. In §CB1 every one of 47 per-stage checks was
green while the assembled `convert_e2e` sat at cos 0.40 — per-stage checks are
input-aligned and never test the wiring between stages.
