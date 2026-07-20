# Beatrice v2 — port blueprint (§CB2)

Reference: `fierce-cats/beatrice-trainer`, file `beatrice_trainer/__main__.py`
(4519 lines; `__init__.py` is only `from .__main__ import *`). **MIT**, source
and trained models alike.

This is the HARD RULE #1 read: every claim below was checked against that file
or measured, not summarised. Where a detail is a silent-bug risk it says so.

Status: **`PitchEstimator` converter + reference dump DONE and validated.** No
ggml graph yet. `PhoneExtractor`, `VectorQuantizer`, `ConverterNetwork` and
`Vocoder` are not yet read in full.

| artifact | state |
|---|---|
| `models/convert-beatrice-to-gguf.py` | pitch_estimator: 88 tensors, 36 dropped as fusion-neutralised (124 total, balances) |
| `tools/beatrice_torch_parity.py` | 36-stage reference dump, spec reproduces `forward()` **bit-identically** |
| ggml graph (`src/beatrice_pitch.{h,cpp}`) | NOT STARTED |
| `crispasr-diff beatrice` | NOT STARTED |

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

## Reference evaluation — is it any good?

Run before committing to the port, per [[tts-advisory-check-blueprint-first]].
**Verdict: yes on clean input, and the port target is well defined.**

There is **no inference CLI** — `__main__.py`'s argparse is training-only
(`-d/-o/-r/-c`). Driving it takes: build `PhoneExtractor` + `PitchEstimator`,
load their two checkpoints, build `ConverterNetwork(pe, pi, n_speakers=200,
pitch_bins=448, hidden_channels=256, vq_topk=4)`, load `net_g`, then
**`net_g.enable_hook()`** to activate the VectorQuantizer. Only extra dep is
`pyworld`. Scripts: `run_ref.py` / `probe2.py` (scratchpad).

**`load_state_dict` reports 0 missing, 0 unexpected** against the 177-tensor
`net_g`. The architecture read above therefore matches the shipped checkpoint
exactly — the converter can and should assert exact key coverage rather than
using `strict=False` tolerance. This also resolves blueprint item 7:
`embed_quantized_pitch.weight` **is present** (448x256), so the converter does
not need to rebuild the sinusoidal table (but must not assume that for
checkpoints from other training runs).

ASR roundtrip (whisper base), same sentence, 24 kHz output:

| input | transcript of the CONVERTED audio |
|---|---|
| clean TTS speech (in domain) | "And so my fellow **in Erickons**, ask not what your country can do for you, ask what you can do for your country." |
| `samples/jfk.wav` (1961 archival, out of domain) | "And so might fellow **Annacats**, **airsp** not. What your country can do for you..." |

So: on clean input, one degraded word out of twenty and everything else
verbatim. On noisy archival audio it degrades badly and inconsistently. Two
fairness caveats before calling that a quality ceiling: jfk.wav is far out of
domain for a LibriTTS-R model, and **the 151 checkpoint is a `pretrained_file`
bootstrap for fine-tuning, not a finished voice** — Beatrice's actual workflow
trains a target speaker on top of it.

Prepending 1 s of silence did **not** repair the archival degradation (it got
slightly worse), so it is not a simple warmup transient.

Speaker conditioning is real: `embed_speaker` rows differ from row 0 by mean L2
1.98 (a copied/untrained table would be 0.0), and cross-speaker log-mel distance
runs **2.4x-3.7x the RNG floor**.

### Measuring anything here needs an RNG floor

First attempt used **waveform correlation** and produced ~0.00 for every pair —
including a same-input/same-speaker rerun that should have been ~1.0. Cause:
`overlap_add`'s random initial phase (detail 4 above) fully decorrelates the
phase while the audio sounds identical. The measure was invalid, and its
near-zero readings would have been easy to misread as "speakers differ hugely"
or "padding changed everything".

**The discipline that fixes it:** use a phase-invariant measure (log-mel), and
run the *same input twice* first to establish the **floor** — the distance
attributable purely to the RNG. Here the floor is 0.374; every real comparison
is then quoted as a multiple of it. Without the floor, "padded vs raw = 0.565"
means nothing; against it, it is 1.5x — real but small next to the 2.4-3.7x of
an actual speaker change. Applies to the §CB2 harness and to any stochastic
backend. See [[tts-parity-not-by-audio-corr]].

Note the output is stochastic enough that ASR transcribed two runs of the *same*
speaker differently — so acceptance thresholds must be set against the floor,
not against a single golden run.

---

## PitchEstimator — the port target, verified

```
wav 16k -> extract_pitch_features (DSP) -> instfreq[192,T] + corr_diff[256,T] + energy[1,T]
   instfreq -> conv1x1 -> gelu_tanh -> conv1x1  \
                                                 + -> gelu_tanh -> ConvNeXtStack(9) -> head 1x1 -> logits[448,T]
   corr_diff -> conv1x1 -> gelu_tanh -> conv1x1 /
```

`ConvNeXtStack`: `embed` CausalConv1d(k=3, delay=1) -> LayerNorm(**with affine**)
-> 9 x block -> `final_layer_norm` (**with affine**). Each block:
depthwise CausalConv1d(k=33, groups=192, delay=0, strictly causal) -> LayerNorm
(**affine folded away — normalise only**) -> Linear 192->384 -> gelu_tanh ->
Linear 384->192 -> residual add.

**The two LayerNorm forms are not interchangeable.** `merge_weights()` folds the
per-block affine into `pwconv1` but leaves the stack-level `norm` and
`final_layer_norm` affine intact. Using one form throughout is wrong in one
place or the other.

Fusion verified output-preserving on the real checkpoint before being baked into
the converter: **cos 1.0000000000, max_abs 7.6e-06** (f32 noise), and afterwards
`gamma`, `pre_scale`, `post_scale`, `post_scale_weight` are all identically 1.0
and the per-block LayerNorm affine is identity — so all 36 are dropped and the
runtime graph never sees them.

## Negative controls — the gate had two holes

The reference dumper re-implements the forward pass step by step and refuses to
write a file unless it reproduces the module's own `forward()`. Control:
**max_abs 0.000e+00**, bit-identical.

A first-run pass proves nothing ([[parity-harness-negative-control]]), so each
guard was tested by breaking a load-bearing detail. Two of the breaks initially
*passed*:

| break | first result | after fix |
|---|---|---|
| tanh-approx GELU -> exact erf (detail 2) | **PASSED**, cos 0.9999996, max_abs 2.4e-02 | FAILS (rel 1.65e-03) |
| drop the `1e-5` in delta_spec normalisation | **PASSED**, wrote an all-NaN reference, exit 0 | FAILS, names the stage |
| skip the per-block LayerNorm | FAILS, cos -0.53 | FAILS |
| drop the causal trim (detail 3) | raises (shape 432 vs 400) | raises |

Both fixes generalise to the ggml harness:

1. **Gate on relative max-abs, not cosine.** Both arms are torch on the same
   weights, so the control is bit-identical and anything above f32 rounding is
   real. A genuinely wrong activation scored cos 0.9999996 — through any
   `cos > 0.999999` check — while carrying 2.4e-02 of absolute error. HARD RULE
   #2b in the concrete.
2. **Check finiteness before any tolerance test.** Every NaN comparison is
   `False`, so `rel > TOL` is `False` for NaN and a divide-by-zero spec wrote a
   reference full of NaN and exited 0. A tolerance check structurally cannot
   catch this.

A third "control" was invalid and worth recording as a trap: re-applying
`gamma` after the fusion is a **no-op by construction** (post-merge `gamma` is
all-ones), so it passed bit-identically. It tested nothing. A negative control
has to break something the code actually depends on.

### Layout convention

Torch stores `[batch, channels, time]` with **time fastest**; stages are dumped
in exactly that memory order, landing in ggml as `ne = [time, channels]` — what
`ggml_conv_1d` expects. **Do not transpose on either side.** Three separate RVC
port bugs came from transposing here, each producing ~0 cosine on a graph that
was correct.

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
