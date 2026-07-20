# Music transcription in CrispASR

Porting the CometBeat / mus-textbook "transcription → SOTA" model roster
(`docs/TRANSCRIPTION_SOTA_HANDOFF.md` in that repo) from ONNX to CrispASR
ggml/GGUF backends.

## NOW — active work

- **Done**: feasibility triage of all 7 handoff workers (below); fixed a live
  `CAP_SEPARATE`/`CAP_STREAMING` bit collision (both were `1u << 22`; CLI builds
  clean after the move to bit 23). **CREPE converter landed and validated**:
  `models/convert-crepe-to-gguf.py` + `tools/crepe_numpy_parity.py`, cos=1.0 vs
  torchcrepe on both capacities (tiny 1.0 MB, full 44.5 MB at f16).
- **Done**: `src/crepe.{h,cpp}` runtime — **cos = 1.0 vs the numpy spec** on a
  real tone sweep, decoding 220.6 / 440.4 / 881.4 Hz at 0.95–0.97 confidence.
  `tests/test_crepe_parity.cpp` is the acceptance gate.
- **Done**: CREPE wired through the 12-point checklist — `CAP_PITCH = 1u << 24`,
  `examples/cli/crispasr_backend_crepe.cpp` (redirect shim, mirroring the
  htdemucs one), the `--pitch` early dispatcher
  (`examples/cli/crispasr_pitch_cli.{h,cpp}`, mirroring `--separate`), factory /
  roster / arch auto-detect (`crepe`) / filename heuristic in
  `crispasr_backend.cpp`, the session C ABI (`crispasr_session_pitch*` in
  `src/crispasr_c_api.cpp` + `include/crispasr_session.h`), registry entries for
  `cstr/crepe-GGUF` (**tiny is the default**), CMake linkage, README + docs/cli.md.
- **In flight**: nothing.
- **Next**: upload `cstr/crepe-GGUF` (the registry URLs point at it but the repo
  is not published yet); quantize (q8_0/q4_k) and re-measure; then the Dart FFI +
  WASM surfaces. `core/stft.h` extraction is independent (CREPE needs no STFT).

### Performance — measured, M1, quiet box (load 4.0), 10 s audio, median of 3

| model | Metal | CPU |
|---|---|---|
| full (44.5 MB f16) | 20.0 s — **RTF 2.0** | ~400 s — RTF 40 |
| tiny (1.0 MB f16) | 2.8 s — **RTF 0.28** | ~24 s — RTF 2.4 |

CREPE is genuinely expensive: at the reference 10 ms hop it is **1409 MMAC per
frame → 282 GFLOP per second of audio** for `full`, and 36.7 MMAC/frame →
7.3 GFLOP/s for `tiny` (38× cheaper). So **tiny is the shipping default** — it
is also what the handoff asks for ("smallest that hits accuracy"). `full` stays
available and is the right choice offline. Neither is close to real-time on CPU;
the GPU path is not optional here.

Three graph decisions got it from the first working version (RTF 31) to here:

1. **Batching (the big one).** One frame per dispatch wastes the GPU on a model
   this small per-frame. `kBatch = 64` makes each layer one large GEMM.
2. **Channel-fastest layout throughout.** `ggml_conv_1d` ends by permuting back
   to (OL, OC, N), materializing the whole activation every layer. We keep the
   mul_mat's native (OC, OL, N), do bias/relu/BN there — where a plain (OC)
   vector broadcasts along ne[0], ggml's fast path, instead of a stride-0
   (1, OC, 1) broadcast — and pool with `ggml_pool_2d(k0=1, k1=2)`. The one
   transpose im2col forces is deferred until *after* the pool, so it moves half
   the bytes, and the last layer skips it entirely because (OC, OL, N) already
   *is* the channel-fastest flatten the classifier wants.
3. **F32-baked conv kernels** (`ggml_conv_1d` casts an F16 kernel to F32 inside
   the graph — in a persistent graph that re-casts 44 MB per 10 ms frame).
   Gated `CRISPASR_CREPE_NO_BAKE_F32=1`. Honest note: this one measured
   **neutral** here, unlike qwen3-tts CODEC_FASTCONV. Kept gated-on because it
   is provably redundant work, but it was not the win.

Gates: `CRISPASR_CREPE_NO_GPU=1`, `CRISPASR_CREPE_NO_BAKE_F32=1`,
`CRISPASR_CREPE_DEBUG=1`.

### `ggml_conv_1d` returns a tensor whose declared shape contradicts its data for N > 1

**Status: fixed in the fork (`ggml/src/ggml.c`), upstream PR drafted at
`tools/upstream-prs/24-conv-1d-batch-reshape.md` + a standalone repro. NOT yet
merged to main — one audit item is open, see below.**

The im2col is the FIRST `ggml_mul_mat` argument, so the result's ne is
`[N*OL, OC]` (OC slowest). The final `ggml_reshape_3d` declares `[OL, OC, N]`
(N slowest). Those expressions coincide **exactly when N == 1** and differ
otherwise — which is why every shipping caller is correct and this was invisible.

Repro (`tools/upstream-prs/24-conv-1d-batch-reshape.repro.cpp`, standalone,
vs a hand-rolled direct convolution), before the fix:

```
N=1  cos=1.00000000  OK        N=2  cos=0.41129104  MISMATCH        N=3  cos=0.05935857  MISMATCH
```

After: all three `cos=1.0`. Fix reshapes to the true `[OL, N, OC]` then permutes;
the `N == 1` branch is the *unmodified original statement*, so batch-1 callers
are bit-identical **by construction**, not merely by test.

Corroborating facts:

- **Upstream `llama.cpp` has byte-identical code.** Not a fork regression, and
  not fixed upstream.
- **Upstream `test-backend-ops.cpp` has ZERO `conv_1d` cases.** It covers
  `IM2COL` and `MUL_MAT` as ops, but `ggml_conv_1d` is a composite graph
  builder, so the reshape between them is untested. That is the mechanism by
  which this survived.

#### ✅ AUDIT COMPLETE — landed in the fork (`CrispStrobe/ggml@662b05fb`)

The open question was whether any existing caller passes N > 1. Answered, and
**my original safety argument was wrong**:

- **CrispEmbed: zero `ggml_conv_1d` callers.** Unaffected entirely. (Its only 1-D
  conv use is two `ggml_conv_1d_dw` calls, a different function, both N == 1.)
- **CrispASR: 141 call sites** — 11 more than my `grep` found, because
  `ggml_conv_1d_ph` forwards to `ggml_conv_1d` without matching the literal
  string. **136 pass N == 1.** **2 pass N > 1.** 0 unknown.

The two batched callers are `aa_snake_beta_native` in `src/indextts_voc.cpp`
(:508 and :551), which deliberately maps **channels onto the batch axis** so one
depthwise FIR runs across all C channels at once. So "the N == 1 branch is
unmodified, therefore every caller is bit-identical" was **false** — those two
take the new branch.

They are safe for a *different* reason: their filter is `[K,1,1]`, i.e.
**OC == 1**, and with OC == 1 both branches produce the identical flat layout
`n*OL+ol` *and* the identical declared `ne`. Confirmed from the source (the
shape is documented at `indextts_voc.cpp:459-460` and enforced by a downstream
`ggml_reshape_2d` nelements assert) and verified empirically on that exact shape
class at N = 1..4. Neither site compensates for the old transpose, so nothing
depended on the broken layout.

**The branches diverge only when N > 1 AND OC > 1** — which no caller in either
repo does. CREPE would have been the first, which is why it surfaced here.

Gates run: standalone repro (both shape classes, all N) cos = 1.0; CrispASR unit
suite **1032/1032**; CREPE parity unchanged at cos = 1.0.

**Companion, now also landed (`CrispStrobe/ggml@655c14e4`): `ggml_conv_1d_dw`
batch support.** The first description of this (mine, repeating an agent's) was
wrong: it does NOT silently drop the batch dim. It reshapes to `[T,1,C,N]` and
hits `GGML_ASSERT(b->ne[3] == 1)` at `ggml.c:4476`, i.e. it **aborts** — a safe
failure, an unsupported case rather than a correctness bug. Verified by probing
it rather than reading it. Fixed by folding the batch into the channel axis and
tiling the kernel with `ggml_repeat`; verified N = 1..4 at cos = 1.0,
max_abs = 0.0 exactly. No existing caller changes (N == 1 path untouched, and
nothing can depend on an abort).

### CREPE weights are published

**https://huggingface.co/cstr/crepe-GGUF** — all six files (f16/q8_0/q4_k ×
tiny/full), `license: mit` verified present on the card via
`model_info(expand=["cardData"])`, public, ungated. So `-m auto` /
`--auto-download` now resolves. Published deliberately *before* the accuracy
eval, so that eval can be run on real music from the published artifacts.

#### ✅ ACCURACY EVAL ON REAL MUSIC — the octave concern did NOT reproduce

Run on 10 monophonic instrumental recordings (violin arco + pizz, piano, glock,
carillon, cello, flute, three folk melodies, brass), `tools/crepe_music_eval.py`.
No hand-labelled F0, so two load-independent proxies: **tiny-vs-full octave
disagreement** (|log2(a/b)| ~ 1) and **in-tessitura rate** on voiced frames
(voiced_prob >= 0.5).

| | tiny | full |
|---|---|---|
| in-tessitura (weighted) | **89.6%** | 89.0% |
| voiced frames | 8166 | 9165 |
| octave disagreement tiny-vs-full | **2.3%** | — |

**Conclusion: `crepe-tiny` is NOT meaningfully worse than `crepe-full` on real
monophonic music** — 0.6 pt apart on in-tessitura, 2.3% octave disagreement, and
on `10_amazing_brass` tiny is actually *better* (92.8% vs 83.9%). Given tiny is
38x cheaper (RTF 0.28 vs 2.0), **tiny stays the default.** The earlier
`samples/jfk.wav` octave worry was archival *speech* — out of domain for a model
being shipped for music — and it did not generalize.

Per-clip, the failures are domain limits shared by both capacities, not capacity
defects:
- `02_violin_pizz` — 49% / 52% in range. Plucked, fast-decaying transients; most
  frames have no sustained pitch. Worst clip by far for both.
- `05_carillon_ode` — only 39 (tiny) / 117 (full) of 1501 frames voiced at all.
  Bells are inharmonic; CREPE is trained on harmonic pitch. Correctly abstains
  rather than inventing pitch.
- Clean cases (fur_elise, cello, flute, row_boat, old_macdonald, glock) sit at
  96–100% in range with 0–6% octave disagreement.

⚠️ **Caveats on this eval, stated so it is not over-trusted.** (a) The tessitura
bounds are hand-guessed per instrument, so the *absolute* in-range numbers are
soft — `01_violin_scale` reads 86%/80%, which is more likely my bounds than real
error. The tiny-vs-full *comparison* is the robust part, since both are scored
identically. (b) A first version of this script also reported "fraction within
+/-50 cents of the nearest semitone" at exactly 100.0% for every clip and both
models — that metric is **vacuous by construction** (deviation from the *nearest*
semitone is bounded to +/-50c) and was removed. It is not evidence of anything.
(c) Real per-frame ground truth (a labelled MIR dataset) is still the honest way
to get an absolute note-F number against the handoff's "note-F >= 0.9" gate.

### Two measurement traps hit while benchmarking (both in the dev doc already)

- A run piped to `head -2` reported **0.79 s for 30 s of audio** (RTF 0.026,
  which would have been ~10 TFLOP/s — above M1's FP32 peak). SIGPIPE had killed
  it after two lines. The "too good, and the arithmetic disagrees" smell is what
  caught it; the frame-count-scales check (101 / 1001 / 3001) is what confirmed
  the real runs.
- Load average hit **253** mid-session, making every timing meaningless. Numbers
  above were all re-taken at load 4.0.
- **Branch**: `feat/music-transcription`, worktree
  `.claude/worktrees/music-transcription`.

### CREPE blueprint — the geometry the C++ must hit

Traced from `torchcrepe/model.py` + `core.py` + `convert.py` (the *source*, see
the warning below). Input is a 1024-sample 16 kHz frame, per-frame normalized
(`-= mean`, `/= max(std, 1e-10)`); hop is 10 ms; `pad=True` zero-pads
`WINDOW_SIZE//2` each edge.

Per layer: `F.pad -> conv -> F.relu -> batch_norm -> max_pool2d(2)`.

| layer | K | stride | pad (l, r) | out ch (full / tiny) | T out |
|---|---|---|---|---|---|
| conv1 | 512 | 4 | 254, 254 | 1024 / 128 | 1024 → 256 → 128 |
| conv2 | 64 | 1 | 31, **32** | 128 / 16 | 128 → 64 |
| conv3 | 64 | 1 | 31, **32** | 128 / 16 | 64 → 32 |
| conv4 | 64 | 1 | 31, **32** | 128 / 16 | 32 → 16 |
| conv5 | 64 | 1 | 31, **32** | 256 / 32 | 16 → 8 |
| conv6 | 64 | 1 | 31, **32** | 512 / 64 | 8 → 4 |

Then permute to (T, C) — **C is the fast axis** — flatten to `in_features`
(4 × 512 = 2048 full, 4 × 64 = 256 tiny), `classifier` Linear → 360, sigmoid.
Decode: `cents = 20 * bin + 1997.3794084376191`, `Hz = 10 * 2**(cents/1200)`.

Three traps, all now pinned by `tools/crepe_numpy_parity.py`:

1. **ReLU is BEFORE BatchNorm.** So the conv+BN fold is *invalid*. BN ships as a
   standalone per-channel affine (`_BN.scale`, `_BN.offset`, computed in f64).
2. **conv2..6 padding is asymmetric (31, 32)** and Metal rejects an asymmetric
   `GGML_OP_PAD` — use symmetric `p=32` and drop output column 0.
3. **`torchcrepe.convert.bins_to_cents` applies dithering** (triangular noise),
   so the reference is *non-deterministic*. Disable it when dumping parity
   fixtures, and do not implement it in C++. Also note torchcrepe's default
   decoder is **Viterbi**, not the handoff's weighted-average-around-argmax —
   implement `local_average` (original CREPE) and treat Viterbi as optional.

> ⚠️ **Lesson (HARD RULE #1, the expensive way).** The first converter folded BN
> into the conv, because a fetched *summary* of `model.py` listed the ops as
> "Batch Norm ... ReLU activation" in that order. The real source has the relu
> first. The failure looked like plausible numerics, not a structural bug: layer
> 1 at cos=0.83 with ~2× the reference magnitude — because least-squares fitting
> an affine through a *rectified* signal recovers about half the true scale. What
> caught it in one run was printing `|mine|` and `|ref|` per stage and noticing
> `|mine|` was **identical across four different input frames**. A fetched
> summary of source is not reading the source.

---

## Verdict: yes for the neural models, no for two of the seven

The handoff lists 7 workers. They are not the same kind of thing — four are
neural models (a CrispASR port makes sense), two are pure score-level algorithms
(they belong in Dart), and one is already shipped here.

| Worker | CrispASR? | Status / why |
|---|---|---|
| **W-SEP** | ✅ **already done** | HTDemucs (`src/htdemucs.cpp`, §248 full parity, `cstr/htdemucs-GGUF`) + Mel-Band RoFormer (`src/mel_band_roformer.cpp`, waveform bit-exact 2.4e-7). Both shipped with `--separate`, auto-download, C ABI, Python `Session.separate()`. **Don't export Open-Unmix to ONNX — call CrispASR.** |
| **W-CREPE** | ✅ port — start here | 6-layer 1D CNN on raw 16 kHz audio → 360-bin activation. No STFT, no attention, MIT. The single easiest port in the repo's history. |
| **W-PIANO** (slice 1) | ✅ port | Kong/ByteDance high-res piano CNN + biGRU on log-mel. `core/mel.h` covers the front-end; needs a **GRU** in `core/` (only LSTM exists today). |
| **Basic Pitch** | ✅ port | Already ONNX in the app; Apache-2.0, ~4 MB CNN over a harmonic-CQT stack. Needs a **CQT** front-end (absent). |
| **W-HARMONY** | ⚠️ port, licence-gated | Small CRNN/CQT chord model. Architecture is easy; the work is finding a checkpoint whose **licence** is actually permissive. Timebox the checkpoint hunt before the port. |
| **W-DRUMS** | ⚠️ mostly DSP | Onset + band-energy classification is DSP, and DSP belongs where the app is. Only worth a backend if a permissive drum-transcription CNN is chosen. |
| **W-MT3** (slice 2) | ⚠️ frontier, timebox | T5 encoder-decoder over spectrogram frames → MIDI-like tokens, Apache-2.0. The *architecture* is well-trodden in ggml (easier than ONNX, honestly). The risk is the **checkpoint format** — T5X/JAX gin, not HF safetensors — so the converter is the whole job. Feasibility memo before committing. |
| **W-METRE** | ❌ **not CrispASR** | Downbeat DP + metrical quantisation. No model, no tensors. Pure algorithm over a `RhythmGrid`. Keep in Dart. |
| **W-NOTATION** | ❌ **not CrispASR** | Voice separation, staff split, enharmonic spelling — operates on `crisp_notation` score types, not audio. Keep in Dart. |

So: **5 of 7 are worth porting, 1 is already done, 2 should stay in Dart.**

### Why port at all, given ONNX works

1. **W-SEP is the handoff's "biggest lever" and it already exists here**, at
   higher quality than the Open-Unmix fallback the handoff proposes, with
   per-stage cosine parity already validated. That alone justifies the seam.
2. **One runtime for the whole chain.** Separation → F0 → notes currently means
   ONNX Runtime *plus* whatever runs the stems. CrispASR already owns the audio
   IO, resampling, chunking, and model auto-download.
3. **Quantization.** `crispasr-quantize` gives q8_0/q4_k for free; these models
   ship as f32 ONNX. CREPE-full at q8_0 is a phone-sized model.
4. **Metal / CUDA / WASM** come from ggml, not from a per-model ONNX EP story.

The counter-argument is honest and worth stating: for **Basic Pitch and CREPE
specifically**, ONNX already works in the app today, and porting buys speed and
packaging, not capability. The capability wins are W-SEP (done), piano, and MT3.

---

## Architecture: a new task surface, not a `transcribe()` overload

`docs/source-separation-surface.md` already settled this argument for stems: a
task that returns something other than `crispasr_segment`s must **not** be
layered onto `transcribe()`; it gets its own early dispatcher before the ASR
backend is constructed. Music transcription (audio in → note events out) is the
same shape, so it copies that design:

- `src/core/note_events.h` — the result surface, mirroring
  `src/core/separation_io.h` (header-only, unit-testable without linking a
  backend). Carries the Dart-side seam types: `{midi, onMs, offMs, confidence}`
  note events, `{timeMs, f0Hz, voicedProb}` pitch frames.
- `examples/cli/crispasr_music_cli.{h,cpp}` — early route, mirroring
  `crispasr_separate_cli.{h,cpp}`, hooked once from `cli.cpp`.
- `CAP_MUSIC_TRANSCRIBE = 1u << 24` (bit 23 now belongs to `CAP_STREAMING`
  after the collision fix; 22 stays `CAP_SEPARATE`).
- A MIDI writer in `core/` so the CLI can emit `.mid` directly. MusicXML
  engraving stays in Dart — that's `crisp_notation`'s job, not a C runtime's.

**Contract compatibility.** The handoff freezes `contracts.dart`
(`PitchFrame` / `NoteEvent` / `RhythmGrid`). `core/note_events.h` is designed to
be a 1:1 memory-layout match so the Dart FFI binding is a reinterpret, not a
marshal. That is the whole point of the seam — an engine swaps behind it.

---

## Phase 0 — infrastructure (blocks everything else)

The survey turned up three real gaps. None is hard; all are prerequisites.

1. **`core/stft.h` — forward STFT.** `core/istft.h` exists but covers only the
   inverse. HTDemucs rolls its own (`src/htdemucs.cpp:548` `compute_stft`) and
   mel-band-roformer has a second copy. A music backend would be the **third**
   copy. Extract now, before adding to the pile.
   ⚠️ This refactors two *shipped* backends → per the A/B rule, it needs
   byte-identical stem output on both before it lands, gated if not.
2. **`core/cqt.h` — constant-Q / harmonic-CQT.** Absent entirely. Basic Pitch
   and every chord model want log-frequency bins. Built on (1).
3. **`core/gru.h`.** `core/lstm.h` has uni/bidirectional LSTM; the piano model
   needs biGRU. Mirror the LSTM file's structure.

Ordering: (1) → (3) can proceed in parallel with CREPE, which needs neither.

## Phase 1 — CREPE (recommended first backend)

Why first: it needs **zero** new infrastructure. Raw 16 kHz waveform in
(1024-sample frames), 6 conv+batchnorm+maxpool blocks, one 360-unit dense layer
out. No STFT, no attention, no autoregression, no tokenizer. It exercises the
entire new music surface end-to-end — CLI flag, capability bit, note-event
result type, converter, registry, C ABI, bindings — against the simplest
possible model, which is exactly how you want to debug a new surface.

- `models/convert-crepe-to-gguf.py` — from the MIT Keras/`torchcrepe` weights.
- `src/crepe.{h,cpp}` + the 12-point checklist in `docs/contributing.md`.
- Parity: `tools/reference_backends/crepe.py` → `crispasr-diff crepe`, per-stage
  cos ≥ 0.999 vs `torchcrepe`.
- **Acceptance is the decoded output, not cosine** (HARD RULE #3): synth a
  C-major scale → `crepe` → note segmentation → note-F ≥ 0.9 with **zero octave
  errors**, which is the specific failure the handoff wants fixed.

## Phase 2+ — piano, Basic Pitch, harmony, drums, MT3

Sequenced after phase 1 proves the surface. Each follows the same regime:
blueprint read line-by-line → converter → per-stage diff → decoded-output gate →
registry + 12-point checklist. MT3 gets a feasibility memo (checkpoint
conversion viability) **before** any C++ is written.

---

## Licence scoping of the remaining roster (2026-07-20)

Every candidate below was checked against the CometBeat HARD RULE — patent-free
and MIT/Apache-2.0-compatible — by reading the actual LICENSE file or HF card,
not from memory. **Code licence and WEIGHTS licence are tracked separately**,
because for chords they diverge and that divergence is the whole problem.

| Component | Code | Weights | Verdict |
|---|---|---|---|
| CREPE (marl, torchcrepe) | MIT | MIT | ✅ shipped |
| onnxcrepe (yqzhishen) | MIT | converted from torchcrepe + TF CREPE | ✅ useful as an ONNX cross-check |
| mangio-crepe (Mangio-RVC-Fork) | MIT | — same CREPE weights | ✅ **nothing to port** — see below |
| RMVPE (Dream-High) | **Apache-2.0** | **MIT** (`lj1995/VoiceConversionWebUI`) | ✅ clean — best quality tier |
| FCPE (CNChTu/TorchFCPE) | MIT | MIT repo | ✅ clean — cheapest tier |
| w-okada/voice-changer | MIT (6 holders incl. RVC, yxlllc) | mixed; **Beatrice v2 is a custom licence** | ⚠️ integration *reference* only — do NOT vendor |
| anyf0 (SoulMelody) | MIT | wraps crepe/fcpe/rmvpe | ✅ good reference implementation |
| Basic Pitch (Spotify) | Apache-2.0 | Apache-2.0 | ✅ clean — blocked on CQT, not licence |
| piano_transcription (Kong) | MIT | MIT | ✅ clean — in flight |
| **BTC-ISMIR19 (chords)** | **MIT**, ships `btc_model{,_large_voca}.pt` | trained on **Isophonics = CC BY-NC-SA** | ⚠️ **THE GATE** |
| MT3 | Apache-2.0 | T5X/JAX gin checkpoint | ⚠️ converter is the whole job |
| madmom / Essentia / aubio / Vamp | GPL/AGPL + Böck patents | — | ❌ excluded by the hard rule |

### mangio-crepe needs no port

It is **not a different model**. Mangio-RVC-Fork's contribution is a
*configurable `crepe_hop_length`* on the same MIT CREPE weights; its own README
recommends upstream RVC's CREPE for artifact handling. Our `src/crepe.cpp`
already exposes hop as a parameter, so this is covered. Worth stating plainly so
nobody spends a week on it.

### The chord problem is DATA provenance, not code

BTC-ISMIR19 is the obvious port — MIT code, pretrained checkpoints committed to
the repo, architecture we can already build (bi-directional self-attention over
CQT; every op exists in the CrispASR ggml stack). The catch is upstream of the
code: its checkpoints were trained on **Isophonics annotations, which are
CC BY-NC-SA** (non-commercial, share-alike), as are Robbie Williams and
UsPop2002.

Whether NC-licensed *annotations* encumber the resulting weights is legally
unsettled, and the repo ships them under MIT. But "unsettled" is not the bar
this project set. Three options, in order of preference:

1. **Retrain on ChoCo's permissive subset.** ChoCo aggregates 18 chord corpora
   under **CC BY 4.0**, with only three NC exceptions to exclude (Chordify
   Annotator Subjectivity, Mozart Piano Sonata, JAAH). That leaves Billboard,
   Real Book, RWC-Pop, Weimar Jazz, Wikifonia, iReal Pro, Band-in-a-Box, When in
   Rome, Rock Corpus, Nottingham, Schubert-Winterreise — ample for a small CRNN,
   with commercially-clean provenance we can state in the model card.
2. **Synthetic audio.** There is recent work on training chord recognisers on
   artificially generated audio (arXiv 2508.05878). Rendering progressions from
   permissive symbolic sources gives *fully* clean provenance and pairs well
   with option 1 as augmentation.
3. **Ship the chroma-template path** (already in `crisp_notation`
   `chroma_analysis.dart` / `analyze()`) as the default and treat the neural
   chord model as a later premium tier.

### DECISION (2026-07-20): port BTC now, gate the weights non-commercially

Superseding the "train first" recommendation above. We ship the BTC port with
its upstream checkpoints, treated as **non-commercial weights behind a
download-time attestation** — the same posture the repo already takes for
Voxtral-4B-TTS (CC-BY-NC-4.0) and the German moonshine models (CC-BY-NC-SA-4.0).

Why this is sound:

- **Our code stays MIT.** The BTC architecture is written from the paper into
  the CrispASR ggml stack; nothing GPL/AGPL is copied. CrispASR and CometBeat
  can both be commercial.
- **The restriction rides with the WEIGHTS, not the software.** Users obtain
  NC weights only after attesting they will not use them commercially, and the
  restriction is stated in the registry, the model card and the CLI.
- **Commercial users are not blocked** — they get the chroma-template path
  (already in `crisp_notation`), and later the ChoCo-trained Apache-2.0 model.

Training a clean model on the ChoCo CC-BY subset (option 1 above) remains the
target for a *commercially usable* chord tier. It is now a follow-up, not a
prerequisite, and the BTC port is what proves the architecture + surface first.

#### Required mechanism — MIRROR CrispEmbed, do not invent one

**`--i-have-rights` is the WRONG flag.** It attests *speaker consent for voice
cloning* — a third-party-rights question. Licence compliance is unrelated, and
one flag must not silently grant two different permissions. A separate mechanism
is required.

**CrispEmbed already has the right one** (`examples/cli/model_mgr.{h,cpp}`), and
it is stricter than anything CrispASR does today. Mirror it rather than inventing
a parallel design — the two repos should behave identically:

| CrispEmbed (existing) | CrispASR (to add) |
|---|---|
| `license_requires_acceptance(spdx)` — `cc-by-nc*`, `gemma`, `llama*`, `lfm1.0`, `other` | same predicate, same tag list |
| `resolve_model(arg, auto_download, accepted_license)` | extend `crispasr_resolve_model()` with the same parameter |
| `--accept-license <spdx>` | `--accept-license <spdx>` |
| `CRISPEMBED_ACCEPT_LICENSE` env fallback | `CRISPASR_ACCEPT_LICENSE` |
| accepts the exact SPDX tag, or `all` / `*` | same |
| TTY: prints licence + model-card URL, prompts `[y/N]` | same |
| non-TTY without acceptance: **refuses** | same |
| **`auto_download` alone is NOT sufficient** | same — this is the key property |

Two things CrispEmbed's design gets right that a blanket NC flag would not:

1. **Acceptance is per-licence, not blanket.** The user attests to a *specific*
   SPDX tag; `all` exists but is opt-in.
2. **SPDX tags, not substring matching.** CrispASR currently tests
   `license.find("NC")`, and that same NC-detection logic is duplicated in three
   places (`crispasr_model_registry.cpp` + two spots in
   `crispasr_model_mgr_cli.cpp`). Moving to SPDX tags de-duplicates it.

What CrispASR must change:

- Registry `license` becomes an SPDX tag (`cc-by-nc-sa-4.0`) with the prose
  reason kept separately, so the predicate is exact rather than a substring hit.
- `print_license_note()` currently fires AFTER `ensure_cached_file()` — an NC
  model is already on disk when the warning prints. The gate must precede the
  fetch, and must also cover the **cached-hit early return**, which today skips
  the notice entirely.
- Put the predicate + acceptance check in the library so CLI, session C-ABI and
  server all inherit it (the multi-surface rule).

CometBeat mirrors the same gate in its model store before fetching, and states
the restriction in the UI at the point of download — not buried in an About box.

Follow-up once the ChoCo-trained model exists: it is Apache-2.0, needs no gate,
and becomes the default; BTC stays as the opt-in higher-accuracy NC tier.

### CQT is the shared unlock

Basic Pitch and the chord CRNN both need a **constant-Q transform**, which
`core/` does not have (only `core/mel.h` and `core/fft.h`). Building
`core/cqt.h` once unblocks BOTH, and is the highest-leverage remaining
infrastructure item — ahead of either model port.

### F0 tier — CREPE is shipped, RMVPE is the quality upgrade

The handoff already flags RMVPE as "the quality tier after CREPE", and the
licence check confirms it is clean (Apache-2.0 code, MIT weights). It is also
what w-okada's guide recommends for all-purpose use, and it is robust to
accompaniment — which matters because our W-SEP stems are not perfectly clean.
FCPE is the cheap tier if CREPE-tiny proves too slow on low-end hardware.
Priority: RMVPE > FCPE, and neither is urgent while CREPE-tiny hits RTF 0.28.

---

## Open questions

- **Where does the app call this from?** CrispASR has Dart/Flutter bindings, so
  the seam can be FFI. But the handoff's engines are `!kIsWeb`-guarded with a
  pure-Dart web fallback — CrispASR's WASM build could actually *remove* that
  caveat. Worth confirming with the app author before designing the binding.
- **Model hosting.** Existing convention is `cstr/<name>-GGUF` on HF with a
  `license:` YAML tag that must be verified post-upload. CREPE (MIT), RMVPE
  (MIT weights), Basic Pitch (Apache-2.0) and piano_transcription (MIT) are all
  clean. The chord checkpoint is the one that does NOT survive vetting — see the
  licence scoping above; the plan is to train rather than port those weights.
- **Does the chord model need to be neural at all for v1?** The chroma-template
  path already exists in the app. If the ChoCo permissive subset proves thin,
  shipping DSP chords + a documented "premium tier later" may be the better
  trade than a weak model with clean provenance.

## Suggested order (highest leverage first)

0. **Licence-acceptance gate, ported from CrispEmbed** — land BEFORE any NC
   weights are registered, so there is never a window in which they are
   downloadable ungated. Also de-duplicates CrispASR's three copies of
   substring-based NC detection.
1. **`core/cqt.h`** — unblocks Basic Pitch AND the chord model. Infrastructure,
   no licence risk, reusable.
2. **BTC chords** — architecture from the paper, weights gated NC.
3. **Finish piano_transcription** — currently cos 0.971, below the 0.999 gate.
   It is the closest thing to a finished port that is not yet finished.
4. **Basic Pitch** — Apache-2.0 end to end, and the app already depends on it
   via ONNX, so this is a like-for-like replacement with a known-good oracle.
5. **RMVPE** — clean licence, real quality win on sung f0 over accompaniment.
6. **ChoCo-trained chord model** — the commercially-clean tier, Apache-2.0.
7. **MT3** — feasibility memo on the T5X checkpoint conversion FIRST.

Everything above is CPU/Metal-verifiable locally. The **Kaggle/CUDA run should
wait until this roster is complete**, so one clean CUDA session covers every
backend at once rather than being repeated per port.
