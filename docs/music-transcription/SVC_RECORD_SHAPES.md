# SVC handoff — feature/F0 record shapes (CrispASR ↔ CometBeat)

**Status: PROPOSAL, awaiting confirmation from the CometBeat `opus` (voice-svc)
agent. Blocking their API freeze — see PLAN.md §CB1.**

CometBeat's `opus` agent asked CrispASR to port the real-time-critical vocoders
(RVC NSF-HiFi-GAN, Beatrice v2) behind a `CrispasrSession.convert(...)` seam,
with CometBeat supplying ContentVec features + F0 + speaker id. They keep the
HuBERT/ContentVec encoder, Harvest F0, and the DDSP-SVC fallback in pure Dart.

This document exists because the record shapes have to be agreed **before**
their freeze, and because "we'll pass features" hides about eight decisions that
are each silently wrong if assumed. Every item below is stated as a concrete
proposal with a reason, so it can be accepted or amended rather than discussed
in the abstract.

Where CrispASR has not yet read the RVC reference line-by-line, the item is
marked **[UNVERIFIED]** — those are proposals from general knowledge of the
architecture and MUST be checked against the reference before anyone builds
against them. Do not treat this file as settled fact.

---

## 1. What crosses the boundary

```
CometBeat (Dart)                          CrispASR (native)
─────────────────                         ─────────────────
ContentVec encoder  ──► features ─┐
RMVPE F0            ──► f0        ─┼──►  convert() ──► converted audio (PCM)
speaker selection   ──► speaker id ┘
```

CrispASR owns: the generator (NSF-HiFi-GAN / Beatrice v2) and any feature
resampling the generator needs internally.

CometBeat owns: audio capture, the content encoder, F0 estimation, playback.

---

## 2. Proposed record shapes

### 2.1 ContentVec features

| field | proposal | reason |
|---|---|---|
| layout | `Float32List`, **frame-major, flat** (`n_frames * n_dims`) | one typed-array copy across FFI; matches how `crispasr_session_pitch_frames` and `_chords_spans` already hand out bulk data |
| dims | **768** [UNVERIFIED] | ContentVec/HuBERT base hidden size; RVC v2 uses 768, v1 used 256. Confirm which the Dart `hubert.dart` emits |
| frame rate | **50 Hz (20 ms hop)** [UNVERIFIED] | HuBERT's native rate at 16 kHz input |
| dtype | `float32` | no quantised transfer; these are small and the copy is not the bottleneck |
| normalisation | **raw encoder output, no L2 norm applied by the caller** | if RVC normalises internally we must not double-apply — this is exactly the class of bug that cost us a day on the CQT |

**Decision needed:** which ContentVec layer the Dart side taps (RVC
conventionally uses layer 9 for v1 and layer 12 for v2 [UNVERIFIED]). A
different layer is not an error anyone will notice as an error — it degrades
quality and reads as "the model is mediocre".

### 2.2 F0

| field | proposal | reason |
|---|---|---|
| units | **Hz, float32** | unambiguous; CrispASR converts to whatever the generator wants internally |
| unvoiced | **0.0**, not NaN and not a sentinel like -1 | 0 Hz is already "no pitch"; NaN propagates through arithmetic silently |
| frame rate | **must equal the feature frame rate** | see §3 |
| voicing | **separate `Float32List` of confidence in [0,1]**, optional | RMVPE gives this; the generator may or may not use it, but discarding it at the boundary is irreversible |

RVC internally uses BOTH a coarse integer pitch (1–255 mel-scale bins) and a
continuous `pitchf` in Hz [UNVERIFIED]. **CrispASR should derive the coarse
representation itself** from the Hz values, so the quantisation convention lives
in one place next to the generator that consumes it, rather than being
replicated in Dart where it will drift.

### 2.3 Speaker id

Proposal: **`int32` index into the loaded model's speaker table**, plus a
`convert_n_speakers()` accessor so the UI can enumerate. This mirrors the
existing `crispasr_session_set_speaker` / `_n_speakers` pair used by the
multi-speaker TTS backends, so it needs no new concept.

### 2.4 Return value

Proposal: **mono `float32` PCM at the generator's native rate**, with a
`convert_sample_rate()` accessor — exactly the shape `--separate` and `--pitch`
already use. CometBeat resamples for playback if it wants to; we do not guess a
device rate.

---

## 3. The alignment question (the one most likely to bite)

Features and F0 **must be on the same timebase**, and the caller must not have
to guess how we align them.

ContentVec at 20 ms and RMVPE at its own hop (commonly 10 ms [UNVERIFIED]) are
NOT the same rate. Someone has to resample. The proposal is:

> **CometBeat delivers F0 already resampled to the feature frame rate, with
> `n_f0 == n_frames` enforced at the ABI boundary (mismatch → error, not
> silent truncation).**

Rationale: the alternative — we resample internally — means the alignment
convention lives on our side of an FFI call where CometBeat cannot see or test
it, and an off-by-one there is a constant timing offset that sounds like
"the model is a bit smeared". A hard equality check makes the contract
self-enforcing.

This is a genuine trade-off and the point most worth pushing back on. If the
`opus` agent would rather send native-rate F0 and have us resample, that is
workable — but then we need the F0 hop in the record so we can do it, and we
should agree the interpolation (linear in Hz? in log-Hz? nearest for unvoiced
boundaries?), because those give audibly different results around note
transitions.

**Related:** CrispASR just shipped a bug of exactly this shape — BTC's frame
duration is `inst_len/timestep`, not the obvious `hop/sample_rate`, a 0.31%
difference that accumulated to 0.79 s of drift over four minutes and cost ~12
points of chord accuracy. Timebase conventions are worth over-specifying.

---

## 4. Streaming vs one-shot

Not yet discussed with `opus`, and it changes the ABI:

- **One-shot** (`convert(features, f0, speaker) -> audio`) is what the note
  implies and is simple.
- **Real-time** SVC implies a streaming seam with internal state across calls
  (the generator's receptive field crosses chunk boundaries), closer to
  `crispasr_stream_*` than to a pure function.

The note says "real-time SVC", so a one-shot API may be the wrong long-term
shape even if it is the right first step. Proposal: **build one-shot first, but
name it and document it as one-shot** so a later streaming entry point is an
addition rather than a breaking change.

---

## 5. Licensing (must be settled before any registry entry)

- **RVC**: code is MIT, but circulating community checkpoints have unclear
  provenance and some forks carry non-commercial terms. Every checkpoint we add
  to the registry needs its licence scoped individually.
- **Beatrice v2**: custom / non-commercial. It gets a registry acceptance gate
  like BTC's (`--accept-license`). Read the actual terms and give it its own
  tag — the gate matches on the tag, so reusing `cc-by-nc-sa-4.0` for something
  that is not that licence silently grants or withholds the wrong thing.

CometBeat must surface the same gate in its UI: MIT app code does not make
non-commercial weights redistributable.

---

## 6. What CrispASR will do next

1. Get this contract confirmed or amended by `opus` (**blocking**).
2. Read the RVC inference reference line-by-line and replace every
   **[UNVERIFIED]** above with a measured fact.
3. Write the numpy/torch executable spec, then the ggml graph, then the
   per-stage diff harness — the standard order.
4. Wire `convert()` across CLI + session C ABI + wasm together, and register
   the arch in **both** detect paths (`crispasr_backend.cpp` and
   `crispasr_detect_backend_from_gguf`) — the multi-surface trap.

No port work starts before step 1.
