# SVC handoff — feature/F0 record shapes (CrispASR ↔ CometBeat)

**Status: PROPOSAL, awaiting confirmation from the CometBeat voice-svc side.
Blocking their API freeze — see PLAN.md §CB1.**

CometBeat asked CrispASR to port the real-time-critical vocoders (RVC
NSF-HiFi-GAN, Beatrice v2) behind a `CrispasrSession.convert(...)` seam, with
CometBeat supplying ContentVec features + F0 + speaker id. They keep the
HuBERT/ContentVec encoder, Harvest F0, and the DDSP-SVC fallback in pure Dart.

This document exists because the record shapes have to be agreed **before**
their freeze, and because "we'll pass features" hides about eight decisions that
are each silently wrong if assumed.

**Provenance.** Every value below was read from the RVC source
(`RVC-Project/Retrieval-based-Voice-Conversion-WebUI`, shallow clone, 2026-07-20),
with a file:line citation. An earlier draft of this file guessed several of
these from memory and got the frame rate wrong by 2x; nothing here is recalled.
Where a decision is still ours to make rather than the reference's, it is
marked **[OPEN]**.

Primary sources:
- `infer/vc/pipeline.py` — the VC pipeline (F0, feature handling, windowing)
- `infer/hubert.py` — ContentVec feature extraction
- `infer/module/models.py` — the generator
- `infer/vc/modules.py` — model load, target rate

---

## 1. What crosses the boundary

```
CometBeat (Dart)                          CrispASR (native)
─────────────────                         ─────────────────
ContentVec encoder  ──► features ─┐
RMVPE F0            ──► f0        ─┼──►  convert() ──► converted audio (PCM)
speaker selection   ──► speaker id ┘
```

---

## 2. THE FRAME RATE — 100 Hz, and features are DUPLICATED to reach it

This is the single most important fact in this document, and the one the
previous draft got wrong.

RVC's pipeline runs on a **160-sample window at 16 kHz — a 10 ms hop, 100 Hz**
(`pipeline.py:53-54`: `self.sr = 16000`, `self.window = 160`). Frame count is
`p_len = audio0.shape[0] // self.window` (`pipeline.py:203`).

ContentVec/HuBERT natively emits **50 Hz** (20 ms). RVC reconciles the two by
upsampling the features 2x (`pipeline.py`, in `vc()`):

```python
feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
```

`F.interpolate` defaults to `mode='nearest'` — verified empirically, not
assumed: `[1,2,3,4] -> [1,1,2,2,3,3,4,4]`. So **each ContentVec frame is
duplicated, not interpolated.** Anyone "improving" this to linear is changing
the model's input distribution.

Consequences for the contract:

- **F0 is native at 100 Hz.** It does NOT need resampling to meet the features;
  the features are brought up to meet it.
- The **[OPEN]** question is only *who does the 2x duplication*. Proposal:
  **CometBeat sends ContentVec at its native 50 Hz and CrispASR duplicates**,
  because the duplication is part of the model's input contract and belongs
  next to the generator that depends on it. That also keeps the wire format the
  encoder's natural output, with nothing to get wrong in Dart.
- Length reconciliation is `min`, not an error: RVC truncates pitch to the
  feature length when features are shorter (`pipeline.py`, `if feats.shape[1] <
  p_len`). We should mirror that rather than reject, but **log** it.

My earlier draft proposed a hard `n_f0 == n_frames` equality check. That was
wrong for this model — after the 2x upsample the lengths can legitimately differ
by a frame, and the reference silently takes the shorter. Match the reference.

---

## 3. ContentVec features

| field | value | source |
|---|---|---|
| dims | **v1: 256**, **v2: 768** | `hubert.py` — v1 returns `final_proj(hidden_states[9])`, v2 returns `last_hidden_state` |
| layer | **v1: hidden_states[9] then `final_proj`**; **v2: final (12th) layer, no projection** | `hubert.py`, code confirmed (not just its docstring) |
| native rate | **50 Hz (20 ms)**, duplicated 2x to 100 Hz | §2 |
| layout | `Float32List`, frame-major flat (`n_frames * n_dims`) **[OPEN]** | our choice; matches `crispasr_session_pitch_frames` / `_chords_spans` |
| normalisation | raw encoder output — RVC applies none before the generator | `pipeline.py` `vc()` |

**Decision needed: v1 or v2.** They are different feature dimensionalities and
different extraction paths, and a checkpoint is trained for one of them. The
Dart `hubert.dart` must declare which it emits; we should carry it as a GGUF
field and refuse a mismatch loudly rather than run and sound subtly wrong.

There is also an optional **FAISS index retrieval** step
(`pipeline.py`, `index.search(npy, k=8)`, inverse-square-distance weighting,
blended by `index_rate`). It is off when no index is supplied. **[OPEN]** —
proposal: out of scope for v1 of the seam; note it so nobody assumes we dropped
it by accident.

---

## 4. F0 — BOTH representations, and we derive one

The generator takes **two** pitch inputs
(`models.py:664`): `infer(phone, phone_lengths, pitch, nsff0, sid, ...)`

- `nsff0` — continuous F0 in **Hz**, float
- `pitch` — **coarse integer 1..255**, mel-quantised

The quantisation, exactly (`pipeline.py:73-137`):

```python
f0_min, f0_max = 50, 1100
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)
f0_mel = 1127 * np.log(1 + f0 / 700)
f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
f0_mel[f0_mel <= 1]  = 1
f0_mel[f0_mel > 255] = 255
f0_coarse = np.rint(f0_mel).astype(np.int32)
```

**Contract: CometBeat sends Hz only; CrispASR derives the coarse form.** The
constants above are model-side, and replicating them in Dart guarantees drift.

| field | proposal |
|---|---|
| units | **Hz, float32** — matches `nsff0` |
| unvoiced | **0.0** (the reference's `f0_mel <= 1 -> 1` path handles it) |
| rate | **100 Hz / 10 ms**, matching `window=160` at 16 kHz |
| voicing | optional confidence in [0,1] **[OPEN]** — see below |

RVC has a `protect` mechanism that blends the pre-index features back in at
unvoiced frames (`pipeline.py`, `pitchff`, default `protect=0.33`) to reduce
artefacts on consonants. It derives its mask from `pitchf > 0`, i.e. from F0
itself — so a separate voicing array is **not required**. Keep it optional.

---

## 5. Speaker id

`self.emb_g = nn.Embedding(self.spk_embed_dim, gin_channels)` (`models.py:626`)
— `sid` is a plain **integer index** into the checkpoint's speaker table.

Proposal: `int32` plus a `convert_n_speakers()` accessor, mirroring the existing
`crispasr_session_set_speaker` / `_n_speakers` pair used by the multi-speaker
TTS backends. No new concept needed.

---

## 6. Return audio

The output rate is **a property of the checkpoint**, not a constant:
`self.tgt_sr = self.cpt["config"][-1]` (`modules.py:116`) — commonly 32k/40k/48k.
RVC optionally resamples to `resample_sr` afterwards (`pipeline.py:374`).

Proposal: **return mono float32 at the checkpoint's native rate**, with a
`convert_sample_rate()` accessor — the shape `--separate` and `--pitch` already
use. CometBeat resamples for playback; we do not guess a device rate.

Note `change_rms` (`pipeline.py:373`, `rms_mix_rate`) blends the source's
envelope back into the output. **[OPEN]** — proposal: expose as a parameter with
the reference default rather than hardcoding.

---

## 7. Streaming vs one-shot **[OPEN]**

Not yet discussed, and it changes the ABI. RVC's own pipeline is chunked with
padding and crossfade (`x_pad`, `t_query`, `t_center`), and the repo has a
separate `infer/rtrvc.py` for the real-time path — so streaming is a distinct
code path upstream too, not a wrapper.

Proposal: **build one-shot first and name it as one-shot**, so a later streaming
entry point is an addition rather than a breaking change. If CometBeat needs
true real-time in v1, say so now — that changes the port target to `rtrvc.py`.

---

## 8. Licensing

- **RVC**: the repo is MIT, but circulating community checkpoints have unclear
  provenance and some forks add non-commercial terms. Scope every checkpoint
  individually before any registry entry.
- **Beatrice v2**: custom / non-commercial → registry acceptance gate like
  BTC's. Read the actual terms and give it its **own** tag; the gate matches on
  the tag, so reusing `cc-by-nc-sa-4.0` for a different licence silently grants
  or withholds the wrong thing.

CometBeat must surface the same gate in its UI: MIT app code does not make
non-commercial weights redistributable.

---

## 9. Next steps

1. Confirm or amend §2 (who duplicates), §3 (v1 vs v2), §7 (streaming) —
   **blocking**.
2. Write the numpy/torch executable spec for the generator, then the ggml
   graph, then the per-stage diff harness. Standard order.
3. Wire `convert()` across CLI + session C ABI + wasm together, registering the
   arch in **both** detect paths — see `docs/contributing.md` section 7.

No port work starts before step 1.
