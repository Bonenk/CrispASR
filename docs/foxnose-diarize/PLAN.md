# #324 — FoxNoseTech/diarize port

Issue #324 (external reporter) asks for FoxNoseTech/diarize as an alternative
to pyannote. It is not a model port: it is ~1,350 lines of Python glue over
pip packages. The genuinely new work was one embedding model plus ~800 lines
of numerics.

## NOW — active work

Embedder, clustering, smoothing, pipeline, DER harness, CLI wiring and
word-aligned segment splitting all landed. **Automatic speaker counting is the open problem** (below). Not yet
done: WeSpeaker GGUF upload + CC-BY attribution, THIRD_PARTY_NOTICES entry,
docs/architecture.md section, session ABI, real-audio DER.

## What was reused vs. built

| stage | source |
|---|---|
| VAD | already in-tree (Silero) — not rebuilt |
| speaker embedding | **new**: WeSpeaker ResNet34-LM, `src/wespeaker.{h,cpp}` |
| clustering | **new**: `src/core/spectral_diarize.{h,cpp}` |
| temporal smoothing | **new**: `src/core/diarize_smooth.{h,cpp}` |
| orchestration | **new**: `src/core/foxnose_pipeline.{h,cpp}` |
| metric | **new**: `src/core/der.h` |

## Licensing

The repo stays MIT. Three separate questions, only one a constraint:

* **WeSpeaker weights are CC-BY-4.0**, not Apache-2.0 as FoxNose's README
  states. Weights are not code, so this does not touch our source licence —
  but redistributing the GGUF REQUIRES attribution in the model card and
  THIRD_PARTY_NOTICES.txt.
* **FoxNose and wenet-e2e/wespeaker code are Apache-2.0.** Apache is
  compatible with MIT but does not become MIT; copied expression would stay
  Apache and make the repo mixed-licence.
* **Clean-room keeps it uniformly MIT**, and here that was nearly free:
  `clustering.py` is a sequence of scikit-learn CALLS, not implementations, so
  there was nothing to translate. What was taken is the recipe and the tuned
  constants — ideas and parameters, not copyrightable expression. The upstream
  is imported by the reference dumper as an ORACLE only, never vendored.

## Acceptance

Bit-exact sklearn parity is unachievable and always would be: k-means++
seeding, GMM init and the ARPACK eigensolver all ride sklearn's RNG stream.
So the gates are known-answer unit tests plus DER, not label equality.

WeSpeaker embedder, per-stage vs the upstream oracle on samples/jfk.wav:

| stage | cos_mean |
|---|---|
| fbank | 0.999999 |
| stem / layer1-4 | 0.99997 - 0.999995 |
| stats | 0.999999 |
| embedding | 0.999997, cosine(emb, ref) = **0.99999747** |

Clustering + smoothing + pipeline: **375 assertions over 57 hermetic cases**
(no model, no audio, no network), including an end-to-end synthetic
2-speaker timeline at **DER 0.0**.

## Speaker counting — solved by switching estimator

The upstream estimator (PCA + full-covariance GMM BIC, refined by silhouette)
does not work on real speaker embeddings. Silhouette saturates and climbs
monotonically to whatever ceiling it is given: within-speaker cosine is only
~0.595 while cross-speaker is ~0.100, so splitting a real speaker keeps
cutting the intra-cluster term while the inter-cluster term barely moves.

Replaced by the **eigengap** of the normalised Laplacian — the standard
estimator for spectral diarization. It reads cluster structure off the
spectrum rather than scoring partitions, so saturation does not arise.

One thing it needs that the upstream affinity does not provide: the cosine
affinity `(cos+1)/2` is DENSE, sitting near 0.5 even for unrelated windows, so
the graph is nearly complete, the spectrum has one dominant eigenvalue, and
the largest gap is always at k=1 — a naive eigengap reports one speaker for
everything. Row-wise thresholding (keep each row's strongest 15%, attenuate
the rest by 0.01 rather than deleting them so the graph stays connected, then
symmetrise with an elementwise max) restores the block structure.

Measured:

| | synthetic, 5 true-k cases | real audio, `max_speakers=8` |
|---|---|---|
| BIC + silhouette (upstream) | 4/5 exact | **7-8 speakers** — saturated |
| eigengap, no thresholding | 0/5 — always k=1 | — |
| **eigengap + row thresholding** | **5/5 exact** | **2 speakers, correct turns** |

Eigengap is also cheaper: one eigendecomposition instead of a GMM sweep plus
`max_k` spectral runs for silhouette scoring. It wins on quality AND speed, so
it is the DEFAULT; `CRISPASR_DIARIZE_COUNT=bic` restores the upstream path,
which is gated rather than deleted.

Because eigengap is robust to a loose bound, `--diarize-max-speakers` no
longer has to be defensively small: the foxnose default went back from 4 to 8,
so a genuine 5-6 speaker meeting is reachable again.

Two earlier findings, both still in force for the gated BIC path:

* the upstream component ceiling of `n/2 + 1` is far too loose for a FULL
  covariance (not estimable from fewer than `d+1` points) — bounded by
  `n/(d+1)`;
* the `[k-2, k+3]` window recovers an under-count but not a large over-count
  — `CRISPASR_DIARIZE_FULL_K_SEARCH=1` scores the whole range instead.

## Blueprint parity — measured end to end

The upstream Python pipeline (`pip install diarize==0.1.2`) was run on the same
`samples/multispeaker.wav` and compared against this port.

**With the speaker count pinned to 2 on both sides**, the speaker assignment is
identical and the boundaries agree within ~1 s:

| upstream Python | this port |
|---|---|
| 0.30-10.60 SPEAKER_00 (3 pieces, VAD-split) | 0.00-10.50 SPEAKER_00 |
| 11.50-15.60 SPEAKER_01 | 10.50-15.90 SPEAKER_01 |
| 16.30-26.60 SPEAKER_00 (3 pieces) | 15.90-26.70 SPEAKER_00 |
| 27.50-31.50 SPEAKER_01 | 26.70-31.50 SPEAKER_01 |

Scored with the DER harness (0.25 s collar, optimal 1:1 mapping), treating the
upstream output as reference:

    missed        0.00 s
    false alarm   1.05 s
    confusion     0.00 s
    DER           3.93 %

**Zero speaker confusion**: wherever both assign a speaker, they agree. The
entire residual is false alarm, and it is explained — upstream runs its own
Silero VAD and drops silence gaps, while this port tiles the caller's speech
regions contiguously. That is a difference in where the speech segmentation
comes from, not a diarization disagreement.

**On automatic counting this port is better than upstream.** On the same clip
upstream emits 11 speakers across 25 segments (its default `max_speakers=20`);
this port emits 2. The gated `CRISPASR_DIARIZE_COUNT=bic` path reproduces
upstream's failure mode (7-8 speakers), which is what confirms the port is
faithful — the improvement comes from the eigengap switch, not from a
divergence in the shared parts.

### What would still settle it properly

A DER number on labelled audio. There is none in the repo and none in
`cstr/crispasr-regression-fixtures` — this was checked. VoxConverse dev (what
upstream benchmarks on) needs a ~1-2 GB audio download plus RTTM wiring, which
is its own task. Until then the honest recommendation is a **conservative
`--diarize-max-speakers` default (4-6, not 20)**, and pinning
`--diarize-num-speakers` when the count is known.

## Wiring

`--diarize-method foxnose --diarize-embedder <wespeaker.gguf>`, with
`--diarize-max-speakers` / `--diarize-num-speakers`.

The method plugs into the existing `crispasr_diarize_segments` contract: the
caller's segments ARE the speech regions (they come from ASR/VAD upstream), so
FoxNose deliberately runs no VAD of its own — re-segmenting would duplicate
work and desynchronise labels from the segments they attach to.

Segment splitting IS implemented (`split_segments_on_foxnose_turns`): the
method now returns its derived turns through `crispasr_diarize_segments`'s
`out_turns`, and the CLI splits any caller segment spanning several speakers
at word-aligned boundaries. It reuses the same `group_words_into_speaker_runs`
grouping and sub-segment emission as the pyannote splitter — only the per-word
labelling differs (turn-interval lookup instead of posterior scoring).
Measured effect on `samples/multispeaker.wav`: before, both ASR segments
collapsed to `(speaker 0)`; after, the first slice correctly reads
0 -> 1 -> 0 across the ~11 s boundary.

### Cross-slice speaker identity (fixed)

Per-slice diarization cannot give consistent identities: each slice clusters
independently and restarts numbering at 0, so `speaker 0` in one slice is a
different person from `speaker 0` in the next. On `samples/multispeaker.wav`
(2 slices) the final turn came out `speaker 0` where the whole-file pipeline
says SPEAKER_01.

Fixed by running FoxNose in ONE global pass after transcription
(`crispasr_apply_foxnose_global`), using the final segment list as its speech
regions. The pyannote path solves the same problem with a pre-computed
posterior cache (#107); FoxNose does it afterwards instead, because it needs
the segments as speech regions and they do not exist beforehand. The per-slice
path stands down via `params.diarize_foxnose_global`, so the embedder is
loaded once rather than per slice.

Result on `samples/multispeaker.wav` — CLI output now matches the whole-file
pipeline's turns exactly:

| CLI | pipeline truth |
|---|---|
| 0.28-10.84 speaker 0 | 0.00-10.50 SPEAKER_00 |
| 11.64-15.88 speaker 1 | 10.50-15.90 SPEAKER_01 |
| 17.04-26.80 speaker 0 | 15.90-26.70 SPEAKER_00 |
| 26.52-31.52 speaker 1 | 26.70-31.50 SPEAKER_01 |

Note this runs on the `crispasr_run` unified path. The legacy `cli.cpp`
whisper path still diarizes per slice; it is the same fallback situation the
pyannote cache has there.

## Env gates

| var | effect |
|---|---|
| `CRISPASR_DIARIZE_FULL_K_SEARCH=1` | score the full `[min,max]` speaker range on silhouette instead of `[k-2, k+3]` |
| `CRISPASR_WESPEAKER_BENCH=1` | per-stage embedder timings |
| `CRISPASR_WESPEAKER_DEBUG=1` | embedder diagnostics |
