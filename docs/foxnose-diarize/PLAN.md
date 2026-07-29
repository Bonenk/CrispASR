# #324 — FoxNoseTech/diarize port

Issue #324 (external reporter) asks for FoxNoseTech/diarize as an alternative
to pyannote. It is not a model port: it is ~1,350 lines of Python glue over
pip packages. The genuinely new work was one embedding model plus ~800 lines
of numerics.

## NOW — active work

Embedder, clustering, smoothing, pipeline, DER harness and CLI wiring all
landed. **Automatic speaker counting is the open problem** (below). Not yet
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

## ⚠ The open problem: automatic speaker counting

The embedder and the clustering are fine. Counting is not. Measured on
`samples/multispeaker.wav` (31.5 s; opens with all 11 s of `samples/jfk.wav`
then changes speaker):

| config | speakers | turns |
|---|---|---|
| `num_speakers=2` (pinned) | 2 | **correct** — boundary at 10.5 s vs the true 11 s |
| auto, `max_speakers=4` | 2 | **correct**, same turns |
| auto, `max_speakers=8` | 8 | wrong, heavy flicker |
| auto, `max_speakers=8`, `FULL_K_SEARCH=1` | 8 | gate does not help |

Root cause: **silhouette saturates and rises monotonically to the ceiling on
real speaker embeddings.** Within-speaker cosine is only ~0.595 while
cross-speaker is ~0.100 (measured, tests/test_wespeaker_live.cpp), so
splitting one speaker into sub-clusters cuts the intra-cluster term `a`
sharply while the inter-cluster term `b` barely moves. The
`+ 0.04 * log(k)` bonus then pushes the choice to the top of the allowed
range. Upstream defaults `max_speakers` to 20, which is why their README
concedes it "struggles with 8+ speakers".

Two earlier findings, both already fixed or gated:

* the upstream component ceiling of `n/2 + 1` is far too loose for a FULL
  covariance (not estimable from fewer than `d+1` points), so BIC fell
  monotonically — bounded by `n/(d+1)`;
* the `[k-2, k+3]` window recovers an under-count but not a large over-count
  — `CRISPASR_DIARIZE_FULL_K_SEARCH=1` scores the whole range instead (5/5
  vs 4/5 exact on synthetic data), gated off pending DER.

### What would settle it

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

**Consequence worth knowing:** labels are attributed at SEGMENT granularity.
An ASR that emits one 26 s segment spanning several speakers gets a single
label, however good the underlying turns are. Splitting segments at turn
boundaries (as the pyannote path does) is the follow-up.

`--diarize-embedder` was already claimed by the TitaNet remap at THREE call
sites (cli.cpp, crispasr_run.cpp, crispasr_server.cpp). They now all ask
`params.diarize_embedder_is_foxnose()` so a fourth cannot drift — the
multi-surface trap, caught by an end-to-end run rather than by review.

## Env gates

| var | effect |
|---|---|
| `CRISPASR_DIARIZE_FULL_K_SEARCH=1` | score the full `[min,max]` speaker range on silhouette instead of `[k-2, k+3]` |
| `CRISPASR_WESPEAKER_BENCH=1` | per-stage embedder timings |
| `CRISPASR_WESPEAKER_DEBUG=1` | embedder diagnostics |
