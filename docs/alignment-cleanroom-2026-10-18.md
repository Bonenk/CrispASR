# Alignment comparison and clean-room work (#217, 2026-10-18)

The full [#217 thread](https://github.com/CrispStrobe/CrispASR/issues/217) was
reviewed, including the reporter's October 17 follow-up. The referenced
Subtimer default branch (`wip`) currently has one parentless public commit:
[`4a894d0e166191d41ebc734b61d68f5c531fb55b`](https://github.com/Jerry-at-GH/subtimer/commit/4a894d0e166191d41ebc734b61d68f5c531fb55b),
dated October 5. There is no earlier public revision here against which to
reproduce the reported accuracy improvement.

## What can actually be compared

| Aspect | Subtimer at the pinned commit | Current CrispASR |
| --- | --- | --- |
| Main output | Ordered Japanese dialogue-line intervals | Standalone text/SRT alignment; word/character timing where supported; cue-preserving SRT output |
| Input timing guides | Optional guide segments | Align-only SRT cue text/structure is retained, but original cue times are not used as alignment priors |
| Acoustic dependency | Its README requires external CrispASR bundles and a Japanese Parakeet model, plus other speech dependencies | Native ggml encoders and separate CTC/forced-aligner families |
| CTC path search | Includes Triton GPU and Numba CPU kernels | Current character and subword Viterbi loops run on CPU, independently of the acoustic encoder device |
| Boundary method | Research identifies several speech signals, alignment proposals, and fitted boundary refinement | Native aligner timing, transcript normalization, cue/audio interval handling; no equivalent imported fitted selector |
| Published matched benchmark | None found in the inspected release | No matched run against this release has been performed |
| Runtime requirements | README targets Linux x86-64, CUDA 13 and Python 3.12 | Backend/platform support depends on the chosen native model/build |

The [pinned README](https://github.com/Jerry-at-GH/subtimer/blob/4a894d0e166191d41ebc734b61d68f5c531fb55b/README.md)
is an installation/interface description, not an accuracy or speed report.
No named held-out corpus, boundary-error distribution or hardware-pinned timing
table was found by the research agent. Thus neither "theirs is more accurate"
nor "ours is faster" is established. The old six-minutes-per-episode report in
#217 concerns an older CrispASR/model setup and is not a current controlled
comparison. Millisecond output units alone do not establish millisecond accuracy.

A metadata-only inspection confirms JIT declarations in the pinned
[`ctc.py`](https://github.com/Jerry-at-GH/subtimer/blob/4a894d0e166191d41ebc734b61d68f5c531fb55b/subtimer/ctc.py#L12).
This does not establish which path dominates runtime. Neither an assumed DP
bottleneck nor an assumed small quality cost from banding is accepted as a
measured fact. Our first optimization below does not add GPU Viterbi.

Wall time must be split into decoding/preprocessing, encoder inference, CTC
path search, boundary refinement and output. A faster DP kernel does not prove
a faster complete multi-model pipeline. Likewise, a long-file memory failure
can be in the acoustic graph even if the CPU alignment workspace is compact.

## Source and implementation separation

GitHub's repository metadata has no project-wide license declaration; the root
tree and packaging metadata inspected do not supply one. MIT notices exist in
`subtimer/native/LICENSE` and `subtimer/vendor/CrispASR-LICENSE`; an Apache notice
exists in `subtimer/vendor/LICENSE`. These scoped notices are not treated as a
grant for the remaining project or fitted assets. Under the maintainer's explicit
constraint, no Subtimer code, patches, binaries, model weights, fitted tables or
coefficients are ported or vendored.

A fresh research-only agent inspected external text read-only in an isolated
workspace. It did not clone/install/run the project. The implementation agent
received our source and independently derived functional requirements only,
with no external source/history supplied, and explicit instructions forbidding
external-source, research-workspace or network access. Global memory/rule injection was disabled for that fresh agent.
The maintainer independently reviews its patch and supplies baseline/oracle tests.
Local research/session locations remain in ignored operating notes.

Generic ideas retained for independent future work are speech-aware bounded
windows, independently evaluated vocal/VAD signals, confidence-checked time
anchors, robust handling of optional guide offsets, and boundary refinement
trained on our own licensed labels. These are research directions, not an
instruction to reproduce the external feature recipe or trained selector.
[NVIDIA's NFA documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tools/nemo_forced_aligner.html)
independently documents CTC alignment, separate encoder/Viterbi devices,
chunked processing and configurable segmentation. Approximate bands or extra
models require their own quality evidence; their benefits are not assumed.

## Independent optimization and acceptance

Our current `src/align.cpp` character CTC postprocessor normalizes and stores
all `T * V` emissions, allocates a traceback row per frame, and rescans the path
for every word. The separate `canary_ctc_align_words` subword implementation
serves Parakeet/Canary alignment; a character-path speedup must not be reported
as a Japanese Parakeet speedup. Model contexts are already cached in current
`crispasr_aligner.cpp`; the old PERFORMANCE reload warning is stale.

The candidate is opt-in and keeps the existing default. Acceptance compares
serialized word and character timing bit-for-bit against a frozen copy of our
own `8c59ed583` implementation. Independent exhaustive tiny CTC paths check
unique optima; ties, repeated labels, Unicode, unknown labels and failures get
separate regression coverage. Input/model memory and setup are excluded from
kernel heap/time measurements and must not be confused with whole-process VRAM.

The first candidate passed output checks but failed memory acceptance: storing
expanded-state emissions used 161,756 tracked heap bytes versus 62,428 for the
baseline on `T=256, V=16` with 20 repeated two-character words. It was rejected.
Repeated labels/blanks must share unique-token emission columns; correctness
alone is insufficient for accepting an optimization. The failed patch and log
are retained outside the tracked tree. Final measurements and source pins must
be recorded before any default promotion.

## Preliminary local measurements of our kernel

The corrected candidate uses unique emission columns (`T * U`), one contiguous
traceback table and character-derived word bounds. It preserves the full-vocab
normalization operation order and the legacy tie rules. Enable only with
`CRISPASR_CTC_ALIGN_OPT=1`; the default remains the existing path.

These are synthetic character-CTC measurements, **not Subtimer comparisons,
Japanese Parakeet measurements, or end-to-end speedups**. Inputs use `T=256`,
20 repeated two-character words, 59 labels and 4 distinct token IDs including
blank. GCC 13.3, `-O3 -DNDEBUG -fno-fast-math`, single CPU thread on an Intel
Xeon Skylake host; one warm-up per arm and 9 alternating timed calls.

| Vocabulary | Baseline median ms | Opt-in median ms | Tracked peak heap bytes, baseline → opt-in |
| ---: | ---: | ---: | ---: |
| 16 | 0.315 | 0.228 | 62,428 → 46,940 |
| 64 | 0.315 | 0.264 | 114,668 → 50,028 |
| 1,024 | 5.015 | 3.653 | 1,159,180 → 111,500 |
| 4,096 | 17.686 | 11.725 | 4,503,484 → 310,076 |

Timing samples are noisy on this shared host; the measured medians suggest
1.20–1.51x kernel speedups in these fixtures, not a portable guarantee.
The deterministic heap reduction ranges from 1.33x to 14.52x. Every measured
output matches the frozen baseline bit-for-bit, including character spans.
The six acceptance tests also cover 240 randomized cases, 60 unique-optimum
exhaustive-path checks, ties, failures, nonfinite scores and repeated-label
memory behavior. The immutable baseline is our own source from `8c59ed583`;
its SHA256 is `3a09740cb1c4fdaa278c34e63dbd0a613479d13fd02172e436b40f76ab0e9ec8`.
Candidate source SHA256:
`d089c141e03fa61970a5136d1c8f9b92a05e306ae36bab78d6b8b8597b68cfd5`.

The [machine-readable receipt](alignment-ctc-local-2026-10-18.json) retains
all sample ranges and the fact that this was measured as a working-tree
candidate before integration. Its source checksum identifies the exact bytes.
Larger model-free cases and full-library CI must pass on frozen published
source before considering default promotion. This does not reduce the
unmodified raw-logit input or acoustic encoder allocations.

## Separate correctness fix in our subword aligner

Independent inspection of `canary_ctc_align_words` found that it traced back
even when neither terminal CTC state was reachable. A literal production test
with one frame and two distinct transcript labels returned success with zero
word times before the fix. Two identical labels also require an intervening
blank, so they cannot fit into two frames. The function now returns an error
for unreachable or nonfinite terminal states, and rejects null pointers or
out-of-range token IDs before indexing. Known valid paths retain their native centisecond
timestamps. Four model-free tests compile the actual tokenizer and postprocessor
and cover those contracts. This changes error handling, not the subword DP's
speed, encoder computation or tokenization.

## Remaining comparison work

A fair end-to-end comparison needs the same licensed Japanese audio and ordered
transcript, identical cue boundaries, exact model/precision revisions, hardware,
and cold/warm cache state. Measure boundary MAE, median/p95 error, fractions
within declared time tolerances, dropped lines, ordering violations, failures,
wall time and peak RAM/VRAM. Music, noise, repeated dialogue, long sentences,
subtitle omissions and long pauses need separate strata. Preserve whole-line
coverage and original audio offsets when testing chunking.

No external fitted asset may enter that benchmark or the independent training
pipeline under this task's constraint. External pipeline timing remains
unmeasured; the current work measures our own kernel against our own baseline.

## Independent subword workspace experiment

The follow-up is authored from our own `canary_ctc_align_words` source, with no
external implementation or fitted assets. `CRISPASR_SUBWORD_CTC_ALIGN_OPT=1`
enables one-frame log-softmax storage and flat traceback. The default retains
the original implementation, including the separately repaired error contract.
Vocabulary-wide max/sum order, intermediate F32 stores, strict transition ties,
tokenization and word-boundary scans are unchanged. Emission workspace becomes
`V` floats instead of `T * V`; traceback remains `T * (2N+1)` bytes. This is
not GPU Viterbi, encoder chunking or an elimination of the input-logit matrix.

The immutable own reference contains the tokenizer and alignment function from
`deaf9c010acce1410f470d6df421d0f77ff16be2`. Five new tests pass under GCC and
Clang with `-O3 -DNDEBUG -fno-fast-math`: 240 seeded randomized cases, tied and
nonfinite inputs, unknown/byte fallback, repeated labels, Unicode, nonzero blank
and frame duration, output truncation, 60 unique-optimum exhaustive tiny paths,
and allocation/heap acceptance. The four existing subword failure-contract tests
now exercise both gates; all six character tests still pass after extracting
the unchanged allocation tracker into a shared test-only header. The harness
compiles literal production tokenizer/DP functions with only their context
fields; this is not yet full-library/encoder or model acceptance. CTest and the
ordinary Linux unit workflow run the new suite.

Preliminary local synthetic measurements (T=256, 20 words/40 labels, nine
measurements after one warm-up per arm; same noisy shared CPU):

| Vocabulary | Gate off → on median (ms) | Median ratio | Tracked peak heap off → on (bytes) |
| --- | ---: | ---: | ---: |
| 16 | 0.528 → 0.492 | 1.07× | 45,676 → 23,212 |
| 64 | 0.603 → 0.526 | 1.15× | 94,828 → 23,404 |
| 1,024 | 3.487 → 2.047 | 1.70× | 1,077,868 → 27,244 |
| 4,096 | 19.244 → 11.024 | 1.75× | 4,223,596 → 39,532 |

Each candidate call preserves the frozen reference's exact return code, word
bytes and centisecond bounds. All shapes reduce counted allocations from 311
to 54. Timing ranges overlap and show substantial noise; do not interpret these
medians as a guaranteed speedup. The [raw receipt](alignment-subword-local-2026-10-18.json)
records all ranges, CPU/compiler/flags and source hashes. Its measured source
was a dirty draft on the claim commit, identified by exact source SHA256; no
claim that the commit alone contains that draft is made. Heap counters cover
live C++ allocations inside alignment only, excluding input/model/vocabulary
setup, output allocation, IPC and serialization; they are not RSS or VRAM.

`tools/ci-heavy/alignment_subword_bench.py --large` adds hosted long shapes.
Audit its immutable source and terminal artifacts before accepting hosted
numbers. Keep the gate default-off until full-library CI/lint and real model
alignment acceptance pass. A matched Japanese clip still needs encoder versus
postprocessor stage timings and boundary/coverage evidence; the synthetic
workspace result does not establish a Subtimer or whole-pipeline ranking.
