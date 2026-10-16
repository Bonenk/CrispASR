# Dia reference/diff wiring — validation in progress

The working candidate registers the original Dia observer in the generic
Python dispatcher and adds a text-only `crispasr-diff dia` route. It replaces
the old reconstructed reference shim with hooks into the pinned original
`generate()` implementation. Runtime changes add diagnostic output marking
and readback; the current F16 cross cache, sampler, graph math, top-k and
stopping/crop behavior are preserved.

Default captures include text IDs/embedding, all12 encoder layers, normalized
encoder output, first cross K/V cache, all18 decoder layers, first cross
attention, final hidden, every decoder input/logit, original post-reversal
codes, all8 DAC stages and decoded audio. A selected decoder step supports
optional self-attention Q/K/V/attention and16 block boundaries at any requested
layer0–17. Requested extras reach the archive even when the generic dispatcher
supplies an explicit default stage set; diff reads their metadata and compares
them. Input IDs are checked exactly before any numerical interpretation.

Local checks use tiny actual original Dia classes, not reconstructed neural
math: all17 observer tests pass. Three public-dispatch/archive tests pass.
Compiled production comparison/input guards pass20 acceptance/rejection
controls against the actual current GGUF reader and ggml libraries. Doubled
scale with cosine1 is rejected, as are nonfinite, missing/truncated tensors and
invalid DAC indices. Three parser/scratch tests and one device/environment
restoration test pass. The existing original output-protocol test passes.
All237 ordered ggml computation calls match pristine9aa8df20; a graph or F16
precision mutation makes that source guard fail. Whole-runtime syntax passes.

These checks establish local contracts, not full-model neutrality or parity.
The [hosted worker](../tools/ci-heavy/dia-reference-wire/README.md) builds
actual pristine/candidate runtimes, checks complete OFF/ON/OFF generation,
and runs the public dumper and strict diff on fresh original intermediates.
It retains failures and complete raw evidence for independent audit. No full
checklist, current-main speech, GPU, Q4 or speed claim follows yet.

Local logs and compiled controls are retained under
/mnt/storage/crispasr/dia-reference-wire-20261016/local-v1. Early failures are
retained: a scratch-test extraction boundary included the new parser, the
public-list test assumed a fixed number of spaces, and the reader test assumed
GGUFWriter stored empty string metadata. The corrected tests isolate the
intended C++ body, match the public list by fields, accept omitted empty audio
metadata and close the actual mmap before cleanup on CIFS.

Candidate [c7f9da87](https://github.com/CrispStrobe/CrispASR/commit/c7f9da872e87ba238c579083841b3547faee772f)
is pushed. Hosted [full-model validation38883779736](https://github.com/CrispStrobe/CrispASR/actions/runs/38883779736),
[lint38883782438](https://github.com/CrispStrobe/CrispASR/actions/runs/38883782438)
and [general CI38883785994](https://github.com/CrispStrobe/CrispASR/actions/runs/38883785994)
are queued at this checkpoint. All4 unskipped no-secrets checks and
clang-format18 pass locally. Candidate code is not integrated on main yet.
The method link above describes candidate files until integration.
