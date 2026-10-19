# Japanese CTC model profiling and parity

This worker tests **our own** independently implemented subword workspace gate
on actual native logits. No external alignment implementation, fitted recipe
or asset is imported. Production code/defaults are unchanged.

`pins.json` freezes the Japanese Parakeet CTC 1.1B F16 GGUF (Apache 2.0),
FLEURS Japanese test parquet/TSV, exact recording filename and metadata. Sentence
IDs repeat across speakers; matching by ID alone is forbidden. The recording
is CC BY 4.0, attributed to Google FLEURS (Conneau et al., 2022); pinned dataset
card, metadata, source URL, license and transformation notice accompany retained
PCM/WAV. The fixture is 101,760 samples / 6.36 seconds, without cutting or
resampling. It is not the issue reporter's episode or a boundary-labeled corpus.

Run on a frozen source branch:

```sh
gh workflow run heavy-cpu.yml --ref <frozen-branch> \
  -f script=tools/ci-heavy/alignment_model_profile.py \
  -f args='--threads 4 --repeats 5' \
  -f pip='numpy==1.26.4 huggingface_hub==0.36.0 pyarrow==19.0.1 soundfile==0.13.1' \
  -f timeout_minutes=120
```

The worker refuses model downloads/build/inference outside hosted Linux
Actions. CPU work belongs here; Kaggle is reserved for actual GPU validation.
Build flags disable CUDA, Metal, Vulkan and BLAS and retain compiler/cache
records. Downloads and model hashes are pinned; no model is uploaded in the
result artifact. `units.cpp` queries the actual linked library's label/display
splitters. No approximate Python reimplementation chooses alignment labels.

Acceptance, in order:

1. Verify model/parquet/TSV bytes and identify the exact recording, including
   filename, transcript and sample count. Retain PCM and float32 WAV, and verify
   actual C ABI file decoding gives the same float32 samples.
2. Build the actual library and CLI; retain their ELFs/dependencies, sources,
   compile commands and hashes. Load the F16 model once; record first-context
   load time. Hash verification pre-reads the model, so this is not cold-disk I/O.
3. Warm the encoder, then record actual OFF/ON/OFF logits and times. Require
   byte-exact finite full-vocabulary logits. Encoder-call time includes frontend,
   graph/scheduling, computation and readback; it is not a GPU kernel timer.
4. On those frozen logits, warm each DP gate once and measure five cycles of
   OFF/ON/ON/OFF. Preserve all raw calls and exact text/centisecond bounds.
   Require every expected unit, positive nonoverlapping in-range spans and a
   nonempty greedy speech control. This does not measure acoustic accuracy.
5. Replay the physical native logits/vocabulary through literal own production
   tokenizer/DP and immutable own `deaf9c010` reference. Require exact outputs
   against the full library and lower measured tracked heap/allocations. These
   counters are from the standalone literal-function harness, not process RSS
   or the actual library's allocator. Its one-call timings are not benchmarks.
6. Invoke the shipped Python wrapper / exported C ABI with cold and warm cache,
   OFF/ON/OFF and a 12.34-second absolute offset. Require exact native timings,
   actual display text and no fabricated subword character spans. Release cache.
7. Run real CLI processes OFF/ON/OFF with the same WAV/text. Require exact
   native/C ABI word text and bounds. CLI calls include cold loading and are
   parity checks, not a timing comparison.

Failures exit nonzero, retain their phase, raw arrays/calls and traceback, and
produce a failed receipt. Never widen coverage/timestamp gates after a failure.
Independently inspect all raw payloads and their ledger, native source/ELFs,
OFF/ON/OFF tensor equality, call distributions and exact reference/public/CLI
output before accepting a receipt. Timings exclude or include different scopes;
do not add medians or present postprocessor speed as end-to-end speed. Native DP timing includes ctypes
entry overhead. Stage
comparison uses warm encoder calls versus warm native DP calls on this one clip.

FLEURS provides no measured word boundaries. No boundary-error, long-form,
original-Python encoder parity, quantization, GPU/VRAM, other-platform, Subtimer
comparison or default-promotion conclusion follows. Those gates remain in
PLAN.md. Earlier frozen general CI/lint must pass before promotion even if this
profile succeeds. The gate remains `CRISPASR_SUBWORD_CTC_ALIGN_OPT=1`, default off.

Tiny local worker contracts (no models/build of the library):

```sh
python3 tests/test_alignment_model_profile.py -v
```

## Long-window follow-up (issue550)

Add `--long-windows` to the worker arguments to repeat the licensed speech seven
times (44.52 seconds) with the complete repeated transcript. The actual CLI runs
full input (`--chunk-seconds 0`), 30-second cores, 10-second cores, then 30-second
cores again. Require complete ordered positive word spans within the existing one-encoder-frame
endpoint tolerance and exact
30-second repeatability. Python/exported C ABI default windowing must match the
30-second CLI words with a12.34-second absolute offset. Retain raw WAV/text,
per-call JSON/logs and measured full-versus-window boundary differences. Different
encoder context/normalization means full and windowed logits/timings need not be
identical. No manual word-boundary or GPU/VRAM acceptance follows. Short-fixture
OFF/ON/OFF and immutable-own-reference gates still apply first.
