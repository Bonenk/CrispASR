# Learnings — CrispASR vs transcribe.cpp Evaluation

Lessons from the systematic head-to-head benchmark against
[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) (July 2026).

## Build & Infrastructure

1. **GGML_CUDA_NO_VMM=ON is essential on Kaggle**: Both CrispASR and transcribe.cpp
   use ggml's CUDA backend. Without this flag, cmake's `FindCUDAToolkit` fails to
   create the `CUDA::cuda_driver` imported target because `libcuda.so` only exists
   in `/usr/local/cuda/lib64/stubs/` on Kaggle. The flag gates the
   `target_link_libraries(... CUDA::cuda_driver)` call entirely. Neither symlinks
   nor `CMAKE_LIBRARY_PATH` nor `-DCUDA_DRIVER_LIBRARY` fix it — those target
   different cmake subsystems than `FindCUDAToolkit`'s internal lookup.

2. **ccache must come from the same environment**: A ccache snapshot from the VPS
   (different g++ version) gives 100% cache miss on Kaggle (g++ 11.4.0).
   ccache keys on preprocessed source + compiler path + flags. Always refresh the
   `crispasr-ccache` dataset from a successful Kaggle build, never from the VPS.
   Warm cache cuts builds from ~25 min to ~3 min.

3. **git submodules need explicit init after --depth 1**: `git clone --depth 1`
   does not fetch submodules. CrispASR's ggml is a submodule since 2026-07-07;
   cmake fails at `add_subdirectory(ggml)` without
   `git submodule update --init --recursive --depth 1`.

## CLI & API

4. **CrispASR uses --backend, not -b**: There is no short form. Using `-b whisper`
   causes "unknown argument" → usage text → exit(0). Exit code 0 (not nonzero)
   makes it look like success with empty output. Always check stderr even when
   exit code is 0.

5. **Companion files must be beside the GGUF**: Moonshine backends need
   `tokenizer.bin` in the same directory as the model GGUF. `--auto-download`
   puts companions in the cache dir, not next to a manually-placed model file.
   When downloading GGUFs manually, also download companions from the same HF repo.

6. **transcribe.cpp's --batch-jsonl gives structured timing**: Single-file mode
   outputs human-readable text; `--batch FILE --batch-jsonl` gives JSON with
   per-utterance `mel_ms`, `encode_ms`, `decode_ms`, `load_ms`. More precise
   than wall-clock timing for RTF comparison.

## Performance

7. **GPU-vs-GPU is competitive, not a blowout**: With both engines on CUDA (P100),
   CrispASR wins 3/9 (SenseVoice, Qwen3, Canary, FunASR), transcribe.cpp wins
   4/9 (Parakeet, Moonshine, Moonshine-Streaming, Nemotron), near-parity on
   Whisper. The earlier "CrispASR 7-10x faster" result was comparing CUDA vs
   CPU-fallback — transcribe.cpp's CUDA build was broken until we fixed it with
   `-DGGML_CUDA_NO_VMM=ON`.

8. **CPU overhead is CrispASR's main disadvantage**: 1.3-3x slower than
   transcribe.cpp on CPU. CrispASR's unified backend path includes VAD slicing,
   segment merging, post-processing (punctuation, truecasing), and LID detection
   that transcribe.cpp doesn't do. This overhead is amortised on GPU but
   dominates on CPU.

9. **Quantisation affects transcription**: Moonshine Tiny Q4_K (CrispASR) produces
   "american asked" while Q8_0 (transcribe.cpp) produces "americans ask" — a real
   accuracy difference from lossy quantisation, not a code bug.

## Performance — Deep Dive (§232 optimisation attempts)

14. **GGML_LLAMAFILE (tinyBLAS) doesn't help Q4_K models on x86**: A/B on Intel
    Xeon Skylake showed neutral results. tinyBLAS optimises the F32 GEMM kernel,
    but Q4_K inference is dominated by dequantisation, not the GEMM itself.
    transcribe.cpp's "~29% faster encoder" claim may apply to F16 or ARM NEON.
    Keep ON as default (zero risk, may help other architectures).

15. **Moonshine decoder is compute-bound, not overhead-bound**: The per-token
    graph rebuild + sched alloc takes ~1-2ms, but the actual matmul compute
    takes ~315ms/token (CPU, Q4_K tiny). The graph CAN'T be cached because
    `cur_pos` is baked into KV cache view byte offsets (`ggml_view_3d` uses
    compile-time offsets). This is a fundamental ggml API limitation — views
    don't support tensor-computed offsets.

16. **In-graph argmax saves GPU transfer, not CPU time**: Adding `ggml_argmax`
    to the decode graph reduces device→host transfer from 128 KB (full vocab
    logits) to 4 bytes (one int32) per token. On CPU this is meaningless
    (shared memory). On GPU it eliminates 3.3 MB of PCIe traffic per
    transcription (26 tokens × 128 KB). Needs Kaggle GPU benchmark to measure.

17. **Sliding-window attention masks cannot be skipped for offline files**:
    Moonshine-streaming's encoder was trained with specific window sizes
    (wl=16, wr=4 per layer). Removing masks for offline processing (to match
    transcribe.cpp's approach) produces degenerate output (repeating tokens).
    The model's learned feature distribution depends on the windowed attention
    pattern. The 21x speed gap requires either:
    - A sparse/banded flash attention kernel that natively skips masked positions
    - Using non-streaming moonshine for offline files (recommend `--backend
      moonshine` instead of `--backend moonshine-streaming` for files)
    - Accepting the gap (streaming model = streaming overhead)

18. **Nemotron streaming attention has the same constraint**: The cache-aware
    FastConformer uses L=56 R=3 attention windows. Like moonshine-streaming,
    switching to full attention for offline would likely break output. The 8.4x
    gap is architectural — streaming models pay streaming overhead even offline.

19. **RTF timing decomposition is critical for diagnosis**: CrispASR's stderr
    RTF excludes model load and LID but includes VAD + inference + post-proc.
    transcribe.cpp's `--batch-jsonl` gives mel/encode/decode separately. On the
    Kaggle benchmark, the subprocess wall-clock (`ca_infer_s`) includes
    everything — this is what matters for the user experience but not for
    engine-level comparison. Always report both.

20. **Batched blank-scan helps modestly on GPU, hurts on CPU**: The batched
    TDT/RNNT decode pre-computes joint logits for 32 frames in one sgemm,
    then scans for the first non-blank. On P100 GPU: Parakeet decode 955ms→828ms
    (13%), Nemotron 385ms→345ms (10%). On CPU: SLOWER (sgemm overhead for small
    matrices exceeds N×sgemv). Gate batched path behind GPU detection or env var.

21. **Transducer decoders are fundamentally CPU-bound in CrispASR**: The LSTM
    predictor + joint network use host-side `cblas_sgemv` with sequential state
    updates. Even with batched blank-scanning, the LSTM step between token
    emissions runs on CPU. The path to matching transcribe.cpp's 29ms Parakeet
    decode is porting LSTM+joint to a ggml graph on GPU. Small matrices (640×640)
    make this challenging — GPU kernel launch latency may dominate.

22. **Moonshine encoder gap is im2col on raw audio**: CrispASR processes raw
    176K audio samples through 3 Conv1d layers via `ggml_im2col + ggml_mul_mat`,
    creating 45.6 MB of F32 intermediates. transcribe.cpp likely uses `ggml_conv_1d`
    directly or a pre-computed mel spectrogram, avoiding the large intermediate.
    The encoder produces 2737 frames (not 550 — moonshine-streaming subsamples
    more aggressively). This accounts for the 5.8x GPU gap.

23. **Cohere works and CrispASR wins**: Fixed URL (repo is `cstr/cohere-transcribe-
    03-2026-GGUF`, not `cstr/cohere-transcribe-GGUF`). CA 0.046 vs TC 0.070 =
    CrispASR 1.5x faster on GPU. Cohere's encoder-decoder architecture benefits
    from CrispASR's GPU-accelerated cross-attention path.

## Model Coverage

10. **CrispASR coverage gaps**: GigaAM v3 family (Russian+EN ASR, 4 variants) and
    MedASR (gated medical) exist only in transcribe.cpp.

11. **transcribe.cpp coverage gaps**: No TTS, no diarization, no LID, no forced
    alignment, no translation (m2m100, madlad), no S2S. CrispASR covers all of
    these plus many unique ASR backends (cohere, granite, voxtral, glm, mimo,
    vibevoice, lfm2-audio, etc.).

## Benchmark Methodology

12. **RTF measurement differs**: CrispASR reports wall-clock including audio I/O,
    VAD, and post-processing. transcribe.cpp's `--batch-jsonl` reports
    mel+encode+decode only. For fair CPU comparison, use wall-clock on both sides.
    For GPU comparison, CrispASR's stderr RTF is the authoritative number.

13. **Normalisation matters for WER**: SenseVoice emits `<|TAG|>` tokens, Nemotron
    emits inline `en-us` language codes. Strip both before WER computation.
    Use lowercase + strip punctuation + normalise whitespace as the baseline.

24. **Batched blank-scan makes GPU WORSE, not better**: v14 showed Parakeet
    decode 0.095→0.833 (8.8x slower) and Nemotron 0.345→1.667 (4.8x slower)
    with CRISPASR_TDT_BATCH=1. Root cause: the "batch" runs a CPU sgemm
    (32×8198 logits) while the GPU sits idle. The sequential path runs 1×8198
    sgemv per step and terminates at the first blank — much cheaper because
    most frames ARE blank. Batching only helps when the sgemm itself runs on
    GPU (i.e., the LSTM+joint is a ggml graph), not when it's a CPU-side
    cblas call between GPU encoder passes. **Lesson: don't batch CPU work
    that feeds a GPU pipeline — batch the GPU work itself.**

25. **A CPU-pinned decode graph must keep its weights on CPU too (moonshine,
    M1 Metal)**: moonshine's self-attn KV cache lives on a CPU buffer, so
    `ggml_backend_sched` runs the whole decode step on the CPU even in GPU
    mode. With the decoder weights loaded onto the GPU (the naive all-GPU
    load), the sched then re-copies every GPU-resident decoder weight
    (incl. the 18-36 MB embed/lm_head) GPU→CPU on *each* per-token graph
    rebuild — the copy can't be cached because each step builds a fresh
    graph. Measured on a *quiet* M1 (jfk, 26 tok): Metal decode q8 39→23 ms
    (−40%), f16 119→50 ms (−58%), bit-identical transcript; f16 hurts most
    because its weights are 2× q8. Fix: `load_weights_split` routing
    `encoder.*`→GPU, `decoder.*`→CPU (moonshine default; `MOONSHINE_ALL_GPU=1`
    restores the old load). The per-token copy also explains the plan's
    440-660 ms/decode-*under-load* figures — the copy balloons under GPU
    contention. **Lesson: when the KV cache pins a decode graph to CPU,
    co-locate that graph's weights on CPU; a GPU weight buffer feeding a
    CPU split is a per-step cross-backend copy, not free residency.**
    (Corollary to the §232 CP_DIRECT finding: for these tiny models the win
    is avoiding cross-backend traffic, not sched-free dispatch.)

26. **For moonshine *tiny* on Apple Silicon at idle, pure CPU beats GPU
    outright**: measured totals (jfk, q8) — CPU 79 ms (enc 47 + dec 30) vs
    all-GPU 110 ms (enc 67 + dec 39). The model is small enough that Metal
    launch + the per-layer encoder attention Metal↔CPU permute bounces cost
    more than they save. The hybrid load (learning 25) recovers the decode
    half; the encoder half (GPU 67 vs CPU 47) remains a launch/bounce tax,
    left as-is (fixing the flash-attn layout bounce is a shared-code, higher-
    risk change). Net: GPU-mode moonshine-tiny is now enc-bound, not the
    earlier "GPU decode is 440 ms" story (that was the CPU-run-mislabeled-as-
    GPU bug × the weight-copy tax, both now addressed).

27. **Keeping a small-model encoder fully on Metal (manual attn) is SLOWER
    than the flash_attn CPU-bounce**: moonshine's encoder head_dim=36 has no
    Metal flash_attn kernel, so `ggml_flash_attn_ext` runs on CPU and the
    sched bounces each layer MTL→CPU→MTL (Q/K/V copies ≈514 KB each). Replacing
    it with manual `mul_mat + soft_max_ext + mul_mat` (Metal-supported at any
    head_dim) keeps the whole encoder on-backend — but measured ~40% SLOWER on
    M1 (enc 162 vs 114 ms, even with the flash path's CPU work under load).
    The T² scores tensor ([T,T,nh] ≈ 6.8 MB/layer) + 3 `cont`s spawn many small
    Metal kernels whose launch cost exceeds the cheap bounce copies — the same
    "death by kernel count" that sinks sched-free dispatch for these tiny
    graphs. Transcript identical. **Lesson: a cross-backend bounce of a few
    hundred KB is often cheaper than materialising T² attention on the GPU for
    a small model; don't assume "keep it all on one backend" wins — measure.**
    Kept opt-in (`MOONSHINE_ENC_ATTN=manual`) for CUDA/Vulkan/base re-test.

28. **The GPU→CPU decoder-weight copy isn't just slow — it drifts the output**:
    applying the learning-25 hybrid split to `moonshine_streaming` (same
    CPU-KV-pinned decode) in forced-GPU mode (`MOONSHINE_STREAMING_GPU=1`), the
    hybrid path reproduces the pure-CPU transcript **exactly**, while the legacy
    all-GPU path stably drops a comma ("so my" vs "so, my", deterministic across
    reps). The per-token GPU→CPU weight copy perturbs a borderline decode logit
    enough to shift a token boundary → different punctuation. So co-locating a
    CPU-pinned graph's weights on CPU is a *correctness* win too, not only perf.
    (moonshine_streaming stays CPU-by-default — the encoder is launch-bound on
    GPU; the fix is latent until the §232 Fix-2 batch encoder lands.)

29. **Gate a GPU transducer-decode port on Kaggle BEFORE building it — the naive
    per-step version loses.** Scoped the Parakeet TDT decode port (target: TC's
    29 ms P100 decode). Three pieces of evidence all point the same way: (a)
    LEARNINGS 24 — batched GPU joint measured 8.8× WORSE (CPU sgemm, idle GPU);
    (b) LEARNINGS 25-27 — per-step GPU dispatch of ≤8198×640 matmuls is
    launch-bound on Metal and loses to CPU cblas; (c) M1 CANNOT measure the win
    (parakeet is already encoder-bound + 6× RT on Metal; the decode gap is a
    CUDA-vs-CUDA competitiveness issue). TC's 29 ms almost certainly comes from
    **CUDA-graph capture** (whole step loop as one replayable graph), not per-step
    ggml dispatch. **Lesson: for a transducer/AR GPU-decode port whose only
    payoff is on CUDA, don't hand-write a large speculative ggml decode from an
    M1 session — scope the exact math (mind stale comments: the parakeet joint is
    ReLU, not the tanh the header claims), record the design, and build+A/B it on
    Kaggle where it can actually be measured. Correctness (transcript/WER parity)
    is HW-independent and can be pre-validated on M1; perf cannot.** Design in
    PLAN §232 "RNNT/TDT GPU decode".

30. **The Parakeet "955 ms P100 decode gap" is substantially a slow-CPU-BLAS
    artifact, not pure GPU-idle.** After actually implementing the ggml-graph
    TDT decode (LSTM predictor + joint on `ctx->backend`, opt-in
    `PARAKEET_GGML_DECODE=1`), the M1 numbers reframe the whole gap: cblas decode
    is only **~60 ms on M1** (Apple Accelerate) vs the **955 ms** the §232 Kaggle
    kernel reported (OpenBLAS on the Kaggle CPU) — same code, same audio, ~16×
    difference purely from CPU BLAS quality. The ggml GPU decode is **55-59 ms on
    M1 Metal** — neutral/slightly faster, and **transcript-identical** to cblas
    (jfk + multispeaker, CPU & Metal), so the port is correct and safe to ship
    gated. **Lessons:** (a) before attributing a CPU-path "gap" to GPU-idle /
    architecture and building a GPU port, measure the baseline on the SAME BLAS
    the comparison used — a large chunk of the parakeet gap may close by linking a
    faster CPU BLAS, not by a GPU decode. (b) A correct hand-written ggml
    transducer decode is a modest, contained amount of code (one LSTM-layer helper
    + two step builders) and validates on M1 for correctness — so "build it, gate
    it, let Kaggle judge perf" was tractable after all; the reservation in
    LEARNINGS 29 was about not FLIPPING THE DEFAULT unvalidated, which still holds.
    Kaggle P100 A/B: `tools/kaggle/parakeet-ggml-decode-ab/`.
