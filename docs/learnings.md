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
