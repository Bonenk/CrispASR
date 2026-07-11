# Handover: FastConformer CPU performance optimization (#81)

## Context

CrispASR's FastConformer encoder (used by parakeet-ctc, parakeet-tdt, canary,
canary-ctc backends) is ~1.5× slower than ONNX Runtime on CPU for parakeet
models. This was investigated in issue #81. A first round of optimizations
(commit `1c5478a1`, `perf: pre-cast F32 conv_dw_w + manual attn gate`) shipped
small gains. This handover is for deeper optimization work.

## The problem

**Benchmark on the VPS** (4-core Intel Xeon Skylake, no GPU, 7.6 GB RAM):

| Engine | Model | Quant | Mean (JFK 11s) | x-RT |
|--------|-------|-------|----------------|------|
| **CrispASR** | parakeet-ctc-0.6b | Q8_0 | **5.16s** | **2.1×** |
| **CrispASR** | parakeet-ctc-0.6b | Q4_K | 6.07s | 1.8× |
| **CrispASR** | parakeet-tdt-0.6b-v3 | Q8_0 | 7.40s | 1.5× |
| onnx-asr | parakeet-ctc-0.6b | int8 | **3.58s** | **3.1×** |
| onnx-asr | parakeet-tdt-0.6b-v2 | int8 | 3.68s | 3.0× |

**Goal**: Close the 1.5× gap. Getting CrispASR parakeet-ctc Q8_0 from 5.1s → <4s
would make it competitive. Getting it to ~3.5s would match ONNX.

External user reports (issue #81):
- Tamnac (Tiger Lake i7, AVX-512+VNNI): CrispASR ~4.7× RT vs onnx-asr ~21× RT
- grikdotnet (RTX 4070 Laptop, CUDA): onnx-asr 40-45× RT

## What was already tried / ruled out

1. **Flash attention vs manual attention**: A/B'd via `CRISPASR_FC_NO_FLASH=1`.
   No difference on CPU at T=137 (~5.1s both). The flash path is NOT the bottleneck.

2. **Thread count**: 2 threads is faster than 4 threads on this VPS (5.1s vs 6.6s).
   At 1 thread it's 8.9s. Scaling from 1→2 is only 1.68×. The parallelism is
   poor for these narrow matrices (d=1024, T=137).

3. **Q4_K vs Q8_0**: Q4_K is ~18% SLOWER than Q8_0 on this CPU. The dequant
   overhead exceeds the memory bandwidth savings at these small matrix sizes.

4. **Graph build vs compute**: graph_build=4ms, sched_alloc=3ms,
   graph_compute=5500ms. 99.9% of time is in `ggml_backend_sched_graph_compute`.

5. **Conv_dw_w F16→F32 cast**: Pre-cast to F32 at load time (shipped in
   `1c5478a1`). Small improvement but not the main bottleneck.

6. **Mel spectrogram**: 26ms — negligible.

## Architecture of the hot path

The encoder is 24 × Conformer blocks. Each block contains:

```
FFN1: LayerNorm → Linear(1024→4096) → SiLU → Linear(4096→1024) × 0.5 + residual
Attn: LayerNorm → Q,K,V proj(1024→1024) → rel-pos BD computation → flash_attn_ext → out proj
Conv: LayerNorm → PW1(1024→2048) → SigLU → DepthwiseConv(K=9) → SiLU → PW2(1024→1024)
FFN2: same as FFN1
Output LayerNorm
```

For JFK 11s audio: 176k samples → 128-mel × ~1100 frames → 8× subsampling → T=137 encoder frames.

Key files:
- `src/core/fastconformer.h` — `build_block()` builds one Conformer block's ggml graph
- `src/canary_ctc.cpp` — CTC runtime (mel → encoder → CTC logits → greedy decode)
- `src/parakeet.cpp` — TDT/RNNT runtime (mel → encoder → transducer decode)
- `PERFORMANCE.md` — benchmark history and methodology

The graph has ~40 ops per block × 24 blocks ≈ ~960 ggml ops. Each op materializes
its full output tensor before the next op starts — this intermediate traffic is the
main source of the gap vs ONNX Runtime.

## Per-block FLOPs estimate (T=137, d=1024)

- FFN1/FFN2: 4 × matmul(1024×4096, 1024×137) ≈ 4.6 GFLOP
- Attention: Q/K/V/pos/out projections = 5 × matmul(1024×1024, 1024×137) ≈ 1.4 GFLOP
  + BD matmul + flash_attn ≈ 0.15 GFLOP
- Conv: PW1(1024×2048), PW2(1024×1024) ≈ 0.9 GFLOP + depthwise conv (tiny)
- Total per block: ~7 GFLOP
- Total 24 blocks: ~168 GFLOP

On a 4-core Xeon at ~50-80 GFLOP/s F32, pure compute is ~2-3s. We see 5.1s →
~40-50% of time is memory traffic / intermediate materialization overhead.

## Concrete optimization directions to investigate

### 1. Fused Q/K/V projection (high impact, medium effort)

The three Q, K, V projections share the same input `x` and have the same output
size. They can be fused into a single matmul: `[Q;K;V] = [Wq;Wk;Wv] × x`, then
split the result with `ggml_view`. This saves 2 matmul kernel launches + 2×
intermediate reads of `x`.

In `build_block()` (fastconformer.h:288-291), replace:
```cpp
ggml_tensor* Q = mm_bias(e.attn_q_w, x, e.attn_q_b);
ggml_tensor* K_ = mm_bias(e.attn_k_w, x, e.attn_k_b);
ggml_tensor* V = mm_bias(e.attn_v_w, x, e.attn_v_b);
```
with a single concat+matmul+view. The weights `attn_q_w`, `attn_k_w`, `attn_v_w`
are all `(1024, 1024)` — concatenate them at load time into a single `(1024, 3072)`
tensor, do one matmul, then view-split the result.

**Important**: the pos projection `R = ggml_mul_mat(e.attn_pos_w, pos_enc)` has a
different input so it can't be fused into this.

### 2. Fused FFN (high impact, medium effort)

Each FFN has: `y = Linear2(SiLU(Linear1(LayerNorm(x))))`. The two linears + SiLU
are three separate ggml ops. Consider:
- Fusing LayerNorm + Linear1 (the norm output is only used by this one matmul)
- Using `ggml_mul_mat_id` or similar batched ops

### 3. Reduce ggml_cont calls (medium impact, low effort)

The attention path has ~6 `ggml_cont` calls per block (each is a full tensor copy):
- `ggml_cont(R)` before BD matmul
- `ggml_cont(BD)` after rel_shift (strided view → contiguous)
- `ggml_cont(Q_u)` before flash_attn_ext
- `ggml_cont(K_)` before flash_attn_ext
- `ggml_cont(V_)` inside V permute
- `ggml_cont(attn_out)` after output permute

Some of these can be avoided by reorganizing the reshape/permute order. For example,
if Q_u is already contiguous (which it might be after reshape_3d + permute in a
specific order), the cont is a no-op but still allocates + copies.

### 4. Switch from sched to gallocr for CPU-only (medium impact, low effort)

`ggml_backend_sched` has overhead for multi-backend scheduling. For the CPU-only
case (no GPU), switching to raw `ggml_gallocr` avoids the split/copy logic.
The encoder graph is a single-backend graph — `gallocr` is simpler and faster.

Gate this behind `ggml_backend_is_cpu(ctx->backend)`.

### 5. Depthwise conv without transpose (low-medium impact, medium effort)

The conv module does:
```cpp
cnv = ggml_cont(ctx0, ggml_transpose(ctx0, cnv)); // (d, T) → (T, d)
cnv = ggml_reshape_4d(ctx0, cnv, T, 1, d, 1);
cnv = ggml_conv_2d_dw_direct(ctx0, dw_w_4d, cnv, ...);
cnv = ggml_cont(ctx0, ggml_permute(ctx0, cnv, 1, 2, 0, 3));
cnv = ggml_reshape_2d(ctx0, cnv, d, T);
```
Two `ggml_cont` calls (one for the transpose, one for the post-conv permute) are
pure memory copies. Consider using `ggml_conv_1d` directly if it can handle the
(d, T) layout without transposing.

### 6. Thread-count auto-tuning (low impact, easy)

The default `n_threads=4` is suboptimal on many CPUs. Consider:
- Auto-detect based on `std::thread::hardware_concurrency()` and CPU topology
- For conformer-shape workloads (narrow M dimension), 2 threads is often optimal
- Could profile the first call and set threads for subsequent calls

### 7. ggml upstream kernel work (high impact, out of scope for CrispASR)

The fundamental gap is ggml's per-op intermediate materialization vs ONNX's fused
subgraphs. These need ggml upstream changes:
- Conv1d + BN + activation fusion
- LayerNorm + GEMM fusion
- Per-tensor int8 activations (4× less activation bandwidth)
- Better thread scheduling for narrow matmuls

## How to benchmark

```bash
# Build in a worktree (NEVER build on main tree directly)
cd /mnt/volume1/CrispASR
git worktree add .claude/worktrees/feat-fc-perf -b feat/fc-perf
cd .claude/worktrees/feat-fc-perf
git submodule update --init --recursive

# Build with OpenBLAS (MKL on this VPS has a broken avx512 symbol)
CCACHE_DIR=/mnt/volume1/.ccache cmake -G Ninja -B build \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=build/install \
  -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DBLA_VENDOR=OpenBLAS
cmake --build build -j$(nproc) --target crispasr-lib

# Benchmark
CRISPASR_LIB_PATH=$(pwd)/build/src/libcrispasr.so python -c "
from crispasr._binding import Session
import soundfile as sf, time
pcm, sr = sf.read('/mnt/volume1/CrispASR/samples/jfk.wav', dtype='float32')
dur = len(pcm)/sr
s = Session('/mnt/storage/gguf-models/parakeet-ctc-0.6b-q8_0.gguf', n_threads=2)
for _ in range(3): s.transcribe(pcm, language='en')  # warmup
times = []
for _ in range(5):
    t0 = time.perf_counter()
    s.transcribe(pcm, language='en')
    t1 = time.perf_counter()
    times.append(t1-t0)
mean = sum(times)/len(times)
print(f'mean={mean:.3f}s = {dur/mean:.1f}x RT  runs={[round(t,2) for t in times]}')
"

# Per-stage timing (canary-ctc backend only)
CANARY_CTC_BENCH=1 ...same command...

# Compare with onnx-asr baseline
HF_HUB_DISABLE_SYMLINKS_WARNING=1 python -c "
import onnx_asr, time
m = onnx_asr.load_model('nemo-parakeet-ctc-0.6b', quantization='int8', providers=['CPUExecutionProvider'])
for _ in range(3): m.recognize('/mnt/volume1/CrispASR/samples/jfk.wav')
times = []
for _ in range(5):
    t0 = time.perf_counter()
    m.recognize('/mnt/volume1/CrispASR/samples/jfk.wav')
    t1 = time.perf_counter()
    times.append(t1-t0)
mean = sum(times)/len(times)
print(f'onnx-asr: mean={mean:.3f}s = {11/mean:.1f}x RT')
"
```

**Baseline to beat**: CrispASR CTC Q8_0 = 5.1s (2.1× RT), onnx-asr int8 = 3.6s (3.1× RT).

## Rules

- Read `/mnt/volume1/crispasr-crispembed-dev.md` for full project rules.
- NEVER build >1 thing at once (8 GB RAM).
- NEVER use `/tmp` — use `/mnt/volume1/tmp-overflow`.
- Always A/B perf changes against ground truth transcript before flipping defaults.
- Keep BOTH paths behind env-var gates. NEVER delete the working path.
- Format with `./tools/format.sh --fix <files>` before committing (clang-format v18).
- Commit to `main` directly, no PRs. Push with `git push origin main`.
