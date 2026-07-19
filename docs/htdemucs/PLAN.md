# HTDemucs — parity validation (feat/htdemucs-parity-diff)

HTDemucs (Meta Demucs v4, 42M params, 533 tensors) had a complete-looking C++
runtime in `src/htdemucs.cpp` but **zero cosine-similarity validation** — the
binary had never produced output compared against the Python reference. The VPS
could not run it (8 GB, OOM). This branch does that validation on the M1.

## NOW — active work

- [x] Worktree `feat/htdemucs-parity-diff` + submodules
- [x] `input_wav` stage added to `tools/reference_backends/htdemucs.py` so the
      diff replays the reference's exact 44.1 kHz waveform (resampler-independent
      input gate, per the dev-guide "gate input alignment first" rule)
- [x] Reference dumped: 22 stages, F32 GGUF converted (533 tensors, 160 MB)
- [x] Per-stage capture + `htdemucs_diff()` in `src/htdemucs.cpp`
      (self-contained runner, dots-tts/mel-band-roformer pattern)
- [x] `htdemucs` wired into `examples/cli/crispasr_diff_main.cpp` (checklist #9)
- [x] **BUG 1 FIXED** — `read_tensor_f32` infinite recursion (see below)
- [x] First full per-stage run (was: 6/21 stages)
- [x] **BUG 2 FIXED** — DConv GroupNorms
- [x] **BUG 4 FIXED** — time encoder skipped DConv entirely
- [x] **BUG 5 FIXED** — tdecoder rewrite used a 1x1 helper on a K=3 Conv1d
- [x] **BUG 6 FIXED** — freq_emb ne[] swapped
- [x] **ENCODER NOW FULLY PASSES** — 15/25 stages, every encoder stage
      (freq + time, all 4 layers) and both pre_transformer_* at cos >= 0.999998
- [ ] IN FLIGHT: CrossTransformer — `post_transformer_z` cos=0.035,
      |mine|=715.5 vs |ref|=86.5 (8x magnitude blowup => structural, not drift)
- [ ] BUG 3: `tests/test_htdemucs_smoke.cpp` never registered in CMake

## Bugs found

### BUG 1 — `read_tensor_f32` infinite recursion (FIXED)

```cpp
if (t->type == GGML_TYPE_F32) {
    auto _rd = read_tensor_f32(t);   // calls itself -> stack overflow
```

A find/replace that swapped `ggml_backend_tensor_get` for the `read_tensor_f32`
wrapper also rewrote the call *inside the wrapper's own definition*. Effect:
**any F32 GGUF crashed with SIGSEGV (stack exhaustion) on the first weight
read** — the model could never run at all in F32. Only the F16 path worked,
which is why the shipped F16 GGUF appeared functional. Fixed to call
`ggml_backend_tensor_get` directly.

Diagnosis note: the crash presented as SIGKILL/137 with a 78 GB "peak memory
footprint" under the time branch and SIGSEGV/139 without it. `maximum resident
set size` read only 1.8 GB (compression-capped — see the macOS
peak-footprint-not-RSS learning); lldb could not unwind the corrupted stack, so
the location was pinned with fflush'd markers instead.

### BUG 2 — DConv GroupNorms skipped (CONFIRMED, unfixed)

`src/htdemucs.cpp` DConv sublayer forward:

```cpp
// GroupNorm(1) + GELU (skip norm for now — LayerScale init=1e-3 dominates)
```

Both `GroupNorm(1, hidden)` (after the dilated conv) and `GroupNorm(1, 2C)`
(after the 1x1 conv) are omitted. The justification is wrong: `1e-3` is the
LayerScale *init*, not its trained value. The weights are real and non-identity
— `encoder.0.dconv.layers.0.1` has weight mean 1.563 / std 0.496, bias absmax
0.554. The converter exports them and `bind_dconv` binds them into
`norm1_w/norm1_b/norm2_w/norm2_b`; the forward simply never reads those fields
(dead weights).

### BUG 3 — smoke test never built

`tests/test_htdemucs_smoke.cpp` exists but appears in no `CMakeLists.txt`, so it
has never compiled or run. Checklist item 12 is unmet.

## Layout contract (verified, not assumed)

Both the reference and the C++ use `(C, Fq, T)` row-major with `t` fastest
(`x[t + fq*T + c*T*Fq]`, confirmed in `cpu_conv2d_freq`). No permutation is
needed in the diff. Time stages are `(C, T)` channel-major; the final outputs
are captured channel-major *before* the interleave.

### BUG 4 — time encoder skipped DConv entirely (FIXED)

`HEncLayer.forward` is `conv -> GELU(norm1) -> dconv -> GLU(norm2(rewrite))`.
The frequency branch had the DConv; the time branch went straight from GELU to
the rewrite. All 4 tencoder layers have a real DConv, so every `enc_time_*` was
wrong. (`apply_dconv()` existed for exactly this but was never called — dead
code, and itself buggy: it splits the GLU on `ne[0]`, which is *time* for
`ggml_conv_1d` output, not channels.)

Fixed by extracting the DConv stack into `cpu_dconv_inplace()` shared by both
branches, and splitting the time encoder into two ggml graphs
(conv+GELU | rewrite+GLU) with the CPU DConv between them. ggml's 2D `(T, C)`
layout is flat `t + c*T`, identical to the channel-major `(C, T)` convention
`xt_buf` uses, so no transpose is needed at the handoff.

Result: `enc_time_0/1/2` and `pre_transformer_xt` went to cos = 1.000000.

### BUG 5 — tdecoder rewrite is a K=3 Conv1d, not 1x1 (FIXED)

`cpu_conv2d_1x1` was called on `tdecoder.N.rewrite.weight`, whose ne is
`(K=3, IC, OC)`. That helper reads `out_C = ne[3]`, correct for a 1x1 **Conv2d**
`(1,1,IC,OC)` but = 1 for a Conv1d — so the output collapsed, `cpu_glu` returned
an empty buffer, and the following `ConvTranspose1d` dereferenced
`std::vector::data()` on it: the **null-pointer crash at address 0x0** that
killed every run at `dec[0]`. Also the helper assumes K=1, ignoring the K=3
context window.

Added `cpu_conv1d_time()` (proper K, symmetric padding = `hp.context`) and a
`GGML_ASSERT` in `cpu_conv2d_1x1` so a Conv1d weight can never be silently
misread as 1x1 again. The encoder's own 1x1 use (`(1,1,48,96)`) is legitimate.

### BUG 6 — freq_emb ne[] swapped (FIXED)

`nn.Embedding.weight` is `(num_embeddings, embedding_dim)` row-major = ggml
`ne(embedding_dim, num_embeddings)`. The code read `n_freqs = ne[0]` and
`C = ne[1]`, exactly backwards, which both strided by 512 instead of 48 AND made
`min(x_Fq, emb_n_freqs) = min(512, 48) = 48`, so only 48 of 512 frequency bands
received the embedding at all.

The `10 * 0.2 = 2.0` total scale was already correct — verified numerically
against the reference (`max|eff - gguf.T*2.0| = 0.0`) before touching the code.

Bisected by adding `enc0_conv / enc0_gelu / enc0_dconv / enc0_rewrite` stages to
both the dumper and the runtime: all four read cos = 1.000000 while `enc_freq_0`
(the same tensor plus freq_emb) read 0.884, isolating it to the embedding add.

### Harness bug — dangling `first_fail` (FIXED)

The summary reported `FIRST DIVERGENCE: output_vocals` when the real first
failure was `enc_freq_0`. `first_fail` was a `const char*` assigned from
`("enc_freq_" + std::to_string(i)).c_str()` — a pointer into a temporary that
dies at the end of the full expression. Now a `std::string`.
