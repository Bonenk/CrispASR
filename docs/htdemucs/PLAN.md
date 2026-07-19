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
- [x] **CROSSTRANSFORMER NOW FULLY PASSES** — all 5 layers + post_transformer_*
      at cos = 1.000000 (BUGS 7, 8, 9)
- [ ] IN FLIGHT: decoder — dec_freq_0..3 cos 0.96 -> 0.90, and the final
      outputs are ~100x too small (|mine|=0.0015 vs |ref|=0.1543), which looks
      like a separate CaC-unmask / iSTFT / denormalize issue
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

### BUG 7 — transformer input: pos-emb added BEFORE norm_in (FIXED)

`CrossTransformerEncoder.forward` is `x = norm_in(x)` **then**
`x = x + weight_pos_embed * pos_emb`. The C++ added the position embedding
first and then LayerNorm'd it, which renormalised the embedding away. Both sin
embedding formulas themselves were already correct (verified term by term
against `create_2d_sin_embedding` / `create_sin_embedding`).

### BUG 8 — spurious transpose scrambled both transformer branches (FIXED)

`ggml_conv_1d` output has `ne[0] = seq`, i.e. seq is the **fast** axis, so the
channel upsampler already left the buffers as `[dim][seq]` in C memory — exactly
what the attention code indexes (`tmp[d*seq_len + s]`). The code read that as
`(seq, dim)` and transposed it (with a matching transpose back before
`channel_down`, so the two cancelled and the shapes stayed plausible) — meaning
every transformer layer ran on scrambled data. Both transposes removed.

The freq branch keeps its native `s = fr*T1 + t1` order rather than Python's
`(t1 fr)`; the position embedding is indexed to match. Token order is otherwise
irrelevant (self-attn is permutation-equivariant, cross-attn sums over the whole
key/value set, FFN/LayerNorm are per-token) and the decoder reads the same order.

### BUG 9 — norm_out is GroupNorm(1), not LayerNorm (FIXED) — the big one

Each transformer layer ends with `norm_out`, a `MyGroupNorm(num_groups=1)`:
**one mean/std over all channels AND all tokens jointly**. The C++ applied
`cpu_layernorm`, which computes a **separate mean/std per token**. That left
~1% error in every layer.

Why it looked like a layer-4 bug: layer 4's pre-`norm_out` activations are
outlier-dominated (norm ~85,589 normalised down to ~78), so that step amplifies
any upstream relative error enormously — `ct_l3_z` cos 0.992 became `ct_l4_z`
cos 0.217. Verified with forward hooks on the real module that this collapse is
genuine reference behaviour, NOT a reference-replication artifact, before
concluding the fault was upstream.

Fixing the normalization alone took the whole transformer to cos = 1.000000:

  ct_l0_z 0.987908 -> 1.000000      ct_l4_z 0.217092 -> 1.000000
  post_transformer_z 0.406606 -> 1.000000

Lesson: a uniform ~1% per-layer error is worth chasing as structural, not
written off as drift — the amplifying stage downstream is the symptom, not
the cause.
