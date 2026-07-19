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
- [ ] Full per-stage cos run — IN FLIGHT
- [ ] BUG 2: DConv GroupNorms skipped (confirmed, not yet fixed)
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
