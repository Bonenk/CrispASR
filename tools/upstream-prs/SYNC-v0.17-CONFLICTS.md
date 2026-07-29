# ggml sync v0.10.2 → v0.17.0 — resolution record (2026-07-29)

Merge: `CrispStrobe/ggml` `crispstrobe-ops` (bfe8ea22 + 2 SVE cherry-picks)
← `ggml-org/ggml` master (9be31331). **435 commits, 19 conflicted files, 53 hunks.**

Branch: `sync/upstream-v0.17` on `CrispStrobe/ggml`.
**CrispASR's submodule pin has NOT moved** — see "Pin decision" at the end.

An earlier revision of this file was an inventory written mid-merge that stopped
at three resolved files. All 19 are now resolved and the tree builds and passes.
What follows is what was actually decided and why.

## The two blockers, as resolved

### `col2im_1d` — upstream's op wins, our callsite changed

Upstream gained its own `ggml_col2im_1d` (`fda9d536`) with the same name and
signature as ours but different semantics:

    upstream   p0 crops BOTH sides,  T_out = (T_in-1)*s0 + K - 2*p0,  type-preserving
    ours #160  p0 is a LEFT offset,  T_out = (T_in-1)*s0 + K -   p0,  always F32 out

Both kernels compute `t_abs = t_out + p0` identically; only the length rule and
the output dtype differ. We adopted **upstream's** op and deleted ours, then
rewrote the single production callsite — `core_convt::convt1d_decomp` in
`src/core/conv.h` — to pass `p0 = 0` and do both crops with a view. That is the
only way to express the asymmetric causal crop (`crop_left=0`,
`crop_right=K-stride`) that the symmetric formula cannot represent.

This was the right shape of change because the blast radius is one function:
20 decoder callsites use `convt1d_decomp`, and `convt1d_decomp_tf` just
delegates to it. Nothing else in CrispASR calls `ggml_col2im_1d`.

### `GGML_OP_*` enum — rebuilt by hand

Both sides inserted ops into an ordered enum with two parallel tables. Rebuilt
entry-by-entry; `GGML_OP_COUNT == 103` now asserts at both sites in `ggml.c`.

## What the merge silently broke

Auto-merge "kept both sides" in nine places. None produced a test failure —
they were compile errors, but only in backends nobody had compiled:

| file | duplicate |
| --- | --- |
| `ggml-cpu/ggml-cpu.c` | `COL2IM_1D` forward dispatch ×2; duplicate case label in `ggml_get_n_tasks` |
| `ggml-cpu/ops.cpp` | our F32-only col2im kernel, now redundant |
| `ggml-metal/*` | duplicate case + missing brace; a typedef; an excess initializer |
| `ggml-vulkan/ggml-vulkan.cpp` | duplicate `pipeline_col2im_1d_f32` member; duplicate **brace-truncated** `vk_op_col2im_1d_push_constants`; duplicate pipeline creation; duplicate `GGML_OP_COL2IM_1D` case |

Two of my own conflict resolutions also dropped code — a keep-both concatenation
lost a closing brace (`crispasr_metal_pipeline_cache_flush`), and two "theirs"
resolutions dropped our function bodies (`get_pipeline_aa_snake_beta`,
`ggml_metal_op_aa_snake_beta`, the latter surfacing only as a link error). Both
recovered from `crispstrobe-ops`.

For the surviving Vulkan push-constant struct, the tiebreak was the shader:
`col2im_1d.comp`'s `layout(push_constant)` block and the dispatch site both use
`T_out, OC, K_OC, T_in, K, stride, p0`.

## Our issue-#38 patch had to be extended, not reverted

Upstream's **direct** `CONV_2D`/`CONV_3D` CPU path built its im2col patch buffer
in the kernel type and typed both `ggml_call_mul_mat` operands the same. That
holds upstream, where F16's `vec_dot_type` is F16. Ours is F32
(`ggml_vec_dot_f16_f32`, to avoid the saturating F32→F16 cast — `vfmaq_f16`
accumulates in an F16 register and overflows at 65504), so an F16 patch buffer
tripped `GGML_ASSERT(src1->type == GGML_TYPE_F32)`.

Checked whether the patch could simply be dropped: **no.** Upstream still has
`#define GGML_F16x8_FMA(a, b, c) vfmaq_f16(a, b, c)`; it has not moved NEON F16
dot accumulation to F32, so the saturation is still there.

The patch buffer *is* `mul_mat`'s src1, so it must be built in the kernel type's
`vec_dot_type` — the idiom flash-attn already uses. No-op upstream; under our
traits it also keeps conv patches out of F16, which is what #38 is for.

## Validation

Apple M1, CPU + Metal + BLAS:

    test-backend-ops -o CONV_2D     1573 cases, 3/3 backends
    test-backend-ops -o CONV_3D      261 cases, 3/3 backends
    test-backend-ops (full)        39601 lines, 0 failures, exit 0
      COL2IM_1D 66, CONV_2D 3140, CONV_3D 516, CONV_TRANSPOSE_1D 232

`tests/test_ggml_audio_ops_backend.cpp` gained a `convt1d_decomp` case — the
production path, not the op in isolation — checked against a host
ConvTranspose1d including the weight permutation and both crop patterns:

    convt1d_decomp_causal       max_abs=0  rmse=0   (Metal)
    convt1d_decomp_symmetric    max_abs=0  rmse=0   (Metal)
    convt1d_decomp_causal_cpu   max_abs=0  rmse=0   (CPU)

Negative control: restoring the old `p0`-as-left-offset call makes the symmetric
case return `[2,8]` instead of `[2,10]` and the test fails. The causal case
still passes there — `crop_left=0` makes both semantics identical — which is
exactly why the symmetric case is in the suite.

## The fork now has CI, and it immediately found three bugs

Upstream's `ci.yml` runs CUDA/Vulkan/Metal on **self-hosted** runners this fork
does not have, so those backends were never built on our branches at all.
`.github/workflows/crispasr-ops.yml` builds cpu (x64 + arm64), Metal, Vulkan
(executed against lavapipe, not just compiled) and CUDA (compile-only) on hosted
runners. First run:

1. **A real heap overflow in our tree.** `test-quantize-fns` sizes its scratch
   buffers `2*test_size` bytes, assuming no `vec_dot_type` is wider than 2 bytes
   per element. Ours is F32 for F16, so `from_float` wrote 4 bytes per element
   and ran `test_size` bytes past the end. glibc aborts; **macOS malloc silently
   tolerates it**, so it had been latent for as long as the patch existed.
   ASan: `heap-buffer-overflow ... WRITE of size 16384 in ggml_cpu_fp32_to_fp32`.
2. **CUDA had not compiled since the merge.** Our F16 kernel-weight template
   makes `src0_t` float or half, and `half * float` is ambiguous to nvcc. Fixing
   that exposed the ~590-line resurrection described above, and then a third
   problem: upstream's VMM pool links `CUDA::cuda_driver` unconditionally, but
   that imported target is undefined on Kaggle (configure hard-fails) and empty
   in `nvidia/cuda` devel containers. `src/ggml-cuda/CMakeLists.txt` now resolves
   libcuda by path first and degrades to `GGML_CUDA_NO_VMM` rather than failing.
3. A missing `spirv-headers` CI dependency.

All six jobs now pass: cpu-x64, cpu-arm64, metal, vulkan (executed on lavapipe),
cuda (compile), carried patches.

### GPU execution — Kaggle, since hosted runners have none

`tools/kaggle/ggml-v017-cuda/` builds the branch and runs test-backend-ops on a
real GPU. It detects the device's compute capability instead of hardcoding one:
Kaggle hands out either a T4 (sm_75) or a P100 (sm_60) and gives no choice, so a
fixed arch fails at *run* time after a 20-minute build.

It drew a **P100 (sm_60)** — the best case available, because our #38/#125 F16
cuBLAS fix targets exactly that hardware. Everything the merge touched passed:
COL2IM_1D 32, CONV_TRANSPOSE_1D 104, CONV_2D 1566, MUL_MAT 1050, NORM 19.

It also found a genuine bug that CI structurally cannot: 10 failures in
`GET_ROWS` with k-quants and `be1=7` (`ERR ~1.75`). **Not** a merge regression —
`getrows.cu` is byte-identical to `crispstrobe-ops`. Our k-quant GET_ROWS patch
flattened the index space and ignored `nb02/nb03` and `nb2/nb3`, which is correct
only when `ne11 == ne12 == 1` (embedding lookups, what it was written for) and
wrong for any broadcast — which upstream has since added and now tests. Fixed to
mirror `k_get_rows_float`'s indexing exactly.

`ci/check-crispasr-patches.sh` + `ci/crispasr-patches.txt` turn the 37
"MUST RE-APPLY after ggml bump" comments into an executable check over 23 named
patches. A merge that resolves a hunk as "theirs" drops one of our patches with
no compile error and no test failure — this is the only thing that catches that.
Writing the manifest immediately caught an entry for a Metal `norm_affine`
kernel that never existed (NORM_AFFINE is deliberately CPU + CUDA only).

## Pin decision

**Do not bump CrispASR's `ggml` submodule pin yet.** The pin stays at
`52165e4c`. The merge is validated on CPU and Metal, and CI covers the other
backends' *compilation*, but no GPU-executed CUDA test has run, and the
`conv.h` rewrite + pin bump are **coupled** — the rewrite is wrong against the
old ggml and the old callsite is wrong against the new one. They must land in a
single commit, gated on a CrispASR CI run that builds CUDA and Vulkan and on the
TTS regression (f5-tts, cosyvoice3, TADA, vocoders), since every decoder's
ConvTranspose1d goes through the rewritten function.

## Backported ahead of the sync

The F16 conv_2d fix is **also on `crispstrobe-ops`** (`00285218`), because a
parallel effort hit it on the current pin: F16 kernels abort the direct CONV_2D
CPU path with `GGML_ASSERT(src1->type == GGML_TYPE_F32)`. That is our #38 patch,
not a ggml limitation — the pin has the identical conv structure. Verified on
that exact commit: before, `test-backend-ops -o CONV_2D` aborts (exit 134);
after, CONV_2D 1571 and CONV_3D 261 cases pass. So F16 conv_2d works on the
current pin without waiting for this sync to land.

## Still outstanding

- CUDA execution has now run on sm_60 (P100) only; sm_75+ untested, and Vulkan
  has executed only against lavapipe, not a real GPU.
- The k-quant GET_ROWS broadcast fix is on the sync branch but NOT yet
  backported to `crispstrobe-ops`; it is latent there too.
- `MASTER-AUDIT.md` (outbound) was last cross-checked 435 commits ago.
- Dropped in the merge, re-derive if regressions appear: our GH #65 `cpy.cu`
  grid-y perf tiling, and the Metal im2col batch-1 occupancy tuning
  (`23695065`) — upstream reworked that dispatch, and the orphaned `occ` state
  was removed rather than left reading uninitialised.
- `upstream-prs` 03 / 14 / 17 / 20 are superseded by upstream's own versions.
