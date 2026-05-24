**Title:** `ggml-backend : sched-internal CPU→GPU input copy delivers stale data in long mixed-residency graphs`

---

Bug report. A long F32 graph with mixed CPU+Metal weight residency
produces structurally wrong output (and NaN under certain
`set_output` layouts) on the Metal backend. The root cause is that
the scheduler's auto-copy of `GGML_TENSOR_FLAG_INPUT` tensors from
the CPU backend to the consuming Metal split's backend silently
delivers stale data — the Metal kernel reads whatever was at the
destination offset before the copy ran, rather than the data the
user uploaded via `ggml_backend_tensor_set`.

CPU backend produces correct output in every configuration tested.
The bug only surfaces when the consuming sub-graph runs on the
Metal backend.

## Application context (chatterbox-tts S3Gen UNet1D)

UNet1D denoiser graph: ~14 sub-blocks (1 down + 12 mid + 1 up +
final), ~396 mul_mats per pass, F32 activations throughout, ~2700
nodes per CFM step, 29 backend splits at the production input
shape (T_mel=382).

Three runtime input tensors per CFM step:
- `unet_input` (F32, T_mel × 320)
- `time_emb` (F32, 1024)
- `mask` (F32, T_mel)

Weights GPU-resident; sched auto-assigns the three inputs to the
CPU backend (presumably because `ggml_backend_tensor_set` from
host data is cheaper to CPU). Sched creates auto-copies
`MTL0#unet_input#0`, `MTL0#time_emb#0`, `MTL0#mask#0` for the first
Metal split that consumes them.

## Minimum repro

Application-level repro using the chatterbox CLI (50 MB GGUF):

```
# Without the workaround — vocoder mel rms ~14, audio garbled
CHATTERBOX_FORCE_GPU=1 CHATTERBOX_S3GEN_UNET_GPU_RESIDENCY=1 \
  ./chatterbox-cli --tts "Hello." --voice prompt.wav --seed 42

# With explicit input-pin via ggml_backend_sched_set_tensor_backend
# on the three inputs — vocoder mel rms 5.1, matches CPU reference
```

Standalone `test-backend-ops`-style repro: a 30-split, 200-node F32
graph with weights on backend A and three flag-INPUT tensors
auto-routed to backend B, where the first split on backend A reads
the inputs. The auto-copy fires (`ggml_backend_tensor_copy` at
`ggml-backend.cpp:1567`) but the kernel reading the resulting
backend-A tensor sees stale bytes. We can prepare a minimised
`test-backend-ops` case — flag the issue and we'll extract one.

## Bisect path

1. **CPU vs GPU `after_im2col` already differs at cos=-0.08.** First
   kernel of the UNet on the GPU side. Kernel arithmetic (causal
   zero-padding) is correct — visible by inspecting the dump bytes
   directly. The kernel is reading wrong values from x.
2. **Host-side input is bit-identical** in both runs (verified by
   logging `input[0..7]` before
   `ggml_backend_tensor_set`).
3. **`ggml_backend_tensor_get` readback right after the upload returns
   the correct values** — the CPU buffer holds the right data.
4. **`ggml_backend_sched_get_tensor_backend(sched, unet_input)`
   returns `CPU`** — even though the consuming kernel will run on Metal.
5. **Pinning the input to the Metal backend via
   `ggml_backend_sched_set_tensor_backend(sched, unet_input, metal)`
   makes the kernel read correct values** and downstream output
   matches CPU reference at `cos_min=0.999976`.
6. **Per-node `ggml_gallocr` trace** shows the auto-copy
   `MTL0#unet_input#0` IS allocated. The plan is internally
   consistent (no overlap with other tensors). So the bug is in the
   runtime delivery of the copy, not in the allocation plan.

## What's ruled out

- Allocator aliasing — per-node gallocr trace + overlap scan find
  zero overlapping live ranges in the affected configuration.
- `kernel_norm_fuse_impl` short-row reduction bug — already patched
  in a separate fork hunk (filed as a different patch). Not the
  cause here.
- `kernel_flash_attn_ext` F32 Q downcast — patching changes the
  drift profile but doesn't fix this path (and the bug fires before
  any attention kernel runs).
- `GGML_METAL_CONCURRENCY_DISABLE=1` — no effect.
- `GGML_NO_INPLACE=1` — actively breaks (sign-flip artifact).
- Per-op CPU pin (`mul_mat`, `norm`, `add`, `cont`) — all produce
  NaN at production T (the original PIN_CPU_OP "fix" was a
  measurement artifact reported as round-tripped through the
  pre-`compare_with_row_width` non-finite guard).

## Investigation pointers in ggml

- `ggml-backend.cpp:1555-1567` — the sched compute-splits loop
  copies input-flagged tensors via `ggml_backend_tensor_copy` after
  a prior `ggml_backend_synchronize`. The path looks correct. We
  suspect either:
  - The `ggml_backend_synchronize(split_backend)` doesn't actually
    wait for the destination's pre-existing residency (some
    Metal-specific ordering issue with the autocopy buffer).
  - `ggml_backend_buffer_is_host(src->buffer)` returns true and
    the `ggml_backend_tensor_set` at line 485 of
    `ggml_backend_tensor_copy` ends up writing to a destination
    Metal buffer whose `tensor->data` offset isn't yet the one the
    consumer reads from.
- `ggml-metal/ggml-metal-device.m:1940-1992` — `ggml_metal_buffer_set_tensor`
  on private buffers uses a synchronous blit (`commandBufferWithUnretainedReferences`
  + `dispatch_semaphore_wait`). The blit destination is computed via
  `ggml_metal_buffer_get_id(buf, tensor)`. If `buf` here is the
  destination Metal buffer for `MTL0#unet_input#0`, this should
  land correctly. Worth instrumenting to confirm the actual byte
  offset that's being written.
- `ggml-backend.cpp:1330-1349` — input-tensor-with-multiple-copies
  bookkeeping. We test with `n_copies=1`; the `n_copies > 1` path
  uses double-buffering events and might behave differently.

## Application-side workaround

Explicit pin in the user code, conditional on the sub-graph
running on a non-CPU backend:

```cpp
if (running_on_gpu_backend) {
    ggml_backend_sched_set_tensor_backend(sched, unet_input, gpu_backend);
    ggml_backend_sched_set_tensor_backend(sched, time_emb,   gpu_backend);
    ggml_backend_sched_set_tensor_backend(sched, mask,       gpu_backend);
}
```

This is a robust workaround at the application layer but the
underlying ggml-side issue should still be fixed — silently failing
input copies are a class of bug other users will hit and have a
very confusing surface (cos -0.08 from the first kernel, no
allocator/kernel symptom to grep for).
