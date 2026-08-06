**Title:** `metal : fix im2col occupancy for batch-1 (inference) convolutions`

---

`ggml_metal_op_im2col` sizes the threadgroup's first dimension from the batch
count `N`:

```c
const uint64_t ntptg0 = std::min(max_threads/(KH*KW), N);
...
dispatch_threadgroups(enc, IC, OH, OW, ntptg0, KH, KW);
```

At **inference batch `N==1`** this collapses to `ntptg0 = 1`, so every
threadgroup runs only `KH*KW` threads — 3..11 for typical 1D convs
(K∈{3,7,11}, KH=1). That is ~10–34% of a single 32-wide simdgroup, across
millions of tiny threadgroups. The im2col kernel then runs far below memory
bandwidth (measured ~40× below peak on Apple M1). For conv-heavy inference
graphs this dominates: on a HiFi-GAN vocoder decode (melotts) per-node
profiling put `IM2COL` at **58%** of total GPU time, vs `MUL_MAT` at only 12%.

Batch `N` is a poor parallelization axis — it is 1 for essentially all
inference. The fix parallelizes the threadgroup over a block of the output
width `OW` instead (grid Z = `ceil(OW / OW_PER_TG)`, threads =
`OW_PER_TG*KH*KW ≈ max_threads`), looping over `N` inside the kernel. `OW` is
large for audio/vision convs, so threadgroups fill.

Bit-exact: im2col is a gather-copy — each destination element is written exactly
once with the same value regardless of which thread writes it (no FP reorder).

Patch: `23-metal-im2col-batch1-occupancy.patch` (im2col kernel + dispatch + one
kargs field for the explicit `OW`).

**Verification.** Apple M1, F32/F16 im2col. Measured ~2–3× on the melotts
HiFi-GAN decode (im2col-dominated: GPU 1.88 s → 0.83 s) and 1.65× on a small
conv-preprocessor ASR (moonshine), no regression on paraformer. Bit-parity
confirmed with a deterministic ASR (moonshine transcript byte-identical with the
change on vs off) and a melotts TTS→ASR round-trip. Recommend running
`test-backend-ops` `IM2COL` cases under Metal (incl. `N>1`) to confirm
bit-identical output before merge.

**Note for the PR.** The CrispASR fork ships this behind an env A/B gate
(`CRISPASR_METAL_IM2COL_OCC`, auto-on for `N==1`) and keeps the original path for
`N>1`. The upstream PR should simply auto-select the OW-blocked dispatch when it
gives better occupancy (e.g. `N==1`, or whenever `N*KH*KW < max_threads`) and
drop the fork env var. CUDA is unaffected — `im2col_cuda` already parallelizes
threads over `IC*KH*KW`, not `N`.

---

**SUPERSEDED 2026-08-06 — re-derived against the v0.17 dispatch as
`kernel_im2col_flat`** (fork commit `89a2039d`, branch `sync/upstream-v0.17`;
this OW-blocked patch targets the pre-v0.17 dispatch and no longer applies —
the v0.17 sync dropped the OCC variant, see `SYNC-v0.17-CONFLICTS.md`). The
successor covers this draft's batch-1 case via the predicate
`N*KH*KW < 128` AND a second pathological class this draft missed: the
`ggml_conv_2d_dw` lowering (IC==1, N=C*batch), whose per-thread N/ntg0 loop
strides a full plane on both src and dst (~0.4 GB/s on M1; 70% of the
CrispEmbed PP-OCRv6 medium recognizer graph by per-op profile). One thread
per dst element from a `(ceil(OW*CHW/256), OH, N)` grid; row-local 32-bit
divmods only — a first cut with a fully flat int64-divmod index was SLOWER
than the broken kernel (Apple GPUs emulate int64 division), which is worth
stating in the PR to preempt the "why not one flat index" review question.
Fresh M1 numbers, outputs byte-identical: PP-OCRv6 medium recognize
13.4-14.0 s -> 5.8-6.2 s (2.3x), layout_detect (RT-DETR, N==1 convs)
1.6x. CrispASR-side numbers re-measured 2026-08-06 after the pin bump to
`89a2039d` (interleaved M1 pairs, PERFORMANCE.md top entry): melotts
hifigan_decode 2.02 s -> 1.10 s (**1.85x**, round-trip clean), moonshine
transcript byte-identical with RTF neutral-to-slightly-better, paraformer
transcript byte-identical with its two im2col nodes 8.5-9.1 ->
0.4-1.0 ms/node (9-20x, wall-invisible at its share). The old 2-3x/1.65x
figures above were the pre-v0.17 variant. Re-draft the upstream PR from
`kernel_im2col_flat`; keep this file's CUDA note (still true) and the
auto-select-no-env-var guidance (the fork's `CRISPASR_METAL_IM2COL_FLAT`
opt-out stays fork-only).
