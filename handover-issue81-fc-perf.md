
---

## STATUS UPDATE 2026-07-11 (M1 session, commit 4100c367)

Per-node profiling (`CRISPASR_FC_PROFILE=1`, added to canary_ctc) **disproved
the bottleneck theory above**: intermediate materialization (conts/casts) was
~3% and sched ~1%. The real cost was **35% of encoder time in the F16
conv.pw1/pw2 matmuls** — stored 3D so crispasr-quantize skipped them, and
ggml's CPU F16 mul_mat is ~6x slower per FLOP than Q8_0. Shipped (all
env-gated, transcripts verified, fused-QKV/strided-attn bit-identical):

- `core_conformer::repack_conv_pw_q8` (CRISPASR_FC_PW_Q8, auto-on for
  quantized models) — parakeet/canary/canary_ctc
- `core_conformer::fuse_qkv` (CRISPASR_FC_FUSED_QKV, default on)
- strided flash-attn views (CRISPASR_FC_ATTN_CONT=1 restores legacy conts)
- crispasr-quantize now quantizes 3D conv pw as 2D — new GGUFs need no repack

M1 CPU: parakeet-ctc q8_0 1.13→0.73 s (re-quantized GGUF 0.60 s = 18.4x RT);
tdt-v3 q4_k 1.89→1.03 s. Full data in PERFORMANCE.md + LEARNINGS.md.

### Remaining to close #81

1. **Re-bench on the VPS** (the 5.16 s baseline box) with the new defaults —
   expect a similar jump; the OpenBLAS build flag was a no-op for this path
   (blas backend never in the sched).
2. **Re-quantize + re-upload the HF GGUFs** (cstr/parakeet-ctc-*-GGUF etc.)
   with the fixed quantizer (~5% smaller, faster than runtime repack).
3. **Extend repack+fusion to nemotron / canary_qwen / lfm2_audio** — same
   BlockWeights/naming, calls are two lines per runtime (see canary_ctc).
4. Directions 2/4/5/6 above are RULED OUT by measurement (see PERFORMANCE.md
   entry); don't re-attempt without new profile data.

## STATUS UPDATE 2026-07-11 (later, same session)

- Extension shipped (cdc6dbc4): conv-pw Q8_0 repack wired into nemotron
  (repack only — its streaming builder bypasses build_block), canary_qwen +
  lfm2_audio (repack + fuse_qkv). nemotron transcript-verified identical
  locally; box too loaded for local timing (another session's builds).
- crispasr-quantize now pins conv pw to a Q8_0 floor for sub-8-bit targets,
  so re-quantized GGUFs byte-match the runtime repack output.
- HF fleet re-quantization runs on Kaggle:
  `tools/kaggle/fc-pw-requant/` → kernel chr1str/crispasr-fc-pw-requant
  (c8930529). 42 repos, per-file strict transcript-equality validation +
  per-backend legacy-vs-new timing on the Kaggle box. Progress lands in
  cstr/crispasr-kaggle-progress; results.json in the kernel output.
  Re-push to resume (already-fixed files no-op).
- Still open: VPS re-bench (item 1 above) — the parallel session's
  issue81-onnx-bench kernel may already cover the GPU side.
