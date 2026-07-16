# CrispASR perf campaign — fleet-wide codec/CFG optimization

Cross-cutting speedups discovered during the OmniVoice #254 work, generalized to
the whole backend fleet. Each item ships **env-gated** (A/B toggle, never a silent
default), **A/B'd** against ground truth (byte-equivalence + ASR/decoded-output
roundtrip on F16 AND a quant), and **unit-tested** where a model-free test is
possible. Default flips only on a proven speed AND quality win.

## NOW — active work

**Status (2026-07-16): shared FASTCONV helper landed + 2 backends migrated/piloted.**

- ✅ **Item 1 — shared `core_dac::fastconv_cache` + `conv1d`/`res_unit`/`dec_block`
  fast path** (`203f28f01`). Unit test `test-fastconv` (206 assertions): fast ≈
  legacy (K=7 + k=1), byte-identical when disabled. `res_unit`/`dec_block` take an
  optional `fc` (default nullptr = legacy, so all existing callers untouched).
- ✅ **Item 6 — OmniVoice migrated to the shared helper** (`191a7ebe4`) — deleted
  its ~90-line local copy; verified equivalent (max|d| 23/32768, ASR exact).
- ✅ **Item 7 pilot — irodori_tts** (`CRISPASR_IRODORI_FASTCONV`, default on):
  bakes 92 F16 decode conv kernels → F32; wired `decode_dac_window` through
  `core_dac::conv1d(...,&fc)` + `dec_block(...,&fc)`. **Codec-only A/B (seed 42,
  isolating the flow-matching RNG): BYTE-IDENTICAL (0/32768).**
  - ⚠ **Lesson:** irodori is non-deterministic without a seed (`std::mt19937` from
    `random_device`, flow-matching noise). A naive on/off byte-diff showed
    max|d|=45607 (RNG, NOT fastconv). ON-vs-ON @seed42 = 0 confirmed determinism;
    the seeded on/off = 0 confirmed the codec change. [[tts-parity-not-by-audio-corr]].
- ✅ **zonos_tts** (`CRISPASR_ZONOS_FASTCONV`, default on): threaded the cache
  through `core_dac::build_decode_graph(...,&fc)` (extended with an optional fc
  param; nullptr = legacy). Bakes the dac-44khz F16 decode kernels. **Codec A/B
  (seed 42): BYTE-IDENTICAL (0/32768), non-silent.** So all three core_dac-family
  backends (omnivoice/irodori/zonos) now share one FASTCONV impl.
- ⏭ **Next:** the remaining backends have their OWN local conv lambdas (not
  `core_dac`). Wire the reusable `fastconv_cache` into each: add a cache member,
  bake its convs at load, call `fc->get(w)` in its lambda (or route to
  `core_dac::conv1d`). Then items 2 (interval-CFG), 3/4 (Metal q4_k, CI perf gate).

---

**Kickoff notes (superseded above): 26 backends, only 2 had FASTCONV.**

### Evidence (grep of `src/*.cpp`, 2026-07-16)
- **26 TTS backends** route their vocoder/codec through `core_dac::conv1d` /
  `core_convt::*` (which cast the F16 kernel → F32 inside EVERY graph).
- **Only 2** (omnivoice, qwen3_tts) have a baked-F32 fast path. **24 un-migrated.**
- `PERFORMANCE.md` claims FASTCONV "landed for qwen3-tts, voxtral-tts, omnivoice,
  tada, chatterbox" — but only 2 actually have it. Same overclaim class as the
  omnivoice one corrected this session; **coverage doc is unreliable, audit it.**
- **11 backends** use classifier-free guidance (chatterbox, cosyvoice3, dia, dots,
  f5, irodori, tada, vibevoice, voxcpm2, voxtral, zonos) — interval-CFG candidates.
- Rich benchmark harness already exists (`tools/benchmark_asr_engines.py`,
  `tests/benchmark_*`) but is NOT a CI gate; PERFORMANCE.md is hand-maintained + drifts.

---

## Item 1 — Shared FASTCONV in `core_dac` (24 backends) ★ highest ROI
The fork's `ggml_conv_1d` does `ggml_cast(F16→F32)` on the kernel inside every graph
when activations are F32 (~dozens of casts per codec decode), plus a pure-copy im2col
for k=1 convs. OmniVoice proved baking F32 kernels once at load + k=1→matmul = **2.9×
decode, output-equivalent**.

**Design (shared, no per-backend reinvention):**
- `core_dac::fastconv_cache` — owns a `ctx_f32` + backend buffer + `unordered_map<const
  ggml_tensor*, ggml_tensor*>`; `bake(backend, {conv weights})` at load converts each
  F16 kernel → F32 once; `get(w)` returns the baked F32 (or `w` if not F16/absent).
- `core_dac::conv1d(ctx, x, w, b, K, dil, const fastconv_cache* fc)` overload — when
  `fc` present+enabled: k=1 → `ggml_mul_mat` (skip im2col), K>1 → `ggml_conv_1d` with
  the baked F32 kernel (cast becomes a no-op). `fc == nullptr` → identical to today.
- Per backend: add a `fastconv_cache` member, `bake(...)` the codec convs at load
  (gated `CRISPASR_<BACKEND>_FASTCONV`, default per-backend after A/B), pass `&fc_`.
- **Unit test** (`tests/test-fastconv.cpp`, model-free): random F16 conv kernel + input,
  assert `conv1d(...,&fc)` ≈ `conv1d(...)` within F32 tol, and k=1 path exact.

**Rollout:** shared helper → migrate omnivoice (dogfood/de-dup) → pilot kokoro/piper/
melotts → remaining ~21, each byte+ASR A/B'd.

## Item 2 — Interval-CFG for guidance backends (~11)
Port the OmniVoice uncond-skip-every-K trick (recompute uncond only every K steps,
reuse cached logits; cond fresh every step; first+last always recompute). Gated
`<BACKEND>_CFG_INTERVAL`, default 1 (exact). ~20–40% on flow/diffusion stages.
Per-backend ASR-validated (approximate → opt-in). Candidates: f5 (ODE), chatterbox
(CFM), vibevoice (DPM), voxcpm2, cosyvoice3, dia, zonos.

## Item 3 — Metal q4_k
Measured: q4_k is slower AND lower-quality than q8 on Metal (Apple q4_k dequant path).
- **Quick:** registry/auto-select prefers q8 on Apple Silicon (config-level).
- **Deep:** ggml-fork Metal q4_k dequant kernel — helps every quantized backend on M-series.

## Item 4 — Perf-regression CI gate
Wire the existing benchmark harness into CI: per-backend RTF + decoded-output snapshot,
fail on regression. Keeps PERFORMANCE.md honest (it demonstrably drifts) and guides
optimization with real numbers.

---

## Discipline (every item)
1. Env-gated (`CRISPASR_<BACKEND>_<FEATURE>`), both paths kept, never remove gates.
2. A/B vs ground truth: byte/near-equivalence + ASR/decoded roundtrip, F16 AND quant.
3. Unit test where model-free (the shared conv helper especially).
4. Flip default only on speed AND quality win; approximate paths stay opt-in.
5. Checkpoint: update this PLAN + push to main at each landed backend.
