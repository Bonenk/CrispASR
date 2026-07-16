# CrispASR perf campaign — fleet-wide codec/CFG optimization

Cross-cutting speedups discovered during the OmniVoice #254 work, generalized to
the whole backend fleet. Each item ships **env-gated** (A/B toggle, never a silent
default), **A/B'd** against ground truth (byte-equivalence + ASR/decoded-output
roundtrip on F16 AND a quant), and **unit-tested** where a model-free test is
possible. Default flips only on a proven speed AND quality win.

## NOW — active work

**Status (2026-07-16): 2 shared FASTCONV helpers landed; 3 DAC-family backends
wired+verified; HiFi-GAN family now wired for speecht5 (verified). All on `main`,
all green.**

Commits (pushed to `main`):
`203f28f01` shared core_dac cache+conv1d · `191a7ebe4` omnivoice migrate ·
`8f2b17e4a` irodori · `8d8e6d9c8` zonos · `8231e8144` core_hifigan overload ·
`1c558b4f0` speecht5 FASTCONV (branch `perf/fastconv-hifigan-fleet`).

### ⚠ HiFi-GAN family — coverage reality (measured 2026-07-16, GGUF-parsed)
The handover's "one overload sets up 3 backends" is over-optimistic. FASTCONV
only engages when the vocoder conv kernels are **F16** (the cast it kills). Actual
shipped dtypes (voc conv1d-routed kernels, ups.* excluded):
- **speecht5** — registry default `-f16` → **F16 ×74 → ENGAGES.** ✅ wired+verified.
- **fastpitch** — default `-q8_0` → F32 ×74 → **no-op**; only the non-default
  `-f16` variant has F16 ×74. Wired locally but NOT committed: its `-f16` GGUF hits
  a *pre-existing* loader bug (`gguf_init_from_file_ptr: failed to read tensor data`,
  in the GGUF reader before any graph code — file is byte-valid, md5-stable across
  two downloads) so it can't be run end-to-end here; and the q8_0 default is a no-op
  anyway. Low value until the f16 loader is fixed. **Discipline: don't ship what you
  can't run.**
- **bananamind** — ships only `-q8_0` and `-f32` (NO f16 variant in either
  en/de repo) → F32 ×74 → FASTCONV can **never** engage. Do NOT wire (dead code)
  unless an F16 build is published.

### Recipe — wire FASTCONV into one backend (proven 3×)
1. Add `core_dac::fastconv_cache <name>_fc;` to the backend's context struct.
2. At codec/vocoder load (after weights resolved, on the codec compute backend):
   collect all decode conv-kernel `ggml_tensor*` into a vector and
   `ctx-><name>_fc.bake(codec_backend, convs, env_on);` gated
   `CRISPASR_<BACKEND>_FASTCONV` (default on — the change is numerically equivalent).
3. In the decode graph, route each conv through the fc-aware overload:
   `core_dac::conv1d(g,x,w,b,K,dil,&fc)` / `dec_block(...,&fc)` /
   `build_decode_graph(...,&fc)` (DAC family) or `core_hifigan::conv1d(...,&fc)`
   (HiFi-GAN family). For a bespoke local lambda, replace `w` with `fc.get(w)`.
4. `<name>_fc.free();` in the backend's free().
5. **A/B (MANDATORY, seed-aware):** build, then synth fastconv ON vs OFF. If the
   model is stochastic (flow-matching / AR sampling), PASS `--seed N` — else the
   RNG dominates and a huge diff is FALSE (the irodori trap below). Gate:
   ON-vs-ON @same seed must be 0 (deterministic) BEFORE trusting ON-vs-OFF.
   Expect ON-vs-OFF ≈ 0 (byte-identical) or ≤~F16-codec drift; ASR/decoded output
   must be intact. Confirm fastconv ENGAGED (kernels are F16, not a no-op).


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
- ✅ **Shared `core_hifigan::conv1d` FASTCONV overload** — the HiFi-GAN vocoder is
  shared by fastpitch/speecht5/bananamind, so one overload (reusing
  `core_dac::fastconv_cache`) sets up 3 backends. Time-major layout so no
  k=1→matmul, but the F16-cast kill (the main win) applies. Unit test extended
  (`core_hifigan` case, cos>0.99999); 210 assertions total. `nullptr`-default so
  nothing changes until a backend bakes + passes `&fc`.
- 🟡 **chatterbox_s3gen** (`CRISPASR_S3GEN_FASTCONV`, **default OFF / opt-in**) — `41dbc88cc`.
  Cast-kill only (bitwise-identical): bakes F32 copies of the 275 F16 conv kernels
  (100 K>1 + 175 K=1) and pointer-swaps `c->tensors` (same idiom as the ctx_f16 Metal
  fix), so `ggml_conv_1d` skips its F16→F32 cast. Split-load aware (two
  `fastconv_cache`, one per backend); `.ups.` conv_transpose excluded (permute path).
  **Verified:** builds clean; load-time bake+swap+permute runs, engages exactly
  (275 baked/swapped, all GPU on this box, via `CRISPASR_S3GEN_FASTCONV_DEBUG`).
  **NOT yet:** full seeded decoded-output A/B — chatterbox is flow-matching + AR, each
  synth is ~30 min on the current contended M1 (load 300+), too slow for a 3-run
  byte-diff. Default-flip pending that A/B on a quiet box / CUDA. ⚠ GGUF-parse note: only
  the F16 conv kernels benefit; the 79 F32 + 3 ups kernels are untouched (correct).
  **k=1→matmul NOT done** (175 K=1 kernels) — a further win, but changes reduction order
  so it needs its own A/B on the drift-prone GPU path; left for later.
- ✅ **speecht5_tts** (`CRISPASR_SPEECHT5_FASTCONV`, default on) — `1c558b4f0`.
  Threaded the fastconv cache through `core_hifigan::forward()`/`resblock_forward()`
  (one change wires the whole family) + `collect_fastconv_kernels()` helper (excludes
  ups.*). Bakes 74 F16 vocoder convs → F32. **A/B on `speecht5-tts-f16` (deterministic,
  no RNG): ON-vs-ON 0/32768, ON-vs-OFF 0/32768 (byte-identical).** Engagement proven
  via `CRISPASR_SPEECHT5_FASTCONV_DEBUG`: ON bakes 74/74, OFF bakes 0. ASR roundtrip
  intact.
- ⏭ **Next:** fastpitch (blocked on f16 loader, see above) / bananamind (no f16, skip);
  backends with bespoke conv lambdas (chatterbox_s3gen 12 convs, cosyvoice3,
  indextts_voc, kokoro) one at a time — first GGUF-parse each to confirm F16 kernels
  before wiring (the dtype check above is now mandatory triage). Then items 2
  (interval-CFG), 3/4 (Metal q4_k, CI perf gate).

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
