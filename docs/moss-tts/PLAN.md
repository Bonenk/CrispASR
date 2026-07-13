# MOSS-TTS-v1.5 port (#249) — implementation status & validation plan

Branch `feat/moss-tts-249`. Spec: `docs/moss-tts/STUDY.md`. Reference:
`github.com/pwilkin/openmoss` (validated C++ port; we graft its codec + delay
onto CrispASR's in-house Qwen3 runtime — no libllama).

## NOW — active work (update at every checkpoint; push to main)

### 4B `moss-tts-local` (#249 second deliverable) — codec-v2 IN FLIGHT (2026-07-13)

Branch `feat/moss-tts-local-4b` (off `feat/moss-tts-parity-diff`, which carries
the validated P0/P1/P2 — STUDY-4B, converter, backbone+local runtime; NOT yet on
`main`). Spec: `docs/moss-tts/STUDY-4B.md`. Worktree
`.claude/worktrees/moss-tts-local-4b`.

- **DONE (inherited, validated on Kaggle):** P0 STUDY, P1 converter (9.11 GB F16,
  438 tensors, hosted `cstr/moss-tts-local-v1.5-GGUF`), P2 runtime (smoke PASS on
  P100 — valid (12,T) grid, natural stop).
- **IN FLIGHT — P3 codec (MOSS-Audio-Tokenizer-v2) decode.** Read
  `modeling_moss_audio_tokenizer.py` line-by-line (HARD RULE #1). **Decode path
  locked:**
  - Quantizer (ResidualLFQ.decode_codes): per cb `codebook[1024,8][code] →
    WNConv1d 8→512 (+bias)`, sum over the 12 used cbs, then `output_proj WNConv1d
    512→768 (+bias)`. Weight-norm reconstruct `w=g·v/‖v‖` (v1 pattern transfers).
    Codebooks are LEARNED nn.Embedding; decode is a plain lookup (no L2-norm — that
    is encode-only).
  - Decoder = 6 ProjectedTransformer + 6 PatchedPretransform (pure reshape
    upsamplers, NO weights). Dims/ctx (12.5Hz→96kHz interleaved): dec.0 768→1280
    d1280/20h/32L ctx125 · p2 · dec.2 640→768 d768/12h/12L ctx250 · p2 · dec.4
    384→768 ctx400 · p2 · dec.6 384→768 ctx400 · p2 · dec.8 384→768 ctx400 · p2 ·
    dec.10 384→240 ctx400 · p240 → (1, 7680·T) mono-interleaved → de-interleave to
    STEREO (2, 3840·T) @48kHz. Synthesize downmixes (L+R)/2 to the mono API.
  - ⚠ vs v1: **12** cbs (not 32); hop **3840**; **6** stages; **input_proj AND
    output_proj exist on EVERY stage** (e.g. dec.0 output_proj is a real 1280→1280
    matrix — v1's dim-conditional skip would be WRONG); stereo out.
  - Transformer block: pre-norm LayerNorm(w+b, eps 1e-5), fused-QKV NO bias, RoPE
    **NORMAL** (adjacent-pair, base 1e4, head_dim 64), sliding-window causal (ctx),
    scale 1/√64, FFN Linear→**GELU-erf**→Linear (no bias), LayerScale 0.01
    per-channel. No final norm. (Reuses `moss_tts_codec.cpp` block math.)
  - Converter: emit a focused DECODE-only codec GGUF (arch `moss-tts-local-codec`)
    — skip encoder + quantizers 12..31 (~halves size). Runtime:
    `src/moss_tts_local_codec.{h,cpp}`; wire `set_codec_path` + `synthesize`.
- **NEXT:** P4 12-point integration, P5 Kaggle ASR round-trip (the only gate).

---

- **2026-07-13:** Phase 3 code-parity RESOLVED (tokenizer bug fixed; frame-0
  divergence proven = Q4_K near-tie argmax flip, not a bug — see the resolution
  section below). Merged to `main` @ `dd5a21a4`.
- **2026-07-13:** Voice-cloning round-trip **VALIDATED** on Kaggle P100
  (`tools/kaggle/moss-tts-voiceclone/`, clean green `all_pass=True`). Closed-loop
  R/B/C clips, Q4_K: GATE1 no_crash PASS, GATE2 intelligible PASS (word overlap
  1.00 — cloned clip ASR == clone text), GATE3 voice_moved PASS (speaker cosine
  clone↔ref **0.853** vs baseline↔ref **0.471**, Δ **+0.382**). `--voice ref.wav`
  transfers the timbre AND preserves the words.
- **8B (MOSS-TTS-v1.5 / MossTTSDelay) COMPLETE** — Phase 6, code-parity, voice
  cloning all on `main`; shipped in **v0.8.10** (released 2026-07-13).
- **NOW: 4B port (MOSS-TTS-Local-Transformer-v1.5 / MossTTSLocal), #249 second
  deliverable.** Phase 0 STUDY **done** → `docs/moss-tts/STUDY-4B.md` (architecture
  decoded line-by-line: Qwen3-4B backbone [2560d, tied] + a 1-layer local/depth
  transformer that AR-generates 12 codebooks per frame — the delay pattern is
  replaced; codec = MOSS-Audio-Tokenizer-v2 @ 48 kHz).
- **P1 converter DONE + EMPIRICALLY VALIDATED** — `convert-moss-tts-local-to-gguf.py`
  ran on the real 4B weights on Kaggle (`tools/kaggle/moss-tts-local-convert`):
  **9.11 GB F16 GGUF, 438 tensors, arch moss-tts-local, all 6 structural checks
  PASS**, shapes confirmed. See STUDY-4B.md.
- **P2 runtime DONE + COMPILES** — `src/moss_tts_local.{h,cpp}` (+ CMake
  `moss_tts_local` lib, builds clean → `libmoss_tts_local.a`). Backbone reused via
  `core_attn::kv_self_attn`/`core_ffn::swiglu` (2560d, tied lm_head); NEW: pad-masked
  input embeddings (audio_embed size 1024), the 1-layer local/depth transformer
  forward (LayerNorm+bias, fused-QKV, RoPE NEOX 1e6, SiLU), and the depth-first
  generate loop (backbone→local→binary stop→12 codebooks AR). `generate_codes`
  emits the (12, T) grid.
- **Runtime SMOKE test PASSED** (2026-07-13, Kaggle P100,
  `tools/kaggle/moss-tts-local-smoke`) — the hand-written runtime RUNS on real
  weights: loaded n_vq=12/hidden=2560/48kHz, `generate_codes` produced a valid
  (12, T=35) grid, all codes in [1,1022] (0 out-of-range), natural termination
  (binary stop head fired), no crash. Backbone + local/depth transformer +
  depth-first generate loop + stop head all structurally correct. GGUF uploaded to
  `cstr/moss-tts-local-v1.5-GGUF` (9.11 GB, reusable).
- Next: **P3** codec-v2 (48 kHz decode of the codes → audio — the acceptance test
  needs it), **P4** 12-point integration, **P5** ASR round-trip validate. Code
  parity is diagnostic only (quantized AR — see the 8B logit-rank note).
- Issue #249 stays **OPEN** until the 4B ships (only the 8B half is done).

## Done (compiles clean; NOT yet parity-validated)

| Phase | What | Files |
|-------|------|-------|
| 0 | Study (verified vs 2 HF configs + Python blueprint) | `docs/moss-tts/STUDY.md` |
| 1 | GGUF converter (backbone `moss-tts` + companion `moss-tts-codec`); backbone/audio name map unit-tested; codec shortener validated vs all 1600 real codec tensor names | `models/convert-moss-tts-to-gguf.py` |
| 2 | Qwen3-8B backbone (clone of qwen3_asr KV path: QK-norm, NEOX RoPE 1e6, GQA 4:1, SwiGLU) + `hidden_last` output + 2 aux graphs (summed embed, 32 heads) | `src/moss_tts.{h,cpp}` |
| 3 | Delay state machine (openmoss port, incl. sentinel/off-by-one/unique-rep-penalty gotchas) + special-token BPE + prompt builder + AR code-gen loop | `src/moss_tts.cpp` |
| 4 | Transformer RVQ codec decode (weight-norm reconstruction, 4 ProjectedTransformer stages, sliding-window mask, patch upsamples) + end-to-end `synthesize` | `src/moss_tts_codec.{h,cpp}` |
| 5 | CLI adapter, `--backend moss-tts` factory + filename/arch detect, CMake, registry entry, quantize keep-list, **session-ABI inline synthesize** (bindings/server) | `examples/cli/crispasr_backend_moss_tts.cpp`, `crispasr_backend.cpp`, `crispasr_c_api.cpp`, registry, quantize, CMake |

Verified locally: whole runtime builds into `libmoss_tts.a`; `crispasr --backend
moss-tts` routes through the session ABI to the runtime and the registry entry
resolves (backbone + codec companion). All builds 0 errors.

## Remaining Phase 5 (peripheral)

- [ ] **`bindings/go/whisper.go` cgo LDFLAGS sync — CI-ENFORCED, regen on LINUX.**
      A new backend lib (`moss_tts`) without this fails the `Bindings Tests (Go)`
      job (`undefined reference` / `cgo-ldflags-drift`). Run on the VPS:
      `python tools/sync_go_cgo_ldflags.py` then `--check`. Do NOT run on macOS
      (Metal/BLAS pollute the `#cgo linux` line). This is the one item that will
      red CI until done.
- [ ] Bindings docstrings: `python/crispasr/_binding.py`, `bindings/go/`, `flutter/`.
- [ ] Diff-harness reference backend: `tools/reference_backends/moss_tts.py`
      (`dump()` + `DEFAULT_STAGES`) + register in `tools/dump_reference.py`; a
      `moss_tts_<stage>_diff` self-runner in the `.cpp` (dots-tts/voxtral-tts
      pattern) is cleaner than exposing stage APIs.
- [ ] Live test `tests/test_moss_tts_live.cpp` + `tests/env-live-tests.sh` entry.
- [ ] README.md + `docs/{tts,architecture}.md`.

## Phase 6 — VALIDATED (2026-07-12, Kaggle P100, kernel run 1)

**The port works.** Decoded round-trip (HARD RULE #3) **PASSES on Q4_K, CUDA**:
- convert OK (F16 backbone 16.99 GB + codec 3.55 GB), quantize OK (→ Q4_K 7.0 GB,
  252/463 tensors quantized; audio embeds/heads + norms kept F16).
- Q4_K short "Hello world." → ASR "HEllo world!."; Q4_K long → ASR reproduces the
  whole passage ("the quick round fox jumps over the lazy dog. speech synthesis
  should stay intelligible over a longer passage. …"; brown→round, codec→Codex are
  ASR mishears). rms 0.10–0.16, 1.68 s / 22.04 s, proof-of-work TRUE (21→55 words).
- **F16 FAIL = P100 VRAM only, NOT a bug**: the 16.99 GB F16 backbone doesn't fit a
  16.27 GB P100 (`cudaMalloc out of memory` at load). Needs a >24 GB GPU (L4/A100)
  or CPU; Q4_K is the practical target and is proven.
- code-parity ref dump: torch OOM loading the 8B alongside the crispasr process on
  the same 16 GB P100 (non-gating; expected).

**Kernel refinements for a cleaner re-run** (both known bugs, not port issues):
1. `CCACHE_DIR=/kaggle/working/.ccache` bloats the output with the ccache tree →
   `progress.txt` sorts past the 500-file `kernels_output` page cap (usage #22).
   Move ccache to `/kaggle/temp/.ccache` after `install_build_toolchain`; keep
   `/kaggle/working` to just `progress.txt`+`results/`. (Log is still reachable via
   `KaggleApi().kernels_logs(slug)` — used to diagnose this run.)
2. Treat an F16-backbone load-OOM as **SKIP** on ≤16 GB GPUs, not FAIL — gate only
   on Q4_K there; run F16 only when VRAM ≥ ~20 GB.

Ship next: upload GGUFs to `cstr/moss-tts-v1.5-GGUF` (backbone Q4_K + F16 codec;
daemon-thread + timeout + server-side verify per the HF-upload note), populate the
registry `license` (Apache-2.0), version bump, HISTORY + LEARNINGS.

## Phase 3 code-parity — RESOLVED (2026-07-13, Kaggle P100)

Ran the greedy code-parity (C++ Q4_K vs HF BF16 reference on CPU, temps=0) and
methodically diffed the divergence. Two of the PLAN's "known-suspect" areas were
both confirmed — one a real bug, one an intrinsic limit:

1. **Tokenizer (real bug, FIXED — `41c08e8f`).** The moss-tts prompt tokenizer
   (cloned from `qwen3_asr`) used a crude whitespace pre-splitter that split `>`
   from a trailing `\n`. Qwen's pre-tokenizer regex `[^\s\p{L}\p{N}]+[\r\n]*`
   groups punctuation with trailing newlines (`>\n`=397, `):\n`=982). The
   `moss-tts-promptdiff` kernel isolated it: prompt TEXT identical, tokenization
   differed at the first newline. Fix: a proper Qwen2/3 pre-tokenizer
   (`mt_qwen_pretokenize`). After the fix the 67-token conditioning prompt is
   **byte-identical** to the HF processor (`first_mismatch=null`, `prefix=67`).

2. **F16-vs-BF16 rounding through the AR loop (intrinsic — NOT a bug).** Even with
   byte-identical prompts, greedy codes diverge from **frame 0** (~0.5% exact
   match). The `moss-tts-logit0` probe settles it: at step-0 head-0, the C++ Q4_K
   logits are essentially the reference's distribution — the reference's greedy
   pick (143) is the C++'s **rank-1 runner-up**, only **0.135 logits** below the
   C++ argmax (1021). Q4_K rounding (O(0.1–0.5) on logits) flips this near-tie
   onset token; the MOSS delay makes every un-delayed frame span many raw AR
   steps, so one flip cascades (~n_vq·T decisions → ~0.1% survival = the 0.5%
   observed). Codebook-0 (coarse RVQ) still re-syncs to exact reference values at
   scattered frames (807, 578, 756; 2-frame-shifted runs 400/575/254) — the
   models produce the same coarse audio, differing only on quant-sensitive fine
   residuals.

**Conclusion:** exact greedy code-parity between ggml-Q4_K and torch-BF16 is
*unachievable* for this AR audio LM (near-tie onset flips + AR chaos + dtype
mismatch), exactly as the last PLAN bullet predicted ("judge by the deterministic
prefix + round-trip, not aggregate cos"). The port is **structurally confirmed**:
byte-identical prompt + near-identical step-0 logit distribution + coarse-codebook
re-sync + the passing ASR round-trip (HARD RULE #3). The correct acceptance gate
is the round-trip (passes) and the step-0 logit-rank probe (ref pick = C++ rank-1),
NOT byte-exact greedy codes. Kernels: `tools/kaggle/moss-tts-{promptdiff,parity,
logit0}/`.

## Phase 6 — original validation plan (the ONLY acceptance test; HARD RULE #3)

8B backbone won't fit the 8 GB VPS and is tight on the 16 GB Mac with the 1.6 B
codec → run on **Kaggle** (P100/T4). Reference kernels:
`tools/kaggle/voxtral-diff-harness/`, `tools/kaggle/tada-bucket-ab/`.

1. **Convert** on Kaggle (stage model pulls under `/tmp` ~70 GB, not
   `/kaggle/working` ~20 GB): `python models/convert-moss-tts-to-gguf.py --input
   OpenMOSS-Team/MOSS-TTS-v1.5 --codec OpenMOSS-Team/MOSS-Audio-Tokenizer
   --output moss-tts-v1.5-f16.gguf` → F16 backbone + F16 codec. Then
   `crispasr-quantize moss-tts-v1.5-f16.gguf moss-tts-v1.5-q4_k.gguf q4_k`.
2. **Code parity** (Phase 3 gate): greedy code streams byte-identical to the
   Python/pwilkin reference for a fixed text+seed. Build the parity dumper first
   (reference backend). Watch the delay-fill boundary + the sentinel warm-up.
3. **Codec parity** (Phase 4 gate): per-stage cos ≥ 0.999 vs the ONNX/PyTorch
   tokenizer, then the decoded audio. Gate input alignment BEFORE trusting
   per-layer cos. ⚠ CUDA `get_rows` needs a contiguous index (dev-guide §232) —
   audit the codebook lookups if decode aborts on P100.
4. **Decoded round-trip** (the real test): `crispasr --backend moss-tts
   --model <gguf> --tts "..." --tts-output out.wav` → ASR `out.wav` (whisper) →
   text recognizable. Test **F16 AND Q4_K** (quant amplifies divergence). Add an
   `expected_text` regression entry.

Known-suspect areas to check first if parity fails (all faithful clones but
unverified): the codec attention (manual SDPA + sliding-window mask vs openmoss's
flash path — output-equivalent but confirm), the prompt tokenization (special-token
BPE must match the Python tokenizer exactly), and F16 vs BF16 rounding through the
deep AR loop (judge by the deterministic prefix + round-trip, not aggregate cos).

## Follow-ups (documented, out of core scope)

- Voice cloning: the codec **encoder** (ref audio → codes). openmoss `codec.cpp`
  encode path + `core_rvq::encode_euclidean`; wire the reference-audio prompt grid
  in the AR loop. Separate PR.
- 4B `MossTTSLocal` (48 kHz stereo, depth-transformer, tokenizer-v2). No C++ ref.
- Perf: codec uses manual SDPA (O(T²) scores/mask); switch to flash_attn_ext +
  persistent graph if long-audio memory bites (openmoss's reason for flash).
