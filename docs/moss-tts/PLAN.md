# MOSS-TTS-v1.5 port (#249) — implementation status & validation plan

Branch `feat/moss-tts-249`. Spec: `docs/moss-tts/STUDY.md`. Reference:
`github.com/pwilkin/openmoss` (validated C++ port; we graft its codec + delay
onto CrispASR's in-house Qwen3 runtime — no libllama).

## NOW — active work (update at every checkpoint; push to main)

- **2026-07-13:** Phase 3 code-parity RESOLVED (tokenizer bug fixed; frame-0
  divergence proven = Q4_K near-tie argmax flip, not a bug — see the resolution
  section below). Merged to `main` @ `dd5a21a4`.
- **2026-07-13:** Voice-cloning round-trip **VALIDATED** on Kaggle P100
  (`tools/kaggle/moss-tts-voiceclone/`). Closed-loop R/B/C clips, Q4_K:
  GATE1 no_crash PASS (rc=0, non-silent), GATE3 voice_moved PASS strongly
  (speaker cosine clone↔ref **0.845** vs baseline↔ref **0.585**, Δ **+0.26**),
  GATE2 intelligible = real transcript "Cloning should keep my voice while
  changing the words entirely." (== clone text, 100% word overlap; the reported
  FAIL was an ASR-stdout truncation artifact in the harness, now fixed). Net:
  `--voice ref.wav` transfers the timbre AND preserves the words.
- **In flight:** clean re-run of the voiceclone kernel with the fixed ASR parser
  to record a green `all_pass=True` artifact.
- **Next:** on clean green, mark voice cloning validated in HISTORY/LEARNINGS +
  docs and close task #8.

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
