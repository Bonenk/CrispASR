# MOSS-TTS-v1.5 port (#249) — implementation status & validation plan

Branch `feat/moss-tts-249`. Spec: `docs/moss-tts/STUDY.md`. Reference:
`github.com/pwilkin/openmoss` (validated C++ port; we graft its codec + delay
onto CrispASR's in-house Qwen3 runtime — no libllama).

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

## Phase 6 — validation (the ONLY acceptance test; HARD RULE #3)

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
