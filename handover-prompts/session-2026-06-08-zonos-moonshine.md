# Handover — 2026-06-08 Zonos/moonshine/O15 session

## What was done

### §140 pocket-tts GPU/sched migration — DONE
- Migrated from `gguf_init_from_file` + `gallocr` to `core_gguf::load_weights`
  (mmap) + `ggml_backend_sched`. All 6 TTS backends now sched-wired.
- **Bug fix:** weights must load to CPU (`backend_cpu`) not GPU, because
  the flow head + AR loop use eager CPU code (`tensor_f32_data` reads
  `t->data` directly). Sched auto-copies for graph ops.

### Moonshine mmap/sched migration — DONE
- Both `moonshine.cpp` and `moonshine_streaming.cpp` migrated.
- Replaced `gguf_init_from_file` + manual fread + per-graph `gallocr`
  with `core_gguf::open_metadata` + `load_weights` + `sched`.
- Added `use_gpu` to both param structs, wired via C API.

### §130 Zonos TTS — IN PROGRESS
- **DAC decoder graph-building** implemented in `core/dac_decoder.h`:
  `conv1d`, `convt1d`, `res_unit`, `dec_block`, `build_decode_graph`.
- **DAC wired into `zonos_tts_synthesize`**: lazy codec load from GGUF,
  replaces the 440 Hz sine placeholder.
- **Tensor name mismatch fixed**: converter outputs `decoder.block.0.conv_t1.weight`
  style names, not the `dac.dec.blk.0.1.weight` from header docs.
- **DAC GGUF** converted and uploaded to `cstr/dac-44khz-GGUF` (104 MB).
- **Speaker emb env var**: `ZONOS_SPEAKER_EMB_PATH` override added.
- **Kaggle GPU test kernel** pushed: `tools/kaggle/zonos-tts-gpu-test/`

**Verified locally:** model loads (246 tensors), conditioning + prefill +
AR decode start works. Full end-to-end needs GPU (500M params + CFG = too
slow for the 8GB VPS).

**Remaining for Zonos:**
1. GPU end-to-end via Kaggle kernel (pushed, awaiting results)
2. ASR roundtrip verification
3. Speaker encoder (ResNet293) — currently random Gaussian fallback
4. `docs/tts.md` entry
5. Phonemizer: espeak-ng popen works, shared G2P dict cascade blocked
   by link dependency (zonos-tts lib doesn't link phonemizer.o)

### §52 O15 CUDA test — kernel v4 running
Kernel went through 4 iterations fixing test harness bugs:
- v1: no `--voice` (qwen3-tts requires voice ref)
- v2: no `--i-have-rights` (consent gate)
- v3: 16kHz jfk.wav (qwen3-tts requires 24kHz)
- v4: scipy resample 16k→24k — should finally work

**v4 running on Kaggle P100, ~15 min remaining.**

### §155 CONV_TRANSPOSE_1D optimization — PLAN entry added
Crash fixed in `f8fc8b8e`, but kernel still 3× slower than CPU for TTS
codec workloads. User @Rafa00127: codec GPU 1198ms vs CPU-fallback 396ms
on RX 7900 XTX. Documented optimization approaches in PLAN.md.

### §115 mimo-asr GPU — CLOSED (from previous session, committed here)

### §52 O15 — CONFIRMED BROKEN on CUDA
Kernel v4 on P100: O15=OFF works (27.3 ms/frame, ASR OK), O15=ON crashes
(rc=-6, CUDA error). Cross-architecture (sm_60 + sm_87). Stays default OFF.

### CI fixes — 20 failures → 0
- piper_tts.cpp: `crispasr_cache::ensure_cached_file` link error — guarded
  behind `#ifdef CRISPASR_BUILD` + added define to CMake
- clang-format-18: g2p_en.h, ipa_convert.h, piper_tts.cpp
- bark-small: missing voice_preset in regression manifest

### Kaggle harness HF token fix
Kaggle changed dataset mount from `/kaggle/input/<slug>/` to
`/kaggle/input/datasets/<owner>/<slug>/`. Fixed `kaggle_token_from_dataset()`
to scan both roots. Also fixed kernel-metadata.json boolean→string.
**Confirmed working:** v6 kernel log shows token loaded from new path.

## Build

```bash
cd /mnt/volume1/CrispASR
cmake --build build --target crispasr-cli -j$(nproc)
```

## Kaggle kernels

```bash
# kernels status endpoint returns 500 — use kernels_output instead:
PYTHONPATH=/tmp/kaggle_pkg python3 -m kaggle kernels output ${KAGGLE_ACCOUNT}/crispasr-zonos-tts-gpu-test -p /tmp/zonos-results
PYTHONPATH=/tmp/kaggle_pkg python3 -m kaggle kernels output ${KAGGLE_ACCOUNT}/crispasr-zonos-diff -p /tmp/zonos-diff-results

# Max 2 concurrent GPU sessions. Check running:
PYTHONPATH=/tmp/kaggle_pkg python3 -c "
import json
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
for r in api.kernels_list(user='${KAGGLE_ACCOUNT}', page_size=5):
    d = json.loads(str(r))
    print(d.get('ref'), d.get('lastRunTime'))
"
```

## Key paths

| What | Path |
|------|------|
| pocket-tts (migrated) | `src/pocket_tts.cpp` |
| moonshine (migrated) | `src/moonshine.cpp` |
| moonshine streaming (migrated) | `src/moonshine_streaming.cpp` |
| Zonos TTS runtime | `src/zonos_tts.cpp` |
| DAC decoder core | `src/core/dac_decoder.h` |
| DAC converter | `models/convert-dac-to-gguf.py` |
| Zonos converter | `models/convert-zonos-to-gguf.py` |
| Zonos reference | `tools/reference_backends/zonos_tts_reference.py` |
| Zonos CLI adapter | `examples/cli/crispasr_backend_zonos.cpp` |
| O15 kernel (done) | `tools/kaggle/qwen3-tts-o15-cuda/` |
| Zonos test kernel | `tools/kaggle/zonos-tts-gpu-test/` |
| Zonos diff kernel | `tools/kaggle/zonos-diff/` |
| Kaggle harness | `tools/kaggle/kaggle_harness.py` |
| PLAN | `PLAN.md` (§52, §130, §155) |
| HISTORY | `HISTORY.md` (2026-06-08 entries) |

## What's next

1. **Check Zonos v6 kernel** — has en-us fix + harness HF token fix.
   If ASR still hallucinates, the diff kernel is the next step.
2. **Zonos diff harness** — `tools/kaggle/zonos-diff/` kernel compares
   Python reference vs C++ stage-by-stage. Push when GPU slot opens.
3. **Zonos quality**: the `cb0 argmax=110` being identical across all
   conditioning variants suggests the backbone may not be routing the
   prefix conditioning through the KV cache correctly. Diff harness
   will pinpoint where.
4. **Zonos speaker encoder** — ResNet293 port or pre-computed embeddings.
5. **§155 optimization** — tiled conv_transpose_1d kernel.
6. **§58 MOSS-Audio** — next large backend.
