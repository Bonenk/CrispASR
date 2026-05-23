# CrispASR — Claude Code project memo

## What this is
CrispASR: single C++ binary, 24 ASR backends + 5 TTS engines + translation, zero Python deps. Fork of whisper.cpp. Remote: `cohere` → github.com/CrispStrobe/CrispASR.

## Build
```bash
cd /mnt/storage/whisper.cpp          # build dir lives here, not akademie_storage
cmake --build build -j$(nproc) --target crispasr-cli
```
The cmake cache points at `/mnt/storage/whisper.cpp/build`. Binary: `build/bin/crispasr`.

## Test
```bash
cd /mnt/storage/whisper.cpp/build && ctest --output-on-failure -j4   # unit + integration
build/bin/crispasr -m models/ggml-base.en.bin -f samples/jfk.wav     # whisper smoke test
```
For TTS, always run ASR roundtrip via whisper or parakeet on the generated wav.

## Storage layout
- `/mnt/storage` → symlink to `/mnt/akademie_storage` (5 TB Hetzner storage box, CIFS mount)
- Main disk `/dev/sda1` is 75 GB — keep it under 80 %. All large files on `/mnt/storage`.
- Worktrees: `git worktree add /mnt/storage/worktree-<name>` (NOT on main disk)
- Build dirs: `/tmp/build-<name>` (local SSD, avoids CIFS symlink issues with cmake)
- Auto-download cache: `--cache-dir /mnt/storage/cache` to keep models off main disk

## Models
- Small test models: `models/` dir (ggml-base.en.bin, ggml-tiny.en.bin, silero VAD)
- Full GGUFs: `/mnt/storage/` subfolders — cohere/, parakeet-tdt-0.6b-v3/, chatterbox/, etc.
- HF cache GGUFs: `/mnt/storage/huggingface/hub/models--cstr--*/`
- Auto-downloaded: `/mnt/storage/cache/` (canary-1b-v2-q4_k.gguf, etc.)
- Quantized models: `/mnt/storage/models/` (moonshine, wav2vec2, indextts, titanet, etc.)
- Pyannote seg: `/mnt/akademie_storage/test_cohere/pyannote-seg-3.0.gguf`
- Diarization test audio: `/mnt/storage/mini-diarization/` (Kaggle mini-speech-diarization)
- Benchmark test audio: `/mnt/storage/test-audio/` (en/de/ja/zh × 4 durations from FLEURS)

## Key paths
- `src/` — all library code, per-model runtimes, C-ABI (`crispasr_c_api.cpp`)
- `examples/cli/` — CLI binary, backend adapters (`crispasr_backend_*.cpp`)
- `examples/cli/crispasr_run.cpp` — main dispatch: VAD → slice → backend → diarize → output
- `examples/cli/crispasr_llm_pipeline.h` — shared LLM decode loop (voxtral, granite, glm, etc.)
- `src/core/` — shared primitives (mel, FFN, rotary, weight loading, greedy_decode)
- `src/core/greedy_decode.h` — `core_greedy_decode::Config` with seed/temperature
- `include/crispasr.h` — public C API header
- `tools/test-all-backends.py` — regression matrix runner

## Formatting
clang-format v18 only (never v22). Use `./tools/format.sh --fix`.

## Git workflow
- `cohere` remote = CrispStrobe/CrispASR (our fork)
- `origin` remote = ggerganov/whisper.cpp (upstream)
- Work on `main` branch, push to `cohere`
- **Always use `git worktree add`** for branch work — never checkout branches directly in this working tree

## SCP to MacBook
```bash
scp root@168.119.190.252:/mnt/akademie_storage/whisper.cpp/handover-prompts/<file>.md .
```

## Seed / reproducibility
`--seed N` wired through all backends with stochastic sampling. 0 = non-deterministic.
Backends: parakeet, canary, cohere, qwen3-asr, voxtral4b, granite, glm-asr, kyutai-stt,
moonshine, qwen3-tts, chatterbox, indextts, voxcpm2. Also available via server API
`/v1/audio/speech` JSON `"seed"` field.
