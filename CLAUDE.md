# CrispASR — Claude Code project instructions

## ⛔ READ FIRST — `crispasr-crispembed-dev.md`

**Before writing any code, read the development guide in full**: it sits beside
this repo's checkout (on the Mac: `~/code/crispasr-crispembed-dev.md`). It is
~1000 lines and is *not* summarised anywhere — this file deliberately does not
restate it.

It is the method: the HARD RULES, the convert → quantize → dump-reference →
diff → parity port pipeline, the ggml/GPU portability gotchas, the A/B
discipline, the machine + storage layout for both boxes. Skipping it is how the
recurring bugs recur. If a rule here and a rule there disagree, **the dev guide
wins** — this file is a router, not a source.

## Where everything lives

| Need | Read |
|---|---|
| How to work here (method, rules, patterns) | `crispasr-crispembed-dev.md` — **all of it, first** |
| A specific past lesson ("has this bitten us?") | `LEARNINGS.md` — grep it, never read it whole (800 KB, 227 lessons) |
| Is this backend fast / which quant ships | `PERFORMANCE.md` |
| What is in flight right now | `PLAN.md` — and claim your task there before starting |
| What already shipped | `HISTORY.md` — archive; consult only to confirm a claim |
| Adding a backend | `docs/contributing.md` (12-point checklist) |

**Budget your reading.** These total >2 MB. Only the dev guide is meant to be
read end to end; everything else is grep-first, and `PLAN.md`/`HISTORY.md`
prose is frequently stale — audit against the CODE, never the note.

## Build

```bash
CCACHE_DIR=/mnt/volume1/.ccache cmake -G Ninja -B build \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Test

```bash
# Unit tests (no models needed, ~5s):
ctest --test-dir build -L unit --timeout 30

# Integration / live tests (need models):
export CRISPASR_MODELS_DIR=/mnt/storage/gguf-models
source tests/env-live-tests.sh
ctest --test-dir build --rerun-failed --output-on-failure --timeout 300
```

Key env vars: `CRISPASR_MODELS_DIR`, `CRISPASR_MODEL_WHISPER`, `CRISPASR_PARAFORMER_MODEL`.
See `tests/env-live-tests.sh` for the full list.

## Lint

```bash
./tools/format.sh --fix <changed .cpp/.h files>
```

Must use clang-format v18 — `tools/format.sh` enforces this.

## Storage

- **GGUF models**: `/mnt/storage/gguf-models` — all converted GGUF files live here
- **NEVER use `/tmp`** — it is a tiny tmpfs that fills up and kills processes. Use `/mnt/volume1/tmp-overflow` for temp files, or write directly to `/mnt/storage` or `/mnt/volume1`.
- **HuggingFace cache**: `/mnt/akademie_storage/huggingface/hub/`
- **ccache**: `/mnt/volume1/.ccache`

## Commit workflow

- No PRs — merge locally and push to main directly
- Always run `tools/format.sh --fix` on changed C/C++ files before committing
- Work in git worktrees for non-trivial changes
