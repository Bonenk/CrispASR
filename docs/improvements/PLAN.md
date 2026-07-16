# CrispASR — cross-cutting improvements program

Five high-leverage improvements surfaced by the #257 marathon. Every runtime
change ships **behind an env gate** (A/B without recompile, never delete the
working path — per the dev-guide), with an **A/B method** and **unit tests**.

## NOW — active work

- [ ] **Phase 0** — cross-surface parity harness + test (safety net; must land first)
- [ ] **Phase 1** — collapse the dual dispatch (session C-ABI → shared adapter)
- [ ] **Phase 2** — unified encoder memory policy (proactive, replaces ad-hoc gates)
- [ ] **Phase 3** — diff-harness parity in CI (per-stage cos + decoded roundtrip)
- [ ] **Phase 4** — server throughput (batching / worker pool)

Execution order is deliberately **0 before 1**: the parity test is the guard that
proves the dispatch unification changes nothing observable. Each phase merges to
`main` green before the next starts.

---

## Phase 0 — cross-surface parity harness + test

**Why:** #257 needed the identical fix in CLI adapter, server, and session C-ABI
because they are three separate code paths. A test that pins "all surfaces agree"
would have caught the wiring gap on commit 1, and is the precondition for safely
unifying them (Phase 1).

**Deliverable:** `tests/test-surface-parity.sh` (live, needs a model) — runs the
same clip through (a) `crispasr` CLI, (b) an in-process session via the Python
binding, (c) the HTTP server, and asserts identical segment text/offsets. Plus a
pure **unit test** for the params→`whisper_params` marshalling helper Phase 1 adds.

**Env gate:** none (test infra). Gated to run only when `CRISPASR_MODELS_DIR` +
a parakeet model are present (label `live;parity`).

**A/B method:** the test itself is the A/B — CLI vs session vs server on the same
seed/clip; diff the JSON `transcription[]`.

**Acceptance:** three surfaces byte-identical on parakeet (F16 + Q4_K) for a short
clip and a `--chunk-seconds 7` clip; unit test green.

## Phase 1 — collapse the dual dispatch

**Why:** `crispasr_session_transcribe*` reimplements each backend's transcribe
inline (dev-guide HARD RULE #6) instead of calling the `CrispasrBackend` adapter
the CLI/server use. Every fix/feature/default risks landing in one path but not
the other (JA-detection was patched in ~5 places; #257 segmentation in 3).

**Approach:** add a session→`whisper_params` marshaller (sticky `source/target_lang`,
chunk, att-context, hotwords, …) and route `crispasr_session_transcribe*` through
`backend->transcribe()` for backends that have an adapter. Delete the inline
per-backend branch once parity holds. Start with **parakeet** (freshest), then
canary/cohere/granite/etc.

**Env gate:** `CRISPASR_SESSION_UNIFIED_DISPATCH` — `1` routes through the adapter
(new), `0` keeps the inline path (old). Default `0` until parity proven per
backend, then flip to `1` and keep `0` as the A/B escape hatch.

**A/B method:** Phase-0 parity test with the gate `0` vs `1` — must be byte-identical
before flipping. Judge decoded output (HARD RULE #3), F16 + Q4_K.

**Unit tests:** marshaller mapping (session fields → `whisper_params`), incl. the
sticky-language and chunk/att-context/overlap fields; a regression that the gate
default is the proven value.

**Acceptance:** parakeet session output identical gate 0 vs 1 across the parity
matrix; measured LOC deleted from the inline branch reported in this doc.

## Phase 2 — unified encoder memory policy

**Why:** windowed local attention is bit-exact + ~3× faster + lower memory yet
still opt-in (`--att-context` + `CRISPASR_FC_WINDOWED_ATTN`), and we just bolted a
*reactive* single-pass-OOM fallback on top (issue #257). Three scattered gates +
a catch-and-retry instead of one decision.

**Approach:** a `parakeet_pick_encode_strategy(T, backend, free_vram?)` that chooses
single-pass / windowed / streamed proactively (bound the O(T²) bias you can't
afford before allocating it). Keep the reactive fallback as a backstop.

**Env gate:** `CRISPASR_PARAKEET_MEM_POLICY` = `auto` (new default) | `single` |
`windowed` | `streamed` | `off` (current reactive-only behaviour). Never removes
the existing `--att-context` / `--chunk-seconds` / `CRISPASR_FC_WINDOWED_ATTN`.

**A/B method:** back-to-back on the reporter's 225 s clip + a long clip; decoded
output equality vs single-pass within tolerance + peak-footprint (`phys_footprint`
on macOS) and, where possible, a CUDA cross-check (R5 open item).

**Unit tests:** the pure strategy-selection function (T, caps, vram → strategy)
across boundary cases; no model needed.

**Acceptance:** `auto` never OOMs where single-pass would, output within tolerance,
peak memory ≤ single-pass; policy table documented.

## Phase 3 — diff-harness parity in CI

**Why:** the `dump_reference → crispasr-diff` methodology is the crown jewel but
run by hand. Automating it catches the "cos 0.99 snowballs into a hallucination"
class (and the TTS "component cos=0.999 but audio garbage" case) on every push.

**Approach:** a CI job (nightly / label-gated) that, per backend with a hosted
`ref.gguf` (dataset `cstr/crispasr-regression-fixtures`), runs the per-stage diff
+ the decoded-output roundtrip and reds on cos / word-overlap regression. Extend
the manifest beyond ASR-transcript-only to include a TTS→ASR roundtrip gate.

**Env gate:** n/a (CI infra); thresholds configurable via the manifest.

**A/B method:** self-checking (compares runtime vs hosted reference).

**Unit tests:** the threshold/verdict logic (cos_min, word-overlap → PASS/FAIL)
as a pure function.

**Acceptance:** at least parakeet + one TTS backend gated in CI; a deliberately
broken quant reds the job.

## Phase 4 — server throughput

**Why:** the server serialises on a single `model_mutex` — one request at a time.

**Approach:** request batching / a small worker pool / KV reuse. Bigger lift;
scoped last.

**Env gate:** `CRISPASR_SERVER_WORKERS=N` (default 1 = current behaviour).

**A/B method:** concurrent-request throughput + latency, N=1 vs N>1, identical
transcripts.

**Unit tests:** the queue/dispatch logic (no model).

**Acceptance:** throughput scales with workers, transcripts unchanged vs N=1.
