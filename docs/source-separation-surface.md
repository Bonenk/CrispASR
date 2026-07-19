# Source-separation surface (§248) — shared design

Music/voice **source separation** is a new task category, distinct from ASR
(text out) and TTS (synthetic audio out): it takes one mixed audio input and
returns **N named stems of the user's own audio**. Two backends target it —
`htdemucs` (4-stem: drums/bass/other/vocals) and `mel-band-roformer`
(vocal/instrumental) — plus future ones (mel-band RoFormer variants, etc.).

This doc is the **single agreed surface** both backends route through, so we do
not grow two parallel CLI flags / output conventions. Authored by the M1/Metal
session (mel-band-roformer), 2026-07-19, at the maintainer's instruction to
design the shared surface now. The htdemucs session should adopt it.

## Why not the `transcribe()` backend interface

Separation returns **audio, not `crispasr_segment`s**, so it is NOT a
capability layered onto the ASR `transcribe()` path. It gets its own dispatch.
A capability bit (`CAP_SEPARATE`, TBD when wired) is only for detection/help
text, not for routing through the transcription loop.

## CLI

```
crispasr --separate -m <model.gguf> -f mix.flac [--stems vocals,drums] [--sep-output-dir DIR]
```

- **`--separate`** enables the task (alias intent: `--task separate`). Routes to
  the separation dispatcher BEFORE the ASR backend is constructed.
- Backend is auto-detected from the GGUF `general.architecture`
  (`htdemucs` / `mel-band-roformer`) via the normal `-m` resolution, so the user
  never names the backend for separation.
- **`--stems LIST`** — comma-separated subset to write (case-insensitive);
  empty / `all` writes every stem. A backend ignores names it doesn't have.
- **Output**: one WAV per selected stem, named
  `<input-stem>_<source>.wav` next to the input, or under `--sep-output-dir`.
  Stereo, 16-bit PCM, at the model's native rate (44100). **No AI-provenance
  INFO chunk** — the audio is the user's, not AI-generated.

## Shared code (landed, additive)

`src/core/separation_io.h` (header-only, unit-tested — `tests/test-separation-io.cpp`):

- `struct crispasr_separation_view` — backend-agnostic, **non-owning** result:
  `n_sources, n_channels, n_frames, sample_rate, sources[], source_names[]`.
  Each backend fills this from its own result struct (e.g. `htdemucs_result`,
  which already has exactly these fields) — no backend refactor required.
- `crispasr_stem_output_path(input, source, out_dir, ext="wav")` — deterministic
  naming (extension-strip, dir handling, lowercase source).
- `crispasr_stem_selected(csv, source)` — `--stems` membership.
- `crispasr_stem_to_wav(view, s)` — one stem → interleaved WAV blob via
  `crispasr_make_wav_int16_interleaved` (new multi-channel writer in
  `crispasr_wav_writer.h`, no AI tag).

## Dispatcher (next increment)

`examples/cli/crispasr_separate_cli.{h,cpp}` (NEW file — keeps the shared
`cli.cpp` footprint to just flag parsing + one early-dispatch hook, minimizing
collision with the in-flight htdemucs integration):

```
int crispasr_run_separate(const whisper_params& params);
  1. resolve -m, detect arch
  2. read audio -> stereo float @ model rate (reuse audio_resample/wav_reader)
  3. arch == htdemucs         -> htdemucs_init_from_file + htdemucs_separate
     arch == mel-band-roformer-> mel_band_roformer_init + _separate  (pending)
  4. wrap result in crispasr_separation_view
  5. for each source: if crispasr_stem_selected(--stems) -> write
     crispasr_stem_output_path(...) with crispasr_stem_to_wav(...)
```

Both backend `*_separate()` C APIs already return the same shape
(`float** sources` + `const char** source_names`), so the wrapper is a few
lines per backend.

## Coordination

- The shared header + WAV writer + tests are **additive** (no existing file
  depends on them) — safe to land while htdemucs is mid-integration.
- The htdemucs session, when it wires its CLI, should call
  `crispasr_run_separate` and adopt this naming, not add a second `--separate`
  path. If it has already started one, we reconcile to THIS spec (one surface).
- `mel_band_roformer_{init,separate}` (the C API mirroring `htdemucs.h`) lands
  with the MBR C++ backend; until then the dispatcher's MBR branch is stubbed.

## RESOLVED — one surface (2026-07-19)

The `--separate` dispatcher (this spec) shipped and drives BOTH backends. But
the htdemucs session independently took a **different** surface:
`examples/cli/crispasr_backend_htdemucs.cpp` — an `HtdemucsBackend :
CrispasrBackend` with a `CAP_SEPARATE` capability whose `transcribe()` runs
separation and returns a synthetic `"[separated: …]"` segment (the audio can't
travel through the transcript path, so the stems are stashed for the CLI to
pull out separately). So there are now TWO ways to separate:

| surface | trigger | shape |
|---|---|---|
| **`--separate`** (this spec, agreed) | `--separate` | standalone task, audio out, both backends via one dispatcher |
| `CAP_SEPARATE` adapter | `--backend htdemucs` | routed through `transcribe()`, fake text segment, stem-stash hack |

They don't crash together (`--separate` early-routes in `cli.cpp` before
backend detection), but shipping two is confusing and the adapter fights the
transcribe contract. **Per the maintainer's decision that `--separate` is the
shared surface, the `CAP_SEPARATE` transcribe-adapter should be removed and
`--backend htdemucs` (if kept at all) should just point users at `--separate`.**
Not deleting the other session's active file unilaterally — flagged here for the
maintainer to have that session converge, or to authorize this session to do it.
