# crispasr

Node.js package for CrispASR speech recognition

Package: https://www.npmjs.com/package/crispasr

## Details

The performance is comparable to when running `crispasr` in the browser via WASM.

The API is currently very rudimentary: [bindings/javascript/emscripten.cpp](/bindings/javascript/emscripten.cpp)

For sample usage check [tests/test-crispasr.js](/Users/christianstrobele/code/CrispASR/tests/test-crispasr.js)

## Package building + test

```bash
# load emscripten
source /path/to/emsdk/emsdk_env.sh

# clone repo
git clone https://github.com/CrispStrobe/CrispASR
cd crispasr

# grab base.en model
./models/download-ggml-model.sh base.en

# prepare PCM sample for testing
ffmpeg -i samples/jfk.wav -f f32le -acodec pcm_f32le samples/jfk.pcmf32

# build
mkdir build-em && cd build-em
emcmake cmake .. && make -j

# run test
node ../tests/test-crispasr.js

# For Node.js versions prior to v16.4.0, experimental features need to be enabled:
node --experimental-wasm-threads --experimental-wasm-simd ../tests/test-crispasr.js

# publish npm package
make publish-npm
```

## Sample run

```text
$ node --experimental-wasm-threads --experimental-wasm-simd ../tests/test-crispasr.js

whisper_model_load: loading model from 'whisper.bin'
whisper_model_load: n_vocab       = 51864
whisper_model_load: n_audio_ctx   = 1500
whisper_model_load: n_audio_state = 512
whisper_model_load: n_audio_head  = 8
whisper_model_load: n_audio_layer = 6
whisper_model_load: n_text_ctx    = 448
whisper_model_load: n_text_state  = 512
whisper_model_load: n_text_head   = 8
whisper_model_load: n_text_layer  = 6
whisper_model_load: n_mels        = 80
whisper_model_load: f16           = 1
whisper_model_load: type          = 2
whisper_model_load: adding 1607 extra tokens
whisper_model_load: mem_required  =  506.00 MB
whisper_model_load: ggml ctx size =  140.60 MB
whisper_model_load: memory size   =   22.83 MB
whisper_model_load: model size    =  140.54 MB

system_info: n_threads = 8 / 10 | AVX = 0 | AVX2 = 0 | AVX512 = 0 | NEON = 0 | F16C = 0 | FP16_VA = 0 | WASM_SIMD = 1 | BLAS = 0 |

operator(): processing 176000 samples, 11.0 sec, 8 threads, 1 processors, lang = en, task = transcribe ...

[00:00:00.000 --> 00:00:11.000]   And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.

whisper_print_timings:     load time =   162.37 ms
whisper_print_timings:      mel time =   183.70 ms
whisper_print_timings:   sample time =     4.27 ms
whisper_print_timings:   encode time =  8582.63 ms / 1430.44 ms per layer
whisper_print_timings:   decode time =   436.16 ms / 72.69 ms per layer
whisper_print_timings:    total time =  9370.90 ms
```

## Note transcription

Open a note model with `ttsOpenExplicit(path, backend, nThreads)`, where
`backend` is `basic-pitch`, `piano-transcription`, `mt3`, `onsets-and-frames`,
or `hft-transformer`. `sessionPianoSampleRate()` returns its input rate
(22050 Hz for basic-pitch, 16000 Hz for the other four), or 0 without a note
model. Pass mono `Float32Array` PCM at that rate to `sessionPianoNotes(audio)`:

```js
const notes = module.sessionPianoNotes(pcm);
// [{onMs, offMs, midi, velocity, program}, ...]
```

Times are milliseconds. MT3 reports General MIDI programs; other backends use
`program: -1`. Empty input, no session, or a failed inference returns `[]`.


## Session load options and registry

`asrOpenWithOptions(path, backend, options)` and
`ttsOpenWithOptions(path, backend, options)` expose the native v2 load settings:

```js
const opened = module.asrOpenWithOptions('/models/model.gguf', '', {
  n_threads: 1,
  use_gpu: false,
  verbosity: 0,
  flash_attn: true,
  n_gpu_layers: -1,
});
```

Use an empty backend for native auto-detection. `null` options use defaults:
one thread in a single-thread module, four in a threaded module, GPU enabled
when available, quiet logging, flash attention enabled and maximum GPU layers.
Zero threads selects the module default. A single-thread module rejects thread
counts above one. Invalid options throw before replacing the current session;
a valid open closes the old singleton first and returns `false` if loading fails.
These synchronous functions have the same browser scheduling constraints as
`asrOpen`/`ttsOpenExplicit`. In a proxied pthread module, use
`asrOpenWithOptionsAsync(path, backend, options, callback)` or
`ttsOpenWithOptionsAsync(...)` for nonblocking loading.

`registryLookup(backend)` and `registryLookupByFilename(filename)` return
`{filename, url, approxSize}`, or `null` on a miss. `registryBackends()` returns
an array of names. `registryDefaultBundle(backend)` returns
`{backend, license, requiresAcceptance, artifacts}`, or `null` for an unknown
backend. Each artifact contains the entry fields plus `kind`: 0 primary,
1 companion, 2 extra. A known backend with an empty bundle returns an empty
artifact array. License metadata does not itself record user acceptance.

`cacheDirectory(overrideDir)` and
`cacheEnsureFile(filename, url, quiet, overrideDir)` expose native cache helpers;
an empty override selects the native default. They throw on native failure.
Paths refer to the module filesystem. Browser network/download support and
persistence require separate environment validation; a registry hit does not
prove a model is downloadable or supported by this module. The compiled Node
contract checks argument/status/buffer transport without making downloads.


## Diarization and VAD transcription

`diarizeSegments(audio, rightOrNull, segments, options)` returns owned labeled
segments. `diarizeSegmentsWithTurns(...)` returns `{segments, turns}`. Input
segments use `{t0, t1}` in absolute integer centiseconds and are not modified.
Audio is mono 16 kHz PCM; supply a same-length right channel for stereo methods.
Options are `method` (0–5), `n_threads`, `slice_t0_cs`, `model_path`,
`embedder_path`, `min_speakers`, `max_speakers`, and `num_speakers`. Methods 3/5
use `model_path` (pyannote/Sortformer); method 4 uses `embedder_path` (FoxNose).
Speaker turns share the absolute centisecond timeline. Methods without derived
turns return an empty turns array. Buffer retries are bounded; native failures
and invalid returned counts throw.

`asrTranscribeVad(audio, sampleRate, vadModelPath, language, optionsOrNull)`
uses the ASR singleton and returns the same owned centisecond segments and
word/token metadata as `asrTranscribe`. Empty language selects the native
backend default. `null` options preserve native VAD defaults; an options object
accepts `threshold`, `min_speech_duration_ms`, `min_silence_duration_ms`,
`speech_pad_ms`, `chunk_seconds`, and `n_threads`. Object defaults are
0.5/250/100/30/30 and the module thread default. A zero chunk limit disables
maximum-length splitting. Invalid settings or native transcription failure throw.
Both operations are synchronous and need the same browser scheduling care as
other synchronous inference calls.


`drainStreamedTokenBytes()` returns owned `Uint8Array` fragments, preserving
partial UTF-8 sequences and empty tokens. Assemble the fragments before decoding
text. The wrapper copies every native string before creating JS arrays, so JS
constructor re-entry cannot change later tokens. Negative counts or positive
counts with a null buffer throw. Token/segment drain and count queries require idle compute; the shared guard
rejects them while an async native operation runs. Retained ASR callback
delivery is available through the retained setters described below. Native
segment/token setters replace their corresponding default pollers; clear them
to restore polling.


## Proxied browser dispatch

Threaded `PROXY_TO_PTHREAD` modules expose the async ASR/open/TTS routes.
`browserComputeReady()` reports readiness. Async calls run native compute on
its dedicated pthread, then deliver owned results to the caller's servicer
thread. Only one native operation may be active: synchronous inference,
close/replacement and another async submission raise while compute is busy.
`asrGetProgress()` remains available. Completion callbacks may submit the next
operation. Error completions are asynchronous too.

Native/encoding failures deliver `{error: message}` through the callback.
Check for the `error` property; an exception message may be empty.
A callback exception is contained and stored as `module.crispasrLastCallbackError`;
inspect and clear that property when reporting application errors. The native
delivery is released even when encoding or the callback throws. TTS delivers an
owned `Float32Array`, including empty output, and frees native PCM before JS
encoding. A failed replacement load leaves its singleton closed.

The model-free contract compiles a real proxied pthread module and runs it in
Node. Set `CRISPASR_CHROME` to a Chromium/Chrome executable to run the same checks
in a headless browser served with COOP/COEP headers; the binding CI requires this
browser pass. These checks cover dispatch, load-option transport and TTS-buffer
ownership. Full model accuracy, module size and other hardware paths remain
separate validation gates.


## Retained ASR streams

After opening the ASR singleton, `asrStreamKind()` reports native availability:
0 unavailable, 1 rolling windows, 2 persistent caches, 3 text-prefix streaming.
Only one stream/inference operation per model session is supported. Batch ASR
calls reject while a retained stream is open.

Single-thread modules provide `asrStreamOpen(options)`, returning an opaque
integer ticket. Options are `n_threads` (1), `step_ms` (3000), `length_ms`
(10000), `keep_ms` (200), `language` (automatic when empty), and `translate`
(false). Step must be positive, length at least step, keep nonnegative, and
length plus keep must fit native int32 sample counts at 16 kHz. Translation
support depends on the backend. Pass a nonempty `Float32Array` of 16 kHz mono
PCM to `asrStreamFeed(ticket, pcm)`. `asrStreamFlush(ticket)` processes the tail;
both preserve the native nonnegative status and throw on native failure.
`asrStreamSetLiveDecode(ticket, enabled)` forwards the backend toggle.
`asrStreamGetText(ticket)` returns an owned `{text, t0, t1, counter}` snapshot,
with times in seconds and an exactly representable nonnegative JS counter.
It retries truncated native text rather than silently clipping it.
`asrStreamClose(ticket)` retires a ticket; repeated close is harmless and
other operations on retired tickets fail. Tickets never contain native pointers
and are never reused. Parent close/replacement closes its child first.

Threaded modules require the corresponding `asrStreamOpenAsync`,
`asrStreamFeedAsync`, `asrStreamFlushAsync`, `asrStreamGetTextAsync`,
`asrStreamSetLiveDecodeAsync`, and `asrStreamCloseAsync` methods, with a final
completion callback. These routes require a ready `PROXY_TO_PTHREAD` compute
thread; an ordinary pthread module without it refuses dispatch. Stream open
defaults to four threads. These run native
work and backend cleanup on compute, and deliver owned results on the servicer
using the shared dispatch/error contract above. Async live-decode and close
complete with zero. Use `asrCloseAsync(callback)` or an async ASR open to retire
a parent with a retained child; synchronous parent close/replacement refuses
that case without changing the live parent. Sync stream methods also refuse
in threaded modules. Finish one callback before submitting the next operation.

The tiny Node/browser probes establish transport, scheduling and ownership.
Actual backend/model accuracy, browser memory and decoder responsiveness remain
separate acceptance work in PLAN.md.


## Retained ASR events

`asrSetSegmentCallback(callback)`, `asrSetTokenCallback(callback)`, and
`asrSetProgressCallback(callback)` retain a function on the caller's JS thread.
Pass null or undefined to clear one. Register or clear only while compute is
idle and before opening a retained stream. Parent close/replacement clears all
three native hooks, discards pending events, and restores native polling hooks.
The servicer retires JS callback references on its next pump, before async
completion.
Callback replacement discards pending events for that hook; stale native
userdata generations cannot call a replacement function.

Callbacks receive one owned object:

| Hook | Event |
| --- | --- |
| Segment | `{text, t0, t1, index}`; timestamps are native centiseconds |
| Token | `{rawText, index}`; `rawText` is a copied `Uint8Array` |
| Progress | `{processed, total}`; counts are native input samples |

Token fragments may be incomplete UTF-8 or empty. Concatenate bytes before
text decoding. Native threads only enqueue copied data; JS callbacks never run
on compute or native producer threads. Events are delivered by a 10 ms servicer
pump and drained before an async operation's completion callback. In a sync
single-thread call, delivery follows the native return; `asrDispatchEvents()`
can drain immediately afterwards. This function also works during async
compute because it reads copied events rather than model state.

Event callbacks cannot re-enter wrapper operations, including registration,
close or another event drain. Errors are contained in
`module.crispasrLastCallbackError`, with native event allocations retired even
when JS encoding or a callback throws. Applications may inspect and clear this
property. The native queue is bounded at 4096 events and 8 MiB of text; overflow,
unrepresentable timestamps or allocation failure report an error there and mean
incremental events were lost. The native operation continues. The pump stops
when no callbacks remain.

These setters expose native hooks; they do not make a backend emit unsupported
events. In particular, progress depends on native chunked transcription and a
retained stream backend need not emit session hooks. Verify actual model/backend
behavior separately from the compiled transport and ownership probes.
