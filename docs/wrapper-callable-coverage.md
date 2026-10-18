# Callable wrapper audit

This ledger records reviewed public entry points and their operation probes.
It is incremental: unmapped functions remain pending in PLAN.md. A native
symbol declaration, regex match or generated stub is not a callable API.
The census scope is the 297 unique `crispasr_*` functions in the two shared
headers, plus the separate 14 chat exports. It does not inventory every
upstream `whisper_*` declaration in `include/crispasr.h`.

## TitaNet

Native functions: `crispasr_titanet_init`, `crispasr_titanet_embed`,
`crispasr_titanet_cosine_sim`, `crispasr_titanet_free`.
Embedding succeeds with exactly **192 floats written**, not zero or one.
Similarity is the native dot product for normalized model embeddings.

| Wrapper | Public entry points | Operation proof |
| --- | --- | --- |
| Python | `TitaNet(...)`, `.embed`, `.cosine_sim`, `.close` | `test_python_legacy_pointer.py` |
| C# | `TitaNet.Open`, `.Embed`, `.CosineSim`, `.Dispose` | `CSharpTitaNetContract` |
| Java | `CrispasrAuxiliary.TitaNet`, `.embed`, `.cosineSimilarity`, `.close` | `JavaAuxWrapperContract` |
| Go | `TitaNetInit`, `TitaNetContext.Embed`, `TitaNetCosineSim`, `.Close` | `TestProbeTitaNetFloatCountAndClosedContext` |
| Rust | `TitaNet::new`, `.embed`, `::cosine_sim`, `Drop` | `crispasr/tests/probe.rs` |
| Dart | `CrispasrTitaNet`, `.embed`, `.cosineSim`, `.close` | `ffi_behavior_test.dart` |
| Ruby | `Session.titanet_init/embed/cosine_sim/free` | `ruby-contract-program.rb` |
| WASM | `TitaNet`, `.embed`, `titanetCosineSimilarity`, `.close`/`.delete` | `WasmWrapperContract` |

The shared compiled fixture checks success, zero/short/oversized native counts
and resource lifetime. That proves transport, not real speaker accuracy.

## Closed-roster speaker database

Native functions: `crispasr_speaker_db_open`, `_count`, `_match`, `_enroll2`,
`_free` (all with the `crispasr_speaker_db` prefix). The legacy `_load` and
`_enroll` exports are retired refusing stubs; they are not implementation
alternatives. Open and enrollment preserve the native roster/attestation
contract. A negative match score means **no match**, including at a negative
threshold. Enrollment through `_enroll2` writes a profile to disk; reopen an
existing database to refresh its retained roster.

| Wrapper | Public entry points | Operation proof |
| --- | --- | --- |
| Python | `SpeakerDB(...)`, `.count`, `.match`, `.enroll`, `.close` | `test_python_legacy_pointer.py` |
| C# | `SpeakerDb.Open`, `.Count`, `.Match`, `SpeakerDb.Enroll`, `.Dispose` | `CSharpSpeakerDbContract` |
| Java | `CrispasrAuxiliary.SpeakerDb`, `.count`, `.match`, `SpeakerDb.enroll`, `.close` | `JavaAuxWrapperContract` |
| Go | `SpeakerDBOpen`, `.Count`, `.Match`, `.Enroll`, `.Close` | `TestProbeSpeakerDatabaseRosterScoresEnrollmentAndLifetime` |
| Rust | `SpeakerDB::open`, `.count`, `.match_embedding`, `.enroll`, `Drop` | `probe_speaker_database_roster_scores_enrollment_and_lifetime` |
| Dart | `CrispasrSpeakerDB`, `.count`, `.match`, `.enroll`, `.close` | `ffi_behavior_test.dart` |
| Ruby | `Session.speaker_db_open/count/match/enroll/free` | `ruby-contract-program.rb` |
| WASM | `SpeakerDB`, `.count`, `.match`, `.enroll`, `.close`/`.delete` | `WasmWrapperContract` |

Dart optionally uses the newer handle-enrollment export when present; its
existing return-code handling remains compatible. Rust's old `load` method
now explicitly directs callers to `open`, rather than invoking a retired
native function. The operation fixture checks actual UTF-8 result copies,
missing consent/roster, native failure codes, no-match semantics and lifetime.

## Generic speaker primitives: Ruby and WASM follow-up

Native `crispasr_speaker_embedder_*_abi` has make/free/dim/name/embed operations;
embedding succeeds with **1**. `crispasr_speaker_cluster_abi` returns the number
of clusters, with negative failure. Pyannote cache compute/free/apply uses
zero-success apply and native centisecond segment/offset fields.

| Wrapper | Public operations | Operation proof |
| --- | --- | --- |
| Ruby | `CrispASR::SpeakerEmbedder.new`, `.dimension`, `.name`, `.embed`, `.close`; `CrispASR.speaker_cluster(rows, threshold, maximum)`; `CrispASR::PyannoteCache.new`, `.apply(segments, offset_cs)`, `.close` | `RubyWrapperContract`, native shared fixture; TypedData GC and recursive coercion/GC checks |
| WASM | `SpeakerEmbedder`, `.dimension`, `.name`, `.embed`, `.close`/`.delete`; `speakerCluster(rows, threshold, maximum)`; `PyannoteCache`, `.apply(segments, offsetCs)`, `.close`/`.delete` | `WasmWrapperContract`, same shared fixture compiled into WASM |

Clustering returns labels plus `n_clusters` (Ruby) / `nClusters` (WASM).
Cache apply returns owned `{t0,t1,speaker}` records and leaves input records
unchanged. Explicit close is idempotent; subsequent operations raise. Other
languages' existing generic primitives still need their ledger rows reviewed.

## Go/Rust options, text and polling follow-up

| Native family | Go public operation | Rust public operation | Operation proof |
| --- | --- | --- | --- |
| Session open with params | `SessionOpenWithOptions`, `DefaultSessionOptions` | `Session::open_with_options`, `SessionOptions::default` | Literal runtime 48-byte struct, custom/default values and zero reserved fields |
| Backend capabilities | `BackendCapabilities`, `ListBackendCapabilities` | Existing `Session::backend_caps` / `list_backends_with_caps` | New Go growth/unknown-backend probe; existing Rust methods still need their ledger proof row |
| Truecase / PCS init/process/free text/free | `TruecaseInit`, `PCSInit`, promoted `.Process`/`.Close` | `TruecaseModel::open`, `PCSModel::open`, `.process`, Drop | Prefix-specific allocation/free oracle, empty text and failures |
| Candidate scores | Existing `Session.ScoreTexts` | `Session::score_texts` | Actual candidate/option transport, owned scores/counts and positive native error |
| Owned streamed segment result | `DrainStreamedSegments` | `drain_streamed_segments` | Shared metadata reader, copied raw bytes/alternatives/speaker and result free |
| Progress/count/reset polling | Existing `GetProgress`, `ResetProgress`, `GetStreamed*Count`, `ResetStreamed*` | `get_progress`, `reset_progress`, `get_streamed_*_count`, `reset_streamed_*` | Serialized global polling fixture |
| Borrowed token polling | `DrainStreamedTokens` with documented caller serialization | **unsafe** `drain_streamed_token_bytes` / `drain_streamed_tokens` | Protected-page raw fragments/empty tokens and malformed counts |

Global polling selects the most recently streaming session. An owned segment
result can be copied after the call; the borrowed token buffer cannot be used
concurrently with another drain/inference/close. Rust makes that requirement
an explicit unsafe contract. These checks do not prove concurrent model behavior.

## Ruby session lifecycle and v2 loader

`Session.open`, `.open_explicit` and `.open_with_options` return opaque integer
tickets; `Session.close` closes child `Session::Stream` handles first. Stream
open/feed/text/flush/live-decode/close paths validate ownership and reject use
after parent close. The new loader accepts all five v2 configuration controls,
with runtime ABI version/reserved fields generated internally.

The 155 guarded Session/Stream/Chat operations preserve arities and aliases and
hold per-context borrows across coercion/callback/thread switches. This count
is a guard synchronization fact, not a native export coverage percentage.
Direct context, DB and text handles also use opaque tickets. Tests force native
address reuse and still reject old closed tickets; native callbacks continue to
use actual native pointers internally.

`RubyWrapperContract` validates this through concrete stream and loader probes,
including native close ordering and callback userdata. Remaining mic and legacy
Whisper ownership paths remain separate tasks in PLAN.md.

## Additional reviewed paths

- C# `Session.TranscribeVad(pcm, rate, model, language, options)` and Java
  `CrispasrSession.transcribeVad(pcm, rate, model, language, options)` expose
  all six VAD ABI fields. The three-argument calls retain their original
  default/native route. Probes compile the literal native option layout and
  verify each field, language, input checks and callback-error cleanup.
- Java `translateText`, `enhanceAudioRnnoise` and
  `TextProcessor(model, TextProcessorKind.PUNCTUATION)` are public operations.
  Allocated UTF-8 output remains a pointer until copied and freed. Empty
  output and native failure are distinct; RNNoise unavailability raises.
- Go `DrainStreamedTokens` reads exactly the native count of NUL-terminated
  strings. It preserves raw fragments and empty tokens. A protected-page
  fixture reproduced the previous guessed-byte-length read crashing before
  the repair. Polling still selects the most recently streaming native
  session; callers must serialize polling with its inference and close.
- Dart `diarizeSegments` uses target-width FFI structs and an arena freed on
  every exit. Its probe verifies all option fields, centisecond conversion,
  turn-buffer retry, invalid native counts and stereo input bounds. Java's
  corresponding pointer offsets now use `Native.POINTER_SIZE`; a 32-bit JVM
  execution remains a platform validation task.

Real-model output, browser scheduling, full WASM size and native build tests
remain separate acceptance gates. See PLAN.md for the remaining audit rows.


## WASM load options and registry/cache

`asrOpenWithOptions`/`ttsOpenWithOptions` call
`crispasr_session_open_with_params` using all five v2 controls and zero reserved
padding. The probe compiles the literal runtime POD. Invalid options preserve
the live singleton; a native load failure after valid replacement leaves it closed.
Single-thread modules enforce one thread. Threaded/browser scheduling remains
an independent pending gate.

`registryLookup`, `registryLookupByFilename`, `registryBackends`,
`registryDefaultBundle`, `cacheDirectory` and `cacheEnsureFile` cover the seven
registry/cache ABI exports. Bundle info and artifact enumeration are combined
into one owned result. The compiled Node contract checks string transport,
unknown versus empty bundles, all three artifact kinds, license metadata,
cache arguments, native errors, buffer growth, retry ceilings and unterminated
native output. This is a model-free transport test, not browser download proof.


## WASM generic diarization and language-aware VAD

`diarizeSegments` and `diarizeSegmentsWithTurns` cover the two generic diarization
ABI exports. The compiled fixture uses literal runtime segment/turn/options
structs on wasm32. It checks field transport, stereo channels, absolute times,
owned output, empty turns, negative/oversized counts, native failures and bounded
buffer growth. Model methods retain their distinct path fields.

`asrTranscribeVad` covers `crispasr_session_transcribe_vad_lang`, a superset of
the no-language export. The compiled fixture verifies all six literal VAD
option fields, language, native-default null options, invalid-input rejection,
result ownership and metadata. These contracts do not establish model accuracy
or threaded browser scheduling.


WASM's `sessionTranscribeWithLogits` uses a shared native-to-owned transcription
helper. The operation probe checks negative segment/word/alternative/token
counts, negative/overflowing logit dimensions, missing buffers, copied grids,
native-result freeing and logit-flag reset on C++ and JS encoding errors.


## Ruby microphone ownership (separate microphone header)

`Whisper::CrispASR::Mic.open/start/stop/close/default_device_name` wrap the five
microphone exports declared in `src/crispasr_mic.h`, outside the shared 311-export
census. Handles are opaque tickets. The compiled native-thread fixture checks
PCM copies, callback-close, repeated/concurrent close, foreign/retired tickets,
start/stop/open failures, queue overflow, GC, callback exceptions, pump kill and
an injected copy exception after dequeue. A test-only atomic allocation counter
checks queued/pending chunks return to zero. This does not prove audio-device
access on a target platform.
