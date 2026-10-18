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
