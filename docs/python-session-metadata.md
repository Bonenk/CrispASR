# Python session metadata (#549)

`Session.transcribe`, `transcribe_chunked`, `transcribe_vad`, and
`transcribe_with_logits` return the same segment metadata. Each
`SessionSegment.tokens` is a list of `SessionToken` values with `text`, `start`,
`end`, `confidence`, and `raw_text`. Times are seconds; negative native timing
sentinels become `None`. Token confidence is the backend probability, or `-1`
when unavailable. Word confidence retains its existing fallback of `1.0`.

| Backend | Segment output | Word/token timing |
| --- | --- | --- |
| Cohere | Transcript and native token stream | Existing runtime estimates, matching CLI token metadata; not forced alignment |
| Parakeet | Shared CLI/session orchestration with words and tokens | Backend TDT timings; available tokens are retained |
| MOSS-Transcribe-Diarize | Native parsed timestamp/speaker segments | Native segment times; no word/token timings exposed by this backend |
| VibeVoice | Parsed native utterances and speaker labels | Native utterance times; word/token times unavailable |

Tokens remain empty when a backend or route does not produce them. In
particular, legacy Parakeet word-only merge and Japanese gap-fill routes do not
provide a token stream. Default non-Japanese Parakeet orchestration retains
the shared CLI token stream, including explicit chunked calls.

Speaker labels use the CLI form `(Speaker N) `, stored separately from text.
Speaker numbers are local to a decode/chunk; matching numbers across calls do
not establish speaker identity. MOSS parsing uses the same parser as the native
segment API. Plain output that contains no parseable MOSS segments retains the
text fallback, rather than discarding the transcript.

## Missing timestamps and byte fragments

```python
for segment in session.transcribe(pcm_f32):
    for word in segment.words:
        if word.has_timestamps:
            print(word.start, word.end, word.text)
    token_bytes = b"".join(token.raw_text for token in segment.tokens)
```

`SessionWord.start/end` can now be `None`, replacing the old misleading `-0.01`
value for unavailable times. Existing valid numeric times are unchanged.
Chunk offsetting and VAD remapping preserve missing timing sentinels. No
replacement word timestamps are interpolated by the Python binding.

Byte-level BPE tokens can split UTF-8 characters. `SessionToken.text` decodes
individual fragments with replacement for display; concatenate `raw_text`
before decoding to reconstruct the token bytes losslessly. Assembled segment
and word text retain their existing UTF-8 finalization.

## Forced alignment and cache lifetime

```python
from crispasr import align_words, free_aligner_cache

try:
    words = align_words("aligner.gguf", transcript, pcm_f32)
finally:
    free_aligner_cache()
# Or session.free_aligner_cache() for the library already loaded by a session.
```

The cache belongs to the loaded native library, shared by its callers. Release
waits for any active alignment and is safe to repeat. The next alignment
reloads the requested model; session transcription contexts remain usable.
When selecting a custom native library, pass the same `lib_path` to alignment
and module-level cache release. Backend/device allocators may retain their
own allocations, so this does not promise a particular VRAM counter drop.

## Compatibility and validation

The C ABI adds `crispasr_session_result_n_tokens`, `token_text`, `token_t0`,
`token_t1`, `token_p` (each with the `crispasr_session_result_` prefix), and
`crispasr_free_aligner_cache`. Results stay opaque. Getter text pointers remain
owned by the result; copy before freeing it. Native times are centiseconds.
Token getters return empty text, `-1` times/probability, and zero count for
invalid indices or null results.

The Python wrapper accepts older libraries: token lists are empty when the
accessors are absent, while explicit cache release raises `RuntimeError` when
unsupported. Existing positional `SessionSegment` arguments retain their order;
`tokens` is appended with an independent empty-list default.

`python3 tests/test_session_metadata.py -v` compiles production packaging and
getter functions against decoded fixtures. It covers Cohere/Parakeet token
preservation, MOSS native segments/fallback, missing times, byte fragments,
chunk/VAD timing transforms, all four Python entry points, older libraries,
null/bounds checks, and cache-release forwarding. With `CRISPASR_LIB` pointing
to a complete build it also exercises exported getters and real ctypes
signature setup. CTest and GitHub CI run this suite; the linked native ABI
null test also exercises the actual cache release twice without a model.
These are packaging checks, not a new neural accuracy or GPU-memory benchmark.
