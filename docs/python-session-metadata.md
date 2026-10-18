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

Tokens remain empty when the native source does not produce them. Parakeet
session paths retain tokens through default/shared orchestration, legacy
window splicing/deduplication and Japanese slice/gap-fill. Metadata moves with
the selected word's native token block; equal timestamps do not identify token
ownership. The native grouping rules are shared with word construction, so
probabilities and raw byte fragments survive without retokenizing word text.
Empty/BOS tokens before the first word belong to it; skipped tokens between
words and at the end belong to the preceding word. Removing a word removes its
block. Entirely wordless windows still follow the existing merge's empty-output
behavior; no empty segment or invented token is added.

Default non-Japanese shared gap repair also selects tokens by these groups,
preventing duplicate/cross-assigned tokens at equal-frame word boundaries. The
legacy and shared explicit-output splitters also use native word/token ranges:
a word's continuation at the next word's first frame stays with its own segment.
Timestamp fallback is retained for results with no consistent word/token groups;
no-word token-only single results retain their native token bounds. The generic CLI postprocessor's own gap-fill logic
is a separate path, so model-level parity still needs the actual audio checks
in PLAN.md; these packaging tests do not establish long-form speech accuracy.

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
null/bounds checks, cache-release forwarding, and a direct comparison of the
legacy and shared CLI splitters over identical native results. With `CRISPASR_LIB` pointing
to a complete build it also exercises exported getters and real ctypes
signature setup. CTest and GitHub CI run this suite; the linked native ABI
null test also exercises the actual cache release twice without a model.
These are packaging checks, not a new neural accuracy or GPU-memory benchmark.

`python3 tests/test_parakeet_window_metadata.py -v` additionally reproduces and
repairs the former Japanese/legacy empty-token output and shared-repair tied-frame
duplicates. It compiles literal production grouping/window/repair functions and
reads production C ABI getters through the shipped Python result reader. Six
checks cover splice/dedup, two offset Japanese slices with gap recovery, equal
frames, split UTF-8 bytes, empty/BOS ownership and token-less native words.
A 2,000-case comparison against frozen own native grouping preserves exact word
text, times and F32 probabilities. `CRISPASR_LIB` selects actual built-library
getters; full-library CI and real Japanese model validation remain pending for
this follow-up. No public POD layout or Python constructor changed.

The metadata suite also creates an actual native-grouped word whose final
continuation token shares the next segment's first frame. Both shared and
legacy splitters retain that token with the preceding word; all token bytes
still appear exactly once. This case failed on both paths before the follow-up.

Final ownership follow-up source is `ebf69c43721e56a43b7c8f637873902098f523ba`,
frozen on `diag/549-token-boundaries-20261018`: [CI](https://github.com/CrispStrobe/CrispASR/actions/runs/39117921894)
and [lint](https://github.com/CrispStrobe/CrispASR/actions/runs/39117923785) are
queued at dispatch audit. Earlier source's green checks do not validate this
follow-up; preserve exact-source receipts before release.

## Word alternatives

`SessionWord.alternatives` copies native `(text, probability)` candidates before
freeing the result, where the backend and decode options supply them. The list
is empty when absent, including older native libraries. Existing positional
word constructors remain valid. See [the broader API inventory](python-api-coverage.md)
for auxiliary operations and legacy Whisper token/DTW readout.
