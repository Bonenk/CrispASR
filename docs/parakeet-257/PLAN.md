# parakeet-tdt: word list + chunking fixes (issue #257)

## DONE (2026-07-14) — both fixes on main; issue closeable

Branch `fix/parakeet-257`. Reporter (AppleSheeple) on parakeet-tdt-1.1b:
(A) `-ojf` JSON has text + tokens but **no words list**; (B) `--chunk-seconds 7
--chunk-overlap 2` cuts transcript tails (first cut > overlap) and emits weird
mid-sentence split segments. Not seen with cohere/granite.

### Reproduced (parakeet-tdt-0.6b-v3-q4_k, samples/jfk.wav)
- **(B) CONFIRMED, serious.** Baseline (no chunk): "And so, my fellow Americans,
  ask not what your country can do for you, ask what you can do for your country."
  With `--chunk-seconds 7 --chunk-overlap 2`:
    "And so, my fellow Americans,"                                  ← tail CUT ("ask not..." gone)
    "what your country can do for you, ask what you can do for yourself."  ← boundary loss + "your country"→"yourself"
    "you can do for your country."                                  ← duplicated split fragment
- **(A) not repro'd on 0.6b-v3** (22 words emitted, with & without --no-punctuation).
  Likely 1.1b-specific OR the streamed/chunked path.

### Root causes (code)
- **(A)** `parakeet_decode_frames` (parakeet.cpp:3431) sets `r->words=nullptr;
  r->n_words=0` with a FALSE comment ("the backend adapter builds words" — the
  adapter `result_to_segment` only *copies* r->words). Both `parakeet_transcribe_
  streamed` (3605) and `parakeet_transcribe_chunked` (3605-ish return) delegate to
  decode_frames → **no words on streamed/chunked paths**. transcribe_ex (single-
  pass) builds words inline (3966), so short single-file gets words. → FIX: extract
  the word-grouping into a shared helper and call it in decode_frames too.
  (Also: transcribe_chunked builds a full `r` at 3568-3591 then throws it away by
  `return parakeet_decode_frames(...)` → double-decode + word loss. Clean that up.)
- **(B)** `--chunk-seconds` forces the DISPATCHER chunk+merge (bypasses parakeet's
  internal long-audio handling). The per-chunk parakeet decode is fine; the
  MERGE/LCS-dedup drops boundary content for parakeet (works for cohere/granite).
  → INVESTIGATING the dispatcher chunk+merge + LCS dedup (crispasr_c_api
  session_transcribe_chunked path).

### Progress
- (A) DONE: extracted `parakeet_group_words()` helper, called from decode_frames +
  transcribe_ex. Streamed path (STREAM_THRESHOLD=0) now emits 22 words on jfk (was 0).

### (B) root cause + fix
- `--chunk-seconds` forces the DISPATCHER per-slice transcribe + overlap-save
  trim + LCS merge. parakeet is a full-attention FastConformer (`CAP_UNBOUNDED_INPUT
  + CAP_INTERNAL_CHUNKING`): short context-extended slices decode DEGRADED
  ("your country"→"yourself") and the trim drops boundary words ("ask not" lost).
- cohere/granite lack `CAP_INTERNAL_CHUNKING` → dispatcher chunking is correct for
  them (matches user: "not seen with cohere/granite").
- parakeet's INTERNAL streaming at 7s/2s (STREAM_THRESHOLD=0 STREAM_CHUNK=7) gives
  the CORRECT full transcript. → FIX: for CAP_INTERNAL_CHUNKING backends, when
  --chunk-seconds is explicit, DON'T dispatcher-slice; pass the whole audio and let
  the backend chunk internally. parakeet adapter honors params.chunk_seconds/overlap
  by routing to parakeet_transcribe_streamed.

### DONE
- (A) parakeet_group_words() shared helper → words on streamed/chunked paths.
- (B) backend_self_chunks_on_explicit() gate: CAP_INTERNAL_CHUNKING + explicit
  --chunk-seconds bypasses dispatcher slicing (crispasr_run.cpp); parakeet adapter
  routes to parakeet_transcribe_streamed(chunk,overlap). jfk --chunk-seconds 7
  --chunk-overlap 2 now == baseline ("...ask not what your country can do for you.
  Ask what you can do for your country.", 22 words). 4 new unit tests in
  test-issue-114-chunk-context-gate. Full unit suite green (12 'failures' = unbuilt
  metal/vad binaries in the targeted build, not regressions).
2. (B) find why the chunk-merge drops parakeet tails; fix; verify jfk chunked == baseline.
3. Unit test(s) for the word-grouping helper + a chunk-merge regression.
4. Build, run unit tests, merge to main, comment #257.

## REOPENED (2026-07-14) — reporter feedback: partial fix

Reporter (AppleSheeple) on parakeet-tdt-1.1b: (A) words now present ✓ but weird
single-word split-outs appear; (B) chunked still cuts tails + splits. Key detail:
tokens array is FULL, but the SEGMENT offsets.to is cut and text/words are filtered
to end before it.

ROOT CAUSE: `is_ja_model_ = (parakeet_n_vocab <= 4096)` (adapter:75) MISDETECTS
parakeet-tdt-1.1b (English, vocab ~1024) as Japanese. JA-ness is vocab CONTENT not
size: 0.6b-ja vocab_size=3073 is ~97% CJK/kana; v3 vocab_size=8192 is 0% CJK; 1.1b
English small vocab is ~0% CJK. Misdetected-JA → CAP_INTERNAL_CHUNKING off, (B) fix
gated off (!is_ja), JA 8-12s small-chunk path used on an English full-attention
model → split-outs (default) + dispatcher-slice corruption (chunked). Same bug in
the lib's `vocab_size < 4000` heuristics (parakeet.cpp:3654,3774).

FIX: detect JA by scanning the vocab for Japanese script (kana/kanji fraction),
not vocab size. Verify: 0.6b-ja stays JA, 0.6b-v3 non-JA, 1.1b-EN non-JA. Download
cstr/parakeet-tdt-1.1b-GGUF to reproduce the reporter's exact case.
