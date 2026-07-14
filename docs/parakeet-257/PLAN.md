# parakeet-tdt: word list + chunking fixes (issue #257)

## NOW — active work (2026-07-14)

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

### Next
1. (A) done.
2. (B) find why the chunk-merge drops parakeet tails; fix; verify jfk chunked == baseline.
3. Unit test(s) for the word-grouping helper + a chunk-merge regression.
4. Build, run unit tests, merge to main, comment #257.
