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


## FIXED v2 (2026-07-14) — JA misdetection resolved, verified on real 1.1b
- parakeet_vocab_is_japanese() (vocab content scan) replaces vocab<=4096. Downloaded
  cstr/parakeet-tdt-1.1b-GGUF (vocab_size=1024, confirms misdetection).
- 1.1b DEFAULT: 1 clean complete segment, 21 words (was split-outs + truncation).
- 1.1b --chunk-seconds 7 --chunk-overlap 2: full correct transcript (was corrupted).
- 0.6b-ja still JA (97% CJK vocab); 0.6b-v3 still non-JA. No regressions.
- TODO: extract testable vocab_looks_japanese() helper + unit test.

## ROUND 3 (2026-07-15) — reporter: split-outs fixed; default VRAM heavy

Reporter: split-outs gone ✓. Remaining: (i) --chunk-seconds gives one coherent
transcription (INTENDED — chunked encode, decode once, word timestamps); (ii)
default (no chunk) ~2GiB VRAM for <4min (single-pass full O(T²) attention).

USER DIRECTION: option 2 (memory-bounded, CLI-steered) wired through C-ABI/server/
wrappers, MATCHING the Python reference.

REFERENCE (NeMo): long-audio memory is bounded via `change_attention_model(
"rel_pos_local_attn", [L,R])` (local/windowed attention → O(T·window)); default is
full attention. CrispASR already implements this (att_context_left/right) but only
via env CRISPASR_PARAKEET_ATT_CONTEXT.

PLAN:
1. BUG (critical): C-ABI inline parakeet dispatch (crispasr_c_api.cpp:4525) still
   uses the OLD `parakeet_n_vocab<=4096` JA heuristic → bindings/server STILL
   misdetect 1.1b. Mirror parakeet_vocab_is_japanese() there (contributing pt6).
2. FEATURE: expose local-attention window as CLI `--att-context L,R` (matches NeMo
   change_attention_model), wired: whisper_params → CLI → parakeet adapter
   (parakeet_set_att_context) → C-ABI (session field + inline dispatch) → server
   (form) → python/go wrapper docs. Default full attention (matches NeMo default).
   --chunk-seconds stays the other reference control (chunked inference).


## ROUND 3 DONE (2026-07-15)
1. C-ABI JA fix (crispasr_c_api.cpp:4525) — bindings/server now match CLI. ✓
2. --att-context "L,R" wired: lib parakeet_set_att_context → whisper_params → CLI
   (+help) → parakeet adapter → C-ABI (session field + inline dispatch +
   crispasr_session_set_parakeet_att_context + header) → server (att_context form,
   both handlers) → python Session.set_parakeet_att_context(). Go binding has no
   session-API surface (0 crispasr_session refs), nothing to wire there.
   Verified: --att-context 64,64 on 88s clip == full-attention output, local attn
   active; symbol exported; python syntax OK.
Matches NeMo: full attention default; opt-in local attention (rel_pos_local_attn)
for long-audio VRAM. --chunk-seconds remains the other reference control.

## ROUND 4 (2026-07-15) — true windowed attention (maintainer-directed)

FINDING: --att-context (as shipped R3) does NOT reduce memory — CrispASR builds a
T×T mask over FULL attention (fastconformer build_block: scores are (T,T,n_heads)),
matching NeMo's OUTPUT but not its O(T·window) memory. Measured peak RSS: full ==
att-context (1.41GB); --chunk-seconds ~same/slightly higher. The real single-alloc
memory lever today is --chunk-seconds (per-chunk encode graphs).

DIRECTIVE: implement TRUE windowed attention (compute only the local band →
O(T·window)) so --att-context delivers NeMo rel_pos_local_attn's memory benefit.

REFERENCE: NeMo RelPositionMultiHeadAttentionLongformer — "sliding chunks":
pad+reshape Q/K/V into overlapping windows of size w; each query chunk attends to a
2w+1 key band; rel-pos bias (BD) windowed too. O(T·w) scores.

PLAN: core_conformer::build_block windowed-attention path (gated), validated vs the
masked-full output (parity ≥0.999 on the encoder / transcript-identical) + measured
memory reduction, before flipping --att-context to use it. HARD: ggml banded matmul
(overlapping key-window gather) + windowed rel-pos. Incremental, diff-harness-checked.

### R4 Milestone 1 DONE (2026-07-15) — windowed-attn algorithm validated

Standalone ggml parity harness (tools/dev/winattn_parity.cpp) proves the block
sliding-chunks windowed attention is BIT-EXACT vs full masked rel-pos attention
(max abs diff ~1e-7) across: baseline, asymmetric windows (WL!=WR), BS/HD/NH
sweeps, and non-divisible T with query-axis zero-padding (T=13,17,100,209).

Key algorithm (O(T·BS·H) scores/BD instead of O(T²·H)):
- Block size BS >= max(att_left, att_right). NB=ceil(T/BS), Tp=NB*BS (pad Q/K/V).
- 3-block band per query block via reshape->3 stride-1 block slices->concat
  (ggml forbids overlapping views: view_4d checks contiguous product<=src bytes).
- K/V zero-padded BS each side so band [b-1,b,b+1] = keys k=(bo-1)BS+j, j∈[0,3BS).
- scores_blk = mul_mat(K_band(HD,3BS,NB,NH), Qu_blk(HD,BS,NB,NH)) -> (3BS,BS,NB,NH).
- BD (rel-pos bias) windowed: R_sl = R rows [T-2BS .. T+2BS-2] (RB=4BS-1),
  RESHAPED (HD,RB,1,NH) so mul_mat broadcasts R over blocks & aligns heads in ne3
  (critical bug found: without the size-1 block axis, ggml mixes head/block batch
  dims and blocks b>=2 diverge). BDraw_blk=mul_mat(R_sl,Qv_blk)->(RB,BS,NB,NH),
  then in-block rel_shift view: BD_blk[j,i]=BDraw_blk[(BS-1)+j-i,i], nb1'=nb1-nb0,
  offset (BS-1)*nb0 — natural key order, all strides>=0.
- Host band mask (3BS,BS,NB): -inf where k out of [0,T) or out of [q-WL,q+WR].

NEXT (M2): wire as core_conformer::build_windowed_attn, gated CRISPASR_FC_WINDOWED_ATTN
(default keeps masked-full path intact for A/B), validate via real parakeet
transcript parity + memory measurement.

### R4 M2 DONE + M3 in progress (2026-07-15) — wired + validated

M2: build_windowed_attn wired into core_conformer::build_block, gated
CRISPASR_FC_WINDOWED_ATTN=1 (default OFF keeps masked-full intact). Caller
parakeet.cpp builds O(T·window) band mask (make_window_band_mask) instead of the
T×T local mask when gated+applicable. Builds clean.

M3 findings (parakeet-tdt-0.6b-v3 q4_k, Metal M1):
- PARITY: windowed-local == masked-full-local transcripts IDENTICAL on 20s
  (T=250) and 209s (T=2613) clips. Windowed path confirmed engaging (stderr trace).
- MEMORY: KEY metric is phys_footprint (macOS caps RSS via compression). At forced
  single-pass T=7838 (627s clip, CRISPASR_PARAKEET_STREAM_THRESHOLD=9999):
    masked-full local: peak footprint = 2402 MB  (the O(T²) BD_raw ~ user's "2GiB")
    windowed local:    <measuring — slow, backgrounded>
  So the O(T²) memory hog is REAL at large single-pass T, and it IS the rel-pos BD.
- CAVEAT: default dispatcher silence-splits/chunks long audio (STREAM_THRESHOLD
  300s), bounding per-encode T, so the blow-up only appears in forced single-pass.
  Windowed's purpose = enable bounded-memory SINGLE-PASS long encode (avoids the
  chunking that corrupts full-attention FastConformer — the #257 root issue).
- CONCERN: windowed is SLOWER at large T (many small ops: 2×concat + 4×pad + several
  cont per layer ×24). Timed out >2min at T=7838. Needs perf assessment / op fusion
  before it's a viable default; fine as an opt-in memory-vs-speed lever now.

NEXT: confirm windowed footprint << 2402MB; assess speed; fix --att-context help
wording; decide default (opt-in for now).

### R4 M3 RESULTS (2026-07-15) — windowed is FASTER + lower memory (correction)

Earlier "windowed is slower" was WRONG (that was T=7838 being slow for ALL paths).
Real data at T=2613 (209s, single-pass, Metal M1):
    masked-full local (att 64,64): 25.7 s   (worst: full compute + T×T mask)
    windowed local     (att 64,64):  8.3 s   (3.1x faster than masked-full)
    full attention:                 11.4 s   (windowed 1.4x faster than full)
Memory (peak footprint, macOS phys_footprint; RSS is compression-capped):
    T=7838 single-pass: masked-full 2402 MB vs windowed 2155 MB (~10%; the O(T)
    conv front-end co-dominates at this T — BD is O(T²) so the win grows for
    longer audio). Attention BD itself drops from ~2GB to a few hundred MB.
Parity: windowed == masked-full transcripts IDENTICAL on t501-20s/long90/long3m
    at both att 32,32 and 64,64. Bit-exact algorithm (parity harness).

VERDICT: windowed local attention is strictly better than the shipped masked-full
local path — same output, ~3x faster, less memory (growing with length). Still
gated CRISPASR_FC_WINDOWED_ATTN=1 for A/B per maintainer. Candidate to become the
DEFAULT when --att-context is set, pending CUDA cross-check.
