// src/session_autochunk.h — pure applicability decision for the session
// long-audio auto-chunker (fix/session-long-audio).
//
// The raw session transcribe is a single pass; for short-segment models this
// degrades past ~30 s while the CLI dispatcher chunks. `transcribe_autochunk`
// slices long audio and transcribes each piece. This header holds only the pure
// "should we auto-chunk this call?" decision so it can be unit-tested without a
// model.
#pragma once

#include <string>

namespace core_session {

// True iff the session should slice this buffer into chunks before transcribing.
//   enabled          — CRISPASR_SESSION_AUTOCHUNK (default on)
//   backend          — session backend id
//   n_samples, sr    — buffer length
//   chunk_seconds    — window (CRISPASR_SESSION_CHUNK_SECONDS, default 30)
//   return_logits    — session opted into per-frame CTC logits (can't merge)
//   already_chunking — an explicit chunked request is in flight (force_chunk>=0)
//
// Skips backends that already chunk internally (parakeet/reazonspeech
// self-chunk in transcribe_single), the logits path, an explicit chunk request,
// and audio at/under the window.
inline bool session_autochunk_applicable(bool enabled, const std::string& backend, int n_samples, int sr,
                                         int chunk_seconds, bool return_logits, bool already_chunking) {
    if (!enabled || return_logits || already_chunking)
        return false;
    if (backend == "parakeet" || backend == "reazonspeech")
        return false; // self-chunk in transcribe_single
    if (sr <= 0 || chunk_seconds <= 0)
        return false;
    return (long long)n_samples > (long long)chunk_seconds * sr;
}

} // namespace core_session
