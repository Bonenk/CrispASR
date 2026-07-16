// src/parakeet_orchestrate.h — shared parakeet transcription orchestration
// (improvements Phase 1: collapse the CLI-adapter vs session-C-ABI dual dispatch).
//
// The path-selection + long-audio + segmentation logic that decides HOW to run
// parakeet on a buffer used to be written twice — once in the CLI backend
// adapter (examples/cli/crispasr_backend_parakeet.cpp) and once inline in the
// session C-ABI (src/crispasr_c_api.cpp). That divergence is why issue #257 (and
// the JA-detection fix before it) had to be applied in multiple places.
//
// This header hoists that orchestration into the library so BOTH surfaces call
// one implementation. Callers set sticky per-call state on the context first
// (temperature/beam/att-context/hotwords/ctc via the existing setters), then
// call parakeet_transcribe_segments() and convert the neutral `parakeet_seg`
// list to their own segment type. The routing DECISION is a pure function
// (parakeet_pick_strategy) so it is unit-tested without a model.
#pragma once

#include "parakeet.h"

#include <cstdint>
#include <string>
#include <vector>

// Neutral, surface-agnostic segment (subset both crispasr_segment and
// crispasr_session_seg can be built from).
struct parakeet_seg {
    struct word {
        std::string text;
        int64_t t0 = 0; // centiseconds, absolute
        int64_t t1 = 0;
        float p = 1.0f;
    };
    struct token {
        std::string text;
        int32_t id = -1;
        int64_t t0 = -1;
        int64_t t1 = -1;
        float p = -1.0f;
    };
    std::string text;
    int64_t t0 = 0;
    int64_t t1 = 0;
    std::vector<word> words;
    std::vector<token> tokens;
};

struct parakeet_orchestrate_opts {
    bool chunk_seconds_explicit = false;
    int chunk_seconds = 0;
    float chunk_overlap_seconds = 2.0f;
    bool no_prints = false;
};

// Long-audio / chunking strategy. Pure decision, unit-tested.
enum class parakeet_strategy {
    CHUNK_SEGMENTED, // explicit --chunk-seconds (non-JA): quality encode → N-second segments
    LONGFORM,        // non-JA above the single-pass cap: silence-split single-pass pieces
    SINGLE_PASS,     // fits the single-pass cap: one full-attention pass (+ OOM fallback)
    STREAMED,        // JA / cap disabled: overlapping streamed encoder, one segment
};

// Inputs mirror the adapter/session env-tunable knobs so the choice is
// reproducible and testable. `is_ja` is content-based JA detection (kana/kanji).
struct parakeet_strategy_in {
    int n_samples = 0;
    int sample_rate = 16000;
    bool is_ja = false;
    bool chunk_seconds_explicit = false;
    int chunk_seconds = 0;
    int stream_threshold_s = 300; // single-pass cap (s); 0 = always streamed
    bool longform_enabled = true; // silence-split above the cap
};

// Pure routing decision — no model, no side effects. Mirrors the adapter:
//   - non-JA + explicit --chunk-seconds>0            → CHUNK_SEGMENTED
//   - longform on + threshold>0 + n > threshold       → LONGFORM
//   - threshold>0 + n <= threshold                    → SINGLE_PASS
//   - else                                            → STREAMED
inline parakeet_strategy parakeet_pick_strategy(const parakeet_strategy_in& in) {
    if (!in.is_ja && in.chunk_seconds_explicit && in.chunk_seconds > 0)
        return parakeet_strategy::CHUNK_SEGMENTED;
    const long long cap = (long long)in.stream_threshold_s * in.sample_rate;
    if (in.longform_enabled && in.stream_threshold_s > 0 && (long long)in.n_samples > cap)
        return parakeet_strategy::LONGFORM;
    if (in.stream_threshold_s > 0 && (long long)in.n_samples <= cap)
        return parakeet_strategy::SINGLE_PASS;
    return parakeet_strategy::STREAMED;
}

// Full orchestration: mel → path selection → decode → segmentation, returning
// the neutral segment list. `is_ja` is passed in (callers already detect it, or
// pass parakeet_vocab_is_japanese(ctx)). Reads the same CRISPASR_PARAKEET_*
// env knobs the adapter did, so behaviour is byte-identical to the pre-hoist
// adapter path.
std::vector<parakeet_seg> parakeet_transcribe_segments(struct parakeet_context* ctx, const float* samples,
                                                       int n_samples, int64_t t_offset_cs, bool is_ja,
                                                       const parakeet_orchestrate_opts& opts);
