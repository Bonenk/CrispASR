// test-parakeet-strategy.cpp — unit tests for parakeet_pick_strategy, the pure
// long-audio routing decision hoisted into the shared orchestration
// (improvements Phase 1). Pinning it here dedups the decision that used to be
// written twice (CLI adapter + session C-ABI) and was the source of the JA /
// #257 routing bugs.

#include <catch2/catch_test_macros.hpp>

#include "parakeet_orchestrate.h"

namespace {
constexpr int SR = 16000;
parakeet_strategy_in mk(int secs, bool is_ja, bool chunk_explicit, int chunk_s, int thr, bool longform) {
    parakeet_strategy_in in;
    in.n_samples = secs * SR;
    in.sample_rate = SR;
    in.is_ja = is_ja;
    in.chunk_seconds_explicit = chunk_explicit;
    in.chunk_seconds = chunk_s;
    in.stream_threshold_s = thr;
    in.longform_enabled = longform;
    return in;
}
} // namespace

TEST_CASE("non-JA explicit --chunk-seconds → CHUNK_SEGMENTED at any length",
          "[unit][parakeet-strategy][improvements]") {
    REQUIRE(parakeet_pick_strategy(mk(20, false, true, 7, 300, true)) == parakeet_strategy::CHUNK_SEGMENTED);
    REQUIRE(parakeet_pick_strategy(mk(600, false, true, 7, 300, true)) == parakeet_strategy::CHUNK_SEGMENTED);
}

TEST_CASE("non-JA short (<= cap) → SINGLE_PASS", "[unit][parakeet-strategy][improvements]") {
    REQUIRE(parakeet_pick_strategy(mk(11, false, false, 0, 300, true)) == parakeet_strategy::SINGLE_PASS);
    REQUIRE(parakeet_pick_strategy(mk(225, false, false, 0, 300, true)) == parakeet_strategy::SINGLE_PASS);
    REQUIRE(parakeet_pick_strategy(mk(300, false, false, 0, 300, true)) == parakeet_strategy::SINGLE_PASS);
}

TEST_CASE("non-JA long (> cap) + longform → LONGFORM", "[unit][parakeet-strategy][improvements]") {
    REQUIRE(parakeet_pick_strategy(mk(301, false, false, 0, 300, true)) == parakeet_strategy::LONGFORM);
}

TEST_CASE("non-JA long + longform disabled → STREAMED", "[unit][parakeet-strategy][improvements]") {
    REQUIRE(parakeet_pick_strategy(mk(600, false, false, 0, 300, false)) == parakeet_strategy::STREAMED);
}

TEST_CASE("threshold 0 (always-streamed) → STREAMED", "[unit][parakeet-strategy][improvements]") {
    REQUIRE(parakeet_pick_strategy(mk(11, false, false, 0, 0, false)) == parakeet_strategy::STREAMED);
    REQUIRE(parakeet_pick_strategy(mk(600, false, false, 0, 0, false)) == parakeet_strategy::STREAMED);
}

TEST_CASE("JA never takes the CHUNK_SEGMENTED branch", "[unit][parakeet-strategy][improvements]") {
    // JA caller passes threshold=12, longform=false. Explicit chunk is ignored
    // (is_ja gate) → falls to the length logic.
    REQUIRE(parakeet_pick_strategy(mk(8, true, true, 7, 12, false)) == parakeet_strategy::SINGLE_PASS);
    REQUIRE(parakeet_pick_strategy(mk(30, true, true, 7, 12, false)) == parakeet_strategy::STREAMED);
}
