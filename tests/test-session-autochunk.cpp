// test-session-autochunk.cpp — unit tests for the session long-audio
// auto-chunk applicability decision (fix/session-long-audio).

#include <catch2/catch_test_macros.hpp>

#include "session_autochunk.h"

using core_session::session_autochunk_applicable;

namespace {
constexpr int SR = 16000;
// enabled, backend, secs, chunk_s, return_logits, already_chunking
bool ac(bool en, const char* b, int secs, int chunk_s, bool logits, bool already) {
    return session_autochunk_applicable(en, b, secs * SR, SR, chunk_s, logits, already);
}
} // namespace

TEST_CASE("auto-chunk fires for long audio on a chunk-needing backend", "[unit][session-autochunk]") {
    REQUIRE(ac(true, "moonshine", 60, 30, false, false));
    REQUIRE(ac(true, "whisper", 45, 30, false, false));
}

TEST_CASE("short audio (<= window) is not chunked", "[unit][session-autochunk]") {
    REQUIRE_FALSE(ac(true, "moonshine", 20, 30, false, false));
    REQUIRE_FALSE(ac(true, "moonshine", 30, 30, false, false)); // exactly at the window
}

TEST_CASE("self-chunking backends are skipped", "[unit][session-autochunk]") {
    REQUIRE_FALSE(ac(true, "parakeet", 300, 30, false, false));
    REQUIRE_FALSE(ac(true, "reazonspeech", 300, 30, false, false));
}

TEST_CASE("disabled gate / logits / explicit-chunk skip", "[unit][session-autochunk]") {
    REQUIRE_FALSE(ac(false, "moonshine", 300, 30, false, false)); // gate off
    REQUIRE_FALSE(ac(true, "moonshine", 300, 30, true, false));   // return_logits
    REQUIRE_FALSE(ac(true, "moonshine", 300, 30, false, true));   // already chunking
}

TEST_CASE("custom window respected", "[unit][session-autochunk]") {
    REQUIRE_FALSE(ac(true, "moonshine", 40, 60, false, false)); // 40s <= 60s window
    REQUIRE(ac(true, "moonshine", 70, 60, false, false));       // 70s > 60s window
}

TEST_CASE("degenerate inputs are safe", "[unit][session-autochunk]") {
    REQUIRE_FALSE(session_autochunk_applicable(true, "moonshine", 999 * SR, 0, 30, false, false)); // sr=0
    REQUIRE_FALSE(session_autochunk_applicable(true, "moonshine", 999 * SR, SR, 0, false, false)); // chunk=0
}
