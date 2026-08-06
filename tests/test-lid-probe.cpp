// test-lid-probe.cpp — hermetic tests for probe-based LID scoring.
// Fixed strings, no model, no audio.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "core/lid_probe.h"

using namespace core_lid_probe;

TEST_CASE("lid_probe: utf8 length counts codepoints, not bytes", "[unit][lid]") {
    REQUIRE(utf8_length("hello") == 5);
    REQUIRE(utf8_length("مرحبا") == 5);    // Arabic: 2 bytes/char
    REQUIRE(utf8_length("你好世界") == 4); // CJK: 3 bytes/char
    REQUIRE(utf8_length("") == 0);
    // The whole point: the same visible length must not score differently by
    // script. "hello" is 5 bytes, "مرحبا" is 10 — byte length would double it.
    REQUIRE(std::string("مرحبا").size() == 10);
}

TEST_CASE("lid_probe: no-space languages tokenise by character", "[unit][lid]") {
    REQUIRE(is_no_space_language("zh"));
    REQUIRE(is_no_space_language("ja"));
    REQUIRE(is_no_space_language("ko"));
    REQUIRE_FALSE(is_no_space_language("en"));
    REQUIRE_FALSE(is_no_space_language("ar"));

    // Chinese has no spaces: whitespace tokenisation would see ONE token and
    // report perfect diversity for any string at all.
    REQUIRE(diversity("你好你好", "zh") == 0.5); // 4 chars, 2 distinct
    REQUIRE(diversity("你好世界", "zh") == 1.0);
    REQUIRE(diversity("你好你好", "en") == 1.0); // as English: one token, no repetition seen
}

TEST_CASE("lid_probe: diversity flags repetition", "[unit][lid]") {
    REQUIRE(diversity("", "en") == 0.0);
    REQUIRE(diversity("the quick brown fox", "en") == 1.0);
    REQUIRE(diversity("the the the the", "en") == 0.25);
    REQUIRE(diversity("a b a b", "en") == 0.5);
}

TEST_CASE("lid_probe: a repetitive hallucination loses to a shorter clean decode", "[unit][lid]") {
    // The failure mode this scoring exists for: a wrong-language prompt does
    // not error, it produces long fluent-looking repetition. Length alone
    // would pick it; diversity squared must not.
    const std::string hallucinated = "the the the the the the the the the the the the";
    const std::string clean = "and now the weather";

    const double bad = score(hallucinated, "en", 0.0);
    const double good = score(clean, "en", 0.0);
    REQUIRE(good > bad);
}

TEST_CASE("lid_probe: text-LID agreement boosts, up to 4x", "[unit][lid]") {
    const std::string t = "and now the weather";
    const double none = score(t, "en", 0.0);
    const double half = score(t, "en", 0.5);
    const double full = score(t, "en", 1.0);

    REQUIRE(half > none);
    REQUIRE(full > half);
    REQUIRE(full == Catch::Approx(none * 4.0));

    // Out-of-range agreement is clamped, never allowed to run away.
    REQUIRE(score(t, "en", 5.0) == Catch::Approx(full));
    REQUIRE(score(t, "en", -1.0) == Catch::Approx(none));
}

TEST_CASE("lid_probe: an empty probe scores zero", "[unit][lid]") {
    // A candidate the model refuses to transcribe must never win, no matter
    // how confident a text detector is about the empty string.
    REQUIRE(score("", "en", 1.0) == 0.0);
}

TEST_CASE("lid_probe: KNOWN LIMITATION — a fluent translation beats repetitive truth", "[unit][lid]") {
    // Measured, not hypothetical. Forcing a 14-candidate probe on samples/jfk.wav
    // picked 'fr' over 'en'. Two things went wrong at once, and BOTH are
    // properties of the scoring, not of the runtime:
    //
    //  1. Cohere Transcribe TRANSLATES when prompted with a language it was not
    //     given. The 'fr' probe returned real French, which the text LID then
    //     confirmed at 1.00 — so "the output is fluent language X" turns out
    //     not to be evidence that the AUDIO is language X.
    //  2. JFK's line is a chiasmus ("ask not what your country can do for you,
    //     ask what you can do for your country"). The TRUE language is
    //     legitimately repetitive, so diversity^2 penalised the right answer.
    //
    // The numbers below are the ones the probe actually logged on that run
    // (length, agreement, diversity) — not reconstructed strings. This test
    // pins the WRONG answer deliberately: if a future scoring change flips it,
    // that is an improvement, and the test should be inverted together with a
    // fresh measurement on real audio — never edited to match a guess.
    const double fr = score_from(108, 1.00, 0.88); // clean French translation
    const double en = score_from(108, 1.00, 0.73); // correct English, but a chiasmus
    REQUIRE(fr > en);                              // observed: 336 vs 228 → picked 'fr'

    // And why the ≤4-language auto gate is the real safety net: on the actual
    // two-language model the same scoring was right both times, because the
    // wrong-language decode DEGENERATED ("the city of Jerry, a large city of
    // Jerry") instead of translating cleanly.
    REQUIRE(score_from(82, 1.00, 1.00) > score_from(64, 1.00, 0.79));  // ar 328 > en 158, Arabic clip
    REQUIRE(score_from(108, 1.00, 0.73) > score_from(59, 1.00, 0.73)); // en 228 > ar 125, English clip
}

TEST_CASE("lid_probe: agreement can overturn a length advantage", "[unit][lid]") {
    // Same script, both plausible: 'de' output is shorter but the text
    // detector confirms it, while the 'en' output is longer and unconfirmed.
    const double de = score("guten morgen liebe leser", "de", 0.99);
    const double en = score("good morning dear readers today we", "en", 0.0);
    REQUIRE(de > en);
}
