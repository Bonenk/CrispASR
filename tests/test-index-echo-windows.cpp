#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "core/index_echo_windows.h"

using core_index_echo::windows;
using Catch::Approx;

TEST_CASE("Index-Echo speech windows preserve padding and source offsets", "[unit][index-echo]") {
    auto out = windows({{1, 3}, {4, 6}}, 8);
    REQUIRE(out.size() == 1);
    REQUIRE(out[0].start == Approx(0.7));
    REQUIRE(out[0].end == Approx(6.5));
    REQUIRE_FALSE(out[0].hard_cut);
}
TEST_CASE("Index-Echo long speech cuts cannot be merged across the hard boundary", "[unit][index-echo]") {
    auto out = windows({{0, 125}}, 126);
    REQUIRE(out.size() == 3);
    REQUIRE(out[0].start == 0); REQUIRE(out[0].end == 60);
    REQUIRE(out[1].start == 60); REQUIRE(out[1].end == 120);
    REQUIRE(out[2].start == 120); REQUIRE(out[2].end == Approx(125.5));
    REQUIRE(out[0].hard_cut); REQUIRE(out[1].hard_cut); REQUIRE_FALSE(out[2].hard_cut);
}
TEST_CASE("Index-Echo short final window folds only when the combined span fits", "[unit][index-echo]") {
    auto out = windows({{0, 55}, {58, 59.8}}, 60);
    REQUIRE(out.size() == 1); REQUIRE(out[0].end == 60);
    auto separate = windows({{0, 55}, {62, 63}}, 64);
    REQUIRE(separate.size() == 2);
    REQUIRE(separate[1].start == Approx(61.7));
}
TEST_CASE("Index-Echo silence has no windows", "[unit][index-echo]") {
    REQUIRE(windows({}, 120).empty());
}
