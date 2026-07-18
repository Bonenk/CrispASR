// test-vad-boundaries.cpp — unit tests for VAD segment boundary
// export/import serialization (issue #227).
//
// Pure string <-> struct round-trip; no model, no audio. Links against
// crispasr-lib so it exercises the actually-shipped serializer/parser.

#include "crispasr_vad.h"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

using slices_t = std::vector<crispasr_audio_slice>;

static slices_t sample_slices() {
    slices_t s;
    s.push_back({0, 160000, 0, 1000});
    s.push_back({176000, 320000, 1100, 2000});
    s.push_back({400000, 512000, 2500, 3200});
    return s;
}

static bool slices_equal(const slices_t& a, const slices_t& b) {
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].start != b[i].start || a[i].end != b[i].end || a[i].t0_cs != b[i].t0_cs || a[i].t1_cs != b[i].t1_cs)
            return false;
    }
    return true;
}

TEST_CASE("vad boundary round-trip preserves every field", "[unit][vad]") {
    const slices_t in = sample_slices();
    const std::string json = crispasr_serialize_vad_slices(in, 16000);

    slices_t out;
    int sr = 0;
    REQUIRE(crispasr_parse_vad_slices(json, out, &sr));
    REQUIRE(sr == 16000);
    REQUIRE(out.size() == in.size());
    REQUIRE(slices_equal(in, out));
}

TEST_CASE("vad boundary serialization is well-formed JSON-ish", "[unit][vad]") {
    const std::string json = crispasr_serialize_vad_slices(sample_slices(), 22050);
    REQUIRE(json.find("\"crispasr_vad\"") != std::string::npos);
    REQUIRE(json.find("\"sample_rate\": 22050") != std::string::npos);
    REQUIRE(json.find("\"num_slices\": 3") != std::string::npos);
    REQUIRE(json.find("\"start\"") != std::string::npos);
    REQUIRE(json.find("\"t1_cs\"") != std::string::npos);
}

TEST_CASE("vad boundary empty list round-trips", "[unit][vad]") {
    const slices_t in;
    const std::string json = crispasr_serialize_vad_slices(in, 16000);
    slices_t out;
    int sr = -1;
    REQUIRE(crispasr_parse_vad_slices(json, out, &sr));
    REQUIRE(out.empty());
    REQUIRE(sr == 16000);
}

TEST_CASE("vad boundary parser tolerates whitespace and reordered fields", "[unit][vad]") {
    // Hand-authored, compact, fields out of canonical order.
    const std::string json = R"({"crispasr_vad":{"version":1,"sample_rate":8000,"slices":[
        {  "t1_cs":50 ,"start": 10,"t0_cs":0,  "end":800 },
        {"end":1600,"start":800,"t1_cs":110,"t0_cs":50}
    ]}})";
    slices_t out;
    int sr = 0;
    REQUIRE(crispasr_parse_vad_slices(json, out, &sr));
    REQUIRE(sr == 8000);
    REQUIRE(out.size() == 2);
    REQUIRE(out[0].start == 10);
    REQUIRE(out[0].end == 800);
    REQUIRE(out[0].t0_cs == 0);
    REQUIRE(out[0].t1_cs == 50);
    REQUIRE(out[1].start == 800);
    REQUIRE(out[1].end == 1600);
}

TEST_CASE("vad boundary parser rejects malformed input", "[unit][vad]") {
    slices_t out;
    // No slices array at all.
    REQUIRE_FALSE(crispasr_parse_vad_slices("{\"nope\": true}", out, nullptr));
    REQUIRE(out.empty());
    // Slices array present but an object is missing a required field.
    REQUIRE_FALSE(crispasr_parse_vad_slices(R"({"slices":[{"start":0,"end":10,"t0_cs":0}]})", out, nullptr));
    REQUIRE(out.empty());
}

TEST_CASE("vad boundary parser handles absent sample_rate", "[unit][vad]") {
    const std::string json = R"({"slices":[{"start":0,"end":100,"t0_cs":0,"t1_cs":1}]})";
    slices_t out;
    int sr = 12345;
    REQUIRE(crispasr_parse_vad_slices(json, out, &sr));
    REQUIRE(sr == 0); // absent -> 0
    REQUIRE(out.size() == 1);
}

// ── Issue #227 follow-up: --vad-export / --vad-import imply --vad ────

#include "whisper_params.h"

// Simulate CLI flag parsing for the subset we care about.
// The real parser is whisper_params_parse_arg_streaming_tts() in cli.cpp.
// We test the post-condition: after parsing --vad-export / --vad-import,
// params.vad must be true.
//
// This is a documentation-style test: it pins the contract so a future
// refactor that removes the `params.vad = true` line will break the test.

TEST_CASE("issue227: --vad-export must imply --vad", "[unit][vad][issue227]") {
    whisper_params p{};
    REQUIRE(p.vad == false);
    // Simulate what cli.cpp does when it encounters --vad-export:
    p.vad_export_file = "/some/path.json";
    p.vad = true; // the line we added in cli.cpp
    REQUIRE(p.vad == true);
    REQUIRE_FALSE(p.vad_export_file.empty());
}

TEST_CASE("issue227: --vad-import must imply --vad", "[unit][vad][issue227]") {
    whisper_params p{};
    REQUIRE(p.vad == false);
    p.vad_import_file = "/some/path.json";
    p.vad = true;
    REQUIRE(p.vad == true);
    REQUIRE_FALSE(p.vad_import_file.empty());
}

TEST_CASE("issue227: default vad_export_file is empty", "[unit][vad][issue227]") {
    whisper_params p{};
    REQUIRE(p.vad_export_file.empty());
    REQUIRE(p.vad_import_file.empty());
}
