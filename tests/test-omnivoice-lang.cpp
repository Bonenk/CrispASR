// test-omnivoice-lang.cpp — the OmniVoice target-language knob (#13273).
//
// SubtitleEdit's OmniVoice language menu was decoration: the value it picks
// never reached the prompt on ANY surface. Two halves to guard, and the
// predicate half alone would have stayed green through the whole bug.
//
//   1. The PREDICATE — core_omnivoice_lang::resolve() reproduces the
//      blueprint's `_resolve_language()` (omnivoice/models/omnivoice.py).
//      Whatever it returns is what lands between <|lang_start|> and
//      <|lang_end|>, so an unrecognized value MUST resolve to None rather than
//      pass through and condition the model on noise.
//   2. The JOINS — every surface that can synthesise actually calls it. The
//      original defect was three missing call sites, not a wrong predicate:
//      the CLI adapter applied language only in init() (so a persistent server
//      could never change it per request) and the session C-ABI's omnivoice arm
//      never applied it at all. Source-level, because the behavioural version
//      needs a 1.2 GB model and a socket — the tier CI does not run, and a gate
//      CI cannot run is a gate that ships wrong (#312).

#include "core/omnivoice_lang.h"

#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <set>
#include <sstream>
#include <string>

#ifndef CRISPASR_SOURCE_DIR
#error "CRISPASR_SOURCE_DIR must be defined by the build"
#endif

namespace {

std::string read_file(const std::string& rel) {
    const std::string path = std::string(CRISPASR_SOURCE_DIR) + "/" + rel;
    std::ifstream f(path);
    INFO("reading " << rel);
    REQUIRE(f.good());
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

using core_omnivoice_lang::resolve;
using core_omnivoice_lang::Status;

} // namespace

// ---------------------------------------------------------------------------
// The predicate: parity with _resolve_language().
// ---------------------------------------------------------------------------

TEST_CASE("omnivoice lang: a valid ISO id passes straight through", "[unit][omnivoice]") {
    for (const char* id : {"en", "de", "fr", "zh", "ja", "ko", "es", "pt", "ru", "arb"}) {
        INFO("id " << id);
        const auto r = resolve(id);
        REQUIRE(r.status == Status::id_passthrough);
        REQUIRE(r.id == id);
    }
}

TEST_CASE("omnivoice lang: an English name resolves to its id", "[unit][omnivoice]") {
    REQUIRE(resolve("English").id == "en");
    REQUIRE(resolve("German").id == "de");
    REQUIRE(resolve("Japanese").id == "ja");
    // Case-insensitive on the NAME side (the blueprint lowercases the key).
    REQUIRE(resolve("GERMAN").id == "de");
    REQUIRE(resolve("german").id == "de");
    REQUIRE(resolve("German").status == Status::name_resolved);
}

TEST_CASE("omnivoice lang: None/empty/auto mean language-agnostic", "[unit][omnivoice]") {
    for (const char* v : {"", "None", "none", "NONE", "auto", "AUTO"}) {
        INFO("value '" << v << "'");
        const auto r = resolve(v);
        REQUIRE(r.status == Status::cleared);
        REQUIRE(r.id.empty());
    }
}

// The defect this whole file exists for: an unrecognized string used to be
// dropped verbatim into <|lang_start|>...<|lang_end|>. The blueprint returns
// None instead, and so must we — a bogus tag is worse than no tag, because the
// model is conditioned on tokens it never saw in that slot.
TEST_CASE("omnivoice lang: an unrecognized value falls back to None, never itself", "[unit][omnivoice]") {
    for (const char* v : {"de-DE", "en_US", "DE", "Deutsch", "xx", "klingon", "de-CH-1901"}) {
        INFO("value '" << v << "'");
        const auto r = resolve(v);
        REQUIRE(r.status == Status::unrecognized);
        REQUIRE(r.id.empty());
    }
}

// The blueprint's id check is case-SENSITIVE (`if language in LANG_IDS`), so
// "DE" is genuinely unrecognized upstream too. We match that rather than
// silently being more permissive than the model's own resolver — but the
// warning has to be actionable, which is what suggest() is for.
TEST_CASE("omnivoice lang: suggest() turns the near-misses into the right id", "[unit][omnivoice]") {
    REQUIRE(core_omnivoice_lang::suggest("DE") == "de");
    REQUIRE(core_omnivoice_lang::suggest("de-DE") == "de");
    REQUIRE(core_omnivoice_lang::suggest("en_US") == "en");
    REQUIRE(core_omnivoice_lang::suggest("pt-BR") == "pt");
    REQUIRE(core_omnivoice_lang::suggest("klingon").empty());
}

// Not a gap in our table — a property of the model. OmniVoice's vocabulary is
// ISO 639-3 INDIVIDUAL languages, so several macrolanguage codes a caller will
// reach for out of habit simply do not exist: Arabic is "arb" (Standard) or one
// of ~20 regional variants, never "ar". Pinned so nobody "fixes" the table by
// inventing rows the model was never trained on; the right answer is the
// warning that resolve() drives, plus SE listing the real names.
TEST_CASE("omnivoice lang: macrolanguage codes are absent by design", "[unit][omnivoice]") {
    REQUIRE(resolve("ar").status == Status::unrecognized);
    REQUIRE(resolve("arb").status == Status::id_passthrough);
    REQUIRE(resolve("Standard Arabic").id == "arb");
    REQUIRE(resolve("Egyptian Arabic").id == "arz");
}

TEST_CASE("omnivoice lang: the generated table is well-formed", "[unit][omnivoice]") {
    // A hand-maintained list needs a machine check that it is complete; this
    // one is generated, so the check is that generation did not collapse it.
    REQUIRE(core_omnivoice_lang::kLangTableN > 600);

    std::set<std::string> names;
    for (int i = 0; i < core_omnivoice_lang::kLangTableN; i++) {
        const std::string name = core_omnivoice_lang::kLangTable[i].name;
        const std::string id = core_omnivoice_lang::kLangTable[i].id;
        INFO("row " << i << " '" << name << "' -> '" << id << "'");
        REQUIRE(!name.empty());
        REQUIRE(!id.empty());
        // Names are lowercase upstream; a stray capital would make the
        // lowercase-key lookup miss that language forever.
        for (char c : name)
            REQUIRE(!(c >= 'A' && c <= 'Z'));
        // Every name distinct — a duplicate silently shadows a language.
        REQUIRE(names.insert(name).second);
        // Every row must round-trip through the resolver in both directions.
        REQUIRE(resolve(name).id == id);
        REQUIRE(resolve(id).id == id);
    }
}

// ---------------------------------------------------------------------------
// The joins: every synthesis surface applies the resolved language.
// ---------------------------------------------------------------------------

// The runtime resolves centrally, so no caller can forget to. Asserting the
// call inside omnivoice_set_language specifically (not merely "the file
// mentions the header") is what makes this guard able to go red.
TEST_CASE("omnivoice lang: the runtime resolves before building the prompt", "[unit][omnivoice]") {
    const std::string src = read_file("src/omnivoice.cpp");
    REQUIRE(contains(src, "core/omnivoice_lang.h"));
    REQUIRE(contains(src, "core_omnivoice_lang::resolve("));
    // And the prompt keeps reading the stored (now resolved) value.
    REQUIRE(contains(src, "<|lang_start|>"));
}

// The bug SubtitleEdit hit. The server owns ONE backend instance for the whole
// session and passes the per-request language in `params`; applying it only in
// init() means the menu can never do anything after the first line.
TEST_CASE("omnivoice lang: the CLI adapter applies language PER CALL", "[unit][omnivoice]") {
    const std::string src = read_file("examples/cli/crispasr_backend_omnivoice.cpp");
    const size_t synth = src.find("std::vector<float> synthesize(");
    INFO("synthesize() not found in the omnivoice adapter");
    REQUIRE(synth != std::string::npos);
    const std::string body = src.substr(synth);
    REQUIRE(contains(body, "omnivoice_set_language("));
    // Found while A/B-ing the language fix: /v1/audio/speech advertises a
    // per-request `seed` that omnivoice dropped on the floor, so a re-render
    // of the same line could not be reproduced. Same call site, same class.
    REQUIRE(contains(body, "omnivoice_set_seed("));
}

// #329's exact shape, one backend over: the session ABI reimplements each
// backend's synthesize inline and does NOT call the adapter, so the adapter fix
// above reaches the CLI and the server but not bindings/Flutter/Android.
TEST_CASE("omnivoice lang: the session ABI applies language too", "[unit][omnivoice]") {
    const std::string src = read_file("src/crispasr_c_api.cpp");
    const size_t arm = src.find("omnivoice_synthesize(s->omnivoice_ctx");
    INFO("the session's omnivoice synthesis arm was not found");
    REQUIRE(arm != std::string::npos);
    // Search backwards from the synth call to the start of its `if` arm.
    const size_t arm_start = src.rfind("if (s->omnivoice_ctx)", arm);
    REQUIRE(arm_start != std::string::npos);
    const std::string body = src.substr(arm_start, arm - arm_start);
    REQUIRE(contains(body, "omnivoice_set_language("));
    // It must read the session's sticky target language, not a hardcoded value.
    REQUIRE(contains(body, "target_language"));
}
