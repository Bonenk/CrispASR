// test-compliance-wiring.cpp — the EU AI Act gates are WIRED UP on every
// surface that can reach them.
//
// The pure predicates are guarded elsewhere: test-voice-clone-policy.cpp knows
// what a clone is, test-marking-policy.cpp knows what a container-less surface
// must do. Both were green throughout every failure this file exists for,
// because none of those failures were in the predicate. They were in the
// plumbing — a surface that never called the gate, a baker that never stamped
// the pack the gate reads, an upload endpoint nobody had added a gate to.
//
// The pattern, four times now:
//
//   #172  Wyoming shipped for four releases marking nothing at all — it was
//         never added to the surface list the compliance work walked.
//   9a91e4 chatterbox clones only through a baked .gguf, so a `.wav` suffix
//         test could never see the headline cloning backend.
//   a66fc2 --make-ref built a reusable voiceprint with no attestation asked
//         anywhere, because it returned before the TTS block's gate.
//   (here) cosyvoice3 keeps every voice inside one bundle and selects by name,
//         so --voice named no file, no metadata was read, and a zero-shot
//         voice clone scored as a preset on CLI, server, Wyoming and ABI.
//
// So this is a SOURCE-level guard, deliberately, and it tests the joins rather
// than the logic. The behavioural version needs models and sockets, which puts
// it in the live tier CI does not run — and a compliance gate nobody can run on
// the tier CI actually runs is a gate that ships wrong (#312's lesson).
//
// It is coarse by design: it cannot prove the gate is called correctly, only
// that the call is still there. That is exactly the failure mode above.

#include <catch2/catch_test_macros.hpp>

#include <fstream>
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

} // namespace

// ---------------------------------------------------------------------------
// Voice BANKS reach the clone gate on every surface.
//
// cosyvoice3's voices live in one voices.gguf, discovered as a sibling of the
// model, and --voice selects an entry by name. Only the backend knows which
// bundle it resolved, so it has to hand the path to the gate; without that the
// gate sees an unresolvable bare name and returns "preset" for a voice clone.
// ---------------------------------------------------------------------------

TEST_CASE("the backend base class offers a voice-bank hook", "[unit][compliance]") {
    const std::string src = read_file("examples/cli/crispasr_backend.h");
    REQUIRE(contains(src, "virtual std::string voice_bank_path() const"));
}

TEST_CASE("cosyvoice3 hands its voice bundle to the gate", "[unit][compliance]") {
    const std::string src = read_file("examples/cli/crispasr_backend_cosyvoice3.cpp");
    // Overrides the hook...
    REQUIRE(contains(src, "std::string voice_bank_path() const override"));
    // ...and actually records the bundle it resolved, rather than returning a
    // member nothing ever assigns. That would compile, pass a naive test, and
    // leave the gate exactly as blind as before.
    REQUIRE(contains(src, "voices_path_ = voices_path;"));
}

TEST_CASE("every synthesis surface passes the bank to classify_voice", "[unit][compliance]") {
    // Four surfaces, one gate. The Wyoming lesson is that a surface inherits no
    // compliance behaviour — it has to be wired up, and only a check like this
    // will say whether it was.
    SECTION("CLI") {
        REQUIRE(contains(read_file("examples/cli/crispasr_run.cpp"), "backend->voice_bank_path()"));
    }
    SECTION("HTTP server") {
        REQUIRE(contains(read_file("examples/cli/crispasr_server.cpp"), "backend->voice_bank_path()"));
    }
    SECTION("Wyoming") {
        REQUIRE(contains(read_file("examples/cli/wyoming.cpp"), "g_backend->voice_bank_path()"));
    }
    SECTION("C ABI") {
        // No CrispasrBackend here — the session resolves the bundle itself.
        const std::string src = read_file("src/crispasr_c_api.cpp");
        REQUIRE(contains(src, "s->cosyvoice3_voices_path = std::move(cv3_voices);"));
        REQUIRE(contains(src, "s->cosyvoice3_voices_path)"));
    }
}

// ---------------------------------------------------------------------------
// Every producer that consumes a recording gates and stamps.
//
// Baking IS the cloning step: everything downstream just replays the pack. A
// baker that skips the stamp produces a clone the runtime reads as a preset —
// no consent asked, no [CONSENT] line, no Art. 50(4) audible disclosure.
// ---------------------------------------------------------------------------

TEST_CASE("the cosyvoice3 bank baker gates and stamps", "[unit][compliance]") {
    const std::string src = read_file("models/convert-cosyvoice3-voices-to-gguf.py");
    // Assert the REFUSAL, not the flag. `--i-have-rights` appears in the
    // argparse help whether or not anything checks it, and `raise SystemExit(`
    // appears in any script that validates anything — a guard built from those
    // two stayed green with the gate gutted, which is how this test earned its
    // own red-proof.
    REQUIRE(contains(src, "if not args.i_have_rights:"));
    REQUIRE(contains(src, "baking a CosyVoice3 voice bundle requires --i-have-rights."));
    // Per-entry stamp: a bundle can hold a preset and a clone at once, and a
    // bank-wide flag would have to gate both or free both.
    REQUIRE(contains(src, "crispasr.voice.{name}.cloned_from_recording"));
    // The sentinel that makes an ABSENT per-voice key mean "preset" rather than
    // "baked before the stamp existed". Without it every entry falls back to
    // the producer architecture and re-gates the presets.
    REQUIRE(contains(src, "crispasr.voice.bank_stamped"));
}

TEST_CASE("the kugelaudio voice baker gates the --audio path only", "[unit][compliance]") {
    const std::string src = read_file("models/convert-kugelaudio-voice-to-gguf.py");
    REQUIRE(contains(src, "if not args.i_have_rights:"));
    REQUIRE(contains(src, "encoding a voice from --audio requires --i-have-rights."));
    REQUIRE(contains(src, "crispasr.voice.cloned_from_recording"));
    // --voice-pt converts an upstream pre-encoded voice: no recording of a
    // natural person, so no attestation and no stamp. That dual mode is also
    // why `kugelaudio-voice` must NOT go on the architecture list — it cannot
    // tell the two apart, and the stamp is the only predicate that can.
    REQUIRE(contains(src, "cloned_from_recording=True"));
    REQUIRE_FALSE(contains(read_file("examples/cli/crispasr_voice_clone_policy.h"), "arch == \"kugelaudio-voice\""));
}

TEST_CASE("the recording-derived producer list matches the bakers", "[unit][compliance]") {
    // Classification by producer, for packs baked before the stamp existed.
    // The header claims "every recording-derived producer in this repo is
    // enumerated above" — that claim was false for a year, which is what let
    // cosyvoice3 bundles through. Pin it.
    const std::string policy = read_file("examples/cli/crispasr_voice_clone_policy.h");
    REQUIRE(contains(policy, "\"chatterbox-voice\""));
    REQUIRE(contains(policy, "\"qwen3tts.voicepack\""));
    REQUIRE(contains(policy, "\"cosyvoice3-voices\""));
}

// ---------------------------------------------------------------------------
// Voice ENROLLMENT over the network is gated where the recording enters.
// ---------------------------------------------------------------------------

TEST_CASE("uploading a voice reference requires a consent attestation", "[unit][compliance]") {
    const std::string src = read_file("examples/cli/crispasr_server.cpp");
    // POST /v1/voices stores an arbitrary uploaded recording as a reusable
    // voiceprint — the network equivalent of --make-ref, which demands
    // --i-have-rights. It accepted uploads from anyone who could reach the
    // endpoint and logged only a byte count.
    REQUIRE(contains(src, "scope=voice-upload"));
    REQUIRE(contains(src, "uploading a voice reference requires a 'consent_attestation' form field"));
}

// ---------------------------------------------------------------------------
// The bindings' watermark helper marks at a DETECTABLE strength.
//
// This is the call the docs point synthesize_raw() callers at: opting out of
// automatic marking makes marking the result their duty, and this discharges
// it. Three bindings defaulted it to 0.005 — the strength the C ABI itself
// documents as "too faint to reliably detect on real speech, so it did not
// robustly satisfy EU AI Act Art. 50 'detectable' marking". The ABI fixed that
// by treating alpha <= 0 as the robust default; an explicit positive 0.005
// walks straight past the fallback. Marking you cannot detect is not marking.
// ---------------------------------------------------------------------------

TEST_CASE("bindings request the robust default watermark strength", "[unit][compliance]") {
    SECTION("python") {
        const std::string src = read_file("python/crispasr/_binding.py");
        REQUIRE(contains(src, "def watermark_embed(pcm: \"numpy.ndarray\", alpha: float = -1.0)"));
        REQUIRE_FALSE(contains(src, "alpha: float = 0.005"));
    }
    SECTION("dart / flutter") {
        const std::string src = read_file("flutter/crispasr/lib/src/crispasr.dart");
        REQUIRE(contains(src, "{double alpha = -1.0,"));
        REQUIRE_FALSE(contains(src, "double alpha = 0.005"));
    }
    SECTION("go") {
        const std::string src = read_file("bindings/go/crispasr_session.go");
        REQUIRE(contains(src, "C.int(len(pcm)), -1.0)"));
        REQUIRE_FALSE(contains(src, "C.int(len(pcm)), 0.005)"));
    }
}

TEST_CASE("every binding that can opt out of marking can also mark", "[unit][compliance]") {
    // Rust, Ruby, Java and C# exposed synthesize_raw (UNMARKED audio, behind an
    // attestation) while exposing no watermark call at all — so the documented
    // remediation, "mark the result yourself", was unreachable from the binding
    // that handed you the unmarked buffer.
    SECTION("rust") {
        REQUIRE(contains(read_file("crispasr/src/lib.rs"), "pub fn watermark_embed(pcm: &mut [f32])"));
    }
    SECTION("ruby") {
        REQUIRE(contains(read_file("bindings/ruby/ext/ruby_crispasr_session.c"),
                         "\"watermark_embed\", rb_watermark_embed"));
    }
    SECTION("java") {
        REQUIRE(contains(read_file("bindings/java/src/main/java/io/github/ggerganov/whispercpp/CrispasrSession.java"),
                         "public static void watermarkEmbed(float[] pcm)"));
    }
    SECTION("csharp") {
        REQUIRE(contains(read_file("bindings/csharp/CrispASR/Session.cs"),
                         "public static void WatermarkEmbed(float[] pcm)"));
    }
}

// ---------------------------------------------------------------------------
// Synthetic TEXT surfaces disclose (Art. 50(1)) and carry marking metadata
// (Art. 50(2)).
//
// CrispASR marks audio; nothing marks short-form text, and the docs said so —
// for ONE of the four surfaces that generate it. The chat ABI, the installed
// crispasr-chat binary and the Flutter chat binding were all unlisted, and the
// Flutter one is exactly where §6.3's "a CLI is obvious to a reasonably
// well-informed person" stops being true.
// ---------------------------------------------------------------------------

TEST_CASE("the chat ABI publishes a canonical AI disclosure", "[unit][compliance]") {
    const std::string hdr = read_file("include/crispasr_chat.h");
    REQUIRE(contains(hdr, "crispasr_chat_ai_disclosure_text(void)"));
    // The header is where an integrator learns the duty exists. It carried no
    // AI Act text at all, unlike crispasr.h and crispasr_session.h.
    REQUIRE(contains(hdr, "Art. 50(1)"));
    REQUIRE(contains(hdr, "Art. 50(2)"));

    const std::string impl = read_file("src/chat.cpp");
    REQUIRE(contains(impl, "crispasr_chat_ai_disclosure_text(void)"));
    // Pinned wording: four surfaces read this string, and a drift between them
    // is a disclosure that differs depending on which one you called.
    REQUIRE(contains(impl, "You are interacting with an AI system. "
                           "Responses are generated by artificial intelligence."));
}

TEST_CASE("the chat CLI discloses at startup", "[unit][compliance]") {
    const std::string src = read_file("examples/cli/crispasr_chat_main.cpp");
    REQUIRE(contains(src, "crispasr_chat_ai_disclosure_text()"));
}

TEST_CASE("the chat endpoint marks its responses as AI-generated", "[unit][compliance]") {
    // No watermark-equivalent survives a copy-paste of a sentence, so the
    // Commission's guidance points at metadata travelling with the response.
    // For an HTTP API that is a header, and it must be set on the buffered and
    // the SSE branch alike — set_header after a body write is too late.
    const std::string src = read_file("examples/cli/crispasr_server.cpp");
    REQUIRE(contains(src, "X-Crispasr-Ai-Generated"));
    REQUIRE(contains(src, "X-Crispasr-Ai-Disclosure"));
}

TEST_CASE("the flutter chat binding surfaces the disclosure", "[unit][compliance]") {
    const std::string src = read_file("flutter/crispasr/lib/src/chat.dart");
    REQUIRE(contains(src, "aiDisclosureText"));
    REQUIRE(contains(src, "crispasr_chat_ai_disclosure_text"));
}
