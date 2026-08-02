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
// Whose voice a PRESET voice is reaches the disclosure decision, on every
// surface.
//
// `is_clone == false` used to mean both "no consent owed" (right) and "nothing
// to disclose" (wrong). A preset shipped inside a model can be an identifiable
// person, and Art. 3(60) does not care which pipeline produced the audio. The
// mechanism is guarded in test-speaker-identity.cpp; this is the plumbing.
// ---------------------------------------------------------------------------

TEST_CASE("the backend base class resolves identity per checkpoint", "[unit][compliance]") {
    const std::string src = read_file("examples/cli/crispasr_backend.h");
    // Takes the model path, because one backend serves several checkpoints with
    // different answers — `orpheus` runs both Canopy's base model and
    // Kartoffel's German fine-tune. A no-argument hook cannot tell them apart
    // and would have to pick one verdict for both.
    REQUIRE(contains(src, "virtual crispasr_voice::SpeakerIdentity declared_speaker_identity(const std::string& "
                          "model_path) const"));
    REQUIRE(contains(src, "crispasr_voice::identity_for_model(name(), model_path)"));
}

TEST_CASE("every surface passes the actual checkpoint to the lookup", "[unit][compliance]") {
    // Passing an empty string here would compile, resolve every model to
    // Unknown, and look exactly like "nothing is researched yet" — a fix that
    // ships inert. Pin the argument.
    REQUIRE(contains(read_file("examples/cli/crispasr_run.cpp"), "declared_speaker_identity(params.model)"));
    REQUIRE(contains(read_file("examples/cli/crispasr_server.cpp"), "declared_speaker_identity(params.model)"));
    REQUIRE(contains(read_file("examples/cli/wyoming.cpp"), "declared_speaker_identity(g_params.model)"));
}

TEST_CASE("the model stamp reaches the resolver on every surface", "[unit][compliance]") {
    // The durable half of the identity answer. A reader nothing consults is the
    // inert-fix failure mode (#324), so pin the call at each join.
    REQUIRE(contains(read_file("examples/cli/crispasr_run.cpp"),
                     "crispasr_voice::read_model_speaker_identity(params.model)"));
    REQUIRE(contains(read_file("examples/cli/crispasr_server.cpp"),
                     "crispasr_voice::read_model_speaker_identity(params.model)"));
    REQUIRE(
        contains(read_file("examples/cli/wyoming.cpp"), "crispasr_voice::read_model_speaker_identity(g_params.model)"));
}

TEST_CASE("a converter can write the stamp the runtime reads", "[unit][compliance]") {
    // A read path with no writer is a feature nobody can use. The key spelling
    // has to match on both sides or the whole thing fails open, silently.
    const std::string conv = read_file("models/convert-orpheus-to-gguf.py");
    REQUIRE(contains(conv, "--speaker-identity"));
    // The WRITE CALL, not the key name: the flag's own help text quotes the key
    // too, so matching the bare string stayed green with the add_string()
    // deleted. Fourth time the red-proof has caught a guard of mine matching
    // prose instead of code — the pattern is "assert the token that only exists
    // when the behaviour does".
    REQUIRE(contains(conv, "w.add_string(\"crispasr.voice.speaker_identity\", args.speaker_identity)"));
    // Never written as a guess: an omitted or "unknown" value writes no key.
    REQUIRE(contains(conv, "if args.speaker_identity and args.speaker_identity != \"unknown\":"));
}

TEST_CASE("declared sources combine by strongest duty", "[unit][compliance]") {
    // The rule that keeps a stale stamp from cancelling a researched verdict.
    // Guarded here as well as in the pure test because it is the kind of thing
    // a later "simplification" turns back into precedence.
    const std::string src = read_file("examples/cli/crispasr_speaker_identity.h");
    REQUIRE(contains(src, "duty_rank"));
    REQUIRE(contains(src, "if (duty_rank(model_value) > duty_rank(best))"));
    REQUIRE(contains(src, "if (duty_rank(backend_value) > duty_rank(best))"));
}

TEST_CASE("the verdict table records its evidence", "[unit][compliance]") {
    // These are research results, not code. The value of the table is the
    // reasoning beside each entry — a bare `return RealPerson;` is unreviewable
    // and gets flipped by whoever finds it inconvenient.
    const std::string src = read_file("examples/cli/crispasr_speaker_identity_models.h");
    // The SECTION BANNER, not a bare mention: "OPEN QUESTIONS" also appears as
    // a cross-reference inside the orpheus branch, so matching the short string
    // stayed green with the backlog itself deleted. The red-proof caught that.
    REQUIRE(contains(src, "OPEN QUESTIONS — models whose card has NOT been read"));
    // The conflict found while porting CrispTTS's research must stay visible:
    // kokoro is synthetic upstream, but CrispASR's German backbone is trained
    // on a named-narrator corpus.
    REQUIRE(contains(src, "HUI-Audio-Corpus-German"));
    // And the cost of matching on a file name has to be stated, not hidden.
    REQUIRE(contains(src, "WHAT MATCHING ON A FILE NAME COSTS"));
}

TEST_CASE("every synthesis surface resolves the speaker identity", "[unit][compliance]") {
    SECTION("CLI") {
        const std::string src = read_file("examples/cli/crispasr_run.cpp");
        REQUIRE(contains(src, "resolve_speaker_identity"));
        REQUIRE(contains(src, "needs_spoken_disclosure"));
        // ...and USES it: the disclaimer must key on the resolved answer, not
        // on is_voice_clone. A resolve whose result is never read is the exact
        // shape of a fix that ships inert.
        REQUIRE(contains(src, "if (needs_spoken_disclosure && !params.tts_no_spoken_disclaimer)"));
        REQUIRE_FALSE(contains(src, "if (is_voice_clone && !params.tts_no_spoken_disclaimer)"));
    }
    SECTION("HTTP server") {
        const std::string src = read_file("examples/cli/crispasr_server.cpp");
        REQUIRE(contains(src, "resolve_speaker_identity"));
        REQUIRE(contains(src, "crispasr_marking::decide(needs_spoken_disclosure,"));
    }
    SECTION("Wyoming") {
        const std::string src = read_file("examples/cli/wyoming.cpp");
        REQUIRE(contains(src, "resolve_speaker_identity"));
        REQUIRE(contains(src, "clone_decision.is_clone, rp.tts_voice_clone_consent, needs_spoken_disclosure"));
    }
    SECTION("C ABI") {
        const std::string src = read_file("src/crispasr_c_api.cpp");
        REQUIRE(contains(src, "crispasr_session_set_speaker_identity"));
        REQUIRE(contains(src, "s->voice_pack_identity = voice_decision.pack_identity;"));
        REQUIRE(contains(src, "requires_spoken_disclosure(s->voice_is_clone, identity)"));
    }
}

TEST_CASE("the identity is readable from a pack and a bank entry", "[unit][compliance]") {
    const std::string src = read_file("examples/cli/crispasr_voice_provenance.h");
    REQUIRE(contains(src, "speaker_identity_key()"));
    REQUIRE(contains(src, "speaker_identity_key_for(voice_name)"));
    // Carried out of classify_voice so the server's hot path does not reopen
    // the GGUF per request just to ask a second question about it.
    REQUIRE(contains(src, "d.pack_identity = is_bank_entry ? bank.identity : p.identity;"));
}

TEST_CASE("the operator can answer the question on every surface", "[unit][compliance]") {
    // A warning nobody can act on is one they learn to ignore, and this one
    // fires on every unresearched preset backend in the project.
    // Matched WITH the closing quote: a bare "--speaker-identity" substring
    // survives renaming the flag to "--speaker-identity-disabled", which the
    // red-proof duly demonstrated on an earlier draft of this line.
    REQUIRE(contains(read_file("examples/cli/cli.cpp"), "arg == \"--speaker-identity\""));
    REQUIRE(contains(read_file("examples/cli/crispasr_server.cpp"), "body.value(\"speaker_identity\""));
    REQUIRE(contains(read_file("include/crispasr_session.h"), "crispasr_session_set_speaker_identity"));
    REQUIRE(contains(read_file("include/crispasr.h"), "crispasr_session_set_speaker_identity"));
}

TEST_CASE("consent stays keyed on cloning alone", "[unit][compliance]") {
    // The other half of the split, and the one that is easy to get wrong in the
    // "safe" direction: making a real-person preset demand --i-have-rights
    // would gate every documented preset example behind an attestation the
    // operator cannot truthfully give.
    const std::string policy = read_file("examples/cli/crispasr_speaker_identity.h");
    REQUIRE(contains(policy, "inline bool requires_consent_attestation(bool is_clone, SpeakerIdentity /*identity*/) {\n"
                             "    return is_clone;\n"
                             "}"));
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
