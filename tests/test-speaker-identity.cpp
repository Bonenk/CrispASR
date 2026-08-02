// test-speaker-identity.cpp — whose voice is a PRESET voice, and which duty
// that triggers.
//
// The gap this guards: `is_clone == false` was doing two jobs at once. It
// correctly meant "this operator owes no consent attestation", and it
// incorrectly meant "there is nothing to disclose". A preset voice shipped
// inside a model can be an identifiable individual — a named donor, a corpus
// speaker such as VCTK's p225 — and Art. 3(60) attaches to the audio resembling
// that person, not to which pipeline produced it.
//
// Pure predicate, so it is guarded on the tier CI actually runs, for the same
// reason as test-marking-policy.cpp and test-voice-clone-policy.cpp.
//
// NOTE ON SCOPE: this file tests the MECHANISM. It deliberately asserts nothing
// about which backend is which — those verdicts come from reading each
// provider's model card, land as declared_speaker_identity() overrides and GGUF
// stamps, and get their own coverage. Pinning a verdict here that nobody has
// researched would be the same mistake as guessing "synthetic" to quiet a
// warning.

#include "crispasr_speaker_identity.h"

#include <catch2/catch_test_macros.hpp>

using crispasr_voice::parse_speaker_identity;
using crispasr_voice::requires_consent_attestation;
using crispasr_voice::requires_spoken_disclosure;
using crispasr_voice::resolve_speaker_identity;
using crispasr_voice::should_warn_unknown_identity;
using crispasr_voice::SpeakerIdentity;

// ---------------------------------------------------------------------------
// The split: two duties, two triggers.
// ---------------------------------------------------------------------------

TEST_CASE("a real-person preset is disclosed", "[unit][compliance]") {
    // The bug, pinned. Not a clone — nothing went through a baker — but the
    // voice is a specific person, so the output is a deep fake and owes the
    // Art. 50(4) audible label.
    REQUIRE(requires_spoken_disclosure(/*is_clone=*/false, SpeakerIdentity::RealPerson));
}

TEST_CASE("a real-person preset is NOT consent-gated", "[unit][compliance]") {
    // Whether that donor agreed to the model being trained is a licensing
    // question settled upstream between them and whoever trained it. The
    // operator downstream cannot attest to it, so demanding --i-have-rights
    // would be theatre — and would break every documented preset example.
    REQUIRE_FALSE(requires_consent_attestation(/*is_clone=*/false, SpeakerIdentity::RealPerson));
}

TEST_CASE("cloning triggers both duties", "[unit][compliance]") {
    // The operator IS the one taking a specific person's voice here, so they
    // attest; and the output is a deep fake, so it is disclosed.
    REQUIRE(requires_consent_attestation(true, SpeakerIdentity::Unknown));
    REQUIRE(requires_spoken_disclosure(true, SpeakerIdentity::Unknown));
    // ...and the identity of a cloned voice cannot weaken either. A pack that
    // claims "synthetic" while being handed a real recording must not escape.
    REQUIRE(requires_spoken_disclosure(true, SpeakerIdentity::Synthetic));
    REQUIRE(requires_consent_attestation(true, SpeakerIdentity::Synthetic));
}

TEST_CASE("a synthetic preset owes neither", "[unit][compliance]") {
    // Nobody's voice, no deep fake. Art. 50(2) marking still applies and is
    // handled elsewhere — synthesis always watermarks.
    REQUIRE_FALSE(requires_spoken_disclosure(false, SpeakerIdentity::Synthetic));
    REQUIRE_FALSE(requires_consent_attestation(false, SpeakerIdentity::Synthetic));
}

TEST_CASE("unknown does not force a disclosure, but does warn", "[unit][compliance]") {
    // Treating unknown as real_person would prepend a spoken sentence to every
    // stock TTS voice in the project and teach operators to reach for
    // --no-spoken-disclaimer, which is worse than the disease. Treating it as
    // synthetic would silently assert the convenient answer on exactly the
    // models nobody has checked. So: no disclosure, but say so.
    REQUIRE_FALSE(requires_spoken_disclosure(false, SpeakerIdentity::Unknown));
    REQUIRE(should_warn_unknown_identity(false, SpeakerIdentity::Unknown));
}

TEST_CASE("a clone never warns about unknown identity", "[unit][compliance]") {
    // It is disclosed and gated regardless, so the question cannot change the
    // outcome and the warning would be pure noise on the one path that is
    // already fully handled.
    REQUIRE_FALSE(should_warn_unknown_identity(true, SpeakerIdentity::Unknown));
    // Nor does an answered question warn.
    REQUIRE_FALSE(should_warn_unknown_identity(false, SpeakerIdentity::Synthetic));
    REQUIRE_FALSE(should_warn_unknown_identity(false, SpeakerIdentity::RealPerson));
}

// ---------------------------------------------------------------------------
// Parsing — the values arrive from a CLI flag, a JSON field and GGUF metadata.
// ---------------------------------------------------------------------------

TEST_CASE("the three values parse, case- and space-tolerantly", "[unit][compliance]") {
    REQUIRE(parse_speaker_identity("real_person") == SpeakerIdentity::RealPerson);
    REQUIRE(parse_speaker_identity("REAL_PERSON") == SpeakerIdentity::RealPerson);
    REQUIRE(parse_speaker_identity("  Real_Person  ") == SpeakerIdentity::RealPerson);
    REQUIRE(parse_speaker_identity("synthetic") == SpeakerIdentity::Synthetic);
    REQUIRE(parse_speaker_identity("unknown") == SpeakerIdentity::Unknown);
}

TEST_CASE("an unset value is unknown and is NOT a typo", "[unit][compliance]") {
    // Absence is the normal case — most packs declare nothing — and must not
    // produce a warning telling the operator they mistyped something.
    bool recognised = false;
    REQUIRE(parse_speaker_identity("", &recognised) == SpeakerIdentity::Unknown);
    REQUIRE(recognised);
}

TEST_CASE("a typo is reported, not silently downgraded", "[unit][compliance]") {
    // "real-person" resolving quietly to unknown would drop the disclosure the
    // operator explicitly asked for — a wrong answer in the direction that
    // removes a duty, which is the direction that must never be silent.
    for (const char* bad : {"real-person", "human", "person", "REALPERSON", "true"}) {
        INFO("value=" << bad);
        bool recognised = true;
        REQUIRE(parse_speaker_identity(bad, &recognised) == SpeakerIdentity::Unknown);
        REQUIRE_FALSE(recognised);
    }
}

TEST_CASE("the round trip through to_string is stable", "[unit][compliance]") {
    // These strings are written into GGUF metadata by bakers and read back by
    // the gate; a drift between writer and reader fails open.
    for (auto id : {SpeakerIdentity::RealPerson, SpeakerIdentity::Synthetic, SpeakerIdentity::Unknown}) {
        REQUIRE(parse_speaker_identity(crispasr_voice::to_string(id)) == id);
    }
    REQUIRE(std::string(crispasr_voice::to_string(SpeakerIdentity::RealPerson)) == "real_person");
    REQUIRE(std::string(crispasr_voice::speaker_identity_key()) == "crispasr.voice.speaker_identity");
    REQUIRE(crispasr_voice::speaker_identity_key_for("af_heart") == "crispasr.voice.af_heart.speaker_identity");
}

// ---------------------------------------------------------------------------
// Resolution precedence: override > pack > backend > unknown.
// ---------------------------------------------------------------------------

TEST_CASE("the operator override outranks everything", "[unit][compliance]") {
    // They may have read the model card the pack was built before anyone wrote.
    REQUIRE(resolve_speaker_identity(SpeakerIdentity::RealPerson, SpeakerIdentity::Synthetic,
                                     SpeakerIdentity::Synthetic) == SpeakerIdentity::RealPerson);
    // ...and it must move the answer in BOTH directions. Someone who knows a
    // "real_person" label is wrong has to be able to say so, or the flag is
    // only usable for adding duties and gets ignored for the other half.
    REQUIRE(resolve_speaker_identity(SpeakerIdentity::Synthetic, SpeakerIdentity::RealPerson,
                                     SpeakerIdentity::RealPerson) == SpeakerIdentity::Synthetic);
}

TEST_CASE("the pack outranks the backend default", "[unit][compliance]") {
    // The backend default describes its built-in voices; a pack knows about
    // itself, and is the more specific claim.
    REQUIRE(resolve_speaker_identity(SpeakerIdentity::Unknown, SpeakerIdentity::RealPerson,
                                     SpeakerIdentity::Synthetic) == SpeakerIdentity::RealPerson);
}

TEST_CASE("the backend default fills in when nothing else speaks", "[unit][compliance]") {
    REQUIRE(resolve_speaker_identity(SpeakerIdentity::Unknown, SpeakerIdentity::Unknown, SpeakerIdentity::RealPerson) ==
            SpeakerIdentity::RealPerson);
}

TEST_CASE("nothing declared resolves to unknown", "[unit][compliance]") {
    // The default state of the project today: the mechanism ships before the
    // research does, and claims nothing until a model card has been read.
    REQUIRE(resolve_speaker_identity(SpeakerIdentity::Unknown, SpeakerIdentity::Unknown, SpeakerIdentity::Unknown) ==
            SpeakerIdentity::Unknown);
}

// ---------------------------------------------------------------------------
// The once-per-model warning.
// ---------------------------------------------------------------------------

TEST_CASE("the unknown-identity warning fires once per model", "[unit][compliance]") {
    crispasr_voice::reset_unknown_identity_warnings_for_test();
    REQUIRE(crispasr_voice::claim_unknown_identity_warning("kokoro"));
    REQUIRE_FALSE(crispasr_voice::claim_unknown_identity_warning("kokoro"));
    // Keyed per model, not per process: a server may load several backends and
    // each unanswered one is a separate thing the operator needs to know.
    REQUIRE(crispasr_voice::claim_unknown_identity_warning("piper"));
    REQUIRE_FALSE(crispasr_voice::claim_unknown_identity_warning("piper"));
}

TEST_CASE("the test reset actually clears the state", "[unit][compliance]") {
    // Guarding the guard: an earlier draft kept the set and the reset in
    // separate function-local statics, so the reset compiled, ran, and did
    // nothing — which would have made every case above order-dependent.
    crispasr_voice::reset_unknown_identity_warnings_for_test();
    REQUIRE(crispasr_voice::claim_unknown_identity_warning("same-model"));
    crispasr_voice::reset_unknown_identity_warnings_for_test();
    REQUIRE(crispasr_voice::claim_unknown_identity_warning("same-model"));
}

TEST_CASE("the warning names what to do about it", "[unit][compliance]") {
    // A warning an operator cannot act on is one they learn to ignore, and this
    // one fires on every unresearched preset backend in the project.
    const std::string w = crispasr_voice::unknown_identity_warning("kokoro");
    REQUIRE(w.find("kokoro") != std::string::npos);
    REQUIRE(w.find("--speaker-identity real_person") != std::string::npos);
    REQUIRE(w.find("--speaker-identity synthetic") != std::string::npos);
    REQUIRE(w.find("Art. 50(4)") != std::string::npos);
    // An unnamed model still produces a usable line rather than a dangling quote.
    REQUIRE(crispasr_voice::unknown_identity_warning("").find("<unknown-model>") != std::string::npos);
}
