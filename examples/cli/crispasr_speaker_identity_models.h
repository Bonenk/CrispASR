// crispasr_speaker_identity_models.h — the researched verdicts.
//
// crispasr_speaker_identity.h is the MECHANISM: three values, two duties, a
// resolution order. This file is the DATA — which shipped model produces whose
// voice, with the evidence for each answer beside it.
//
// Kept as one table rather than 50 adapter overrides on purpose. These are
// research results, not code: they come from reading each provider's own model
// card, they get revised when a provider says more, and they need to be
// reviewable in one place by someone checking the reasoning rather than the
// plumbing. A verdict scattered across `crispasr_backend_*.cpp` is a verdict
// nobody re-reads.
//
// HOW A VERDICT IS REACHED
// ------------------------
// From the provider's own documentation, never from the file name, the repo
// name, or what would be convenient. Two rules earned elsewhere in this
// project apply directly:
//
//   * Guessing "synthetic" to quiet the warning is the costly error. It is
//     silent, and it is wrong in the direction that REMOVES a disclosure.
//     Unknown is a question; synthetic is a claim.
//   * A model whose card does not say gets Unknown, and the check is recorded
//     here so the next reader does not re-litigate it.
//
// WHAT MATCHING ON A FILE NAME COSTS
// ----------------------------------
// One CrispASR backend serves many checkpoints — `orpheus` runs Canopy's base
// model AND Kartoffel's German fine-tune, and they have different answers — so
// the verdict cannot key on the backend alone. There is no metadata that
// distinguishes them either: models/convert-orpheus-to-gguf.py writes
// general.name = "orpheus-<variant>" for every one of them.
//
// So the table matches on the model's file name, which this project's own rule
// 3 says not to trust ("classify by provenance, not by filename"). It is used
// here because the alternative is no answer at all, and because the failure is
// SAFE: a renamed file matches nothing, falls through to Unknown, and warns.
// A rename cannot silently turn real_person into synthetic — it can only turn a
// known answer back into a question.
//
// The durable fix is a `crispasr.voice.speaker_identity` stamp in the published
// GGUFs, which this project controls for its own cstr/ mirrors. Until those are
// re-converted, this table is the legacy fallback — exactly the role
// architecture_is_recording_derived() plays for unstamped voice packs.

#pragma once

#include "crispasr_speaker_identity.h"

#include <string>

namespace crispasr_voice {

// Case-insensitive "does `haystack` contain `needle`". Local so this header
// stays pure and unit-testable with no model, no GGUF and no filesystem.
inline bool contains_ci(const std::string& haystack, const std::string& needle) {
    if (needle.empty() || haystack.size() < needle.size())
        return false;
    for (size_t i = 0; i + needle.size() <= haystack.size(); ++i) {
        size_t j = 0;
        for (; j < needle.size(); ++j) {
            char a = haystack[i + j];
            char b = needle[j];
            if (a >= 'A' && a <= 'Z')
                a = (char)(a - 'A' + 'a');
            if (b >= 'A' && b <= 'Z')
                b = (char)(b - 'A' + 'a');
            if (a != b)
                break;
        }
        if (j == needle.size())
            return true;
    }
    return false;
}

// Whose voice does this (backend, model) produce?
//
// `backend` is CrispasrBackend::name(); `model_path` is the loaded checkpoint
// (full path or bare name — only the tail is inspected).
inline SpeakerIdentity identity_for_model(const std::string& backend, const std::string& model_path) {
    // ── piper ────────────────────────────────────────────────────────────
    // rhasspy/piper voices are single-speaker VITS models, each trained on one
    // named donor's recordings. Every voice in this project's registry is an
    // identifiable person: en_US-lessac (the Lessac Technologies corpus),
    // de_DE-thorsten (Thorsten Müller, who released his voice under CC0 —
    // released, which is consent to publish, not a reason to stop disclosing),
    // de_DE-kerstin. Blanket verdict: there is no synthetic piper voice here to
    // wrongly gate.
    if (backend == "piper")
        return SpeakerIdentity::RealPerson;

    // ── kokoro ───────────────────────────────────────────────────────────
    // hexgrad/Kokoro-82M's voicepacks are style vectors, documented upstream as
    // designed/blended rather than any one person.
    //
    // EXCEPT the German fine-tune, which is left Unknown deliberately — see the
    // OPEN QUESTIONS note at the bottom of this file. Its base is trained on
    // HUI-Audio-Corpus-German, whose narrators are NAMED, and inheriting
    // "synthetic" from the English verdict would be assuming the answer on the
    // one variant where there is a reason to doubt it.
    if (backend == "kokoro") {
        if (contains_ci(model_path, "hui"))
            return SpeakerIdentity::Unknown;
        return SpeakerIdentity::Synthetic;
    }

    // ── orpheus ──────────────────────────────────────────────────────────
    // One backend, several checkpoints, different answers.
    //
    // kartoffel-orpheus-de-natural: the card describes it as fine-tuned
    // "primarily on natural human speech recordings" — permissively licensed
    // podcasts, lectures and OER — and its 19 speakers were EXTRACTED from
    // those recordings ("not all speakers could be reconstructed"). Real people
    // who spoke in public.
    if (backend == "orpheus") {
        if (contains_ci(model_path, "kartoffel-orpheus-de-natural"))
            return SpeakerIdentity::RealPerson;
        // Everything else on this backend stays Unknown, including:
        //   orpheus-3b-0.1-ft   Canopy Labs disclose 100k+ h of "permissive /
        //                       non-copyrighted" audio and nothing at all about
        //                       the origin of tara, leah, jess, leo, dan, mia,
        //                       zac, zoe. HF card, GitHub repo and web checked.
        //   Orpheus-3b-German-FT (lex-au)   No training-data documentation.
        //   kartoffel-orpheus-de-synthetic  NOT researched — see OPEN QUESTIONS.
        return SpeakerIdentity::Unknown;
    }

    // Every other backend: not yet researched. Unknown warns once per model and
    // names the fix; it does not claim the voice is synthetic.
    return SpeakerIdentity::Unknown;
}

// ─────────────────────────────────────────────────────────────────────────
// OPEN QUESTIONS — models whose card has NOT been read, or where the evidence
// points somewhere the current verdict does not. Listed so they are picked up
// rather than rediscovered, and so "unknown" is visibly a backlog rather than
// an opinion.
//
//   kokoro-de-hui-*     CONFLICT. The sibling project CrispTTS classifies
//                       kokoro as synthetic, and for the English voicepacks
//                       that is right. But this project's German backbone is
//                       `kokoro-de-hui-base`, trained on HUI-Audio-Corpus-
//                       German — the same corpus, with the same NAMED narrators
//                       (Bernd, Hokuspokus, Friedrich, Eva, Karlsson, Sonja),
//                       that CrispTTS itself cites when marking the NeMo
//                       FastPitch German model real_person. Whether a Kokoro
//                       style vector derived from that corpus is recognisably
//                       one of them is a real question and is not answered
//                       here. Held at Unknown rather than inheriting either
//                       neighbouring verdict.
//
//   fastpitch           CrispASR ships fastpitch-en (NVIDIA, English, single
//                       speaker) — NOT the German NeMo model CrispTTS marks
//                       real_person. Different weights, so that verdict does
//                       not port. NVIDIA's English FastPitch is conventionally
//                       trained on LJSpeech, which is one identifiable narrator
//                       — a strong hypothesis, deliberately not asserted here
//                       without reading the card.
//
//   speecht5            CrispASR ships microsoft/speecht5_tts, whose speaker
//                       comes from a 512-d x-vector the OPERATOR supplies via
//                       --voice. The identity is therefore per-invocation, not
//                       per-model, and no backend-level verdict can be right.
//                       CrispTTS's real_person applies to a German fine-tune
//                       with CMU ARCTIC x-vectors baked in, which is a
//                       different model. Operators pass --speaker-identity.
//
//   melotts, bananamind-tts   No training-data documentation found.
//
//   bark, csm, parler-tts, and the rest   Not yet examined.
// ─────────────────────────────────────────────────────────────────────────

} // namespace crispasr_voice
