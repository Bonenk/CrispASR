// src/core/omnivoice_duration.h — OmniVoice target-length estimation.
//
// OmniVoice is a masked *iterative* generator, not autoregressive: it is handed
// a target length T up front and fills those frames in. There is no EOS to stop
// at and no way to ask for more room, so if T is too small for the text the
// tail simply does not fit — you get the beginning of the utterance and a
// fragment where the end should be (#363).
//
// T comes from a speaking rate. With a voice prompt the rate is measured from
// the reference the caller supplied — weight(ref_text) / ref_T — which makes it
// user-controlled, and wrong whenever the transcript does not match the audio:
// a typo, the wrong file, the target text pasted by mistake, a noisy clip that
// auto-transcribed badly, or a partial transcript of a long recording.
//
// Measured on an M1 with omnivoice-f16, cloning a 2.60 s reference and asking
// for a 22-word line. With a ref_text matching the audio: 6.60 s, complete. With
// a ref_text 3.2x too long for the same audio, and nothing else changed:
//
//     2.04 s, and the words spoken were
//     "and then continued running through the field for f and then continued
//      running the field for a very long time."
//
// — fragments, of the end of the *reference* text. So this file also carries
// the plausibility guard: a rate no human could produce means the reference
// pair is inconsistent, and falling back to the built-in anchor is strictly
// better than rendering that.
//
// Extracted verbatim from omnivoice.cpp so it can be unit-tested; the weights
// themselves are unchanged.
#pragma once

#include "core/crispasr_env.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace core_omnivoice_duration {

// Per-codepoint "phonetic weight" (relative speech duration). Adapted from
// OmniVoice's rule-based estimator (omnivoice/utils/duration.py — Apache-2.0,
// © Xiaomi/k2-fsa; C++ mirror in ServeurpersoCom/omnivoice.cpp — MIT). Weights:
// mark 0, separator 0.2, punct/symbol 0.5, digit 3.5 (spoken as words),
// latin/cyrillic/greek 1.0, arabic/hebrew/thai 1.5, indic 1.8, kana 2.2,
// hangul 2.5, cjk 3.0.
inline double duration_cp_weight(uint32_t cp) {
    if ((cp >= 0x30 && cp <= 0x39) || (cp >= 0xFF10 && cp <= 0xFF19))
        return 3.5; // digits — spoken as words
    if ((cp >= 0x41 && cp <= 0x5A) || (cp >= 0x61 && cp <= 0x7A))
        return 1.0; // ASCII letters
    if (cp < 0x80)
        return (cp == 0x20 || cp == 0x09 || cp == 0x0A || cp == 0x0D) ? 0.2 : 0.5;
    if ((cp >= 0x0300 && cp <= 0x036F) || (cp >= 0x1DC0 && cp <= 0x1DFF) || (cp >= 0x20D0 && cp <= 0x20FF))
        return 0.0; // combining marks — silent
    if (cp == 0x3000)
        return 0.2; // ideographic space
    if ((cp >= 0x3001 && cp <= 0x303F) || (cp >= 0xFF00 && cp <= 0xFF0F) || (cp >= 0xFF1A && cp <= 0xFF20) ||
        (cp >= 0xFF3B && cp <= 0xFF40) || (cp >= 0xFF5B && cp <= 0xFF65))
        return 0.5; // CJK punctuation
    if (cp >= 0x00C0 && cp <= 0x024F)
        return 1.0; // Latin extended
    if (cp >= 0x0370 && cp <= 0x03FF)
        return 1.0; // Greek
    if (cp >= 0x0400 && cp <= 0x04FF)
        return 1.0; // Cyrillic
    if (cp >= 0x0590 && cp <= 0x06FF)
        return 1.5; // Hebrew + Arabic
    if (cp >= 0x0E00 && cp <= 0x0EFF)
        return 1.5; // Thai + Lao
    if (cp >= 0x0900 && cp <= 0x0DFF)
        return 1.8; // Indic (Devanagari..Sinhala)
    if (cp >= 0x3040 && cp <= 0x30FF)
        return 2.2; // Hiragana + Katakana
    if ((cp >= 0xAC00 && cp <= 0xD7AF) || (cp >= 0x1100 && cp <= 0x11FF))
        return 2.5; // Hangul
    if ((cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x20000 && cp <= 0x2FA1F))
        return 3.0; // CJK ideographs
    return 1.0;     // default letter-ish
}

inline double duration_text_weight(const std::string& text) {
    double w = 0.0;
    for (size_t i = 0; i < text.size();) {
        unsigned char c = text[i];
        uint32_t cp;
        int adv;
        if (c < 0x80) {
            cp = c;
            adv = 1;
        } else if (c < 0xE0) {
            cp = c & 0x1F;
            adv = 2;
        } else if (c < 0xF0) {
            cp = c & 0x0F;
            adv = 3;
        } else {
            cp = c & 0x07;
            adv = 4;
        }
        for (int k = 1; k < adv && i + (size_t)k < text.size(); k++)
            cp = (cp << 6) | ((unsigned char)text[i + k] & 0x3F);
        w += duration_cp_weight(cp);
        i += adv;
    }
    return w;
}

// Estimate target length (in ref_dur's unit) from a reference (text, dur) pair,
// mirroring RuleDurationEstimator.estimate_duration:
//   speed = weight(ref_text) / ref_dur;  est = weight(target) / speed
// with a power-curve BOOST when the linear estimate is short (< low_threshold):
//   est = low_threshold * (est/low_threshold)^(1/boost_strength)
// The boost lengthens short clips so they don't render too fast / skip
// characters (#254). Defaults (low_threshold=50, boost_strength=3) match Python.
inline double duration_estimate(const std::string& target_text, const std::string& ref_text, double ref_dur,
                                double low_threshold = 50.0, double boost_strength = 3.0) {
    if (ref_dur <= 0.0 || ref_text.empty())
        return 0.0;
    double ref_weight = duration_text_weight(ref_text);
    if (ref_weight == 0.0)
        return 0.0;
    double speed = ref_weight / ref_dur;
    double est = duration_text_weight(target_text) / speed;
    if (est < low_threshold)
        return low_threshold * std::pow(est / low_threshold, 1.0 / boost_strength);
    return est;
}

// Plausible speaking rates, in weighted characters per frame. Frames are 40 ms
// (24 kHz / 960 samples), so these are 6.25 and 37.5 weighted chars per second.
//
// The built-in anchor sits at 14.1/25 = 0.564, and measured English synthesis
// lands at 0.57-0.59, so the band is roughly a third to two and a half times
// normal speech. It is deliberately wide: the goal is to reject a ref_text that
// does not describe the audio at all, not to police an unusual speaker. The
// 3.2x-too-long reference that produced the fragments quoted above implies
// 1.92, comfortably outside; a correct transcript of the same clip gives 0.571,
// comfortably inside.
inline constexpr double kMinRefRate = 0.25;
inline constexpr double kMaxRefRate = 1.50;

// True when (ref_text, ref_T) imply a speaking rate a human could produce.
// A false result means the pair is internally inconsistent — almost always a
// transcript that does not match the reference audio.
inline bool ref_rate_is_plausible(const std::string& ref_text, int ref_T) {
    if (ref_T <= 0 || ref_text.empty())
        return false;
    const double w = duration_text_weight(ref_text);
    if (w <= 0.0)
        return false;
    const double rate = w / (double)ref_T;
    return rate >= kMinRefRate && rate <= kMaxRefRate;
}

// Target audio length (frames). With a voice prompt the anchor IS the reference
// (ref_text → ref_T frames) so the length tracks the reference speaker's actual
// rate (fixes "duration doesn't change with a reference voice", #254). Without
// one, fall back to the canonical "Nice to meet you." ≈ 25-frame anchor.
// OMNIVOICE_FRAMES_PER_CHAR overrides the no-ref frames/weighted-char rate.
//
// A reference implying an impossible rate is rejected in favour of that anchor
// (#363). Trusting it instead does not produce merely a mistimed clip: too high
// a rate yields a T the text cannot fit, and because the generator is masked
// iterative rather than autoregressive it cannot ask for more room, so the
// utterance ends mid-phrase. Set CRISPASR_OMNIVOICE_REF_RATE_CHECK=0 to trust
// the reference regardless.
inline int estimate_target_tokens(const std::string& text, const std::string& ref_text = std::string(), int ref_T = 0,
                                  float speed = 1.0f) {
    std::string rt;
    double rd;
    bool have_ref = ref_T > 0 && !ref_text.empty();

    if (have_ref && !ref_rate_is_plausible(ref_text, ref_T)) {
        const char* e = crispasr_env::get("CRISPASR_OMNIVOICE_REF_RATE_CHECK");
        const bool enforce = !(e && e[0] == '0');
        // Warn either way: when the check is off this is still the most likely
        // explanation for whatever comes out.
        fprintf(stderr,
                "omnivoice: reference transcript implies %.2f weighted chars/frame, outside the "
                "plausible %.2f-%.2f — --ref-text likely does not match --voice.%s\n",
                duration_text_weight(ref_text) / (double)ref_T, kMinRefRate, kMaxRefRate,
                enforce ? " Using the built-in duration anchor instead (#363); set "
                          "CRISPASR_OMNIVOICE_REF_RATE_CHECK=0 to trust it anyway."
                        : " CRISPASR_OMNIVOICE_REF_RATE_CHECK=0 is set, using it anyway.");
        if (enforce)
            have_ref = false;
    }

    if (have_ref) {
        rt = ref_text;
        rd = (double)ref_T;
    } else {
        rt = "Nice to meet you.";
        rd = 25.0;
        if (const char* e = crispasr_env::get("CRISPASR_OMNIVOICE_FRAMES_PER_CHAR")) {
            float v = (float)atof(e);
            if (v > 0.0f)
                rd = v * duration_text_weight(rt); // env = frames per weighted char
        }
    }
    double est = duration_estimate(text, rt, rd) / speed;
    return std::max((int)est, 10);
}
} // namespace core_omnivoice_duration
