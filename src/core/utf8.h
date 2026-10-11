// core/utf8.h — UTF-8 codepoint counting, in one place.
//
// The same four-line loop had been written out four separate times before this
// header existed: `core_lid_probe::utf8_length` (core/lid_probe.h),
// a file-static `utf8_len` in crispasr.cpp, a lambda in kokoro.cpp, and — as of
// the f5-tts duration fix — a fourth in f5_tts.cpp. They agree, which is luck
// rather than design: each is a hand-rolled continuation-byte test.
//
// Counting matters because a BYTE count is not a character count outside ASCII,
// and the two get mixed. f5-tts derived a speech rate as chars/sec from the
// reference and then multiplied it by a byte count of the target, so Devanagari
// (3 bytes/char) inflated the estimate ~3x; since the ODE solve scales with the
// mel sequence length, a two-second line became minutes of synthesis.
#pragma once

#include <cstddef>
#include <string>

namespace core_utf8 {

// Number of UTF-8 codepoints in [s, s + n_bytes).
//
// Counts NON-CONTINUATION bytes, i.e. every byte that is not 10xxxxxx. That is
// the leading byte of each sequence, so well-formed input gives the codepoint
// count exactly.
//
// On MALFORMED input it degrades rather than rejecting: a stray continuation
// byte is not counted, and an over-long or truncated sequence still counts its
// leading byte once. Callers here are estimating a duration or a length ratio,
// where a slightly-off count on invalid input beats throwing. length() does not
// validate UTF-8, and it should not be used as a validity check.
inline size_t length(const char* s, size_t n_bytes) {
    size_t n = 0;
    for (size_t i = 0; i < n_bytes; i++)
        if (((unsigned char)s[i] & 0xC0) != 0x80)
            n++;
    return n;
}

inline size_t length(const std::string& s) {
    return length(s.data(), s.size());
}

// NUL-terminated overload.
inline size_t length(const char* s) {
    if (!s)
        return 0;
    size_t n = 0;
    for (; *s; ++s)
        if (((unsigned char)*s & 0xC0) != 0x80)
            n++;
    return n;
}

// RFC 3629 sequence length, or zero for malformed/truncated input.
inline size_t sequence_length(const std::string& s, size_t i) {
    if (i >= s.size())
        return 0;
    const unsigned char c = (unsigned char)s[i];
    if (c < 0x80)
        return 1;
    size_t n;
    unsigned char lo = 0x80, hi = 0xBF;
    if (c >= 0xC2 && c <= 0xDF)
        n = 2;
    else if (c >= 0xE0 && c <= 0xEF) {
        n = 3;
        if (c == 0xE0)
            lo = 0xA0;
        else if (c == 0xED)
            hi = 0x9F;
    } else if (c >= 0xF0 && c <= 0xF4) {
        n = 4;
        if (c == 0xF0)
            lo = 0x90;
        else if (c == 0xF4)
            hi = 0x8F;
    } else
        return 0;
    if (s.size() - i < n)
        return 0;
    for (size_t k = 1; k < n; ++k) {
        const unsigned char b = (unsigned char)s[i + k];
        if (k == 1 ? (b < lo || b > hi) : (b < 0x80 || b > 0xBF))
            return 0;
    }
    return n;
}

// Normalize an ASSEMBLED string, never individual byte-level BPE pieces.
// Match the CLI JSON policy: replace each malformed byte with U+FFFD.
// Valid strings take no allocation and retain their original storage.
inline void replace_invalid(std::string& s) {
    size_t i = 0;
    while (i < s.size()) {
        const size_t n = sequence_length(s, i);
        if (!n)
            break;
        i += n;
    }
    if (i == s.size())
        return;
    std::string clean(s, 0, i);
    while (i < s.size()) {
        const size_t n = sequence_length(s, i);
        if (n) {
            clean.append(s, i, n);
            i += n;
        } else {
            clean += "\xEF\xBF\xBD";
            ++i;
        }
    }
    s.swap(clean);
}

} // namespace core_utf8
