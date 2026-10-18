#pragma once

#include "crispasr_session.h"
#include <emscripten/bind.h>
#include <cstdint>
#include <climits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Plain owned data can cross the browser compute queue; emval objects cannot.
struct binding_piece {
    std::string text;
    int64_t t0 = -1, t1 = -1;
    float p = -1;
};
struct binding_word : binding_piece {
    std::vector<binding_piece> alts;
};
struct binding_segment : binding_piece {
    std::string speaker;
    std::vector<binding_word> words;
    std::vector<binding_piece> tokens;
};
static std::string binding_string(const char* p) {
    return p ? p : "";
}
struct binding_result_deleter {
    void operator()(crispasr_session_result* r) const {
        if (r)
            crispasr_session_result_free(r);
    }
};
using binding_result_owner = std::unique_ptr<crispasr_session_result, binding_result_deleter>;
static int binding_result_count(int n) {
    if (n < 0)
        throw std::runtime_error("Negative native result count");
    return n;
}
static std::vector<binding_segment> binding_copy_segments(crispasr_session_result* r) {
    std::vector<binding_segment> out;
    const int count = binding_result_count(crispasr_session_result_n_segments(r));
    for (int i = 0; i < count; ++i) {
        binding_segment seg;
        seg.text = binding_string(crispasr_session_result_segment_text(r, i));
        seg.t0 = crispasr_session_result_segment_t0(r, i);
        seg.t1 = crispasr_session_result_segment_t1(r, i);
        seg.p = crispasr_session_result_segment_no_speech_prob(r, i);
        seg.speaker = binding_string(crispasr_session_result_segment_speaker(r, i));
        const int words = binding_result_count(crispasr_session_result_n_words(r, i));
        for (int j = 0; j < words; ++j) {
            binding_word w;
            w.text = binding_string(crispasr_session_result_word_text(r, i, j));
            w.t0 = crispasr_session_result_word_t0(r, i, j);
            w.t1 = crispasr_session_result_word_t1(r, i, j);
            w.p = crispasr_session_result_word_p(r, i, j);
            const int alts = binding_result_count(crispasr_session_result_word_n_alts(r, i, j));
            for (int k = 0; k < alts; ++k) {
                binding_piece alt;
                alt.text = binding_string(crispasr_session_result_word_alt_text(r, i, j, k));
                alt.p = crispasr_session_result_word_alt_p(r, i, j, k);
                w.alts.push_back(std::move(alt));
            }
            seg.words.push_back(std::move(w));
        }
        const int tokens = binding_result_count(crispasr_session_result_n_tokens(r, i));
        for (int j = 0; j < tokens; ++j) {
            binding_piece t;
            t.text = binding_string(crispasr_session_result_token_text(r, i, j));
            t.t0 = crispasr_session_result_token_t0(r, i, j);
            t.t1 = crispasr_session_result_token_t1(r, i, j);
            t.p = crispasr_session_result_token_p(r, i, j);
            seg.tokens.push_back(std::move(t));
        }
        out.push_back(std::move(seg));
    }
    return out;
}
static std::vector<binding_segment> binding_take_segments(crispasr_session_result* r) {
    binding_result_owner owned(r);
    return binding_copy_segments(r);
}
static emscripten::val binding_bytes(const std::string& text) {
    auto out = emscripten::val::global("Uint8Array").new_(text.size());
    out.call<void>("set", emscripten::val(emscripten::typed_memory_view(
                              text.size(), reinterpret_cast<const unsigned char*>(text.data()))));
    return out;
}
static emscripten::val binding_encode_piece(const binding_piece& p, double units, bool raw = false) {
    auto out = emscripten::val::object();
    bool timed = p.t0 >= 0 && p.t1 >= p.t0;
    out.set("hasTimestamps", timed);
    out.set("t0", timed ? p.t0 / units : -1.0);
    out.set("t1", timed ? p.t1 / units : -1.0);
    out.set("p", p.p);
    if (raw) {
        auto bytes = binding_bytes(p.text);
        out.set("rawText", bytes);
        out.set("text", emscripten::val::global("TextDecoder").new_().call<emscripten::val>("decode", bytes));
    } else
        out.set("text", p.text);
    return out;
}
static emscripten::val binding_encode_segments(const std::vector<binding_segment>& segments, double units) {
    auto out = emscripten::val::array();
    for (const auto& seg : segments) {
        auto item = binding_encode_piece(seg, units);
        item.set("noSpeechProb", seg.p);
        item.set("speaker", seg.speaker);
        auto words = emscripten::val::array(), tokens = emscripten::val::array();
        for (const auto& w : seg.words) {
            auto word = binding_encode_piece(w, units), alts = emscripten::val::array();
            for (const auto& a : w.alts) {
                auto alt = emscripten::val::object();
                alt.set("text", a.text);
                alt.set("p", a.p);
                alts.call<void>("push", alt);
            }
            word.set("alts", alts);
            words.call<void>("push", word);
        }
        for (const auto& t : seg.tokens)
            tokens.call<void>("push", binding_encode_piece(t, units, true));
        item.set("words", words);
        item.set("tokens", tokens);
        out.call<void>("push", item);
    }
    return out;
}
