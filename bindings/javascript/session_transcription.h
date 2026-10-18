#pragma once
#include "session_results.h"

struct binding_transcription {
    bool present = false;
    int frames = 0, vocabulary = 0;
    std::vector<binding_segment> segments;
    std::vector<float> logits;
};
// No JS calls while the session's logit flag and result buffers are borrowed.
// Plain owned data also permits later compute/servicer dispatch reuse.
static binding_transcription binding_transcribe_logits(crispasr_session* session, const std::vector<float>& pcm,
                                                       const std::string& language) {
    binding_transcription out;
    if (!session)
        return out;
    if (language.find('\0') != std::string::npos)
        throw std::invalid_argument("Embedded NUL in language");
    if (crispasr_session_set_return_logits(session, 1) < 0)
        throw std::runtime_error("Could not enable logits");
    struct Reset {
        crispasr_session* session;
        ~Reset() { crispasr_session_set_return_logits(session, 0); }
    } reset{session};
    auto raw = language.empty() ? crispasr_session_transcribe(session, pcm.data(), pcm.size())
                                : crispasr_session_transcribe_lang(session, pcm.data(), pcm.size(), language.c_str());
    binding_result_owner owned(raw);
    if (!raw)
        return out;
    out.segments = binding_copy_segments(raw);
    out.frames = crispasr_session_result_n_logit_frames(raw);
    out.vocabulary = crispasr_session_result_n_logit_vocab(raw);
    const float* data = crispasr_session_result_logits(raw);
    if (out.frames < 0 || out.vocabulary < 0 || (out.vocabulary && out.frames > INT_MAX / out.vocabulary))
        throw std::runtime_error("Invalid native logit dimensions");
    const int total = out.frames * out.vocabulary;
    if (total && !data)
        throw std::runtime_error("Missing native logit buffer");
    if (total)
        out.logits.assign(data, data + total);
    out.present = true;
    return out;
}
static emscripten::val binding_encode_transcription(const binding_transcription& input) {
    using emscripten::val;
    if (!input.present)
        return val::null();
    auto out = val::object();
    out.set("segments", binding_encode_segments(input.segments, 100.0));
    if (input.logits.empty()) {
        out.set("logits", val::null());
    } else {
        auto data = val::global("Float32Array").new_(input.logits.size());
        data.call<void>("set", val(emscripten::typed_memory_view(input.logits.size(), input.logits.data())));
        auto logits = val::object();
        logits.set("nFrames", input.frames);
        logits.set("nVocab", input.vocabulary);
        logits.set("data", data);
        out.set("logits", logits);
    }
    return out;
}
