#pragma once
#include "session_lifetime.h"
#include "session_options.h"
#include <memory>

struct crispasr_vad_abi_opts {
    float threshold;
    int32_t min_speech_duration_ms;
    int32_t min_silence_duration_ms;
    int32_t speech_pad_ms;
    int32_t chunk_seconds;
    int32_t n_threads;
};
static_assert(sizeof(crispasr_vad_abi_opts) == 24, "VAD options ABI changed");
static emscripten::val binding_transcribe_vad(const emscripten::val& audio, int sampleRate, const std::string& model,
                                              const std::string& language, const emscripten::val& options) {
    using emscripten::val;
    binding_check_reentry();
    if (!g_asr_session)
        throw std::runtime_error("ASR session is closed");
    if (sampleRate <= 0 || model.empty() || model.find('\0') != std::string::npos ||
        language.find('\0') != std::string::npos)
        throw std::invalid_argument("Invalid VAD sample rate, model or language");
    auto pcm = binding_pcm(audio);
    crispasr_vad_abi_opts p{};
    bool hasOptions = !options.isNull() && !options.isUndefined();
    if (hasOptions) {
        if (options.typeOf().as<std::string>() != "object" || val::global("Array").call<bool>("isArray", options))
            throw std::invalid_argument("VAD options must be an object");
        auto threshold = options["threshold"];
        if (!threshold.isUndefined() && threshold.typeOf().as<std::string>() != "number")
            throw std::invalid_argument("VAD threshold must be numeric");
        double t = threshold.isUndefined() ? .5 : threshold.as<double>();
        if (!std::isfinite(t) || t < 0 || t > 1)
            throw std::invalid_argument("VAD threshold must be in [0,1]");
        p.threshold = float(t);
        p.min_speech_duration_ms = binding_option_integer(options, "min_speech_duration_ms", 250);
        p.min_silence_duration_ms = binding_option_integer(options, "min_silence_duration_ms", 100);
        p.speech_pad_ms = binding_option_integer(options, "speech_pad_ms", 30);
        p.chunk_seconds = binding_option_integer(options, "chunk_seconds", 30);
#ifdef __EMSCRIPTEN_PTHREADS__
        const int defaults = 4;
#else
        const int defaults = 1;
#endif
        p.n_threads = binding_option_integer(options, "n_threads", defaults);
        if (p.min_speech_duration_ms < 0 || p.min_silence_duration_ms < 0 || p.speech_pad_ms < 0 ||
            p.chunk_seconds < 0 || p.n_threads <= 0)
            throw std::invalid_argument("Invalid VAD options");
    }
    binding_check_reentry();
    binding_require_no_stream(g_asr_session);
    auto r =
        crispasr_session_transcribe_vad_lang(g_asr_session, pcm.data(), pcm.size(), sampleRate, model.c_str(),
                                             hasOptions ? &p : nullptr, language.empty() ? nullptr : language.c_str());
    if (!r)
        throw std::runtime_error("VAD transcription failed");
    std::unique_ptr<crispasr_session_result, decltype(&crispasr_session_result_free)> owned(
        r, crispasr_session_result_free);
    return binding_encode_segments(binding_copy_segments(r), 1.0);
}
static void binding_register_vad() {
    emscripten::function("asrTranscribeVad", &binding_transcribe_vad);
}
