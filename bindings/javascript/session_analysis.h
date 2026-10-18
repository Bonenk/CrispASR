#pragma once
#include "crispasr_session.h"
#include <memory>
#include <cmath>
#include <stdexcept>

// Alignment uses native centiseconds; VAD returns native seconds.
static emscripten::val binding_alignment(const char* text, int64_t t0, int64_t t1) {
    auto out = emscripten::val::object();
    out.set("text", std::string(text ? text : ""));
    out.set("t0", double(t0));
    out.set("t1", double(t1));
    out.set("hasTimestamps", t0 >= 0 && t1 >= t0);
    return out;
}
class BindingParakeet {
    parakeet_context* handle = nullptr;

public:
    BindingParakeet(const std::string& model, int threads, bool flash) {
        binding_check_reentry();
        handle = crispasr_parakeet_init(model.c_str(), threads, flash ? 1 : 0);
        if (!handle)
            throw std::runtime_error("Parakeet model could not open");
    }
    BindingParakeet(const BindingParakeet&) = delete;
    ~BindingParakeet() {
        if (handle)
            crispasr_parakeet_free(handle);
    }
    void close() {
        binding_check_reentry();
        if (handle) {
            crispasr_parakeet_free(handle);
            handle = nullptr;
        }
    }
    emscripten::val transcribe(const emscripten::val& audio, double offsetCs) {
        binding_check_reentry();
        if (!handle)
            throw std::runtime_error("Parakeet context is closed");
        auto pcm = binding_pcm(audio);
        if (!std::isfinite(offsetCs) || offsetCs < double(INT64_MIN) || offsetCs >= double(INT64_MAX))
            throw std::invalid_argument("Invalid centisecond offset");
        std::unique_ptr<parakeet_result, decltype(&crispasr_parakeet_result_free)> result(
            crispasr_parakeet_transcribe(handle, pcm.data(), pcm.size(), int64_t(offsetCs)),
            crispasr_parakeet_result_free);
        if (!result)
            throw std::runtime_error("Parakeet transcription failed");
        auto r = result.get();
        auto out = emscripten::val::object(), words = emscripten::val::array(), tokens = emscripten::val::array();
        out.set("text", binding_string(crispasr_parakeet_result_text(r)));
        int nw = crispasr_parakeet_result_n_words(r), nt = crispasr_parakeet_result_n_tokens(r);
        if (nw < 0 || nt < 0)
            throw std::runtime_error("Invalid Parakeet count");
        for (int j = 0; j < nw; ++j) {
            binding_piece piece;
            piece.text = binding_string(crispasr_parakeet_result_word_text(r, j));
            piece.t0 = crispasr_parakeet_result_word_t0(r, j);
            piece.t1 = crispasr_parakeet_result_word_t1(r, j);
            words.call<void>("push", binding_encode_piece(piece, 1.0));
        }
        for (int j = 0; j < nt; ++j) {
            binding_piece piece;
            piece.text = binding_string(crispasr_parakeet_result_token_text(r, j));
            piece.t0 = crispasr_parakeet_result_token_t0(r, j);
            piece.t1 = crispasr_parakeet_result_token_t1(r, j);
            piece.p = crispasr_parakeet_result_token_p(r, j);
            tokens.call<void>("push", binding_encode_piece(piece, 1.0, true));
        }
        out.set("words", words);
        out.set("tokens", tokens);
        return out;
    }
};
class BindingTitaNet {
    void* handle = nullptr;

public:
    BindingTitaNet(const std::string& model, int threads) {
        binding_check_reentry();
        handle = crispasr_titanet_init(model.c_str(), threads);
        if (!handle)
            throw std::runtime_error("TitaNet model could not open");
    }
    BindingTitaNet(const BindingTitaNet&) = delete;
    ~BindingTitaNet() {
        if (handle)
            crispasr_titanet_free(handle);
    }
    void close() {
        binding_check_reentry();
        if (handle) {
            crispasr_titanet_free(handle);
            handle = nullptr;
        }
    }
    emscripten::val embed(const emscripten::val& audio) {
        binding_check_reentry();
        if (!handle)
            throw std::runtime_error("TitaNet context is closed");
        auto pcm = binding_pcm(audio);
        std::vector<float> output(192);
        int n = crispasr_titanet_embed(handle, pcm.data(), pcm.size(), output.data());
        if (n != 192)
            throw std::runtime_error("TitaNet must return 192 floats");
        return binding_floats(output.data(), output.size());
    }
};
static void binding_register_analysis() {
    using namespace emscripten;
    class_<BindingTitaNet>("TitaNet")
        .constructor<const std::string&, int>()
        .function("embed", &BindingTitaNet::embed)
        .function("close", &BindingTitaNet::close);
    function("titanetCosineSimilarity", optional_override([](const val& left, const val& right) {
                 binding_check_reentry();
                 auto a = binding_pcm(left, true), b = binding_pcm(right, true);
                 if (a.size() != b.size())
                     throw std::invalid_argument("Embedding dimensions must match");
                 return crispasr_titanet_cosine_sim(a.data(), b.data(), a.size());
             }));
    class_<BindingParakeet>("ParakeetContext")
        .constructor<const std::string&, int, bool>()
        .function("transcribe", &BindingParakeet::transcribe)
        .function("close", &BindingParakeet::close);
    function("alignWords", optional_override([](const val& audio, const std::string& model,
                                                const std::string& transcript, double offsetCs, int threads) {
                 binding_check_reentry();
                 auto pcm = binding_pcm(audio);
                 if (!std::isfinite(offsetCs) || offsetCs < double(INT64_MIN) || offsetCs >= double(INT64_MAX))
                     throw std::invalid_argument("Invalid centisecond offset");
                 std::unique_ptr<crispasr_align_result, decltype(&crispasr_align_result_free)> result(
                     crispasr_align_words_abi(model.c_str(), transcript.c_str(), pcm.data(), pcm.size(),
                                              int64_t(offsetCs), threads),
                     crispasr_align_result_free);
                 if (!result)
                     throw std::runtime_error("Alignment failed");
                 auto out = val::array();
                 int n = crispasr_align_result_n_words(result.get());
                 if (n < 0)
                     throw std::runtime_error("Invalid aligned word count");
                 for (int i = 0; i < n; ++i) {
                     auto word = binding_alignment(crispasr_align_result_word_text(result.get(), i),
                                                   crispasr_align_result_word_t0(result.get(), i),
                                                   crispasr_align_result_word_t1(result.get(), i));
                     auto characters = val::array();
                     int count = crispasr_align_result_n_characters(result.get(), i);
                     if (count < 0)
                         throw std::runtime_error("Invalid aligned character count");
                     for (int j = 0; j < count; ++j)
                         characters.call<void>(
                             "push", binding_alignment(crispasr_align_result_character_text(result.get(), i, j),
                                                       crispasr_align_result_character_t0(result.get(), i, j),
                                                       crispasr_align_result_character_t1(result.get(), i, j)));
                     word.set("characters", characters);
                     out.call<void>("push", word);
                 }
                 return out;
             }));
    function("vadSpans",
             optional_override([](const val& audio, const std::string& model, int sampleRate, float threshold,
                                  int minSpeechMs, int minSilenceMs, int threads, bool gpu) {
                 binding_check_reentry();
                 auto pcm = binding_pcm(audio);
                 float* spans = nullptr;
                 int n = crispasr_vad_segments(model.c_str(), pcm.data(), pcm.size(), sampleRate, threshold,
                                               minSpeechMs, minSilenceMs, threads, gpu, &spans);
                 std::unique_ptr<float, decltype(&crispasr_vad_free)> owned(spans, crispasr_vad_free);
                 if (n < 0 || (n && !spans))
                     throw std::runtime_error("VAD failed");
                 auto out = val::array();
                 for (int i = 0; i < n; ++i) {
                     auto span = val::object();
                     span.set("start", spans[2 * i]);
                     span.set("end", spans[2 * i + 1]);
                     out.call<void>("push", span);
                 }
                 return out;
             }));
    function("vadSlices", optional_override([](const val& audio, const std::string& model, int sampleRate,
                                               float threshold, int minSpeechMs, int minSilenceMs, int paddingMs,
                                               float maxChunkSeconds, int threads) {
                 binding_check_reentry();
                 auto pcm = binding_pcm(audio);
                 float* spans = nullptr;
                 int n = crispasr_vad_slices(model.c_str(), pcm.data(), pcm.size(), sampleRate, threshold, minSpeechMs,
                                             minSilenceMs, paddingMs, maxChunkSeconds, threads, &spans);
                 std::unique_ptr<float, decltype(&crispasr_vad_free)> owned(spans, crispasr_vad_free);
                 if (n < 0 || (n && !spans))
                     throw std::runtime_error("VAD slices failed");
                 auto out = val::array();
                 for (int i = 0; i < n; ++i) {
                     auto span = val::object();
                     span.set("start", spans[2 * i]);
                     span.set("end", spans[2 * i + 1]);
                     out.call<void>("push", span);
                 }
                 return out;
             }));
}
