#pragma once
#include "crispasr_session.h"
#include <emscripten.h>
#include <emscripten/bind.h>
#include <climits>
#include <vector>
#include <stdexcept>

#include "session_guard.h"
// Catch JS exceptions on the JS side, before they can cross native synthesis.
EM_JS(void, binding_pcm_deliver, (uintptr_t handle, const float* pcm, int n, int final), {
    var state = Emval.toValue(handle);
    if (state.failed)
        return;
    Module['crispasrInCallback'] = true;
    try {
        state.callback(HEAPF32.slice(pcm >> 2, (pcm >> 2) + n), !!final);
    } catch (e) {
        state.failed = true;
        state.error = e;
    } finally {
        Module['crispasrInCallback'] = false;
    }
});
static std::vector<float> binding_pcm(const emscripten::val& audio, bool optional = false) {
    if (optional && (audio.isNull() || audio.isUndefined()))
        return {};
    int n = audio["length"].as<int>();
    if (n <= 0)
        throw std::invalid_argument("nonempty PCM required");
    std::vector<float> out(n);
    emscripten::val(emscripten::typed_memory_view(out.size(), out.data())).call<void>("set", audio);
    return out;
}
static emscripten::val binding_floats(const float* p, size_t n) {
    if (!p && n)
        throw std::runtime_error("null native buffer");
    auto out = emscripten::val::global("Float32Array").new_(n);
    if (n)
        out.call<void>("set", emscripten::val(emscripten::typed_memory_view(n, p)));
    return out;
}
static void binding_require_session() {
    binding_check_reentry();
    if (!g_tts_session)
        throw std::runtime_error("session is closed");
}
static void binding_status(int rc) {
    if (rc < 0)
        throw std::runtime_error("native session operation failed: " + std::to_string(rc));
}
static size_t binding_count(int n, int width) {
    if (n < 0 || width < 0 || (width && n > INT_MAX / width))
        throw std::runtime_error("invalid native dimensions");
    return size_t(n) * width;
}
static void binding_stream_pcm(const float* p, int n, int final, void* user) {
    auto* state = static_cast<emscripten::val*>(user);
    binding_pcm_deliver(reinterpret_cast<uintptr_t>(state->as_handle()), p, n, final);
}
static void binding_register_session_tasks() {
    using emscripten::val;
    emscripten::function("sessionBeats", emscripten::optional_override([](const val& audio) {
                             binding_require_session();
                             auto pcm = binding_pcm(audio);
                             int rate = crispasr_session_beats_sample_rate(g_tts_session);
                             binding_status(crispasr_session_beats(g_tts_session, pcm.data(), pcm.size(), rate));
                             int n = 0;
                             auto p = crispasr_session_beats_events(g_tts_session, &n);
                             auto out = val::object();
                             out.set("events", binding_floats(p, binding_count(n, 2)));
                             out.set("sampleRate", rate);
                             out.set("tempoBpm", crispasr_session_beats_tempo_bpm(g_tts_session));
                             return out;
                         }));
    emscripten::function("sessionTab", emscripten::optional_override([](const val& audio, int rate) {
                             binding_require_session();
                             auto pcm = binding_pcm(audio);
                             binding_status(crispasr_session_tab(g_tts_session, pcm.data(), pcm.size(), rate));
                             int n = 0, strings = 0, classes = 0;
                             auto p = crispasr_session_tab_emissions(g_tts_session, &n, &strings, &classes);
                             auto out = val::object();
                             size_t rows = binding_count(n, strings);
                             if (classes < 0 || (classes && rows > INT_MAX / classes))
                                 throw std::runtime_error("invalid tab dimensions");
                             out.set("emissions", binding_floats(p, rows * classes));
                             out.set("nFrames", n);
                             out.set("nStrings", strings);
                             out.set("nClasses", classes);
                             out.set("silentClass", crispasr_session_tab_silent_class(g_tts_session));
                             out.set("framePeriod", crispasr_session_tab_frame_period(g_tts_session));
                             auto midi = val::array();
                             for (int i = 0; i < strings; ++i)
                                 midi.call<void>("push", crispasr_session_tab_string_open_midi(g_tts_session, i));
                             out.set("openMidi", midi);
                             return out;
                         }));
    emscripten::function(
        "sessionConvert", emscripten::optional_override([](const val& content, const val& f0, int speaker,
                                                           const val& noiseZp, const val& noiseSine) {
            binding_require_session();
            auto c = binding_pcm(content), f = binding_pcm(f0);
            int dim = crispasr_session_convert_content_dim(g_tts_session), n = f.size();
            if (dim <= 0 || c.size() != binding_count(n, dim) || speaker < 0 ||
                speaker >= crispasr_session_convert_n_speakers(g_tts_session))
                throw std::invalid_argument("invalid content/speaker");
            bool hasZp = !noiseZp.isNull() && !noiseZp.isUndefined(),
                 hasSine = !noiseSine.isNull() && !noiseSine.isUndefined();
            auto z = binding_pcm(noiseZp, true), v = binding_pcm(noiseSine, true);
            if (hasZp) {
                int d = crispasr_session_convert_inter_channels(g_tts_session);
                if (d <= 0 || z.size() != binding_count(n, d))
                    throw std::invalid_argument("noiseZp length mismatch");
            }
            if (hasSine) {
                int d = crispasr_session_convert_upsample_product(g_tts_session);
                if (d <= 0 || v.size() != binding_count(n, d))
                    throw std::invalid_argument("noiseSine length mismatch");
            }
            int rc = crispasr_session_convert(g_tts_session, c.data(), n, f.data(), speaker, hasZp ? z.data() : nullptr,
                                              hasSine ? v.data() : nullptr);
            if (rc <= 0)
                throw std::runtime_error("conversion failed");
            int count = 0;
            auto p = crispasr_session_convert_audio(g_tts_session, &count);
            return binding_floats(p, binding_count(count, 1));
        }));
    emscripten::function("sessionSetVoiceSamples",
                         emscripten::optional_override([](const val& audio, int rate, const std::string& text) {
                             binding_require_session();
                             auto pcm = binding_pcm(audio);
                             if (rate <= 0)
                                 throw std::invalid_argument("invalid rate");
                             return crispasr_session_set_voice_samples(g_tts_session, pcm.data(), pcm.size(), rate,
                                                                       text.empty() ? nullptr : text.c_str());
                         }));
    emscripten::function("sessionSetTtsCfgScale", emscripten::optional_override([](float x) {
                             binding_require_session();
                             return crispasr_session_set_tts_cfg_scale(g_tts_session, x);
                         }));
    emscripten::function("sessionTadaSetMakerefModels",
                         emscripten::optional_override([](const std::string& encoder, const std::string& aligner) {
                             binding_require_session();
                             return crispasr_session_tada_set_makeref_models(
                                 g_tts_session, encoder.empty() ? nullptr : encoder.c_str(),
                                 aligner.empty() ? nullptr : aligner.c_str());
                         }));
    emscripten::function("sessionLastSynthError", emscripten::optional_override([]() {
                             binding_require_session();
                             return binding_string(crispasr_session_last_synth_error(g_tts_session));
                         }));
    emscripten::function("sessionDisclaimerText", emscripten::optional_override([]() {
                             return binding_string(crispasr_session_disclaimer_text());
                         }));
    emscripten::function("sessionGetDisclaimerPcm", emscripten::optional_override([]() {
                             binding_require_session();
                             int n = 0;
                             float* p = crispasr_session_get_disclaimer_pcm(g_tts_session, &n);
                             std::unique_ptr<float, decltype(&crispasr_pcm_free)> owned(p, crispasr_pcm_free);
                             if (!p)
                                 throw std::runtime_error("disclaimer unavailable");
                             return binding_floats(p, binding_count(n, 1));
                         }));
    // Synchronous sentence chunks; callback errors are returned after native unwinds.
    // {status, failed, error}; throwing inside callback never unwinds C++ frames.
    emscripten::function(
        "sessionSynthesizeStreaming", emscripten::optional_override([](const std::string& text, const val& callback) {
            binding_require_session();
            auto state = val::object();
            state.set("callback", callback);
            state.set("failed", false);
            state.set("error", val::null());
            int rc = crispasr_session_synthesize_streaming(g_tts_session, text.c_str(), binding_stream_pcm, &state);
            state.set("status", rc);
            state.delete_("callback");
            return state;
        }));
}
