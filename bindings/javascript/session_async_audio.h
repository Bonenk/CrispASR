#pragma once
#include <memory>
#include <vector>
#ifdef __EMSCRIPTEN_PTHREADS__
// Catch JS allocation/constructor errors before they can skip C++ emval
// temporaries. The copied output (or error object) owns its JS memory.
// clang-format off
EM_JS(uintptr_t, binding_async_audio_encode, (const float* pcm, int count), {
    try { return Emval.toHandle(new Float32Array(HEAPF32.subarray(pcm >> 2, (pcm >> 2) + count))); }
    catch (error) { return Emval.toHandle({error: error && error.message ? error.message : String(error)}); }
});
// clang-format on
static std::vector<float> binding_synthesize_owned(const std::string& text) {
    int n = 0;
    float* pcm = g_tts_session ? crispasr_session_synthesize(g_tts_session, text.c_str(), &n) : nullptr;
    std::unique_ptr<float, decltype(&crispasr_pcm_free)> owned(pcm, crispasr_pcm_free);
    if (n < 0 || (n && !pcm))
        throw std::runtime_error("Invalid native synthesis buffer");
    return n ? std::vector<float>(pcm, pcm + n) : std::vector<float>{};
}
static void binding_register_async_audio() {
    using namespace emscripten;
    function("ttsSynthesizeAsync", optional_override([](const std::string& text, val callback) {
                 binding_check_reentry();
                 if (text.find('\0') != std::string::npos)
                     throw std::invalid_argument("Embedded NUL in synthesis text");
                 browser_dispatch<std::vector<float>>([text]() { return binding_synthesize_owned(text); },
                                                      [](std::vector<float>& pcm) {
                                                          return val::take_ownership(reinterpret_cast<EM_VAL>(
                                                              binding_async_audio_encode(pcm.data(), pcm.size())));
                                                      },
                                                      std::move(callback));
             }));
}
#else
static void binding_register_async_audio() {}
#endif
