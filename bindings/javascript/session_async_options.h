#pragma once
#ifdef __EMSCRIPTEN_PTHREADS__
static void binding_open_options_async(bool asr, const std::string& model, const std::string& backend,
                                       const emscripten::val& options, emscripten::val callback) {
    binding_check_reentry();
    binding_validate_open_path(model, backend);
    auto params = binding_open_options(options);
    browser_dispatch<bool>(
        [asr, model, backend, params]() {
            return binding_native_open_options(asr ? g_asr_session : g_tts_session, model, backend, params);
        },
        [](bool& opened) { return emscripten::val(opened); }, std::move(callback));
}
static void binding_register_async_options() {
    using namespace emscripten;
    function("asrOpenWithOptionsAsync", optional_override([](const std::string& model, const std::string& backend,
                                                             const val& options, val callback) {
                 binding_open_options_async(true, model, backend, options, std::move(callback));
             }));
    function("ttsOpenWithOptionsAsync", optional_override([](const std::string& model, const std::string& backend,
                                                             const val& options, val callback) {
                 binding_open_options_async(false, model, backend, options, std::move(callback));
             }));
}
#else
static void binding_register_async_options() {}
#endif
