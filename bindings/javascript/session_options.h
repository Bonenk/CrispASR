#pragma once
#include "crispasr_session.h"
#include "session_lifetime.h"
#include <cmath>
#include <climits>
#include <cstddef>

// Private ABI POD; operation probes compile the literal runtime declaration.
struct crispasr_open_params_v1 {
    int abi_version;
    int n_threads;
    int use_gpu;
    int verbosity;
    int flash_attn;
    int n_gpu_layers;
    int reserved[6];
};
static_assert(sizeof(crispasr_open_params_v1) == 48, "Open options ABI changed");
static_assert(offsetof(crispasr_open_params_v1, flash_attn) == 16, "Open options offset changed");
static int binding_option_integer(const emscripten::val& options, const char* key, int fallback) {
    auto value = options[key];
    if (value.isUndefined())
        return fallback;
    if (value.typeOf().as<std::string>() != "number")
        throw std::invalid_argument(std::string(key) + " must be an integer");
    double n = value.as<double>();
    if (!std::isfinite(n) || n < INT_MIN || n > INT_MAX || n != std::trunc(n))
        throw std::invalid_argument(std::string(key) + " must be an int32");
    return int(n);
}
static int binding_option_boolean(const emscripten::val& options, const char* key, bool fallback) {
    auto value = options[key];
    if (value.isUndefined())
        return fallback;
    if (!value.isTrue() && !value.isFalse())
        throw std::invalid_argument(std::string(key) + " must be boolean");
    return value.isTrue();
}
static crispasr_open_params_v1 binding_open_options(const emscripten::val& options) {
    using emscripten::val;
    auto input = options;
    if (input.isNull() || input.isUndefined())
        input = val::object();
    if (input.typeOf().as<std::string>() != "object" || val::global("Array").call<bool>("isArray", input))
        throw std::invalid_argument("Options must be an object");
#ifdef __EMSCRIPTEN_PTHREADS__
    const int defaults = 4;
#else
    const int defaults = 1;
#endif
    crispasr_open_params_v1 p{};
    p.abi_version = 2;
    p.n_threads = binding_option_integer(input, "n_threads", defaults);
    if (p.n_threads == 0)
        p.n_threads = defaults;
    p.use_gpu = binding_option_boolean(input, "use_gpu", true);
    p.verbosity = binding_option_integer(input, "verbosity", 0);
    p.flash_attn = binding_option_boolean(input, "flash_attn", true);
    p.n_gpu_layers = binding_option_integer(input, "n_gpu_layers", -1);
    if (p.n_threads < 1 || p.verbosity < 0 || p.n_gpu_layers < -1)
        throw std::invalid_argument("Invalid load options");
#ifndef __EMSCRIPTEN_PTHREADS__
    if (p.n_threads != 1)
        throw std::invalid_argument("Single-thread module requires n_threads=1");
#endif
    return p;
}
static void binding_validate_open_path(const std::string& model, const std::string& backend) {
    if (model.empty() || model.find('\0') != std::string::npos || backend.find('\0') != std::string::npos)
        throw std::invalid_argument("Invalid model path or backend");
}
// Plain native inputs only: usable on compute without creating any emval.
static bool binding_native_open_options(crispasr_session*& session, const std::string& model,
                                        const std::string& backend, const crispasr_open_params_v1& params) {
    binding_native_close_session(session);
    session = crispasr_session_open_with_params(model.c_str(), backend.empty() ? nullptr : backend.c_str(), &params);
    return session != nullptr;
}
static bool binding_open_with_options(crispasr_session*& session, const std::string& model, const std::string& backend,
                                      const emscripten::val& options) {
    binding_check_reentry();
    binding_validate_open_path(model, backend);
    // Validate before replacing a live singleton; close first bounds model memory.
    auto params = binding_open_options(options);
    binding_check_reentry();
    return binding_native_open_options(session, model, backend, params);
}
static void binding_register_open_options() {
    using namespace emscripten;
    function("asrOpenWithOptions",
             optional_override([](const std::string& model, const std::string& backend, const val& options) {
                 return binding_open_with_options(g_asr_session, model, backend, options);
             }));
    function("ttsOpenWithOptions",
             optional_override([](const std::string& model, const std::string& backend, const val& options) {
                 return binding_open_with_options(g_tts_session, model, backend, options);
             }));
}
