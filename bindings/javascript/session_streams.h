#pragma once
#include "session_guard.h"
#include "session_lifetime.h"
#include "session_options.h"
#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

struct binding_stream_options {
    int threads, step, length, keep, translate;
    std::string language;
};
static binding_stream_options binding_parse_stream_options(const emscripten::val& input) {
    using emscripten::val;
    auto options = input;
    if (options.isNull() || options.isUndefined())
        options = val::object();
    if (options.typeOf().as<std::string>() != "object" || val::global("Array").call<bool>("isArray", options))
        throw std::invalid_argument("Stream options must be an object");
#ifdef __EMSCRIPTEN_PTHREADS__
    const int defaults = 4;
#else
    const int defaults = 1;
#endif
    binding_stream_options p{
        binding_option_integer(options, "n_threads", defaults), binding_option_integer(options, "step_ms", 3000),
        binding_option_integer(options, "length_ms", 10000),    binding_option_integer(options, "keep_ms", 200),
        binding_option_boolean(options, "translate", false),    ""};
    auto lang = options["language"];
    if (!lang.isUndefined() && !lang.isNull()) {
        if (lang.typeOf().as<std::string>() != "string")
            throw std::invalid_argument("Stream language must be a string");
        p.language = lang.as<std::string>();
    }
    // Milliseconds are converted to int32 sample counts by native rolling windows.
    if (p.threads < 1 || p.step < 1 || p.length < p.step || p.keep < 0 || int64_t(p.length) + p.keep > INT_MAX / 16 ||
        p.language.find('\0') != std::string::npos)
        throw std::invalid_argument("Invalid stream options");
#ifndef __EMSCRIPTEN_PTHREADS__
    if (p.threads != 1)
        throw std::invalid_argument("Single-thread module requires n_threads=1");
#endif
    return p;
}
static int binding_native_stream_open(crispasr_session* parent, const binding_stream_options& p) {
    if (!parent)
        throw std::runtime_error("ASR session is closed");
    binding_require_no_stream(parent);
    std::unique_ptr<crispasr_stream, decltype(&crispasr_stream_close)> stream(
        crispasr_session_stream_open(parent, p.threads, p.step, p.length, p.keep,
                                     p.language.empty() ? nullptr : p.language.c_str(), p.translate),
        crispasr_stream_close);
    if (!stream)
        throw std::runtime_error("Native ASR stream open failed");
    std::lock_guard<std::mutex> lock(binding_stream_mutex);
    if (binding_next_stream == INT_MAX)
        throw std::runtime_error("Stream ticket space exhausted");
    int ticket = binding_next_stream++;
    binding_streams.emplace(ticket, binding_stream_entry{parent, stream.get()});
    stream.release();
    return ticket;
}
static crispasr_stream* binding_native_stream(int ticket) {
    std::lock_guard<std::mutex> lock(binding_stream_mutex);
    auto it = binding_streams.find(ticket);
    if (it == binding_streams.end())
        throw std::runtime_error("ASR stream is closed or unknown");
    return it->second.handle;
}
static int binding_native_stream_close(int ticket) {
    crispasr_stream* stream = nullptr;
    {
        std::lock_guard<std::mutex> lock(binding_stream_mutex);
        auto it = binding_streams.find(ticket);
        if (it == binding_streams.end())
            return 0;
        stream = it->second.handle;
        binding_streams.erase(it);
    }
    crispasr_stream_close(stream);
    return 0;
}
struct binding_stream_text {
    std::string text;
    double t0 = 0, t1 = 0;
    int64_t counter = 0;
};
static binding_stream_text binding_native_stream_text(int ticket) {
    auto* stream = binding_native_stream(ticket);
    binding_stream_text out;
    // Native get_text returns the full byte length even when truncated. Retry
    // without consuming output; cap untrusted native growth at 16 MiB.
    std::vector<char> bytes(256);
    for (int retry = 0; retry < 16; ++retry) {
        std::fill(bytes.begin(), bytes.end(), '\0');
        int n = crispasr_stream_get_text(stream, bytes.data(), bytes.size(), &out.t0, &out.t1, &out.counter);
        if (n < 0 || n >= 16 * 1024 * 1024)
            throw std::runtime_error("Invalid native stream text length");
        if (size_t(n) < bytes.size()) {
            if (bytes[n] != '\0' || std::memchr(bytes.data(), '\0', n))
                throw std::runtime_error("Invalid native stream text buffer");
            if (out.counter < 0 || out.counter > 9007199254740991LL || !std::isfinite(out.t0) || !std::isfinite(out.t1))
                throw std::runtime_error("Invalid native stream metadata");
            out.text.assign(bytes.data(), n);
            return out;
        }
        bytes.resize(size_t(n) + 1);
    }
    throw std::runtime_error("Native stream text keeps growing");
}
// A literal JS object avoids user setters on Object.prototype during encoding.
// clang-format off
EM_JS(emscripten::EM_VAL, binding_stream_text_encode, (const char* text, double t0, double t1, double counter), {
    return Emval.toHandle({text: UTF8ToString(text >>> 0), t0: t0, t1: t1, counter: counter});
});
// clang-format on
static emscripten::val binding_encode_stream_text(binding_stream_text& text) {
    return emscripten::val::take_ownership(
        binding_stream_text_encode(text.text.c_str(), text.t0, text.t1, double(text.counter)));
}
static void binding_stream_status(int rc) {
    if (rc < 0)
        throw std::runtime_error("Native ASR stream operation failed: " + std::to_string(rc));
}
// JS input access can invoke getters. Re-check scheduling/parent identity after
// all coercions, before the first native access or queued submission.
static binding_stream_options binding_stream_input(const emscripten::val& options) {
    binding_check_reentry();
    auto* parent = g_asr_session;
    auto epoch = binding_session_epoch.load();
    auto p = binding_parse_stream_options(options);
    binding_check_reentry();
    if (parent != g_asr_session || epoch != binding_session_epoch.load())
        throw std::runtime_error("ASR session changed during stream option conversion");
    return p;
}
// Catch JS throws inside JS: legacy Emscripten EH does not unwind C++ scratch.
// clang-format off
EM_JS(int, binding_stream_pcm_length, (emscripten::EM_VAL handle), {
    try {
        var audio = Emval.toValue(handle);
        if (!(audio instanceof Float32Array)) return -1;
        var getLength = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(Float32Array.prototype), 'length').get;
        var n = getLength.call(audio);
        return n > 0 && n <= 2147483647 ? n : -1;
    } catch (_) { return -1; }
});
EM_JS(int, binding_stream_pcm_copy, (emscripten::EM_VAL handle, float* pcm, int n), {
    try {
        HEAPF32.subarray(pcm >>> 2, (pcm >>> 2) + n).set(Emval.toValue(handle));
        return 1;
    } catch (_) { return 0; }
});
// clang-format on
static std::vector<float> binding_stream_pcm_input(const emscripten::val& audio) {
    int n = binding_stream_pcm_length(audio.as_handle());
    if (n < 1)
        throw std::invalid_argument("Stream PCM must be a nonempty Float32Array");
    binding_check_reentry();
    std::vector<float> pcm(size_t(n), 0.0f);
    if (!binding_stream_pcm_copy(audio.as_handle(), pcm.data(), n))
        throw std::runtime_error("Stream PCM copy failed");
    binding_check_reentry();
    return pcm;
}
static int binding_stream_ticket(double value) {
    if (!std::isfinite(value) || value < 1 || value >= INT_MAX || value != std::trunc(value))
        throw std::invalid_argument("Invalid stream ticket");
    return int(value);
}
static void binding_require_sync_stream() {
    binding_check_reentry();
#ifdef __EMSCRIPTEN_PTHREADS__
    throw std::runtime_error("Threaded modules require the asrStream*Async methods");
#endif
}
static emscripten::val binding_stream_encode_number(int& value) {
    return emscripten::val(value);
}
static void binding_register_streams() {
    using namespace emscripten;
    function("asrStreamKind", optional_override([]() {
                 binding_check_reentry();
                 return g_asr_session ? crispasr_session_stream_kind(g_asr_session) : 0;
             }));
    function("asrStreamOpen", optional_override([](const val& options) {
                 binding_require_sync_stream();
                 auto p = binding_stream_input(options);
                 return binding_native_stream_open(g_asr_session, p);
             }));
    function("asrStreamClose", optional_override([](double ticketValue) {
                 int ticket = binding_stream_ticket(ticketValue);
                 binding_require_sync_stream();
                 return binding_native_stream_close(ticket);
             }));
    function("asrStreamFeed", optional_override([](double ticketValue, const val& audio) {
                 int ticket = binding_stream_ticket(ticketValue);
                 binding_require_sync_stream();
                 auto pcm = binding_stream_pcm_input(audio);
                 int rc = crispasr_stream_feed(binding_native_stream(ticket), pcm.data(), pcm.size());
                 binding_stream_status(rc);
                 return rc;
             }));
    function("asrStreamFlush", optional_override([](double ticketValue) {
                 int ticket = binding_stream_ticket(ticketValue);
                 binding_require_sync_stream();
                 int rc = crispasr_stream_flush(binding_native_stream(ticket));
                 binding_stream_status(rc);
                 return rc;
             }));
    function("asrStreamGetText", optional_override([](double ticketValue) {
                 int ticket = binding_stream_ticket(ticketValue);
                 binding_require_sync_stream();
                 auto text = binding_native_stream_text(ticket);
                 return binding_encode_stream_text(text);
             }));
    function("asrStreamSetLiveDecode", optional_override([](double ticketValue, bool enabled) {
                 int ticket = binding_stream_ticket(ticketValue);
                 binding_require_sync_stream();
                 crispasr_stream_set_live_decode(binding_native_stream(ticket), enabled);
             }));
#ifdef __EMSCRIPTEN_PTHREADS__

    function("asrStreamOpenAsync", optional_override([](const val& options, val callback) {
                 auto p = binding_stream_input(options);
                 browser_dispatch<int>([p]() { return binding_native_stream_open(g_asr_session, p); },
                                       binding_stream_encode_number, callback);
             }));
    function("asrStreamCloseAsync", optional_override([](double ticketValue, val callback) {
                 int ticket = binding_stream_ticket(ticketValue);
                 browser_dispatch<int>([ticket]() { return binding_native_stream_close(ticket); },
                                       binding_stream_encode_number, callback);
             }));
    function("asrStreamFeedAsync", optional_override([](double ticketValue, const val& audio, val callback) {
                 int ticket = binding_stream_ticket(ticketValue);
                 binding_check_reentry();
                 auto pcm = binding_stream_pcm_input(audio);
                 browser_dispatch<int>(
                     [ticket, pcm = std::move(pcm)]() {
                         int rc = crispasr_stream_feed(binding_native_stream(ticket), pcm.data(), pcm.size());
                         binding_stream_status(rc);
                         return rc;
                     },
                     binding_stream_encode_number, callback);
             }));
    function("asrStreamFlushAsync", optional_override([](double ticketValue, val callback) {
                 int ticket = binding_stream_ticket(ticketValue);
                 browser_dispatch<int>(
                     [ticket]() {
                         int rc = crispasr_stream_flush(binding_native_stream(ticket));
                         binding_stream_status(rc);
                         return rc;
                     },
                     binding_stream_encode_number, callback);
             }));
    function("asrStreamGetTextAsync", optional_override([](double ticketValue, val callback) {
                 int ticket = binding_stream_ticket(ticketValue);
                 browser_dispatch<binding_stream_text>([ticket]() { return binding_native_stream_text(ticket); },
                                                       binding_encode_stream_text, callback);
             }));
    function("asrStreamSetLiveDecodeAsync", optional_override([](double ticketValue, bool enabled, val callback) {
                 int ticket = binding_stream_ticket(ticketValue);
                 browser_dispatch<int>(
                     [ticket, enabled]() {
                         crispasr_stream_set_live_decode(binding_native_stream(ticket), enabled);
                         return 0;
                     },
                     binding_stream_encode_number, callback);
             }));
    function("asrCloseAsync", optional_override([](val callback) {
                 browser_dispatch<int>(
                     []() {
                         binding_native_close_session(g_asr_session);
                         return 0;
                     },
                     binding_stream_encode_number, callback);
             }));
#endif
}
