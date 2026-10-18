#pragma once
#include "crispasr_session.h"
#include "session_asr_events_native.h"
#include <atomic>
#include <climits>
#include <map>
#include <mutex>
#include <stdexcept>
#ifdef __EMSCRIPTEN_PTHREADS__
#include <emscripten/threading.h>
#endif

// Native-only ownership: tickets never contain addresses and are never reused.
// The dispatcher excludes servicer operations while compute mutates this table.
struct binding_stream_entry {
    crispasr_session* parent;
    crispasr_stream* handle;
};
static std::mutex binding_stream_mutex;
static std::map<int, binding_stream_entry> binding_streams;
static int binding_next_stream = 1;
static std::atomic<uint64_t> binding_session_epoch{0};
static bool binding_has_stream(crispasr_session* parent) {
    std::lock_guard<std::mutex> lock(binding_stream_mutex);
    for (const auto& item : binding_streams)
        if (item.second.parent == parent)
            return true;
    return false;
}
static void binding_require_no_stream(crispasr_session* parent) {
    if (parent && binding_has_stream(parent))
        throw std::runtime_error("Close the active ASR stream before batch inference");
}
static void binding_native_close_session(crispasr_session*& session) {
    if (!session)
        return;
#ifdef __EMSCRIPTEN_PTHREADS__
    // Backend stream release can join native decoder threads. Never do that on
    // the browser servicer. Keep the session intact when refusing this route.
    if (binding_has_stream(session) && emscripten_is_main_runtime_thread())
        throw std::runtime_error("Use asrCloseAsync or an async open to close a streamed session");
#endif
    binding_asr_clear_events(session);
    auto* parent = session;
    session = nullptr;
    ++binding_session_epoch;
    crispasr_stream* child = nullptr;
    {
        std::lock_guard<std::mutex> lock(binding_stream_mutex);
        for (auto it = binding_streams.begin(); it != binding_streams.end();) {
            if (it->second.parent == parent) {
                child = it->second.handle;
                it = binding_streams.erase(it);
            } else {
                ++it;
            }
        }
    }
    // Only one retained stream per parent is supported by the native ABI.
    if (child)
        crispasr_stream_close(child);
    crispasr_session_close(parent);
}
