#pragma once
#include "session_guard.h"
#include "session_lifetime.h"
#include <emscripten/bind.h>

// JS functions and callback errors stay on the creating servicer. No retained
// emval or JS callback ever enters native compute or a backend producer thread.
// clang-format off
EM_JS(int, binding_asr_callback_valid, (emscripten::EM_VAL handle), {
    var callback = Emval.toValue(handle);
    return callback == null || typeof callback === 'function';
});
EM_JS(int, binding_asr_callback_store, (int kind, int version, emscripten::EM_VAL handle), {
    try {
        var state = Module['crispasrAsrCallbacks'];
        if (!state) state = Module['crispasrAsrCallbacks'] = new Array(3).fill(null);
        state[kind] = version ? {version: version, callback: Emval.toValue(handle)} : null;
        return 1;
    } catch (error) {
        try { Module['crispasrLastCallbackError'] = error; } catch (_) {}
        return 0;
    }
});
EM_JS(void, binding_asr_event_deliver_js,
      (int kind, int version, const char* text, int n, int index, double t0, double t1, int processed, int total), {
    var state = Module['crispasrAsrCallbacks'];
    var slot = state && state[kind];
    if (!slot || slot.version !== version) return;
    var previous = Module['crispasrInCallback'];
    Module['crispasrInCallback'] = true;
    try {
        var event;
        text = text >>> 0;
        if (kind === 0) event = {text: UTF8ToString(text), t0: t0, t1: t1, index: index};
        else if (kind === 1) event = {rawText: HEAPU8.slice(text, text + n), index: index};
        else event = {processed: processed, total: total};
        slot.callback(event);
    } catch (error) {
        Module['crispasrLastCallbackError'] = error;
    } finally {
        Module['crispasrInCallback'] = previous;
    }
});
// clang-format on
static void binding_asr_event_deliver(uintptr_t raw) {
    auto& event = *reinterpret_cast<binding_asr_event*>(raw);
    binding_asr_event_deliver_js(event.kind, event.version, event.text.c_str(), event.text.size(), event.index,
                                 double(event.t0), double(event.t1), event.processed, event.total);
}
// Every popped native event has a JS finally; JS throws do not unwind legacy
// Emscripten C++ frames. This also protects against unexpected encoder throws.
// clang-format off
EM_JS(int, binding_asr_events_drain_js,
      (uintptr_t pop, uintptr_t deliver, uintptr_t destroy, uintptr_t getVersion, uintptr_t takeError), {
    var state = Module['crispasrAsrCallbacks'];
    if (!state) return 0;
    try {
        for (var kind = 0; kind < 3; ++kind) {
            if (state[kind] && state[kind].version !== wasmTable.get(getVersion)(kind)) state[kind] = null;
        }
        if (wasmTable.get(takeError)())
            Module['crispasrLastCallbackError'] = new Error('ASR callback queue overflow, invalid event, or allocation failure; events were lost');
        for (var i = 0; i < 4096; ++i) {
            var event = wasmTable.get(pop)();
            if (!event) break;
            try { wasmTable.get(deliver)(event); }
            catch (error) { Module['crispasrLastCallbackError'] = error; }
            finally { wasmTable.get(destroy)(event); }
        }
        // A failed pop can raise the native error flag during this drain.
        if (wasmTable.get(takeError)())
            Module['crispasrLastCallbackError'] = new Error('ASR callback queue overflow, invalid event, or allocation failure; events were lost');
    } catch (error) {
        Module['crispasrLastCallbackError'] = error;
    }
    return state[0] !== null || state[1] !== null || state[2] !== null;
});
// clang-format on
static int binding_asr_events_drain() {
    return binding_asr_events_drain_js(
        reinterpret_cast<uintptr_t>(binding_asr_event_pop), reinterpret_cast<uintptr_t>(binding_asr_event_deliver),
        reinterpret_cast<uintptr_t>(binding_asr_event_destroy), reinterpret_cast<uintptr_t>(binding_asr_event_version),
        reinterpret_cast<uintptr_t>(binding_asr_event_take_error));
}
static bool binding_asr_events_timer_armed = false;
static void binding_asr_events_poll(void*) {
    if (binding_asr_events_drain()) {
        emscripten_async_call(binding_asr_events_poll, nullptr, 10);
    } else {
        binding_asr_events_timer_armed = false;
    }
}
static void binding_asr_set_callback(int kind, const emscripten::val& callback) {
    binding_check_reentry();
    if (!g_asr_session)
        throw std::runtime_error("ASR session is closed");
    if (binding_has_stream(g_asr_session))
        throw std::runtime_error("Set ASR callbacks before opening a retained stream");
    if (!binding_asr_callback_valid(callback.as_handle()))
        throw std::invalid_argument("ASR callback must be a function or null");
    bool enabled = !callback.isNull() && !callback.isUndefined();
    int version = binding_asr_install_event(g_asr_session, kind, enabled);
    if (!binding_asr_callback_store(kind, version, callback.as_handle())) {
        binding_asr_install_event(g_asr_session, kind, false);
        throw std::runtime_error("Could not retain ASR callback on the servicer");
    }
    if (!binding_asr_events_timer_armed) {
        binding_asr_events_timer_armed = true;
        emscripten_async_call(binding_asr_events_poll, nullptr, 0);
    }
}
static void binding_register_asr_callbacks() {
    using namespace emscripten;
    function("asrSetSegmentCallback", optional_override([](const val& cb) { binding_asr_set_callback(0, cb); }));
    function("asrSetTokenCallback", optional_override([](const val& cb) { binding_asr_set_callback(1, cb); }));
    function("asrSetProgressCallback", optional_override([](const val& cb) { binding_asr_set_callback(2, cb); }));
    function("asrDispatchEvents", optional_override([]() {
                 // Deliberately allowed while compute is active: only copied
                 // events are drained, never model state or borrowed buffers.
                 binding_check_callback();
                 binding_asr_events_drain();
             }));
}
