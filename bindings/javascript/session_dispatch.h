#pragma once
#include "session_asr_events.h"
#ifdef __EMSCRIPTEN_PTHREADS__
#include <emscripten.h>
#include <emscripten/bind.h>
#include <atomic>
#include <functional>
#include <memory>
#include <stdexcept>

// JS callback/encoding exceptions do not unwind C++ frames in the default
// Emscripten EH mode. The JS finally below therefore explicitly retires the
// delivery on its creating servicer. Only plain C++ work data reaches compute.
struct binding_browser_delivery {
    emscripten::val callback;
    bool owns_busy = false;
    explicit binding_browser_delivery(emscripten::val cb) : callback(std::move(cb)) {}
    virtual ~binding_browser_delivery() = default;
    virtual bool ready() const = 0;
    virtual emscripten::val encode() = 0;
};
static uintptr_t binding_browser_encode(uintptr_t raw) {
    auto* delivery = reinterpret_cast<binding_browser_delivery*>(raw);
    try {
        return reinterpret_cast<uintptr_t>(delivery->encode().release_ownership());
    } catch (const std::exception& e) {
        auto out = emscripten::val::object();
        out.set("error", std::string(e.what()));
        return reinterpret_cast<uintptr_t>(out.release_ownership());
    } catch (...) {
        auto out = emscripten::val::object();
        out.set("error", std::string("Unknown browser encoding exception"));
        return reinterpret_cast<uintptr_t>(out.release_ownership());
    }
}
static void binding_browser_destroy(uintptr_t raw) {
    delete reinterpret_cast<binding_browser_delivery*>(raw);
}
// clang-format off
EM_JS(void, binding_browser_deliver, (uintptr_t job, uintptr_t encode, uintptr_t destroy, uintptr_t callback), {
    try {
        var output;
        try {
            var handle = wasmTable.get(encode)(job);
            try { output = Emval.toValue(handle); }
            finally { __emval_decref(handle); }
        } catch (error) {
            output = {error: error && error.message ? error.message : String(error)};
        }
        try { Emval.toValue(callback)(output); }
        catch (error) { Module['crispasrLastCallbackError'] = error; }
    } finally {
        wasmTable.get(destroy)(job);
    }
});
// clang-format on
static void binding_browser_poll(void* raw) {
    auto* job = static_cast<binding_browser_delivery*>(raw);
    if (!job->ready()) {
        emscripten_async_call(binding_browser_poll, raw, 1);
        return;
    }
    // Deliver all committed ASR events before the operation completion. Event
    // callbacks keep re-entry blocked even after native work has finished.
    binding_asr_events_drain();
    // Native work has ended and its results are owned. Release before encoding
    // and callback delivery so completion callbacks can enqueue the next job.
    if (job->owns_busy)
        binding_browser_busy.store(false, std::memory_order_release);
    binding_browser_deliver(reinterpret_cast<uintptr_t>(job), reinterpret_cast<uintptr_t>(binding_browser_encode),
                            reinterpret_cast<uintptr_t>(binding_browser_destroy),
                            reinterpret_cast<uintptr_t>(job->callback.as_handle()));
}
template <typename T> struct binding_browser_state {
    T value{};
    std::string error;
    bool failed = false;
    std::atomic<bool> complete{false};
};
template <typename T> struct binding_browser_result : binding_browser_delivery {
    binding_browser_state<T> state;
    std::function<emscripten::val(T&)> encoder;
    binding_browser_result(emscripten::val cb, std::function<emscripten::val(T&)> fn)
        : binding_browser_delivery(std::move(cb)), encoder(std::move(fn)) {}
    bool ready() const override { return state.complete.load(std::memory_order_acquire); }
    emscripten::val encode() override {
        if (!state.failed)
            return encoder(state.value);
        auto out = emscripten::val::object();
        out.set("error", state.error);
        return out;
    }
};
template <typename T>
static void browser_dispatch(std::function<T()> work, std::function<emscripten::val(T&)> encode,
                             emscripten::val callback) {
    binding_check_reentry();
    if (callback.typeOf().as<std::string>() != "function")
        throw std::invalid_argument("Browser completion callback must be a function");
    auto job = std::make_unique<binding_browser_result<T>>(std::move(callback), std::move(encode));
    auto* state = &job->state;
    if (!g_compute_thread_set.load() || g_compute_thread == pthread_self()) {
        state->failed = true;
        state->error = "Proxied compute thread is not ready";
        state->complete = true;
    } else {
        bool idle = false;
        if (!binding_browser_busy.compare_exchange_strong(idle, true))
            throw std::runtime_error("Browser compute is busy");
        job->owns_busy = true;
        try {
            if (!g_proxy_queue.proxyAsync(g_compute_thread, [state, work = std::move(work)]() {
                    try {
                        state->value = work();
                    } catch (const std::exception& e) {
                        state->failed = true;
                        state->error = e.what();
                    } catch (...) {
                        state->failed = true;
                        state->error = "Unknown native compute exception";
                    }
                    // Last access to state on this thread. The servicer may
                    // retire it immediately after this release publication.
                    state->complete.store(true, std::memory_order_release);
                })) {
                state->failed = true;
                state->error = "Could not dispatch browser compute";
                state->complete = true;
            }
        } catch (const std::exception& e) {
            state->failed = true;
            state->error = e.what();
            state->complete = true;
        } catch (...) {
            state->failed = true;
            state->error = "Could not dispatch browser compute";
            state->complete = true;
        }
    }
    // Polling is scheduled on the creating servicer before returning. There is
    // no return proxy queue that can fail and strand an emval on compute.
    emscripten_async_call(binding_browser_poll, job.release(), 0);
}
#endif
