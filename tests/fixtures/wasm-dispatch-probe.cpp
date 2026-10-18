#include <emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/proxying.h>
#include <pthread.h>
#include <atomic>
#include <functional>
#include <memory>
#include <stdexcept>
#include <thread>
struct ProbeQueue {
    emscripten::ProxyingQueue queue;
    int failure = 0;
    template <typename F> bool proxyAsync(pthread_t target, F work) {
        int mode = failure;
        failure = 0;
        if (mode == 1)
            return false;
        if (mode == 2)
            throw std::runtime_error("queue sentinel");
        return queue.proxyAsync(target, std::move(work));
    }
};
static ProbeQueue g_proxy_queue;
static pthread_t g_compute_thread;
static std::atomic<bool> g_compute_thread_set{false};
static std::atomic<int> live_payloads{0}, progress{0};
#include "crispasr_session.h"
static crispasr_session* g_asr_session = nullptr;
static crispasr_session* g_tts_session = nullptr;
#include "session_guard.h"
#include "session_dispatch.h"
#include "session_options.h"
#include "session_async_options.h"
#include "session_async_audio.h"
#include "session_streams.h"
#include "wasm-stream-probe.h"
extern "C" int probe_async_audio_live();
extern "C" int probe_async_open_live();
extern "C" int probe_async_open_calls();
extern "C" void probe_async_open_defaults(int);
EM_JS(void, raise_encode_js, (), { throw new Error('JS encoding sentinel'); });
struct Payload {
    std::unique_ptr<int> value = std::make_unique<int>(0);
    Payload() { ++live_payloads; }
    Payload(Payload&& other) noexcept : value(std::move(other.value)) { ++live_payloads; }
    Payload& operator=(Payload&&) = default;
    ~Payload() { --live_payloads; }
};
int main() {
    g_compute_thread = pthread_self();
    g_compute_thread_set = true;
    emscripten_exit_with_live_runtime();
}
static void dispatch_probe(int mode, emscripten::val callback) {
    binding_check_reentry();
    progress = 0;
    if (mode == 4)
        g_proxy_queue.failure = 1;
    if (mode == 5)
        g_compute_thread_set = false;
    if (mode == 6)
        g_proxy_queue.failure = 2;
    browser_dispatch<Payload>(
        [mode]() {
            for (int i = 1; i <= 4; ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                progress = i;
            }
            if (mode == 1)
                throw std::runtime_error("work sentinel");
            if (mode == 7)
                throw 7;
            if (mode == 8)
                throw std::runtime_error("");
            Payload result;
            *result.value = 42;
            return result;
        },
        [mode](Payload& value) {
            if (mode == 2)
                throw std::runtime_error("encoding sentinel");
            if (mode == 3)
                raise_encode_js();
            return emscripten::val(*value.value);
        },
        callback);
    if (mode == 5)
        g_compute_thread_set = true;
}
EMSCRIPTEN_BINDINGS(dispatch_probe) {
    binding_register_async_options();
    binding_register_async_audio();
    binding_register_streams();
    binding_register_asr_callbacks();
    register_stream_probe();
    emscripten::function("audioLive", &probe_async_audio_live);
    emscripten::function("openLive", &probe_async_open_live);
    emscripten::function("openCalls", &probe_async_open_calls);
    emscripten::function("openDefaults", &probe_async_open_defaults);
    emscripten::function("closeAsr", emscripten::optional_override([]() {
                             binding_check_reentry();
                             binding_native_close_session(g_asr_session);
                         }));
    emscripten::function("closeTts", emscripten::optional_override([]() {
                             binding_check_reentry();
                             binding_native_close_session(g_tts_session);
                         }));
    emscripten::function("dispatchProbe", &dispatch_probe);
    emscripten::function("computeReady", emscripten::optional_override([]() { return g_compute_thread_set.load(); }));
    emscripten::function("livePayloads", emscripten::optional_override([]() { return live_payloads.load(); }));
    emscripten::function("probeProgress", emscripten::optional_override([]() {
                             binding_check_callback();
                             return progress.load();
                         }));
    emscripten::function("checkIdle", emscripten::optional_override([]() {
                             binding_check_reentry();
                             return true;
                         }));
}
