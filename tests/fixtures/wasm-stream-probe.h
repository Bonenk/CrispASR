#pragma once
#include <chrono>
#include <thread>
static std::atomic<bool> probe_event_queue_held{false}, probe_event_queue_release{false};
extern "C" int probe_stream_live();
extern "C" int probe_stream_closes();
extern "C" int probe_stream_order_errors();
extern "C" int probe_stream_calls();
extern "C" void probe_stream_mode(int);
extern "C" void probe_asr_emit(crispasr_session*, int);
extern "C" int probe_asr_close_errors();
extern "C" void probe_asr_emit_prepare();
extern "C" int probe_asr_emit_finished();
static void register_stream_probe() {
    using namespace emscripten;
    function("eventFinished", &probe_asr_emit_finished);
    function("callbackCloseErrors", &probe_asr_close_errors);
    function("exhaustCallbackVersions", optional_override([]() {
                 std::lock_guard<std::mutex> lock(binding_asr_events_mutex);
                 binding_asr_event_next_version = INT_MAX;
             }));
    function("eventTimerArmed", optional_override([]() { return int(binding_asr_events_timer_armed); }));
    function("eventLive", optional_override([]() { return binding_asr_event_live.load(); }));
    function("eventQueued", optional_override([]() {
                 std::lock_guard<std::mutex> lock(binding_asr_events_mutex);
                 return int(binding_asr_events.size());
             }));
#ifdef __EMSCRIPTEN_PTHREADS__
    function("eventQueueHeld", optional_override([]() { return int(probe_event_queue_held.load()); }));
    function("releaseEventQueue", optional_override([]() { probe_event_queue_release = true; }));
    function("holdEventQueueAsync", optional_override([](val callback) {
                 binding_check_reentry();
                 probe_event_queue_held = false;
                 probe_event_queue_release = false;
                 browser_dispatch<int>(
                     []() {
                         std::lock_guard<std::mutex> lock(binding_asr_events_mutex);
                         probe_event_queue_held = true;
                         auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
                         while (!probe_event_queue_release.load() && std::chrono::steady_clock::now() < deadline)
                             std::this_thread::sleep_for(std::chrono::milliseconds(1));
                         if (!probe_event_queue_release.load())
                             throw std::runtime_error("Servicer blocked on ASR queue lock");
                         return 0;
                     },
                     [](int& value) { return val(value); }, callback);
             }));
    function("emitAsrEventsAsync", optional_override([](int mode, val cb) {
                 binding_check_reentry();
                 probe_asr_emit_prepare();
                 browser_dispatch<int>(
                     [mode]() {
                         probe_asr_emit(g_asr_session, mode);
                         return 0;
                     },
                     [](int& value) { return val(value); }, cb);
             }));
#else
    function("emitAsrEvents", optional_override([](int mode) {
                 binding_check_reentry();
                 probe_asr_emit(g_asr_session, mode);
             }));
#endif
    function("streamLive", &probe_stream_live);
    function("streamCloses", &probe_stream_closes);
    function("streamOrderErrors", &probe_stream_order_errors);
    function("streamCalls", &probe_stream_calls);
    function("streamMode", &probe_stream_mode);
    function("batchAllowed", optional_override([]() {
                 binding_check_reentry();
                 binding_require_no_stream(g_asr_session);
                 return true;
             }));
}
