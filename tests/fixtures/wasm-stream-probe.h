#pragma once
extern "C" int probe_stream_live();
extern "C" int probe_stream_closes();
extern "C" int probe_stream_order_errors();
extern "C" int probe_stream_calls();
extern "C" void probe_stream_mode(int);
static void register_stream_probe() {
    using namespace emscripten;
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
