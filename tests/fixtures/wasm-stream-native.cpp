#include "crispasr_session.h"
#include <emscripten/threading.h>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
struct crispasr_stream {
    crispasr_session* parent;
    bool live_decode = false;
};
static crispasr_session* stream_parent = nullptr;
static std::atomic<int> live{0}, closes{0}, order_errors{0}, mode{0}, calls{0};
extern "C" int probe_stream_live() {
    return live.load();
}
extern "C" int probe_stream_closes() {
    return closes.load();
}
extern "C" int probe_stream_order_errors() {
    return order_errors.load();
}
extern "C" int probe_stream_calls() {
    return calls.load();
}
extern "C" void probe_stream_mode(int value) {
    mode = value;
}
// The wrapper permits one child per ASR singleton. Native fixture independently
// rejects closing the parent while its retained child remains alive.
extern "C" void probe_stream_parent_closing(crispasr_session* parent) {
    if (live && parent == stream_parent)
        ++order_errors;
}
static void check_thread() {
#ifdef __EMSCRIPTEN_PTHREADS__
    if (emscripten_is_main_runtime_thread())
        throw std::runtime_error("stream on servicer");
#endif
}
extern "C" int crispasr_session_stream_kind(crispasr_session* s) {
    return s ? 2 : 0;
}
extern "C" crispasr_stream* crispasr_session_stream_open(crispasr_session* parent, int threads, int step, int length,
                                                         int keep, const char* language, int translate) {
    check_thread();
    ++calls;
#ifdef __EMSCRIPTEN_PTHREADS__
    const int expected_threads = 4;
#else
    const int expected_threads = 1;
#endif
    if (!parent || threads != expected_threads || step != 100 || length != 300 || keep != 20 || !language ||
        std::strcmp(language, "ja") || translate != 0)
        throw std::runtime_error("stream option mismatch");
    if (mode == 1)
        return nullptr;
    ++live;
    stream_parent = parent;
    return new crispasr_stream{parent};
}
extern "C" void crispasr_stream_close(crispasr_stream* s) {
    if (!s)
        return;
    check_thread();
    --live;
    ++closes;
    delete s;
}
extern "C" void probe_asr_emit(crispasr_session*, int);
extern "C" int crispasr_stream_feed(crispasr_stream* s, const float* pcm, int n) {
    check_thread();
    ++calls;
    if (!s || !pcm || n != 3 || pcm[0] != .25f || pcm[1] != .5f || pcm[2] != .75f)
        throw std::runtime_error("stream PCM mismatch");
#ifdef __EMSCRIPTEN_PTHREADS__
    probe_asr_emit(s->parent, 5);
#endif
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    return mode == 2 ? -2 : 1;
}
extern "C" int crispasr_stream_flush(crispasr_stream* s) {
    check_thread();
    ++calls;
    return s && mode != 2 ? 1 : -2;
}
extern "C" void crispasr_stream_set_live_decode(crispasr_stream* s, int enabled) {
    check_thread();
    ++calls;
    s->live_decode = enabled != 0;
}
extern "C" int crispasr_stream_get_text(crispasr_stream* s, char* text, int cap, double* t0, double* t1,
                                        int64_t* counter) {
    check_thread();
    ++calls;
    if (mode == 3)
        return -2;
    if (mode == 4)
        return 2147483647;
    std::string content = mode == 5 ? std::string(5000, 'a') + "日本語" : "日本語";
    std::strncpy(text, content.c_str(), cap - 1);
    text[cap - 1] = 0;
    *t0 = 1.25;
    *t1 = 2.5;
    *counter = mode == 6 ? 9007199254740992LL : s->live_decode ? 4294967297LL : 3;
    return int(content.size());
}
