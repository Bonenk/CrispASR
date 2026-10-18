#include "crispasr_session.h"
#include "runtime-open-layout.h"
#include <emscripten/threading.h>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <thread>
struct crispasr_session {
    int tag;
    crispasr_segment_callback segment = nullptr;
    crispasr_token_callback token = nullptr;
    crispasr_progress_callback progress = nullptr;
    void* segment_data = nullptr;
    void* token_data = nullptr;
    void* progress_data = nullptr;
};
static std::atomic<int> callback_close_errors{0};
extern "C" int probe_asr_close_errors() {
    return callback_close_errors.load();
}
extern "C" void crispasr_session_set_segment_callback(crispasr_session* s, crispasr_segment_callback cb, void* ud) {
    s->segment = cb;
    s->segment_data = ud;
}
extern "C" void crispasr_session_set_token_callback(crispasr_session* s, crispasr_token_callback cb, void* ud) {
    s->token = cb;
    s->token_data = ud;
}
extern "C" void crispasr_session_set_progress_callback(crispasr_session* s, crispasr_progress_callback cb, void* ud) {
    s->progress = cb;
    s->progress_data = ud;
}
static crispasr_token_callback saved_token = nullptr;
static void* saved_token_data = nullptr;
static std::atomic<bool> event_finished{true};
extern "C" void probe_asr_emit_prepare() {
    event_finished = false;
}
extern "C" int probe_asr_emit_finished() {
    return event_finished.load();
}
extern "C" void probe_asr_emit(crispasr_session* s, int mode) {
    struct Finished {
        ~Finished() { event_finished = true; }
    } finished;
#ifdef __EMSCRIPTEN_PTHREADS__
    if (emscripten_is_main_runtime_thread())
        throw std::runtime_error("ASR events emitted on servicer");
#endif
    if (!s)
        throw std::runtime_error("ASR emitter missing parent");
    if (mode == 1) {
        saved_token = s->token;
        saved_token_data = s->token_data;
        return;
    }
    if (mode == 2) {
        if (saved_token)
            saved_token("stale", 99, saved_token_data);
        return;
    }
    if (mode == 3) {
        for (int i = 0; i < 4200; ++i)
            if (s->token)
                s->token("x", i, s->token_data);
        return;
    }
    if (mode == 4) {
        if (s->segment)
            s->segment("bad", 9007199254740992LL, 9007199254740993LL, 0, s->segment_data);
        return;
    }
    if (mode == 5) {
        std::thread producer([s]() {
            if (s->token)
                s->token("foreign", 5, s->token_data);
        });
        producer.join();
        return;
    }
    if (s->progress)
        s->progress(16, 64, s->progress_data);
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    char segment[] = "日本語";
    char first[] = {char(0xe6), char(0x97), 0}, second[] = {char(0xa5), 0};
    if (s->segment)
        s->segment(segment, 125, 250, 7, s->segment_data);
    if (s->token) {
        s->token(first, 0, s->token_data);
        s->token(second, 1, s->token_data);
        s->token("", 2, s->token_data);
    }
    std::memset(segment, 'z', sizeof(segment) - 1);
    first[0] = 'x';
    second[0] = 'x';
    if (s->progress)
        s->progress(64, 64, s->progress_data);
}
static std::atomic<int> live{0}, calls{0};
static int defaults;
extern "C" int probe_async_open_live() {
    return live.load();
}
extern "C" int probe_async_open_calls() {
    return calls.load();
}
extern "C" void probe_async_open_defaults(int value) {
    defaults = value;
}
extern "C" crispasr_session* crispasr_session_open_with_params(const char* model, const char* backend,
                                                               const crispasr_open_params_v1* p) {
    ++calls;
#ifdef __EMSCRIPTEN_PTHREADS__
    if (pthread_equal(pthread_self(), emscripten_main_runtime_thread_id()))
        return nullptr;
    const int default_threads = 4;
#else
    const int default_threads = 1;
#endif
    if (!std::strcmp(model, "throw"))
        throw std::runtime_error("open sentinel");
    if (std::strcmp(model, "model"))
        return nullptr;
    if (p->abi_version != 2 || p->n_threads != (defaults ? default_threads : 7) || p->use_gpu != (defaults ? 1 : 0) ||
        p->verbosity != (defaults ? 0 : 2) || p->flash_attn != (defaults ? 1 : 0) ||
        p->n_gpu_layers != (defaults ? -1 : 3))
        return nullptr;
    if ((defaults && backend) || (!defaults && (!backend || std::strcmp(backend, "foo"))))
        return nullptr;
    for (int n : p->reserved)
        if (n)
            return nullptr;
    ++live;
    return new crispasr_session{42};
}
extern "C" void probe_stream_parent_closing(crispasr_session*);
extern "C" void crispasr_session_close(crispasr_session* session) {
    if (session) {
        probe_stream_parent_closing(session);
        if (session->segment || session->token || session->progress)
            ++callback_close_errors;
        --live;
        delete session;
    }
}

static std::atomic<int> audio_live{0};
extern "C" int probe_async_audio_live() {
    return audio_live.load();
}
extern "C" float* crispasr_session_synthesize(crispasr_session* session, const char* text, int* count) {
    if (!std::strcmp(text, "throw"))
        throw std::runtime_error("synthesis sentinel");
    if (!std::strcmp(text, "empty")) {
        *count = 0;
        return nullptr;
    }
    if (!std::strcmp(text, "missing")) {
        *count = 3;
        return nullptr;
    }
    auto* pcm = static_cast<float*>(std::malloc(3 * sizeof(float)));
    ++audio_live;
    pcm[0] = .25f;
    pcm[1] = .5f;
    pcm[2] = .75f;
    *count = !std::strcmp(text, "zero-allocated") ? 0 : !std::strcmp(text, "bad") ? -1 : 3;
    return pcm;
}
extern "C" void crispasr_pcm_free(float* pcm) {
    if (pcm) {
        --audio_live;
        std::free(pcm);
    }
}
