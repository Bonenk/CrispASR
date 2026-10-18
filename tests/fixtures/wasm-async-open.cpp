#include "crispasr_session.h"
#include "runtime-open-layout.h"
#include <emscripten/threading.h>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
struct crispasr_session {
    int tag;
};
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
