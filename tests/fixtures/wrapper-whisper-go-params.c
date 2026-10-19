// Included by the Go/Rust probe; real public-header struct transport.
static char legacy_prompt[256], legacy_model[256];
#include <stdatomic.h>
#include <pthread.h>
static int legacy_foreign_callback;
void probe_legacy_foreign_callback(int enabled) {
    legacy_foreign_callback = enabled;
}
struct probe_encoder_call {
    struct whisper_context* context;
    struct whisper_full_params* params;
    bool accepted;
};
static void* probe_encoder_foreign(void* data) {
    struct probe_encoder_call* call = data;
    call->accepted =
        call->params->encoder_begin_callback(call->context, NULL, call->params->encoder_begin_callback_user_data);
    return NULL;
}
static int legacy_mode;
static _Atomic int legacy_calls;
void probe_legacy_expect(const char* prompt, const char* model) {
    snprintf(legacy_prompt, sizeof(legacy_prompt), "%s", prompt);
    snprintf(legacy_model, sizeof(legacy_model), "%s", model);
}
void probe_legacy_mode(int mode) {
    legacy_mode = mode;
}
int probe_legacy_calls(void) {
    return legacy_calls;
}
static int probe_legacy_full(struct whisper_context* ctx, struct whisper_full_params p, const float* samples, int n) {
    ++legacy_calls;
    if (!ctx || !samples || n != 3 || samples[0] != 1 || samples[2] != 5 || p.n_threads != 3 || p.greedy.best_of != 7 ||
        !p.detect_language || p.suppress_blank || !p.tdrz_enable || p.alt_n != 32 || !p.initial_prompt ||
        strcmp(p.initial_prompt, legacy_prompt) || !p.vad_model_path || strcmp(p.vad_model_path, legacy_model))
        return -9;
    if (legacy_mode == 1)
        return -1;
    if (p.encoder_begin_callback) {
        if (legacy_foreign_callback) {
            struct probe_encoder_call call = {ctx, &p, false};
            pthread_t thread;
            if (pthread_create(&thread, NULL, probe_encoder_foreign, &call) != 0)
                return -3;
            if (pthread_join(thread, NULL) != 0)
                abort();
            if (!call.accepted)
                return -2;
        } else if (!p.encoder_begin_callback(ctx, NULL, p.encoder_begin_callback_user_data)) {
            return -2;
        }
    }
    if (p.new_segment_callback)
        p.new_segment_callback(ctx, NULL, 1, p.new_segment_callback_user_data);
    if (p.progress_callback)
        p.progress_callback(ctx, NULL, 50, p.progress_callback_user_data);
    return 0;
}
