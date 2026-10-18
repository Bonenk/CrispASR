// ABI transport only. The real runtime's scalar setters/token/parallel adapters
// are compiled alongside this fake decoder, using the actual public structs.
#include "crispasr.h"
#include "crispasr_session.h"
#include <cstdlib>
#include <cstring>
#include <thread>
static int live, contexts, calls, processors, dtw, mode, saw_zero;
static whisper_context_params captured_context{};
extern "C" {
int probe_legacy_gpu_device() {
    return captured_context.gpu_device;
}
int probe_legacy_use_gpu() {
    return captured_context.use_gpu;
}
int probe_legacy_flash() {
    return captured_context.flash_attn;
}
int probe_legacy_head_count() {
    return int(captured_context.dtw_aheads.n_heads);
}
int probe_legacy_head_value() {
    return captured_context.dtw_aheads.heads
               ? captured_context.dtw_aheads.heads[0].n_text_layer * 100 + captured_context.dtw_aheads.heads[0].n_head
               : -1;
}
uint64_t probe_legacy_dtw_memory() {
    return captured_context.dtw_mem_size;
}
int probe_legacy_full_size() {
    return sizeof(whisper_full_params);
}

int probe_legacy_live() {
    return live;
}
int probe_legacy_contexts() {
    return contexts;
}
int probe_legacy_calls() {
    return calls;
}
int probe_legacy_processors() {
    return processors;
}
int probe_legacy_dtw() {
    return dtw;
}
int probe_legacy_progress_zero() {
    return saw_zero;
}
void probe_legacy_mode(int value) {
    mode = value;
}
whisper_context_params* whisper_context_default_params_by_ref() {
    if (mode == 4)
        return nullptr;
    auto p = new whisper_context_params{};
    p->use_gpu = true;
    p->flash_attn = true;
    ++live;
    return p;
}
void whisper_free_context_params(whisper_context_params* p) {
    if (p) {
        delete p;
        --live;
    }
}
whisper_full_params* whisper_full_default_params_by_ref(enum whisper_sampling_strategy strategy) {
    if (mode == 3)
        return nullptr;
    auto p = new whisper_full_params{};
    p->strategy = strategy;
    p->n_threads = 4;
    p->language = "en";
    p->suppress_blank = true;
    ++live;
    return p;
}
void whisper_free_params(whisper_full_params* p) {
    if (p) {
        delete p;
        --live;
    }
}
whisper_context* whisper_init_from_file_with_params_by_ref(const char* path, whisper_context_params* p) {
    captured_context = *p;
    dtw = p->dtw_token_timestamps ? (int)p->dtw_aheads_preset * 100 + p->dtw_n_top : 0;
    if (std::strstr(path, "fail"))
        return nullptr;
    ++contexts;
    return reinterpret_cast<whisper_context*>(uintptr_t(0x100000001ULL));
}
void whisper_free(whisper_context* ctx) {
    if (ctx)
        --contexts;
}
int whisper_full_by_ref(whisper_context* ctx, whisper_full_params* p, const float* pcm, int n) {
    ++calls;
    if (mode == 2)
        return -1;
    if (!ctx || !p || n != 3 || pcm[0] != 1 || pcm[2] != 5)
        return -9;
    if (p->language && std::strcmp(p->language, "de"))
        return -8;
    if (p->initial_prompt && std::strcmp(p->initial_prompt, "Grüße"))
        return -7;
    if (p->progress_callback) {
        auto callback = [&]() { p->progress_callback(ctx, nullptr, 37, p->progress_callback_user_data); };
        if (mode == 6) {
            std::thread producer(callback);
            producer.join();
        } else
            callback();
    }
    return 0;
}
int whisper_full_parallel(whisper_context* ctx, whisper_full_params p, const float* pcm, int n, int count) {
    processors = count;
    saw_zero = crispasr_get_progress() == 0;
    return whisper_full_by_ref(ctx, &p, pcm, n);
}
int whisper_full_lang_id(whisper_context*) {
    return 0;
}
const char* whisper_lang_str(int) {
    return "de";
}
int whisper_full_n_segments(whisper_context* ctx) {
    return mode == 7 ? -1 : mode == 1 ? 0 : 1;
}
const char* whisper_full_get_segment_text(whisper_context* ctx, int index) {
    return "日";
}
int64_t whisper_full_get_segment_t0(whisper_context* ctx, int index) {
    return 100;
}
int64_t whisper_full_get_segment_t1(whisper_context* ctx, int index) {
    return 200;
}
float whisper_full_get_segment_no_speech_prob(whisper_context* ctx, int index) {
    return .125f;
}
int whisper_full_n_tokens(whisper_context* ctx, int index) {
    return 2;
}
const char* whisper_full_get_token_text(whisper_context* ctx, int segment, int token) {
    static const char first[] = {char(0xe6), char(0x97), 0};
    static const char second[] = {char(0xa5), 0};
    return token ? second : first;
}
whisper_token whisper_full_get_token_id(whisper_context* ctx, int segment, int token) {
    return token + 10;
}
whisper_token_data whisper_full_get_token_data(whisper_context* ctx, int segment, int token) {
    whisper_token_data p{};
    p.id = token + 10;
    p.t0 = token ? 123 : -1;
    p.t1 = token ? 150 : -1;
    p.t_dtw = 140;
    p.p = token ? .75f : .5f;
    return p;
}
int whisper_full_get_token_n_alts(whisper_context* ctx, int segment, int token) {
    return 1;
}
whisper_token whisper_full_get_token_alt_id(whisper_context* ctx, int segment, int token, int alternative) {
    return 42;
}
float whisper_full_get_token_alt_p(whisper_context* ctx, int segment, int token, int alternative) {
    return .25f;
}
const char* whisper_token_to_str(whisper_context* ctx, whisper_token token) {
    return token == 42 ? "alternative" : "token";
}
}
