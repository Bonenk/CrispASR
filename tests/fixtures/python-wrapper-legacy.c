#include "crispasr.h"
#include "crispasr_session.h"
#include <stdlib.h>
#include <string.h>
static int live;
int probe_live(void) {
    return live;
}
void whisper_free(struct whisper_context* ctx) {}
struct whisper_context_params* whisper_context_default_params_by_ref(void) {
    ++live;
    return calloc(1, sizeof(struct whisper_context_params));
}
void whisper_free_context_params(struct whisper_context_params* p) {
    --live;
    free(p);
}
struct whisper_full_params* whisper_full_default_params_by_ref(int strategy) {
    struct whisper_full_params* p = calloc(1, sizeof(*p));
    p->marker = 123 + strategy;
    ++live;
    return p;
}
void whisper_free_params(struct whisper_full_params* p) {
    --live;
    free(p);
}
void crispasr_ctx_params_set_dtw(struct whisper_context_params* p, bool enable, int preset, int top) {
    p->enabled = enable;
    p->preset = preset;
    p->top = top;
}
struct whisper_context* whisper_init_from_file_with_params(const char* model, struct whisper_context_params p) {
    if (p.enabled && (p.preset != 1 || p.top != 2))
        return NULL;
    return (struct whisper_context*)(uintptr_t)0x100000001ULL;
}
int whisper_full(struct whisper_context* ctx, struct whisper_full_params p, const float* pcm, int n) {
    if (p.marker != 123 || !p.language || strcmp(p.language, "de") || !p.prompt || strcmp(p.prompt, "Grüße"))
        return -1;
    if (!p.timestamps || n != 3 || pcm[0] != 1 || pcm[1] != 3 || pcm[2] != 5)
        return -2;
    return 0;
}
int crispasr_transcribe_parallel(struct whisper_context* ctx, struct whisper_full_params p, const float* pcm, int n,
                                 int workers) {
    return workers == 3 ? whisper_full(ctx, p, pcm, n) : -3;
}
void crispasr_params_set_language(struct whisper_full_params* p, const char* value) {
    p->language = value;
}
void crispasr_params_set_initial_prompt(struct whisper_full_params* p, const char* value) {
    p->prompt = value;
}
void crispasr_params_set_token_timestamps(struct whisper_full_params* p, int value) {
    p->timestamps = value;
}
int whisper_full_n_segments(struct whisper_context* ctx) {
    return 1;
}
const char* whisper_full_get_segment_text(struct whisper_context* ctx, int i) {
    return "日本語";
}
int64_t whisper_full_get_segment_t0(struct whisper_context* ctx, int i) {
    return 123;
}
int64_t whisper_full_get_segment_t1(struct whisper_context* ctx, int i) {
    return 456;
}
float whisper_full_get_segment_no_speech_prob(struct whisper_context* ctx, int i) {
    return 0.125f;
}
int whisper_full_lang_id(struct whisper_context* ctx) {
    return 0;
}
const char* whisper_lang_str(int i) {
    return "de";
}
int whisper_full_n_tokens(struct whisper_context* ctx, int i) {
    return 2;
}
const char* whisper_full_get_token_text(struct whisper_context* ctx, int i, int j) {
    return j ? "\xa5" : "\xe6\x97";
}
int64_t crispasr_token_t0(struct whisper_context* ctx, int i, int j) {
    return j ? 123 : -1;
}
int64_t crispasr_token_t1(struct whisper_context* ctx, int i, int j) {
    return j ? 456 : -1;
}
int64_t crispasr_token_dtw_t(struct whisper_context* ctx, int i, int j) {
    return j ? 42 : -1;
}
float crispasr_token_p(struct whisper_context* ctx, int i, int j) {
    return 0.75f;
}
int crispasr_token_n_alts(struct whisper_context* ctx, int i, int j) {
    return j ? 0 : 1;
}
int32_t crispasr_token_alt_id(struct whisper_context* ctx, int i, int j, int k) {
    return 1234;
}
float crispasr_token_alt_p(struct whisper_context* ctx, int i, int j, int k) {
    return 0.25f;
}
int crispasr_token_alt_text(struct whisper_context* ctx, int i, int j, int k, char* out, int cap) {
    if (cap < 301)
        return -1;
    memset(out, 'x', 300);
    out[300] = 0;
    return 300;
}
