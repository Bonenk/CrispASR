// Direct Parakeet ABI oracle. Real header signatures; no model quality claim.
#include "crispasr_session.h"
#include <stdlib.h>
#include <string.h>
static int legacy_parakeet_contexts, legacy_parakeet_results;
static int64_t legacy_parakeet_offset;
int probe_parakeet_contexts(void) {
    return legacy_parakeet_contexts;
}
int probe_parakeet_results(void) {
    return legacy_parakeet_results;
}
int64_t probe_parakeet_offset(void) {
    return legacy_parakeet_offset;
}
parakeet_context* crispasr_parakeet_init(const char* model, int threads, int flash) {
    if (!strcmp(model, "fail"))
        return NULL;
    ++legacy_parakeet_contexts;
    return (parakeet_context*)malloc(1);
}
void crispasr_parakeet_free(parakeet_context* ctx) {
    if (ctx) {
        --legacy_parakeet_contexts;
        free(ctx);
    }
}
parakeet_result* crispasr_parakeet_transcribe(parakeet_context* ctx, const float* pcm, int n, int64_t offset) {
    if (!ctx || n != 3 || pcm[0] != 1 || pcm[2] != 5 || (offset != 0 && offset != 123))
        return NULL;
    legacy_parakeet_offset = offset;
    ++legacy_parakeet_results;
    return (parakeet_result*)malloc(1);
}
void crispasr_parakeet_result_free(parakeet_result* r) {
    if (r) {
        --legacy_parakeet_results;
        free(r);
    }
}
const char* crispasr_parakeet_result_text(parakeet_result* r) {
    return "日本語";
}
int crispasr_parakeet_result_n_words(parakeet_result* r) {
    return 1;
}
const char* crispasr_parakeet_result_word_text(parakeet_result* r, int i) {
    return "日本語";
}
int64_t crispasr_parakeet_result_word_t0(parakeet_result* r, int i) {
    return 100 + legacy_parakeet_offset;
}
int64_t crispasr_parakeet_result_word_t1(parakeet_result* r, int i) {
    return 200 + legacy_parakeet_offset;
}
int crispasr_parakeet_result_n_tokens(parakeet_result* r) {
    return 2;
}
const char* crispasr_parakeet_result_token_text(parakeet_result* r, int i) {
    static const char first[] = {(char)0xe6, (char)0x97, 0}, second[] = {(char)0xa5, 0};
    return i ? second : first;
}
int64_t crispasr_parakeet_result_token_t0(parakeet_result* r, int i) {
    return i ? 123 + legacy_parakeet_offset : -1;
}
int64_t crispasr_parakeet_result_token_t1(parakeet_result* r, int i) {
    return i ? 150 + legacy_parakeet_offset : -1;
}
float crispasr_parakeet_result_token_p(parakeet_result* r, int i) {
    return .75f;
}
