#include "crispasr_session.h"
#include <stdlib.h>
#include <string.h>
static int live, mode;
int probe_analysis_live(void) {
    return live;
}
void probe_analysis_mode(int value) {
    mode = value;
}
crispasr_align_result* crispasr_align_words_abi(const char* model, const char* transcript, const float* pcm, int32_t n,
                                                int64_t offset, int32_t threads) {
    if (mode == 2)
        return NULL;
    if (n != 3 || offset != 123 || strcmp(transcript, "日本語"))
        abort();
    ++live;
    return (crispasr_align_result*)1;
}
void crispasr_align_result_free(crispasr_align_result* r) {
    if (r)
        --live;
}
int crispasr_align_result_n_words(crispasr_align_result* r) {
    return mode == 1 ? 0 : 1;
}
const char* crispasr_align_result_word_text(crispasr_align_result* r, int i) {
    return "日本語";
}
int64_t crispasr_align_result_word_t0(crispasr_align_result* r, int i) {
    return 123;
}
int64_t crispasr_align_result_word_t1(crispasr_align_result* r, int i) {
    return 456;
}
int crispasr_align_result_n_characters(crispasr_align_result* r, int i) {
    return 2;
}
const char* crispasr_align_result_character_text(crispasr_align_result* r, int i, int j) {
    return j ? "本" : "日";
}
int64_t crispasr_align_result_character_t0(crispasr_align_result* r, int i, int j) {
    return j ? 140 : 123;
}
int64_t crispasr_align_result_character_t1(crispasr_align_result* r, int i, int j) {
    return j ? 165 : 140;
}
static int spans(float** out) {
    *out = malloc(2 * sizeof(float));
    ++live;
    (*out)[0] = 1.25f;
    (*out)[1] = 2.5f;
    return mode == 2 ? -3 : mode == 1 ? 0 : 1;
}
int crispasr_vad_segments(const char* model, const float* pcm, int n, int rate, float threshold, int minSpeech,
                          int minSilence, int threads, bool gpu, float** out) {
    return spans(out);
}
int crispasr_vad_slices(const char* model, const float* pcm, int n, int rate, float threshold, int minSpeech,
                        int minSilence, int pad, float maxChunk, int threads, float** out) {
    return spans(out);
}
void crispasr_vad_free(float* p) {
    if (p) {
        free(p);
        --live;
    }
}
