// Shared decoded-result probe for language wrappers. No model inference.
#include "crispasr_session.h"
#include <string.h>
#ifdef CRISPASR_PROBE_VAD_ARGUMENTS
#include "wrapper-vad-arguments.h"
#endif
static char token_head[3], token_tail[2];
static int result_frees;
static int result_count_mode;
void probe_result_count_mode(int mode) {
    result_count_mode = mode;
}
extern void probe_callbacks(void);
int probe_result_frees(void) {
    return result_frees;
}
static crispasr_session_result* new_result(void) {
    token_head[0] = (char)0xe6;
    token_head[1] = (char)0x97;
    token_head[2] = 0;
    token_tail[0] = (char)0xa5;
    token_tail[1] = 0;
    probe_callbacks();
    return (crispasr_session_result*)1;
}
static crispasr_session* probe_session_handle(void) {
#ifdef CRISPASR_PROBE_UNIQUE_SESSIONS
    static uintptr_t next = 0x1000;
    return (crispasr_session*)++next;
#else
    return (crispasr_session*)1;
#endif
}
crispasr_session* crispasr_session_open(const char* model, int threads) {
    return probe_session_handle();
}
crispasr_session* crispasr_session_open_explicit(const char* model, const char* backend, int threads) {
    return probe_session_handle();
}
crispasr_session_result* crispasr_session_transcribe(crispasr_session* s, const float* p, int n) {
    return new_result();
}
crispasr_session_result* crispasr_session_transcribe_lang(crispasr_session* s, const float* p, int n,
                                                          const char* lang) {
    return new_result();
}
crispasr_session_result* crispasr_session_transcribe_chunked_lang(crispasr_session* s, const float* p, int n, int c,
                                                                  int overlap, const char* lang) {
    return new_result();
}
crispasr_session_result* crispasr_session_transcribe_vad(crispasr_session* s, const float* p, int n, int sr,
                                                         const char* model, const crispasr_vad_abi_opts* opts) {
#ifdef CRISPASR_PROBE_VAD_ARGUMENTS
    if (!probe_vad_arguments(p, n, sr, model, opts, NULL))
        return NULL;
#endif
    return new_result();
}
crispasr_session_result* crispasr_session_transcribe_vad_lang(crispasr_session* s, const float* p, int n, int sr,
                                                              const char* model, const crispasr_vad_abi_opts* opts,
                                                              const char* lang) {
#ifdef CRISPASR_PROBE_VAD_ARGUMENTS
    if (!probe_vad_arguments(p, n, sr, model, opts, lang))
        return NULL;
#endif
    return new_result();
}
static int result_logits_enabled;
int probe_result_logits_enabled(void) {
    return result_logits_enabled;
}
int crispasr_session_set_return_logits(crispasr_session* s, int value) {
    result_logits_enabled = value;
    return 0;
}
int crispasr_session_result_n_logit_frames(crispasr_session_result* r) {
    return result_count_mode == 5 ? -1 : result_count_mode == 6 ? INT32_MAX : result_count_mode >= 7 ? 1 : 0;
}
int crispasr_session_result_n_logit_vocab(crispasr_session_result* r) {
    return result_count_mode >= 6 ? 2 : 0;
}
const float* crispasr_session_result_logits(crispasr_session_result* r) {
    static float logits[] = {-1, 2};
    return result_count_mode == 8 ? logits : NULL;
}
int crispasr_session_result_n_segments(crispasr_session_result* r) {
    if (result_count_mode == 1)
        return -1;
    return 1;
}
const char* crispasr_session_result_segment_text(crispasr_session_result* r, int i) {
    return "Japanese";
}
int64_t crispasr_session_result_segment_t0(crispasr_session_result* r, int i) {
    return 123;
}
int64_t crispasr_session_result_segment_t1(crispasr_session_result* r, int i) {
    return 456;
}
float crispasr_session_result_segment_no_speech_prob(crispasr_session_result* r, int i) {
    return .125f;
}
const char* crispasr_session_result_segment_speaker(crispasr_session_result* r, int i) {
    return "speaker";
}
int crispasr_session_result_n_words(crispasr_session_result* r, int i) {
    if (result_count_mode == 2)
        return -1;
    return 1;
}
const char* crispasr_session_result_word_text(crispasr_session_result* r, int i, int j) {
    return "word";
}
int64_t crispasr_session_result_word_t0(crispasr_session_result* r, int i, int j) {
    return -1;
}
int64_t crispasr_session_result_word_t1(crispasr_session_result* r, int i, int j) {
    return -1;
}
float crispasr_session_result_word_p(crispasr_session_result* r, int i, int j) {
    return .75f;
}
int crispasr_session_result_word_n_alts(crispasr_session_result* r, int i, int j) {
    if (result_count_mode == 3)
        return -1;
    return 1;
}
const char* crispasr_session_result_word_alt_text(crispasr_session_result* r, int i, int j, int k) {
    return "alternative";
}
float crispasr_session_result_word_alt_p(crispasr_session_result* r, int i, int j, int k) {
    return .25f;
}
void crispasr_session_result_free(crispasr_session_result* r) {
    ++result_frees;
    token_head[0] = 'X';
}
#ifndef CRISPASR_PROBE_OLD_TOKENS
int crispasr_session_result_n_tokens(crispasr_session_result* r, int i) {
    if (result_count_mode == 4)
        return -1;
    return 2;
}
const char* crispasr_session_result_token_text(crispasr_session_result* r, int i, int j) {
    return j ? token_tail : token_head;
}
int64_t crispasr_session_result_token_t0(crispasr_session_result* r, int i, int j) {
    return j ? 123 : -1;
}
int64_t crispasr_session_result_token_t1(crispasr_session_result* r, int i, int j) {
    return j ? 124 : -1;
}
float crispasr_session_result_token_p(crispasr_session_result* r, int i, int j) {
    return .75f;
}
#endif
