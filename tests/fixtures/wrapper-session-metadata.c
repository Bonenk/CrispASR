// Shared decoded-result probe for language wrappers. No model inference.
#include "crispasr_session.h"
#include <string.h>
static char token_head[3], token_tail[2];
static int result_frees;
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
crispasr_session* crispasr_session_open(const char* model, int threads) {
    return (crispasr_session*)1;
}
crispasr_session* crispasr_session_open_explicit(const char* model, const char* backend, int threads) {
    return (crispasr_session*)1;
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
    return new_result();
}
crispasr_session_result* crispasr_session_transcribe_vad_lang(crispasr_session* s, const float* p, int n, int sr,
                                                              const char* model, const crispasr_vad_abi_opts* opts,
                                                              const char* lang) {
    return new_result();
}
int crispasr_session_set_return_logits(crispasr_session* s, int value) {
    return 0;
}
int crispasr_session_result_n_logit_frames(crispasr_session_result* r) {
    return 0;
}
int crispasr_session_result_n_logit_vocab(crispasr_session_result* r) {
    return 0;
}
const float* crispasr_session_result_logits(crispasr_session_result* r) {
    return 0;
}
int crispasr_session_result_n_segments(crispasr_session_result* r) {
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
