// Binary and allocation transport only; no decoder/model or signing backend.
#include "crispasr.h"
#include "crispasr_session.h"
#include <stdlib.h>
#include <string.h>
static int utility_live, utility_mode;
void probe_utility_mode(int mode) {
    utility_mode = mode;
}
int probe_utility_live(void) {
    return utility_live;
}
static float* utility_pcm(void) {
    float* p = malloc(3 * sizeof(float));
    p[0] = .25f;
    p[1] = .5f;
    p[2] = .75f;
    ++utility_live;
    return p;
}
int crispasr_audio_load_at_rate(const char* path, int target, float** pcm, int* n, int* rate) {
    *pcm = utility_pcm();
    *n = 3;
    *rate = target;
    return utility_mode == 2 ? -1 : 0;
}
int crispasr_audio_load(const char* path, float** pcm, int* n, int* rate) {
    return crispasr_audio_load_at_rate(path, 16000, pcm, n, rate);
}
int crispasr_audio_load_stereo(const char* path, float** left, float** right, int* n, int* rate, int* channels) {
    *left = utility_pcm();
    *right = utility_mode == 3 ? *left : utility_pcm();
    *n = 3;
    *rate = 16000;
    *channels = utility_mode == 3 ? 1 : 2;
    return utility_mode == 2 ? -1 : 0;
}
int crispasr_audio_resample(const float* pcm, int n, int source, int target, float** out, int* count) {
    if (n == 0) {
        *out = NULL;
        *count = 0;
        return 0;
    }
    *out = utility_pcm();
    *count = 3;
    return utility_mode == 2 ? -1 : 0;
}
void crispasr_audio_free(float* pcm) {
    if (pcm) {
        --utility_live;
        free(pcm);
    }
}
unsigned char* crispasr_pcm_to_wav(const float* pcm, int n, int rate, size_t* size) {
    unsigned char* p = malloc(4);
    memcpy(p, "W\0AV", 4);
    *size = utility_mode == 5 ? 0 : utility_mode == 6 ? SIZE_MAX : 4;
    ++utility_live;
    return p;
}
unsigned char* crispasr_c2pa_sign(const unsigned char* data, size_t len, const char* format, const char* cert,
                                  const char* key, size_t* size) {
    if (utility_mode == 2) {
        *size = 0;
        return NULL;
    }
    unsigned char* p = malloc(len);
    memcpy(p, data, len);
    *size = utility_mode == 5 ? 0 : utility_mode == 6 ? SIZE_MAX : len;
    ++utility_live;
    return p;
}
void crispasr_c2pa_free(unsigned char* p) {
    if (p) {
        --utility_live;
        free(p);
    }
}
#ifndef CRISPASR_PROBE_AUDIO_ONLY
int crispasr_watermark_load_model(const char* path) {
    return utility_mode == 2 ? -1 : 0;
}
int crispasr_session_score_texts(crispasr_session* s, const float* pcm, int n, const char* lang, const char* prompt,
                                 const char** texts, int count, float* scores, int* tokens) {
    for (int i = 0; i < count; ++i) {
        scores[i] = -(float)i;
        tokens[i] = (int)strlen(texts[i]);
    }
    return 0;
}

bool crispasr_kokoro_lang_is_german_abi(const char* lang) {
    return !strcmp(lang, "de");
}
bool crispasr_kokoro_lang_has_native_voice_abi(const char* lang) {
    return !strcmp(lang, "en");
}

#endif

#ifdef CRISPASR_PROBE_JAVA_RNNOISE
int crispasr_enhance_audio_rnnoise(const float* pcm, int32_t n, float* out, int32_t cap) {
    if (!pcm || n <= 0 || cap < n)
        return -1;
    if (utility_mode == 2)
        return -2;
    if (utility_mode == 4)
        return 1;
    for (int i = 0; i < n; ++i)
        out[i] = pcm[i] * .5f;
    return 0;
}
#endif
