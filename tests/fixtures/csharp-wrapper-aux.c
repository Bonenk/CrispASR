#include "crispasr_session.h"
#include "runtime-abi-layout.h"
#include <stdlib.h>
#include <string.h>
static int mode, live;
void probe_aux_mode(int value) {
    mode = value;
}
int probe_aux_live(void) {
    return live;
}
int probe_aux_segment_size(void) {
    return sizeof(crispasr_diarize_seg_abi);
}
int probe_aux_options_size(void) {
    return sizeof(crispasr_diarize_opts_abi);
}
void* crispasr_speaker_embedder_make_abi(const char* model, int threads, const char* cache) {
    ++live;
    return malloc(1);
}
void crispasr_speaker_embedder_free_abi(void* p) {
    if (p) {
        --live;
        free(p);
    }
}
int32_t crispasr_speaker_embedder_dim_abi(const void* p) {
    return mode == 17 ? 0 : 3;
}
const char* crispasr_speaker_embedder_name_abi(const void* p) {
    return "embed-probe";
}
int32_t crispasr_speaker_embedder_embed_abi(void* p, const float* pcm, int n, float* out) {
    if (mode == 2)
        return 0;
    if (mode == 15)
        return 2;
    if (mode == 16)
        return -1;
    out[0] = .25f;
    out[1] = .5f;
    out[2] = .75f;
    return 1;
}
int32_t crispasr_speaker_cluster_abi(const float* embeddings, int n, int dim, float threshold, int max, int* labels) {
    if (dim != 3 || n != 2)
        return -1;
    labels[0] = mode == 14 ? -1 : 0;
    labels[1] = mode == 14 ? -1 : (mode == 12 ? 2 : 1);
    if (mode == 11)
        return 3;
    if (mode == 13)
        return -1;
    if (mode == 14)
        return 0;
    return 2;
}
void* crispasr_pyannote_cache_compute_abi(const float* pcm, int n, const char* model, int threads) {
    ++live;
    return malloc(1);
}
void crispasr_pyannote_cache_free_abi(void* p) {
    if (p) {
        --live;
        free(p);
    }
}
int32_t crispasr_pyannote_cache_apply_abi(const void* cache, int64_t offset, crispasr_diarize_seg_abi* segs, int n) {
    if (offset != 250 || n != 1)
        return -1;
    segs[0].speaker = 2;
    return 0;
}
static int diarize(crispasr_diarize_seg_abi* segs, int n, const crispasr_diarize_opts_abi* opts) {
    if (n != 1 || segs[0].t0_cs != 100 || segs[0].t1_cs != 200 || opts->method != 4 || opts->n_threads != 7 ||
        opts->slice_t0_cs != 125 || !opts->pyannote_model_path || strcmp(opts->pyannote_model_path, "model") ||
        !opts->foxnose_embedder_path || strcmp(opts->foxnose_embedder_path, "embed") || opts->min_speakers != 1 ||
        opts->max_speakers != 3 || opts->num_speakers != 2)
        return 1;
    segs[0].speaker = 1;
    return 0;
}
int crispasr_diarize_segments_abi(const float* left, const float* right, int n, int stereo,
                                  crispasr_diarize_seg_abi* segs, int count, const crispasr_diarize_opts_abi* opts) {
    return diarize(segs, count, opts);
}
int crispasr_diarize_segments_turns_abi(const float* left, const float* right, int n, int stereo,
                                        crispasr_diarize_seg_abi* segs, int count,
                                        const crispasr_diarize_opts_abi* opts, crispasr_diarize_turn_abi* turns,
                                        int cap, int* needed) {
    int rc = diarize(segs, count, opts);
    if (rc)
        return rc;
    *needed = 1;
    if (!turns)
        return 0;
    if (cap < 1)
        return 2;
    turns[0].t0_cs = 125;
    turns[0].t1_cs = 250;
    turns[0].speaker = 1;
    turns[0]._pad = 0;
    return 0;
}
int crispasr_text_detect_language(const char* text, const char* model, int threads, char* label, int cap,
                                  float* confidence) {
    if (mode == 2)
        return 1;
    if (cap < 3)
        return 2;
    strcpy(label, "de");
    *confidence = .875f;
    return 0;
}
int crispasr_vad_slices(const char* model, const float* pcm, int n, int rate, float threshold, int minSpeech,
                        int minSilence, int pad, float maxChunk, int threads, float** spans) {
    if (mode == 1) {
        *spans = NULL;
        return 0;
    }
    *spans = malloc(2 * sizeof(float));
    ++live;
    (*spans)[0] = 1.25f;
    (*spans)[1] = 2.5f;
    return mode == 2 ? -3 : 1;
}
void crispasr_vad_free(float* p) {
    if (p) {
        --live;
        free(p);
    }
}
static int entry(const char* name, char* filename, int fcap, char* url, int ucap, char* size, int scap) {
    if (!strcmp(name, "missing"))
        return 1;
    if (fcap < 11 || ucap < 18 || scap < 4)
        return -2;
    strcpy(filename, "model.gguf");
    strcpy(url, "https://test/model");
    strcpy(size, "1MB");
    return 0;
}
int crispasr_registry_lookup_abi(const char* name, char* file, int fcap, char* url, int ucap, char* size, int scap) {
    return entry(name, file, fcap, url, ucap, size, scap);
}
int crispasr_registry_lookup_by_filename_abi(const char* name, char* file, int fcap, char* url, int ucap, char* size,
                                             int scap) {
    return entry(name, file, fcap, url, ucap, size, scap);
}
int crispasr_registry_list_backends_abi(char* data, int cap) {
    strcpy(data, "foo,bar");
    return 7;
}
int crispasr_registry_default_bundle_info_abi(const char* backend, char* name, int ncap, char* license, int lcap,
                                              int* acceptance) {
    strcpy(name, "foo");
    strcpy(license, "Apache-2.0");
    *acceptance = 0;
    return 1;
}
int crispasr_registry_default_bundle_artifact_abi(const char* backend, int index, int* kind, char* filename, int fcap,
                                                  char* url, int ucap, char* size, int scap) {
    *kind = 1;
    return entry(backend, filename, fcap, url, ucap, size, scap);
}
int crispasr_cache_dir_abi(const char* directory, char* path, int cap) {
    if (mode == 2)
        return 2;
    strcpy(path, "cache");
    return 0;
}
int crispasr_cache_ensure_file_abi(const char* filename, const char* url, int quiet, const char* directory, char* path,
                                   int cap) {
    if (mode == 2)
        return 1;
    strcpy(path, "cache/model.gguf");
    return 0;
}
bool crispasr_kokoro_lang_is_german_abi(const char* lang) {
    return !strcmp(lang, "de");
}
bool crispasr_kokoro_lang_has_native_voice_abi(const char* lang) {
    return !strcmp(lang, "en");
}
int crispasr_lcs_dedup_prefix_count(const int* previous, int a, const int* current, int b, int minimum) {
    return a && b && previous[a - 1] == current[0] ? 1 : 0;
}
