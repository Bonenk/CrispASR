// openvoice2.h — OpenVoice V2 Tone Color Converter (voice cloning)
//
// Converts source audio to match a reference speaker's voice timbre.
// Architecture: ReferenceEncoder → speaker embedding,
//               PosteriorEncoder (WaveNet) → latent z,
//               ResidualCouplingFlow (WaveNet) forward/reverse → voice conversion,
//               HiFi-GAN decoder → waveform.
//
// Usage:
//   auto * ctx = openvoice2_init_from_file("openvoice2-tcc-f16.gguf", params);
//   openvoice2_convert(ctx, src_pcm, n_src, src_sr, ref_pcm, n_ref, ref_sr, &out_pcm, &n_out);
//   openvoice2_free(ctx);

#ifndef OPENVOICE2_H
#define OPENVOICE2_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct openvoice2_context;

struct openvoice2_context_params {
    int n_threads;
    int verbosity; // 0=silent, 1=info, 2=debug
    bool use_gpu;
    float tau; // posterior encoder sampling temperature (default 0: deterministic; upstream uses 0.3)
};

struct openvoice2_context_params openvoice2_context_default_params(void);

// Load the Tone Color Converter GGUF model.
struct openvoice2_context* openvoice2_init_from_file(const char* path, struct openvoice2_context_params params);

// Native converted output rate from the loaded model metadata (0 for null).
int openvoice2_sample_rate(const struct openvoice2_context* ctx);

// Convert source audio to match reference speaker's voice.
// src_pcm/ref_pcm: mono float PCM (any sample rate, resampled internally to 22050).
// out_pcm: allocated by callee, caller must free().
// Returns true on success.
bool openvoice2_convert(struct openvoice2_context* ctx, const float* src_pcm, int n_src, int src_sr,
                        const float* ref_pcm, int n_ref, int ref_sr, float** out_pcm, int* n_out);

// Extract speaker embedding from reference audio (256-d).
// Returns true on success.
bool openvoice2_extract_speaker_embedding(struct openvoice2_context* ctx, const float* ref_pcm, int n_ref, int ref_sr,
                                          float* out_embedding); // must point to float[256]

void openvoice2_free(struct openvoice2_context* ctx);

// Diff harness: dump intermediates to directory.
// Explicit source identity; rejects unknown names without changing the selection.
// NULL clears selection and extracts identity from the actual source PCM.
bool openvoice2_set_source_speaker(struct openvoice2_context* ctx, const char* name);

// Average independently encoded, ordered non-overlapping speech clips.
// Starts/ends are sample indices in ref_pcm (end exclusive); no VAD is run.
// This matches upstream extract_se(list_of_clips), including equal clip weights.
bool openvoice2_extract_speaker_embedding_segments(struct openvoice2_context* ctx, const float* ref_pcm, int n_ref,
                                                   int ref_sr, const int* starts, const int* ends, int n_segments,
                                                   float* out_embedding);

// Optional caller-prepared target identity (256 finite floats); NULL clears it.
// Copied on success; an invalid request leaves the prior override unchanged.
bool openvoice2_set_target_embedding(struct openvoice2_context* ctx, const float* embedding);

// Diagnostic teacher forcing: owned standard-normal posterior noise in (T,C)
// order. NULL/0 clears it. A conversion rejects a mismatched frame count.
bool openvoice2_set_posterior_noise(struct openvoice2_context* ctx, const float* noise, int n);

// Synchronous diagnostic snapshots, valid only during callback; no re-entry.
typedef void (*openvoice2_stage_callback)(const char*, const float*, size_t, void*);
void openvoice2_set_stage_callback(struct openvoice2_context* ctx, openvoice2_stage_callback callback, void* user);

void openvoice2_set_dump_dir(struct openvoice2_context* ctx, const char* dir);

#ifdef __cplusplus
}
#endif

#endif // OPENVOICE2_H
