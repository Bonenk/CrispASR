// src/beatrice_pitch.h — Beatrice v2 PitchEstimator (Project Beatrice, MIT).
//
// The first of Beatrice's three networks. Takes 16 kHz mono PCM and produces,
// at a 100 Hz frame rate, 448 pitch-bin logits plus a per-frame energy — and
// the banded-argmax quantised pitch derived from them.
//
// Beatrice's LICENCE IS MIT, source and trained models alike
// (fierce-cats/beatrice-trainer). Unlike RVC this needs no acceptance gate. The
// GGUF still carries its own tag because a checkpoint from some other training
// run need not share the base's terms.
//
// UNLIKE the rest of the model, this component is DETERMINISTIC — the RNG in
// Beatrice lives in the vocoder's overlap_add, not here. So this one can be
// validated by direct comparison, with no noise injection.
//
// See docs/music-transcription/BEATRICE_BLUEPRINT.md for the ~10 non-obvious
// details this reproduces, and for what the parity harness cannot see.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct beatrice_pitch_context;

struct beatrice_pitch_params {
    int n_threads; // 0 = auto
    bool use_gpu;
    int gpu_device;
};

beatrice_pitch_params beatrice_pitch_default_params(void);

beatrice_pitch_context* beatrice_pitch_init_from_file(const char* model_path, beatrice_pitch_params params);
void beatrice_pitch_free(beatrice_pitch_context* ctx);

// Geometry, read from the GGUF rather than assumed.
int beatrice_pitch_n_bins(const beatrice_pitch_context* ctx);          // 448
int beatrice_pitch_bins_per_octave(const beatrice_pitch_context* ctx); // 96
int beatrice_pitch_sample_rate(const beatrice_pitch_context* ctx);     // 16000
int beatrice_pitch_hop_length(const beatrice_pitch_context* ctx);      // 160 -> 100 Hz

struct beatrice_pitch_result {
    // BIN-MAJOR, TIME FASTEST: element (bin, frame) at bin*n_frames + frame.
    // This is the ggml-native layout of the graph output and matches the parity
    // reference, so nothing has to transpose. It is NOT frame-major.
    float* logits;  // n_bins * n_frames
    int* quantized; // n_frames, banded argmax; 0 marks unvoiced
    float* energy;  // n_frames
    int n_frames;
    int n_bins;
};

// `pcm` is mono 16 kHz. Frame count is
// floor((n_samples + (win-hop) - win) / hop) + 1 with win=560, hop=160.
beatrice_pitch_result* beatrice_pitch_estimate(beatrice_pitch_context* ctx, const float* pcm, int n_samples);
void beatrice_pitch_result_free(beatrice_pitch_result* r);

// Per-stage parity diff against tools/beatrice_torch_parity.py's dump.
// Returns 0 when every stage passes, 1 on a parity failure, 2 on a load error.
int beatrice_pitch_diff(const char* model_gguf, const char* ref_gguf, int verbosity);

#ifdef __cplusplus
}
#endif
