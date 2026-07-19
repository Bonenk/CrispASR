// src/htdemucs.cpp — HTDemucs source separation runtime.
//
// Phase 1: GGUF loader + STFT/iSTFT + weight loading.
// Phase 2: encoder + decoder U-Net.
// Phase 3: CrossTransformer.
// Phase 4: Full forward + overlap-add chunking.

#include "htdemucs.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "core/gguf_loader.h"
#include "core/fft.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Debug gating
// ---------------------------------------------------------------------------
static bool htdemucs_debug() {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("CRISPASR_HTDEMUCS_DEBUG");
        v = (e && atoi(e) != 0) ? 1 : 0;
    }
    return v != 0;
}

// ---------------------------------------------------------------------------
// Hparams
// ---------------------------------------------------------------------------
struct htdemucs_hparams {
    int audio_channels = 2;
    int channels = 48;
    int nfft = 4096;
    int depth = 4;
    int bottom_channels = 512;
    int samplerate = 44100;
    float segment = 7.8f;
    bool cac = true;
    int kernel_size = 8;
    int stride = 4;
    int context = 1;
    int n_sources = 4;
    int dconv_depth = 2;
    int dconv_compress = 8;
    bool has_rewrite = true;
    bool has_freq_emb = true;
    float freq_emb_scale = 0.2f;

    // Transformer
    int t_layers = 5;
    int t_heads = 8;
    float t_max_period = 10000.0f;
    float t_weight_pos_embed = 1.0f;
    int t_classic_parity = 0; // 0 = cross-first-is-odd, 1 = cross-first-is-even

    int hop_length() const { return nfft / 4; }
    int training_length() const { return (int)(segment * samplerate); }
};

// ---------------------------------------------------------------------------
// Per-layer weight structs
// ---------------------------------------------------------------------------

// DConv: 2 residual sub-layers, each: Conv1d→GroupNorm→GELU→Conv1d→GroupNorm→GLU→LayerScale
struct htdemucs_dconv_sublayer {
    ggml_tensor* conv1_w = nullptr; // dilated conv weight
    ggml_tensor* conv1_b = nullptr;
    ggml_tensor* norm1_w = nullptr; // GroupNorm(1)
    ggml_tensor* norm1_b = nullptr;
    ggml_tensor* conv2_w = nullptr; // 1x1 → 2*channels
    ggml_tensor* conv2_b = nullptr;
    ggml_tensor* norm2_w = nullptr; // GroupNorm(1)
    ggml_tensor* norm2_b = nullptr;
    ggml_tensor* scale = nullptr; // LayerScale
};

struct htdemucs_dconv {
    std::vector<htdemucs_dconv_sublayer> layers; // dconv_depth layers
};

// Encoder layer (freq branch = Conv2d, time branch = Conv1d)
struct htdemucs_enc_layer {
    ggml_tensor* conv_w = nullptr; // main conv
    ggml_tensor* conv_b = nullptr;
    ggml_tensor* norm1_w = nullptr; // GroupNorm after conv (if not empty)
    ggml_tensor* norm1_b = nullptr;
    ggml_tensor* rewrite_w = nullptr; // 1x1 rewrite → 2*ch (GLU)
    ggml_tensor* rewrite_b = nullptr;
    ggml_tensor* norm2_w = nullptr; // GroupNorm after rewrite
    ggml_tensor* norm2_b = nullptr;
    htdemucs_dconv dconv;
    bool empty = false; // last freq layer before merge
    bool freq = true;   // true = freq (Conv2d), false = time (Conv1d)
    bool has_norm = false;
    int kernel_size = 8;
    int stride_val = 4;
    int pad = 2;
};

// Decoder layer
struct htdemucs_dec_layer {
    ggml_tensor* conv_tr_w = nullptr; // ConvTranspose
    ggml_tensor* conv_tr_b = nullptr;
    ggml_tensor* norm2_w = nullptr; // GroupNorm after ConvTranspose
    ggml_tensor* norm2_b = nullptr;
    ggml_tensor* rewrite_w = nullptr;
    ggml_tensor* rewrite_b = nullptr;
    ggml_tensor* norm1_w = nullptr;
    ggml_tensor* norm1_b = nullptr;
    htdemucs_dconv dconv;
    bool empty = false;
    bool freq = true;
    bool last = false;
    int kernel_size = 8;
    int stride_val = 4;
    int pad = 2;
};

// CrossTransformer self-attention layer
struct htdemucs_self_attn_layer {
    ggml_tensor* in_proj_w = nullptr; // (3*d, d)
    ggml_tensor* in_proj_b = nullptr;
    ggml_tensor* out_proj_w = nullptr; // (d, d)
    ggml_tensor* out_proj_b = nullptr;
    ggml_tensor* norm1_w = nullptr;
    ggml_tensor* norm1_b = nullptr;
    ggml_tensor* norm2_w = nullptr;
    ggml_tensor* norm2_b = nullptr;
    ggml_tensor* linear1_w = nullptr; // FFN
    ggml_tensor* linear1_b = nullptr;
    ggml_tensor* linear2_w = nullptr;
    ggml_tensor* linear2_b = nullptr;
    ggml_tensor* gamma1_scale = nullptr; // LayerScale
    ggml_tensor* gamma2_scale = nullptr;
    ggml_tensor* norm_out_w = nullptr; // optional norm_out (GroupNorm(1))
    ggml_tensor* norm_out_b = nullptr;
};

// CrossTransformer cross-attention layer
struct htdemucs_cross_attn_layer {
    ggml_tensor* cross_attn_in_proj_w = nullptr;
    ggml_tensor* cross_attn_in_proj_b = nullptr;
    ggml_tensor* cross_attn_out_proj_w = nullptr;
    ggml_tensor* cross_attn_out_proj_b = nullptr;
    ggml_tensor* norm1_w = nullptr;
    ggml_tensor* norm1_b = nullptr;
    ggml_tensor* norm2_w = nullptr;
    ggml_tensor* norm2_b = nullptr;
    ggml_tensor* norm3_w = nullptr;
    ggml_tensor* norm3_b = nullptr;
    ggml_tensor* linear1_w = nullptr;
    ggml_tensor* linear1_b = nullptr;
    ggml_tensor* linear2_w = nullptr;
    ggml_tensor* linear2_b = nullptr;
    ggml_tensor* gamma1_scale = nullptr;
    ggml_tensor* gamma2_scale = nullptr;
    ggml_tensor* norm_out_w = nullptr;
    ggml_tensor* norm_out_b = nullptr;
};

struct htdemucs_transformer_layer {
    bool is_cross = false; // false = self-attn, true = cross-attn
    htdemucs_self_attn_layer self_attn;
    htdemucs_cross_attn_layer cross_attn;
};

// Full model weights
struct htdemucs_model {
    htdemucs_hparams hparams;

    // Encoder: depth freq layers + (depth-1) time layers
    std::vector<htdemucs_enc_layer> encoder;  // freq branch
    std::vector<htdemucs_enc_layer> tencoder; // time branch

    // Decoder: depth freq layers + (depth-1) time layers
    std::vector<htdemucs_dec_layer> decoder;
    std::vector<htdemucs_dec_layer> tdecoder;

    // Frequency embedding
    ggml_tensor* freq_emb_w = nullptr; // (n_freqs, channels)

    // Channel up/downsamplers around transformer
    ggml_tensor* channel_up_w = nullptr; // (bottom_ch, transformer_ch, 1)
    ggml_tensor* channel_up_b = nullptr;
    ggml_tensor* channel_down_w = nullptr;
    ggml_tensor* channel_down_b = nullptr;
    ggml_tensor* channel_up_t_w = nullptr;
    ggml_tensor* channel_up_t_b = nullptr;
    ggml_tensor* channel_down_t_w = nullptr;
    ggml_tensor* channel_down_t_b = nullptr;

    // CrossTransformer
    ggml_tensor* norm_in_w = nullptr; // LayerNorm for spec input
    ggml_tensor* norm_in_b = nullptr;
    ggml_tensor* norm_in_t_w = nullptr; // LayerNorm for time input
    ggml_tensor* norm_in_t_b = nullptr;
    std::vector<htdemucs_transformer_layer> ct_layers;   // spec layers
    std::vector<htdemucs_transformer_layer> ct_layers_t; // time layers

    // Source names
    std::vector<std::string> source_names;
};

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------
struct htdemucs_context {
    htdemucs_model model;
    htdemucs_params params;

    // ggml state
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    ggml_context* ctx_w = nullptr;

    // Precomputed Hann window (nfft)
    std::vector<float> hann_window;
};

// ---------------------------------------------------------------------------
// GGUF loading
// ---------------------------------------------------------------------------

static bool load_hparams(htdemucs_hparams& hp, gguf_context* meta) {
    hp.audio_channels = core_gguf::kv_u32(meta, "htdemucs.audio_channels", 2);
    hp.channels = core_gguf::kv_u32(meta, "htdemucs.channels", 48);
    hp.nfft = core_gguf::kv_u32(meta, "htdemucs.nfft", 4096);
    hp.depth = core_gguf::kv_u32(meta, "htdemucs.depth", 4);
    hp.bottom_channels = core_gguf::kv_u32(meta, "htdemucs.bottom_channels", 512);
    hp.samplerate = core_gguf::kv_u32(meta, "htdemucs.samplerate", 44100);
    hp.segment = core_gguf::kv_f32(meta, "htdemucs.segment", 7.8f);
    hp.cac = core_gguf::kv_u32(meta, "htdemucs.cac", 1) != 0;
    hp.kernel_size = core_gguf::kv_u32(meta, "htdemucs.kernel_size", 8);
    hp.stride = core_gguf::kv_u32(meta, "htdemucs.stride", 4);
    hp.context = core_gguf::kv_u32(meta, "htdemucs.context", 1);
    hp.n_sources = core_gguf::kv_u32(meta, "htdemucs.n_sources", 4);
    hp.dconv_depth = core_gguf::kv_u32(meta, "htdemucs.dconv_depth", 2);
    hp.dconv_compress = core_gguf::kv_u32(meta, "htdemucs.dconv_compress", 8);
    hp.has_rewrite = core_gguf::kv_u32(meta, "htdemucs.has_rewrite", 1) != 0;
    hp.has_freq_emb = core_gguf::kv_u32(meta, "htdemucs.has_freq_emb", 1) != 0;
    hp.freq_emb_scale = core_gguf::kv_f32(meta, "htdemucs.freq_emb_scale", 0.2f);
    hp.t_layers = core_gguf::kv_u32(meta, "htdemucs.t_layers", 5);
    hp.t_heads = core_gguf::kv_u32(meta, "htdemucs.t_heads", 8);
    hp.t_max_period = core_gguf::kv_f32(meta, "htdemucs.t_max_period", 10000.0f);
    hp.t_weight_pos_embed = core_gguf::kv_f32(meta, "htdemucs.t_weight_pos_embed", 1.0f);
    hp.t_classic_parity = core_gguf::kv_u32(meta, "htdemucs.t_classic_parity", 0);
    return true;
}

// Placeholder: full weight loading will be Phase 2.
// For now just load hparams and verify the GGUF structure.

static htdemucs_context* htdemucs_init_impl(const char* model_path, htdemucs_params params) {
    auto ctx = new htdemucs_context();
    ctx->params = params;

    // Pass 1: read metadata / hparams
    gguf_context* meta = core_gguf::open_metadata(model_path);
    if (!meta) {
        fprintf(stderr, "htdemucs: failed to open %s\n", model_path);
        delete ctx;
        return nullptr;
    }

    if (!load_hparams(ctx->model.hparams, meta)) {
        fprintf(stderr, "htdemucs: failed to read hparams from %s\n", model_path);
        core_gguf::free_metadata(meta);
        delete ctx;
        return nullptr;
    }
    core_gguf::free_metadata(meta);

    auto& hp = ctx->model.hparams;
    fprintf(stderr,
            "htdemucs: loaded hparams: depth=%d, channels=%d, nfft=%d, "
            "samplerate=%d, segment=%.1f, t_layers=%d\n",
            hp.depth, hp.channels, hp.nfft, hp.samplerate, hp.segment, hp.t_layers);

    // Pass 2: load weights
    ctx->backend = ggml_backend_cpu_init();
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(model_path, ctx->backend, "htdemucs", wl)) {
        fprintf(stderr, "htdemucs: failed to load weights from %s\n", model_path);
        delete ctx;
        return nullptr;
    }
    ctx->ctx_w = wl.ctx;
    ctx->buf_w = wl.buf;

    fprintf(stderr, "htdemucs: loaded %zu tensors\n", wl.tensors.size());

    // TODO: bind named tensors into model struct fields

    // Source names
    ctx->model.source_names = {"drums", "bass", "other", "vocals"};

    // Precompute Hann window (periodic, matching torch.hann_window)
    ctx->hann_window.resize(hp.nfft);
    for (int i = 0; i < hp.nfft; i++) {
        ctx->hann_window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / hp.nfft));
    }

    return ctx;
}

// ---------------------------------------------------------------------------
// CPU STFT (complex output)
// ---------------------------------------------------------------------------

// Forward STFT with center=True, normalized=True, Hann window.
// Input: pcm[n_channels][n_samples]
// Output: complex spectrogram as separate real/imag arrays.
//   real[n_channels][n_freqs][n_frames], imag[same]
//   where n_freqs = nfft/2 + 1.
struct stft_result {
    int n_channels;
    int n_freqs;
    int n_frames;
    std::vector<float> real; // [n_channels * n_freqs * n_frames]
    std::vector<float> imag;
};

static stft_result compute_stft(const float* pcm_channels, int n_channels, int n_samples, int nfft, int hop,
                                const float* window) {
    stft_result r;
    r.n_channels = n_channels;
    r.n_freqs = nfft / 2 + 1;

    // center=True: pad nfft/2 on each side
    int pad = nfft / 2;
    int padded_len = n_samples + 2 * pad;

    // Reflect padding
    std::vector<float> padded(padded_len);

    r.n_frames = (padded_len - nfft) / hop + 1;
    size_t spec_size = (size_t)n_channels * r.n_freqs * r.n_frames;
    r.real.resize(spec_size, 0.0f);
    r.imag.resize(spec_size, 0.0f);

    // Normalization factor for normalized=True
    float norm_factor = 1.0f / sqrtf((float)nfft);

    // Temp buffers for FFT (need power-of-2 for radix-2)
    // nfft=4096 is already power of 2
    std::vector<float> fft_re(nfft), fft_im(nfft);

    for (int ch = 0; ch < n_channels; ch++) {
        const float* src = pcm_channels + (size_t)ch * n_samples;

        // Reflect pad
        for (int i = 0; i < padded_len; i++) {
            int idx = i - pad;
            if (idx < 0)
                idx = -idx; // reflect left
            if (idx >= n_samples)
                idx = 2 * n_samples - 2 - idx; // reflect right
            idx = std::max(0, std::min(n_samples - 1, idx));
            padded[i] = src[idx];
        }

        for (int f = 0; f < r.n_frames; f++) {
            int start = f * hop;
            // Apply window and copy to FFT buffer
            for (int i = 0; i < nfft; i++) {
                fft_re[i] = padded[start + i] * window[i];
                fft_im[i] = 0.0f;
            }
            // In-place radix-2 FFT
            core_fft::fft_radix2_inplace(fft_re.data(), fft_im.data(), nfft);

            // Store half-spectrum (n_freqs = nfft/2+1), normalized
            size_t base = (size_t)ch * r.n_freqs * r.n_frames + (size_t)f;
            for (int k = 0; k < r.n_freqs; k++) {
                r.real[base + (size_t)k * r.n_frames] = fft_re[k] * norm_factor;
                r.imag[base + (size_t)k * r.n_frames] = fft_im[k] * norm_factor;
            }
        }
    }
    return r;
}

// Inverse STFT
static void compute_istft(const float* real, const float* imag, int n_channels, int n_freqs, int n_frames, int nfft,
                          int hop, const float* window, int output_length, float* out) {
    float norm_factor = 1.0f / sqrtf((float)nfft);
    int pad = nfft / 2;

    std::vector<float> fft_re(nfft), fft_im(nfft);
    std::vector<float> win_sum;

    for (int ch = 0; ch < n_channels; ch++) {
        int out_len = (n_frames - 1) * hop + nfft;
        std::vector<float> signal(out_len, 0.0f);
        win_sum.assign(out_len, 0.0f);

        size_t ch_base = (size_t)ch * n_freqs * n_frames;

        for (int f = 0; f < n_frames; f++) {
            // Reconstruct full spectrum from half
            for (int k = 0; k < n_freqs; k++) {
                fft_re[k] = real[ch_base + (size_t)k * n_frames + f] * norm_factor;
                fft_im[k] = imag[ch_base + (size_t)k * n_frames + f] * norm_factor;
            }
            // Mirror for negative frequencies
            for (int k = n_freqs; k < nfft; k++) {
                fft_re[k] = fft_re[nfft - k];
                fft_im[k] = -fft_im[nfft - k];
            }

            // Inverse FFT = conjugate + forward FFT + conjugate + scale
            for (int k = 0; k < nfft; k++)
                fft_im[k] = -fft_im[k];
            core_fft::fft_radix2_inplace(fft_re.data(), fft_im.data(), nfft);
            float scale = 1.0f / (float)nfft;
            for (int k = 0; k < nfft; k++) {
                fft_re[k] *= scale;
                fft_im[k] = -fft_im[k] * scale; // unused but correct
            }

            // Overlap-add with window
            int start = f * hop;
            for (int i = 0; i < nfft; i++) {
                signal[start + i] += fft_re[i] * window[i];
                win_sum[start + i] += window[i] * window[i];
            }
        }

        // Normalize by window sum
        for (int i = 0; i < out_len; i++) {
            if (win_sum[i] > 1e-8f)
                signal[i] /= win_sum[i];
        }

        // Remove center padding and copy to output
        float* dst = out + (size_t)ch * output_length;
        for (int i = 0; i < output_length; i++) {
            dst[i] = signal[pad + i];
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

htdemucs_params htdemucs_default_params(void) {
    htdemucs_params p;
    memset(&p, 0, sizeof(p));
    p.n_threads = 0;
    p.use_gpu = false;
    p.gpu_device = 0;
    return p;
}

htdemucs_context* htdemucs_init_from_file(const char* model_path, htdemucs_params params) {
    return htdemucs_init_impl(model_path, params);
}

void htdemucs_free(htdemucs_context* ctx) {
    if (!ctx)
        return;
    if (ctx->buf_w)
        ggml_backend_buffer_free(ctx->buf_w);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

htdemucs_result* htdemucs_separate(htdemucs_context* ctx, const float* pcm_stereo, int n_samples) {
    if (!ctx || !pcm_stereo || n_samples <= 0)
        return nullptr;

    // TODO Phase 2: full forward pass
    // For now, return a stub result with zeros to verify the pipeline compiles

    auto& hp = ctx->model.hparams;
    auto r = new htdemucs_result();
    r->n_sources = hp.n_sources;
    r->n_channels = hp.audio_channels;
    r->n_samples = n_samples;
    r->sample_rate = hp.samplerate;

    r->sources = new float*[hp.n_sources];
    r->source_names = new const char*[hp.n_sources];
    for (int s = 0; s < hp.n_sources; s++) {
        size_t sz = (size_t)hp.audio_channels * n_samples;
        r->sources[s] = new float[sz](); // zero-init
        r->source_names[s] = ctx->model.source_names[s].c_str();
    }

    fprintf(stderr, "htdemucs: separated %d samples → %d sources (STUB — weights not loaded yet)\n", n_samples,
            hp.n_sources);

    return r;
}

void htdemucs_result_free(htdemucs_result* r) {
    if (!r)
        return;
    for (int s = 0; s < r->n_sources; s++)
        delete[] r->sources[s];
    delete[] r->sources;
    delete[] r->source_names;
    delete r;
}

int htdemucs_sample_rate(const htdemucs_context* ctx) {
    return ctx ? ctx->model.hparams.samplerate : 44100;
}

int htdemucs_n_sources(const htdemucs_context* ctx) {
    return ctx ? ctx->model.hparams.n_sources : 4;
}

const char* htdemucs_source_name(const htdemucs_context* ctx, int idx) {
    if (!ctx || idx < 0 || idx >= (int)ctx->model.source_names.size())
        return nullptr;
    return ctx->model.source_names[idx].c_str();
}
