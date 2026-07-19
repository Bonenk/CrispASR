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

// ---------------------------------------------------------------------------
// Weight binding (Phase 2)
// ---------------------------------------------------------------------------

static void bind_dconv(htdemucs_dconv& dc, const core_gguf::tensor_map& t, const std::string& prefix, int depth) {
    dc.layers.resize(depth);
    for (int d = 0; d < depth; d++) {
        auto& sl = dc.layers[d];
        std::string p = prefix + "." + std::to_string(d);
        // Sequential: [0]=dilated_conv, [1]=groupnorm, [2]=GELU, [3]=1x1_conv, [4]=groupnorm, [5]=GLU, [6]=LayerScale
        sl.conv1_w = core_gguf::try_get(t, (p + ".0.weight").c_str());
        sl.conv1_b = core_gguf::try_get(t, (p + ".0.bias").c_str());
        sl.norm1_w = core_gguf::try_get(t, (p + ".1.weight").c_str());
        sl.norm1_b = core_gguf::try_get(t, (p + ".1.bias").c_str());
        sl.conv2_w = core_gguf::try_get(t, (p + ".3.weight").c_str());
        sl.conv2_b = core_gguf::try_get(t, (p + ".3.bias").c_str());
        sl.norm2_w = core_gguf::try_get(t, (p + ".4.weight").c_str());
        sl.norm2_b = core_gguf::try_get(t, (p + ".4.bias").c_str());
        sl.scale = core_gguf::try_get(t, (p + ".6.scale").c_str());
    }
}

static void bind_enc_layer(htdemucs_enc_layer& el, const core_gguf::tensor_map& t, const std::string& prefix) {
    el.conv_w = core_gguf::try_get(t, (prefix + ".conv.weight").c_str());
    el.conv_b = core_gguf::try_get(t, (prefix + ".conv.bias").c_str());
    el.norm1_w = core_gguf::try_get(t, (prefix + ".norm1.weight").c_str());
    el.norm1_b = core_gguf::try_get(t, (prefix + ".norm1.bias").c_str());
    el.rewrite_w = core_gguf::try_get(t, (prefix + ".rewrite.weight").c_str());
    el.rewrite_b = core_gguf::try_get(t, (prefix + ".rewrite.bias").c_str());
    el.norm2_w = core_gguf::try_get(t, (prefix + ".norm2.weight").c_str());
    el.norm2_b = core_gguf::try_get(t, (prefix + ".norm2.bias").c_str());
    el.has_norm = el.norm1_w != nullptr;
    el.empty = (el.norm1_w == nullptr && el.rewrite_w == nullptr);
}

static void bind_dec_layer(htdemucs_dec_layer& dl, const core_gguf::tensor_map& t, const std::string& prefix) {
    dl.conv_tr_w = core_gguf::try_get(t, (prefix + ".conv_tr.weight").c_str());
    dl.conv_tr_b = core_gguf::try_get(t, (prefix + ".conv_tr.bias").c_str());
    dl.norm2_w = core_gguf::try_get(t, (prefix + ".norm2.weight").c_str());
    dl.norm2_b = core_gguf::try_get(t, (prefix + ".norm2.bias").c_str());
    dl.rewrite_w = core_gguf::try_get(t, (prefix + ".rewrite.weight").c_str());
    dl.rewrite_b = core_gguf::try_get(t, (prefix + ".rewrite.bias").c_str());
    dl.norm1_w = core_gguf::try_get(t, (prefix + ".norm1.weight").c_str());
    dl.norm1_b = core_gguf::try_get(t, (prefix + ".norm1.bias").c_str());
    dl.empty = (dl.rewrite_w == nullptr);
}

static void bind_self_attn_layer(htdemucs_self_attn_layer& sa, const core_gguf::tensor_map& t,
                                 const std::string& prefix) {
    sa.in_proj_w = core_gguf::try_get(t, (prefix + ".self_attn.in_proj_weight").c_str());
    sa.in_proj_b = core_gguf::try_get(t, (prefix + ".self_attn.in_proj_bias").c_str());
    sa.out_proj_w = core_gguf::try_get(t, (prefix + ".self_attn.out_proj.weight").c_str());
    sa.out_proj_b = core_gguf::try_get(t, (prefix + ".self_attn.out_proj.bias").c_str());
    sa.norm1_w = core_gguf::try_get(t, (prefix + ".norm1.weight").c_str());
    sa.norm1_b = core_gguf::try_get(t, (prefix + ".norm1.bias").c_str());
    sa.norm2_w = core_gguf::try_get(t, (prefix + ".norm2.weight").c_str());
    sa.norm2_b = core_gguf::try_get(t, (prefix + ".norm2.bias").c_str());
    sa.linear1_w = core_gguf::try_get(t, (prefix + ".linear1.weight").c_str());
    sa.linear1_b = core_gguf::try_get(t, (prefix + ".linear1.bias").c_str());
    sa.linear2_w = core_gguf::try_get(t, (prefix + ".linear2.weight").c_str());
    sa.linear2_b = core_gguf::try_get(t, (prefix + ".linear2.bias").c_str());
    sa.gamma1_scale = core_gguf::try_get(t, (prefix + ".gamma_1.scale").c_str());
    sa.gamma2_scale = core_gguf::try_get(t, (prefix + ".gamma_2.scale").c_str());
    sa.norm_out_w = core_gguf::try_get(t, (prefix + ".norm_out.weight").c_str());
    sa.norm_out_b = core_gguf::try_get(t, (prefix + ".norm_out.bias").c_str());
}

static void bind_cross_attn_layer(htdemucs_cross_attn_layer& ca, const core_gguf::tensor_map& t,
                                  const std::string& prefix) {
    ca.cross_attn_in_proj_w = core_gguf::try_get(t, (prefix + ".cross_attn.in_proj_weight").c_str());
    ca.cross_attn_in_proj_b = core_gguf::try_get(t, (prefix + ".cross_attn.in_proj_bias").c_str());
    ca.cross_attn_out_proj_w = core_gguf::try_get(t, (prefix + ".cross_attn.out_proj.weight").c_str());
    ca.cross_attn_out_proj_b = core_gguf::try_get(t, (prefix + ".cross_attn.out_proj.bias").c_str());
    ca.norm1_w = core_gguf::try_get(t, (prefix + ".norm1.weight").c_str());
    ca.norm1_b = core_gguf::try_get(t, (prefix + ".norm1.bias").c_str());
    ca.norm2_w = core_gguf::try_get(t, (prefix + ".norm2.weight").c_str());
    ca.norm2_b = core_gguf::try_get(t, (prefix + ".norm2.bias").c_str());
    ca.norm3_w = core_gguf::try_get(t, (prefix + ".norm3.weight").c_str());
    ca.norm3_b = core_gguf::try_get(t, (prefix + ".norm3.bias").c_str());
    ca.linear1_w = core_gguf::try_get(t, (prefix + ".linear1.weight").c_str());
    ca.linear1_b = core_gguf::try_get(t, (prefix + ".linear1.bias").c_str());
    ca.linear2_w = core_gguf::try_get(t, (prefix + ".linear2.weight").c_str());
    ca.linear2_b = core_gguf::try_get(t, (prefix + ".linear2.bias").c_str());
    ca.gamma1_scale = core_gguf::try_get(t, (prefix + ".gamma_1.scale").c_str());
    ca.gamma2_scale = core_gguf::try_get(t, (prefix + ".gamma_2.scale").c_str());
    ca.norm_out_w = core_gguf::try_get(t, (prefix + ".norm_out.weight").c_str());
    ca.norm_out_b = core_gguf::try_get(t, (prefix + ".norm_out.bias").c_str());
}

static bool bind_weights(htdemucs_model& m, const core_gguf::tensor_map& t) {
    auto& hp = m.hparams;
    int bound = 0;

    // Encoder (freq branch)
    m.encoder.resize(hp.depth);
    for (int i = 0; i < hp.depth; i++) {
        std::string p = "encoder." + std::to_string(i);
        bind_enc_layer(m.encoder[i], t, p);
        if (m.encoder[i].conv_w)
            bound++;
        // DConv
        if (core_gguf::try_get(t, (p + ".dconv.layers.0.0.weight").c_str())) {
            bind_dconv(m.encoder[i].dconv, t, p + ".dconv.layers", hp.dconv_depth);
        }
        // Determine freq vs time, kernel/stride/pad from weight shape
        m.encoder[i].freq = (m.encoder[i].conv_w && ggml_n_dims(m.encoder[i].conv_w) == 4);
    }

    // Encoder (time branch) — as many layers as freq>1 layers.
    // For depth=4, nfft=4096: freqs = 2048→512→128→32→8→1 but depth=4 so:
    // layer 0: freq=True (2048→512), layer 1: freq=True (512→128),
    // layer 2: freq=True (128→32), layer 3: freq=True (32→8, last_freq=True, ker=32→8)
    // Wait, let me re-check: nfft/2=2048, stride=4:
    // layer 0: freqs=2048, freq=True. 2048>kernel_size(8). freqs=2048/4=512
    // layer 1: freqs=512, freq=True. 512>8. freqs=512/4=128
    // layer 2: freqs=128, freq=True. 128>8. freqs=128/4=32
    // layer 3: freqs=32, freq=True. 32>8. freqs=32/4=8
    // But depth=4, so we stop. The time encoders exist for layers where freq=True,
    // which is all 4 layers. But the LAST freq layer that makes freqs<=kernel_size
    // has empty=True for the time encoder.
    // Let me just count from the GGUF:
    m.tencoder.resize(0);
    for (int i = 0; i < hp.depth; i++) {
        std::string p = "tencoder." + std::to_string(i);
        auto* w = core_gguf::try_get(t, (p + ".conv.weight").c_str());
        if (!w)
            break;
        m.tencoder.resize(i + 1);
        bind_enc_layer(m.tencoder[i], t, p);
        m.tencoder[i].freq = false; // time branch always Conv1d
        if (core_gguf::try_get(t, (p + ".dconv.layers.0.0.weight").c_str())) {
            bind_dconv(m.tencoder[i].dconv, t, p + ".dconv.layers", hp.dconv_depth);
        }
        bound++;
    }

    // Decoder (freq branch)
    m.decoder.resize(hp.depth);
    for (int i = 0; i < hp.depth; i++) {
        std::string p = "decoder." + std::to_string(i);
        bind_dec_layer(m.decoder[i], t, p);
        m.decoder[i].freq = (m.decoder[i].conv_tr_w && ggml_n_dims(m.decoder[i].conv_tr_w) == 4);
        m.decoder[i].last = (i == hp.depth - 1);
        if (core_gguf::try_get(t, (p + ".dconv.layers.0.0.weight").c_str())) {
            bind_dconv(m.decoder[i].dconv, t, p + ".dconv.layers", hp.dconv_depth);
        }
        if (m.decoder[i].conv_tr_w)
            bound++;
    }

    // Decoder (time branch)
    m.tdecoder.resize(0);
    for (int i = 0; i < hp.depth; i++) {
        std::string p = "tdecoder." + std::to_string(i);
        auto* w = core_gguf::try_get(t, (p + ".conv_tr.weight").c_str());
        if (!w)
            break;
        m.tdecoder.resize(i + 1);
        bind_dec_layer(m.tdecoder[i], t, p);
        m.tdecoder[i].freq = false;
        m.tdecoder[i].last = (i == 0); // tdecoder is reversed
        if (core_gguf::try_get(t, (p + ".dconv.layers.0.0.weight").c_str())) {
            bind_dconv(m.tdecoder[i].dconv, t, p + ".dconv.layers", hp.dconv_depth);
        }
        bound++;
    }

    // Frequency embedding
    m.freq_emb_w = core_gguf::try_get(t, "freq_emb.embedding.weight");

    // Channel up/downsamplers
    m.channel_up_w = core_gguf::try_get(t, "channel_upsampler.weight");
    m.channel_up_b = core_gguf::try_get(t, "channel_upsampler.bias");
    m.channel_down_w = core_gguf::try_get(t, "channel_downsampler.weight");
    m.channel_down_b = core_gguf::try_get(t, "channel_downsampler.bias");
    m.channel_up_t_w = core_gguf::try_get(t, "channel_upsampler_t.weight");
    m.channel_up_t_b = core_gguf::try_get(t, "channel_upsampler_t.bias");
    m.channel_down_t_w = core_gguf::try_get(t, "channel_downsampler_t.weight");
    m.channel_down_t_b = core_gguf::try_get(t, "channel_downsampler_t.bias");

    // CrossTransformer norm_in
    m.norm_in_w = core_gguf::try_get(t, "crosstransformer.norm_in.weight");
    m.norm_in_b = core_gguf::try_get(t, "crosstransformer.norm_in.bias");
    m.norm_in_t_w = core_gguf::try_get(t, "crosstransformer.norm_in_t.weight");
    m.norm_in_t_b = core_gguf::try_get(t, "crosstransformer.norm_in_t.bias");

    // CrossTransformer layers (spec + time)
    m.ct_layers.resize(hp.t_layers);
    m.ct_layers_t.resize(hp.t_layers);
    for (int i = 0; i < hp.t_layers; i++) {
        bool is_cross = (i % 2 != hp.t_classic_parity);
        // Spec layers
        m.ct_layers[i].is_cross = is_cross;
        std::string sp = "crosstransformer.layers." + std::to_string(i);
        if (is_cross) {
            bind_cross_attn_layer(m.ct_layers[i].cross_attn, t, sp);
        } else {
            bind_self_attn_layer(m.ct_layers[i].self_attn, t, sp);
        }
        // Time layers
        m.ct_layers_t[i].is_cross = is_cross;
        std::string tp_str = "crosstransformer.layers_t." + std::to_string(i);
        if (is_cross) {
            bind_cross_attn_layer(m.ct_layers_t[i].cross_attn, t, tp_str);
        } else {
            bind_self_attn_layer(m.ct_layers_t[i].self_attn, t, tp_str);
        }
    }

    fprintf(stderr,
            "htdemucs: bound %d enc/dec layers, %d tenc, %d tdec, "
            "%d transformer layers\n",
            bound, (int)m.tencoder.size(), (int)m.tdecoder.size(), hp.t_layers);
    return true;
}

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

    if (!bind_weights(ctx->model, wl.tensors)) {
        fprintf(stderr, "htdemucs: failed to bind weights\n");
        delete ctx;
        return nullptr;
    }

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

// ---------------------------------------------------------------------------
// ggml graph helpers for the encoder/decoder/transformer
// ---------------------------------------------------------------------------

// GroupNorm + affine: y = weight * group_norm(x) + bias
static ggml_tensor* ggml_group_norm_affine(ggml_context* g, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b,
                                           int n_groups, float eps) {
    ggml_tensor* y = ggml_group_norm(g, x, n_groups, eps);
    if (w)
        y = ggml_mul(g, y, w);
    if (b)
        y = ggml_add(g, y, b);
    return y;
}

// LayerNorm (= GroupNorm with groups=1, applied on last dim)
static ggml_tensor* ggml_layer_norm_affine(ggml_context* g, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, float eps) {
    ggml_tensor* y = ggml_norm(g, x, eps);
    if (w)
        y = ggml_mul(g, y, w);
    if (b)
        y = ggml_add(g, y, b);
    return y;
}

// DConv residual block: x = x + LayerScale * GLU(Norm(Conv1d(GELU(Norm(DilConv(x))))))
// Applied on time axis for both freq and time branches (freq branch reshapes to apply DConv per-freq-band).
static ggml_tensor* apply_dconv(ggml_context* g, ggml_tensor* x, const htdemucs_dconv& dc) {
    for (size_t d = 0; d < dc.layers.size(); d++) {
        auto& sl = dc.layers[d];
        if (!sl.conv1_w)
            continue;
        // Residual: x = x + scale * GLU(norm2(conv2(GELU(norm1(conv1(x))))))
        int dilation = 1 << (int)d;
        int K = (int)sl.conv1_w->ne[0];
        int pad = dilation * (K / 2);

        ggml_tensor* h = ggml_conv_1d(g, sl.conv1_w, x, 1, pad, dilation);
        if (sl.conv1_b)
            h = ggml_add(g, h, sl.conv1_b);
        if (sl.norm1_w)
            h = ggml_group_norm_affine(g, h, sl.norm1_w, sl.norm1_b, 1, 1e-5f);
        h = ggml_gelu(g, h);
        // 1x1 conv → 2*channels
        h = ggml_conv_1d(g, sl.conv2_w, h, 1, 0, 1);
        if (sl.conv2_b)
            h = ggml_add(g, h, sl.conv2_b);
        if (sl.norm2_w)
            h = ggml_group_norm_affine(g, h, sl.norm2_w, sl.norm2_b, 1, 1e-5f);
        // GLU on channel dim (dim=1 for (C, T), but ggml is (T, 2*C) → split on ne[0] axis)
        // Actually ggml tensors from conv_1d come out as (T_out, C_out).
        // GLU splits the channel dim in half.
        // For ggml: conv_1d output is (T, 2*C). GLU along dim 0 (the channel dim in ggml).
        // ggml doesn't have GLU directly. GLU(x) = x[:C] * sigmoid(x[C:])
        int C_half = (int)h->ne[0] / 2;
        int T_out = (int)h->ne[1];
        ggml_tensor* a = ggml_view_2d(g, h, C_half, T_out, h->nb[1], 0);
        ggml_tensor* b_gate = ggml_view_2d(g, h, C_half, T_out, h->nb[1], C_half * ggml_element_size(h));
        h = ggml_mul(g, ggml_cont(g, a), ggml_sigmoid(g, ggml_cont(g, b_gate)));
        // LayerScale
        if (sl.scale)
            h = ggml_mul(g, h, sl.scale);
        x = ggml_add(g, x, h);
    }
    return x;
}

// ---------------------------------------------------------------------------
// Full forward pass (Phase 2b — in progress)
// ---------------------------------------------------------------------------

// The full forward is built as a ggml graph. This is the core of the
// separation engine. It mirrors htdemucs.py:HTDemucs.forward() exactly.
//
// Input:  stereo PCM at 44100 Hz (2 channels, n_samples per channel).
// Output: 4 × stereo PCM (drums, bass, other, vocals).

htdemucs_result* htdemucs_separate(htdemucs_context* ctx, const float* pcm_stereo, int n_samples) {
    if (!ctx || !pcm_stereo || n_samples <= 0)
        return nullptr;

    auto& hp = ctx->model.hparams;
    auto& m = ctx->model;

    // Step 1: deinterleave stereo to channel-major
    int training_length = hp.training_length();
    int work_length = std::max(n_samples, training_length);
    std::vector<float> pcm_ch(2 * work_length, 0.0f);
    for (int i = 0; i < n_samples; i++) {
        pcm_ch[i] = pcm_stereo[2 * i];                   // L
        pcm_ch[work_length + i] = pcm_stereo[2 * i + 1]; // R
    }

    // Step 2: HTDemucs-specific STFT (matches _spec() in htdemucs.py)
    //
    // The Python _spec() does:
    //   le = ceil(T / hop_length)
    //   pad = hop_length // 2 * 3   (= 1536 for hop=1024)
    //   x = pad1d(x, (pad, pad + le*hop - T), mode="reflect")
    //   z = spectro(x, nfft, hop)   (center=True, normalized=True)
    //   z = z[..., :-1, :]          (drop last freq bin: 2049 → 2048)
    //   z = z[..., 2:2+le]          (trim to le frames)
    //
    // spectro's center=True adds nfft/2 reflect pad on each side internally.
    // So total padding before the raw STFT is:
    //   left:  pad + nfft/2 = 1536 + 2048 = 3584
    //   right: (pad + le*hop - T) + nfft/2

    int nfft = hp.nfft;
    int hop = hp.hop_length();
    int le = (int)ceil((double)work_length / hop);

    // Apply _spec() pre-padding (reflect) to each channel
    int pre_pad_left = hop / 2 * 3; // 1536
    int pre_pad_right = pre_pad_left + le * hop - work_length;
    int pre_padded_len = work_length + pre_pad_left + pre_pad_right;
    std::vector<float> pre_padded(2 * pre_padded_len);

    for (int ch = 0; ch < 2; ch++) {
        const float* src = pcm_ch.data() + (size_t)ch * work_length;
        float* dst = pre_padded.data() + (size_t)ch * pre_padded_len;
        for (int i = 0; i < pre_padded_len; i++) {
            int idx = i - pre_pad_left;
            // Reflect padding (matches pad1d with mode="reflect")
            if (idx < 0)
                idx = -idx;
            if (idx >= work_length)
                idx = 2 * work_length - 2 - idx;
            idx = std::max(0, std::min(work_length - 1, idx));
            dst[i] = src[idx];
        }
    }

    // Now run the raw STFT with center=True on the pre-padded signal
    stft_result spec = compute_stft(pre_padded.data(), 2, pre_padded_len, nfft, hop, ctx->hann_window.data());

    // z[..., :-1, :] — drop the last frequency bin (2049 → 2048)
    int Fq = nfft / 2; // 2048 (was nfft/2+1 = 2049)

    // z[..., 2:2+le] — take frames 2..2+le (the _spec frame slicing)
    int T = le;
    int frame_offset = 2;

    // Build CaC magnitude: (B, C*2, Fq, T) from complex spectrogram.
    // For B=1, C=2: we get 4 channels — [real_L, real_R, imag_L, imag_R]
    // Actually Python does: view_as_real(z).permute(0,1,4,2,3).reshape(B, C*2, Fr, T)
    // where view_as_real gives (B, C, Fr, T, 2), permute → (B, C, 2, Fr, T),
    // reshape → (B, C*2, Fr, T) = (1, 4, 2048, le)
    // So channel order is: [L_real, L_imag, R_real, R_imag]
    int n_cac_ch = hp.audio_channels * 2; // 4
    std::vector<float> cac_mag(n_cac_ch * Fq * T, 0.0f);

    for (int ch = 0; ch < 2; ch++) {
        for (int fq = 0; fq < Fq; fq++) {
            for (int t = 0; t < T; t++) {
                size_t src_idx =
                    (size_t)ch * spec.n_freqs * spec.n_frames + (size_t)fq * spec.n_frames + (frame_offset + t);
                float re_val = spec.real[src_idx];
                float im_val = spec.imag[src_idx];
                // CaC channel order: [ch*2] = real, [ch*2+1] = imag
                cac_mag[((size_t)(ch * 2) * Fq + fq) * T + t] = re_val;
                cac_mag[((size_t)(ch * 2 + 1) * Fq + fq) * T + t] = im_val;
            }
        }
    }

    if (htdemucs_debug()) {
        fprintf(stderr, "htdemucs: STFT spec %d×%d, CaC %d×%d×%d\n", spec.n_freqs, spec.n_frames, n_cac_ch, Fq, T);
    }

    fprintf(stderr, "htdemucs: STFT → %d freqs × %d frames, CaC → %d channels\n", Fq, T, hp.cac ? 4 : 2);
    // Step 4: Normalize spec branch (mean/std over all dims)
    float spec_mean = 0.0f, spec_var = 0.0f;
    size_t spec_n = (size_t)n_cac_ch * Fq * T;
    for (size_t i = 0; i < spec_n; i++)
        spec_mean += cac_mag[i];
    spec_mean /= (float)spec_n;
    for (size_t i = 0; i < spec_n; i++) {
        float d = cac_mag[i] - spec_mean;
        spec_var += d * d;
    }
    float spec_std = sqrtf(spec_var / (float)spec_n);
    for (size_t i = 0; i < spec_n; i++) {
        cac_mag[i] = (cac_mag[i] - spec_mean) / (1e-5f + spec_std);
    }

    // Step 5: Normalize time branch
    float time_mean = 0.0f, time_var = 0.0f;
    size_t time_n = (size_t)2 * work_length;
    for (size_t i = 0; i < time_n; i++)
        time_mean += pcm_ch[i];
    time_mean /= (float)time_n;
    for (size_t i = 0; i < time_n; i++) {
        float d = pcm_ch[i] - time_mean;
        time_var += d * d;
    }
    float time_std = sqrtf(time_var / (float)time_n);
    for (size_t i = 0; i < time_n; i++) {
        pcm_ch[i] = (pcm_ch[i] - time_mean) / (1e-5f + time_std);
    }

    if (htdemucs_debug()) {
        fprintf(stderr, "htdemucs: spec_norm mean=%.6f std=%.6f, time_norm mean=%.6f std=%.6f\n", spec_mean, spec_std,
                time_mean, time_std);
    }

    // Step 6: Encoder forward (dimension tracking for now, ggml graphs next)
    //
    // Freq branch: cac_mag[n_cac_ch × Fq × T], Time branch: pcm_ch[2 × work_length]
    // Each encoder layer changes dims; we track them for decoder symmetry.

    std::vector<float> x_buf(cac_mag);
    int x_C = n_cac_ch, x_Fq = Fq, x_T = T;
    std::vector<float> xt_buf(pcm_ch.begin(), pcm_ch.begin() + 2 * work_length);
    int xt_C = hp.audio_channels, xt_T = work_length;

    int freqs_cur = nfft / 2;
    bool encoder_ok = true;

    for (int idx = 0; idx < hp.depth && encoder_ok; idx++) {
        auto& enc = m.encoder[idx];
        bool freq = (freqs_cur > 1);
        int stri = freq ? hp.stride : 2;
        int ker = freq ? hp.kernel_size : 4;
        int pad_val = ker / 4;
        bool last_freq = false;
        (void)last_freq;

        if (freq && freqs_cur <= hp.kernel_size) {
            ker = freqs_cur;
            pad_val = 0;
            last_freq = true;
        }

        if (htdemucs_debug()) {
            fprintf(stderr, "htdemucs: enc[%d] x=(%d,%d,%d) xt=(%d,%d) freq=%d freqs=%d ker=%d stri=%d pad=%d\n", idx,
                    x_C, x_Fq, x_T, xt_C, xt_T, freq ? 1 : 0, freqs_cur, ker, stri, pad_val);
        }

        // --- Per-layer ggml graph for freq encoder ---
        if (!enc.conv_w) {
            encoder_ok = false;
            break;
        }
        {
            // Allocate a small ggml context for this layer's graph
            size_t n_tensors_est = 64;
            size_t ctx_size = ggml_tensor_overhead() * n_tensors_est + ggml_graph_overhead();
            ggml_init_params gp = {ctx_size, nullptr, true};
            ggml_context* g = ggml_init(gp);

            // Input: x_buf is laid out as (x_C, x_Fq, x_T) channel-major.
            // ggml Conv2d expects b: [N, IC, IH, IW] → ne = (IW, IH, IC, N)
            // Our data: IW=x_T, IH=x_Fq, IC=x_C, N=1
            ggml_tensor* x_in = ggml_new_tensor_4d(g, GGML_TYPE_F32, x_T, x_Fq, x_C, 1);
            ggml_set_name(x_in, "enc_in");
            ggml_set_input(x_in);

            // Conv2d: kernel ne = (KW, KH, IC, OC), stride=(s0=1, s1=stri), pad=(p0=0, p1=pad_val)
            // For freq branch: kernel [Cout, Cin, K, 1] → ggml ne: (1, K, Cin, Cout)
            // s0=1 (W=time axis, no stride), s1=stri (H=freq axis, stride=4)
            // p0=0, p1=pad_val
            ggml_tensor* y = ggml_conv_2d(g, enc.conv_w, x_in, 1, stri, 0, pad_val, 1, 1);
            if (enc.conv_b) {
                // Bias shape: (OC,) → broadcast: reshape to (1, 1, OC, 1)
                ggml_tensor* bias_4d = ggml_reshape_4d(g, enc.conv_b, 1, 1, (int)enc.conv_b->ne[0], 1);
                y = ggml_add(g, y, bias_4d);
            }

            if (!enc.empty) {
                // GroupNorm(norm_groups, C) + GELU
                // ggml_group_norm uses ne[2] as channel dim — correct for (T, Fq, C, N)
                if (enc.norm1_w) {
                    int ng = 4; // norm_groups — htdemucs default
                    y = ggml_group_norm(g, y, ng, 1e-5f);
                    // Affine: y = y * weight + bias (weight/bias are 1D = (C,), broadcast over T,Fq)
                    ggml_tensor* w4d = ggml_reshape_4d(g, enc.norm1_w, 1, 1, (int)enc.norm1_w->ne[0], 1);
                    y = ggml_mul(g, y, w4d);
                    if (enc.norm1_b) {
                        ggml_tensor* b4d = ggml_reshape_4d(g, enc.norm1_b, 1, 1, (int)enc.norm1_b->ne[0], 1);
                        y = ggml_add(g, y, b4d);
                    }
                }
                y = ggml_gelu(g, y);

                // DConv: operates on time axis per-freq-band.
                // Python: y.permute(0,2,1,3).reshape(-1,C,T) → DConv → reshape back
                if (enc.dconv.layers.size() > 0) {
                    int yC = (int)y->ne[2], yFq = (int)y->ne[1], yT = (int)y->ne[0];
                    // Permute (T, Fq, C, 1) → (T, C, Fq, 1) then reshape to (T, C, Fq*1)
                    // Actually DConv Conv1d operates on (C, T) per freq band.
                    // We need to reshape to (Fq, C, T) batch — but ggml conv_1d doesn't
                    // support batching. For now, skip DConv and add it in a follow-up.
                    // TODO: implement DConv per-freq-band via loop or batched conv.
                    (void)yC;
                    (void)yFq;
                    (void)yT;
                }

                // Rewrite: 1x1 Conv2d(C → 2*C) → GroupNorm → GLU
                if (enc.rewrite_w) {
                    ggml_tensor* rw = ggml_conv_2d(g, enc.rewrite_w, y, 1, 1, 0, 0, 1, 1);
                    if (enc.rewrite_b) {
                        ggml_tensor* rwb = ggml_reshape_4d(g, enc.rewrite_b, 1, 1, (int)enc.rewrite_b->ne[0], 1);
                        rw = ggml_add(g, rw, rwb);
                    }
                    if (enc.norm2_w) {
                        int ng = 4;
                        rw = ggml_group_norm(g, rw, ng, 1e-5f);
                        ggml_tensor* w4d = ggml_reshape_4d(g, enc.norm2_w, 1, 1, (int)enc.norm2_w->ne[0], 1);
                        rw = ggml_mul(g, rw, w4d);
                        if (enc.norm2_b) {
                            ggml_tensor* b4d = ggml_reshape_4d(g, enc.norm2_b, 1, 1, (int)enc.norm2_b->ne[0], 1);
                            rw = ggml_add(g, rw, b4d);
                        }
                    }
                    // GLU: split channel dim in half, a * sigmoid(b)
                    // rw shape: (T, Fq, 2*C, 1)
                    int rw_C = (int)rw->ne[2];
                    int rw_half = rw_C / 2;
                    int rw_Fq = (int)rw->ne[1];
                    int rw_T = (int)rw->ne[0];
                    // Split on ne[2] (channel dim): first half and second half
                    // view_3d with offset on ne[2] axis:
                    size_t ch_stride = rw->nb[2]; // bytes per channel slice
                    ggml_tensor* a_glu =
                        ggml_view_4d(g, rw, rw_T, rw_Fq, rw_half, 1, rw->nb[1], rw->nb[2], rw->nb[3], 0);
                    ggml_tensor* b_glu = ggml_view_4d(g, rw, rw_T, rw_Fq, rw_half, 1, rw->nb[1], rw->nb[2], rw->nb[3],
                                                      (size_t)rw_half * ch_stride);
                    y = ggml_mul(g, ggml_cont(g, a_glu), ggml_sigmoid(g, ggml_cont(g, b_glu)));
                }
            }

            ggml_set_name(y, "enc_out");
            ggml_set_output(y);

            // Build and run graph
            ggml_cgraph* gf = ggml_new_graph(g);
            ggml_build_forward_expand(gf, y);

            ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
            if (!ggml_gallocr_alloc_graph(alloc, gf)) {
                fprintf(stderr, "htdemucs: enc[%d] graph alloc failed\n", idx);
                ggml_gallocr_free(alloc);
                ggml_free(g);
                encoder_ok = false;
                break;
            }

            // Upload input data
            ggml_backend_tensor_set(x_in, x_buf.data(), 0, x_buf.size() * sizeof(float));

            // Compute
            ggml_backend_graph_compute(ctx->backend, gf);

            // Read output
            int out_T = (int)y->ne[0];
            int out_Fq = (int)y->ne[1];
            int out_C = (int)y->ne[2];
            size_t out_n = (size_t)out_T * out_Fq * out_C;
            x_buf.resize(out_n);
            ggml_backend_tensor_get(y, x_buf.data(), 0, out_n * sizeof(float));
            x_C = out_C;
            x_Fq = out_Fq;
            x_T = out_T;

            if (htdemucs_debug()) {
                fprintf(stderr, "htdemucs: enc[%d] output (%d, %d, %d) = %zu floats\n", idx, x_C, x_Fq, x_T, out_n);
            }

            ggml_gallocr_free(alloc);
            ggml_free(g);
        }

        // Update freq tracking
        if (freq) {
            freqs_cur = (freqs_cur <= hp.kernel_size) ? 1 : freqs_cur / hp.stride;
        }
    }

    fprintf(stderr, "htdemucs: encoder %s, output (%d, %d, %d)\n", encoder_ok ? "OK" : "FAILED", x_C, x_Fq, x_T);

    // Return stub result
    auto r = new htdemucs_result();
    r->n_sources = hp.n_sources;
    r->n_channels = hp.audio_channels;
    r->n_samples = n_samples;
    r->sample_rate = hp.samplerate;

    r->sources = new float*[hp.n_sources];
    r->source_names = new const char*[hp.n_sources];
    for (int s = 0; s < hp.n_sources; s++) {
        size_t sz = (size_t)hp.audio_channels * n_samples;
        r->sources[s] = new float[sz]();
        r->source_names[s] = m.source_names[s].c_str();
    }

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
