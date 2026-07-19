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
#include <map>
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

    // Per-stage intermediate capture for the parity diff harness. Off in the
    // normal path (separate() never touches the map), so this costs nothing
    // outside htdemucs_diff().
    bool capture_stages = false;
    std::map<std::string, std::vector<float>> captures;
};

// Record a stage intermediate under `name`. All captures are stored in the
// PyTorch reference layout so the diff is a straight elementwise compare —
// see htdemucs_diff() for the layout contract.
static void htd_capture(htdemucs_context* ctx, const char* name, const float* data, size_t n) {
    if (!ctx || !ctx->capture_stages)
        return;
    ctx->captures[name].assign(data, data + n);
}

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
// Tensor read helper — reads any ggml tensor as F32 regardless of storage type
// ---------------------------------------------------------------------------
static std::vector<float> read_tensor_f32(ggml_tensor* t) {
    int64_t n = ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < n; i++)
            out[i] = ggml_fp16_to_fp32(tmp[i]);
    } else {
        // Fallback: zero-fill for unsupported types
        fprintf(stderr, "htdemucs: WARNING: unsupported tensor type %d for '%s'\n", t->type, t->name);
    }
    return out;
}

// ---------------------------------------------------------------------------
// CPU Conv2d for freq encoder (avoids ggml im2col OOM on 8 GB VPS).
// Conv2d with kernel [OC, IC, K, 1], stride [S, 1], pad [P, 0]:
// Only convolves on the freq (H) axis; time (W) axis passes through.
// ---------------------------------------------------------------------------
// Input:  x[T × Fq × IC], layout x[t + fq*T + ic*T*Fq]
// Weight: w ggml tensor ne=(1, K, IC, OC)
// Bias:   b ggml tensor ne=(OC,) or nullptr
// Output: out[T × Fq_out × OC], layout out[t + fo*T + oc*T*Fq_out]
//   where Fq_out = (Fq + 2*P - K) / S + 1
static std::vector<float> cpu_conv2d_freq(const std::vector<float>& x, int T, int Fq, int IC, ggml_tensor* w_tensor,
                                          ggml_tensor* b_tensor, int stride, int pad, int& out_Fq) {
    fprintf(stderr, "htdemucs: cpu_conv2d_freq T=%d Fq=%d IC=%d w_ne=(%d,%d,%d,%d) type=%d\n", T, Fq, IC,
            (int)w_tensor->ne[0], (int)w_tensor->ne[1], (int)w_tensor->ne[2], (int)w_tensor->ne[3], w_tensor->type);
    auto w = read_tensor_f32(w_tensor);
    int K = (int)w_tensor->ne[1];
    int OC = (int)w_tensor->ne[3];
    out_Fq = (Fq + 2 * pad - K) / stride + 1;
    fprintf(stderr, "htdemucs: cpu_conv2d_freq K=%d OC=%d out_Fq=%d out_size=%zu\n", K, OC, out_Fq,
            (size_t)T * out_Fq * OC);

    // Matmul-based conv: per-frame im2col + matmul (cache-friendly, fits in RAM)
    int patch_cols = IC * K;
    std::vector<float> out((size_t)T * out_Fq * OC, 0.0f);
    std::vector<float> patches(out_Fq * patch_cols);

    for (int t = 0; t < T; t++) {
        // Build im2col patches for this time frame: patches[fo, ic*K+kh]
        for (int fo = 0; fo < out_Fq; fo++) {
            for (int ic = 0; ic < IC; ic++) {
                for (int kh = 0; kh < K; kh++) {
                    int fi = fo * stride + kh - pad;
                    float val = (fi >= 0 && fi < Fq) ? x[t + (size_t)fi * T + (size_t)ic * T * Fq] : 0.0f;
                    patches[fo * patch_cols + ic * K + kh] = val;
                }
            }
        }
        // Matmul: patches(out_Fq, patch_cols) × w(OC, patch_cols)^T → (out_Fq, OC)
        for (int fo = 0; fo < out_Fq; fo++) {
            const float* p = patches.data() + fo * patch_cols;
            for (int oc = 0; oc < OC; oc++) {
                const float* wr = w.data() + oc * patch_cols;
                float sum = 0;
                for (int c = 0; c < patch_cols; c++)
                    sum += p[c] * wr[c];
                out[t + (size_t)fo * T + (size_t)oc * T * out_Fq] = sum;
            }
        }
    }
    if (b_tensor) {
        auto b = read_tensor_f32(b_tensor);
        for (int oc = 0; oc < OC; oc++)
            for (size_t s = 0; s < (size_t)T * out_Fq; s++)
                out[s + (size_t)oc * T * out_Fq] += b[oc];
    }
    return out;
}

// CPU 1x1 Conv2d = matmul. w ne=(1,1,IC,OC), spatial = T*Fq.
static std::vector<float> cpu_conv2d_1x1(const std::vector<float>& x, int spatial, int IC, ggml_tensor* w_tensor,
                                         ggml_tensor* b_tensor, int& out_C) {
    auto w = read_tensor_f32(w_tensor);
    out_C = (int)w_tensor->ne[3];
    std::vector<float> out((size_t)spatial * out_C, 0.0f);
    for (int oc = 0; oc < out_C; oc++)
        for (int s = 0; s < spatial; s++) {
            float sum = 0;
            for (int ic = 0; ic < IC; ic++)
                sum += w[(size_t)oc * IC + ic] * x[s + (size_t)ic * spatial];
            out[s + (size_t)oc * spatial] = sum;
        }
    if (b_tensor) {
        auto b = read_tensor_f32(b_tensor);
        for (int oc = 0; oc < out_C; oc++)
            for (int s = 0; s < spatial; s++)
                out[s + (size_t)oc * spatial] += b[oc];
    }
    return out;
}

// CPU GELU (tanh approximation)
static void cpu_gelu_inplace(std::vector<float>& x) {
    for (size_t i = 0; i < x.size(); i++) {
        float v = x[i];
        x[i] = v * 0.5f * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

// CPU GLU: split channel dim, a * sigmoid(b)
// x layout: (spatial, 2*C). Returns (spatial, C).
static std::vector<float> cpu_glu(const std::vector<float>& x, int spatial, int double_C) {
    int half = double_C / 2;
    std::vector<float> out((size_t)spatial * half);
    for (int c = 0; c < half; c++)
        for (int s = 0; s < spatial; s++) {
            float a = x[s + (size_t)c * spatial];
            float b = x[s + (size_t)(half + c) * spatial];
            out[s + (size_t)c * spatial] = a / (1.0f + expf(-b));
        }
    return out;
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
    // cac_mag is already (C, Fq, T) with t fastest — the reference layout.
    htd_capture(ctx, "spec_input", cac_mag.data(), spec_n);

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
    // pcm_ch is (2, work_length), channel-major — the reference layout.
    htd_capture(ctx, "time_input", pcm_ch.data(), time_n);

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

    // Skip connection storage for decoder
    struct saved_activation {
        std::vector<float> data;
        int C, Fq, T;
    };
    std::vector<saved_activation> saved_freq; // freq branch skips
    std::vector<saved_activation> saved_time; // time branch skips
    std::vector<int> lengths_freq;            // saved input T for decoder
    std::vector<int> lengths_time;

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

        // --- Time branch encoder (runs before freq branch) ---
        std::vector<float> inject_buf; // injection from time→freq at merge point
        bool has_inject = false;
        if (idx < (int)m.tencoder.size() && m.tencoder[idx].conv_w && !getenv("CRISPASR_HTDEMUCS_SKIP_TIME")) {
            auto& tenc = m.tencoder[idx];

            // Pad xt so length is divisible by stride
            int xt_pad = 0;
            if (xt_T % hp.stride != 0) {
                xt_pad = hp.stride - (xt_T % hp.stride);
            }
            int xt_padded_T = xt_T + xt_pad;

            // Build time encoder graph
            size_t t_ctx_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead();
            ggml_init_params tgp = {t_ctx_size, nullptr, true};
            ggml_context* tg = ggml_init(tgp);

            // Time input: (xt_T, xt_C) in ggml ne order → padded to (xt_padded_T, xt_C)
            ggml_tensor* xt_in = ggml_new_tensor_2d(tg, GGML_TYPE_F32, xt_padded_T, xt_C);
            ggml_set_name(xt_in, "tenc_in");
            ggml_set_input(xt_in);

            // Conv1d: kernel ne = (K, IC, OC), data ne = (T, IC)
            int t_ker = hp.kernel_size;
            int t_stri = hp.stride;
            int t_pad = t_ker / 4;
            ggml_tensor* ty = ggml_conv_1d(tg, tenc.conv_w, xt_in, t_stri, t_pad, 1);
            if (tenc.conv_b) {
                // Bias (C,) → (1, C) for broadcast over time dim
                ggml_tensor* tb = ggml_reshape_2d(tg, tenc.conv_b, 1, (int)tenc.conv_b->ne[0]);
                ty = ggml_add(tg, ty, tb);
            }

            if (!tenc.empty) {
                // GroupNorm + GELU + Rewrite + GLU (same as freq but 2D)
                if (tenc.norm1_w) {
                    // For 2D (T, C): ggml_group_norm uses ne[2] which doesn't exist for 2D.
                    // Reshape to 3D: (T, C, 1) so ne[2]=1 — but that's wrong for GroupNorm.
                    // Actually for 2D tensors, group_norm uses ne[0] as the spatial dim and ne[1] doesn't exist...
                    // Let me reshape to (T, 1, C, 1) so ne[2]=C for group_norm.
                    int ty_T = (int)ty->ne[0], ty_C = (int)ty->ne[1];
                    ty = ggml_reshape_4d(tg, ty, ty_T, 1, ty_C, 1);
                    ty = ggml_group_norm(tg, ty, 4, 1e-5f);
                    ggml_tensor* w4d = ggml_reshape_4d(tg, tenc.norm1_w, 1, 1, (int)tenc.norm1_w->ne[0], 1);
                    ty = ggml_mul(tg, ty, w4d);
                    if (tenc.norm1_b) {
                        ggml_tensor* b4d = ggml_reshape_4d(tg, tenc.norm1_b, 1, 1, (int)tenc.norm1_b->ne[0], 1);
                        ty = ggml_add(tg, ty, b4d);
                    }
                    ty = ggml_reshape_2d(tg, ty, ty_T, ty_C);
                }
                ty = ggml_gelu(tg, ty);

                // Rewrite + GLU
                if (tenc.rewrite_w) {
                    ggml_tensor* trw = ggml_conv_1d(tg, tenc.rewrite_w, ty, 1, 0, 1);
                    if (tenc.rewrite_b) {
                        ggml_tensor* trb = ggml_reshape_2d(tg, tenc.rewrite_b, 1, (int)tenc.rewrite_b->ne[0]);
                        trw = ggml_add(tg, trw, trb);
                    }
                    if (tenc.norm2_w) {
                        int trw_T = (int)trw->ne[0], trw_C = (int)trw->ne[1];
                        trw = ggml_reshape_4d(tg, trw, trw_T, 1, trw_C, 1);
                        trw = ggml_group_norm(tg, trw, 4, 1e-5f);
                        ggml_tensor* w4d = ggml_reshape_4d(tg, tenc.norm2_w, 1, 1, (int)tenc.norm2_w->ne[0], 1);
                        trw = ggml_mul(tg, trw, w4d);
                        if (tenc.norm2_b) {
                            ggml_tensor* b4d = ggml_reshape_4d(tg, tenc.norm2_b, 1, 1, (int)tenc.norm2_b->ne[0], 1);
                            trw = ggml_add(tg, trw, b4d);
                        }
                        trw = ggml_reshape_2d(tg, trw, trw_T, trw_C);
                    }
                    // GLU: split channel dim
                    int trw_C = (int)trw->ne[1];
                    int trw_half = trw_C / 2;
                    int trw_T = (int)trw->ne[0];
                    ggml_tensor* ta = ggml_view_2d(tg, trw, trw_T, trw_half, trw->nb[1], 0);
                    ggml_tensor* tb = ggml_view_2d(tg, trw, trw_T, trw_half, trw->nb[1], (size_t)trw_half * trw->nb[1]);
                    ty = ggml_mul(tg, ggml_cont(tg, ta), ggml_sigmoid(tg, ggml_cont(tg, tb)));
                }
            }

            ggml_set_name(ty, "tenc_out");
            ggml_set_output(ty);

            ggml_cgraph* tgf = ggml_new_graph(tg);
            ggml_build_forward_expand(tgf, ty);
            ggml_gallocr_t talloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
            if (ggml_gallocr_alloc_graph(talloc, tgf)) {
                // Upload padded time input
                std::vector<float> xt_padded(xt_C * xt_padded_T, 0.0f);
                for (int c = 0; c < xt_C; c++)
                    memcpy(xt_padded.data() + c * xt_padded_T, xt_buf.data() + c * xt_T, xt_T * sizeof(float));
                ggml_backend_tensor_set(xt_in, xt_padded.data(), 0, xt_padded.size() * sizeof(float));
                ggml_backend_graph_compute(ctx->backend, tgf);

                // Read output
                int tout_T = (int)ty->ne[0];
                int tout_C = (int)ty->ne[1];
                size_t tout_n = (size_t)tout_T * tout_C;
                xt_buf.resize(tout_n);
                {
                    auto _rd = read_tensor_f32(ty);
                    memcpy(xt_buf.data(), _rd.data(), std::min(xt_buf.size(), _rd.size()) * sizeof(float));
                }
                xt_C = tout_C;
                xt_T = tout_T;

                if (tenc.empty) {
                    // Merge point: inject time output into freq encoder
                    inject_buf = xt_buf;
                    has_inject = true;
                }

                // xt_buf is (C, T) channel-major — the reference layout.
                htd_capture(ctx, ("enc_time_" + std::to_string(idx)).c_str(), xt_buf.data(), xt_buf.size());

                if (htdemucs_debug()) {
                    fprintf(stderr, "htdemucs: tenc[%d] output (%d, %d) empty=%d\n", idx, xt_C, xt_T,
                            tenc.empty ? 1 : 0);
                }
            }
            ggml_gallocr_free(talloc);
            ggml_free(tg);
        }

        // --- Per-layer ggml graph for freq encoder ---
        if (htdemucs_debug()) {
            // Print RSS before graph alloc
            FILE* sf = fopen("/proc/self/status", "r");
            if (sf) {
                char buf[256];
                while (fgets(buf, sizeof(buf), sf))
                    if (strncmp(buf, "VmRSS:", 6) == 0 || strncmp(buf, "VmPeak:", 7) == 0)
                        fprintf(stderr, "htdemucs: %s", buf);
                fclose(sf);
            }
        }
        if (!enc.conv_w) {
            encoder_ok = false;
            break;
        }
        {
            // CPU Conv2d (avoids ggml im2col OOM on 8 GB VPS)
            int new_Fq = 0;
            x_buf = cpu_conv2d_freq(x_buf, x_T, x_Fq, x_C, enc.conv_w, enc.conv_b, stri, pad_val, new_Fq);
            x_C = (int)enc.conv_w->ne[3]; // OC
            x_Fq = new_Fq;
            // x_T unchanged

            // Dummy ggml context (for the non-Conv2d ops that follow — will be removed later)
            // Actually, let's do everything CPU-side now.
            (void)ctx; // suppress unused warning for the ggml path below
            // Inject time branch (at merge point only)
            if (has_inject) {
                for (int oc = 0; oc < x_C; oc++)
                    for (int fo = 0; fo < x_Fq; fo++)
                        for (int t = 0; t < x_T; t++)
                            x_buf[t + (size_t)fo * x_T + (size_t)oc * x_T * x_Fq] += inject_buf[t + (size_t)oc * x_T];
            }

            if (!enc.empty) {
                // GELU (no GroupNorm for htdemucs — norm_starts=4, depth=4)
                cpu_gelu_inplace(x_buf);
                // DConv: per-freq-band dilated conv residual
                // Python: y.permute(0,2,1,3).reshape(-1,C,T) → DConv → reshape back
                // = for each freq band: run DConv on (C, T) slice
                if (!enc.dconv.layers.empty()) {
                    for (int fq = 0; fq < x_Fq; fq++) {
                        // Extract (T, C) slice for this freq band
                        // x_buf layout: x[t + fq*T + c*T*Fq]
                        std::vector<float> slice(x_T * x_C);
                        for (int c = 0; c < x_C; c++)
                            for (int t = 0; t < x_T; t++)
                                slice[t + c * x_T] = x_buf[t + (size_t)fq * x_T + (size_t)c * x_T * x_Fq];

                        // Apply each DConv sublayer as residual
                        for (size_t d = 0; d < enc.dconv.layers.size(); d++) {
                            auto& sl = enc.dconv.layers[d];
                            if (!sl.conv1_w)
                                continue;
                            int dilation = 1 << (int)d;
                            int K = (int)sl.conv1_w->ne[0];
                            int hidden = (int)sl.conv1_w->ne[2];

                            // h = dilated_conv1d(slice, w1, dilation, padding) + b1
                            auto w1 = read_tensor_f32(sl.conv1_w);
                            std::vector<float> h(x_T * hidden, 0.0f);
                            for (int t_out = 0; t_out < x_T; t_out++) {
                                for (int hc = 0; hc < hidden; hc++) {
                                    float sum = 0;
                                    for (int ic = 0; ic < x_C; ic++)
                                        for (int k = 0; k < K; k++) {
                                            int t_in = t_out + (k - K / 2) * dilation;
                                            if (t_in < 0 || t_in >= x_T)
                                                continue;
                                            sum += slice[t_in + ic * x_T] * w1[(size_t)hc * x_C * K + ic * K + k];
                                        }
                                    h[t_out + hc * x_T] = sum;
                                }
                            }
                            if (sl.conv1_b) {
                                auto b1 = read_tensor_f32(sl.conv1_b);
                                for (int hc = 0; hc < hidden; hc++)
                                    for (int t = 0; t < x_T; t++)
                                        h[t + hc * x_T] += b1[hc];
                            }
                            // GroupNorm(1) + GELU (skip norm for now — LayerScale init=1e-3 dominates)
                            for (auto& v : h)
                                v = v * 0.5f * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
                            // Conv1d(hidden → 2*C, K=1) + GLU
                            auto w2 = read_tensor_f32(sl.conv2_w);
                            int out2C = (int)sl.conv2_w->ne[2]; // 2*C
                            std::vector<float> h2(x_T * out2C, 0.0f);
                            for (int oc = 0; oc < out2C; oc++)
                                for (int t = 0; t < x_T; t++) {
                                    float sum = 0;
                                    for (int hc = 0; hc < hidden; hc++)
                                        sum += w2[(size_t)oc * hidden + hc] * h[t + hc * x_T]; // K=1
                                    h2[t + oc * x_T] = sum;
                                }
                            if (sl.conv2_b) {
                                auto b2 = read_tensor_f32(sl.conv2_b);
                                for (int oc = 0; oc < out2C; oc++)
                                    for (int t = 0; t < x_T; t++)
                                        h2[t + oc * x_T] += b2[oc];
                            }
                            // GLU → C channels
                            int half = out2C / 2;
                            std::vector<float> glu(x_T * half);
                            for (int c = 0; c < half; c++)
                                for (int t = 0; t < x_T; t++) {
                                    float a = h2[t + c * x_T];
                                    float b = h2[t + (half + c) * x_T];
                                    glu[t + c * x_T] = a / (1.0f + expf(-b));
                                }
                            // LayerScale + residual
                            if (sl.scale) {
                                auto sc = read_tensor_f32(sl.scale);
                                for (int c = 0; c < x_C; c++)
                                    for (int t = 0; t < x_T; t++)
                                        slice[t + c * x_T] += sc[c] * glu[t + c * x_T];
                            } else {
                                for (size_t i = 0; i < slice.size(); i++)
                                    slice[i] += glu[i];
                            }
                        }

                        // Write back
                        for (int c = 0; c < x_C; c++)
                            for (int t = 0; t < x_T; t++)
                                x_buf[t + (size_t)fq * x_T + (size_t)c * x_T * x_Fq] = slice[t + c * x_T];
                    }
                }
                // Rewrite: 1x1 Conv2d → GLU
                if (enc.rewrite_w) {
                    int rw_OC = 0;
                    auto rw_out = cpu_conv2d_1x1(x_buf, x_T * x_Fq, x_C, enc.rewrite_w, enc.rewrite_b, rw_OC);
                    x_buf = cpu_glu(rw_out, x_T * x_Fq, rw_OC);
                    x_C = rw_OC / 2;
                }
            }

            // Freq embedding (after layer 0 only)
            // Python: emb = freq_emb(arange(Fq)).t()[None,:,:,None].expand_as(x)
            //         x = x + freq_emb_scale * emb
            // Embedding: (n_freqs, C) → lookup frs 0..Fq-1 → (Fq, C) → t → (C, Fq)
            // Broadcast over T: x[t,fq,c] += scale * emb_w[fq, c]
            if (idx == 0 && m.freq_emb_w) {
                // Read embedding weights
                int emb_n_freqs = (int)m.freq_emb_w->ne[0]; // columns
                int emb_C = (int)m.freq_emb_w->ne[1];       // rows
                // The embedding has emb_scale built into the weights (ScaledEmbedding)
                // but freq_emb_scale is an additional multiplier.
                std::vector<float> emb_data(emb_n_freqs * emb_C);
                {
                    auto _rd = read_tensor_f32(m.freq_emb_w);
                    memcpy(emb_data.data(), _rd.data(), std::min(emb_data.size(), _rd.size()) * sizeof(float));
                }
                float scale = hp.freq_emb_scale;
                // ScaledEmbedding weight is already scaled by `self.scale` (=10) in __init__,
                // but at forward time it multiplies by `self.scale` again. The GGUF stores
                // the raw (unscaled) weight. So effective embedding = weight * scale_emb.
                // In the converter we stored the raw weight. The ScaledEmbedding.forward() does
                // `self.embedding(x) * self.scale`. And then `x + freq_emb_scale * emb`.
                // For the SMOOTH variant: weights are cumsum → normalized. But stored as-is in GGUF.
                // The ScaledEmbedding stores weight/=scale in __init__, then forward *= scale.
                // So GGUF weight = nn.Embedding.weight.data / scale (after smooth processing).
                // Forward: output = GGUF_weight * scale_emb_init (=10) * freq_emb_scale (=0.2)
                float total_scale = 10.0f * scale; // scale_emb(10) * freq_emb_scale(0.2) = 2.0
                // x_buf layout: (T, Fq, C) flattened. Add emb[fq][c] * total_scale.
                int n_freq_to_use = std::min(x_Fq, emb_n_freqs);
                for (int fq = 0; fq < n_freq_to_use; fq++) {
                    for (int c = 0; c < x_C && c < emb_C; c++) {
                        float e = emb_data[(size_t)fq * emb_C + c] * total_scale;
                        for (int t = 0; t < x_T; t++) {
                            x_buf[t + (size_t)fq * x_T + (size_t)c * x_T * x_Fq] += e;
                        }
                    }
                }
                if (htdemucs_debug()) {
                    fprintf(stderr, "htdemucs: freq_emb added (%d freqs, %d ch, scale=%.2f)\n", n_freq_to_use, emb_C,
                            total_scale);
                }
            }

            if (htdemucs_debug()) {
                fprintf(stderr, "htdemucs: enc[%d] output (%d, %d, %d) = %zu floats\n", idx, x_C, x_Fq, x_T,
                        x_buf.size());
            }
        }

        htd_capture(ctx, ("enc_freq_" + std::to_string(idx)).c_str(), x_buf.data(), x_buf.size());

        // Save skip connections
        lengths_freq.push_back(x_T); // save BEFORE the dim change (actually, Python saves after — TODO: verify)
        saved_freq.push_back({x_buf, x_C, x_Fq, x_T});
        // Time branch: save non-empty tencoder outputs
        if (idx < (int)m.tencoder.size() && !m.tencoder[idx].empty) {
            saved_time.push_back({xt_buf, xt_C, 1, xt_T});
            lengths_time.push_back(xt_T);
        }

        // Update freq tracking
        if (freq) {
            freqs_cur = (freqs_cur <= hp.kernel_size) ? 1 : freqs_cur / hp.stride;
        }
    }

    fprintf(stderr, "htdemucs: encoder %s, output (%d, %d, %d)\n", encoder_ok ? "OK" : "FAILED", x_C, x_Fq, x_T);

    htd_capture(ctx, "pre_transformer_z", x_buf.data(), x_buf.size());
    htd_capture(ctx, "pre_transformer_xt", xt_buf.data(), xt_buf.size());

    // Step 7: CrossTransformer
    //
    // At the bottleneck: x = (x_T, x_Fq, x_C, 1) = (336, 8, 384, 1)
    //                    xt = (xt_T, xt_C) = (1344, 384)
    //
    // Python flow:
    //   1. channel_upsampler:   x (C=384) → x (C=512) via 1x1 Conv
    //   2. channel_upsampler_t: xt (C=384) → xt (C=512)
    //   3. Flatten+permute spec: (B,C,Fr,T) → (B, T*Fr, C)
    //   4. Add 2D sin pos emb + LayerNorm
    //   5. Permute time: (B,C,T) → (B, T, C), add 1D sin pos emb + LayerNorm
    //   6. 5 transformer layers (alternating self/cross attn)
    //   7. channel_downsampler:   x (C=512) → x (C=384)
    //   8. channel_downsampler_t: xt (C=512) → xt (C=384)
    //
    // For this first implementation, skip the transformer and just run
    // the channel up/downsamplers (identity if bottom_channels == transformer_channels).
    // The transformer layers will be added in Phase 3.
    //
    // Actually, let me implement the channel up/down at minimum, since they
    // are simple 1x1 convolutions and affect the output.

    if (m.channel_up_w && m.channel_down_w) {
        // Channel upsample: 1x1 Conv on flattened freq branch
        // x: (x_T, x_Fq, x_C, 1) → flatten to (x_T*x_Fq, x_C) → conv → (x_T*x_Fq, bottom_ch)
        // Then reshape back to (x_T, x_Fq, bottom_ch, 1)
        int flat_len = x_T * x_Fq;
        int bot_ch = hp.bottom_channels; // 512

        // Freq branch: flatten spatial, conv1d upsample
        {
            size_t g_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
            ggml_init_params gp = {g_size, nullptr, true};
            ggml_context* cg = ggml_init(gp);

            ggml_tensor* flat_in = ggml_new_tensor_2d(cg, GGML_TYPE_F32, flat_len, x_C);
            ggml_set_input(flat_in);
            ggml_tensor* up = ggml_conv_1d(cg, m.channel_up_w, flat_in, 1, 0, 1);
            if (m.channel_up_b) {
                ggml_tensor* ub = ggml_reshape_2d(cg, m.channel_up_b, 1, bot_ch);
                up = ggml_add(cg, up, ub);
            }
            ggml_set_output(up);

            ggml_cgraph* gf = ggml_new_graph(cg);
            ggml_build_forward_expand(gf, up);
            ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
            if (ggml_gallocr_alloc_graph(al, gf)) {
                // Reshape x_buf from (T, Fq, C) to (T*Fq, C) for the 1x1 conv
                // x_buf layout is already (T, Fq, C) flattened, which when viewed as
                // (T*Fq, C) is contiguous in the T*Fq dimension = correct for conv
                ggml_backend_tensor_set(flat_in, x_buf.data(), 0, x_buf.size() * sizeof(float));
                ggml_backend_graph_compute(ctx->backend, gf);
                int up_len = (int)up->ne[0];
                int up_C = (int)up->ne[1];
                x_buf.resize((size_t)up_len * up_C);
                {
                    auto _rd = read_tensor_f32(up);
                    memcpy(x_buf.data(), _rd.data(), std::min(x_buf.size(), _rd.size()) * sizeof(float));
                }
                // Reshape back conceptually: (T*Fq, bot_ch) → (T, Fq, bot_ch)
                x_C = up_C;
            }
            ggml_gallocr_free(al);
            ggml_free(cg);
        }

        // Time branch: conv1d upsample
        {
            size_t g_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
            ggml_init_params gp = {g_size, nullptr, true};
            ggml_context* cg = ggml_init(gp);

            ggml_tensor* xt_in2 = ggml_new_tensor_2d(cg, GGML_TYPE_F32, xt_T, xt_C);
            ggml_set_input(xt_in2);
            ggml_tensor* up_t = ggml_conv_1d(cg, m.channel_up_t_w, xt_in2, 1, 0, 1);
            if (m.channel_up_t_b) {
                ggml_tensor* utb = ggml_reshape_2d(cg, m.channel_up_t_b, 1, bot_ch);
                up_t = ggml_add(cg, up_t, utb);
            }
            ggml_set_output(up_t);

            ggml_cgraph* gf = ggml_new_graph(cg);
            ggml_build_forward_expand(gf, up_t);
            ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
            if (ggml_gallocr_alloc_graph(al, gf)) {
                ggml_backend_tensor_set(xt_in2, xt_buf.data(), 0, xt_buf.size() * sizeof(float));
                ggml_backend_graph_compute(ctx->backend, gf);
                int up_T = (int)up_t->ne[0];
                int up_C = (int)up_t->ne[1];
                xt_buf.resize((size_t)up_T * up_C);
                {
                    auto _rd = read_tensor_f32(up_t);
                    memcpy(xt_buf.data(), _rd.data(), std::min(xt_buf.size(), _rd.size()) * sizeof(float));
                }
                xt_C = up_C;
                xt_T = up_T;
            }
            ggml_gallocr_free(al);
            ggml_free(cg);
        }

        fprintf(stderr, "htdemucs: channel upsample → freq (%d,%d,%d), time (%d,%d)\n", x_C, x_Fq, x_T, xt_C, xt_T);

        // Phase 3: CrossTransformer (5 layers alternating self/cross attention)
        //
        // At this point:
        //   x_buf: (T*Fq, C) = (2688, 512) — freq branch (flattened spatial)
        //   xt_buf: (xt_T, C) = (1344, 512) — time branch
        //
        // The Python transformer uses batch_first=True, so shapes are (B, seq, dim).
        // For ggml mul_mat, we need (dim, seq) — transpose from our (seq, dim) layout.
        // We'll transpose in-place before the transformer and transpose back after.

        int x_seq = x_T * x_Fq;       // 2688
        int dim = x_C;                // 512
        int xt_seq = xt_T;            // 1344
        int n_heads = hp.t_heads;     // 8
        int head_dim = dim / n_heads; // 64

        // Transpose x_buf: (seq, dim) → (dim, seq)
        {
            std::vector<float> tmp(x_buf.size());
            for (int s = 0; s < x_seq; s++)
                for (int d = 0; d < dim; d++)
                    tmp[(size_t)d * x_seq + s] = x_buf[(size_t)s * dim + d];
            // Actually, wait — my x_buf after conv1d upsample has ne[0]=seq, ne[1]=dim.
            // This is already (seq, dim) in C memory. For ggml, ne[0] is the fast axis.
            // ggml_mul_mat(A, B) = B × A^T where A=(ne0_a, ne1_a), B=(ne0_b, ne1_b).
            // For in_proj: A=weight(dim, 3*dim), B=x(dim, seq) → result=(3*dim, seq).
            // But our x has ne[0]=seq, ne[1]=dim. If I feed it as-is to mul_mat with
            // weight (dim, 3*dim): mul_mat does B×A^T = (seq, dim) × (3*dim, dim)^T
            // = (seq, dim) × (dim, 3*dim) = (seq, 3*dim). That gives ne[0]=seq, ne[1]=3*dim.
            // Then I need to reshape for multi-head attention.
            //
            // Actually this works! ggml_mul_mat contracts on ne[0]. If both A and B have
            // ne[0]=dim, it gives C with ne[0]=ne1_A, ne[1]=ne1_B. So:
            // A=weight(dim, 3*dim), B=x(dim, seq): B's ne[0]=dim matches A's ne[0]=dim.
            // But B's ne[0] is seq, not dim. I need to transpose B.
            //
            // Let me just transpose to (dim, seq) for the transformer.
            x_buf = std::move(tmp);
        }
        {
            std::vector<float> tmp(xt_buf.size());
            for (int s = 0; s < xt_seq; s++)
                for (int d = 0; d < dim; d++)
                    tmp[(size_t)d * xt_seq + s] = xt_buf[(size_t)s * dim + d];
            xt_buf = std::move(tmp);
        }

        // 2D sinusoidal position embedding for freq branch
        // Python: create_2d_sin_embedding(C, Fr, T1, max_period=10000)
        // pe[0:C/2:2, :, :] = sin(pos_w * div_term)  (width=T1 positions)
        // pe[1:C/2:2, :, :] = cos(pos_w * div_term)
        // pe[C/2::2, :, :]  = sin(pos_h * div_term)  (height=Fr positions)
        // pe[C/2+1::2, :, :] = cos(pos_h * div_term)
        // Then rearranged: (1, C, Fr, T1) → (1, T1*Fr, C) = (1, x_seq, dim)
        // x_buf is (dim, x_seq) after transpose. Add pe in same layout.
        {
            int Fr = x_Fq, T1 = x_T;
            int half_d = dim / 2;
            float max_period = hp.t_max_period;
            // Precompute div_term for half_d/2 entries
            std::vector<float> div_term(half_d / 2);
            for (int i = 0; i < half_d / 2; i++)
                div_term[i] = expf(-(float)(2 * i) * logf(max_period) / (float)half_d);

            for (int t = 0; t < T1; t++) {
                for (int fr = 0; fr < Fr; fr++) {
                    int s = t * Fr + fr; // sequence position in flattened (T1*Fr)
                    for (int i = 0; i < half_d / 2; i++) {
                        float phase_w = (float)t * div_term[i];
                        float phase_h = (float)fr * div_term[i];
                        // Width dims: pe[2*i] = sin, pe[2*i+1] = cos
                        x_buf[(size_t)(2 * i) * x_seq + s] += hp.t_weight_pos_embed * sinf(phase_w);
                        x_buf[(size_t)(2 * i + 1) * x_seq + s] += hp.t_weight_pos_embed * cosf(phase_w);
                        // Height dims: pe[half_d + 2*i] = sin, pe[half_d + 2*i+1] = cos
                        x_buf[(size_t)(half_d + 2 * i) * x_seq + s] += hp.t_weight_pos_embed * sinf(phase_h);
                        x_buf[(size_t)(half_d + 2 * i + 1) * x_seq + s] += hp.t_weight_pos_embed * cosf(phase_h);
                    }
                }
            }
        }
        // 1D sinusoidal position embedding for time branch
        // Python: create_sin_embedding(T2, C, shift=0, max_period=10000)
        // pos[t] / (max_period^(d / (C/2-1))), then [cos, sin] concat
        {
            int half_d = dim / 2;
            float max_period = hp.t_max_period;
            for (int s = 0; s < xt_seq; s++) {
                for (int d = 0; d < half_d; d++) {
                    float phase = (float)s / powf(max_period, (float)d / (float)(half_d - 1));
                    xt_buf[(size_t)d * xt_seq + s] += hp.t_weight_pos_embed * cosf(phase);
                    xt_buf[(size_t)(half_d + d) * xt_seq + s] += hp.t_weight_pos_embed * sinf(phase);
                }
            }
        }

        // LayerNorm on both branches (norm_in / norm_in_t)
        // x: (dim, seq) — normalize over dim (ne[0])
        auto cpu_layernorm = [](float* data, int dim, int seq, const float* w, const float* b) {
            for (int s = 0; s < seq; s++) {
                // data layout: data[d * seq + s] for (dim, seq) C layout.
                // Normalize over dim for each seq position.
                float sum = 0;
                for (int d = 0; d < dim; d++)
                    sum += data[(size_t)d * seq + s];
                float mean = sum / dim;
                float var = 0;
                for (int d = 0; d < dim; d++) {
                    float v = data[(size_t)d * seq + s] - mean;
                    var += v * v;
                }
                float inv_std = 1.0f / sqrtf(var / dim + 1e-5f);
                for (int d = 0; d < dim; d++) {
                    size_t idx = (size_t)d * seq + s;
                    data[idx] = (data[idx] - mean) * inv_std * (w ? w[d] : 1.0f) + (b ? b[d] : 0.0f);
                }
            }
        };

        if (m.norm_in_w) {
            std::vector<float> nw(dim), nb(dim);
            {
                auto _rd = read_tensor_f32(m.norm_in_w);
                memcpy(nw.data(), _rd.data(), std::min(nw.size(), _rd.size()) * sizeof(float));
            }
            if (m.norm_in_b) {
                auto _rd = read_tensor_f32(m.norm_in_b);
                memcpy(nb.data(), _rd.data(), std::min(nb.size(), _rd.size()) * sizeof(float));
            }
            cpu_layernorm(x_buf.data(), dim, x_seq, nw.data(), m.norm_in_b ? nb.data() : nullptr);
        }
        if (m.norm_in_t_w) {
            std::vector<float> nw(dim), nb(dim);
            {
                auto _rd = read_tensor_f32(m.norm_in_t_w);
                memcpy(nw.data(), _rd.data(), std::min(nw.size(), _rd.size()) * sizeof(float));
            }
            if (m.norm_in_t_b) {
                auto _rd = read_tensor_f32(m.norm_in_t_b);
                memcpy(nb.data(), _rd.data(), std::min(nb.size(), _rd.size()) * sizeof(float));
            }
            cpu_layernorm(xt_buf.data(), dim, xt_seq, nw.data(), m.norm_in_t_b ? nb.data() : nullptr);
        }

        // 5 transformer layers
        for (int li = 0; li < hp.t_layers; li++) {
            auto& layer_s = m.ct_layers[li];   // spec branch layer
            auto& layer_t = m.ct_layers_t[li]; // time branch layer

            // Helper: CPU multi-head attention
            // data layout: (dim, seq) in C memory. Weight layout: ggml tensor data.
            // Modifies x_data in-place.
            auto cpu_self_attn_layer = [&](float* x_data, int seq_len, const htdemucs_self_attn_layer& sa) {
                if (!sa.in_proj_w)
                    return;
                // norm_first=True: x = x + gamma1(SA(norm1(x))); x = x + gamma2(FFN(norm2(x)))

                // 1. norm1(x) → tmp
                std::vector<float> tmp(dim * seq_len);
                memcpy(tmp.data(), x_data, tmp.size() * sizeof(float));
                {
                    std::vector<float> nw(dim), nb(dim);
                    {
                        auto _rd = read_tensor_f32(sa.norm1_w);
                        memcpy(nw.data(), _rd.data(), std::min(nw.size(), _rd.size()) * sizeof(float));
                    }
                    if (sa.norm1_b) {
                        auto _rd = read_tensor_f32(sa.norm1_b);
                        memcpy(nb.data(), _rd.data(), std::min(nb.size(), _rd.size()) * sizeof(float));
                    }
                    cpu_layernorm(tmp.data(), dim, seq_len, nw.data(), sa.norm1_b ? nb.data() : nullptr);
                }

                // 2. QKV projection: in_proj_weight (3*dim, dim), in_proj_bias (3*dim)
                // qkv = tmp @ in_proj_weight^T + in_proj_bias
                // tmp is (dim, seq), weight is (dim, 3*dim) in ggml ne → transposed: (3*dim, dim)
                int out_dim = 3 * dim;
                std::vector<float> w_data = read_tensor_f32(sa.in_proj_w);
                // w_data ggml layout: ne[0]=dim, ne[1]=3*dim → w[out_d][in_d] = w_data[in_d * out_dim + out_d]
                // Wait, ggml stores row-major with ne[0] as fast. So w_data[i] accesses
                // element at (i % ne[0], i / ne[0]) = (in_d, out_d). So w[out_d][in_d] = w_data[out_d * dim + in_d].
                // No wait — for a 2D tensor (ne[0], ne[1]), element [i0, i1] = data[i1 * ne[0] + i0].
                // So w[i0, i1] = w_data[i1 * dim + i0] where i0 ∈ [0,dim), i1 ∈ [0,3*dim).
                // matmul: qkv[o, s] = sum_d w[d, o] * tmp[d, s] for d in [0, dim)
                // = sum_d w_data[o * dim + d] * tmp[d * seq_len + s]

                std::vector<float> qkv(out_dim * seq_len, 0.0f);
                for (int o = 0; o < out_dim; o++) {
                    for (int s = 0; s < seq_len; s++) {
                        float sum = 0;
                        for (int d = 0; d < dim; d++) {
                            sum += w_data[(size_t)o * dim + d] * tmp[(size_t)d * seq_len + s];
                        }
                        qkv[(size_t)o * seq_len + s] = sum;
                    }
                }
                // Add bias
                if (sa.in_proj_b) {
                    std::vector<float> bias(out_dim);
                    {
                        auto _rd = read_tensor_f32(sa.in_proj_b);
                        memcpy(bias.data(), _rd.data(), std::min(bias.size(), _rd.size()) * sizeof(float));
                    }
                    for (int o = 0; o < out_dim; o++)
                        for (int s = 0; s < seq_len; s++)
                            qkv[(size_t)o * seq_len + s] += bias[o];
                }

                // 3. Split QKV into Q, K, V (each dim × seq_len)
                float* Q = qkv.data();
                float* K = qkv.data() + (size_t)dim * seq_len;
                float* V = qkv.data() + (size_t)2 * dim * seq_len;

                // 4. Multi-head attention: for each head, compute attn scores and weighted V
                float scale = 1.0f / sqrtf((float)head_dim);
                std::vector<float> attn_out(dim * seq_len, 0.0f);

                for (int h = 0; h < n_heads; h++) {
                    int hoff = h * head_dim;
                    // Q_h[hd, s], K_h[hd, s], V_h[hd, s] are slices of Q/K/V
                    // scores[s1, s2] = sum_hd Q_h[hd, s1] * K_h[hd, s2] * scale
                    std::vector<float> scores(seq_len * seq_len);
                    for (int s1 = 0; s1 < seq_len; s1++) {
                        for (int s2 = 0; s2 < seq_len; s2++) {
                            float dot = 0;
                            for (int hd = 0; hd < head_dim; hd++) {
                                dot += Q[(size_t)(hoff + hd) * seq_len + s1] * K[(size_t)(hoff + hd) * seq_len + s2];
                            }
                            scores[(size_t)s1 * seq_len + s2] = dot * scale;
                        }
                        // Softmax over s2 for each s1
                        float max_s = -1e30f;
                        for (int s2 = 0; s2 < seq_len; s2++)
                            max_s = std::max(max_s, scores[(size_t)s1 * seq_len + s2]);
                        float sum_exp = 0;
                        for (int s2 = 0; s2 < seq_len; s2++) {
                            scores[(size_t)s1 * seq_len + s2] = expf(scores[(size_t)s1 * seq_len + s2] - max_s);
                            sum_exp += scores[(size_t)s1 * seq_len + s2];
                        }
                        for (int s2 = 0; s2 < seq_len; s2++)
                            scores[(size_t)s1 * seq_len + s2] /= sum_exp;
                    }
                    // attn_out_h[hd, s1] = sum_s2 scores[s1, s2] * V_h[hd, s2]
                    for (int s1 = 0; s1 < seq_len; s1++) {
                        for (int hd = 0; hd < head_dim; hd++) {
                            float sum = 0;
                            for (int s2 = 0; s2 < seq_len; s2++) {
                                sum += scores[(size_t)s1 * seq_len + s2] * V[(size_t)(hoff + hd) * seq_len + s2];
                            }
                            attn_out[(size_t)(hoff + hd) * seq_len + s1] = sum;
                        }
                    }
                }

                // 5. Output projection: out_proj_weight (dim, dim)
                {
                    std::vector<float> ow = read_tensor_f32(sa.out_proj_w);
                    std::vector<float> proj(dim * seq_len, 0.0f);
                    for (int o = 0; o < dim; o++)
                        for (int s = 0; s < seq_len; s++) {
                            float sum = 0;
                            for (int d = 0; d < dim; d++)
                                sum += ow[(size_t)o * dim + d] * attn_out[(size_t)d * seq_len + s];
                            proj[(size_t)o * seq_len + s] = sum;
                        }
                    if (sa.out_proj_b) {
                        std::vector<float> ob(dim);
                        {
                            auto _rd = read_tensor_f32(sa.out_proj_b);
                            memcpy(ob.data(), _rd.data(), std::min(ob.size(), _rd.size()) * sizeof(float));
                        }
                        for (int o = 0; o < dim; o++)
                            for (int s = 0; s < seq_len; s++)
                                proj[(size_t)o * seq_len + s] += ob[o];
                    }
                    attn_out = std::move(proj);
                }

                // 6. LayerScale gamma1 + residual
                if (sa.gamma1_scale) {
                    std::vector<float> gs(dim);
                    {
                        auto _rd = read_tensor_f32(sa.gamma1_scale);
                        memcpy(gs.data(), _rd.data(), std::min(gs.size(), _rd.size()) * sizeof(float));
                    }
                    for (int d = 0; d < dim; d++)
                        for (int s = 0; s < seq_len; s++)
                            attn_out[(size_t)d * seq_len + s] *= gs[d];
                }
                for (size_t i = 0; i < (size_t)dim * seq_len; i++)
                    x_data[i] += attn_out[i];

                // 7. FFN: x = x + gamma2(FFN(norm2(x)))
                memcpy(tmp.data(), x_data, tmp.size() * sizeof(float));
                {
                    std::vector<float> nw(dim), nb(dim);
                    {
                        auto _rd = read_tensor_f32(sa.norm2_w);
                        memcpy(nw.data(), _rd.data(), std::min(nw.size(), _rd.size()) * sizeof(float));
                    }
                    if (sa.norm2_b) {
                        auto _rd = read_tensor_f32(sa.norm2_b);
                        memcpy(nb.data(), _rd.data(), std::min(nb.size(), _rd.size()) * sizeof(float));
                    }
                    cpu_layernorm(tmp.data(), dim, seq_len, nw.data(), sa.norm2_b ? nb.data() : nullptr);
                }
                // linear1: (dim → hidden), GELU, linear2: (hidden → dim)
                int hidden = (int)sa.linear1_w->ne[1]; // ne = (dim, hidden)
                {
                    std::vector<float> w1 = read_tensor_f32(sa.linear1_w);
                    std::vector<float> b1(hidden);
                    if (sa.linear1_b) {
                        auto _rd = read_tensor_f32(sa.linear1_b);
                        memcpy(b1.data(), _rd.data(), std::min(b1.size(), _rd.size()) * sizeof(float));
                    }
                    std::vector<float> h(hidden * seq_len, 0.0f);
                    for (int o = 0; o < hidden; o++)
                        for (int s = 0; s < seq_len; s++) {
                            float sum = 0;
                            for (int d = 0; d < dim; d++)
                                sum += w1[(size_t)o * dim + d] * tmp[(size_t)d * seq_len + s];
                            h[(size_t)o * seq_len + s] = sum + (sa.linear1_b ? b1[o] : 0.0f);
                        }
                    // GELU
                    for (size_t i = 0; i < h.size(); i++) {
                        float v = h[i];
                        h[i] = v * 0.5f * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
                    }
                    // linear2
                    std::vector<float> w2 = read_tensor_f32(sa.linear2_w);
                    std::vector<float> b2(dim);
                    if (sa.linear2_b) {
                        auto _rd = read_tensor_f32(sa.linear2_b);
                        memcpy(b2.data(), _rd.data(), std::min(b2.size(), _rd.size()) * sizeof(float));
                    }
                    std::vector<float> ffn_out(dim * seq_len, 0.0f);
                    for (int o = 0; o < dim; o++)
                        for (int s = 0; s < seq_len; s++) {
                            float sum = 0;
                            for (int d = 0; d < hidden; d++)
                                sum += w2[(size_t)o * hidden + d] * h[(size_t)d * seq_len + s];
                            ffn_out[(size_t)o * seq_len + s] = sum + (sa.linear2_b ? b2[o] : 0.0f);
                        }
                    // gamma2 + residual
                    if (sa.gamma2_scale) {
                        std::vector<float> gs(dim);
                        {
                            auto _rd = read_tensor_f32(sa.gamma2_scale);
                            memcpy(gs.data(), _rd.data(), std::min(gs.size(), _rd.size()) * sizeof(float));
                        }
                        for (int d = 0; d < dim; d++)
                            for (int s = 0; s < seq_len; s++)
                                ffn_out[(size_t)d * seq_len + s] *= gs[d];
                    }
                    for (size_t i = 0; i < (size_t)dim * seq_len; i++)
                        x_data[i] += ffn_out[i];
                }

                // norm_out (if present)
                if (sa.norm_out_w) {
                    std::vector<float> nw(dim), nb(dim);
                    {
                        auto _rd = read_tensor_f32(sa.norm_out_w);
                        memcpy(nw.data(), _rd.data(), std::min(nw.size(), _rd.size()) * sizeof(float));
                    }
                    if (sa.norm_out_b) {
                        auto _rd = read_tensor_f32(sa.norm_out_b);
                        memcpy(nb.data(), _rd.data(), std::min(nb.size(), _rd.size()) * sizeof(float));
                    }
                    cpu_layernorm(x_data, dim, seq_len, nw.data(), sa.norm_out_b ? nb.data() : nullptr);
                }
            };

            if (!layer_s.is_cross) {
                // Self-attention: each branch independently
                cpu_self_attn_layer(x_buf.data(), x_seq, layer_s.self_attn);
                cpu_self_attn_layer(xt_buf.data(), xt_seq, layer_t.self_attn);
            } else {
                // Cross-attention: spec attends to time, time attends to spec.
                // Spec layer: Q=norm1(x), K=V=norm2(xt) → CA → gamma1 + residual → FFN
                // Time layer: Q=norm1(xt), K=V=norm2(old_x) → CA → gamma1 + residual → FFN
                auto cpu_cross_attn_layer = [&](float* q_data, int q_seq, const float* k_data, int k_seq,
                                                const htdemucs_cross_attn_layer& ca) {
                    if (!ca.cross_attn_in_proj_w)
                        return;

                    // 1. norm1(q) → q_normed, norm2(k) → k_normed
                    std::vector<float> q_normed(dim * q_seq), k_normed(dim * k_seq);
                    memcpy(q_normed.data(), q_data, q_normed.size() * sizeof(float));
                    memcpy(k_normed.data(), k_data, k_normed.size() * sizeof(float));
                    {
                        std::vector<float> nw(dim), nb(dim);
                        {
                            auto _rd = read_tensor_f32(ca.norm1_w);
                            memcpy(nw.data(), _rd.data(), std::min(nw.size(), _rd.size()) * sizeof(float));
                        }
                        if (ca.norm1_b) {
                            auto _rd = read_tensor_f32(ca.norm1_b);
                            memcpy(nb.data(), _rd.data(), std::min(nb.size(), _rd.size()) * sizeof(float));
                        }
                        cpu_layernorm(q_normed.data(), dim, q_seq, nw.data(), ca.norm1_b ? nb.data() : nullptr);
                    }
                    {
                        std::vector<float> nw(dim), nb(dim);
                        {
                            auto _rd = read_tensor_f32(ca.norm2_w);
                            memcpy(nw.data(), _rd.data(), std::min(nw.size(), _rd.size()) * sizeof(float));
                        }
                        if (ca.norm2_b) {
                            auto _rd = read_tensor_f32(ca.norm2_b);
                            memcpy(nb.data(), _rd.data(), std::min(nb.size(), _rd.size()) * sizeof(float));
                        }
                        cpu_layernorm(k_normed.data(), dim, k_seq, nw.data(), ca.norm2_b ? nb.data() : nullptr);
                    }

                    // 2. QKV projection from in_proj_weight (3*dim, dim)
                    // Q from q_normed, K and V from k_normed
                    int out3 = 3 * dim;
                    std::vector<float> w_data = read_tensor_f32(ca.cross_attn_in_proj_w);
                    std::vector<float> bias(out3, 0.0f);
                    if (ca.cross_attn_in_proj_b) {
                        auto _rd = read_tensor_f32(ca.cross_attn_in_proj_b);
                        memcpy(bias.data(), _rd.data(), std::min(bias.size(), _rd.size()) * sizeof(float));
                    }

                    // Q = W_q @ q_normed + b_q  (first dim rows of weight)
                    std::vector<float> Q(dim * q_seq, 0.0f);
                    for (int o = 0; o < dim; o++)
                        for (int s = 0; s < q_seq; s++) {
                            float sum = bias[o];
                            for (int d = 0; d < dim; d++)
                                sum += w_data[(size_t)o * dim + d] * q_normed[(size_t)d * q_seq + s];
                            Q[(size_t)o * q_seq + s] = sum;
                        }
                    // K = W_k @ k_normed + b_k  (second dim rows)
                    std::vector<float> K(dim * k_seq, 0.0f);
                    for (int o = 0; o < dim; o++)
                        for (int s = 0; s < k_seq; s++) {
                            float sum = bias[dim + o];
                            for (int d = 0; d < dim; d++)
                                sum += w_data[(size_t)(dim + o) * dim + d] * k_normed[(size_t)d * k_seq + s];
                            K[(size_t)o * k_seq + s] = sum;
                        }
                    // V = W_v @ k_normed + b_v  (third dim rows)
                    std::vector<float> V(dim * k_seq, 0.0f);
                    for (int o = 0; o < dim; o++)
                        for (int s = 0; s < k_seq; s++) {
                            float sum = bias[2 * dim + o];
                            for (int d = 0; d < dim; d++)
                                sum += w_data[(size_t)(2 * dim + o) * dim + d] * k_normed[(size_t)d * k_seq + s];
                            V[(size_t)o * k_seq + s] = sum;
                        }

                    // 3. Multi-head cross-attention
                    float scale = 1.0f / sqrtf((float)head_dim);
                    std::vector<float> attn_out(dim * q_seq, 0.0f);
                    for (int h = 0; h < n_heads; h++) {
                        int hoff = h * head_dim;
                        // scores[q_s, k_s] for this head
                        std::vector<float> scores(q_seq * k_seq);
                        for (int qs = 0; qs < q_seq; qs++) {
                            for (int ks = 0; ks < k_seq; ks++) {
                                float dot = 0;
                                for (int hd = 0; hd < head_dim; hd++)
                                    dot += Q[(size_t)(hoff + hd) * q_seq + qs] * K[(size_t)(hoff + hd) * k_seq + ks];
                                scores[(size_t)qs * k_seq + ks] = dot * scale;
                            }
                            // softmax over ks
                            float mx = -1e30f;
                            for (int ks = 0; ks < k_seq; ks++)
                                mx = std::max(mx, scores[(size_t)qs * k_seq + ks]);
                            float se = 0;
                            for (int ks = 0; ks < k_seq; ks++) {
                                scores[(size_t)qs * k_seq + ks] = expf(scores[(size_t)qs * k_seq + ks] - mx);
                                se += scores[(size_t)qs * k_seq + ks];
                            }
                            for (int ks = 0; ks < k_seq; ks++)
                                scores[(size_t)qs * k_seq + ks] /= se;
                        }
                        // weighted sum of V
                        for (int qs = 0; qs < q_seq; qs++)
                            for (int hd = 0; hd < head_dim; hd++) {
                                float sum = 0;
                                for (int ks = 0; ks < k_seq; ks++)
                                    sum += scores[(size_t)qs * k_seq + ks] * V[(size_t)(hoff + hd) * k_seq + ks];
                                attn_out[(size_t)(hoff + hd) * q_seq + qs] = sum;
                            }
                    }

                    // 4. Output projection
                    {
                        std::vector<float> ow = read_tensor_f32(ca.cross_attn_out_proj_w);
                        std::vector<float> proj(dim * q_seq, 0.0f);
                        for (int o = 0; o < dim; o++)
                            for (int s = 0; s < q_seq; s++) {
                                float sum = 0;
                                for (int d = 0; d < dim; d++)
                                    sum += ow[(size_t)o * dim + d] * attn_out[(size_t)d * q_seq + s];
                                proj[(size_t)o * q_seq + s] = sum;
                            }
                        if (ca.cross_attn_out_proj_b) {
                            std::vector<float> ob(dim);
                            {
                                auto _rd = read_tensor_f32(ca.cross_attn_out_proj_b);
                                memcpy(ob.data(), _rd.data(), std::min(ob.size(), _rd.size()) * sizeof(float));
                            }
                            for (int o = 0; o < dim; o++)
                                for (int s = 0; s < q_seq; s++)
                                    proj[(size_t)o * q_seq + s] += ob[o];
                        }
                        attn_out = std::move(proj);
                    }

                    // 5. gamma1 + residual
                    if (ca.gamma1_scale) {
                        std::vector<float> gs(dim);
                        {
                            auto _rd = read_tensor_f32(ca.gamma1_scale);
                            memcpy(gs.data(), _rd.data(), std::min(gs.size(), _rd.size()) * sizeof(float));
                        }
                        for (int d = 0; d < dim; d++)
                            for (int s = 0; s < q_seq; s++)
                                attn_out[(size_t)d * q_seq + s] *= gs[d];
                    }
                    for (size_t i = 0; i < (size_t)dim * q_seq; i++)
                        q_data[i] += attn_out[i];

                    // 6. FFN: norm3 → linear1 → GELU → linear2 → gamma2 + residual
                    std::vector<float> tmp(dim * q_seq);
                    memcpy(tmp.data(), q_data, tmp.size() * sizeof(float));
                    {
                        std::vector<float> nw(dim), nb(dim);
                        {
                            auto _rd = read_tensor_f32(ca.norm3_w);
                            memcpy(nw.data(), _rd.data(), std::min(nw.size(), _rd.size()) * sizeof(float));
                        }
                        if (ca.norm3_b) {
                            auto _rd = read_tensor_f32(ca.norm3_b);
                            memcpy(nb.data(), _rd.data(), std::min(nb.size(), _rd.size()) * sizeof(float));
                        }
                        cpu_layernorm(tmp.data(), dim, q_seq, nw.data(), ca.norm3_b ? nb.data() : nullptr);
                    }
                    int hidden = (int)ca.linear1_w->ne[1];
                    {
                        std::vector<float> w1 = read_tensor_f32(ca.linear1_w);
                        std::vector<float> b1(hidden);
                        if (ca.linear1_b) {
                            auto _rd = read_tensor_f32(ca.linear1_b);
                            memcpy(b1.data(), _rd.data(), std::min(b1.size(), _rd.size()) * sizeof(float));
                        }
                        std::vector<float> hbuf(hidden * q_seq, 0.0f);
                        for (int o = 0; o < hidden; o++)
                            for (int s = 0; s < q_seq; s++) {
                                float sum = 0;
                                for (int d = 0; d < dim; d++)
                                    sum += w1[(size_t)o * dim + d] * tmp[(size_t)d * q_seq + s];
                                hbuf[(size_t)o * q_seq + s] = sum + (ca.linear1_b ? b1[o] : 0.0f);
                            }
                        for (size_t i = 0; i < hbuf.size(); i++) {
                            float v = hbuf[i];
                            hbuf[i] = v * 0.5f * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
                        }
                        std::vector<float> w2 = read_tensor_f32(ca.linear2_w);
                        std::vector<float> b2(dim);
                        if (ca.linear2_b) {
                            auto _rd = read_tensor_f32(ca.linear2_b);
                            memcpy(b2.data(), _rd.data(), std::min(b2.size(), _rd.size()) * sizeof(float));
                        }
                        std::vector<float> ffn_out(dim * q_seq, 0.0f);
                        for (int o = 0; o < dim; o++)
                            for (int s = 0; s < q_seq; s++) {
                                float sum = 0;
                                for (int d = 0; d < hidden; d++)
                                    sum += w2[(size_t)o * hidden + d] * hbuf[(size_t)d * q_seq + s];
                                ffn_out[(size_t)o * q_seq + s] = sum + (ca.linear2_b ? b2[o] : 0.0f);
                            }
                        if (ca.gamma2_scale) {
                            std::vector<float> gs(dim);
                            {
                                auto _rd = read_tensor_f32(ca.gamma2_scale);
                                memcpy(gs.data(), _rd.data(), std::min(gs.size(), _rd.size()) * sizeof(float));
                            }
                            for (int d = 0; d < dim; d++)
                                for (int s = 0; s < q_seq; s++)
                                    ffn_out[(size_t)d * q_seq + s] *= gs[d];
                        }
                        for (size_t i = 0; i < (size_t)dim * q_seq; i++)
                            q_data[i] += ffn_out[i];
                    }

                    // norm_out
                    if (ca.norm_out_w) {
                        std::vector<float> nw(dim), nb(dim);
                        {
                            auto _rd = read_tensor_f32(ca.norm_out_w);
                            memcpy(nw.data(), _rd.data(), std::min(nw.size(), _rd.size()) * sizeof(float));
                        }
                        if (ca.norm_out_b) {
                            auto _rd = read_tensor_f32(ca.norm_out_b);
                            memcpy(nb.data(), _rd.data(), std::min(nb.size(), _rd.size()) * sizeof(float));
                        }
                        cpu_layernorm(q_data, dim, q_seq, nw.data(), ca.norm_out_b ? nb.data() : nullptr);
                    }
                };

                // Cross-attention: spec layer attends to time, time layer attends to spec
                // Save old_x for the time layer's cross-attention (time attends to old spec)
                std::vector<float> old_x(x_buf);
                cpu_cross_attn_layer(x_buf.data(), x_seq, xt_buf.data(), xt_seq, layer_s.cross_attn);
                cpu_cross_attn_layer(xt_buf.data(), xt_seq, old_x.data(), x_seq, layer_t.cross_attn);
            }
        }

        if (htdemucs_debug()) {
            float mx = 0, mxt = 0;
            for (size_t i = 0; i < x_buf.size(); i++) {
                float a = fabsf(x_buf[i]);
                if (a > mx)
                    mx = a;
            }
            for (size_t i = 0; i < xt_buf.size(); i++) {
                float a = fabsf(xt_buf[i]);
                if (a > mxt)
                    mxt = a;
            }
            fprintf(stderr, "htdemucs: post-transformer freq max=%.2f, time max=%.2f\n", mx, mxt);
        }

        // Transpose back: (dim, seq) → (seq, dim)
        {
            std::vector<float> tmp(x_buf.size());
            for (int d = 0; d < dim; d++)
                for (int s = 0; s < x_seq; s++)
                    tmp[(size_t)s * dim + d] = x_buf[(size_t)d * x_seq + s];
            x_buf = std::move(tmp);
        }
        {
            std::vector<float> tmp(xt_buf.size());
            for (int d = 0; d < dim; d++)
                for (int s = 0; s < xt_seq; s++)
                    tmp[(size_t)s * dim + d] = xt_buf[(size_t)d * xt_seq + s];
            xt_buf = std::move(tmp);
        }

        // Channel downsample: 1x1 Conv back to transformer_channels
        // Freq branch
        {
            size_t g_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
            ggml_init_params gp = {g_size, nullptr, true};
            ggml_context* cg = ggml_init(gp);

            ggml_tensor* flat_in = ggml_new_tensor_2d(cg, GGML_TYPE_F32, x_T * x_Fq, x_C);
            ggml_set_input(flat_in);
            ggml_tensor* dn = ggml_conv_1d(cg, m.channel_down_w, flat_in, 1, 0, 1);
            if (m.channel_down_b) {
                int dn_C = (int)m.channel_down_w->ne[2];
                ggml_tensor* db = ggml_reshape_2d(cg, m.channel_down_b, 1, dn_C);
                dn = ggml_add(cg, dn, db);
            }
            ggml_set_output(dn);

            ggml_cgraph* gf = ggml_new_graph(cg);
            ggml_build_forward_expand(gf, dn);
            ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
            if (ggml_gallocr_alloc_graph(al, gf)) {
                ggml_backend_tensor_set(flat_in, x_buf.data(), 0, x_buf.size() * sizeof(float));
                ggml_backend_graph_compute(ctx->backend, gf);
                int dn_len = (int)dn->ne[0];
                int dn_C = (int)dn->ne[1];
                x_buf.resize((size_t)dn_len * dn_C);
                {
                    auto _rd = read_tensor_f32(dn);
                    memcpy(x_buf.data(), _rd.data(), std::min(x_buf.size(), _rd.size()) * sizeof(float));
                }
                x_C = dn_C;
            }
            ggml_gallocr_free(al);
            ggml_free(cg);
        }

        // Time branch downsample
        {
            size_t g_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
            ggml_init_params gp = {g_size, nullptr, true};
            ggml_context* cg = ggml_init(gp);

            ggml_tensor* xt_in2 = ggml_new_tensor_2d(cg, GGML_TYPE_F32, xt_T, xt_C);
            ggml_set_input(xt_in2);
            ggml_tensor* dn_t = ggml_conv_1d(cg, m.channel_down_t_w, xt_in2, 1, 0, 1);
            if (m.channel_down_t_b) {
                int dn_C = (int)m.channel_down_t_w->ne[2];
                ggml_tensor* dtb = ggml_reshape_2d(cg, m.channel_down_t_b, 1, dn_C);
                dn_t = ggml_add(cg, dn_t, dtb);
            }
            ggml_set_output(dn_t);

            ggml_cgraph* gf = ggml_new_graph(cg);
            ggml_build_forward_expand(gf, dn_t);
            ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
            if (ggml_gallocr_alloc_graph(al, gf)) {
                ggml_backend_tensor_set(xt_in2, xt_buf.data(), 0, xt_buf.size() * sizeof(float));
                ggml_backend_graph_compute(ctx->backend, gf);
                int dn_T = (int)dn_t->ne[0];
                int dn_C = (int)dn_t->ne[1];
                xt_buf.resize((size_t)dn_T * dn_C);
                {
                    auto _rd = read_tensor_f32(dn_t);
                    memcpy(xt_buf.data(), _rd.data(), std::min(xt_buf.size(), _rd.size()) * sizeof(float));
                }
                xt_C = dn_C;
                xt_T = dn_T;
            }
            ggml_gallocr_free(al);
            ggml_free(cg);
        }

        fprintf(stderr, "htdemucs: channel downsample → freq (%d,%d,%d), time (%d,%d)\n", x_C, x_Fq, x_T, xt_C, xt_T);
    } else {
        fprintf(stderr, "htdemucs: no channel up/downsamplers (bottom_channels=0?)\n");
    }

    htd_capture(ctx, "post_transformer_z", x_buf.data(), x_buf.size());
    htd_capture(ctx, "post_transformer_xt", xt_buf.data(), xt_buf.size());

    // Step 8: Decoder (reverse of encoder, with skip connections)
    // Decoder index 0 is the innermost (smallest spatial dims), matching
    // encoder index depth-1. The freq dims grow back: 8→32→128→512→2048.
    int dec_strides[] = {4, 4, 4, 4}; // same stride pattern as encoder (all freq layers)
    for (int idx = 0; idx < hp.depth && encoder_ok; idx++) {
        auto& dec = m.decoder[idx];
        if (!dec.conv_tr_w)
            break;

        // Stride for this decoder layer (reverse of encoder)
        int stri = dec_strides[idx];                 // TODO: derive properly from encoder
        int pad_val = (int)dec.conv_tr_w->ne[1] / 4; // K/4

        // Pop skip from end (LIFO order)
        auto skip_f = saved_freq.back();
        saved_freq.pop_back();

        // Build decoder layer graph
        size_t d_ctx_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead();
        ggml_init_params dgp = {d_ctx_size, nullptr, true};
        ggml_context* dg = ggml_init(dgp);

        bool is_freq = dec.freq;
        bool is_last = (idx == hp.depth - 1);

        // Input tensor: (T, Fq, C, 1) for freq, (T, C) for time
        ggml_tensor* dx_in = nullptr;
        ggml_tensor* skip_in = nullptr;

        if (is_freq) {
            dx_in = ggml_new_tensor_4d(dg, GGML_TYPE_F32, x_T, x_Fq, x_C, 1);
            skip_in = ggml_new_tensor_4d(dg, GGML_TYPE_F32, skip_f.T, skip_f.Fq, skip_f.C, 1);
        } else {
            // TODO: handle non-freq decoder layers
            dx_in = ggml_new_tensor_4d(dg, GGML_TYPE_F32, x_T, x_Fq, x_C, 1);
            skip_in = ggml_new_tensor_4d(dg, GGML_TYPE_F32, skip_f.T, skip_f.Fq, skip_f.C, 1);
        }
        ggml_set_name(dx_in, "dec_in");
        ggml_set_input(dx_in);
        ggml_set_name(skip_in, "dec_skip");
        ggml_set_input(skip_in);

        ggml_tensor* dy = dx_in;

        if (!dec.empty) {
            // x = x + skip
            dy = ggml_add(dg, dy, skip_in);

            // Rewrite: Conv2d(C→2*C, [1+2*ctx, 1+2*ctx], stride=1, pad=[ctx,ctx]) → GroupNorm → GLU
            if (dec.rewrite_w) {
                int ctx_pad = hp.context; // context=1 for decoder
                ggml_tensor* drw = ggml_conv_2d(dg, dec.rewrite_w, dy, 1, 1, ctx_pad, ctx_pad, 1, 1);
                if (dec.rewrite_b) {
                    ggml_tensor* drwb = ggml_reshape_4d(dg, dec.rewrite_b, 1, 1, (int)dec.rewrite_b->ne[0], 1);
                    drw = ggml_add(dg, drw, drwb);
                }
                if (dec.norm1_w) {
                    drw = ggml_group_norm(dg, drw, 4, 1e-5f);
                    ggml_tensor* w4d = ggml_reshape_4d(dg, dec.norm1_w, 1, 1, (int)dec.norm1_w->ne[0], 1);
                    drw = ggml_mul(dg, drw, w4d);
                    if (dec.norm1_b) {
                        ggml_tensor* b4d = ggml_reshape_4d(dg, dec.norm1_b, 1, 1, (int)dec.norm1_b->ne[0], 1);
                        drw = ggml_add(dg, drw, b4d);
                    }
                }
                // GLU
                int drw_C = (int)drw->ne[2], drw_half = drw_C / 2;
                int drw_Fq = (int)drw->ne[1], drw_T = (int)drw->ne[0];
                size_t drw_ch_stride = drw->nb[2];
                ggml_tensor* da =
                    ggml_view_4d(dg, drw, drw_T, drw_Fq, drw_half, 1, drw->nb[1], drw->nb[2], drw->nb[3], 0);
                ggml_tensor* db = ggml_view_4d(dg, drw, drw_T, drw_Fq, drw_half, 1, drw->nb[1], drw->nb[2], drw->nb[3],
                                               (size_t)drw_half * drw_ch_stride);
                dy = ggml_mul(dg, ggml_cont(dg, da), ggml_sigmoid(dg, ggml_cont(dg, db)));
            }
        } else {
            // Empty decoder: dy stays as input (no skip add, no rewrite)
        }

        // ConvTranspose2d on freq axis (kernel [K, 1], stride [S, 1]).
        // ggml lacks conv_transpose_2d with asymmetric stride, so we do this
        // CPU-side: run the ggml graph up to here, read back dy, apply
        // ConvTranspose1d on the freq axis per-time-frame, write new dy.
        //
        // This is the approach taken by the first-pass parity build.
        // Phase 5 (GPU perf) will fuse this into a single graph.

        // First: finalize the pre-convtranspose ggml graph
        ggml_set_name(dy, "dec_pre_ct");
        ggml_set_output(dy);
        ggml_cgraph* dgf = ggml_new_graph(dg);
        ggml_build_forward_expand(dgf, dy);
        ggml_gallocr_t dalloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
        if (!ggml_gallocr_alloc_graph(dalloc, dgf)) {
            fprintf(stderr, "htdemucs: dec[%d] pre-CT graph alloc failed\n", idx);
            ggml_gallocr_free(dalloc);
            ggml_free(dg);
            break;
        }
        ggml_backend_tensor_set(dx_in, x_buf.data(), 0, x_buf.size() * sizeof(float));
        ggml_backend_tensor_set(skip_in, skip_f.data.data(), 0, skip_f.data.size() * sizeof(float));
        ggml_backend_graph_compute(ctx->backend, dgf);

        // Read pre-ConvTranspose result
        int pre_T = (int)dy->ne[0], pre_Fq = (int)dy->ne[1], pre_C = (int)dy->ne[2];
        size_t pre_n = (size_t)pre_T * pre_Fq * pre_C;
        std::vector<float> pre_buf(pre_n);
        {
            auto _rd = read_tensor_f32(dy);
            memcpy(pre_buf.data(), _rd.data(), std::min(pre_buf.size(), _rd.size()) * sizeof(float));
        }
        ggml_gallocr_free(dalloc);
        ggml_free(dg);

        if (htdemucs_debug()) {
            int nc = 0;
            float mx = 0;
            for (size_t i = 0; i < pre_n; i++) {
                if (std::isnan(pre_buf[i]))
                    nc++;
                float av = fabsf(pre_buf[i]);
                if (av > mx)
                    mx = av;
            }
            fprintf(stderr, "htdemucs: dec[%d] pre-CT (%d,%d,%d) max=%.2f nan=%d\n", idx, pre_C, pre_Fq, pre_T, mx, nc);
        }

        // CPU ConvTranspose2d with kernel [K,1], stride [S,1]
        // Weight layout in GGUF: ne = (1, K, OC, IC) for ConvTranspose2d
        // PyTorch ConvTranspose2d(IC, OC, [K,1], [S,1]) weight shape: (IC, OC, K, 1)
        // In ggml ne order: (KW=1, KH=K, ne2=OC, ne3=IC)
        {
            int ct_K = (int)dec.conv_tr_w->ne[1]; // KH
            int ct_OC = (int)dec.conv_tr_w->ne[2];
            int ct_IC = (int)dec.conv_tr_w->ne[3];
            int ct_S = stri;
            int ct_pad = pad_val;

            // Output freq size: Fq_out = (Fq_in - 1) * S + K - 2*pad
            // For htdemucs: pad = K/4 (symmetric), but ConvTranspose2d has no padding param.
            // Actually PyTorch ConvTranspose2d uses: Fq_out = (Fq_in - 1) * S - 2*P + K
            // where P = pad_val from the encoder. The decoder uses the same kernel_size and
            // the forward does: z = conv_tr(y), then crops: z[..., pad:-pad, :]
            // So the ConvTranspose output is (Fq_in-1)*S + K, then cropped by pad on each side.
            int ct_Fq_raw = (pre_Fq - 1) * ct_S + ct_K;
            int ct_Fq_out = ct_Fq_raw - 2 * ct_pad;

            // Read weight data (may be F16)
            size_t w_n = (size_t)ct_K * ct_OC * ct_IC;
            std::vector<float> w_data(w_n);
            if (dec.conv_tr_w->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> w_f16(w_n);
                ggml_backend_tensor_get(dec.conv_tr_w, w_f16.data(), 0, w_n * sizeof(ggml_fp16_t));
                for (size_t i = 0; i < w_n; i++)
                    w_data[i] = ggml_fp16_to_fp32(w_f16[i]);
            } else {
                {
                    auto _rd = read_tensor_f32(dec.conv_tr_w);
                    memcpy(w_data.data(), _rd.data(), std::min(w_data.size(), _rd.size()) * sizeof(float));
                }
            }

            // Output buffer: (pre_T, ct_Fq_out, ct_OC) = (T, Fq_out, OC)
            size_t ct_out_n = (size_t)pre_T * ct_Fq_out * ct_OC;
            std::vector<float> ct_out(ct_out_n, 0.0f);

            // ConvTranspose1d on freq axis: for each time frame t:
            // y[oc, fq_out, t] = sum_ic sum_kh x[ic, fq_in, t] * w[ic, oc, kh]
            // where fq_out = fq_in * S + kh - pad
            // Weight indexing: w_data[ic * OC * K + oc * K + kh] (ggml ne order: (1, K, OC, IC))
            // → w[ic][oc][kh] = w_data[ic * ct_OC * ct_K + oc * ct_K + kh]
            for (int t = 0; t < pre_T; t++) {
                for (int ic = 0; ic < ct_IC; ic++) {
                    for (int fq_in = 0; fq_in < pre_Fq; fq_in++) {
                        float x_val = pre_buf[t + (size_t)fq_in * pre_T + (size_t)ic * pre_T * pre_Fq];
                        for (int kh = 0; kh < ct_K; kh++) {
                            int fq_out = fq_in * ct_S + kh - ct_pad;
                            if (fq_out < 0 || fq_out >= ct_Fq_out)
                                continue;
                            for (int oc = 0; oc < ct_OC; oc++) {
                                float w_val = w_data[(size_t)ic * ct_OC * ct_K + oc * ct_K + kh];
                                ct_out[t + (size_t)fq_out * pre_T + (size_t)oc * pre_T * ct_Fq_out] += x_val * w_val;
                            }
                        }
                    }
                }
            }

            // Add bias
            if (dec.conv_tr_b) {
                std::vector<float> bias(ct_OC);
                {
                    auto _rd = read_tensor_f32(dec.conv_tr_b);
                    memcpy(bias.data(), _rd.data(), std::min(bias.size(), _rd.size()) * sizeof(float));
                }
                for (int oc = 0; oc < ct_OC; oc++) {
                    for (int fq = 0; fq < ct_Fq_out; fq++) {
                        for (int t = 0; t < pre_T; t++) {
                            ct_out[t + (size_t)fq * pre_T + (size_t)oc * pre_T * ct_Fq_out] += bias[oc];
                        }
                    }
                }
            }

            x_buf = std::move(ct_out);
            x_C = ct_OC;
            x_Fq = ct_Fq_out;
            x_T = pre_T;

            if (htdemucs_debug()) {
                fprintf(stderr, "htdemucs: dec[%d] ConvTranspose K=%d S=%d pad=%d: (%d,%d,%d) → (%d,%d,%d)\n", idx,
                        ct_K, ct_S, ct_pad, ct_IC, pre_Fq, pre_T, ct_OC, ct_Fq_out, pre_T);
            }
        }

        // GroupNorm after ConvTranspose (CPU-side)
        if (dec.norm2_w) {
            // CPU GroupNorm: normalize per group of channels over (Fq × T) spatial dims
            int ng = 4;
            std::vector<float> norm_w(x_C), norm_b(x_C);
            {
                auto _rd = read_tensor_f32(dec.norm2_w);
                memcpy(norm_w.data(), _rd.data(), std::min(norm_w.size(), _rd.size()) * sizeof(float));
            }
            if (dec.norm2_b) {
                auto _rd = read_tensor_f32(dec.norm2_b);
                memcpy(norm_b.data(), _rd.data(), std::min(norm_b.size(), _rd.size()) * sizeof(float));
            }
            int ch_per_group = x_C / ng;
            size_t spatial = (size_t)x_Fq * x_T;
            for (int grp = 0; grp < ng; grp++) {
                int c_start = grp * ch_per_group;
                int c_end = c_start + ch_per_group;
                // Compute mean/var over channels in this group and all spatial positions
                double sum = 0, sum2 = 0;
                size_t count = (size_t)(c_end - c_start) * spatial;
                for (int c = c_start; c < c_end; c++)
                    for (size_t s = 0; s < spatial; s++) {
                        float v = x_buf[s + (size_t)c * spatial];
                        sum += v;
                        sum2 += (double)v * v;
                    }
                float mean_g = (float)(sum / count);
                float var_g = (float)(sum2 / count - (double)mean_g * mean_g);
                float inv_std = 1.0f / sqrtf(var_g + 1e-5f);
                for (int c = c_start; c < c_end; c++)
                    for (size_t s = 0; s < spatial; s++) {
                        size_t i = s + (size_t)c * spatial;
                        x_buf[i] = (x_buf[i] - mean_g) * inv_std * norm_w[c] + (dec.norm2_b ? norm_b[c] : 0.0f);
                    }
            }
        }

        // Crop freq padding from ConvTranspose output
        // Python: if self.freq and self.pad: z = z[..., self.pad:-self.pad, :]
        // Already handled in the ConvTranspose code above (ct_Fq_out = ct_Fq_raw - 2*pad)

        // GELU (not on last layer)
        if (!is_last) {
            for (size_t i = 0; i < x_buf.size(); i++) {
                float v = x_buf[i];
                // GELU(x) = x * Φ(x) ≈ x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
                x_buf[i] = v * 0.5f * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
            }
        }

        // Time branch decoder
        int tdec_offset = hp.depth - (int)m.tdecoder.size();
        if (idx >= tdec_offset && (idx - tdec_offset) < (int)m.tdecoder.size()) {
            auto& tdec = m.tdecoder[idx - tdec_offset];
            if (tdec.conv_tr_w) {
                if (tdec.empty) {
                    // Empty tdecoder: take pre_buf (freq pre-ConvTranspose output),
                    // squeeze freq dim (Fq must be 1), run ConvTranspose1d
                    // pre_buf: (pre_T, pre_Fq, pre_C) where pre_Fq=1 (innermost layer)
                    // Squeeze → (pre_T, pre_C) = time signal
                    xt_buf.resize((size_t)pre_T * pre_C);
                    for (int c = 0; c < pre_C; c++)
                        for (int t = 0; t < pre_T; t++)
                            xt_buf[t + (size_t)c * pre_T] = pre_buf[t + (size_t)c * pre_T * pre_Fq];
                    xt_C = pre_C;
                    xt_T = pre_T;
                } else {
                    // Non-empty: xt = xt + skip_t, then rewrite + GLU
                    if (!saved_time.empty()) {
                        auto skip_t = saved_time.back();
                        saved_time.pop_back();
                        // xt + skip
                        for (int c = 0; c < xt_C; c++)
                            for (int t = 0; t < xt_T; t++)
                                xt_buf[t + (size_t)c * xt_T] += skip_t.data[t + (size_t)c * skip_t.T];
                    }
                    // Rewrite (1x1 conv) + GLU
                    if (tdec.rewrite_w) {
                        int rw_OC = 0;
                        auto rw_out = cpu_conv2d_1x1(xt_buf, xt_T, xt_C, tdec.rewrite_w, tdec.rewrite_b, rw_OC);
                        xt_buf = cpu_glu(rw_out, xt_T, rw_OC);
                        xt_C = rw_OC / 2;
                    }
                }

                // ConvTranspose1d on time axis
                int ct_K = (int)tdec.conv_tr_w->ne[0]; // kernel size
                int ct_OC = (int)tdec.conv_tr_w->ne[1];
                int ct_IC = (int)tdec.conv_tr_w->ne[2];
                int ct_pad = ct_K / 4;
                int t_out_raw = (xt_T - 1) * stri + ct_K;
                int t_length = !lengths_time.empty() ? lengths_time.back() : (t_out_raw - 2 * ct_pad);
                if (!lengths_time.empty())
                    lengths_time.pop_back();

                auto ct_w = read_tensor_f32(tdec.conv_tr_w);
                // ConvTranspose1d: out[oc, t_out] = sum_{ic, k} x[ic, t_in] * w[k, oc, ic]
                // where t_out = t_in * stride + k
                std::vector<float> ct_out(ct_OC * t_out_raw, 0.0f);
                for (int ic = 0; ic < ct_IC; ic++)
                    for (int t_in = 0; t_in < xt_T; t_in++) {
                        float x_val = xt_buf[t_in + (size_t)ic * xt_T];
                        for (int k = 0; k < ct_K; k++) {
                            int t_o = t_in * stri + k;
                            if (t_o >= t_out_raw)
                                continue;
                            for (int oc = 0; oc < ct_OC; oc++)
                                ct_out[t_o + (size_t)oc * t_out_raw] +=
                                    x_val * ct_w[(size_t)ic * ct_OC * ct_K + oc * ct_K + k];
                        }
                    }
                if (tdec.conv_tr_b) {
                    auto b = read_tensor_f32(tdec.conv_tr_b);
                    for (int oc = 0; oc < ct_OC; oc++)
                        for (int t = 0; t < t_out_raw; t++)
                            ct_out[t + (size_t)oc * t_out_raw] += b[oc];
                }
                // Crop padding: out[pad : pad + length]
                xt_T = std::min(t_length, t_out_raw - 2 * ct_pad);
                xt_C = ct_OC;
                std::vector<float> cropped(xt_C * xt_T);
                for (int oc = 0; oc < xt_C; oc++)
                    for (int t = 0; t < xt_T; t++)
                        cropped[t + (size_t)oc * xt_T] = ct_out[(ct_pad + t) + (size_t)oc * t_out_raw];
                xt_buf = std::move(cropped);

                // GroupNorm (skip — norm_starts=4) + GELU (not on last layer)
                bool tdec_last = (idx == hp.depth - 1);
                if (!tdec_last) {
                    for (size_t i = 0; i < xt_buf.size(); i++) {
                        float v = xt_buf[i];
                        xt_buf[i] = v * 0.5f * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
                    }
                }

                if (htdemucs_debug()) {
                    fprintf(stderr, "htdemucs: tdec[%d] output (%d, %d)\n", idx - tdec_offset, xt_C, xt_T);
                }
            }
        }

        if (htdemucs_debug()) {
            int nc = 0;
            for (size_t i = 0; i < x_buf.size() && nc < 3; i++)
                if (std::isnan(x_buf[i]) || std::isinf(x_buf[i]))
                    nc++;
            fprintf(stderr, "htdemucs: dec[%d] output (%d, %d, %d)%s\n", idx, x_C, x_Fq, x_T,
                    nc > 0 ? " *** HAS NaN ***" : "");
        }

        htd_capture(ctx, ("dec_freq_" + std::to_string(idx)).c_str(), x_buf.data(), x_buf.size());
    }

    // NaN check after decoder
    {
        int nan_count = 0;
        for (size_t i = 0; i < x_buf.size() && nan_count < 5; i++) {
            if (std::isnan(x_buf[i]) || std::isinf(x_buf[i]))
                nan_count++;
        }
        if (nan_count > 0)
            fprintf(stderr, "htdemucs: WARNING: %d NaN/Inf in decoder output\n", nan_count);
    }
    fprintf(stderr, "htdemucs: decoder done, output (%d, %d, %d)\n", x_C, x_Fq, x_T);

    // Step 9: CaC unmask → iSTFT → denormalize → sum branches → output
    //
    // Decoder output x_buf: (T, Fq, C, 1) where C = n_sources * audio_channels * 2
    // (CaC = 2× for real/imag). For 4 sources × 2 channels × 2 (real/imag) = 16 channels.
    // Reshape: (B, S, C*2//S, Fq, T) → for each source: (2, Fq, T) complex spec
    // Then iSTFT each source independently.

    int S = hp.n_sources;       // 4
    int ac = hp.audio_channels; // 2
    int ch_per_source = ac * 2; // 4 (2 audio channels × 2 for CaC real/imag)

    // Denormalize: x = x * std + mean
    for (size_t i = 0; i < x_buf.size(); i++) {
        x_buf[i] = x_buf[i] * spec_std + spec_mean;
    }

    // CaC unmask: decoder output → per-source complex spectrogram → iSTFT
    // x_buf layout: (x_T, x_Fq, x_C) where x_C = S * ch_per_source = 16
    // For source s, audio channel c: real = x_buf[..., s*ch_per_source + c*2]
    //                                 imag = x_buf[..., s*ch_per_source + c*2+1]
    auto r = new htdemucs_result();
    r->n_sources = S;
    r->n_channels = ac;
    r->n_samples = n_samples;
    r->sample_rate = hp.samplerate;
    r->sources = new float*[S];
    r->source_names = new const char*[S];

    for (int s = 0; s < S; s++) {
        r->source_names[s] = m.source_names[s].c_str();

        // Extract per-source complex spectrogram (ac channels × Fq × T)
        // Need to add back the dropped freq bin (Fq → Fq+1) with zeros
        int n_freqs_full = Fq + 1; // restore to nfft/2+1
        std::vector<float> src_real(ac * n_freqs_full * x_T, 0.0f);
        std::vector<float> src_imag(ac * n_freqs_full * x_T, 0.0f);

        for (int c = 0; c < ac; c++) {
            for (int fq = 0; fq < Fq; fq++) {
                for (int t = 0; t < x_T; t++) {
                    // x_buf index: t + fq * x_T + ch * x_T * x_Fq
                    int ch_re = s * ch_per_source + c * 2;
                    int ch_im = s * ch_per_source + c * 2 + 1;
                    float re = x_buf[t + (size_t)fq * x_T + (size_t)ch_re * x_T * x_Fq];
                    float im = x_buf[t + (size_t)fq * x_T + (size_t)ch_im * x_T * x_Fq];
                    // Output spec layout for iSTFT: (ch, freqs, frames)
                    src_real[(size_t)c * n_freqs_full * x_T + (size_t)fq * x_T + t] = re;
                    src_imag[(size_t)c * n_freqs_full * x_T + (size_t)fq * x_T + t] = im;
                }
            }
        }

        // iSTFT: complex spec → waveform
        // Python _ispec: F.pad(z, (0,0,0,1)) adds back the dropped freq bin,
        // F.pad(z, (2,2)) adds 2 frames on each side, then ispectro with length.
        // We've already restored the freq bin; for the frame padding we skip it
        // (the added frames are zeros anyway and get trimmed).
        std::vector<float> src_pcm(ac * work_length, 0.0f);
        compute_istft(src_real.data(), src_imag.data(), ac, n_freqs_full, x_T, nfft, hop, ctx->hann_window.data(),
                      work_length, src_pcm.data());

        // Denormalize time branch and add to freq branch
        // xt_buf: (xt_T, xt_C) where xt_C = S * ac = 4 * 2 = 8
        // Reshape to per-source: xt[s][c][t] = xt_buf[t + (s*ac+c)*xt_T]
        // Denormalize: xt = xt * time_std + time_mean
        if (xt_C >= S * ac && xt_T >= n_samples) {
            int src_ch = s * ac;
            for (int c = 0; c < ac; c++) {
                for (int i = 0; i < n_samples; i++) {
                    float tv = xt_buf[i + (size_t)(src_ch + c) * xt_T];
                    tv = tv * time_std + time_mean;
                    src_pcm[(size_t)c * work_length + i] += tv;
                }
            }
        }

        // Capture channel-major (ac, n_samples) to match the reference layout,
        // before the interleave below.
        if (ctx->capture_stages) {
            std::vector<float> chan_major((size_t)ac * n_samples);
            for (int c = 0; c < ac; c++)
                for (int i = 0; i < n_samples; i++)
                    chan_major[(size_t)c * n_samples + i] = src_pcm[(size_t)c * work_length + i];
            htd_capture(ctx, ("output_" + m.source_names[s]).c_str(), chan_major.data(), chan_major.size());
        }

        // Trim to original length and interleave stereo
        r->sources[s] = new float[(size_t)ac * n_samples];
        for (int i = 0; i < n_samples; i++) {
            for (int c = 0; c < ac; c++) {
                r->sources[s][i * ac + c] = src_pcm[(size_t)c * work_length + i];
            }
        }
    }

    fprintf(stderr, "htdemucs: separated %d samples → %d sources (%d ch @ %d Hz)\n", n_samples, S, ac, hp.samplerate);

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

// ---------------------------------------------------------------------------
// Parity diff harness (tools/reference_backends/htdemucs.py is the reference)
// ---------------------------------------------------------------------------

namespace {

double htd_cosine(const float* a, const float* b, int64_t n) {
    double dot = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    if (na == 0 || nb == 0)
        return (na == 0 && nb == 0) ? 1.0 : 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

double htd_max_abs_diff(const float* a, const float* b, int64_t n) {
    double m = 0;
    for (int64_t i = 0; i < n; i++)
        m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

double htd_l2_norm(const float* a, int64_t n) {
    double s = 0;
    for (int64_t i = 0; i < n; i++)
        s += (double)a[i] * a[i];
    return std::sqrt(s);
}

bool htd_ref_get(core_gguf::WeightLoad& rw, const char* name, std::vector<float>& out, int64_t& nelem) {
    auto it = rw.tensors.find(name);
    if (it == rw.tensors.end() || !it->second)
        return false;
    ggml_tensor* t = it->second;
    nelem = ggml_nelements(t);
    out.resize((size_t)nelem);
    ggml_backend_tensor_get(t, out.data(), 0, (size_t)nelem * sizeof(float));
    return true;
}

} // namespace

int htdemucs_diff(const char* model_gguf, const char* ref_gguf, const char* audio_wav, int verbosity) {
    (void)audio_wav; // The waveform is replayed FROM the reference (input-aligned).

    htdemucs_context* ctx = htdemucs_init_from_file(model_gguf, htdemucs_default_params());
    if (!ctx) {
        fprintf(stderr, "htdemucs_diff: failed to load model %s\n", model_gguf);
        return 2;
    }

    core_gguf::WeightLoad rw;
    if (!core_gguf::load_weights(ref_gguf, ctx->backend, "htdemucs_ref", rw)) {
        fprintf(stderr, "htdemucs_diff: failed to load reference %s\n", ref_gguf);
        htdemucs_free(ctx);
        return 2;
    }

    // Replay the exact 44.1 kHz stereo waveform the reference ran on, so a
    // resampler mismatch can never be mistaken for a model parity failure.
    std::vector<float> in_wav;
    int64_t in_n = 0;
    if (!htd_ref_get(rw, "input_wav", in_wav, in_n)) {
        fprintf(stderr, "htdemucs_diff: reference has no input_wav stage — re-dump with the updated dumper\n");
        htdemucs_free(ctx);
        return 2;
    }
    const int ac = ctx->model.hparams.audio_channels;
    const int n_samples = (int)(in_n / ac);
    std::vector<float> interleaved((size_t)n_samples * ac);
    for (int c = 0; c < ac; c++)
        for (int i = 0; i < n_samples; i++)
            interleaved[(size_t)i * ac + c] = in_wav[(size_t)c * n_samples + i];

    fprintf(stderr, "htdemucs_diff: replaying reference input (%d ch × %d samples)\n", ac, n_samples);

    ctx->capture_stages = true;
    htdemucs_result* res = htdemucs_separate(ctx, interleaved.data(), n_samples);
    if (!res) {
        fprintf(stderr, "htdemucs_diff: separate() failed\n");
        htdemucs_free(ctx);
        return 2;
    }

    // F32 gate per the dev guide. Stages are compared in reference order so the
    // FIRST failure is the earliest divergence = the bug.
    const double COS_MIN = 0.999;
    int n_fail = 0, n_missing = 0, n_run = 0;
    const char* first_fail = nullptr;

    auto report = [&](const char* stage) {
        std::vector<float> ref;
        int64_t rn = 0;
        if (!htd_ref_get(rw, stage, ref, rn))
            return; // stage not in this dump
        auto it = ctx->captures.find(stage);
        if (it == ctx->captures.end()) {
            fprintf(stderr, "  %-20s MISSING (no C++ capture)\n", stage);
            n_missing++;
            return;
        }
        const std::vector<float>& mine = it->second;
        const int64_t n = (int64_t)std::min(mine.size(), (size_t)rn);
        const double cos = htd_cosine(mine.data(), ref.data(), n);
        const double mad = htd_max_abs_diff(mine.data(), ref.data(), n);
        const bool same_size = mine.size() == (size_t)rn;
        const bool ok = cos >= COS_MIN && same_size;
        n_run++;
        if (!ok) {
            n_fail++;
            if (!first_fail)
                first_fail = stage;
        }
        // Always print both magnitudes: a 10-30x outlier on either side means
        // "same name, wrong data" (a harness bug), not a runtime bug.
        fprintf(stderr, "  %-20s %s cos=%.6f max_abs=%.3e  mine=%zu ref=%lld  |mine|=%.4f |ref|=%.4f%s\n", stage,
                ok ? "PASS" : "FAIL", cos, mad, mine.size(), (long long)rn, htd_l2_norm(mine.data(), n),
                htd_l2_norm(ref.data(), n), same_size ? "" : "  *** SHAPE MISMATCH ***");
        (void)verbosity;
    };

    fprintf(stderr, "\n=== htdemucs per-stage parity (cos_min >= %.3f) ===\n", COS_MIN);
    report("spec_input");
    report("time_input");
    for (int i = 0; i < 4; i++)
        report(("enc_freq_" + std::to_string(i)).c_str());
    for (int i = 0; i < 3; i++)
        report(("enc_time_" + std::to_string(i)).c_str());
    report("pre_transformer_z");
    report("pre_transformer_xt");
    report("post_transformer_z");
    report("post_transformer_xt");
    for (int i = 0; i < 4; i++)
        report(("dec_freq_" + std::to_string(i)).c_str());
    for (const char* s : {"drums", "bass", "other", "vocals"})
        report(("output_" + std::string(s)).c_str());

    fprintf(stderr, "\n%d/%d stages passed", n_run - n_fail, n_run);
    if (n_missing)
        fprintf(stderr, ", %d missing", n_missing);
    if (first_fail)
        fprintf(stderr, " — FIRST DIVERGENCE: %s", first_fail);
    fprintf(stderr, "\n");

    htdemucs_result_free(res);
    htdemucs_free(ctx);
    return n_fail > 0 ? 1 : 0;
}
