// miocodec.cpp — MioCodec v2 decoder implementation.
//
// Decode path: FSQ codebook lookup → wave_prenet Transformer → conv_upsample
// → interp → ResNet prior → wave_decoder Transformer (AdaLN-Zero) → ResNet
// post → SnakeBeta upsampler → ISTFTHead → 44.1 kHz waveform.

#include "miocodec.h"

#include "core/gguf_loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ============================================================================
// Model hyperparameters (read from GGUF metadata)
// ============================================================================
struct miocodec_hparams {
    uint32_t sample_rate = 44100;
    uint32_t n_fft = 392;
    uint32_t hop_length = 98;
    uint32_t downsample_factor = 2;

    // FSQ
    std::vector<int> fsq_levels = {8, 8, 8, 5, 5};
    uint32_t fsq_input_dim = 768;
    uint32_t fsq_output_dim = 768;

    // Local encoder (for reference — not used in decode-only)
    uint32_t local_enc_dim = 768;
    uint32_t local_enc_n_layers = 6;

    // Wave prenet
    uint32_t wave_prenet_dim = 768;
    uint32_t wave_prenet_output_dim = 512;
    uint32_t wave_prenet_n_layers = 6;
    uint32_t wave_prenet_n_heads = 12;
    uint32_t wave_prenet_window_size = 65;
    float wave_prenet_rope_theta = 10000.0f;

    // Wave decoder (AdaLN-Zero)
    uint32_t wave_dec_dim = 512;
    uint32_t wave_dec_n_layers = 8;
    uint32_t wave_dec_n_heads = 8;
    uint32_t wave_dec_window_size = 65;
    float wave_dec_rope_theta = 10000.0f;
    uint32_t wave_dec_adaln_cond_dim = 128;

    // Wave misc
    uint32_t wave_decoder_dim = 512;
    uint32_t wave_upsample_factor = 2;
    uint32_t wave_resnet_num_blocks = 2;
    uint32_t wave_resnet_kernel_size = 3;
    uint32_t wave_resnet_num_groups = 32;
    std::vector<int> wave_upsampler_factors = {3, 3};
    std::vector<int> wave_upsampler_kernel_sizes = {9, 9};

    // Global encoder
    uint32_t global_enc_output_channels = 128;
};

// ============================================================================
// Model weights
// ============================================================================
struct miocodec_weights {
    // FSQ
    ggml_tensor* fsq_proj_in_w = nullptr;  // [5, 768]
    ggml_tensor* fsq_proj_in_b = nullptr;  // [5]
    ggml_tensor* fsq_proj_out_w = nullptr; // [768, 5]
    ggml_tensor* fsq_proj_out_b = nullptr; // [768]

    // Wave prenet (6 layers)
    struct transformer_layer {
        ggml_tensor* attn_norm_w = nullptr;
        ggml_tensor* attn_norm_b = nullptr;
        ggml_tensor* wq = nullptr;
        ggml_tensor* wk = nullptr;
        ggml_tensor* wv = nullptr;
        ggml_tensor* wo = nullptr;
        ggml_tensor* ffn_norm_w = nullptr;
        ggml_tensor* ffn_norm_b = nullptr;
        ggml_tensor* ffn_w1 = nullptr;
        ggml_tensor* ffn_w2 = nullptr;
        ggml_tensor* ffn_w3 = nullptr;
        // AdaLN-Zero (only for wave_decoder layers)
        ggml_tensor* attn_adaln_w = nullptr; // condition_proj[1].weight
        ggml_tensor* attn_adaln_b = nullptr; // condition_proj[1].bias
        ggml_tensor* ffn_adaln_w = nullptr;
        ggml_tensor* ffn_adaln_b = nullptr;
    };

    // Wave prenet
    ggml_tensor* wave_prenet_input_proj_w = nullptr; // input_proj if dim != input_dim
    ggml_tensor* wave_prenet_output_proj_w = nullptr;
    ggml_tensor* wave_prenet_norm_w = nullptr;
    ggml_tensor* wave_prenet_norm_b = nullptr;
    std::vector<transformer_layer> wave_prenet_layers;

    // Wave conv upsample
    ggml_tensor* wave_conv_up_w = nullptr; // [512, 512, 2]
    ggml_tensor* wave_conv_up_b = nullptr; // [512]

    // Wave prior/post ResNet blocks
    struct resnet_block {
        ggml_tensor* norm1_w = nullptr;
        ggml_tensor* norm1_b = nullptr;
        ggml_tensor* conv1_w = nullptr;
        ggml_tensor* conv1_b = nullptr;
        ggml_tensor* norm2_w = nullptr;
        ggml_tensor* norm2_b = nullptr;
        ggml_tensor* conv2_w = nullptr;
        ggml_tensor* conv2_b = nullptr;
    };
    std::vector<resnet_block> wave_prior_net;
    std::vector<resnet_block> wave_post_net;

    // Wave decoder (8 layers, AdaLN-Zero)
    ggml_tensor* wave_dec_norm_adaln_w = nullptr; // final norm condition_proj
    ggml_tensor* wave_dec_norm_adaln_b = nullptr;
    std::vector<transformer_layer> wave_dec_layers;

    // Wave upsampler
    struct upsampler_stage {
        ggml_tensor* conv_w0 = nullptr; // parametrizations.weight.original0
        ggml_tensor* conv_w1 = nullptr; // parametrizations.weight.original1
        ggml_tensor* conv_b = nullptr;
        ggml_tensor* snake_alpha = nullptr;
        ggml_tensor* snake_beta = nullptr;
        // ResNet block inside upsampler
        resnet_block resblk;
    };
    std::vector<upsampler_stage> wave_upsampler_stages;
    ggml_tensor* wave_upsampler_out_proj_w = nullptr;
    ggml_tensor* wave_upsampler_out_proj_b = nullptr;
    ggml_tensor* wave_upsampler_out_snake_alpha = nullptr;
    ggml_tensor* wave_upsampler_out_snake_beta = nullptr;

    // ISTFT head
    ggml_tensor* istft_out_w = nullptr; // [394, 512]
    ggml_tensor* istft_out_b = nullptr; // [394]
};

// ============================================================================
// Context
// ============================================================================
struct miocodec_context {
    miocodec_hparams hparams;
    miocodec_weights weights;

    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    ggml_context* ctx_w = nullptr; // weight context

    int verbosity = 0;
};

// ============================================================================
// Public API
// ============================================================================

struct miocodec_params miocodec_default_params(void) {
    miocodec_params p = {};
    p.n_threads = 4;
    p.verbosity = 0;
    p.use_gpu = false;
    return p;
}

struct miocodec_context* miocodec_init_from_file(const char* path, struct miocodec_params params) {
    auto* ctx = new miocodec_context();
    ctx->verbosity = params.verbosity;

    // Pass 1: read metadata (hyperparameters)
    gguf_context* gctx = core_gguf::open_metadata(path);
    if (!gctx) {
        fprintf(stderr, "miocodec: failed to open '%s'\n", path);
        delete ctx;
        return nullptr;
    }

    auto& hp = ctx->hparams;
    hp.sample_rate = core_gguf::kv_u32(gctx, "miocodec.sample_rate", 44100);
    hp.n_fft = core_gguf::kv_u32(gctx, "miocodec.n_fft", 392);
    hp.hop_length = core_gguf::kv_u32(gctx, "miocodec.hop_length", 98);
    hp.downsample_factor = core_gguf::kv_u32(gctx, "miocodec.downsample_factor", 2);
    hp.wave_prenet_n_layers = core_gguf::kv_u32(gctx, "miocodec.wave_prenet.n_layers", 6);
    hp.wave_prenet_dim = core_gguf::kv_u32(gctx, "miocodec.wave_prenet.dim", 768);
    hp.wave_prenet_output_dim = core_gguf::kv_u32(gctx, "miocodec.wave_prenet.output_dim", 512);
    hp.wave_prenet_n_heads = core_gguf::kv_u32(gctx, "miocodec.wave_prenet.n_heads", 12);
    hp.wave_prenet_window_size = core_gguf::kv_u32(gctx, "miocodec.wave_prenet.window_size", 65);
    hp.wave_dec_n_layers = core_gguf::kv_u32(gctx, "miocodec.wave_dec.n_layers", 8);
    hp.wave_dec_dim = core_gguf::kv_u32(gctx, "miocodec.wave_dec.dim", 512);
    hp.wave_dec_n_heads = core_gguf::kv_u32(gctx, "miocodec.wave_dec.n_heads", 8);
    hp.wave_dec_adaln_cond_dim = core_gguf::kv_u32(gctx, "miocodec.wave_dec.adaln_cond_dim", 128);
    hp.wave_upsample_factor = core_gguf::kv_u32(gctx, "miocodec.wave_upsample_factor", 2);
    hp.wave_resnet_num_blocks = core_gguf::kv_u32(gctx, "miocodec.wave_resnet_num_blocks", 2);
    hp.wave_decoder_dim = core_gguf::kv_u32(gctx, "miocodec.wave_dec.dim", 512);
    core_gguf::free_metadata(gctx);

    // Pass 2: load weights
    ctx->backend = ggml_backend_cpu_init();
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx->backend, "miocodec", wl)) {
        fprintf(stderr, "miocodec: failed to load weights from '%s'\n", path);
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }
    ctx->ctx_w = wl.ctx;
    ctx->buf_w = wl.buf;

    // Resolve weight tensors
    auto& w = ctx->weights;
    auto T = [&](const char* name) -> ggml_tensor* { return ggml_get_tensor(ctx->ctx_w, name); };

    // FSQ
    w.fsq_proj_in_w = T("local_quantizer.proj_in.weight");
    w.fsq_proj_in_b = T("local_quantizer.proj_in.bias");
    w.fsq_proj_out_w = T("local_quantizer.proj_out.weight");
    w.fsq_proj_out_b = T("local_quantizer.proj_out.bias");

    // ISTFT head
    w.istft_out_w = T("istft_head.out.weight");
    w.istft_out_b = T("istft_head.out.bias");

    // Wave conv upsample
    w.wave_conv_up_w = T("wave_conv_upsample.weight");
    w.wave_conv_up_b = T("wave_conv_upsample.bias");

    // Wave prenet layers
    w.wave_prenet_layers.resize(hp.wave_prenet_n_layers);
    w.wave_prenet_norm_w = T("wave_prenet.norm.weight");
    w.wave_prenet_norm_b = T("wave_prenet.norm.bias");
    w.wave_prenet_output_proj_w = T("wave_prenet.output_proj.weight");
    for (uint32_t i = 0; i < hp.wave_prenet_n_layers; i++) {
        auto& L = w.wave_prenet_layers[i];
        char buf[256];
        auto N = [&](const char* sfx) {
            snprintf(buf, sizeof(buf), "wave_prenet.layers.%u.%s", i, sfx);
            return T(buf);
        };
        L.attn_norm_w = N("attention_norm.weight");
        L.attn_norm_b = N("attention_norm.bias");
        L.wq = N("attention.wq.weight");
        L.wk = N("attention.wk.weight");
        L.wv = N("attention.wv.weight");
        L.wo = N("attention.wo.weight");
        L.ffn_norm_w = N("ffn_norm.weight");
        L.ffn_norm_b = N("ffn_norm.bias");
        L.ffn_w1 = N("feed_forward.w1.weight");
        L.ffn_w2 = N("feed_forward.w2.weight");
        L.ffn_w3 = N("feed_forward.w3.weight");
    }

    // Wave decoder layers (AdaLN-Zero)
    w.wave_dec_layers.resize(hp.wave_dec_n_layers);
    w.wave_dec_norm_adaln_w = T("wave_decoder.norm.condition_proj.1.weight");
    w.wave_dec_norm_adaln_b = T("wave_decoder.norm.condition_proj.1.bias");
    for (uint32_t i = 0; i < hp.wave_dec_n_layers; i++) {
        auto& L = w.wave_dec_layers[i];
        char buf[256];
        auto N = [&](const char* sfx) {
            snprintf(buf, sizeof(buf), "wave_decoder.layers.%u.%s", i, sfx);
            return T(buf);
        };
        L.wq = N("attention.wq.weight");
        L.wk = N("attention.wk.weight");
        L.wv = N("attention.wv.weight");
        L.wo = N("attention.wo.weight");
        L.ffn_w1 = N("feed_forward.w1.weight");
        L.ffn_w2 = N("feed_forward.w2.weight");
        L.ffn_w3 = N("feed_forward.w3.weight");
        L.attn_adaln_w = N("attention_norm.condition_proj.1.weight");
        L.attn_adaln_b = N("attention_norm.condition_proj.1.bias");
        L.ffn_adaln_w = N("ffn_norm.condition_proj.1.weight");
        L.ffn_adaln_b = N("ffn_norm.condition_proj.1.bias");
    }

    // ResNet blocks (prior + post)
    w.wave_prior_net.resize(hp.wave_resnet_num_blocks);
    w.wave_post_net.resize(hp.wave_resnet_num_blocks);
    for (uint32_t i = 0; i < hp.wave_resnet_num_blocks; i++) {
        char buf[256];
        auto PN = [&](const char* sfx) {
            snprintf(buf, sizeof(buf), "wave_prior_net.blocks.%u.%s", i, sfx);
            return T(buf);
        };
        auto& pb = w.wave_prior_net[i];
        pb.norm1_w = PN("norm1.weight");
        pb.norm1_b = PN("norm1.bias");
        pb.conv1_w = PN("conv1.weight");
        pb.conv1_b = PN("conv1.bias");
        pb.norm2_w = PN("norm2.weight");
        pb.norm2_b = PN("norm2.bias");
        pb.conv2_w = PN("conv2.weight");
        pb.conv2_b = PN("conv2.bias");

        auto QN = [&](const char* sfx) {
            snprintf(buf, sizeof(buf), "wave_post_net.blocks.%u.%s", i, sfx);
            return T(buf);
        };
        auto& qb = w.wave_post_net[i];
        qb.norm1_w = QN("norm1.weight");
        qb.norm1_b = QN("norm1.bias");
        qb.conv1_w = QN("conv1.weight");
        qb.conv1_b = QN("conv1.bias");
        qb.norm2_w = QN("norm2.weight");
        qb.norm2_b = QN("norm2.bias");
        qb.conv2_w = QN("conv2.weight");
        qb.conv2_b = QN("conv2.bias");
    }

    // Upsampler stages
    int n_up_stages = (int)hp.wave_upsampler_factors.size();
    if (n_up_stages == 0)
        n_up_stages = 2; // default [3,3]
    w.wave_upsampler_stages.resize(n_up_stages);
    for (int i = 0; i < n_up_stages; i++) {
        char buf[256];
        auto UL = [&](const char* sfx) {
            snprintf(buf, sizeof(buf), "wave_upsampler.upsample_layers.%d.%s", i, sfx);
            return T(buf);
        };
        auto& s = w.wave_upsampler_stages[i];
        s.conv_w0 = UL("parametrizations.weight.original0");
        s.conv_w1 = UL("parametrizations.weight.original1");
        s.conv_b = UL("bias");
        snprintf(buf, sizeof(buf), "wave_upsampler.snake_activations.%d.alpha", i);
        s.snake_alpha = T(buf);
        snprintf(buf, sizeof(buf), "wave_upsampler.snake_activations.%d.beta", i);
        s.snake_beta = T(buf);
        // ResNet block inside upsampler
        auto UR = [&](const char* sfx) {
            snprintf(buf, sizeof(buf), "wave_upsampler.resnet_blocks.%d.%s", i, sfx);
            return T(buf);
        };
        s.resblk.norm1_w = UR("norm1.weight");
        s.resblk.norm1_b = UR("norm1.bias");
        s.resblk.conv1_w = UR("conv1.weight");
        s.resblk.conv1_b = UR("conv1.bias");
        s.resblk.norm2_w = UR("norm2.weight");
        s.resblk.norm2_b = UR("norm2.bias");
        s.resblk.conv2_w = UR("conv2.weight");
        s.resblk.conv2_b = UR("conv2.bias");
    }
    w.wave_upsampler_out_proj_w = T("wave_upsampler.out_proj.weight");
    w.wave_upsampler_out_proj_b = T("wave_upsampler.out_proj.bias");
    w.wave_upsampler_out_snake_alpha = T("wave_upsampler.out_snake.alpha");
    w.wave_upsampler_out_snake_beta = T("wave_upsampler.out_snake.beta");

    if (params.verbosity > 0) {
        fprintf(stderr, "miocodec: loaded model from '%s'\n", path);
        fprintf(stderr, "  sample_rate=%u, n_fft=%u, hop=%u\n", hp.sample_rate, hp.n_fft, hp.hop_length);
        fprintf(stderr, "  wave_prenet: %uL %ud, wave_decoder: %uL %ud (adaln=%u)\n", hp.wave_prenet_n_layers,
                hp.wave_prenet_dim, hp.wave_dec_n_layers, hp.wave_dec_dim, hp.wave_dec_adaln_cond_dim);
    }

    return ctx;
}

void miocodec_free(struct miocodec_context* ctx) {
    if (!ctx)
        return;
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    if (ctx->buf_w)
        ggml_backend_buffer_free(ctx->buf_w);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

uint32_t miocodec_sample_rate(const struct miocodec_context* ctx) {
    return ctx ? ctx->hparams.sample_rate : 44100;
}
uint32_t miocodec_n_fft(const struct miocodec_context* ctx) {
    return ctx ? ctx->hparams.n_fft : 392;
}
uint32_t miocodec_hop_length(const struct miocodec_context* ctx) {
    return ctx ? ctx->hparams.hop_length : 98;
}
uint32_t miocodec_codebook_size(const struct miocodec_context* ctx) {
    (void)ctx;
    return 12800; // product of FSQ levels [8,8,8,5,5]
}
uint32_t miocodec_token_rate(const struct miocodec_context* ctx) {
    (void)ctx;
    return 25;
}

// ============================================================================
// FSQ Decode: token indices → (T, 768) embeddings
// ============================================================================
// FSQ levels = [8, 8, 8, 5, 5], basis = [1, 8, 64, 512, 2560]
// indices_to_codes: index → per-dim codes → normalize to [-1, 1]
// Then proj_out(codes) → 768-dim embeddings

static void fsq_indices_to_codes(const int32_t* indices, int n, float* out_codes) {
    // levels = [8, 8, 8, 5, 5], basis = [1, 8, 64, 512, 2560]
    static const int levels[5] = {8, 8, 8, 5, 5};
    static const int basis[5] = {1, 8, 64, 512, 2560};
    static const int half_width[5] = {4, 4, 4, 2, 2}; // levels // 2

    for (int t = 0; t < n; t++) {
        int idx = indices[t];
        for (int d = 0; d < 5; d++) {
            int code_raw = (idx / basis[d]) % levels[d];
            // _scale_and_shift_inverse: (code - half_width) / half_width
            out_codes[t * 5 + d] = (float)(code_raw - half_width[d]) / (float)half_width[d];
        }
    }
}

float* miocodec_decode(struct miocodec_context* ctx, const int32_t* token_indices, int n_tokens,
                       const float* global_embedding, int target_audio_length, int* out_n_samples) {
    if (!ctx || !token_indices || n_tokens <= 0 || !global_embedding || !out_n_samples)
        return nullptr;

    // TODO: implement full decode graph (Transformer + ResNet + Upsampler + ISTFT)
    // For now, just implement FSQ decode as the first stage to verify parity.

    (void)target_audio_length;
    *out_n_samples = 0;

    fprintf(stderr, "miocodec_decode: not yet implemented (use miocodec_extract_stage for per-stage testing)\n");
    return nullptr;
}

float* miocodec_extract_stage(struct miocodec_context* ctx, const int32_t* token_indices, int n_tokens,
                              const float* global_embedding, int target_audio_length, const char* stage_name,
                              int* out_n) {
    if (!ctx || !token_indices || n_tokens <= 0 || !out_n || !stage_name)
        return nullptr;

    (void)global_embedding;
    (void)target_audio_length;

    // Stage: fsq_decoded — pure CPU computation, no ggml graph needed
    if (strcmp(stage_name, "fsq_decoded") == 0) {
        // Step 1: indices → 5-dim codes
        std::vector<float> codes(n_tokens * 5);
        fsq_indices_to_codes(token_indices, n_tokens, codes.data());

        // Step 2: proj_out(codes) → 768-dim
        // proj_out: Linear(5, 768) → out = codes @ W^T + bias
        const int out_dim = 768;
        float* result = (float*)malloc(sizeof(float) * n_tokens * out_dim);
        if (!result)
            return nullptr;

        // Get weight data from ggml tensors (handle F16 → F32 dequant)
        std::vector<float> proj_w(out_dim * 5);
        std::vector<float> proj_b(out_dim);

        ggml_tensor* tw = ctx->weights.fsq_proj_out_w;
        ggml_tensor* tb = ctx->weights.fsq_proj_out_b;
        if (tw->type == GGML_TYPE_F16) {
            std::vector<uint16_t> tmp(out_dim * 5);
            ggml_backend_tensor_get(tw, tmp.data(), 0, sizeof(uint16_t) * out_dim * 5);
            for (size_t i = 0; i < tmp.size(); i++)
                proj_w[i] = ggml_fp16_to_fp32(tmp[i]);
        } else {
            ggml_backend_tensor_get(tw, proj_w.data(), 0, sizeof(float) * out_dim * 5);
        }
        // Bias is always F32
        ggml_backend_tensor_get(tb, proj_b.data(), 0, sizeof(float) * out_dim);

        // result[t, d] = sum_k(codes[t, k] * W[d, k]) + bias[d]
        // GGUF shape is [5, 768] meaning ne[0]=5, ne[1]=768.
        // Memory layout: element at (d, k) is proj_w[d * 5 + k].
        for (int t = 0; t < n_tokens; t++) {
            for (int d = 0; d < out_dim; d++) {
                float sum = proj_b[d];
                for (int k = 0; k < 5; k++) {
                    sum += codes[t * 5 + k] * proj_w[d * 5 + k];
                }
                result[t * out_dim + d] = sum;
            }
        }

        *out_n = n_tokens * out_dim;
        return result;
    }

    fprintf(stderr, "miocodec_extract_stage: unknown stage '%s'\n", stage_name);
    *out_n = 0;
    return nullptr;
}
