// miotts.cpp — MioTTS backend (Aratako/MioTTS-{0.6B,1.7B} + MioCodec).
//
// LLM: standard Qwen3 forward with KV cache (reuses core_attn::kv_self_attn).
// Codec: FSQ dequant → wave_prenet → conv_upsample → ResNet → wave_decoder
//        (AdaLN-Zero) → ResNet → iSTFT → 24kHz waveform.
//
// The LLM generates speech tokens from text; the codec converts them to audio.
// Voice cloning injects a 128-d global embedding at codec decode time.

#include "miotts.h"

#include "core/attention.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

// ── Hyperparameters ─────────────────────────────────────────────────

struct miotts_hparams {
    // LLM
    uint32_t n_layers = 28;
    uint32_t n_heads = 16;
    uint32_t n_kv_heads = 8;
    uint32_t d_model = 1024;
    uint32_t d_ff = 3072;
    uint32_t vocab_size = 164480;
    uint32_t head_dim = 64;
    uint32_t max_pos = 32768;
    float rope_theta = 1000000.0f;
    float rms_norm_eps = 1e-6f;

    // Speech token range in the vocabulary
    uint32_t speech_token_start = 151669;
    uint32_t speech_token_end = 164469;
    uint32_t eos_token_id = 151645; // <|endoftext|>

    // Codec
    uint32_t codec_sample_rate = 24000;
    uint32_t codec_frame_rate = 25;
    uint32_t codec_n_fft = 1920;
    uint32_t codec_hop_length = 480;
    uint32_t codec_codebook_size = 12800;
    int32_t fsq_levels[5] = {8, 8, 8, 5, 5};
    uint32_t codec_global_dim = 128;
    uint32_t codec_wave_dim = 512;
    uint32_t codec_wave_prenet_layers = 6;
    uint32_t codec_wave_decoder_layers = 8;
};

// ── LLM weight block ────────────────────────────────────────────────

struct miotts_layer {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_output_w = nullptr;
    ggml_tensor* attn_q_norm_w = nullptr;
    ggml_tensor* attn_k_norm_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

// ── Context ─────────────────────────────────────────────────────────

struct miotts_context {
    miotts_context_params params;
    miotts_hparams hp;

    // Backend
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf_weights = nullptr;

    // LLM weights
    ggml_tensor* token_embd = nullptr;
    ggml_tensor* output_norm = nullptr;
    ggml_tensor* output_w = nullptr; // lm_head (may be tied to token_embd)
    std::vector<miotts_layer> layers;

    // KV cache
    ggml_tensor* kv_k = nullptr; // [head_dim, n_kv_heads, max_ctx, n_layers]
    ggml_tensor* kv_v = nullptr;
    ggml_backend_buffer_t buf_kv = nullptr;

    // Codec weights (FSQ dequant projection)
    ggml_tensor* fsq_proj_out_w = nullptr; // (768, 5)
    ggml_tensor* fsq_proj_out_b = nullptr; // (768,)

    // Global embedding for voice cloning (128-d, set by miotts_set_reference)
    std::vector<float> global_embedding;

    // Compute scratch
    std::vector<uint8_t> compute_meta;

    // GGUF contexts (kept alive for weight buffer lifetime)
    ggml_context* ctx_weights = nullptr;
    ggml_context* ctx_kv = nullptr;

    ~miotts_context() {
        if (buf_kv)
            ggml_backend_buffer_free(buf_kv);
        if (buf_weights)
            ggml_backend_buffer_free(buf_weights);
        if (ctx_kv)
            ggml_free(ctx_kv);
        if (ctx_weights)
            ggml_free(ctx_weights);
        if (backend)
            ggml_backend_free(backend);
    }
};

// ── Default params ──────────────────────────────────────────────────

miotts_context_params miotts_context_default_params(void) {
    return {
        /*n_threads*/ 4,
        /*verbosity*/ 1,
        /*use_gpu*/ false,
        /*temperature*/ 0.8f,
        /*seed*/ 0,
        /*max_tokens*/ 750, // 30s at 25Hz
        /*flash_attn*/ false,
    };
}

// ── Load model ──────────────────────────────────────────────────────

static bool load_hparams(gguf_context* meta, miotts_hparams& hp) {
    auto get_u32 = [&](const char* key, uint32_t def) -> uint32_t {
        int idx = gguf_find_key(meta, key);
        return idx >= 0 ? (uint32_t)gguf_get_val_u32(meta, idx) : def;
    };
    auto get_f32 = [&](const char* key, float def) -> float {
        int idx = gguf_find_key(meta, key);
        return idx >= 0 ? gguf_get_val_f32(meta, idx) : def;
    };

    hp.n_layers = get_u32("miotts.block_count", 28);
    hp.n_heads = get_u32("miotts.attention.head_count", 16);
    hp.n_kv_heads = get_u32("miotts.attention.head_count_kv", 8);
    hp.d_model = get_u32("miotts.embedding_length", 1024);
    hp.d_ff = get_u32("miotts.feed_forward_length", 3072);
    hp.vocab_size = get_u32("miotts.vocab_size", 164480);
    hp.head_dim = get_u32("miotts.head_dim", hp.d_model / hp.n_heads);
    hp.max_pos = get_u32("miotts.context_length", 32768);
    hp.rope_theta = get_f32("miotts.rope_theta", 1000000.0f);
    hp.rms_norm_eps = get_f32("miotts.rms_norm_eps", 1e-6f);

    hp.speech_token_start = get_u32("miotts.speech_token_start", 151669);
    hp.speech_token_end = get_u32("miotts.speech_token_end", 164469);
    hp.eos_token_id = get_u32("miotts.eos_token_id", 151645);

    hp.codec_sample_rate = get_u32("miotts.codec.sample_rate", 24000);
    hp.codec_frame_rate = get_u32("miotts.codec.frame_rate", 25);
    hp.codec_n_fft = get_u32("miotts.codec.n_fft", 1920);
    hp.codec_hop_length = get_u32("miotts.codec.hop_length", 480);
    hp.codec_codebook_size = get_u32("miotts.codec.codebook_size", 12800);
    hp.codec_global_dim = get_u32("miotts.codec.global_dim", 128);
    hp.codec_wave_dim = get_u32("miotts.codec.wave_dim", 512);
    hp.codec_wave_prenet_layers = get_u32("miotts.codec.wave_prenet_layers", 6);
    hp.codec_wave_decoder_layers = get_u32("miotts.codec.wave_decoder_layers", 8);

    return true;
}

miotts_context* miotts_init_from_file(const char* path_model, miotts_context_params params) {
    auto c = std::make_unique<miotts_context>();
    c->params = params;

    // Init backend
    c->backend = ggml_backend_init_best();
    if (!c->backend) {
        fprintf(stderr, "miotts: failed to init backend\n");
        return nullptr;
    }

    // Load GGUF
    gguf_init_params gip = {/*.no_alloc=*/true, /*.ctx=*/&c->ctx_weights};
    gguf_context* meta = gguf_init_from_file(path_model, gip);
    if (!meta) {
        fprintf(stderr, "miotts: failed to load '%s'\n", path_model);
        return nullptr;
    }

    if (!load_hparams(meta, c->hp)) {
        fprintf(stderr, "miotts: failed to read hyperparameters\n");
        gguf_free(meta);
        return nullptr;
    }

    const auto& hp = c->hp;
    if (params.verbosity >= 1) {
        fprintf(stderr, "miotts: %u layers, %u heads (%u KV), d=%u, vocab=%u\n", hp.n_layers, hp.n_heads, hp.n_kv_heads,
                hp.d_model, hp.vocab_size);
        fprintf(stderr, "miotts: codec=%uHz, FSQ codebook=%u, n_fft=%u\n", hp.codec_frame_rate, hp.codec_codebook_size,
                hp.codec_n_fft);
    }

    // Allocate weight buffer
    c->buf_weights = ggml_backend_alloc_ctx_tensors(c->ctx_weights, c->backend);
    if (!c->buf_weights) {
        fprintf(stderr, "miotts: failed to allocate weight buffer\n");
        gguf_free(meta);
        return nullptr;
    }

    // Load weights from GGUF into the allocated buffer
    {
        FILE* f = fopen(path_model, "rb");
        if (!f) {
            fprintf(stderr, "miotts: cannot open '%s'\n", path_model);
            gguf_free(meta);
            return nullptr;
        }
        const size_t data_offset = gguf_get_data_offset(meta);
        const int n_tensors = gguf_get_n_tensors(meta);
        for (int i = 0; i < n_tensors; i++) {
            const char* name = gguf_get_tensor_name(meta, i);
            ggml_tensor* t = ggml_get_tensor(c->ctx_weights, name);
            if (!t)
                continue;
            const size_t offset = data_offset + gguf_get_tensor_offset(meta, i);
            const size_t nbytes = ggml_nbytes(t);
            fseek(f, (long)offset, SEEK_SET);
            std::vector<uint8_t> buf(nbytes);
            if (fread(buf.data(), 1, nbytes, f) != nbytes) {
                fprintf(stderr, "miotts: short read on tensor '%s'\n", name);
            }
            ggml_backend_tensor_set(t, buf.data(), 0, nbytes);
        }
        fclose(f);
    }

    // Resolve LLM weight pointers
    auto T = [&](const char* name) -> ggml_tensor* { return ggml_get_tensor(c->ctx_weights, name); };

    c->token_embd = T("token_embd.weight");
    c->output_norm = T("output_norm.weight");
    c->output_w = T("output.weight");
    // Tied embeddings: if output.weight is missing, use token_embd
    if (!c->output_w)
        c->output_w = c->token_embd;

    c->layers.resize(hp.n_layers);
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        auto& b = c->layers[il];
        char buf[128];
        auto tn = [&](const char* suffix) {
            snprintf(buf, sizeof(buf), "blk.%u.%s", il, suffix);
            return T(buf);
        };
        b.attn_norm_w = tn("attn_norm.weight");
        b.attn_q_w = tn("attn_q.weight");
        b.attn_k_w = tn("attn_k.weight");
        b.attn_v_w = tn("attn_v.weight");
        b.attn_output_w = tn("attn_output.weight");
        b.attn_q_norm_w = tn("attn_q_norm.weight");
        b.attn_k_norm_w = tn("attn_k_norm.weight");
        b.ffn_norm_w = tn("ffn_norm.weight");
        b.ffn_gate_w = tn("ffn_gate.weight");
        b.ffn_up_w = tn("ffn_up.weight");
        b.ffn_down_w = tn("ffn_down.weight");
    }

    // Resolve codec FSQ weights
    c->fsq_proj_out_w = T("codec.local_quantizer.proj_out.weight");
    c->fsq_proj_out_b = T("codec.local_quantizer.proj_out.bias");

    // Allocate KV cache
    {
        const int max_ctx = 4096; // sufficient for TTS (30s @ 25Hz = 750 codec tokens + prompt)
        const size_t n_mem = (size_t)max_ctx * hp.n_layers;
        const size_t kv_overhead = 2 * n_mem * ggml_tensor_overhead() + ggml_graph_overhead();
        ggml_init_params kv_ip = {kv_overhead, nullptr, true};
        c->ctx_kv = ggml_init(kv_ip);
        c->kv_k = ggml_new_tensor_4d(c->ctx_kv, GGML_TYPE_F16, hp.head_dim, hp.n_kv_heads, max_ctx, hp.n_layers);
        c->kv_v = ggml_new_tensor_4d(c->ctx_kv, GGML_TYPE_F16, hp.head_dim, hp.n_kv_heads, max_ctx, hp.n_layers);
        ggml_set_name(c->kv_k, "kv_k");
        ggml_set_name(c->kv_v, "kv_v");
        c->buf_kv = ggml_backend_alloc_ctx_tensors(c->ctx_kv, c->backend);
        if (!c->buf_kv) {
            fprintf(stderr, "miotts: failed to allocate KV cache\n");
            gguf_free(meta);
            return nullptr;
        }
        // Zero-init KV cache
        ggml_backend_tensor_set(c->kv_k, std::vector<uint8_t>(ggml_nbytes(c->kv_k), 0).data(), 0, ggml_nbytes(c->kv_k));
        ggml_backend_tensor_set(c->kv_v, std::vector<uint8_t>(ggml_nbytes(c->kv_v), 0).data(), 0, ggml_nbytes(c->kv_v));
    }

    // Compute scratch
    c->compute_meta.resize(512 * 1024 * 1024); // 512 MB compute arena

    // Init global embedding to zeros (no voice cloning by default)
    c->global_embedding.resize(hp.codec_global_dim, 0.0f);

    gguf_free(meta);

    if (params.verbosity >= 1) {
        fprintf(stderr, "miotts: loaded OK (%zu LLM layers, FSQ proj %s)\n", c->layers.size(),
                c->fsq_proj_out_w ? "yes" : "NO");
    }

    return c.release();
}

// ── LLM graph build ─────────────────────────────────────────────────

static ggml_cgraph* build_graph_llm(miotts_context* c, int n_past, int n_tokens) {
    const auto& hp = c->hp;
    const int d = (int)hp.d_model;
    const int n_q = (int)hp.n_heads;
    const int n_kv = (int)hp.n_kv_heads;
    const int hd = (int)hp.head_dim;
    const int n_kv_grp = n_q / n_kv;
    const float eps = hp.rms_norm_eps;
    const float theta = hp.rope_theta;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const int T = n_tokens;

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // Input: token IDs
    ggml_tensor* input_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(input_ids, "input_ids");
    ggml_set_input(input_ids);

    // Embedding lookup
    ggml_tensor* embeds = ggml_get_rows(ctx0, c->token_embd, input_ids);

    // Position IDs
    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // Causal mask (only for prefill, T > 1)
    ggml_tensor* causal_mask = nullptr;
    if (T > 1) {
        const int Lk = n_past + T;
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    const core_attn::KvSelfAttnParams kvp = {
        /*n_heads*/ n_q,
        /*n_kv_heads*/ n_kv,
        /*head_dim*/ hd,
        /*n_kv_grp*/ n_kv_grp,
        /*n_ctx_orig*/ (int)hp.max_pos,
        /*rope_theta*/ theta,
        /*rope_beta_fast*/ 32.0f,
        /*rope_beta_slow*/ 1.0f,
        /*attn_scale*/ attn_scale,
        /*qk_norm_eps*/ eps,
        /*gqa_mode*/ core_attn::GQA_NATIVE,
    };

    ggml_tensor* cur = embeds;
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        const auto& b = c->layers[il];
        ggml_tensor* residual = cur;

        ggml_tensor* x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);

        ggml_tensor* attn = core_attn::kv_self_attn(
            ctx0, gf, x, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_output_w, b.attn_q_norm_w, b.attn_k_norm_w,
            positions, T > 1 ? causal_mask : nullptr, c->kv_k, c->kv_v, (int)il, n_past, kvp);
        cur = ggml_add(ctx0, residual, attn);

        residual = cur;
        x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, c->output_norm);

    // For T > 1, take only the last position's hidden state
    if (T > 1) {
        cur = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
    }

    // Project to vocab logits
    ggml_tensor* logits = ggml_mul_mat(ctx0, c->output_w, cur);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    return gf;
}

// ── Sampling ────────────────────────────────────────────────────────

static int32_t sample_token(const float* logits, int vocab_size, float temperature, std::mt19937& rng) {
    if (temperature <= 0.0f) {
        // Greedy
        return (int32_t)(std::max_element(logits, logits + vocab_size) - logits);
    }
    // Temperature sampling
    std::vector<float> probs(vocab_size);
    float max_val = *std::max_element(logits, logits + vocab_size);
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = std::exp((logits[i] - max_val) / temperature);
        sum += probs[i];
    }
    for (int i = 0; i < vocab_size; i++)
        probs[i] /= sum;

    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return (int32_t)dist(rng);
}

// ── FSQ dequantize ──────────────────────────────────────────────────

float* miotts_fsq_dequant(miotts_context* ctx, const int32_t* indices, int n_indices, int* out_dim) {
    if (!ctx || !indices || n_indices <= 0)
        return nullptr;

    // FSQ levels [8, 8, 8, 5, 5], basis [1, 8, 64, 512, 2560]
    const int levels[5] = {8, 8, 8, 5, 5};
    const int basis[5] = {1, 8, 64, 512, 2560};
    const int fsq_dim = 5;
    const int embed_dim = 768; // proj_out output dim

    // Dequantize: index → codes → normalized codes → projected embedding
    std::vector<float> codes(n_indices * fsq_dim);
    for (int t = 0; t < n_indices; t++) {
        int idx = indices[t];
        for (int d = 0; d < fsq_dim; d++) {
            int code = (idx / basis[d]) % levels[d];
            int half = levels[d] / 2;
            codes[t * fsq_dim + d] = (float)(code - half) / (float)half;
        }
    }

    // Project with proj_out: (T, 5) @ (768, 5)^T + bias → (T, 768)
    if (!ctx->fsq_proj_out_w) {
        // No projection available — return raw codes
        if (out_dim)
            *out_dim = fsq_dim;
        float* result = (float*)malloc(n_indices * fsq_dim * sizeof(float));
        memcpy(result, codes.data(), n_indices * fsq_dim * sizeof(float));
        return result;
    }

    // Read projection weights — handle F16 storage by reading raw bytes
    // then converting if needed.
    const size_t w_nelem = (size_t)embed_dim * fsq_dim;
    std::vector<float> proj_w(w_nelem);
    std::vector<float> proj_b(embed_dim);

    // proj_out.weight
    {
        const size_t nbytes = ggml_nbytes(ctx->fsq_proj_out_w);
        std::vector<uint8_t> raw(nbytes);
        ggml_backend_tensor_get(ctx->fsq_proj_out_w, raw.data(), 0, nbytes);
        if (ctx->fsq_proj_out_w->type == GGML_TYPE_F16) {
            const ggml_fp16_t* src = (const ggml_fp16_t*)raw.data();
            for (size_t i = 0; i < w_nelem; i++)
                proj_w[i] = ggml_fp16_to_fp32(src[i]);
        } else {
            memcpy(proj_w.data(), raw.data(), w_nelem * sizeof(float));
        }
    }
    // proj_out.bias (always F32 — norms/biases are kept F32 by convention)
    ggml_backend_tensor_get(ctx->fsq_proj_out_b, proj_b.data(), 0, embed_dim * sizeof(float));

    float* result = (float*)malloc(n_indices * embed_dim * sizeof(float));
    // Matrix multiply: result[t, d] = sum_k(codes[t, k] * proj_w[d, k]) + proj_b[d]
    // proj_w is (768, 5) row-major = weight[out_dim, in_dim]
    for (int t = 0; t < n_indices; t++) {
        for (int d = 0; d < embed_dim; d++) {
            float val = proj_b[d];
            for (int k = 0; k < fsq_dim; k++) {
                val += codes[t * fsq_dim + k] * proj_w[d * fsq_dim + k];
            }
            result[t * embed_dim + d] = val;
        }
    }

    if (out_dim)
        *out_dim = embed_dim;
    return result;
}

// ── Forward logits (diff harness) ───────────────────────────────────

float* miotts_forward_logits(miotts_context* ctx, const int32_t* token_ids, int n_tokens, int* out_vocab) {
    if (!ctx || !token_ids || n_tokens <= 0)
        return nullptr;

    ggml_cgraph* gf = build_graph_llm(ctx, /*n_past=*/0, n_tokens);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "miotts: graph alloc failed\n");
        ggml_gallocr_free(alloc);
        return nullptr;
    }

    // Set inputs
    ggml_tensor* input_ids_t = ggml_graph_get_tensor(gf, "input_ids");
    ggml_backend_tensor_set(input_ids_t, token_ids, 0, n_tokens * sizeof(int32_t));

    ggml_tensor* positions_t = ggml_graph_get_tensor(gf, "positions");
    std::vector<int32_t> pos(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        pos[i] = i;
    ggml_backend_tensor_set(positions_t, pos.data(), 0, n_tokens * sizeof(int32_t));

    // Causal mask
    if (n_tokens > 1) {
        ggml_tensor* mask_t = ggml_graph_get_tensor(gf, "causal_mask");
        const int Lk = n_tokens;
        std::vector<ggml_fp16_t> mask(Lk * n_tokens);
        for (int q = 0; q < n_tokens; q++) {
            for (int k = 0; k < Lk; k++) {
                mask[q * Lk + k] = (k <= q) ? ggml_fp32_to_fp16(0.0f) : ggml_fp32_to_fp16(-INFINITY);
            }
        }
        ggml_backend_tensor_set(mask_t, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    ggml_backend_graph_compute(ctx->backend, gf);

    // Read logits
    ggml_tensor* logits_t = ggml_graph_get_tensor(gf, "logits");
    const int vocab = (int)logits_t->ne[0];
    float* result = (float*)malloc(vocab * sizeof(float));
    ggml_backend_tensor_get(logits_t, result, 0, vocab * sizeof(float));

    if (out_vocab)
        *out_vocab = vocab;

    ggml_gallocr_free(alloc);
    return result;
}

// ── Synthesize ──────────────────────────────────────────────────────

int miotts_set_reference(miotts_context* ctx, const float* audio_24k, int n_samples) {
    if (!ctx)
        return -1;
    if (!audio_24k || n_samples <= 0) {
        std::fill(ctx->global_embedding.begin(), ctx->global_embedding.end(), 0.0f);
        return 0;
    }
    // TODO: run the global encoder on the reference audio to extract the 128-d embedding.
    // For now, zero embedding (default voice).
    fprintf(stderr, "miotts: warning: reference audio encoding not yet implemented, using default voice\n");
    return 0;
}

float* miotts_synthesize(miotts_context* ctx, const char* text, int* out_n) {
    if (!ctx || !text || !out_n)
        return nullptr;
    *out_n = 0;

    const auto& hp = ctx->hp;

    // TODO: implement full synthesis pipeline:
    // 1. Tokenize text with Qwen3 BPE (ChatML: <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n)
    // 2. Run LLM forward to generate speech tokens
    // 3. Extract speech token indices from generated IDs
    // 4. Run MioCodec decode (FSQ dequant → transformer → iSTFT)
    // The diff harness validates each stage independently first via
    // miotts_forward_logits() and miotts_fsq_dequant().
    (void)hp;
    fprintf(stderr, "miotts: synthesize not yet fully implemented (need tokenizer + codec decode)\n");
    return nullptr;
}

void miotts_free_audio(float* pcm) {
    free(pcm);
}

void miotts_free(miotts_context* ctx) {
    delete ctx;
}
