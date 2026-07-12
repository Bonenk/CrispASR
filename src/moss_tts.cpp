// moss_tts.cpp — MOSS-TTS-v1.5 (MossTTSDelay) ggml runtime.
//
// Backbone: Qwen3-8B KV-cached decoder (cloned from qwen3_asr's Qwen3 path)
// that takes precomputed input embeddings and exposes BOTH the text logits and
// the per-token last hidden state (fed to the 32 audio LM heads). The audio
// extension (32 embed tables + 32 heads), the delay-pattern state machine, and
// the AR code-generation loop are ported from pwilkin/openmoss (validated C++
// reference) and MossTTSDelayModel.generate. The transformer RVQ codec (codes ->
// waveform) lands in Phase 4 (src/moss_tts_codec.*); until then synthesize()
// reports "codec not loaded" and generate_codes() produces the code grid for
// parity testing.
//
// See docs/moss-tts/STUDY.md for the full spec and the delay-pattern gotchas.

#include "moss_tts.h"

#include "core/attention.h"
#include "core/bpe.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===========================================================================
// Hyper-parameters
// ===========================================================================

struct moss_tts_hparams {
    // Qwen3-8B backbone
    uint32_t llm_hidden = 4096;
    uint32_t llm_layers = 36;
    uint32_t llm_n_heads = 32;
    uint32_t llm_n_kv_heads = 8;
    uint32_t llm_head_dim = 128;
    uint32_t llm_ff_dim = 12288;
    uint32_t llm_vocab_size = 155648;
    uint32_t llm_max_pos = 40960;
    float llm_rope_theta = 1000000.0f;
    float llm_rms_eps = 1e-6f;
    uint32_t llm_lm_head_dim = 0; // 0 = use vocab_size

    // Audio / delay
    uint32_t n_vq = 32;
    uint32_t audio_vocab_size = 1024; // WITHOUT +1 pad
    uint32_t audio_pad_code = 1024;
    uint32_t sampling_rate = 24000;
    uint32_t downsample_rate = 1920;

    // Special token ids
    uint32_t tok_audio_start = 151652;
    uint32_t tok_audio_end = 151653;
    uint32_t tok_audio_user_slot = 151654;
    uint32_t tok_audio_gen_slot = 151656;
    uint32_t tok_audio_delay_slot = 151662;
    uint32_t tok_im_start = 151644;
    uint32_t tok_im_end = 151645;
    uint32_t tok_pad = 151643;
};

// ===========================================================================
// Model / vocab / context
// ===========================================================================

struct moss_tts_llm_block {
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

struct moss_tts_model {
    moss_tts_hparams hparams;

    ggml_tensor* token_embd_w = nullptr;  // llm.embed.weight   (hidden, vocab)
    ggml_tensor* output_norm_w = nullptr; // llm.final_norm.weight
    ggml_tensor* output_w = nullptr;      // llm.lm_head.weight  (hidden, vocab)
    std::vector<moss_tts_llm_block> blocks;

    std::vector<ggml_tensor*> audio_embed; // moss.audio_embed.{i}.weight (hidden, 1025)
    std::vector<ggml_tensor*> audio_head;  // moss.audio_head.{i}.weight  (hidden, 1025)

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
};

struct moss_tts_vocab {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;
};

struct moss_tts_context {
    moss_tts_context_params params;
    moss_tts_model model;
    moss_tts_vocab vocab;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // KV cache: (head_dim, max_ctx, n_kv_heads, n_layers) x {K, V}
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    int kv_max_ctx = 0;

    int n_threads = 4;
    uint32_t seed = 0;

    // Codec (Phase 4) — loaded lazily from the companion GGUF.
    std::string codec_path;
    bool codec_loaded = false;
};

// ===========================================================================
// Tensor bind helpers
// ===========================================================================

static ggml_tensor* mt_try(moss_tts_model& m, const char* name) {
    return core_gguf::try_get(m.tensors, name);
}
static ggml_tensor* mt_req(moss_tts_model& m, const char* name) {
    return core_gguf::require(m.tensors, name, "moss_tts");
}

// ===========================================================================
// Model loading
// ===========================================================================

static bool moss_tts_load_model(moss_tts_model& model, moss_tts_vocab& vocab, const char* path,
                                ggml_backend_t backend) {
    // ---- pass 1: metadata + vocab ----
    {
        gguf_context* gctx = core_gguf::open_metadata(path);
        if (!gctx)
            return false;
        auto& hp = model.hparams;
        hp.llm_hidden = core_gguf::kv_u32(gctx, "moss_tts.llm.hidden_size", hp.llm_hidden);
        hp.llm_layers = core_gguf::kv_u32(gctx, "moss_tts.llm.num_layers", hp.llm_layers);
        hp.llm_n_heads = core_gguf::kv_u32(gctx, "moss_tts.llm.num_heads", hp.llm_n_heads);
        hp.llm_n_kv_heads = core_gguf::kv_u32(gctx, "moss_tts.llm.num_kv_heads", hp.llm_n_kv_heads);
        hp.llm_head_dim = core_gguf::kv_u32(gctx, "moss_tts.llm.head_dim", hp.llm_head_dim);
        hp.llm_ff_dim = core_gguf::kv_u32(gctx, "moss_tts.llm.intermediate_size", hp.llm_ff_dim);
        hp.llm_vocab_size = core_gguf::kv_u32(gctx, "moss_tts.llm.vocab_size", hp.llm_vocab_size);
        hp.llm_max_pos = core_gguf::kv_u32(gctx, "moss_tts.llm.max_position_embeddings", hp.llm_max_pos);
        hp.llm_rope_theta = core_gguf::kv_f32(gctx, "moss_tts.llm.rope_theta", hp.llm_rope_theta);
        hp.llm_rms_eps = core_gguf::kv_f32(gctx, "moss_tts.llm.rms_norm_eps", hp.llm_rms_eps);

        hp.n_vq = core_gguf::kv_u32(gctx, "moss.n_vq", hp.n_vq);
        hp.audio_vocab_size = core_gguf::kv_u32(gctx, "moss.audio_vocab_size", hp.audio_vocab_size);
        hp.audio_pad_code = core_gguf::kv_u32(gctx, "moss.audio_pad_code", hp.audio_pad_code);
        hp.sampling_rate = core_gguf::kv_u32(gctx, "moss.sampling_rate", hp.sampling_rate);
        hp.downsample_rate = core_gguf::kv_u32(gctx, "moss.downsample_rate", hp.downsample_rate);

        hp.tok_audio_start = core_gguf::kv_u32(gctx, "moss.token.audio_start", hp.tok_audio_start);
        hp.tok_audio_end = core_gguf::kv_u32(gctx, "moss.token.audio_end", hp.tok_audio_end);
        hp.tok_audio_user_slot = core_gguf::kv_u32(gctx, "moss.token.audio_user_slot", hp.tok_audio_user_slot);
        hp.tok_audio_gen_slot = core_gguf::kv_u32(gctx, "moss.token.audio_gen_slot", hp.tok_audio_gen_slot);
        hp.tok_audio_delay_slot = core_gguf::kv_u32(gctx, "moss.token.audio_delay_slot", hp.tok_audio_delay_slot);
        hp.tok_im_start = core_gguf::kv_u32(gctx, "moss.token.im_start", hp.tok_im_start);
        hp.tok_im_end = core_gguf::kv_u32(gctx, "moss.token.im_end", hp.tok_im_end);
        hp.tok_pad = core_gguf::kv_u32(gctx, "moss.token.pad", hp.tok_pad);

        auto tokens = core_gguf::kv_str_array(gctx, "tokenizer.ggml.tokens");
        if (!tokens.empty()) {
            vocab.id_to_token = std::move(tokens);
            vocab.token_to_id.reserve(vocab.id_to_token.size());
            for (int i = 0; i < (int)vocab.id_to_token.size(); i++)
                vocab.token_to_id[vocab.id_to_token[i]] = i;
        }
        auto merges = core_gguf::kv_str_array(gctx, "tokenizer.ggml.merges");
        for (int i = 0; i < (int)merges.size(); i++)
            vocab.merge_rank[merges[i]] = i;

        core_gguf::free_metadata(gctx);
    }

    // ---- pass 2: tensor data ----
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "moss_tts", wl))
        return false;
    model.ctx = wl.ctx;
    model.buf = wl.buf;
    model.tensors = std::move(wl.tensors);

    // ---- bind ----
    model.token_embd_w = mt_req(model, "llm.embed.weight");
    model.output_norm_w = mt_req(model, "llm.final_norm.weight");
    model.output_w = mt_req(model, "llm.lm_head.weight");

    model.blocks.resize(model.hparams.llm_layers);
    for (uint32_t i = 0; i < model.hparams.llm_layers; i++) {
        char buf[128];
        auto& b = model.blocks[i];
        auto get = [&](const char* suf) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "llm.blk.%u.%s", i, suf);
            return mt_req(model, buf);
        };
        b.attn_norm_w = get("attn_norm.weight");
        b.attn_q_w = get("attn.q.weight");
        b.attn_k_w = get("attn.k.weight");
        b.attn_v_w = get("attn.v.weight");
        b.attn_output_w = get("attn.o.weight");
        b.attn_q_norm_w = get("attn.q_norm.weight");
        b.attn_k_norm_w = get("attn.k_norm.weight");
        b.ffn_norm_w = get("ffn_norm.weight");
        b.ffn_gate_w = get("ffn.gate.weight");
        b.ffn_up_w = get("ffn.up.weight");
        b.ffn_down_w = get("ffn.down.weight");
    }

    // Audio extension tables/heads (kept F16 by the converter).
    model.audio_embed.resize(model.hparams.n_vq, nullptr);
    model.audio_head.resize(model.hparams.n_vq, nullptr);
    for (uint32_t i = 0; i < model.hparams.n_vq; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "moss.audio_embed.%u.weight", i);
        model.audio_embed[i] = mt_req(model, buf);
        snprintf(buf, sizeof(buf), "moss.audio_head.%u.weight", i);
        model.audio_head[i] = mt_req(model, buf);
    }

    const auto& hp = model.hparams;
    fprintf(stderr, "moss_tts: loaded %u LLM layers (d=%u), n_vq=%u, audio_vocab=%u, vocab=%u\n", hp.llm_layers,
            hp.llm_hidden, hp.n_vq, hp.audio_vocab_size, hp.llm_vocab_size);
    return true;
}

// ===========================================================================
// Backbone graph (KV-cached) — text logits + hidden_last, last position only.
// ===========================================================================

static ggml_cgraph* moss_tts_build_graph_llm_kv(moss_tts_context* ctx, int n_past, int n_tokens) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int d = (int)hp.llm_hidden;
    const int n_q = (int)hp.llm_n_heads;
    const int n_kv = (int)hp.llm_n_kv_heads;
    const int hd = (int)hp.llm_head_dim;
    const int n_kv_grp = n_q / n_kv;
    const float eps = hp.llm_rms_eps;
    const float theta = hp.llm_rope_theta;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const int T = n_tokens;
    const int Lk = n_past + T;

    GGML_ASSERT(ctx->kv_k && ctx->kv_v);
    GGML_ASSERT(Lk <= ctx->kv_max_ctx);

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
    ggml_set_name(embeds, "inputs_embeds");
    ggml_set_input(embeds);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    ggml_tensor* causal_mask = nullptr;
    if (T > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    const core_attn::KvSelfAttnParams kvp = {
        /*n_heads*/ n_q,
        /*n_kv_heads*/ n_kv,
        /*head_dim*/ hd,
        /*n_kv_grp*/ n_kv_grp,
        /*n_ctx_orig*/ (int)hp.llm_max_pos,
        /*rope_theta*/ theta,
        /*rope_beta_fast*/ 32.0f,
        /*rope_beta_slow*/ 1.0f,
        /*attn_scale*/ attn_scale,
        /*qk_norm_eps*/ eps,
        /*gqa_mode*/ core_attn::GQA_MANUAL_CONT,
    };

    ggml_tensor* cur = embeds;
    for (uint32_t il = 0; il < hp.llm_layers; il++) {
        const auto& b = m.blocks[il];
        ggml_tensor* residual = cur;

        ggml_tensor* x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);

        ggml_tensor* attn = core_attn::kv_self_attn(
            ctx0, gf, x, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_output_w, b.attn_q_norm_w, b.attn_k_norm_w,
            positions, (T == 1) ? nullptr : causal_mask, ctx->kv_k, ctx->kv_v, (int)il, n_past, kvp, /*qkv_w*/ nullptr);
        cur = ggml_add(ctx0, residual, attn);

        residual = cur;
        x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, m.output_norm_w);

    // Slice to the LAST position (autoregressive loop only needs the tail).
    if (T > 1) {
        cur = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
    }
    // hidden_last: the input to the 32 audio heads (post final-norm, pre lm_head).
    ggml_tensor* hidden_last = ggml_cont(ctx0, cur);
    ggml_set_name(hidden_last, "hidden_last");
    ggml_set_output(hidden_last);
    ggml_build_forward_expand(gf, hidden_last);

    ggml_tensor* logits = ggml_mul_mat(ctx0, m.output_w, hidden_last);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    ggml_free(ctx0);
    return gf;
}

// ===========================================================================
// Public: KV init / reset
// ===========================================================================

extern "C" bool moss_tts_kv_init(moss_tts_context* ctx, int max_ctx) {
    if (!ctx || max_ctx <= 0)
        return false;
    if (ctx->kv_k && ctx->kv_max_ctx >= max_ctx)
        return true;
    if (ctx->kv_buf) {
        ggml_backend_buffer_free(ctx->kv_buf);
        ctx->kv_buf = nullptr;
    }
    if (ctx->kv_ctx) {
        ggml_free(ctx->kv_ctx);
        ctx->kv_ctx = nullptr;
    }
    const auto& hp = ctx->model.hparams;
    const int hd = (int)hp.llm_head_dim;
    const int n_kv = (int)hp.llm_n_kv_heads;
    const int nl = (int)hp.llm_layers;
    const ggml_type kv_type = core_attn::kv_dtype_from_env("moss_tts");

    ggml_init_params ip = {ggml_tensor_overhead() * 4, nullptr, true};
    ctx->kv_ctx = ggml_init(ip);
    if (!ctx->kv_ctx)
        return false;
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, kv_type, hd, max_ctx, n_kv, nl);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, kv_type, hd, max_ctx, n_kv, nl);
    ggml_set_name(ctx->kv_k, "kv_k");
    ggml_set_name(ctx->kv_v, "kv_v");
    ctx->kv_buf = ggml_backend_alloc_ctx_tensors(ctx->kv_ctx, ctx->backend);
    if (!ctx->kv_buf) {
        ggml_free(ctx->kv_ctx);
        ctx->kv_ctx = nullptr;
        ctx->kv_k = ctx->kv_v = nullptr;
        return false;
    }
    ctx->kv_max_ctx = max_ctx;
    return true;
}

extern "C" void moss_tts_kv_reset(moss_tts_context* ctx) {
    // KV is position-addressed by n_past in every graph; a logical reset just
    // means the caller restarts at n_past = 0. Nothing to zero.
    (void)ctx;
}

// ===========================================================================
// Public: backbone step
// ===========================================================================

extern "C" float* moss_tts_run_llm_kv(moss_tts_context* ctx, const float* inputs_embeds, int n_tokens, int n_past,
                                      int* out_vocab_size, float** out_hidden) {
    if (!ctx || !inputs_embeds || n_tokens <= 0)
        return nullptr;
    if (!ctx->kv_k) {
        fprintf(stderr, "moss_tts: kv cache not initialized — call moss_tts_kv_init first\n");
        return nullptr;
    }
    if (n_past + n_tokens > ctx->kv_max_ctx) {
        fprintf(stderr, "moss_tts: kv overflow (n_past=%d + n_tokens=%d > max=%d)\n", n_past, n_tokens,
                ctx->kv_max_ctx);
        return nullptr;
    }
    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.llm_hidden;
    const int vocab = (int)(hp.llm_lm_head_dim ? hp.llm_lm_head_dim : hp.llm_vocab_size);
    const int Lk = n_past + n_tokens;

    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        positions[i] = n_past + i;

    std::vector<ggml_fp16_t> mask;
    if (n_tokens > 1) {
        const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neg_h = ggml_fp32_to_fp16(-INFINITY);
        mask.assign((size_t)Lk * n_tokens, zero_h);
        for (int q = 0; q < n_tokens; q++)
            for (int k = n_past + q + 1; k < Lk; k++)
                mask[(size_t)q * Lk + k] = neg_h;
    }

    ggml_cgraph* gf = moss_tts_build_graph_llm_kv(ctx, n_past, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "moss_tts: failed to alloc llm_kv graph\n");
        return nullptr;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inputs_embeds"), inputs_embeds, 0,
                            (size_t)d * n_tokens * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), positions.data(), 0,
                            positions.size() * sizeof(int32_t));
    if (n_tokens > 1)
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0,
                                mask.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "moss_tts: llm_kv graph compute failed\n");
        return nullptr;
    }

    ggml_tensor* logits_t = ggml_graph_get_tensor(gf, "logits");
    if (!logits_t)
        return nullptr;
    float* logits = (float*)malloc((size_t)vocab * sizeof(float));
    ggml_backend_tensor_get(logits_t, logits, 0, (size_t)vocab * sizeof(float));

    if (out_hidden) {
        ggml_tensor* h_t = ggml_graph_get_tensor(gf, "hidden_last");
        *out_hidden = (float*)malloc((size_t)d * sizeof(float));
        ggml_backend_tensor_get(h_t, *out_hidden, 0, (size_t)d * sizeof(float));
    }
    if (out_vocab_size)
        *out_vocab_size = vocab;
    return logits;
}

// ===========================================================================
// Public: aux graphs (summed input embedding + 32 audio heads)
// ===========================================================================

extern "C" float* moss_tts_compute_input_embeddings(moss_tts_context* ctx, const int32_t* grid, int n_pos) {
    if (!ctx || !grid || n_pos <= 0)
        return nullptr;
    const auto& m = ctx->model;
    const int n_vq = (int)m.hparams.n_vq;
    const int hidden = (int)m.hparams.llm_hidden;
    const int stride = 1 + n_vq;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4 * n_vq + 64, false);

    ggml_tensor* text_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(text_ids, "text_ids");
    ggml_set_input(text_ids);
    std::vector<ggml_tensor*> audio_ids(n_vq);
    for (int i = 0; i < n_vq; i++) {
        audio_ids[i] = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
        char nm[32];
        snprintf(nm, sizeof(nm), "audio_ids_%d", i);
        ggml_set_name(audio_ids[i], nm);
        ggml_set_input(audio_ids[i]);
    }

    ggml_tensor* out = ggml_get_rows(ctx0, m.token_embd_w, text_ids); // (hidden, n_pos)
    out = ggml_cast(ctx0, out, GGML_TYPE_F32);
    for (int i = 0; i < n_vq; i++) {
        ggml_tensor* e = ggml_get_rows(ctx0, m.audio_embed[i], audio_ids[i]);
        e = ggml_cast(ctx0, e, GGML_TYPE_F32);
        out = ggml_add(ctx0, out, e);
    }
    ggml_set_name(out, "input_embeds");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "moss_tts: alloc input-embed graph failed\n");
        return nullptr;
    }
    std::vector<int32_t> col(n_pos);
    for (int r = 0; r < n_pos; r++)
        col[r] = grid[r * stride + 0];
    ggml_backend_tensor_set(text_ids, col.data(), 0, (size_t)n_pos * sizeof(int32_t));
    for (int i = 0; i < n_vq; i++) {
        for (int r = 0; r < n_pos; r++)
            col[r] = grid[r * stride + 1 + i];
        ggml_backend_tensor_set(audio_ids[i], col.data(), 0, (size_t)n_pos * sizeof(int32_t));
    }
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "moss_tts: input-embed compute failed\n");
        return nullptr;
    }
    ggml_tensor* out_t = ggml_graph_get_tensor(gf, "input_embeds");
    float* result = (float*)malloc((size_t)n_pos * hidden * sizeof(float));
    ggml_backend_tensor_get(out_t, result, 0, (size_t)n_pos * hidden * sizeof(float));
    return result;
}

extern "C" float* moss_tts_compute_audio_logits(moss_tts_context* ctx, const float* hidden) {
    if (!ctx || !hidden)
        return nullptr;
    const auto& m = ctx->model;
    const int n_vq = (int)m.hparams.n_vq;
    const int Vfull = (int)m.hparams.audio_vocab_size + 1;
    const int hsz = (int)m.hparams.llm_hidden;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4 * n_vq + 64, false);

    ggml_tensor* h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hsz, 1);
    ggml_set_name(h, "hidden");
    ggml_set_input(h);

    ggml_tensor* stacked = nullptr;
    for (int i = 0; i < n_vq; i++) {
        ggml_tensor* y = ggml_mul_mat(ctx0, m.audio_head[i], h); // (Vfull, 1)
        stacked = stacked ? ggml_concat(ctx0, stacked, y, 1) : y;
    }
    ggml_set_name(stacked, "audio_logits");
    ggml_set_output(stacked);
    ggml_build_forward_expand(gf, stacked);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "moss_tts: alloc audio-logits graph failed\n");
        return nullptr;
    }
    ggml_backend_tensor_set(h, hidden, 0, (size_t)hsz * sizeof(float));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "moss_tts: audio-logits compute failed\n");
        return nullptr;
    }
    ggml_tensor* out_t = ggml_graph_get_tensor(gf, "audio_logits");
    float* result = (float*)malloc((size_t)n_vq * Vfull * sizeof(float));
    ggml_backend_tensor_get(out_t, result, 0, (size_t)n_vq * Vfull * sizeof(float));
    return result;
}

// ===========================================================================
// Sampling helpers (CPU) — ported from openmoss delay.cpp
// ===========================================================================

namespace {

struct Rng {
    std::mt19937_64 g;
    explicit Rng(uint64_t seed) {
        if (seed == 0) {
            std::random_device rd;
            seed = ((uint64_t)rd() << 32) | rd();
        }
        g.seed(seed);
    }
    float uniform01() { return std::uniform_real_distribution<float>(0.f, 1.f)(g); }
};

void apply_repetition_penalty(float* logits, int vocab, const std::vector<int32_t>& history, float penalty) {
    if (penalty == 1.0f)
        return;
    std::vector<bool> seen((size_t)vocab, false);
    for (int32_t id : history) {
        if (id < 0 || id >= vocab || seen[(size_t)id])
            continue;
        seen[(size_t)id] = true;
        if (logits[id] > 0)
            logits[id] /= penalty;
        else
            logits[id] *= penalty;
    }
}

void softmax_inplace(float* x, int n) {
    float mx = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; i++)
        if (x[i] > mx)
            mx = x[i];
    if (!std::isfinite(mx)) {
        std::fill(x, x + n, 0.f);
        return;
    }
    float sum = 0.f;
    for (int i = 0; i < n; i++) {
        x[i] = std::exp(x[i] - mx);
        sum += x[i];
    }
    if (sum <= 0.f) {
        std::fill(x, x + n, 0.f);
        return;
    }
    const float inv = 1.f / sum;
    for (int i = 0; i < n; i++)
        x[i] *= inv;
}

int32_t sample_one(float* logits, int vocab, float temperature, float top_p, int top_k, bool do_sample, Rng& rng) {
    if (!do_sample) {
        int best = 0;
        for (int i = 1; i < vocab; i++)
            if (logits[i] > logits[best])
                best = i;
        return best;
    }
    if (temperature > 0.f && temperature != 1.f) {
        const float inv = 1.f / temperature;
        for (int i = 0; i < vocab; i++)
            logits[i] *= inv;
    }
    if (top_k > 0 && top_k < vocab) {
        std::vector<std::pair<float, int32_t>> v;
        v.reserve(vocab);
        for (int i = 0; i < vocab; i++)
            if (std::isfinite(logits[i]))
                v.emplace_back(logits[i], i);
        if ((int)v.size() > top_k) {
            std::nth_element(v.begin(), v.begin() + top_k, v.end(), [](auto& a, auto& b) { return a.first > b.first; });
            v.resize(top_k);
        }
        std::vector<bool> keep(vocab, false);
        for (auto& p : v)
            keep[p.second] = true;
        for (int i = 0; i < vocab; i++)
            if (!keep[i])
                logits[i] = -std::numeric_limits<float>::infinity();
    }
    softmax_inplace(logits, vocab);
    if (top_p > 0.f && top_p < 1.f) {
        std::vector<int32_t> order(vocab);
        for (int i = 0; i < vocab; i++)
            order[i] = i;
        std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
        float acc = 0.f;
        size_t cut = order.size();
        for (size_t i = 0; i < order.size(); i++) {
            acc += logits[order[i]];
            if (acc >= top_p) {
                cut = i + 1;
                break;
            }
        }
        float sum = 0.f;
        for (size_t i = 0; i < cut; i++)
            sum += logits[order[i]];
        for (size_t i = cut; i < order.size(); i++)
            logits[order[i]] = 0.f;
        if (sum > 0.f) {
            const float inv = 1.f / sum;
            for (size_t i = 0; i < cut; i++)
                logits[order[i]] *= inv;
        }
    }
    const float u = rng.uniform01();
    float acc = 0.f;
    for (int i = 0; i < vocab; i++) {
        acc += logits[i];
        if (acc >= u)
            return i;
    }
    for (int i = vocab - 1; i >= 0; i--)
        if (logits[i] > 0.f)
            return i;
    return 0;
}

// Delay-pattern state machine — port of openmoss delay.cpp / MossTTSDelay.
struct DelayState {
    const moss_tts_hparams& hp;
    int step_idx = 0;
    bool is_audio = false;
    bool is_stopping = false;
    int64_t audio_length = 0;
    int64_t delayed_length = -1;               // -1 = INT64_MAX sentinel
    std::vector<std::vector<int32_t>> history; // (T, 1+n_vq)
    Rng rng;

    DelayState(const moss_tts_hparams& hp_, const std::vector<std::vector<int32_t>>& prompt, uint64_t seed)
        : hp(hp_), history(prompt), rng(seed) {
        if (!prompt.empty()) {
            const int32_t last_text = prompt.back().front();
            if (last_text == (int32_t)hp.tok_audio_start || last_text == (int32_t)hp.tok_audio_gen_slot) {
                is_audio = true;
                for (int64_t i = (int64_t)prompt.size() - 1; i >= 0; i--) {
                    if (prompt[i].front() == (int32_t)hp.tok_audio_start) {
                        audio_length = (int64_t)prompt.size() - i;
                        break;
                    }
                }
            }
        }
    }

    // Returns the next (1+n_vq) row; sets stop=true when the model emits im_end.
    std::vector<int32_t> step(const float* text_logits, const float* audio_logits, const moss_tts_synth_params& sc,
                              bool& stop) {
        const int n_vq = (int)hp.n_vq;
        const int aud_v = (int)hp.audio_vocab_size;
        const int aud_v_full = aud_v + 1;
        const int pad_code = (int)hp.audio_pad_code;
        stop = false;

        std::vector<int32_t> ids((size_t)(1 + n_vq), pad_code);
        int32_t next_text = (int32_t)hp.tok_pad;

        if (is_stopping) {
            ids[0] = (int32_t)hp.tok_pad;
            stop = true;
            return ids;
        }

        if (delayed_length >= 0 && delayed_length < n_vq) {
            next_text = (int32_t)hp.tok_audio_delay_slot;
        } else if (delayed_length == n_vq) {
            next_text = (int32_t)hp.tok_audio_end;
            is_audio = false;
        } else if (is_audio && delayed_length < 0 && sc.max_audio_frames > 0 &&
                   audio_length >= (int64_t)sc.max_audio_frames) {
            next_text = (int32_t)hp.tok_audio_delay_slot;
        } else {
            const int text_vocab = (int)hp.llm_vocab_size;
            std::vector<float> tmp(text_logits, text_logits + text_vocab);
            auto mask = [&](int id) {
                if (id >= 0 && id < text_vocab)
                    tmp[id] = -std::numeric_limits<float>::infinity();
            };
            if (!is_audio) {
                mask((int)hp.tok_pad);
                mask((int)hp.tok_audio_gen_slot);
                mask((int)hp.tok_audio_delay_slot);
                mask((int)hp.tok_audio_end);
            } else {
                for (int i = 0; i < text_vocab; i++)
                    if (i != (int)hp.tok_audio_gen_slot && i != (int)hp.tok_audio_delay_slot)
                        tmp[i] = -std::numeric_limits<float>::infinity();
            }
            if (step_idx == 0)
                mask((int)hp.tok_audio_delay_slot);
            if (step_idx <= n_vq)
                mask((int)hp.tok_im_end);
            if (is_audio && delayed_length < 0 && audio_length < (int64_t)sc.min_audio_frames)
                mask((int)hp.tok_audio_delay_slot);

            next_text = sample_one(tmp.data(), text_vocab, sc.text_temperature, sc.text_top_p, sc.text_top_k,
                                   sc.text_temperature > 0.f, rng);
            if (next_text == (int32_t)hp.tok_audio_start)
                is_audio = true;
            if (next_text == (int32_t)hp.tok_im_end)
                is_stopping = true;
        }
        ids[0] = next_text;

        const bool delayed_in_sentinel = (delayed_length < 0);
        std::vector<float> abuf((size_t)aud_v_full);
        std::vector<int32_t> hist_col(history.size());
        for (int i = 0; i < n_vq; i++) {
            const bool pre = audio_length > (int64_t)i;
            const bool post = delayed_in_sentinel ? true : (int64_t)i >= delayed_length;
            if (!(pre && post)) {
                ids[(size_t)(1 + i)] = pad_code;
                continue;
            }
            std::memcpy(abuf.data(), audio_logits + (size_t)i * aud_v_full, (size_t)aud_v_full * sizeof(float));
            abuf[(size_t)pad_code] = -std::numeric_limits<float>::infinity();
            if (sc.audio_repetition_penalty != 1.f) {
                for (size_t h = 0; h < history.size(); h++)
                    hist_col[h] = history[h][(size_t)(1 + i)];
                apply_repetition_penalty(abuf.data(), aud_v_full, hist_col, sc.audio_repetition_penalty);
            }
            ids[(size_t)(1 + i)] = sample_one(abuf.data(), aud_v_full, sc.audio_temperature, sc.audio_top_p,
                                              sc.audio_top_k, sc.audio_temperature > 0.f, rng);
        }

        if (next_text == (int32_t)hp.tok_audio_start || next_text == (int32_t)hp.tok_audio_gen_slot ||
            next_text == (int32_t)hp.tok_audio_delay_slot)
            audio_length++;
        if (next_text == (int32_t)hp.tok_audio_end)
            audio_length = 0;
        if (delayed_length < 0 && next_text == (int32_t)hp.tok_audio_delay_slot)
            delayed_length = 0;
        else if (delayed_length >= 0) {
            delayed_length++;
            if (delayed_length > n_vq)
                delayed_length = -1;
        }
        history.push_back(ids);
        step_idx++;
        return ids;
    }

    // (n_vq, T_audio) row-major, delay-unshifted. Returns empty if too short.
    std::vector<int32_t> extract_audio_codes(int& n_vq_out, int& t_audio) const {
        const int n_vq = (int)hp.n_vq;
        n_vq_out = n_vq;
        t_audio = 0;
        int64_t start = -1;
        for (int64_t i = (int64_t)history.size() - 1; i >= 0; i--) {
            if (history[(size_t)i].front() == (int32_t)hp.tok_audio_start) {
                start = i + 1;
                break;
            }
        }
        if (start < 0)
            return {};
        int64_t end = (int64_t)history.size();
        for (int64_t i = start; i < end; i++) {
            if (history[(size_t)i].front() == (int32_t)hp.tok_audio_end) {
                end = i;
                break;
            }
        }
        const int64_t T = end - start;
        if (T <= (int64_t)n_vq)
            return {};
        const int64_t T_audio = T - (int64_t)n_vq;
        t_audio = (int)T_audio;
        std::vector<int32_t> out((size_t)n_vq * (size_t)T_audio);
        for (int cb = 0; cb < n_vq; cb++)
            for (int64_t t = 0; t < T_audio; t++)
                out[(size_t)cb * (size_t)T_audio + (size_t)t] = history[(size_t)(start + t + cb)][(size_t)(1 + cb)];
        return out;
    }
};

} // namespace

// ===========================================================================
// Prompt builder + code generation (AR loop)
// ===========================================================================

static std::string mt_tok_str(const moss_tts_context* ctx, uint32_t id) {
    if (id < ctx->vocab.id_to_token.size())
        return ctx->vocab.id_to_token[id];
    return "";
}

// Build the assistant prompt string (special tokens in their textual form) that
// tokenizes back to the exact id sequence, mirroring openmoss pipeline.cpp.
static std::string mt_build_prompt_text(const moss_tts_context* ctx, const char* text,
                                        const moss_tts_synth_params& sp) {
    const auto& hp = ctx->model.hparams;
    const std::string im_start = mt_tok_str(ctx, hp.tok_im_start);
    const std::string im_end = mt_tok_str(ctx, hp.tok_im_end);
    const std::string audio_start = mt_tok_str(ctx, hp.tok_audio_start);
    const std::string instruction = sp.instruction ? std::string(sp.instruction) : "None";
    const std::string language = sp.language ? std::string(sp.language) : "None";

    std::string body;
    body += "<user_inst>\n";
    body += "- Reference(s):\nNone\n";
    body += "- Instruction:\n" + instruction + "\n";
    body += "- Tokens:\nNone\n";
    body += "- Quality:\nNone\n";
    body += "- Sound Event:\nNone\n";
    body += "- Ambient Sound:\nNone\n";
    body += "- Language:\n" + language + "\n";
    body += "- Text:\n" + std::string(text ? text : "") + "\n";
    body += "</user_inst>";

    std::string out;
    out += im_start + "user\n" + body + im_end + "\n" + im_start + "assistant\n" + audio_start;
    return out;
}

// Shared AR loop → DelayState with the generated history. Returns false on error.
static bool mt_generate(moss_tts_context* ctx, const char* text, const moss_tts_synth_params& sp,
                        std::unique_ptr<DelayState>& state_out) {
    const auto& hp = ctx->model.hparams;
    const int n_vq = (int)hp.n_vq;
    const int max_new = sp.max_new_tokens > 0 ? sp.max_new_tokens : 4096;

    // Tokenize the prompt to the (col-0) id sequence.
    std::string prompt = mt_build_prompt_text(ctx, text, sp);
    int n_ids = 0;
    int32_t* ids = moss_tts_tokenize(ctx, prompt.c_str(), &n_ids);
    if (!ids || n_ids <= 0) {
        free(ids);
        return false;
    }
    const int prompt_len = n_ids;
    const int stride = 1 + n_vq;

    // Prompt grid: col 0 = text ids, audio cols = pad. (No reference audio.)
    std::vector<int32_t> grid((size_t)prompt_len * stride, (int32_t)hp.audio_pad_code);
    for (int r = 0; r < prompt_len; r++)
        grid[(size_t)r * stride + 0] = ids[r];
    free(ids);

    if (!moss_tts_kv_init(ctx, prompt_len + max_new + 8))
        return false;
    moss_tts_kv_reset(ctx);

    // Seed DelayState history from the prompt grid.
    std::vector<std::vector<int32_t>> history;
    history.reserve((size_t)prompt_len + (size_t)max_new);
    for (int r = 0; r < prompt_len; r++)
        history.emplace_back(grid.begin() + (size_t)r * stride, grid.begin() + (size_t)(r + 1) * stride);
    auto state = std::make_unique<DelayState>(hp, history, sp.seed ? sp.seed : ctx->seed);

    // Prefill.
    float* prefill_emb = moss_tts_compute_input_embeddings(ctx, grid.data(), prompt_len);
    if (!prefill_emb)
        return false;
    int vocab = 0;
    float* hidden_vec = nullptr;
    float* text_logits = moss_tts_run_llm_kv(ctx, prefill_emb, prompt_len, 0, &vocab, &hidden_vec);
    free(prefill_emb);
    if (!text_logits || !hidden_vec) {
        free(text_logits);
        free(hidden_vec);
        return false;
    }

    int pos = prompt_len;
    for (int step = 0; step < max_new; step++) {
        float* audio_logits = moss_tts_compute_audio_logits(ctx, hidden_vec);
        if (!audio_logits) {
            free(text_logits);
            free(hidden_vec);
            return false;
        }
        bool stop = false;
        std::vector<int32_t> row = state->step(text_logits, audio_logits, sp, stop);
        free(text_logits);
        free(hidden_vec);
        free(audio_logits);
        text_logits = nullptr;
        hidden_vec = nullptr;
        if (stop)
            break;

        // Embed the (1, 1+n_vq) row and decode one step.
        float* next_emb = moss_tts_compute_input_embeddings(ctx, row.data(), 1);
        if (!next_emb)
            return false;
        text_logits = moss_tts_run_llm_kv(ctx, next_emb, 1, pos, &vocab, &hidden_vec);
        free(next_emb);
        if (!text_logits || !hidden_vec) {
            free(text_logits);
            free(hidden_vec);
            return false;
        }
        pos++;
    }
    free(text_logits);
    free(hidden_vec);
    state_out = std::move(state);
    return true;
}

extern "C" int32_t* moss_tts_generate_codes(moss_tts_context* ctx, const char* text, const moss_tts_synth_params* sp_in,
                                            int* out_n_vq, int* out_t_audio) {
    if (out_n_vq)
        *out_n_vq = 0;
    if (out_t_audio)
        *out_t_audio = 0;
    if (!ctx || !text)
        return nullptr;
    moss_tts_synth_params sp = sp_in ? *sp_in : moss_tts_synth_default_params();
    std::unique_ptr<DelayState> state;
    if (!mt_generate(ctx, text, sp, state) || !state)
        return nullptr;
    int nvq = 0, t_audio = 0;
    std::vector<int32_t> codes = state->extract_audio_codes(nvq, t_audio);
    if (t_audio <= 0 || codes.empty())
        return nullptr;
    if (out_n_vq)
        *out_n_vq = nvq;
    if (out_t_audio)
        *out_t_audio = t_audio;
    int32_t* out = (int32_t*)malloc(codes.size() * sizeof(int32_t));
    std::memcpy(out, codes.data(), codes.size() * sizeof(int32_t));
    return out;
}

extern "C" float* moss_tts_synthesize(moss_tts_context* ctx, const char* text, const moss_tts_synth_params* sp,
                                      int* out_n_samples) {
    if (out_n_samples)
        *out_n_samples = 0;
    if (!ctx || !text)
        return nullptr;
    if (!ctx->codec_loaded) {
        fprintf(stderr, "moss_tts: codec not loaded — call moss_tts_set_codec_path() with the companion "
                        "GGUF (codec decode lands in Phase 4)\n");
        return nullptr;
    }
    // Phase 4: generate codes, then codec_decode -> waveform.
    (void)sp;
    return nullptr;
}

// ===========================================================================
// Tokenizer (Qwen3 gpt2-style BPE, special-token aware) — clone of qwen3_asr.
// ===========================================================================

extern "C" int32_t* moss_tts_tokenize(moss_tts_context* ctx, const char* text, int* out_n_tokens) {
    if (!ctx || !text) {
        if (out_n_tokens)
            *out_n_tokens = 0;
        return nullptr;
    }
    const auto& v = ctx->vocab;
    std::vector<int32_t> result;

    auto match_special = [&](const std::string& str, size_t pos, int32_t& id) -> size_t {
        if (str[pos] != '<')
            return 0;
        const bool pipe_form = pos + 1 < str.size() && str[pos + 1] == '|';
        const size_t close = pipe_form ? str.find("|>", pos + 2) : str.find('>', pos + 1);
        if (close == std::string::npos)
            return 0;
        const size_t len = close + (pipe_form ? 2 : 1) - pos;
        if (!pipe_form && len > 24)
            return 0;
        auto it = v.token_to_id.find(str.substr(pos, len));
        if (it == v.token_to_id.end())
            return 0;
        id = it->second;
        return len;
    };

    const std::string s = text;
    size_t i = 0;
    while (i < s.size()) {
        if (s[i] == '<') {
            int32_t sp_id = 0;
            const size_t sp_len = match_special(s, i, sp_id);
            if (sp_len > 0) {
                result.push_back(sp_id);
                i += sp_len;
                continue;
            }
        }
        size_t j = i;
        if (s[j] == '<')
            j++;
        while (j < s.size()) {
            if (s[j] == '<') {
                int32_t sp_id = 0;
                if (match_special(s, j, sp_id) > 0)
                    break;
            }
            j++;
        }
        std::string chunk = s.substr(i, j - i);
        i = j;
        if (chunk.empty())
            continue;
        size_t k = 0;
        while (k < chunk.size()) {
            size_t start = k;
            if (chunk[k] == ' ' || chunk[k] == '\t' || chunk[k] == '\n')
                k++;
            while (k < chunk.size() && chunk[k] != ' ' && chunk[k] != '\t' && chunk[k] != '\n')
                k++;
            if (k == start)
                k++;
            std::string pre(chunk, start, k - start);
            std::string encoded = core_bpe::bytes_to_unicode(pre.data(), pre.size());
            core_bpe::bpe_one(v.token_to_id, v.merge_rank, encoded, result);
        }
    }

    if (out_n_tokens)
        *out_n_tokens = (int)result.size();
    int32_t* out = (int32_t*)malloc(result.size() * sizeof(int32_t));
    if (!out) {
        if (out_n_tokens)
            *out_n_tokens = 0;
        return nullptr;
    }
    std::memcpy(out, result.data(), result.size() * sizeof(int32_t));
    return out;
}

extern "C" const char* moss_tts_token_text(moss_tts_context* ctx, int id) {
    if (!ctx || id < 0 || id >= (int)ctx->vocab.id_to_token.size())
        return "";
    return ctx->vocab.id_to_token[id].c_str();
}

// ===========================================================================
// Init / free / params / accessors
// ===========================================================================

extern "C" moss_tts_context_params moss_tts_context_default_params(void) {
    moss_tts_context_params p = {};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    p.flash_attn = true;
    return p;
}

extern "C" moss_tts_synth_params moss_tts_synth_default_params(void) {
    moss_tts_synth_params p = {};
    p.max_new_tokens = 4096;
    p.text_temperature = 1.5f;
    p.text_top_p = 1.0f;
    p.text_top_k = 50;
    p.audio_temperature = 1.7f;
    p.audio_top_p = 0.8f;
    p.audio_top_k = 25;
    p.audio_repetition_penalty = 1.0f;
    p.min_audio_frames = 0;
    p.max_audio_frames = 0;
    p.seed = 0;
    p.language = nullptr;
    p.instruction = nullptr;
    return p;
}

extern "C" moss_tts_context* moss_tts_init_from_file(const char* path, moss_tts_context_params params) {
    moss_tts_context* ctx = new moss_tts_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    ctx->backend = params.use_gpu ? crispasr_init_gpu_backend() : ggml_backend_cpu_init();
    if (!ctx->backend)
        ctx->backend = ggml_backend_cpu_init();
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (ctx->backend_cpu)
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    if (ggml_backend_is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);

    if (!moss_tts_load_model(ctx->model, ctx->vocab, path, ctx->backend)) {
        delete ctx;
        return nullptr;
    }

    int n_be = 0;
    ggml_backend_t backends[2];
    backends[n_be++] = ctx->backend;
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        backends[n_be++] = ctx->backend_cpu;
    ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    if (params.verbosity >= 1)
        fprintf(stderr, "moss_tts: loaded %s (llm %u layers, vocab %u)\n", path ? path : "(null)",
                ctx->model.hparams.llm_layers, (uint32_t)ctx->vocab.id_to_token.size());
    return ctx;
}

extern "C" bool moss_tts_set_codec_path(moss_tts_context* ctx, const char* path_codec) {
    if (!ctx || !path_codec)
        return false;
    ctx->codec_path = path_codec;
    // Phase 4: open the codec GGUF, bind moss.codec.* tensors, set codec_loaded.
    ctx->codec_loaded = false;
    return false;
}

extern "C" void moss_tts_free(moss_tts_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->model.buf)
        ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

extern "C" void moss_tts_set_seed(moss_tts_context* ctx, uint32_t seed) {
    if (ctx)
        ctx->seed = seed;
}

extern "C" int moss_tts_n_vq(const moss_tts_context* ctx) {
    return ctx ? (int)ctx->model.hparams.n_vq : 0;
}
extern "C" int moss_tts_hidden_size(const moss_tts_context* ctx) {
    return ctx ? (int)ctx->model.hparams.llm_hidden : 0;
}
extern "C" int moss_tts_audio_vocab_size(const moss_tts_context* ctx) {
    return ctx ? (int)ctx->model.hparams.audio_vocab_size : 0;
}
extern "C" int moss_tts_sampling_rate(const moss_tts_context* ctx) {
    return ctx ? (int)ctx->model.hparams.sampling_rate : 0;
}
extern "C" bool moss_tts_codec_loaded(const moss_tts_context* ctx) {
    return ctx ? ctx->codec_loaded : false;
}
