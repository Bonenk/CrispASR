// voxtral_tts.cpp — Voxtral-4B-TTS runtime (mistralai/Voxtral-4B-TTS-2603).
//
// Three-component TTS pipeline:
//   1. Ministral-3B AR backbone (26L GQA) — predicts semantic tokens per frame
//   2. Acoustic FM transformer (3L bidirectional, 8-step Euler ODE) — predicts
//      36 acoustic FSQ values per frame
//   3. Voxtral codec decoder (4 conv+transformer blocks) — decodes to 24 kHz PCM
//
// Reuses:
//   - core_attn::llama_self_attn_kv() for the LLM backbone (GQA + RoPE + KV cache)
//   - core_ffn::swiglu() for SwiGLU FFN in both LLM and FM
//   - Tekken BPE tokenizer pattern from voxtral4b.cpp

#include "voxtral_tts.h"

#include "core/attention.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Hyperparameters
// ---------------------------------------------------------------------------

struct voxtral_tts_hparams {
    // LLM backbone (Ministral 3B)
    int llm_n_layers = 26;
    int llm_dim = 3072;
    int llm_n_heads = 32;
    int llm_n_kv = 8;
    int llm_head_dim = 128;
    int llm_ff_dim = 9216;
    float llm_rope_theta = 1000000.0f;
    float llm_norm_eps = 1e-5f;
    int llm_vocab_size = 131072;
    bool tied_embeddings = true;

    // FM transformer (acoustic flow-matching)
    int fm_n_layers = 3;
    int fm_dim = 3072;
    int fm_n_heads = 32;
    int fm_n_kv = 8;
    int fm_head_dim = 128;
    int fm_ff_dim = 9216;
    float fm_rope_theta = 10000.0f;
    float fm_sigma = 1e-5f;
    float fm_sigma_max = 1.0f;

    // Audio encoding
    int semantic_cb_size = 8192;
    int acoustic_cb_size = 21; // FSQ levels
    int n_acoustic_cb = 36;
    int n_codebooks = 37; // 1 semantic + 36 acoustic
    int sample_rate = 24000;
    float frame_rate = 12.5f;

    // Special tokens
    int audio_token_id = 24;
    int begin_audio_token_id = 25;
    int cond_dropped_token_id = 42;
    int bos_token_id = 1;

    // Codec decoder
    int codec_dim = 1024;
    int codec_hidden_dim = 4096;
    int codec_n_heads = 8;
    int codec_n_kv = 8;
    int codec_head_dim = 128;
    int codec_semantic_dim = 256;
    int codec_acoustic_dim = 36;
    int codec_patch_size = 240;
    int codec_patch_kernel = 7;
    int codec_attn_window = 16;
    float codec_norm_eps = 0.01f;
    float codec_qk_norm_eps = 1e-6f;
    bool codec_qk_norm = true;
    bool codec_layer_scale = true;
    std::vector<int> codec_conv_strides; // {1, 2, 2, 2}
    std::vector<int> codec_conv_kernels; // {3, 4, 4, 4}
    std::vector<int> codec_tfm_lengths;  // {2, 2, 2, 2}
};

// ---------------------------------------------------------------------------
// Tekken BPE tokenizer (shared pattern with voxtral4b.cpp)
// ---------------------------------------------------------------------------

struct voxtral_tts_vocab {
    std::string pre_pattern;
    std::vector<std::string> specials;
    int n_specials = 0;
    int n_vocab = 0;

    // BPE merge table: pair → merged piece
    std::vector<std::pair<std::string, std::string>> merges;
    // Piece → token ID (specials first, then BPE vocab)
    std::map<std::string, int> piece_to_id;
    // Token ID → piece
    std::vector<std::string> id_to_piece;

    std::vector<uint8_t> tekken_vocab_blob;
};

// ---------------------------------------------------------------------------
// Model weights
// ---------------------------------------------------------------------------

struct voxtral_tts_llm_layer {
    ggml_tensor* attn_norm = nullptr;
    ggml_tensor* attn_q = nullptr;
    ggml_tensor* attn_k = nullptr;
    ggml_tensor* attn_v = nullptr;
    ggml_tensor* attn_o = nullptr;
    ggml_tensor* ffn_norm = nullptr;
    ggml_tensor* ffn_gate = nullptr;
    ggml_tensor* ffn_up = nullptr;
    ggml_tensor* ffn_down = nullptr;
};

struct voxtral_tts_fm_layer {
    ggml_tensor* attn_norm = nullptr;
    ggml_tensor* attn_q = nullptr;
    ggml_tensor* attn_k = nullptr;
    ggml_tensor* attn_v = nullptr;
    ggml_tensor* attn_o = nullptr;
    ggml_tensor* ffn_norm = nullptr;
    ggml_tensor* ffn_gate = nullptr;
    ggml_tensor* ffn_up = nullptr;
    ggml_tensor* ffn_down = nullptr;
};

struct voxtral_tts_codec_tfm_layer {
    ggml_tensor* attn_norm = nullptr;
    ggml_tensor* attn_q = nullptr;
    ggml_tensor* attn_k = nullptr;
    ggml_tensor* attn_v = nullptr;
    ggml_tensor* attn_o = nullptr;
    ggml_tensor* q_norm = nullptr;
    ggml_tensor* k_norm = nullptr;
    ggml_tensor* ffn_norm = nullptr;
    ggml_tensor* ffn_gate = nullptr;
    ggml_tensor* ffn_up = nullptr;
    ggml_tensor* ffn_down = nullptr;
    ggml_tensor* attn_scale = nullptr; // layer_scale
    ggml_tensor* ffn_scale = nullptr;
};

struct voxtral_tts_codec_conv {
    ggml_tensor* weight = nullptr;
    ggml_tensor* bias = nullptr;
};

struct voxtral_tts_model {
    // LLM backbone
    ggml_tensor* token_embd = nullptr; // (vocab_size, dim)
    ggml_tensor* audio_embd = nullptr; // (9088, dim) — combined audio codebook embeddings
    ggml_tensor* output_norm = nullptr;
    std::vector<voxtral_tts_llm_layer> llm_layers;

    // FM transformer
    ggml_tensor* fm_input_proj = nullptr;      // (dim, 36)
    ggml_tensor* fm_llm_proj = nullptr;        // (dim, dim)
    ggml_tensor* fm_time_proj = nullptr;       // (dim, dim)
    ggml_tensor* fm_semantic_output = nullptr; // (8320, dim)
    ggml_tensor* fm_acoustic_output = nullptr; // (36, dim)
    ggml_tensor* fm_norm = nullptr;
    std::vector<voxtral_tts_fm_layer> fm_layers;

    // Codec decoder
    std::vector<voxtral_tts_codec_conv> codec_convs;                        // 4 conv layers
    std::vector<std::vector<voxtral_tts_codec_tfm_layer>> codec_tfm_blocks; // 4 blocks × 2 layers
    voxtral_tts_codec_conv codec_output;
    voxtral_tts_codec_conv codec_patch_proj;
    ggml_tensor* codec_semantic_cb = nullptr; // (8192, 256)

    // Voice embeddings
    std::map<std::string, ggml_tensor*> voice_tensors;
};

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct voxtral_tts_context {
    voxtral_tts_hparams hp;
    voxtral_tts_model model;
    voxtral_tts_vocab vocab;
    voxtral_tts_context_params params;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_context* ctx_w = nullptr; // weight context

    // KV cache for LLM backbone
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr; // (head_dim, max_ctx, n_kv, n_layers)
    ggml_tensor* kv_v = nullptr;
    int kv_used = 0;

    // Voice name list (for list_voices API)
    std::vector<std::string> voice_names;
    std::vector<const char*> voice_name_ptrs;

    int verbosity = 1;
};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static bool env_bool(const char* name) {
    const char* v = std::getenv(name);
    return v && (*v == '1' || *v == 'y' || *v == 'Y');
}

static std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> out;
    size_t start = 0;
    while (start < s.size()) {
        size_t end = s.find(',', start);
        if (end == std::string::npos)
            end = s.size();
        out.push_back(std::stoi(s.substr(start, end - start)));
        start = end + 1;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Tekken BPE tokenizer (adapted from voxtral4b.cpp)
// ---------------------------------------------------------------------------

static void tekken_build_vocab(voxtral_tts_vocab& v) {
    // Decode the packed vocab blob into piece strings and build the merge table.
    const uint8_t* p = v.tekken_vocab_blob.data();
    const uint8_t* end = p + v.tekken_vocab_blob.size();
    v.id_to_piece.clear();
    v.piece_to_id.clear();

    // Specials come first (IDs 0 .. n_specials-1)
    for (int i = 0; i < v.n_specials && i < (int)v.specials.size(); i++) {
        v.id_to_piece.push_back(v.specials[i]);
        v.piece_to_id[v.specials[i]] = i;
    }
    // Pad if fewer specials stored
    while ((int)v.id_to_piece.size() < v.n_specials) {
        v.id_to_piece.push_back("");
    }

    // BPE vocab entries
    int bpe_id = v.n_specials;
    while (p + 2 <= end) {
        uint16_t len = *(const uint16_t*)p;
        p += 2;
        if (p + len > end)
            break;
        std::string piece((const char*)p, len);
        p += len;
        v.id_to_piece.push_back(piece);
        v.piece_to_id[piece] = bpe_id;
        bpe_id++;
    }
}

static void tekken_bpe_encode(const voxtral_tts_vocab& v, const uint8_t* data, size_t len, std::vector<int32_t>& out) {
    if (len == 0)
        return;

    // Initialize: each byte is its own piece
    std::vector<std::string> pieces;
    for (size_t i = 0; i < len; i++) {
        pieces.push_back(std::string(1, (char)data[i]));
    }

    // Greedy BPE merge: repeatedly find the highest-priority merge
    while (pieces.size() > 1) {
        int best_pos = -1;
        int best_id = INT32_MAX;
        for (int i = 0; i < (int)pieces.size() - 1; i++) {
            std::string merged = pieces[i] + pieces[i + 1];
            auto it = v.piece_to_id.find(merged);
            if (it != v.piece_to_id.end() && it->second < best_id) {
                best_id = it->second;
                best_pos = i;
            }
        }
        if (best_pos < 0)
            break;
        pieces[best_pos] = pieces[best_pos] + pieces[best_pos + 1];
        pieces.erase(pieces.begin() + best_pos + 1);
    }

    // Map pieces to IDs
    for (auto& pc : pieces) {
        auto it = v.piece_to_id.find(pc);
        if (it != v.piece_to_id.end()) {
            out.push_back(it->second);
        } else {
            // Fallback: encode each byte as its own token
            for (unsigned char c : pc) {
                std::string s(1, (char)c);
                auto it2 = v.piece_to_id.find(s);
                if (it2 != v.piece_to_id.end()) {
                    out.push_back(it2->second);
                }
            }
        }
    }
}

// Simple pre-tokenization: split on whitespace and punctuation boundaries
static std::vector<std::string> tekken_pre_tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    // For now, just pass the full text — Tekken's BPE handles splitting.
    // A proper implementation would use the regex pattern from tekken.json.
    if (!text.empty())
        tokens.push_back(text);
    return tokens;
}

static std::vector<int32_t> voxtral_tts_tokenize(voxtral_tts_context* ctx, const std::string& text) {
    std::vector<int32_t> ids;
    auto& v = ctx->vocab;

    // Check for special tokens in the text
    size_t pos = 0;
    while (pos < text.size()) {
        // Look for special token at current position
        bool found_special = false;
        for (int si = 0; si < (int)v.specials.size(); si++) {
            const auto& sp = v.specials[si];
            if (sp.empty())
                continue;
            if (text.compare(pos, sp.size(), sp) == 0) {
                ids.push_back(si);
                pos += sp.size();
                found_special = true;
                break;
            }
        }
        if (found_special)
            continue;

        // Find the next special token
        size_t next_special = text.size();
        for (const auto& sp : v.specials) {
            if (sp.empty())
                continue;
            size_t f = text.find(sp, pos);
            if (f != std::string::npos && f < next_special)
                next_special = f;
        }

        // BPE encode the text between specials
        if (pos < next_special) {
            auto pre_tokens = tekken_pre_tokenize(text.substr(pos, next_special - pos));
            for (auto& pt : pre_tokens) {
                tekken_bpe_encode(v, (const uint8_t*)pt.data(), pt.size(), ids);
            }
        }
        pos = next_special;
    }
    return ids;
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

extern "C" voxtral_tts_context_params voxtral_tts_context_default_params(void) {
    voxtral_tts_context_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.temperature = 0.0f;
    p.n_ode_steps = 0;  // default = 8
    p.cfg_alpha = 0.0f; // default = 1.2
    return p;
}

extern "C" voxtral_tts_context* voxtral_tts_init_from_file(const char* path_model, voxtral_tts_context_params params) {
    auto* ctx = new voxtral_tts_context();
    ctx->params = params;
    ctx->verbosity = params.verbosity;
    auto& hp = ctx->hp;

    // Open GGUF metadata
    gguf_context* gctx = core_gguf::open_metadata(path_model);
    if (!gctx) {
        fprintf(stderr, "voxtral_tts: failed to open '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }

    // Read hyperparameters
    hp.llm_n_layers = core_gguf::kv_u32(gctx, "voxtral_tts.llm.n_layers", hp.llm_n_layers);
    hp.llm_dim = core_gguf::kv_u32(gctx, "voxtral_tts.llm.dim", hp.llm_dim);
    hp.llm_n_heads = core_gguf::kv_u32(gctx, "voxtral_tts.llm.n_heads", hp.llm_n_heads);
    hp.llm_n_kv = core_gguf::kv_u32(gctx, "voxtral_tts.llm.n_kv_heads", hp.llm_n_kv);
    hp.llm_head_dim = core_gguf::kv_u32(gctx, "voxtral_tts.llm.head_dim", hp.llm_head_dim);
    hp.llm_ff_dim = core_gguf::kv_u32(gctx, "voxtral_tts.llm.hidden_dim", hp.llm_ff_dim);
    hp.llm_rope_theta = core_gguf::kv_f32(gctx, "voxtral_tts.llm.rope_theta", hp.llm_rope_theta);
    hp.llm_norm_eps = core_gguf::kv_f32(gctx, "voxtral_tts.llm.norm_eps", hp.llm_norm_eps);
    hp.llm_vocab_size = core_gguf::kv_u32(gctx, "voxtral_tts.llm.vocab_size", hp.llm_vocab_size);
    hp.tied_embeddings = core_gguf::kv_bool(gctx, "voxtral_tts.llm.tied_embeddings", hp.tied_embeddings);

    hp.fm_n_layers = core_gguf::kv_u32(gctx, "voxtral_tts.fm.n_layers", hp.fm_n_layers);
    hp.fm_dim = core_gguf::kv_u32(gctx, "voxtral_tts.fm.dim", hp.fm_dim);
    hp.fm_n_heads = core_gguf::kv_u32(gctx, "voxtral_tts.fm.n_heads", hp.fm_n_heads);
    hp.fm_n_kv = core_gguf::kv_u32(gctx, "voxtral_tts.fm.n_kv_heads", hp.fm_n_kv);
    hp.fm_head_dim = core_gguf::kv_u32(gctx, "voxtral_tts.fm.head_dim", hp.fm_head_dim);
    hp.fm_ff_dim = core_gguf::kv_u32(gctx, "voxtral_tts.fm.hidden_dim", hp.fm_ff_dim);
    hp.fm_rope_theta = core_gguf::kv_f32(gctx, "voxtral_tts.fm.rope_theta", hp.fm_rope_theta);
    hp.fm_sigma = core_gguf::kv_f32(gctx, "voxtral_tts.fm.sigma", hp.fm_sigma);
    hp.fm_sigma_max = core_gguf::kv_f32(gctx, "voxtral_tts.fm.sigma_max", hp.fm_sigma_max);

    hp.semantic_cb_size = core_gguf::kv_u32(gctx, "voxtral_tts.semantic_codebook_size", hp.semantic_cb_size);
    hp.acoustic_cb_size = core_gguf::kv_u32(gctx, "voxtral_tts.acoustic_codebook_size", hp.acoustic_cb_size);
    hp.n_acoustic_cb = core_gguf::kv_u32(gctx, "voxtral_tts.n_acoustic_codebook", hp.n_acoustic_cb);
    hp.n_codebooks = core_gguf::kv_u32(gctx, "voxtral_tts.n_codebooks", hp.n_codebooks);
    hp.sample_rate = core_gguf::kv_u32(gctx, "voxtral_tts.sample_rate", hp.sample_rate);
    hp.frame_rate = core_gguf::kv_f32(gctx, "voxtral_tts.frame_rate", hp.frame_rate);

    hp.audio_token_id = core_gguf::kv_u32(gctx, "voxtral_tts.audio_token_id", hp.audio_token_id);
    hp.begin_audio_token_id = core_gguf::kv_u32(gctx, "voxtral_tts.begin_audio_token_id", hp.begin_audio_token_id);
    hp.cond_dropped_token_id =
        core_gguf::kv_u32(gctx, "voxtral_tts.condition_dropped_token_id", hp.cond_dropped_token_id);
    hp.bos_token_id = core_gguf::kv_u32(gctx, "voxtral_tts.bos_token_id", hp.bos_token_id);

    hp.codec_dim = core_gguf::kv_u32(gctx, "voxtral_tts.codec.dim", hp.codec_dim);
    hp.codec_hidden_dim = core_gguf::kv_u32(gctx, "voxtral_tts.codec.hidden_dim", hp.codec_hidden_dim);
    hp.codec_n_heads = core_gguf::kv_u32(gctx, "voxtral_tts.codec.n_heads", hp.codec_n_heads);
    hp.codec_n_kv = core_gguf::kv_u32(gctx, "voxtral_tts.codec.n_kv_heads", hp.codec_n_kv);
    hp.codec_head_dim = core_gguf::kv_u32(gctx, "voxtral_tts.codec.head_dim", hp.codec_head_dim);
    hp.codec_semantic_dim = core_gguf::kv_u32(gctx, "voxtral_tts.codec.semantic_dim", hp.codec_semantic_dim);
    hp.codec_acoustic_dim = core_gguf::kv_u32(gctx, "voxtral_tts.codec.acoustic_dim", hp.codec_acoustic_dim);
    hp.codec_patch_size = core_gguf::kv_u32(gctx, "voxtral_tts.codec.patch_size", hp.codec_patch_size);
    hp.codec_patch_kernel = core_gguf::kv_u32(gctx, "voxtral_tts.codec.patch_proj_kernel", hp.codec_patch_kernel);
    hp.codec_attn_window = core_gguf::kv_u32(gctx, "voxtral_tts.codec.attn_window", hp.codec_attn_window);
    hp.codec_norm_eps = core_gguf::kv_f32(gctx, "voxtral_tts.codec.norm_eps", hp.codec_norm_eps);
    hp.codec_qk_norm_eps = core_gguf::kv_f32(gctx, "voxtral_tts.codec.qk_norm_eps", hp.codec_qk_norm_eps);
    hp.codec_qk_norm = core_gguf::kv_bool(gctx, "voxtral_tts.codec.qk_norm", hp.codec_qk_norm);
    hp.codec_layer_scale = core_gguf::kv_bool(gctx, "voxtral_tts.codec.layer_scale", hp.codec_layer_scale);

    std::string strides_str = core_gguf::kv_str(gctx, "voxtral_tts.codec.conv_strides", "1,2,2,2");
    std::string kernels_str = core_gguf::kv_str(gctx, "voxtral_tts.codec.conv_kernels", "3,4,4,4");
    std::string tfm_lens_str = core_gguf::kv_str(gctx, "voxtral_tts.codec.tfm_lengths", "2,2,2,2");
    hp.codec_conv_strides = parse_int_list(strides_str);
    hp.codec_conv_kernels = parse_int_list(kernels_str);
    hp.codec_tfm_lengths = parse_int_list(tfm_lens_str);

    // Voice names
    auto voice_names = core_gguf::kv_str_array(gctx, "voxtral_tts.voice_names");
    ctx->voice_names = voice_names;
    for (auto& vn : ctx->voice_names) {
        ctx->voice_name_ptrs.push_back(vn.c_str());
    }
    ctx->voice_name_ptrs.push_back(nullptr);

    // Tekken tokenizer
    ctx->vocab.pre_pattern = core_gguf::kv_str(gctx, "tokenizer.tekken.pattern", "");
    ctx->vocab.specials = core_gguf::kv_str_array(gctx, "tokenizer.tekken.specials");
    ctx->vocab.n_specials = core_gguf::kv_u32(gctx, "tokenizer.tekken.n_specials", 1000);
    ctx->vocab.n_vocab = core_gguf::kv_u32(gctx, "tokenizer.tekken.n_vocab", 150000);

    core_gguf::free_metadata(gctx);

    // Load weights
    ggml_backend_t be = ggml_backend_cpu_init();
    ctx->backend_cpu = be;
    if (params.use_gpu) {
        ggml_backend_t gpu = crispasr_init_gpu_backend();
        if (gpu) {
            ctx->backend = gpu;
            be = gpu;
        }
    }
    if (!ctx->backend)
        ctx->backend = be;

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, ctx->backend, "voxtral_tts", wl)) {
        fprintf(stderr, "voxtral_tts: failed to load weights from '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }
    ctx->ctx_w = wl.ctx;
    ctx->buf = wl.buf;

    auto get = [&](const std::string& name) -> ggml_tensor* {
        auto it = wl.tensors.find(name);
        if (it == wl.tensors.end())
            return nullptr;
        return it->second;
    };

    // Bind LLM weights
    ctx->model.token_embd = get("token_embd.weight");
    ctx->model.audio_embd = get("audio_embd.weight");
    ctx->model.output_norm = get("output_norm.weight");
    ctx->model.llm_layers.resize(hp.llm_n_layers);
    for (int i = 0; i < hp.llm_n_layers; i++) {
        auto& l = ctx->model.llm_layers[i];
        auto b = [&](const std::string& s) { return get("blk." + std::to_string(i) + "." + s); };
        l.attn_norm = b("attn_norm.weight");
        l.attn_q = b("attn_q.weight");
        l.attn_k = b("attn_k.weight");
        l.attn_v = b("attn_v.weight");
        l.attn_o = b("attn_output.weight");
        l.ffn_norm = b("ffn_norm.weight");
        l.ffn_gate = b("ffn_gate.weight");
        l.ffn_up = b("ffn_up.weight");
        l.ffn_down = b("ffn_down.weight");
    }

    // Bind FM weights
    ctx->model.fm_input_proj = get("fm.input_proj.weight");
    ctx->model.fm_llm_proj = get("fm.llm_proj.weight");
    ctx->model.fm_time_proj = get("fm.time_proj.weight");
    ctx->model.fm_semantic_output = get("fm.semantic_output.weight");
    ctx->model.fm_acoustic_output = get("fm.acoustic_output.weight");
    ctx->model.fm_norm = get("fm.norm.weight");
    ctx->model.fm_layers.resize(hp.fm_n_layers);
    for (int i = 0; i < hp.fm_n_layers; i++) {
        auto& l = ctx->model.fm_layers[i];
        auto b = [&](const std::string& s) { return get("fm.blk." + std::to_string(i) + "." + s); };
        l.attn_norm = b("attn_norm.weight");
        l.attn_q = b("attn_q.weight");
        l.attn_k = b("attn_k.weight");
        l.attn_v = b("attn_v.weight");
        l.attn_o = b("attn_output.weight");
        l.ffn_norm = b("ffn_norm.weight");
        l.ffn_gate = b("ffn_gate.weight");
        l.ffn_up = b("ffn_up.weight");
        l.ffn_down = b("ffn_down.weight");
    }

    // Bind codec weights
    int n_conv_blocks = (int)hp.codec_conv_strides.size();
    ctx->model.codec_convs.resize(n_conv_blocks);
    for (int i = 0; i < n_conv_blocks; i++) {
        auto pfx = "codec.dec.conv." + std::to_string(i);
        ctx->model.codec_convs[i].weight = get(pfx + ".weight");
        ctx->model.codec_convs[i].bias = get(pfx + ".bias");
    }

    int n_tfm_blocks = (int)hp.codec_tfm_lengths.size();
    ctx->model.codec_tfm_blocks.resize(n_tfm_blocks);
    for (int bi = 0; bi < n_tfm_blocks; bi++) {
        int n_layers = hp.codec_tfm_lengths[bi];
        ctx->model.codec_tfm_blocks[bi].resize(n_layers);
        for (int li = 0; li < n_layers; li++) {
            auto pfx = "codec.dec.tfm." + std::to_string(bi) + ".blk." + std::to_string(li);
            auto& l = ctx->model.codec_tfm_blocks[bi][li];
            l.attn_norm = get(pfx + ".attn_norm.weight");
            l.attn_q = get(pfx + ".attn_q.weight");
            l.attn_k = get(pfx + ".attn_k.weight");
            l.attn_v = get(pfx + ".attn_v.weight");
            l.attn_o = get(pfx + ".attn_o.weight");
            l.q_norm = get(pfx + ".q_norm.weight");
            l.k_norm = get(pfx + ".k_norm.weight");
            l.ffn_norm = get(pfx + ".ffn_norm.weight");
            l.ffn_gate = get(pfx + ".ffn_gate.weight");
            l.ffn_up = get(pfx + ".ffn_up.weight");
            l.ffn_down = get(pfx + ".ffn_down.weight");
            l.attn_scale = get(pfx + ".attn_scale");
            l.ffn_scale = get(pfx + ".ffn_scale");
        }
    }
    ctx->model.codec_output.weight = get("codec.output.weight");
    ctx->model.codec_output.bias = get("codec.output.bias");
    ctx->model.codec_patch_proj.weight = get("codec.patch_proj.weight");
    ctx->model.codec_patch_proj.bias = get("codec.patch_proj.bias");
    ctx->model.codec_semantic_cb = get("codec.semantic_cb.weight");

    // Bind voice embeddings
    for (auto& [name, tensor] : wl.tensors) {
        if (name.substr(0, 6) == "voice.") {
            ctx->model.voice_tensors[name.substr(6)] = tensor;
        }
    }

    // Load Tekken vocab blob
    {
        ggml_tensor* vt = get("tokenizer.tekken.vocab_tensor");
        if (vt) {
            int n = (int)ggml_nelements(vt);
            std::vector<float> tmp(n);
            ggml_backend_tensor_get(vt, tmp.data(), 0, n * sizeof(float));
            ctx->vocab.tekken_vocab_blob.resize(n);
            for (int i = 0; i < n; i++) {
                ctx->vocab.tekken_vocab_blob[i] = (uint8_t)(int)tmp[i];
            }
        }
    }
    tekken_build_vocab(ctx->vocab);

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "voxtral_tts: LLM %dL d=%d heads=%d/%d\n", hp.llm_n_layers, hp.llm_dim, hp.llm_n_heads,
                hp.llm_n_kv);
        fprintf(stderr, "voxtral_tts: FM  %dL d=%d heads=%d/%d rope_theta=%.0f\n", hp.fm_n_layers, hp.fm_dim,
                hp.fm_n_heads, hp.fm_n_kv, hp.fm_rope_theta);
        fprintf(stderr, "voxtral_tts: Codec d=%d semantic_cb=%d acoustic_fsq=%d×%d\n", hp.codec_dim,
                hp.semantic_cb_size, hp.acoustic_cb_size, hp.n_acoustic_cb);
        fprintf(stderr, "voxtral_tts: %d voices, %d tokens loaded\n", (int)ctx->model.voice_tensors.size(),
                (int)ctx->vocab.id_to_piece.size());
        fprintf(stderr, "voxtral_tts: loaded '%s'\n", path_model);
    }

    return ctx;
}

// ---------------------------------------------------------------------------
// Synthesize (stub — pipeline to be implemented)
// ---------------------------------------------------------------------------

extern "C" float* voxtral_tts_synthesize(voxtral_tts_context* ctx, const char* text, const char* voice,
                                         int* out_n_samples) {
    if (!ctx || !text || !out_n_samples)
        return nullptr;
    *out_n_samples = 0;

    // Step 1: Tokenize text
    std::vector<int32_t> text_ids = voxtral_tts_tokenize(ctx, text);
    if (ctx->verbosity >= 1) {
        fprintf(stderr, "voxtral_tts: tokenized %d tokens from \"%s\"\n", (int)text_ids.size(), text);
    }

    // Step 2: Get voice embedding
    std::string voice_name = voice ? voice : "neutral_female";
    auto vit = ctx->model.voice_tensors.find(voice_name);
    if (vit == ctx->model.voice_tensors.end() && !ctx->model.voice_tensors.empty()) {
        if (ctx->verbosity >= 1) {
            fprintf(stderr, "voxtral_tts: voice '%s' not found, using first available\n", voice_name.c_str());
        }
        vit = ctx->model.voice_tensors.begin();
        voice_name = vit->first;
    }

    // TODO: implement the full pipeline:
    //   3. Build prompt: [voice_tokens] <next> [text_tokens] <repeat>
    //   4. LLM AR decode (with KV cache) → semantic tokens per frame
    //   5. FM ODE per frame (8 Euler steps with CFG) → 36 acoustic FSQ values
    //   6. Codec decode: semantic VQ + acoustic FSQ → 292-d → conv+transformer → PCM
    fprintf(stderr, "voxtral_tts: synthesis pipeline not yet implemented\n");
    return nullptr;
}

// ---------------------------------------------------------------------------
// Cleanup + API
// ---------------------------------------------------------------------------

extern "C" void voxtral_tts_pcm_free(float* pcm) {
    free(pcm);
}

extern "C" int voxtral_tts_sample_rate(void) {
    return 24000;
}

extern "C" const char* const* voxtral_tts_list_voices(voxtral_tts_context* ctx, int* out_n_voices) {
    if (!ctx || !out_n_voices)
        return nullptr;
    *out_n_voices = (int)ctx->voice_names.size();
    return ctx->voice_name_ptrs.data();
}

extern "C" void voxtral_tts_free(voxtral_tts_context* ctx) {
    if (!ctx)
        return;
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->buf)
        ggml_backend_buffer_free(ctx->buf);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}
