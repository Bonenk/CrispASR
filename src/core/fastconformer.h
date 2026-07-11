// src/core/fastconformer.h — shared FastConformer encoder helpers.
//
// Replaces the dw_striding pre-encode + rel-pos sinusoidal table + 24-32×
// macaron Conformer block body that parakeet, canary, and canary_ctc each
// have a near-identical copy of. The only per-model difference is whether
// Q/K/V/output projections, FFN linears, and pointwise convs carry biases
// — parakeet and canary_ctc don't, canary does.
//
// The helper is header-only so the compiler inlines it straight into each
// caller, producing the exact same ggml op sequence as the original inline
// code and preserving bit-identical graph execution on the regression sweep.
//
// Scope:
//   core_conformer::rel_shift        — (T-1)-shift view used by rel-pos attn
//   core_conformer::make_pos_enc     — sinusoidal rel-pos table builder
//   core_conformer::PreEncodeWeights — dw_striding subsampling weights
//   core_conformer::BlockWeights     — one Conformer block's tensors
//   core_conformer::build_pre_encode — conv front-end + linear → (d, T)
//   core_conformer::build_block      — one Conformer block
//
// Each model still owns its own ggml_context setup, input tensor creation,
// and final output head (CTC / RNN-T joint / transformer decoder). What
// moves into here is just the shared encoder body.

#pragma once

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace core_conformer {

// Env-var gate: CRISPASR_FC_NO_FLASH=1 disables flash_attn_ext in the
// FastConformer encoder and uses manual QK^T + softmax + V instead.
// Useful for A/B-ing the flash path on CPU for short sequences.
static inline bool fc_no_flash() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("CRISPASR_FC_NO_FLASH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

// Env-var gate: CRISPASR_FC_ATTN_CONT=1 restores the legacy ggml_cont copies
// of Q/K/V before flash_attn_ext. The kernel reads strided views directly
// (same as llama.cpp's permuted Q), so the copies are pure overhead — this
// gate exists only for regression bisection (issue #81).
static inline bool fc_attn_cont() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("CRISPASR_FC_ATTN_CONT");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

// ---------------------------------------------------------------------------
// Rel-pos "shift" trick from Transformer-XL / Conformer: rewrites the raw
// (Q @ R^T) matrix — which is indexed by (rel_pos, query_pos) with rel_pos
// running over 2T-1 positions — into a square (key_pos, query_pos) matrix
// of size (T, T). This is done as a strided view, no copy.
// Input:  [2T-1, T, H]
// Output: [T,   T, H]
// ---------------------------------------------------------------------------
static inline ggml_tensor* rel_shift(ggml_context* ctx, ggml_tensor* a) {
    const int T = (int)a->ne[1];
    const int H = (int)a->ne[2];
    return ggml_view_3d(ctx, a, T, T, H, a->nb[1] - a->nb[0], a->nb[2], (T - 1) * a->nb[0]);
}

// ---------------------------------------------------------------------------
// Sinusoidal rel-pos table, layout (d_model, 2T-1), with positions running
// descending from +(T-1) to -(T-1). Memory layout is pe[dim + pos*d] so
// that ne[0]=d (fast axis) and ne[1]=2T-1 (slow axis) matches the ggml
// tensor created as ggml_new_tensor_2d(F32, d, 2T-1).
//
// IMPORTANT: the tensor is created as ggml_new_tensor_2d(F32, d, 2T-1),
// giving ne[0]=d (fast) and ne[1]=2T-1 (slow). The CORRECT memory layout
// is therefore `pe[dim + pos*d]`, NOT `pe[(2*i)*K + j]` (which transposes
// the axes). An earlier version of parakeet/cohere shipped with the
// transposed layout — parakeet's TDT decoder is robust enough to mostly
// recover, but canary's encoder–decoder cross-attention is not. If you
// see word boundaries drifting on a new consumer, re-check this first.
// ---------------------------------------------------------------------------
static inline std::vector<float> make_pos_enc(int d_model, int T) {
    const int n_pos = 2 * T - 1;
    std::vector<float> pe((size_t)n_pos * d_model, 0.0f);
    for (int p = 0; p < n_pos; p++) {
        const float pos = (float)(T - 1 - p);
        for (int i = 0; i < d_model / 2; i++) {
            const float div = expf(-logf(10000.0f) * (float)(2 * i) / (float)d_model);
            pe[(size_t)p * d_model + 2 * i] = sinf(pos * div);
            pe[(size_t)p * d_model + 2 * i + 1] = cosf(pos * div);
        }
    }
    return pe;
}

// ---------------------------------------------------------------------------
// Local attention mask builder for rel_pos_local_attn models.
// Returns a flat row-major (T, T) float array where mask[q*T+k] = 0.0 for
// visible positions and -inf for masked positions. Positions within
// [q - left, q + right] are visible. The first `global_tokens` positions
// are always visible to all queries (and can see all positions).
// ---------------------------------------------------------------------------
static inline std::vector<float> make_local_attn_mask(int T, int left, int right, int global_tokens) {
    const float NEG_INF = -1e9f; // large enough for softmax to zero out
    std::vector<float> mask((size_t)T * T, NEG_INF);
    for (int q = 0; q < T; q++) {
        // Global token queries can attend to everything.
        if (q < global_tokens) {
            for (int k = 0; k < T; k++)
                mask[(size_t)q * T + k] = 0.0f;
            continue;
        }
        // Regular positions attend to their local window + global tokens.
        int k_lo = q - left;
        if (k_lo < 0)
            k_lo = 0;
        int k_hi = q + right;
        if (k_hi >= T)
            k_hi = T - 1;
        for (int k = k_lo; k <= k_hi; k++)
            mask[(size_t)q * T + k] = 0.0f;
        // Global token keys are always visible.
        for (int k = 0; k < global_tokens && k < T; k++)
            mask[(size_t)q * T + k] = 0.0f;
    }
    return mask;
}

// ---------------------------------------------------------------------------
// Pre-encode (dw_striding 8× subsampling) weights.
//
//   Conv2d(1→C,  k=3, s=2, p=1) → ReLU
//   Conv2d_dw(C, k=3, s=2, p=1)
//   Conv2d(C→C,  k=1)            → ReLU
//   Conv2d_dw(C, k=3, s=2, p=1)
//   Conv2d(C→C,  k=1)            → ReLU
//   flatten(freq×channel) → Linear(W3*C → d_model)
// ---------------------------------------------------------------------------
struct PreEncodeWeights {
    ggml_tensor *conv0_w = nullptr, *conv0_b = nullptr; // first strided conv
    ggml_tensor *conv2_w = nullptr, *conv2_b = nullptr; // dw
    ggml_tensor *conv3_w = nullptr, *conv3_b = nullptr; // pw
    ggml_tensor *conv5_w = nullptr, *conv5_b = nullptr; // dw
    ggml_tensor *conv6_w = nullptr, *conv6_b = nullptr; // pw
    ggml_tensor *out_w = nullptr, *out_b = nullptr;     // Linear(W3*C → d_model)
};

// Snap a 4D conv output (OW, OH, OC, N) as a 2D named dup (OC*OW, OH) for
// staged comparison.  Feature ordering: k = oc*(OW) + ow  (matches Python's
// x.transpose(1,2).reshape(T, C*Freq) convention).
static inline void snap_conv4d(ggml_context* ctx0, ggml_cgraph* gf, ggml_tensor* t, const char* name) {
    // permute(1, 2, 0, 3): (OW,OH,OC,N) → (OH,OC,OW,N)
    ggml_tensor* p = ggml_cont(ctx0, ggml_permute(ctx0, t, 1, 2, 0, 3));
    // reshape to (OC*OW, OH): ne[0]=OC*OW fastest, ne[1]=OH (T_enc)
    const int64_t C_Freq = t->ne[2] * t->ne[0]; // OC * OW
    const int64_t T_enc = t->ne[1];             // OH
    ggml_tensor* flat = ggml_reshape_2d(ctx0, p, C_Freq, T_enc);
    ggml_tensor* snap = ggml_dup(ctx0, flat);
    ggml_set_name(snap, name);
    ggml_build_forward_expand(gf, snap);
}

// Build the dw_striding pre-encoder. Input `mel` has shape (n_mels, T_mel).
// Returns a (d_model, T_enc) tensor where T_enc is read off the intermediate
// conv output via the caller (write it back through `out_T_enc`).
// When `gf` is non-null, named dup snaps are added after each conv step for
// staged comparison via the diff harness.
static inline ggml_tensor* build_pre_encode(ggml_context* ctx0, ggml_tensor* mel, const PreEncodeWeights& w,
                                            int subsampling_channels, int* out_T_enc, ggml_cgraph* gf = nullptr) {
    auto bias_4d = [&](ggml_tensor* b) {
        return ggml_cast(ctx0, ggml_reshape_4d(ctx0, b, 1, 1, b->ne[0], 1), GGML_TYPE_F32);
    };

    ggml_tensor* cur = ggml_conv_2d(ctx0, w.conv0_w, mel, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(w.conv0_b));
    if (gf)
        snap_conv4d(ctx0, gf, cur, "pre_enc_c0");
    cur = ggml_relu(ctx0, cur);

    cur = ggml_conv_2d_dw(ctx0, w.conv2_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(w.conv2_b));
    if (gf)
        snap_conv4d(ctx0, gf, cur, "pre_enc_c2");
    cur = ggml_conv_2d(ctx0, w.conv3_w, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(w.conv3_b));
    if (gf)
        snap_conv4d(ctx0, gf, cur, "pre_enc_c3");
    cur = ggml_relu(ctx0, cur);

    cur = ggml_conv_2d_dw(ctx0, w.conv5_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(w.conv5_b));
    if (gf)
        snap_conv4d(ctx0, gf, cur, "pre_enc_c5");
    cur = ggml_conv_2d(ctx0, w.conv6_w, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(w.conv6_b));
    if (gf)
        snap_conv4d(ctx0, gf, cur, "pre_enc_c6");
    cur = ggml_relu(ctx0, cur);

    const int H3 = (int)cur->ne[1];
    const int W3 = (int)cur->ne[0];
    const int C = subsampling_channels;
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
    cur = ggml_reshape_2d(ctx0, cur, W3 * C, H3);

    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, w.out_w, cur), w.out_b);

    if (out_T_enc)
        *out_T_enc = H3;
    return cur;
}

// ---------------------------------------------------------------------------
// One Conformer encoder block's weights. Bias tensors may be nullptr — when
// they are, the corresponding ggml_add is skipped. This accommodates the
// three FastConformer flavours we ship:
//
//   parakeet    — no biases on Q/K/V/out, ff linears, conv pw1/pw2
//   canary_ctc  — same as parakeet
//   canary      — biases on everything
//
// conv_dw_b is always present (parakeet/canary_ctc populate it synthetically
// via BN folding; canary has the PyTorch bias natively).
// ---------------------------------------------------------------------------
struct BlockWeights {
    // ---- FFN1 (macaron) ----
    ggml_tensor *norm_ff1_w = nullptr, *norm_ff1_b = nullptr;
    ggml_tensor *ff1_l1_w = nullptr, *ff1_l1_b = nullptr;
    ggml_tensor *ff1_l2_w = nullptr, *ff1_l2_b = nullptr;

    // ---- Self-attention (rel-pos with untied u/v biases) ----
    ggml_tensor *norm_attn_w = nullptr, *norm_attn_b = nullptr;
    ggml_tensor *attn_q_w = nullptr, *attn_q_b = nullptr;
    ggml_tensor *attn_k_w = nullptr, *attn_k_b = nullptr;
    ggml_tensor *attn_v_w = nullptr, *attn_v_b = nullptr;
    // Fused [Wq;Wk;Wv] concat (set by fuse_qkv at load, CRISPASR_FC_FUSED_QKV).
    // When attn_qkv_w is non-null, build_block does one matmul + view-split.
    ggml_tensor *attn_qkv_w = nullptr, *attn_qkv_b = nullptr;
    ggml_tensor *attn_out_w = nullptr, *attn_out_b = nullptr;
    ggml_tensor* attn_pos_w = nullptr; // no bias on rel-pos projection
    ggml_tensor* pos_bias_u = nullptr;
    ggml_tensor* pos_bias_v = nullptr;

    // ---- Conformer convolution module ----
    ggml_tensor *norm_conv_w = nullptr, *norm_conv_b = nullptr;
    ggml_tensor *conv_pw1_w = nullptr, *conv_pw1_b = nullptr; // (2d, d)
    ggml_tensor *conv_dw_w = nullptr, *conv_dw_b = nullptr;   // (d, 1, K)
    ggml_tensor* conv_dw_w_f32 = nullptr;                     // pre-cast F32 copy (set by BN fold)
    ggml_tensor *conv_pw2_w = nullptr, *conv_pw2_b = nullptr; // (d, d)
    // Post-dw-conv LayerNorm affine (NeMo conv_norm_type=layer_norm, e.g.
    // stt_kk_ru hybrid). nullptr for the common batch_norm models, whose
    // BN is folded into conv_dw_w/b at load instead.
    ggml_tensor *conv_ln_w = nullptr, *conv_ln_b = nullptr;

    // ---- FFN2 (macaron) ----
    ggml_tensor *norm_ff2_w = nullptr, *norm_ff2_b = nullptr;
    ggml_tensor *ff2_l1_w = nullptr, *ff2_l1_b = nullptr;
    ggml_tensor *ff2_l2_w = nullptr, *ff2_l2_b = nullptr;

    // ---- Block final LN ----
    ggml_tensor *norm_out_w = nullptr, *norm_out_b = nullptr;
};

// ---------------------------------------------------------------------------
// Load-time Q8_0 repack of the conv pointwise weights (issue #81).
//
// The GGUF stores conv.pw1/pw2 as 3D conv tensors (1, d, 2d)/(1, d, d), so
// crispasr-quantize's 2D-only rule skips them and they ship as F16 even in
// Q8_0/Q4_K models. The ggml CPU F16 mul_mat has no repack fast path and
// measures ~6x slower per FLOP than Q8_0 (M1 per-node profile: the two pw
// matmuls were 35% of encoder time in a q8_0 parakeet-ctc). Repacking them
// to 2D Q8_0 at load moves them onto the optimized int8 kernels.
//
// Gate: CRISPASR_FC_PW_Q8 — "0" forces off, "1" forces on. Unset: enabled
// only when the model is already quantized (pw quantization noise is then
// in-family); pure F16/F32 models keep their exact weights.
// ---------------------------------------------------------------------------
static inline int fc_pw_q8_mode() { // -1 = auto, 0 = off, 1 = on
    static int v = -2;
    if (v == -2) {
        const char* e = std::getenv("CRISPASR_FC_PW_Q8");
        v = (!e || !*e) ? -1 : (*e != '0' ? 1 : 0);
    }
    return v;
}

struct PwRepackBuf {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    void free() {
        if (buf)
            ggml_backend_buffer_free(buf);
        if (ctx)
            ggml_free(ctx);
        buf = nullptr;
        ctx = nullptr;
    }
};

// Repack each layer's F16 conv_pw1_w / conv_pw2_w into fresh 2D Q8_0 tensors
// (allocated in `out`) and repoint the BlockWeights fields. `model_quantized`
// should be true when the surrounding model weights are quantized (used by
// the auto gate). Returns the number of tensors repacked.
static inline int repack_conv_pw_q8(std::vector<BlockWeights*>& layers, ggml_backend_t backend, bool model_quantized,
                                    PwRepackBuf& out, const char* tag) {
    const int mode = fc_pw_q8_mode();
    if (mode == 0 || (mode == -1 && !model_quantized))
        return 0;

    auto eligible = [](ggml_tensor* t) {
        if (!t || t->type != GGML_TYPE_F16 || !ggml_is_contiguous(t))
            return false;
        const int64_t n_per_row = t->ne[0] > 1 ? t->ne[0] : t->ne[1]; // (d,2d) or 3D (1,d,2d)
        return n_per_row % ggml_blck_size(GGML_TYPE_Q8_0) == 0;
    };

    size_t n_tensors = 0;
    for (auto* e : layers)
        n_tensors += (eligible(e->conv_pw1_w) ? 1 : 0) + (eligible(e->conv_pw2_w) ? 1 : 0);
    if (n_tensors == 0)
        return 0;

    ggml_init_params ip = {n_tensors * ggml_tensor_overhead(), nullptr, true};
    out.ctx = ggml_init(ip);
    if (!out.ctx)
        return 0;

    // Pass 1: create the Q8_0 tensors (2D — collapse the leading unit dim).
    std::vector<std::pair<ggml_tensor**, ggml_tensor*>> jobs; // (slot, q8 tensor)
    for (auto* e : layers) {
        for (ggml_tensor** slot : {&e->conv_pw1_w, &e->conv_pw2_w}) {
            ggml_tensor* src = *slot;
            if (!eligible(src))
                continue;
            const int64_t n_per_row = src->ne[0] > 1 ? src->ne[0] : src->ne[1];
            const int64_t n_rows = ggml_nelements(src) / n_per_row;
            ggml_tensor* q8 = ggml_new_tensor_2d(out.ctx, GGML_TYPE_Q8_0, n_per_row, n_rows);
            jobs.push_back({slot, q8});
        }
    }
    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        out.free();
        return 0;
    }

    // Pass 2: F16 → F32 → Q8_0, upload, repoint.
    std::vector<ggml_fp16_t> h16;
    std::vector<float> h32;
    std::vector<uint8_t> hq;
    for (auto& j : jobs) {
        ggml_tensor* src = *j.first;
        const int64_t n = ggml_nelements(src);
        const int64_t n_per_row = j.second->ne[0];
        const int64_t n_rows = j.second->ne[1];
        h16.resize(n);
        h32.resize(n);
        ggml_backend_tensor_get(src, h16.data(), 0, n * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(h16.data(), h32.data(), n);
        hq.resize(ggml_row_size(GGML_TYPE_Q8_0, n_per_row) * n_rows);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, h32.data(), hq.data(), 0, n_rows, n_per_row, nullptr);
        ggml_backend_tensor_set(j.second, hq.data(), 0, hq.size());
        *j.first = j.second;
    }

    fprintf(stderr, "%s: repacked %zu F16 conv pw tensors to Q8_0 (CRISPASR_FC_PW_Q8)\n", tag, jobs.size());
    return (int)jobs.size();
}

// ---------------------------------------------------------------------------
// Load-time Q/K/V weight fusion (issue #81). Concatenates each layer's
// attn_q_w / attn_k_w / attn_v_w (same shape, same type) into one
// (d_in, 3*d_out) tensor so build_block issues a single matmul + view-split
// instead of three matmuls over the same input. Output rows are the same
// independent dot products, so the result is bit-identical to the split path.
//
// Gate: CRISPASR_FC_FUSED_QKV — "0" off, unset/other = on.
// ---------------------------------------------------------------------------
static inline bool fc_fused_qkv_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("CRISPASR_FC_FUSED_QKV");
        v = (e && *e == '0') ? 0 : 1;
    }
    return v != 0;
}

static inline int fuse_qkv(std::vector<BlockWeights*>& layers, ggml_backend_t backend, PwRepackBuf& out,
                           const char* tag) {
    if (!fc_fused_qkv_enabled())
        return 0;

    auto eligible = [](const BlockWeights* e) {
        ggml_tensor *q = e->attn_q_w, *k = e->attn_k_w, *v = e->attn_v_w;
        if (!q || !k || !v)
            return false;
        if (q->type != k->type || q->type != v->type)
            return false;
        if (!ggml_is_contiguous(q) || !ggml_is_contiguous(k) || !ggml_is_contiguous(v))
            return false;
        if (q->ne[0] != k->ne[0] || q->ne[0] != v->ne[0] || q->ne[1] != k->ne[1] || q->ne[1] != v->ne[1] ||
            ggml_n_dims(q) != 2 || ggml_n_dims(k) != 2 || ggml_n_dims(v) != 2)
            return false;
        // Biases: all absent or all present (F32 1-D).
        const int nb = (e->attn_q_b != nullptr) + (e->attn_k_b != nullptr) + (e->attn_v_b != nullptr);
        if (nb != 0 && nb != 3)
            return false;
        if (nb == 3 && (e->attn_q_b->type != GGML_TYPE_F32 || e->attn_k_b->type != GGML_TYPE_F32 ||
                        e->attn_v_b->type != GGML_TYPE_F32))
            return false;
        return true;
    };

    size_t n_tensors = 0;
    for (auto* e : layers)
        if (eligible(e))
            n_tensors += e->attn_q_b ? 2 : 1;
    if (n_tensors == 0)
        return 0;

    ggml_init_params ip = {n_tensors * ggml_tensor_overhead(), nullptr, true};
    out.ctx = ggml_init(ip);
    if (!out.ctx)
        return 0;

    struct Job {
        BlockWeights* e;
        ggml_tensor *w = nullptr, *b = nullptr;
    };
    std::vector<Job> jobs;
    for (auto* e : layers) {
        if (!eligible(e))
            continue;
        Job j;
        j.e = e;
        j.w = ggml_new_tensor_2d(out.ctx, e->attn_q_w->type, e->attn_q_w->ne[0], 3 * e->attn_q_w->ne[1]);
        if (e->attn_q_b)
            j.b = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, 3 * e->attn_q_b->ne[0]);
        jobs.push_back(j);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!buf) {
        out.free();
        return 0;
    }
    out.buf = buf;

    std::vector<uint8_t> h;
    int n_fused = 0;
    for (auto& j : jobs) {
        size_t off = 0;
        for (ggml_tensor* src : {j.e->attn_q_w, j.e->attn_k_w, j.e->attn_v_w}) {
            const size_t nb = ggml_nbytes(src);
            h.resize(nb);
            ggml_backend_tensor_get(src, h.data(), 0, nb);
            ggml_backend_tensor_set(j.w, h.data(), off, nb);
            off += nb;
        }
        j.e->attn_qkv_w = j.w;
        if (j.b) {
            off = 0;
            for (ggml_tensor* src : {j.e->attn_q_b, j.e->attn_k_b, j.e->attn_v_b}) {
                const size_t nb = ggml_nbytes(src);
                h.resize(nb);
                ggml_backend_tensor_get(src, h.data(), 0, nb);
                ggml_backend_tensor_set(j.b, h.data(), off, nb);
                off += nb;
            }
            j.e->attn_qkv_b = j.b;
        }
        n_fused++;
    }

    fprintf(stderr, "%s: fused Q/K/V projections for %d layers (CRISPASR_FC_FUSED_QKV)\n", tag, n_fused);
    return n_fused;
}

struct BlockParams {
    int d; // d_model
    int n_heads;
    int head_dim; // d / n_heads
    int K;        // conv_kernel (usually 9)
    float ln_eps; // LayerNorm epsilon

    // Local attention window (rel_pos_local_attn). Negative = full attention.
    int att_context_left = -1;  // max positions to the left each query sees
    int att_context_right = -1; // max positions to the right
    int global_tokens = 0;      // first N positions are global (visible to all)
};

// Build one Conformer block. `cur` must be (d, T). `pos_enc` is the shared
// sinusoidal rel-pos table (d, 2T-1). `local_attn_mask` is an optional (T, T)
// F32 tensor with 0.0 for visible positions and -inf for masked positions
// (used by rel_pos_local_attn models). Pass nullptr for full attention.
// Returns the post-block (d, T) output.
static inline ggml_tensor* build_block(ggml_context* ctx0, ggml_tensor* cur, ggml_tensor* pos_enc, int T,
                                       const BlockWeights& e, const BlockParams& p,
                                       ggml_tensor* local_attn_mask = nullptr) {
    const int d = p.d;
    const int n_heads = p.n_heads;
    const int head_dim = p.head_dim;
    const int K = p.K;
    const float eps = p.ln_eps;

    // Tiny helper: mul_mat + optional bias add.
    auto mm_bias = [&](ggml_tensor* w, ggml_tensor* x, ggml_tensor* b) {
        ggml_tensor* y = ggml_mul_mat(ctx0, w, x);
        return b ? ggml_add(ctx0, y, b) : y;
    };

    ggml_tensor* inpL = cur;

    // ---- FFN1 (macaron half) ----
    ggml_tensor* x = ggml_norm_affine(ctx0, cur, e.norm_ff1_w, e.norm_ff1_b, eps);
    x = mm_bias(e.ff1_l1_w, x, e.ff1_l1_b);
    x = ggml_silu(ctx0, x);
    x = mm_bias(e.ff1_l2_w, x, e.ff1_l2_b);
    cur = ggml_add(ctx0, inpL, ggml_scale(ctx0, x, 0.5f));

    ggml_tensor* inpAttn = cur;

    // ---- Self-Attention (rel_pos with untied biases) ----
    x = ggml_norm_affine(ctx0, cur, e.norm_attn_w, e.norm_attn_b, eps);

    // Q/K/V projections — fused single matmul + view-split when the load-time
    // concat is available (bit-identical: each output row is the same dot
    // product either way), else three separate matmuls.
    ggml_tensor* Q;  // (d, T)
    ggml_tensor* K3; // (head_dim, n_heads, T)
    ggml_tensor* V3; // (head_dim, n_heads, T)
    if (e.attn_qkv_w) {
        ggml_tensor* qkv = ggml_mul_mat(ctx0, e.attn_qkv_w, x); // (3d, T)
        if (e.attn_qkv_b)
            qkv = ggml_add(ctx0, qkv, e.attn_qkv_b);
        Q = ggml_view_2d(ctx0, qkv, d, T, qkv->nb[1], 0);
        K3 = ggml_view_3d(ctx0, qkv, head_dim, n_heads, T, (size_t)head_dim * sizeof(float), qkv->nb[1],
                          (size_t)d * sizeof(float));
        V3 = ggml_view_3d(ctx0, qkv, head_dim, n_heads, T, (size_t)head_dim * sizeof(float), qkv->nb[1],
                          (size_t)2 * d * sizeof(float));
    } else {
        Q = mm_bias(e.attn_q_w, x, e.attn_q_b);
        K3 = ggml_reshape_3d(ctx0, mm_bias(e.attn_k_w, x, e.attn_k_b), head_dim, n_heads, T);
        V3 = ggml_reshape_3d(ctx0, mm_bias(e.attn_v_w, x, e.attn_v_b), head_dim, n_heads, T);
    }
    ggml_tensor* R = ggml_mul_mat(ctx0, e.attn_pos_w, pos_enc); // no bias

    ggml_tensor* Q_u = ggml_add(ctx0, Q, ggml_reshape_1d(ctx0, e.pos_bias_u, d));
    ggml_tensor* Q_v = ggml_add(ctx0, Q, ggml_reshape_1d(ctx0, e.pos_bias_v, d));

    Q_u = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q_u, head_dim, n_heads, T), 0, 2, 1, 3);
    Q_v = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q_v, head_dim, n_heads, T), 0, 2, 1, 3);
    ggml_tensor* K_ = ggml_permute(ctx0, K3, 0, 2, 1, 3);
    R = ggml_permute(ctx0, ggml_reshape_3d(ctx0, R, head_dim, n_heads, 2 * T - 1), 0, 2, 1, 3);

    // Compute the relative position bias BD = rel_shift(Q_v × R^T).
    // This is query-dependent so it can't be precomputed, but it CAN
    // be passed as the additive mask to ggml_flash_attn_ext, which
    // fuses AC (= Q_u × K^T) + BD + softmax + ×V into one kernel.
    ggml_tensor* BD_raw = ggml_mul_mat(ctx0, ggml_cont(ctx0, R), Q_v);
    ggml_tensor* BD = rel_shift(ctx0, BD_raw);

    // flash_attn_ext computes: softmax(Q_u × K^T * scale + mask) × V
    // We need:                 softmax((Q_u × K^T + BD) * scale)  × V
    // So pass mask = BD * scale to get equivalent semantics.
    const float scale = 1.0f / sqrtf((float)head_dim);
    // BD is a strided view from rel_shift — make contiguous before scale/cast.
    ggml_tensor* BD_c = ggml_cont(ctx0, BD);
    ggml_tensor* BD_scaled = ggml_scale(ctx0, BD_c, scale);

    // Local attention window mask: add -inf for positions outside the window.
    // The mask tensor is created externally and passed via `local_attn_mask`.
    if (local_attn_mask) {
        // Broadcast (T, T) mask across all heads in BD_scaled (T, T, n_heads).
        BD_scaled = ggml_add(ctx0, BD_scaled, local_attn_mask);
    }

    ggml_tensor* attn_out;
    if (fc_no_flash()) {
        // Manual attention: QK^T + BD, softmax, ×V — no flash_attn_ext.
        // Q_u, K_ are (head_dim, T, n_heads) after permute.
        // mul_mat(K, Q) computes Q^T × K^T^T = Q^T × K → (T, T, n_heads).
        ggml_tensor* Q_u_c = ggml_cont(ctx0, Q_u);            // (head_dim, T, n_heads)
        ggml_tensor* K_c = ggml_cont(ctx0, K_);               // (head_dim, T, n_heads)
        ggml_tensor* scores = ggml_mul_mat(ctx0, K_c, Q_u_c); // (T, T, n_heads)
        ggml_tensor* BD_c2 = ggml_cont(ctx0, BD);             // (T, T, n_heads)
        scores = ggml_add(ctx0, scores, BD_c2);
        if (local_attn_mask)
            scores = ggml_add(ctx0, scores, local_attn_mask);
        scores = ggml_soft_max_ext(ctx0, scores, nullptr, scale, 0.0f);
        // V: (head_dim, n_heads, T) → permute(0,2,1,3) → (head_dim, T, n_heads)
        // Then transpose via permute(1,0,2,3) → (T, head_dim, n_heads) for mul_mat
        ggml_tensor* V_3d = ggml_cont(ctx0, ggml_permute(ctx0, V3, 0, 2, 1, 3));
        // V_3d: (head_dim, T, n_heads) — need (T, head_dim, n_heads) for mul_mat
        ggml_tensor* V_t = ggml_cont(ctx0, ggml_permute(ctx0, V_3d, 1, 0, 2, 3)); // (T, head_dim, n_heads)
        attn_out = ggml_mul_mat(ctx0, V_t, scores);                               // (head_dim, T, n_heads)
        attn_out = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, attn_out, 0, 2, 1, 3)), d, T);
    } else {
        // flash_attn_ext mask must be F16
        ggml_tensor* BD_mask = ggml_cast(ctx0, BD_scaled, GGML_TYPE_F16);

        // V needs [head_dim, T, n_heads] layout for flash_attn_ext (same as K).
        // The kernel reads strided views (nb0 == type size) directly — the
        // legacy ggml_cont copies of Q/K/V are restorable via
        // CRISPASR_FC_ATTN_CONT=1 for regression bisection.
        ggml_tensor* Q_f = Q_u;
        ggml_tensor* K_f = K_;
        ggml_tensor* V_f = ggml_permute(ctx0, V3, 0, 2, 1, 3);
        if (fc_attn_cont()) {
            Q_f = ggml_cont(ctx0, Q_f);
            K_f = ggml_cont(ctx0, K_f);
            V_f = ggml_cont(ctx0, V_f);
        }

        attn_out = ggml_flash_attn_ext(ctx0, Q_f, K_f, V_f, BD_mask, scale, 0.0f, 0.0f);
        attn_out = ggml_reshape_2d(ctx0, attn_out, d, T);
    }

    attn_out = mm_bias(e.attn_out_w, attn_out, e.attn_out_b);
    cur = ggml_add(ctx0, inpAttn, attn_out);

    // ---- Conformer convolution module ----
    ggml_tensor* inpConv = cur;
    x = ggml_norm_affine(ctx0, cur, e.norm_conv_w, e.norm_conv_b, eps);

    // pw1: (d → 2d), then sigmoid GLU — fused into one op, avoids strided-view
    // CUDA fallback that plagued the manual sigmoid path (see issue #81 PR #05).
    ggml_tensor* pw1_w = ggml_reshape_2d(ctx0, e.conv_pw1_w, d, 2 * d);
    ggml_tensor* cnv = mm_bias(pw1_w, x, e.conv_pw1_b);
    cnv = ggml_siglu_swapped(ctx0, cnv);

    // dw conv (kernel K, padding K/2). BN was folded into conv_dw_w/b at load.
    // Use pre-cast F32 weights if available (avoids F16→F32 cast per forward).
    ggml_tensor* dw_w_f32 = e.conv_dw_w_f32 ? e.conv_dw_w_f32 : ggml_cast(ctx0, e.conv_dw_w, GGML_TYPE_F32);
    ggml_tensor* dw_w_4d = ggml_reshape_4d(ctx0, dw_w_f32, K, 1, 1, d);
    cnv = ggml_cont(ctx0, ggml_transpose(ctx0, cnv)); // (d, T) → (T, d)
    cnv = ggml_reshape_4d(ctx0, cnv, T, 1, d, 1);
    cnv = ggml_conv_2d_dw_direct(ctx0, dw_w_4d, cnv, 1, 1, (K - 1) / 2, 0, 1, 1);
    cnv = ggml_cont(ctx0, ggml_permute(ctx0, cnv, 1, 2, 0, 3));
    cnv = ggml_reshape_2d(ctx0, cnv, d, T);

    cnv = ggml_add(ctx0, cnv, ggml_reshape_2d(ctx0, e.conv_dw_b, d, 1));
    if (e.conv_ln_w) // conv_norm_type=layer_norm: LN over channels per frame
        cnv = ggml_norm_affine(ctx0, cnv, e.conv_ln_w, e.conv_ln_b, eps);
    cnv = ggml_silu(ctx0, cnv);

    // pw2: (d → d)
    ggml_tensor* pw2_w = ggml_reshape_2d(ctx0, e.conv_pw2_w, d, d);
    cnv = mm_bias(pw2_w, cnv, e.conv_pw2_b);
    cur = ggml_add(ctx0, inpConv, cnv);

    // ---- FFN2 (macaron half) ----
    ggml_tensor* inpFF2 = cur;
    x = ggml_norm_affine(ctx0, cur, e.norm_ff2_w, e.norm_ff2_b, eps);
    x = mm_bias(e.ff2_l1_w, x, e.ff2_l1_b);
    x = ggml_silu(ctx0, x);
    x = mm_bias(e.ff2_l2_w, x, e.ff2_l2_b);
    cur = ggml_add(ctx0, inpFF2, ggml_scale(ctx0, x, 0.5f));

    // ---- Block final LN ----
    cur = ggml_norm_affine(ctx0, cur, e.norm_out_w, e.norm_out_b, eps);

    return cur;
}

} // namespace core_conformer
