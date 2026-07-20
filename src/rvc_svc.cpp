// rvc_svc.cpp — see rvc_svc.h.
//
// Built against tools/rvc_torch_parity.py, which pins every stage at
// cos 1.00000000 against torch. Where this file looks odd, the numpy spec and
// docs/music-transcription/RVC_BLUEPRINT.md say why — most of the oddities are
// upstream quirks baked into the trained weights, not choices.

#include "rvc_svc.h"

#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Hyperparameters
// ---------------------------------------------------------------------------
struct rvc_hparams {
    int content_dim = 768;
    int hidden = 192;
    int inter = 192;
    int n_layers = 6;
    int n_heads = 2;
    int rel_window = 10;
    int n_speakers = 109;
    int gin = 256;
    int sample_rate = 40000;
    int upsample_initial_channel = 512;
    std::vector<int> upsample_rates;
    std::vector<int> upsample_kernel_sizes;
    std::vector<int> resblock_kernel_sizes;
    // Flow geometry is HARDCODED upstream (models.py:624 —
    // ResidualCouplingBlock(inter, hidden, 5, 1, 3)), not config-derived, so
    // these hold for every checkpoint rather than just the one we converted.
    int flow_n_flows = 4;
    int flow_n_layers = 3;
    int flow_kernel = 5;
    int flow_dilation_rate = 1;
    // SineGen. harmonic_num is 0 in GeneratorNSF, which is why the random
    // initial phase is identically zero (rand_ini is one element and the next
    // line zeroes it) — so the ONLY live noise in the source module is additive.
    int harmonic_num = 0;
    float sine_amp = 0.1f;
    float add_noise_std = 0.003f;
    float noise_scale = 0.66666f;

    int upp() const {
        int p = 1;
        for (int r : upsample_rates)
            p *= r;
        return p;
    }
};

struct rvc_svc_context {
    rvc_hparams hp;
    rvc_svc_params params;
    ggml_backend_t backend = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> t; // name -> weight
    // Per-stage capture for the diff harness (HARD RULE #2: intermediates, not
    // just endpoints — endpoints alone cannot localise anything).
    bool capture = false;
    std::map<std::string, ggml_tensor*> caps;
    ggml_tensor* rev_idx_tensor = nullptr; // channel-reversal indices for Flip
};

namespace {

std::vector<int> read_i32_array(gguf_context* g, const char* key) {
    std::vector<int> out;
    const int64_t k = gguf_find_key(g, key);
    if (k < 0)
        return out;
    const int64_t n = gguf_get_arr_n(g, k);
    const auto* d = (const int32_t*)gguf_get_arr_data(g, k);
    out.assign(d, d + n);
    return out;
}

} // namespace

rvc_svc_params rvc_svc_default_params(void) {
    rvc_svc_params p{};
    p.n_threads = 0;
    p.use_gpu = true;
    p.gpu_device = 0;
    return p;
}

void rvc_svc_coarse_pitch(const float* f0_hz, int n, int* out_coarse) {
    // pipeline.py:73-137, exactly. The constants are model-side: emb_pitch is
    // an nn.Embedding(256, hidden) lookup, so an off-by-one here selects a
    // DIFFERENT LEARNED VECTOR rather than producing a small numeric error.
    const double f0_min = 50.0, f0_max = 1100.0;
    const double mel_min = 1127.0 * std::log(1.0 + f0_min / 700.0);
    const double mel_max = 1127.0 * std::log(1.0 + f0_max / 700.0);
    for (int i = 0; i < n; i++) {
        double mel = 1127.0 * std::log(1.0 + (double)f0_hz[i] / 700.0);
        if (mel > 0.0)
            mel = (mel - mel_min) * 254.0 / (mel_max - mel_min) + 1.0;
        if (mel <= 1.0)
            mel = 1.0;
        if (mel > 255.0)
            mel = 255.0;
        out_coarse[i] = (int)std::lround(mel);
    }
}

int rvc_svc_content_dim(const rvc_svc_context* ctx) {
    return ctx ? ctx->hp.content_dim : 0;
}
int rvc_svc_n_speakers(const rvc_svc_context* ctx) {
    return ctx ? ctx->hp.n_speakers : 0;
}
int rvc_svc_sample_rate(const rvc_svc_context* ctx) {
    return ctx ? ctx->hp.sample_rate : 0;
}

rvc_svc_context* rvc_svc_init_from_file(const char* model_path, rvc_svc_params params) {
    gguf_context* meta = core_gguf::open_metadata(model_path);
    if (!meta) {
        fprintf(stderr, "rvc: cannot open %s\n", model_path);
        return nullptr;
    }
    const std::string arch = core_gguf::kv_str(meta, "general.architecture", "");
    if (arch != "rvc") {
        fprintf(stderr, "rvc: '%s' is not an rvc model (arch='%s')\n", model_path, arch.c_str());
        core_gguf::free_metadata(meta);
        return nullptr;
    }

    auto* ctx = new rvc_svc_context();
    ctx->params = params;
    rvc_hparams& hp = ctx->hp;
    hp.content_dim = core_gguf::kv_u32(meta, "rvc.content_dim", 768);
    hp.hidden = core_gguf::kv_u32(meta, "rvc.hidden_channels", 192);
    hp.inter = core_gguf::kv_u32(meta, "rvc.inter_channels", 192);
    hp.n_layers = core_gguf::kv_u32(meta, "rvc.n_layers", 6);
    hp.n_heads = core_gguf::kv_u32(meta, "rvc.n_heads", 2);
    hp.rel_window = core_gguf::kv_u32(meta, "rvc.rel_attn_window", 10);
    hp.n_speakers = core_gguf::kv_u32(meta, "rvc.n_speakers", 109);
    hp.gin = core_gguf::kv_u32(meta, "rvc.gin_channels", 256);
    hp.sample_rate = core_gguf::kv_u32(meta, "rvc.sample_rate", 40000);
    hp.upsample_initial_channel = core_gguf::kv_u32(meta, "rvc.upsample_initial_channel", 512);
    hp.harmonic_num = core_gguf::kv_u32(meta, "rvc.harmonic_num", 0);
    hp.sine_amp = core_gguf::kv_f32(meta, "rvc.sine_amp", 0.1f);
    hp.add_noise_std = core_gguf::kv_f32(meta, "rvc.add_noise_std", 0.003f);
    hp.noise_scale = core_gguf::kv_f32(meta, "rvc.noise_scale", 0.66666f);
    hp.upsample_rates = read_i32_array(meta, "rvc.upsample_rates");
    hp.upsample_kernel_sizes = read_i32_array(meta, "rvc.upsample_kernel_sizes");
    hp.resblock_kernel_sizes = read_i32_array(meta, "rvc.resblock_kernel_sizes");
    core_gguf::free_metadata(meta);

    if (hp.upsample_rates.empty() || hp.upsample_rates.size() != hp.upsample_kernel_sizes.size()) {
        fprintf(stderr, "rvc: bad/missing upsample schedule in %s\n", model_path);
        delete ctx;
        return nullptr;
    }
    // The wire contract promises a 100 Hz feature/F0 rate, and it is derived,
    // not configured: sr / prod(upsample_rates). Refuse a checkpoint that
    // would silently violate it rather than resampling behind the caller.
    const int upp = hp.upp();
    if (upp <= 0 || hp.sample_rate % upp != 0 || hp.sample_rate / upp != 100) {
        fprintf(stderr,
                "rvc: sr/prod(upsample_rates) = %d/%d is not 100 Hz — this checkpoint does not "
                "honour the feature-rate contract (docs/music-transcription/SVC_RECORD_SHAPES.md)\n",
                hp.sample_rate, upp);
        delete ctx;
        return nullptr;
    }

    ctx->backend = params.use_gpu ? crispasr_init_gpu_backend() : nullptr;
    if (!ctx->backend)
        ctx->backend = ggml_backend_cpu_init();

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(model_path, ctx->backend, "rvc", wl)) {
        fprintf(stderr, "rvc: failed to load weights from %s\n", model_path);
        rvc_svc_free(ctx);
        return nullptr;
    }
    ctx->ctx_w = wl.ctx;
    ctx->buf_w = wl.buf;
    ctx->t = wl.tensors;

    // Cross-check the geometry we were told against the geometry the weights
    // actually have. A GGUF whose KVs disagree with its tensors would otherwise
    // fail much later as an unexplained shape error.
    auto need = [&](const char* n) -> ggml_tensor* {
        auto it = ctx->t.find(n);
        return it == ctx->t.end() ? nullptr : it->second;
    };
    ggml_tensor* emb_phone = need("enc_p.emb_phone.weight");
    ggml_tensor* emb_g = need("emb_g.weight");
    if (!emb_phone || !emb_g) {
        fprintf(stderr, "rvc: missing enc_p.emb_phone.weight / emb_g.weight\n");
        rvc_svc_free(ctx);
        return nullptr;
    }
    if ((int)emb_phone->ne[0] != hp.content_dim) {
        fprintf(stderr, "rvc: content_dim KV says %d but emb_phone is %d — refusing a v1/v2 mismatch\n", hp.content_dim,
                (int)emb_phone->ne[0]);
        rvc_svc_free(ctx);
        return nullptr;
    }

    fprintf(stderr, "rvc: content_dim=%d hidden=%d layers=%d heads=%d speakers=%d sr=%d upp=%d (%d fps)\n",
            hp.content_dim, hp.hidden, hp.n_layers, hp.n_heads, hp.n_speakers, hp.sample_rate, upp,
            hp.sample_rate / upp);
    return ctx;
}

void rvc_svc_free(rvc_svc_context* ctx) {
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

void rvc_svc_result_free(rvc_svc_result* r) {
    if (!r)
        return;
    free(r->pcm);
    delete r;
}

// ---------------------------------------------------------------------------
// enc_p — TextEncoder (relative-position transformer)
//
// Validated stage-for-stage against tools/rvc_torch_parity.py (itself cos
// 1.00000000 vs torch). Traps encoded here, all from RVC_BLUEPRINT.md 2b:
//   * LeakyReLU slope 0.1, NOT torch's 0.01 default.
//   * x *= sqrt(hidden) BEFORE the lrelu.
//   * POST-norm residuals: x = norm(x + f(x)).
//   * Relative-position attention over keys AND values (window 10).
//   * FFN is SAME-padded with a plain ReLU.
//   * LayerNorm is over the CHANNEL dim.
// ---------------------------------------------------------------------------

namespace {

// A kernel-1 Conv1d IS a linear, but it is stored with the kernel axis:
// GGUF (out, in, 1) -> ggml ne = [1, in, out]. Drop the leading 1 so mul_mat
// contracts over `in` instead of aborting on a 1-vs-in mismatch.
ggml_tensor* rvc_conv1x1_as_linear(ggml_context* g, ggml_tensor* w) {
    if (w->ne[0] == 1 && w->ne[2] > 1)
        return ggml_reshape_2d(g, w, w->ne[1], w->ne[2]);
    return w;
}

ggml_tensor* rvc_layer_norm(ggml_context* g, ggml_tensor* x, ggml_tensor* gamma, ggml_tensor* beta, float eps) {
    // x: [C, T]. Their LayerNorm transposes so the stats are over C.
    x = ggml_norm(g, x, eps);
    x = ggml_mul(g, x, gamma);
    return ggml_add(g, x, beta);
}

// Slice/pad emb_rel to 2T-1 entries, exactly as _get_relative_embeddings.
// NOTE: the padding is SYMMETRIC (pad_len on BOTH sides). ggml_pad only
// appends, so the front pad is a concat — getting this wrong reads past the
// tensor and trips ggml's view-bounds assert.
ggml_tensor* rvc_rel_embeddings(ggml_context* g, ggml_tensor* emb, int T, int window) {
    const int pad_len = std::max(T - (window + 1), 0);
    const int start = std::max((window + 1) - T, 0);
    // emb: GGUF (1, 2w+1, d) -> ggml ne = [d, 2w+1, 1]
    ggml_tensor* e = ggml_reshape_2d(g, emb, emb->ne[0], emb->ne[1]);
    if (pad_len > 0) {
        ggml_tensor* pre = ggml_new_tensor_2d(g, GGML_TYPE_F32, e->ne[0], pad_len);
        pre = ggml_scale(g, pre, 0.0f);
        e = ggml_concat(g, pre, e, 1);        // front pad
        e = ggml_pad(g, e, 0, pad_len, 0, 0); // back pad
    }
    return ggml_cont(g, ggml_view_2d(g, e, e->ne[0], 2 * T - 1, e->nb[1], (size_t)start * e->nb[1]));
}

// [2T-1, T, H] -> [T, T, H]: the skew from relative to absolute indexing.
ggml_tensor* rvc_rel_to_abs(ggml_context* g, ggml_tensor* x, int T) {
    x = ggml_pad(g, x, 1, 0, 0, 0); // [2T, T, H]
    const int64_t H = x->ne[2];
    x = ggml_cont(g, ggml_reshape_2d(g, x, 2 * T * T, H)); // flatten
    x = ggml_pad(g, x, T - 1, 0, 0, 0);                    // [2T*T + T-1, H]
    x = ggml_cont(g, ggml_reshape_3d(g, x, 2 * T - 1, T + 1, H));
    return ggml_cont(g, ggml_view_3d(g, x, T, T, H, x->nb[1], x->nb[2], (size_t)(T - 1) * x->nb[0]));
}

// [T, T, H] -> [2T-1, T, H]: the inverse skew, for relative VALUES.
ggml_tensor* rvc_abs_to_rel(ggml_context* g, ggml_tensor* x, int T) {
    const int64_t H = x->ne[2];
    x = ggml_pad(g, x, T - 1, 0, 0, 0); // [2T-1, T, H]
    x = ggml_cont(g, ggml_reshape_2d(g, x, T * (2 * T - 1), H));
    // pad the FRONT by T: ggml_pad only appends, so pad the end of a reversed
    // view is not available either — build it with a zero prefix concat.
    ggml_tensor* pre = ggml_new_tensor_2d(g, GGML_TYPE_F32, T, H);
    pre = ggml_scale(g, pre, 0.0f);
    x = ggml_concat(g, pre, x, 0); // [T*(2T-1)+T, H]
    x = ggml_cont(g, ggml_reshape_3d(g, x, 2 * T, T, H));
    return ggml_cont(g, ggml_view_3d(g, x, 2 * T - 1, T, H, x->nb[1], x->nb[2], (size_t)1 * x->nb[0]));
}

} // namespace

namespace {

// Build the enc_p graph. content: [content_dim, T] f32, pitch: [T] i32.
// Returns stats [2*inter, T]; caller splits into m_p / logs_p.
ggml_tensor* rvc_enc_p_graph(ggml_context* g, rvc_svc_context* c, ggml_tensor* content, ggml_tensor* pitch, int T) {
    const rvc_hparams& hp = c->hp;
    auto W = [&](const std::string& n) -> ggml_tensor* {
        auto it = c->t.find(n);
        if (it == c->t.end()) {
            fprintf(stderr, "rvc: missing tensor %s\n", n.c_str());
            return nullptr;
        }
        return it->second;
    };

    // emb_phone is a Linear: [content_dim, hidden] weight in ggml layout.
    // TAP records a tensor for the diff. `chan_time` marks stages whose
    // REFERENCE is stored as row-major (channels, time) — i.e. time fastest —
    // which is how torch stores a (b, C, T) tensor. Our working layout is
    // [C, T] with CHANNELS fastest, the exact transpose, so those stages must
    // be transposed before comparison or the cosine is meaningless.
    // Getting this wrong reads as a catastrophic FAIL (cos ~0) on a correct
    // graph: `agg` passed and `ctx` "failed" purely because the first is
    // compared against a numpy (H,T,hd) buffer that happens to share our order
    // and the second against a (C,T) one that does not.
    auto TAP = [&](const std::string& nm, ggml_tensor* v, bool chan_time = false) {
        if (c->capture) {
            ggml_tensor* o = chan_time ? ggml_cont(g, ggml_transpose(g, v)) : v;
            ggml_set_output(o);
            c->caps[nm] = o;
        }
        return v;
    };

    ggml_tensor* x = ggml_mul_mat(g, W("enc_p.emb_phone.weight"), content); // [hidden, T]
    x = ggml_add(g, x, W("enc_p.emb_phone.bias"));
    // emb_pitch is an EMBEDDING lookup, not a matmul.
    x = ggml_add(g, x, ggml_get_rows(g, W("enc_p.emb_pitch.weight"), pitch));
    x = ggml_scale(g, x, std::sqrt((float)hp.hidden)); // BEFORE the lrelu
    x = ggml_leaky_relu(g, x, 0.1f, false);            // slope 0.1, not 0.01
    TAP("encp_lrelu", x);

    const int H = hp.n_heads;
    const int hd = hp.hidden / H;
    const float scale = 1.0f / std::sqrt((float)hd);

    for (int l = 0; l < hp.n_layers; l++) {
        const std::string p = "enc_p.encoder.attn_layers." + std::to_string(l) + ".";
        // k=1 convs are plain linears over the channel dim.
        ggml_tensor* q =
            ggml_add(g, ggml_mul_mat(g, rvc_conv1x1_as_linear(g, W(p + "conv_q.weight")), x), W(p + "conv_q.bias"));
        ggml_tensor* k =
            ggml_add(g, ggml_mul_mat(g, rvc_conv1x1_as_linear(g, W(p + "conv_k.weight")), x), W(p + "conv_k.bias"));
        ggml_tensor* v =
            ggml_add(g, ggml_mul_mat(g, rvc_conv1x1_as_linear(g, W(p + "conv_v.weight")), x), W(p + "conv_v.bias"));

        // [hidden, T] -> [hd, T, H] -> heads on ne2
        q = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, q, hd, H, T), 0, 2, 1, 3));
        k = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, k, hd, H, T), 0, 2, 1, 3));
        v = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, v, hd, H, T), 0, 2, 1, 3));

        ggml_tensor* qs = ggml_scale(g, q, scale);
        ggml_tensor* scores = ggml_mul_mat(g, k, qs); // [T_k, T_q, H]

        ggml_tensor* rel_k = rvc_rel_embeddings(g, W(p + "emb_rel_k"), T, hp.rel_window); // [hd, 2T-1]
        ggml_tensor* rl = ggml_mul_mat(g, rel_k, qs);                                     // [2T-1, T, H]
        scores = ggml_add(g, scores, rvc_rel_to_abs(g, rl, T));

        ggml_tensor* attn = ggml_soft_max(g, scores); // [T_k, T_q, H]
        if (l == 0)
            TAP("encp_L0_attn_w", attn);
        if (l == 0)
            TAP("encp_L0_v", v);
        ggml_tensor* out = ggml_mul_mat(g, ggml_cont(g, ggml_transpose(g, v)), attn); // [hd, T_q, H]
        if (l == 0)
            TAP("encp_L0_agg", out);

        // relative VALUES — omitting this still "works" and merely degrades.
        ggml_tensor* rel_v = rvc_rel_embeddings(g, W(p + "emb_rel_v"), T, hp.rel_window); // [hd, 2T-1]
        ggml_tensor* ar = rvc_abs_to_rel(g, attn, T);                                     // [2T-1, T, H]
        out = ggml_add(g, out, ggml_mul_mat(g, ggml_cont(g, ggml_transpose(g, rel_v)), ar));

        out = ggml_cont(g, ggml_permute(g, out, 0, 2, 1, 3)); // [hd, H, T]
        out = ggml_reshape_2d(g, out, hp.hidden, T);
        if (l == 0)
            TAP("encp_L0_ctx", out, /*chan_time=*/true);
        ggml_tensor* y =
            ggml_add(g, ggml_mul_mat(g, rvc_conv1x1_as_linear(g, W(p + "conv_o.weight")), out), W(p + "conv_o.bias"));
        TAP("encp_L" + std::to_string(l) + "_attn", y, /*chan_time=*/true);

        const std::string n1 = "enc_p.encoder.norm_layers_1." + std::to_string(l) + ".";
        x = rvc_layer_norm(g, ggml_add(g, x, y), W(n1 + "gamma"), W(n1 + "beta"), 1e-5f); // POST-norm
        TAP("encp_L" + std::to_string(l) + "_norm1", x, /*chan_time=*/true);

        const std::string f = "enc_p.encoder.ffn_layers." + std::to_string(l) + ".";
        // ggml_conv_1d takes input as [length, channels]; our working layout
        // is [channels, time], so transpose in and back out. The bias is added
        // AFTER transposing back so it broadcasts over time, not over channels.
        ggml_tensor* w1 = W(f + "conv_1.weight");
        const int k1 = (int)w1->ne[0];
        ggml_tensor* h = ggml_conv_1d(g, w1, ggml_cont(g, ggml_transpose(g, x)), 1, (k1 - 1) / 2, 1);
        h = ggml_add(g, ggml_cont(g, ggml_transpose(g, h)), W(f + "conv_1.bias"));
        h = ggml_relu(g, h); // plain ReLU: activation != "gelu" here
        ggml_tensor* w2 = W(f + "conv_2.weight");
        const int k2 = (int)w2->ne[0];
        ggml_tensor* y2 = ggml_conv_1d(g, w2, ggml_cont(g, ggml_transpose(g, h)), 1, (k2 - 1) / 2, 1);
        y2 = ggml_add(g, ggml_cont(g, ggml_transpose(g, y2)), W(f + "conv_2.bias"));

        const std::string n2 = "enc_p.encoder.norm_layers_2." + std::to_string(l) + ".";
        TAP("encp_L" + std::to_string(l) + "_ffn", y2, /*chan_time=*/true);
        x = rvc_layer_norm(g, ggml_add(g, x, y2), W(n2 + "gamma"), W(n2 + "beta"), 1e-5f);
        TAP("encp_L" + std::to_string(l) + "_norm2", x, /*chan_time=*/true);
    }

    ggml_tensor* stats =
        ggml_add(g, ggml_mul_mat(g, rvc_conv1x1_as_linear(g, W("enc_p.proj.weight")), x), W("enc_p.proj.bias"));
    return stats; // [2*inter, T]
}

} // namespace

// ---------------------------------------------------------------------------
// flow — ResidualCouplingBlock, REVERSE pass
//
// Traps (RVC_BLUEPRINT.md 2c):
//   * mean_only=True, so `logs` is ZERO and the coupling is purely ADDITIVE:
//     the reverse is x1 = x1 - m, NOT (x1 - m) * exp(-logs).
//   * flows interleaves [Coupling, Flip] x 4 and the reverse walks it
//     backwards, so Flip comes FIRST.
//   * Flip reverses the CHANNEL axis.
//   * The WaveNet is gated: tanh(first half) * sigmoid(second half) of
//     (x_in + g_l), with the speaker conditioning projected once then sliced
//     per layer.
//   * kernel 5 / dilation rate 1 / 3 layers are hardcoded at models.py:624.
// ---------------------------------------------------------------------------

namespace {

// Reverse the CHANNEL axis. ggml has no flip, and a negative-stride view is not
// expressible, so transpose -> get_rows(reversed index) -> transpose back.
ggml_tensor* rvc_flip_channels(ggml_context* g, ggml_tensor* x, ggml_tensor* rev_idx) {
    ggml_tensor* xt = ggml_cont(g, ggml_transpose(g, x)); // [T, C]
    ggml_tensor* f = ggml_get_rows(g, xt, rev_idx);       // [T, C] channels reversed
    return ggml_cont(g, ggml_transpose(g, f));            // [C, T]
}

ggml_tensor* rvc_wn(ggml_context* g, rvc_svc_context* c, const std::string& pre, ggml_tensor* x, ggml_tensor* gcond,
                    int hidden, int n_layers, int kernel, int dil_rate) {
    auto W = [&](const std::string& n) -> ggml_tensor* {
        auto it = c->t.find(n);
        return it == c->t.end() ? nullptr : it->second;
    };
    ggml_tensor* output = nullptr;
    for (int i = 0; i < n_layers; i++) {
        const int d = (int)std::pow((double)dil_rate, (double)i);
        const int pad = (kernel * d - d) / 2;
        ggml_tensor* w = W(pre + "in_layers." + std::to_string(i) + ".weight");
        // conv_1d wants [length, channels]; our layout is [channels, time].
        ggml_tensor* xin = ggml_conv_1d(g, w, ggml_cont(g, ggml_transpose(g, x)), 1, pad, d);
        xin = ggml_add(g, ggml_cont(g, ggml_transpose(g, xin)),
                       W(pre + "in_layers." + std::to_string(i) + ".bias")); // [2*hidden, T]

        // speaker conditioning: one projection, sliced per layer
        ggml_tensor* gl = ggml_cont(g, ggml_view_1d(g, gcond, 2 * hidden, (size_t)i * 2 * hidden * gcond->nb[0]));
        xin = ggml_add(g, xin, gl);

        const int64_t T = xin->ne[1];
        ggml_tensor* ta = ggml_cont(g, ggml_view_2d(g, xin, hidden, T, xin->nb[1], 0));
        ggml_tensor* sa = ggml_cont(g, ggml_view_2d(g, xin, hidden, T, xin->nb[1], (size_t)hidden * xin->nb[0]));
        ggml_tensor* acts = ggml_mul(g, ggml_tanh(g, ta), ggml_sigmoid(g, sa)); // gated

        ggml_tensor* rw = W(pre + "res_skip_layers." + std::to_string(i) + ".weight");
        ggml_tensor* rs = ggml_add(g, ggml_mul_mat(g, rvc_conv1x1_as_linear(g, rw), acts),
                                   W(pre + "res_skip_layers." + std::to_string(i) + ".bias"));
        if (i < n_layers - 1) {
            ggml_tensor* res = ggml_cont(g, ggml_view_2d(g, rs, hidden, T, rs->nb[1], 0));
            ggml_tensor* skp = ggml_cont(g, ggml_view_2d(g, rs, hidden, T, rs->nb[1], (size_t)hidden * rs->nb[0]));
            x = ggml_add(g, x, res);
            output = output ? ggml_add(g, output, skp) : skp;
        } else {
            output = output ? ggml_add(g, output, rs) : rs;
        }
    }
    return output;
}

ggml_tensor* rvc_flow_graph_conds(ggml_context* g, rvc_svc_context* c, ggml_tensor* z_p,
                                  const std::vector<ggml_tensor*>& conds, ggml_tensor* rev_idx, int T) {
    const rvc_hparams& hp = c->hp;
    auto W = [&](const std::string& n) -> ggml_tensor* {
        auto it = c->t.find(n);
        return it == c->t.end() ? nullptr : it->second;
    };
    ggml_tensor* x = z_p;
    const int half = hp.inter / 2;
    for (int idx = hp.flow_n_flows - 1; idx >= 0; idx--) {
        x = rvc_flip_channels(g, x, rev_idx); // Flip comes FIRST in reverse
        const std::string p = "flow.flows." + std::to_string(idx * 2) + ".";
        ggml_tensor* x0 = ggml_cont(g, ggml_view_2d(g, x, half, T, x->nb[1], 0));
        ggml_tensor* x1 = ggml_cont(g, ggml_view_2d(g, x, half, T, x->nb[1], (size_t)half * x->nb[0]));
        ggml_tensor* h =
            ggml_add(g, ggml_mul_mat(g, rvc_conv1x1_as_linear(g, W(p + "pre.weight")), x0), W(p + "pre.bias"));
        h = rvc_wn(g, c, p + "enc.", h, conds[idx], hp.hidden, hp.flow_n_layers, hp.flow_kernel, hp.flow_dilation_rate);
        ggml_tensor* m =
            ggml_add(g, ggml_mul_mat(g, rvc_conv1x1_as_linear(g, W(p + "post.weight")), h), W(p + "post.bias"));
        x1 = ggml_sub(g, x1, m); // mean_only => no exp(-logs)
        x = ggml_concat(g, x0, x1, 0);
        if (c->capture) {
            ggml_tensor* cap = ggml_cont(g, ggml_transpose(g, x));
            ggml_set_output(cap);
            c->caps["flow_c" + std::to_string(idx)] = cap;
        }
    }
    return x;
}

ggml_tensor* rvc_flow_graph(ggml_context* g, rvc_svc_context* c, ggml_tensor* z_p, ggml_tensor* gcond,
                            ggml_tensor* rev_idx, int T) {
    const rvc_hparams& hp = c->hp;
    auto W = [&](const std::string& n) -> ggml_tensor* {
        auto it = c->t.find(n);
        return it == c->t.end() ? nullptr : it->second;
    };
    ggml_tensor* x = z_p;
    const int half = hp.inter / 2;
    for (int idx = hp.flow_n_flows - 1; idx >= 0; idx--) {
        x = rvc_flip_channels(g, x, rev_idx); // Flip comes FIRST in reverse
        const std::string p = "flow.flows." + std::to_string(idx * 2) + ".";
        ggml_tensor* x0 = ggml_cont(g, ggml_view_2d(g, x, half, T, x->nb[1], 0));
        ggml_tensor* x1 = ggml_cont(g, ggml_view_2d(g, x, half, T, x->nb[1], (size_t)half * x->nb[0]));

        ggml_tensor* h =
            ggml_add(g, ggml_mul_mat(g, rvc_conv1x1_as_linear(g, W(p + "pre.weight")), x0), W(p + "pre.bias"));
        h = rvc_wn(g, c, p + "enc.", h, gcond, hp.hidden, hp.flow_n_layers, hp.flow_kernel, hp.flow_dilation_rate);
        ggml_tensor* m =
            ggml_add(g, ggml_mul_mat(g, rvc_conv1x1_as_linear(g, W(p + "post.weight")), h), W(p + "post.bias"));
        x1 = ggml_sub(g, x1, m); // mean_only => no exp(-logs)
        x = ggml_concat(g, x0, x1, 0);
    }
    return x;
}

} // namespace

// ---------------------------------------------------------------------------
// Per-stage diff. Input-aligned: the reference carries input_phone/input_pitch
// AND both noise buffers, which we replay, so the comparison is deterministic
// even though the model is stochastic.
// ---------------------------------------------------------------------------

namespace {

double rvc_cos(const float* a, const float* b, int64_t n) {
    double d = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < n; i++) {
        d += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    const double den = std::sqrt(na) * std::sqrt(nb);
    return den > 0 ? d / den : (na == 0 && nb == 0 ? 1.0 : 0.0);
}

double rvc_max_abs(const float* a, const float* b, int64_t n) {
    double m = 0;
    for (int64_t i = 0; i < n; i++)
        m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

bool rvc_ref_get(core_gguf::WeightLoad& rw, const char* name, std::vector<float>& out) {
    auto it = rw.tensors.find(name);
    if (it == rw.tensors.end() || !it->second)
        return false;
    const int64_t n = ggml_nelements(it->second);
    out.resize((size_t)n);
    ggml_backend_tensor_get(it->second, out.data(), 0, (size_t)n * sizeof(float));
    return true;
}

} // namespace

int rvc_svc_diff(const char* model_gguf, const char* ref_gguf, int verbosity) {
    rvc_svc_params p = rvc_svc_default_params();
    p.use_gpu = false; // structural diff on CPU first
    rvc_svc_context* c = rvc_svc_init_from_file(model_gguf, p);
    if (!c) {
        fprintf(stderr, "rvc_diff: failed to load %s\n", model_gguf);
        return 2;
    }
    core_gguf::WeightLoad rw;
    if (!core_gguf::load_weights(ref_gguf, c->backend, "rvc_ref", rw)) {
        fprintf(stderr, "rvc_diff: failed to load reference %s\n", ref_gguf);
        rvc_svc_free(c);
        return 2;
    }

    std::vector<float> in_phone, in_pitch, ref_mp, ref_logs;
    if (!rvc_ref_get(rw, "input_phone", in_phone) || !rvc_ref_get(rw, "input_pitch", in_pitch)) {
        fprintf(stderr, "rvc_diff: reference lacks input_phone/input_pitch — re-dump with the current spec\n");
        core_gguf::free_weights(rw);
        rvc_svc_free(c);
        return 2;
    }
    const rvc_hparams& hp = c->hp;
    const int T = (int)(in_phone.size() / (size_t)hp.content_dim);

    // Graph
    const size_t mem = (size_t)512 * 1024 * 1024;
    std::vector<uint8_t> buf(mem);
    ggml_init_params ip{mem, buf.data(), true};
    ggml_context* g = ggml_init(ip);
    ggml_tensor* content = ggml_new_tensor_2d(g, GGML_TYPE_F32, hp.content_dim, T);
    ggml_tensor* pitch = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    ggml_set_input(content);
    ggml_set_input(pitch);
    c->capture = true;
    ggml_tensor* stats = rvc_enc_p_graph(g, c, content, pitch, T);

    // flow is INPUT-ALIGNED on the reference's z_p: it is drawn with the
    // model's own noise, so recomputing it here would diverge by construction.
    std::vector<float> ref_zp;
    ggml_tensor* z_out = nullptr;
    ggml_tensor* zp_in = nullptr;
    if (rvc_ref_get(rw, "z_p", ref_zp)) {
        // The reference z_p is (inter, T) row-major -> TIME fastest, whereas
        // our working layout is [inter, T] with CHANNELS fastest. Declare the
        // input in the reference's order and transpose in-graph, rather than
        // uploading a transposed buffer and getting cos ~0 on a correct graph
        // (the same trap that made enc_p look broken).
        zp_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, T, hp.inter); // [T, inter]
        ggml_set_input(zp_in);
        ggml_tensor* zp_ct = ggml_cont(g, ggml_transpose(g, zp_in)); // [inter, T]
        // speaker embedding -> cond projection, done once and sliced per layer
        ggml_tensor* gemb = ggml_cont(g, ggml_view_1d(g, c->t["emb_g.weight"], hp.gin, 0)); // sid 0
        ggml_tensor* rev = ggml_new_tensor_1d(g, GGML_TYPE_I32, hp.inter);
        ggml_set_input(rev);
        c->rev_idx_tensor = rev;
        // one cond_layer per coupling block; project inside rvc_wn's caller
        std::vector<ggml_tensor*> conds;
        for (int i = 0; i < hp.flow_n_flows; i++) {
            const std::string cp = "flow.flows." + std::to_string(i * 2) + ".enc.cond_layer.weight";
            ggml_tensor* cw = rvc_conv1x1_as_linear(g, c->t[cp]);
            ggml_tensor* cb = c->t["flow.flows." + std::to_string(i * 2) + ".enc.cond_layer.bias"];
            conds.push_back(ggml_add(g, ggml_mul_mat(g, cw, gemb), cb));
        }
        z_out = rvc_flow_graph_conds(g, c, zp_ct, conds, rev, T);
        ggml_set_output(z_out);
    }
    if (!stats) {
        ggml_free(g);
        core_gguf::free_weights(rw);
        rvc_svc_free(c);
        return 2;
    }
    // m_p/logs_p in the reference are (inter, T) row-major -> transpose.
    stats = ggml_cont(g, ggml_transpose(g, stats)); // [T, 2*inter]
    ggml_set_output(stats);
    ggml_cgraph* gf = ggml_new_graph_custom(g, 8192, false);
    ggml_build_forward_expand(gf, stats);
    // The transposed tap copies are not reachable from `stats`, so expand the
    // graph over them too — otherwise gallocr never allocates their buffers and
    // ggml_backend_tensor_get aborts with "tensor buffer not set".
    for (auto& kv : c->caps)
        ggml_build_forward_expand(gf, kv.second);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c->backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "rvc_diff: graph allocation failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(g);
        core_gguf::free_weights(rw);
        rvc_svc_free(c);
        return 2;
    }
    ggml_backend_tensor_set(content, in_phone.data(), 0, in_phone.size() * sizeof(float));
    std::vector<int32_t> pi((size_t)T);
    for (int i = 0; i < T; i++)
        pi[(size_t)i] = (int32_t)std::lround(in_pitch[(size_t)i]);
    ggml_backend_tensor_set(pitch, pi.data(), 0, pi.size() * sizeof(int32_t));
    if (zp_in) {
        ggml_backend_tensor_set(zp_in, ref_zp.data(), 0, ref_zp.size() * sizeof(float));
        std::vector<int32_t> rev((size_t)hp.inter);
        for (int i = 0; i < hp.inter; i++)
            rev[(size_t)i] = hp.inter - 1 - i;
        ggml_backend_tensor_set(c->rev_idx_tensor, rev.data(), 0, rev.size() * sizeof(int32_t));
    }
    ggml_backend_graph_compute(c->backend, gf);

    std::vector<float> out((size_t)ggml_nelements(stats));
    ggml_backend_tensor_get(stats, out.data(), 0, out.size() * sizeof(float));

    int n_fail = 0;
    const double COS_MIN = 0.9999;
    auto report = [&](const char* stage, const float* mine, const std::vector<float>& ref) {
        const int64_t n = (int64_t)ref.size();
        const double cs = rvc_cos(mine, ref.data(), n);
        const bool ok = cs >= COS_MIN;
        if (!ok)
            n_fail++;
        if (verbosity >= 1 || !ok)
            fprintf(stderr, "  %-10s %s cos=%.8f max_abs=%.3e\n", stage, ok ? "PASS" : "FAIL", cs,
                    rvc_max_abs(mine, ref.data(), n));
    };

    fprintf(stderr, "rvc diff (T=%d, content_dim=%d, inter=%d):\n", T, hp.content_dim, hp.inter);

    // Per-stage, EARLIEST FIRST — the first FAIL is the bug (HARD RULE #2).
    {
        std::vector<std::string> order;
        order.push_back("encp_lrelu");
        // inside layer 0's attention, earliest first
        order.push_back("encp_L0_q");
        order.push_back("encp_L0_scores_norel");
        order.push_back("encp_L0_rl");
        order.push_back("encp_L0_scores");
        order.push_back("encp_L0_attn_w");
        order.push_back("encp_L0_v");
        order.push_back("encp_L0_agg");
        order.push_back("encp_L0_ctx");
        for (int l = 0; l < hp.n_layers; l++) {
            const std::string L = "encp_L" + std::to_string(l);
            order.push_back(L + "_attn");
            order.push_back(L + "_norm1");
            order.push_back(L + "_ffn");
            order.push_back(L + "_norm2");
        }
        for (int i = hp.flow_n_flows - 1; i >= 0; i--)
            order.push_back("flow_c" + std::to_string(i));
        bool first = true;
        for (const auto& nm : order) {
            auto it = c->caps.find(nm);
            std::vector<float> ref;
            if (it == c->caps.end() || !rvc_ref_get(rw, nm.c_str(), ref))
                continue;
            std::vector<float> mine((size_t)ggml_nelements(it->second));
            ggml_backend_tensor_get(it->second, mine.data(), 0, mine.size() * sizeof(float));
            const int64_t n = (int64_t)std::min(mine.size(), ref.size());
            const double cs = rvc_cos(mine.data(), ref.data(), n);
            const bool ok = cs >= COS_MIN && mine.size() == ref.size();
            if (!ok)
                n_fail++;
            if (verbosity >= 1 || !ok) {
                fprintf(stderr, "  %-16s %s cos=%.8f max_abs=%.3e (mine=%zu ref=%zu)%s\n", nm.c_str(),
                        ok ? "PASS" : "FAIL", cs, rvc_max_abs(mine.data(), ref.data(), n), mine.size(), ref.size(),
                        (!ok && first) ? "  <-- FIRST DIVERGENCE" : "");
            }
            if (!ok)
                first = false;
        }
    }
    // stats is now [T, 2*inter] with T fastest: rows 0..inter-1 are m_p,
    // inter..2*inter-1 are logs_p, each T-contiguous — matching the reference.
    if (rvc_ref_get(rw, "m_p", ref_mp))
        report("m_p", out.data(), ref_mp);
    if (rvc_ref_get(rw, "logs_p", ref_logs))
        report("logs_p", out.data() + (size_t)hp.inter * T, ref_logs);

    fprintf(stderr, "  NOTE: dec (NSF vocoder) graph is not implemented yet — enc_p + flow only.\n");

    ggml_gallocr_free(alloc);
    ggml_free(g);
    core_gguf::free_weights(rw);
    rvc_svc_free(c);
    fprintf(stderr, "rvc diff: %d stage(s) FAILED.\n", n_fail);
    return n_fail == 0 ? 0 : 1;
}
