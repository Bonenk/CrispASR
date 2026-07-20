// beat_this.cpp — Beat This! beat/downbeat tracking runtime (§251b).
//
// Status: front end + weight loading. The ggml graph is the remaining piece;
// see docs/music-transcription/PLAN.md §251b-1 for the traced blueprint.
//
// The front end is implemented and validated FIRST, deliberately: it is the
// part most likely to drift silently (a wrong window, normalization or mel
// layout produces a plausible spectrogram and wrong beats), and it can be
// checked against torchaudio without any of the network existing yet.

#include "beat_this.h"

#include "core/crispasr_env.h"
#include "core/fft.h"
#include "core/gguf_loader.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

// One roformer Attention sub-block (§251b-1). NOTE the bias asymmetry, which
// is not a transcription slip: to_qkv and to_out are bias-free, to_gates has a
// bias. `n_head` is dim/32 — 1, 2, 4 for frontend blocks 0/1/2.
struct bt_attn {
    ggml_tensor* gamma = nullptr;   // norm.gamma — RMSNorm scale, no bias
    ggml_tensor* qkv_w = nullptr;   // (dim, 3*n_head*head_dim)
    ggml_tensor* gates_w = nullptr; // (dim, n_head)  — ONE scalar per head
    ggml_tensor* gates_b = nullptr; // (n_head)
    ggml_tensor* out_w = nullptr;   // (n_head*head_dim, dim)
    int n_head = 1;
    int head_dim = 32;
};

// One roformer FeedForward: RMSNorm -> Linear -> GELU(erf) -> Linear.
struct bt_ff {
    ggml_tensor* gamma = nullptr; // net.0.gamma
    ggml_tensor* w1 = nullptr;    // net.1 (dim, 4*dim)
    ggml_tensor* b1 = nullptr;
    ggml_tensor* w2 = nullptr; // net.4 (4*dim, dim)
    ggml_tensor* b2 = nullptr;
};

struct bt_partial {
    bt_attn attnF;
    bt_ff ffF;
    bt_attn attnT;
    bt_ff ffT;
};

// A frontend block: PartialFTTransformer, then conv2d(k=(2,3), s=(2,1),
// p=(0,1)) + BN + GELU. The BN is already folded into the conv by the
// converter, which is why a bias exists here where torch has bias=False.
struct bt_block {
    bt_partial partial;
    ggml_tensor* conv_w = nullptr;
    ggml_tensor* conv_b = nullptr;
};

struct beat_this_context {
    ggml_backend_t backend = nullptr;
    core_gguf::WeightLoad wl;
    int n_threads = 4;

    int sample_rate = BEAT_THIS_SAMPLE_RATE;
    int n_fft = BEAT_THIS_N_FFT;
    int hop = BEAT_THIS_HOP;
    int mel_bins = BEAT_THIS_MEL_BINS;
    float log_multiplier = 1000.0f;

    // Baked [513, 128] row-major mel filterbank from the reference export.
    // Never re-derived: slaney-vs-htk and the freq/mel layout are classic
    // silent-drift sources, and the export ships the exact matrix the model
    // was trained and exported with.
    std::vector<float> mel_fb;
    int fb_freqs = 0;

    std::vector<float> hann; // periodic, length n_fft
    bool debug = false;

    // --- graph weights (bound in beat_this_init) -------------------------
    ggml_tensor* bn1d_scale = nullptr; // (128) per-FREQUENCY, pre-conv
    ggml_tensor* bn1d_offset = nullptr;
    ggml_tensor* stem_w = nullptr; // ne (kt=3, kf=4, ic=1, oc=32)
    ggml_tensor* stem_b = nullptr; // (32)

    bt_block blocks[3];
};

namespace {

// Periodic Hann (torch.hann_window default), NOT the symmetric np.hanning
// variant — they differ by ~2.4e-7/sample, which is enough to move a cos_min
// into the 0.95 band (see the mel pitfalls note in docs/contributing.md).
void build_hann_periodic(int n, std::vector<float>& w) {
    w.resize((size_t)n);
    const double two_pi = 6.283185307179586476925286766559;
    for (int i = 0; i < n; i++)
        w[(size_t)i] = (float)(0.5 - 0.5 * std::cos(two_pi * (double)i / (double)n));
}

// Reflect-pad index, matching torch's pad_mode="reflect" (no edge repeat).
inline int reflect_index(int i, int n) {
    if (n <= 1)
        return 0;
    while (i < 0 || i >= n) {
        if (i < 0)
            i = -i;
        if (i >= n)
            i = 2 * (n - 1) - i;
    }
    return i;
}

// --- graph pieces -------------------------------------------------------
//
// LAYOUT CONVENTION for everything below. The reference works in torch
// (b, c, f, t); ggml reverses that, so the block-level tensor here is
// ne = (t, f, c). Inside a partial transformer the sequence axis moves to
// ne[1] and the *other* spatial axis becomes the batch, folded into ne[2]
// exactly as einops folds it into the batch dim upstream:
//
//   frequency phase: ne = (c, f, t)   -> 't' independent sequences of length f
//   time phase:      ne = (c, t, f)   -> 'f' independent sequences of length t
//
// RMSNorm eps. Upstream is `F.normalize(x, dim=-1) * sqrt(size) * gamma`,
// which divides by the L2 norm with an eps of 1e-12 *outside* the sqrt. Since
// RMS = L2/sqrt(size), that is standard RMSNorm x gamma; 1e-12 is used here so
// the guard term stays far below the f16 weight noise floor rather than
// contributing its own drift.
ggml_tensor* bt_norm(ggml_context* c, ggml_tensor* x, ggml_tensor* gamma) {
    return ggml_mul(c, ggml_rms_norm(c, x, 1e-12f), gamma);
}

// Attention branch. `x` is ne (C, N, B); returns the BRANCH, ne (C, N, B) —
// the caller adds the residual. `pos` is an I32 vector 0..N-1.
ggml_tensor* bt_attention(ggml_context* c, const bt_attn& a, ggml_tensor* x, ggml_tensor* pos) {
    const int64_t N = x->ne[1], B = x->ne[2];
    const int64_t H = a.n_head, D = a.head_dim;

    // The gates read the NORMED x, not the raw input: upstream rebinds
    // `x = self.norm(x)` before computing both qkv and gates.
    ggml_tensor* xn = bt_norm(c, x, a.gamma);

    ggml_tensor* qkv = ggml_mul_mat(c, a.qkv_w, xn); // (3*H*D, N, B)
    const size_t esz = ggml_element_size(qkv);

    // to_qkv's output is packed "(qkv h d)" with d fastest, so q/k/v are three
    // contiguous H*D slabs and each slab is already (d, h) — no interleaving.
    auto slab = [&](int64_t which) {
        return ggml_cont(c, ggml_view_4d(c, qkv, D, H, N, B, D * esz, qkv->nb[1], qkv->nb[2], which * H * D * esz));
    };

    // RoPE wants (head_dim, heads, seq, batch) and mode NORMAL rotates
    // ADJACENT PAIRS — which is exactly rotary-embedding-torch's convention
    // (rotate_half regroups '... (d r) -> ... d r' with r=2, and freqs are
    // repeated pairwise). GGML_ROPE_TYPE_NEOX would rotate the split halves
    // instead and is silently wrong here. The checkpoint's rotary_embed.freqs
    // were verified bit-identical to the analytic 10000^(-2i/32) schedule, so
    // ggml's internal theta table needs no override.
    ggml_tensor* q = ggml_rope(c, slab(0), pos, (int)D, GGML_ROPE_TYPE_NORMAL);
    ggml_tensor* k = ggml_rope(c, slab(1), pos, (int)D, GGML_ROPE_TYPE_NORMAL);
    ggml_tensor* v = slab(2);

    // (D, H, N, B) -> (D, N, H, B), the shape mul_mat wants per head.
    q = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));
    k = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));

    // Attend() is plain SDPA with the default 1/sqrt(head_dim) scale: the
    // `scale` argument is never passed, so Attention.scale is dead code.
    ggml_tensor* kq = ggml_mul_mat(c, k, q); // (N_k, N_q, H, B)
    kq = ggml_soft_max_ext(c, kq, nullptr, 1.0f / std::sqrt((float)D), 0.0f);

    // v as (N, D, H, B) so mul_mat contracts over the key axis.
    ggml_tensor* vt = ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3));
    ggml_tensor* out = ggml_mul_mat(c, vt, kq); // (D, N_q, H, B)

    // PER-HEAD SIGMOID GATING. to_gates is Linear(dim -> heads): one scalar
    // per head, broadcast over that head's head_dim and over the sequence, and
    // applied BEFORE to_out. Dropping it leaves output that is plausible and
    // wrong — it scales rather than reshapes the activation.
    ggml_tensor* g = ggml_add(c, ggml_mul_mat(c, a.gates_w, xn), a.gates_b); // (H, N, B)
    g = ggml_sigmoid(c, g);
    // (H, N, B) -> (1, N, H, B) so it broadcasts along head_dim.
    g = ggml_cont(c, ggml_permute(c, ggml_reshape_4d(c, g, 1, H, N, B), 0, 2, 1, 3));
    out = ggml_mul(c, out, g);

    // "b h n d -> b n (h d)"
    out = ggml_cont(c, ggml_permute(c, out, 0, 2, 1, 3)); // (D, H, N, B)
    out = ggml_reshape_3d(c, out, D * H, N, B);
    return ggml_mul_mat(c, a.out_w, out);
}

// FeedForward branch. GELU is the EXACT erf form: torch nn.GELU defaults to
// approximate='none'. ggml_gelu is the tanh approximation and drifts.
ggml_tensor* bt_feedforward(ggml_context* c, const bt_ff& f, ggml_tensor* x) {
    ggml_tensor* h = bt_norm(c, x, f.gamma);
    h = ggml_add(c, ggml_mul_mat(c, f.w1, h), f.b1);
    h = ggml_gelu_erf(c, h);
    return ggml_add(c, ggml_mul_mat(c, f.w2, h), f.b2);
}

// Named intermediates, so one graph can serve every parity stage without the
// stage list and the forward drifting apart.
struct bt_stages {
    std::vector<std::pair<std::string, ggml_tensor*>> v;
    void add(const std::string& n, ggml_tensor* t) { v.emplace_back(n, t); }
    ggml_tensor* get(const std::string& n) const {
        for (const auto& p : v)
            if (p.first == n)
                return p.second;
        return nullptr;
    }
};

// PartialFTTransformer. `x` is ne (t, f, c); returns the same shape.
ggml_tensor* bt_partial_ft(ggml_context* c, const bt_partial& p, ggml_tensor* x, ggml_tensor* pos_f, ggml_tensor* pos_t,
                           int blk, bt_stages& sm) {
    const std::string pre = "blk" + std::to_string(blk) + "_";

    // (t, f, c) -> (c, f, t): FREQUENCY is the sequence axis, time the batch.
    ggml_tensor* h = ggml_cont(c, ggml_permute(c, x, 2, 1, 0, 3));

    ggml_tensor* br = bt_attention(c, p.attnF, h, pos_f);
    sm.add(pre + "attnF", br); // the BRANCH, matching the reference hook
    h = ggml_add(c, h, br);
    br = bt_feedforward(c, p.ffF, h);
    sm.add(pre + "ffF", br);
    h = ggml_add(c, h, br);

    // (c, f, t) -> (c, t, f): TIME is now the sequence axis.
    h = ggml_cont(c, ggml_permute(c, h, 0, 2, 1, 3));

    br = bt_attention(c, p.attnT, h, pos_t);
    sm.add(pre + "attnT", br);
    h = ggml_add(c, h, br);
    br = bt_feedforward(c, p.ffT, h);
    sm.add(pre + "ffT", br);
    h = ggml_add(c, h, br);

    // (c, t, f) -> (t, f, c)
    h = ggml_cont(c, ggml_permute(c, h, 2, 0, 1, 3));
    sm.add(pre + "partial", h);
    return h;
}

// Bind one attention sub-block's weights, e.g.
// prefix = "frontend.blocks.0.partial.attnF".
bool bt_bind_attn(core_gguf::tensor_map& tm, const std::string& prefix, int dim, bt_attn& a) {
    a.head_dim = 32; // head_dim is fixed at 32 model-wide (BeatThis(head_dim=32))
    a.n_head = dim / a.head_dim;
    a.gamma = core_gguf::require(tm, (prefix + ".norm.gamma").c_str(), "beat-this");
    a.qkv_w = core_gguf::require(tm, (prefix + ".to_qkv.weight").c_str(), "beat-this");
    a.gates_w = core_gguf::require(tm, (prefix + ".to_gates.weight").c_str(), "beat-this");
    a.gates_b = core_gguf::require(tm, (prefix + ".to_gates.bias").c_str(), "beat-this");
    a.out_w = core_gguf::require(tm, (prefix + ".to_out.0.weight").c_str(), "beat-this");
    return a.gamma && a.qkv_w && a.gates_w && a.gates_b && a.out_w;
}

bool bt_bind_ff(core_gguf::tensor_map& tm, const std::string& prefix, bt_ff& f) {
    f.gamma = core_gguf::require(tm, (prefix + ".net.0.gamma").c_str(), "beat-this");
    f.w1 = core_gguf::require(tm, (prefix + ".net.1.weight").c_str(), "beat-this");
    f.b1 = core_gguf::require(tm, (prefix + ".net.1.bias").c_str(), "beat-this");
    f.w2 = core_gguf::require(tm, (prefix + ".net.4.weight").c_str(), "beat-this");
    f.b2 = core_gguf::require(tm, (prefix + ".net.4.bias").c_str(), "beat-this");
    return f.gamma && f.w1 && f.b1 && f.w2 && f.b2;
}

} // namespace

extern "C" int beat_this_n_frames(int n_samples) {
    if (n_samples <= 0)
        return 0;
    // torchaudio center=true: 1 + n // hop
    return 1 + n_samples / BEAT_THIS_HOP;
}

extern "C" int beat_this_logmel(struct beat_this_context* ctx, const float* pcm, int n_samples, float* out) {
    if (!ctx || !pcm || n_samples <= 0 || !out)
        return 0;
    const int T = beat_this_n_frames(n_samples);
    const int n_fft = ctx->n_fft, hop = ctx->hop, M = ctx->mel_bins;
    const int n_freqs = n_fft / 2 + 1;
    if (ctx->fb_freqs != n_freqs || (int)ctx->mel_fb.size() != n_freqs * M) {
        fprintf(stderr, "beat_this: filterbank is %d x %d, expected %d x %d\n", ctx->fb_freqs,
                ctx->fb_freqs ? (int)(ctx->mel_fb.size() / ctx->fb_freqs) : 0, n_freqs, M);
        return 0;
    }

    // torchaudio's `normalized="frame_length"` divides by SQRT(n_fft), not
    // n_fft — the name is misleading. Verified empirically: the ratio between
    // normalized=False and normalized="frame_length" is exactly 32.0 at
    // n_fft=1024. Getting this wrong scales every mel bin and, because the
    // features are log1p(1000*x), does NOT cancel downstream.
    const float norm = 1.0f / std::sqrt((float)n_fft);

    std::vector<float> frame((size_t)n_fft), spec((size_t)2 * n_freqs), mag((size_t)n_freqs);
    const int half = n_fft / 2;

    for (int t = 0; t < T; t++) {
        // center=true: frame t is centred on sample t*hop, reflect-padded.
        const int start = t * hop - half;
        for (int i = 0; i < n_fft; i++) {
            const int s = start + i;
            const float v = (s >= 0 && s < n_samples) ? pcm[s] : pcm[reflect_index(s, n_samples)];
            frame[(size_t)i] = v * ctx->hann[(size_t)i];
        }
        core_fft::fft_radix2_wrapper(frame.data(), n_fft, spec.data());

        // power=1 => MAGNITUDE, not power. Squaring here is the single easiest
        // way to produce a spectrogram that looks right and tracks wrong.
        for (int k = 0; k < n_freqs; k++) {
            const float re = spec[(size_t)2 * k], im = spec[(size_t)2 * k + 1];
            mag[(size_t)k] = std::sqrt(re * re + im * im) * norm;
        }

        // Project onto the mel filterbank: fb is [n_freqs, n_mels] row-major,
        // so mel[m] = sum_k mag[k] * fb[k * n_mels + m].
        float* dst = out + (size_t)t * M;
        for (int m = 0; m < M; m++)
            dst[m] = 0.0f;
        for (int k = 0; k < n_freqs; k++) {
            const float v = mag[(size_t)k];
            if (v == 0.0f)
                continue;
            const float* frow = ctx->mel_fb.data() + (size_t)k * M;
            for (int m = 0; m < M; m++)
                dst[m] += v * frow[m];
        }
        for (int m = 0; m < M; m++)
            dst[m] = std::log1p(ctx->log_multiplier * dst[m]);
    }
    return T;
}

extern "C" struct beat_this_context* beat_this_init(const char* model_path, int n_threads) {
    if (!model_path)
        return nullptr;
    auto* ctx = new beat_this_context();
    ctx->n_threads = n_threads > 0 ? n_threads : 4;
    ctx->debug = crispasr_env::get("CRISPASR_BEAT_THIS_DEBUG") != nullptr;

    gguf_context* gctx = core_gguf::open_metadata(model_path);
    if (!gctx) {
        fprintf(stderr, "beat_this: cannot open %s\n", model_path);
        delete ctx;
        return nullptr;
    }
    ctx->sample_rate = (int)core_gguf::kv_u32(gctx, "beat-this.sample_rate", BEAT_THIS_SAMPLE_RATE);
    ctx->n_fft = (int)core_gguf::kv_u32(gctx, "beat-this.n_fft", BEAT_THIS_N_FFT);
    ctx->hop = (int)core_gguf::kv_u32(gctx, "beat-this.hop_length", BEAT_THIS_HOP);
    ctx->mel_bins = (int)core_gguf::kv_u32(gctx, "beat-this.mel_bins", BEAT_THIS_MEL_BINS);
    ctx->log_multiplier = core_gguf::kv_f32(gctx, "beat-this.log_multiplier", 1000.0f);
    core_gguf::free_metadata(gctx);

    ctx->backend = ggml_backend_cpu_init();
    if (!ctx->backend) {
        delete ctx;
        return nullptr;
    }
    if (!core_gguf::load_weights(model_path, ctx->backend, "beat-this", ctx->wl)) {
        beat_this_free(ctx);
        return nullptr;
    }

    ggml_tensor* fb = core_gguf::try_get(ctx->wl.tensors, "aux.mel_filterbank");
    if (!fb) {
        fprintf(stderr, "beat_this: model has no aux.mel_filterbank — reconvert with --filterbank\n");
        beat_this_free(ctx);
        return nullptr;
    }
    // ggml ne is reversed vs the [513,128] row-major source: ne[0]=128, ne[1]=513.
    ctx->fb_freqs = (int)fb->ne[1];
    const size_t n = (size_t)ggml_nelements(fb);
    ctx->mel_fb.resize(n);
    ggml_backend_tensor_get(fb, ctx->mel_fb.data(), 0, n * sizeof(float));

    build_hann_periodic(ctx->n_fft, ctx->hann);

    // Bind the stem weights. require() is loud on a miss: a silently-absent
    // weight would compute a plausible-looking activation from zeros.
    ctx->bn1d_scale = core_gguf::require(ctx->wl.tensors, "frontend.stem.bn1d.scale", "beat-this");
    ctx->bn1d_offset = core_gguf::require(ctx->wl.tensors, "frontend.stem.bn1d.offset", "beat-this");
    ctx->stem_w = core_gguf::require(ctx->wl.tensors, "frontend.stem.conv2d.weight", "beat-this");
    ctx->stem_b = core_gguf::require(ctx->wl.tensors, "frontend.stem.conv2d.bias", "beat-this");
    if (!ctx->bn1d_scale || !ctx->bn1d_offset || !ctx->stem_w || !ctx->stem_b) {
        beat_this_free(ctx);
        return nullptr;
    }

    // Three frontend blocks, dims 32 -> 64 -> 128 (heads 1 -> 2 -> 4).
    for (int i = 0; i < 3; i++) {
        const int dim = 32 << i;
        const std::string bp = "frontend.blocks." + std::to_string(i);
        bt_block& b = ctx->blocks[i];
        const bool ok = bt_bind_attn(ctx->wl.tensors, bp + ".partial.attnF", dim, b.partial.attnF) &&
                        bt_bind_ff(ctx->wl.tensors, bp + ".partial.ffF", b.partial.ffF) &&
                        bt_bind_attn(ctx->wl.tensors, bp + ".partial.attnT", dim, b.partial.attnT) &&
                        bt_bind_ff(ctx->wl.tensors, bp + ".partial.ffT", b.partial.ffT);
        b.conv_w = core_gguf::require(ctx->wl.tensors, (bp + ".conv2d.weight").c_str(), "beat-this");
        b.conv_b = core_gguf::require(ctx->wl.tensors, (bp + ".conv2d.bias").c_str(), "beat-this");
        if (!ok || !b.conv_w || !b.conv_b) {
            beat_this_free(ctx);
            return nullptr;
        }
    }

    if (ctx->debug)
        fprintf(stderr, "beat_this: sr=%d n_fft=%d hop=%d mel=%d fb=%dx%d\n", ctx->sample_rate, ctx->n_fft, ctx->hop,
                ctx->mel_bins, ctx->fb_freqs, (int)(n / (size_t)std::max(1, ctx->fb_freqs)));
    return ctx;
}

extern "C" void beat_this_free(struct beat_this_context* ctx) {
    if (!ctx)
        return;
    core_gguf::free_weights(ctx->wl);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

extern "C" int beat_this_sample_rate(const struct beat_this_context* ctx) {
    return ctx ? ctx->sample_rate : 0;
}

extern "C" float beat_this_tempo_bpm(const struct beat_this_event* ev, int n) {
    if (!ev || n < 2)
        return 0.0f;
    std::vector<float> iois;
    iois.reserve((size_t)n - 1);
    for (int i = 1; i < n; i++) {
        const float d = ev[i].time_s - ev[i - 1].time_s;
        if (d > 0.0f)
            iois.push_back(d);
    }
    if (iois.empty())
        return 0.0f;
    // Median, not mean: a single missed or doubled beat skews a mean badly,
    // and beat sequences routinely have both.
    std::sort(iois.begin(), iois.end());
    const float med = iois[iois.size() / 2];
    return med > 0.0f ? 60.0f / med : 0.0f;
}

// Stem forward: log-mel (T,128) -> (T, 32 freq, 32 ch).
//
// LAYOUT. The reference is torch (b, c, f, t); ggml is the reverse, so the
// stem output here is ne = (t, f, c). The log-mel arrives row-major (T,128),
// i.e. ne = (128, T) with mel fastest, and conv2d wants ne[0] = W = TIME — so
// it is transposed first.
//
// The conv kernel is stored as torch (oc, ic, kf, kt) row-major, which is
// already ggml ne = (kt, kf, ic, oc) — exactly ggml_conv_2d's expectation,
// with s0/p0 acting along time and s1/p1 along frequency. No permute needed.
//
// GELU IS THE EXACT (erf) FORM. torch nn.GELU defaults to approximate='none';
// ggml_gelu is the TANH approximation and would drift. See the GELU-variant
// entry in the common-divergence list in docs/contributing.md.
extern "C" int beat_this_debug_stage(struct beat_this_context* ctx, const float* logmel, int T, const char* stage,
                                     float* out, int max_out, int64_t* ne_out) {
    if (!ctx || !logmel || T <= 0 || !out || !stage)
        return 0;
    const int M = ctx->mel_bins;

    const size_t nodes = 1024;
    ggml_init_params ip = {nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true};
    ggml_context* c0 = ggml_init(ip);
    if (!c0)
        return 0;
    ggml_cgraph* gf = ggml_new_graph_custom(c0, nodes, false);

    // (128, T) as stored; ne[0] = mel.
    ggml_tensor* x = ggml_new_tensor_2d(c0, GGML_TYPE_F32, M, T);
    ggml_set_name(x, "logmel");
    ggml_set_input(x);

    // bn1d is per-FREQUENCY and applies BEFORE any conv, so it acts on the mel
    // axis while that axis is still ne[0].
    ggml_tensor* h = ggml_add(c0, ggml_mul(c0, x, ctx->bn1d_scale), ctx->bn1d_offset);

    // -> ne (T, 128, 1, 1): time fastest, as conv2d wants.
    h = ggml_cont(c0, ggml_transpose(c0, h));
    h = ggml_reshape_4d(c0, h, T, M, 1, 1);

    // stride (time=1, freq=4), pad (time=1, freq=0) -> (T, 32, 32, 1)
    h = ggml_conv_2d(c0, ctx->stem_w, h, /*s0*/ 1, /*s1*/ 4, /*p0*/ 1, /*p1*/ 0, /*d0*/ 1, /*d1*/ 1);
    h = ggml_add(c0, h, ggml_reshape_3d(c0, ctx->stem_b, 1, 1, ctx->stem_b->ne[0]));
    h = ggml_gelu_erf(c0, h);

    bt_stages sm;
    sm.add("stem", h);

    // Positions are just 0..N-1; the sequence length differs per axis and per
    // block (frequency halves each time, time never changes).
    ggml_tensor* pos_t = ggml_new_tensor_1d(c0, GGML_TYPE_I32, T);
    ggml_set_input(pos_t);
    ggml_tensor* pos_f[3];

    for (int i = 0; i < 3; i++) {
        const int64_t F = h->ne[1];
        pos_f[i] = ggml_new_tensor_1d(c0, GGML_TYPE_I32, F);
        ggml_set_input(pos_f[i]);

        h = ggml_reshape_3d(c0, h, h->ne[0], F, h->ne[2]);
        h = bt_partial_ft(c0, ctx->blocks[i].partial, h, pos_f[i], pos_t, i, sm);

        // conv (kt=3, kf=2) stride (t=1, f=2) pad (t=1, f=0): halves frequency,
        // doubles channels. Kernel is stored torch (oc, ic, kf, kt) row-major,
        // which is already ggml ne = (kt, kf, ic, oc) — no permute.
        h = ggml_reshape_4d(c0, h, h->ne[0], h->ne[1], h->ne[2], 1);
        h = ggml_conv_2d(c0, ctx->blocks[i].conv_w, h, 1, 2, 1, 0, 1, 1);
        h = ggml_add(c0, h, ggml_reshape_3d(c0, ctx->blocks[i].conv_b, 1, 1, ctx->blocks[i].conv_b->ne[0]));
        h = ggml_gelu_erf(c0, h);
        sm.add("blk" + std::to_string(i), h);
    }

    ggml_tensor* want = sm.get(stage);
    if (!want) {
        fprintf(stderr, "beat_this: unknown debug stage '%s'\n", stage);
        ggml_free(c0);
        return 0;
    }
    if (ne_out)
        for (int i = 0; i < 4; i++)
            ne_out[i] = want->ne[i];
    if ((int)ggml_nelements(want) > max_out) {
        fprintf(stderr, "beat_this: stage '%s' needs %d floats, buffer holds %d\n", stage, (int)ggml_nelements(want),
                max_out);
        ggml_free(c0);
        return 0;
    }

    ggml_set_output(want);
    ggml_build_forward_expand(gf, want);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    bool ok = ga && ggml_gallocr_alloc_graph(ga, gf);
    int n = 0;
    if (ok) {
        ggml_backend_tensor_set(x, logmel, 0, (size_t)T * M * sizeof(float));
        // Only the stages upstream of `want` are in the graph, so the later
        // blocks' position inputs may be unallocated — writing to those would
        // be a null-deref, not a no-op.
        std::vector<int32_t> p((size_t)T);
        auto fill_pos = [&](ggml_tensor* t) {
            if (!t->data)
                return;
            const int n_p = (int)t->ne[0];
            for (int j = 0; j < n_p; j++)
                p[(size_t)j] = j;
            ggml_backend_tensor_set(t, p.data(), 0, (size_t)n_p * sizeof(int32_t));
        };
        fill_pos(pos_t);
        for (int i = 0; i < 3; i++)
            fill_pos(pos_f[i]);
        ok = ggml_backend_graph_compute(ctx->backend, gf) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        n = (int)ggml_nelements(want);
        ggml_backend_tensor_get(want, out, 0, (size_t)n * sizeof(float));
        if (ctx->debug)
            fprintf(stderr, "beat_this: %s ne=(%lld,%lld,%lld,%lld)\n", stage, (long long)want->ne[0],
                    (long long)want->ne[1], (long long)want->ne[2], (long long)want->ne[3]);
    }
    if (ga)
        ggml_gallocr_free(ga);
    ggml_free(c0);
    return n;
}

extern "C" int beat_this_debug_stem(struct beat_this_context* ctx, const float* logmel, int T, float* out) {
    return beat_this_debug_stage(ctx, logmel, T, "stem", out, T * 32 * 32, nullptr);
}

extern "C" int beat_this_track(struct beat_this_context* ctx, const float* pcm, int n_samples,
                               struct beat_this_event* out, int max_events) {
    (void)ctx;
    (void)pcm;
    (void)n_samples;
    (void)out;
    (void)max_events;
    // TODO(§251b-1): ggml graph + 1500-frame windowing (border 6, keep_first)
    // + peak-picking. The front end above is done and validated; this is the
    // remaining half. Returning -1 rather than 0 so a caller cannot mistake
    // "not implemented" for "no beats found".
    fprintf(stderr, "beat_this: track() not implemented yet — the ggml graph is pending (§251b-1)\n");
    return -1;
}
