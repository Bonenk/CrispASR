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
#include <vector>

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
extern "C" int beat_this_debug_stem(struct beat_this_context* ctx, const float* logmel, int T, float* out) {
    if (!ctx || !logmel || T <= 0 || !out)
        return 0;
    const int M = ctx->mel_bins;

    const size_t nodes = 256;
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

    ggml_set_output(h);
    ggml_build_forward_expand(gf, h);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    bool ok = ga && ggml_gallocr_alloc_graph(ga, gf);
    int n = 0;
    if (ok) {
        ggml_backend_tensor_set(x, logmel, 0, (size_t)T * M * sizeof(float));
        ok = ggml_backend_graph_compute(ctx->backend, gf) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        n = (int)ggml_nelements(h);
        ggml_backend_tensor_get(h, out, 0, (size_t)n * sizeof(float));
        if (ctx->debug)
            fprintf(stderr, "beat_this: stem ne=(%lld,%lld,%lld,%lld)\n", (long long)h->ne[0], (long long)h->ne[1],
                    (long long)h->ne[2], (long long)h->ne[3]);
    }
    if (ga)
        ggml_gallocr_free(ga);
    ggml_free(c0);
    return n;
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
