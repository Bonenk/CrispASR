// src/mel_band_roformer.cpp — Mel-Band RoFormer source separation (§248).
//
// Phase 1 (this file): GGUF loader (hparams + baked band-layout aux arrays +
// weight map), the STFT front-end that matches torch.stft(center=True) and the
// binary-band gather, and the front-end half of the diff harness
// (input_audio -> stft_packed -> band_gathered). The transformer graph, mask
// estimator, scatter-average, complex mask and iSTFT are Phase 2 — built and
// validated stage-by-stage against the reference fixture (ref_mbr.gguf), first
// divergence = the bug.
//
// Blueprint: MIT lucidrains/BS-RoFormer. Weights: KimberleyJSN/melbandroformer
// (MIT). Reference pinned at bs-roformer==0.3.10. See docs/mel-band-roformer/PLAN.md.

#include "mel_band_roformer.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include "core/fft.h"         // fft_radix2_wrapper (r2c, interleaved full spectrum)
#include "core/gguf_loader.h" // core_gguf::{open_metadata,kv_u32,load_weights}

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Model context
// ---------------------------------------------------------------------------

struct mel_band_roformer_hparams {
    int dim = 384;
    int depth = 6;
    int heads = 8;
    int dim_head = 64;
    int num_bands = 60;
    int num_stems = 1;
    int time_transformer_depth = 1;
    int freq_transformer_depth = 1;
    int mask_estimator_depth = 2;
    int stereo = 1;
    int audio_channels = 2;
    int sample_rate = 44100;
    int n_fft = 2048;
    int hop = 441;
    int win = 2048;
    int normalized = 0;
};

struct mel_band_roformer_context {
    mel_band_roformer_hparams hp;
    mel_band_roformer_params params{};

    ggml_backend_t backend = nullptr;
    core_gguf::WeightLoad weights;

    // Baked band layout (from the converter's aux.* int32 tensors).
    std::vector<int32_t> freq_indices;       // length = sum over bands of 2*nfreq*ch? no: N gather idx
    std::vector<int32_t> num_bands_per_freq; // length = n_freqs (overlap denominator)
    std::vector<int32_t> num_freqs_per_band; // length = num_bands
    // Per-band packed-input width = 2 (complex) * num_freqs_per_band[b] * channels.
    std::vector<int> band_width;

    std::vector<std::string> source_names_storage;
    std::vector<const char*> source_names_c;

    int n_freqs() const { return hp.n_fft / 2 + 1; }
};

// ---------------------------------------------------------------------------
// STFT front-end (CPU) — matches torch.stft(center=True, Hann-periodic)
// ---------------------------------------------------------------------------

namespace {

// Hann periodic window of length N (torch.hann_window default: periodic=True):
//   w[n] = 0.5 - 0.5*cos(2*pi*n / N)
void hann_periodic(int N, std::vector<float>& w) {
    w.resize(N);
    for (int n = 0; n < N; n++)
        w[n] = 0.5f - 0.5f * std::cos(2.0f * (float)M_PI * (float)n / (float)N);
}

// torch center=True reflect-pad by n_fft/2 on both ends (reflect excludes the
// edge sample, matching numpy/torch 'reflect').
void reflect_pad(const float* x, int n, int pad, std::vector<float>& out) {
    out.resize((size_t)n + 2 * pad);
    for (int i = 0; i < pad; i++)
        out[i] = x[pad - i]; // reflect: x[pad], x[pad-1], ... excludes x[0] mirror center
    for (int i = 0; i < n; i++)
        out[pad + i] = x[i];
    for (int i = 0; i < pad; i++)
        out[pad + n + i] = x[n - 2 - i];
}

// Frame count for center=True: 1 + n_samples/hop (padded length n + n_fft,
// frames = 1 + (padded - n_fft)/hop = 1 + n/hop).
int stft_n_frames(int n_samples, int hop) {
    return 1 + n_samples / hop;
}

// One channel -> complex STFT. Output `spec` is [n_freqs][T][2] laid out as
// spec[(f*T + t)*2 + {0,1}]. n_freqs = n_fft/2+1.
void stft_one_channel(const float* x, int n, int n_fft, int hop, const std::vector<float>& window, int T, int n_freqs,
                      std::vector<float>& spec) {
    const int pad = n_fft / 2;
    std::vector<float> xp;
    reflect_pad(x, n, pad, xp);
    const int padded = (int)xp.size();

    spec.assign((size_t)n_freqs * T * 2, 0.0f);
    std::vector<float> frame(n_fft), full(2 * n_fft);
    for (int t = 0; t < T; t++) {
        const int start = t * hop;
        for (int i = 0; i < n_fft; i++) {
            const int idx = start + i;
            frame[i] = (idx < padded ? xp[idx] : 0.0f) * window[i];
        }
        core_fft::fft_radix2_wrapper(frame.data(), n_fft, full.data()); // full[2k]=re, full[2k+1]=im
        for (int f = 0; f < n_freqs; f++) {
            spec[((size_t)f * T + t) * 2 + 0] = full[2 * f + 0];
            spec[((size_t)f * T + t) * 2 + 1] = full[2 * f + 1];
        }
    }
}

// Build the packed STFT `(f*s, t, 2)` frequency-major / channel-fastest:
//   packed[((f*ch + s)*T + t)*2 + c]. `chan_spec[s]` is one channel's
// [n_freqs][T][2] buffer.
void pack_stft(const std::vector<std::vector<float>>& chan_spec, int n_freqs, int T, int channels,
               std::vector<float>& packed) {
    packed.assign((size_t)n_freqs * channels * T * 2, 0.0f);
    for (int f = 0; f < n_freqs; f++)
        for (int s = 0; s < channels; s++) {
            const int row = f * channels + s;
            for (int t = 0; t < T; t++) {
                packed[((size_t)row * T + t) * 2 + 0] = chan_spec[s][((size_t)f * T + t) * 2 + 0];
                packed[((size_t)row * T + t) * 2 + 1] = chan_spec[s][((size_t)f * T + t) * 2 + 1];
            }
        }
}

// Gather the packed rows named by freq_indices and fold complex into the
// feature axis: band_gathered[t][k*2 + c] where k indexes freq_indices.
// Output shape (T, N*2) with N = freq_indices.size().
void band_gather(const std::vector<float>& packed, const std::vector<int32_t>& freq_indices, int T,
                 std::vector<float>& out) {
    const int N = (int)freq_indices.size();
    out.assign((size_t)T * N * 2, 0.0f);
    for (int t = 0; t < T; t++)
        for (int k = 0; k < N; k++) {
            const int row = freq_indices[k];
            out[((size_t)t * N + k) * 2 + 0] = packed[((size_t)row * T + t) * 2 + 0];
            out[((size_t)t * N + k) * 2 + 1] = packed[((size_t)row * T + t) * 2 + 1];
        }
}

// Read an int32 aux tensor from the loaded weights into a host vector.
bool read_i32(core_gguf::WeightLoad& wl, const char* name, std::vector<int32_t>& out) {
    auto it = wl.tensors.find(name);
    if (it == wl.tensors.end() || !it->second)
        return false;
    ggml_tensor* t = it->second;
    const int64_t n = ggml_nelements(t);
    out.resize((size_t)n);
    ggml_backend_tensor_get(t, out.data(), 0, (size_t)n * sizeof(int32_t));
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

mel_band_roformer_params mel_band_roformer_default_params(void) {
    mel_band_roformer_params p;
    p.n_threads = 0;
    p.use_gpu = false; // CPU front-end in Phase 1
    p.gpu_device = 0;
    return p;
}

mel_band_roformer_context* mel_band_roformer_init_from_file(const char* model_path, mel_band_roformer_params params) {
    gguf_context* meta = core_gguf::open_metadata(model_path);
    if (!meta) {
        fprintf(stderr, "mel_band_roformer: cannot open GGUF '%s'\n", model_path);
        return nullptr;
    }
    const std::string arch = core_gguf::kv_str(meta, "general.architecture", "");
    if (arch != "mel-band-roformer") {
        fprintf(stderr, "mel_band_roformer: GGUF arch is '%s', expected 'mel-band-roformer'\n", arch.c_str());
        core_gguf::free_metadata(meta);
        return nullptr;
    }

    auto* ctx = new mel_band_roformer_context();
    ctx->params = params;
    auto& hp = ctx->hp;
    hp.dim = (int)core_gguf::kv_u32(meta, "mel-band-roformer.dim", hp.dim);
    hp.depth = (int)core_gguf::kv_u32(meta, "mel-band-roformer.depth", hp.depth);
    hp.heads = (int)core_gguf::kv_u32(meta, "mel-band-roformer.heads", hp.heads);
    hp.dim_head = (int)core_gguf::kv_u32(meta, "mel-band-roformer.dim_head", hp.dim_head);
    hp.num_bands = (int)core_gguf::kv_u32(meta, "mel-band-roformer.num_bands", hp.num_bands);
    hp.num_stems = (int)core_gguf::kv_u32(meta, "mel-band-roformer.num_stems", hp.num_stems);
    hp.time_transformer_depth =
        (int)core_gguf::kv_u32(meta, "mel-band-roformer.time_transformer_depth", hp.time_transformer_depth);
    hp.freq_transformer_depth =
        (int)core_gguf::kv_u32(meta, "mel-band-roformer.freq_transformer_depth", hp.freq_transformer_depth);
    hp.mask_estimator_depth =
        (int)core_gguf::kv_u32(meta, "mel-band-roformer.mask_estimator_depth", hp.mask_estimator_depth);
    hp.stereo = (int)core_gguf::kv_u32(meta, "mel-band-roformer.stereo", hp.stereo);
    hp.audio_channels = (int)core_gguf::kv_u32(meta, "mel-band-roformer.audio_channels", hp.audio_channels);
    hp.sample_rate = (int)core_gguf::kv_u32(meta, "mel-band-roformer.sample_rate", hp.sample_rate);
    hp.n_fft = (int)core_gguf::kv_u32(meta, "mel-band-roformer.stft_n_fft", hp.n_fft);
    hp.hop = (int)core_gguf::kv_u32(meta, "mel-band-roformer.stft_hop_length", hp.hop);
    hp.win = (int)core_gguf::kv_u32(meta, "mel-band-roformer.stft_win_length", hp.win);
    hp.normalized = (int)core_gguf::kv_u32(meta, "mel-band-roformer.stft_normalized", hp.normalized);
    core_gguf::free_metadata(meta);

    ctx->backend = ggml_backend_cpu_init();
    if (!ctx->backend) {
        fprintf(stderr, "mel_band_roformer: ggml_backend_cpu_init failed\n");
        delete ctx;
        return nullptr;
    }
    if (!core_gguf::load_weights(model_path, ctx->backend, "mel_band_roformer", ctx->weights)) {
        fprintf(stderr, "mel_band_roformer: failed to load weights from '%s'\n", model_path);
        mel_band_roformer_free(ctx);
        return nullptr;
    }

    if (!read_i32(ctx->weights, "aux.freq_indices", ctx->freq_indices) ||
        !read_i32(ctx->weights, "aux.num_bands_per_freq", ctx->num_bands_per_freq) ||
        !read_i32(ctx->weights, "aux.num_freqs_per_band", ctx->num_freqs_per_band)) {
        fprintf(stderr, "mel_band_roformer: GGUF missing baked aux.* band-layout arrays\n");
        mel_band_roformer_free(ctx);
        return nullptr;
    }
    ctx->band_width.resize(ctx->num_freqs_per_band.size());
    for (size_t b = 0; b < ctx->num_freqs_per_band.size(); b++)
        ctx->band_width[b] = 2 * ctx->num_freqs_per_band[b] * hp.audio_channels;

    // Stem names: vocals model emits {vocals, other}. Generic fallback stemN.
    if (hp.num_stems == 1) {
        ctx->source_names_storage = {"vocals", "other"};
    } else {
        for (int i = 0; i < hp.num_stems; i++)
            ctx->source_names_storage.push_back("stem" + std::to_string(i));
    }
    for (auto& s : ctx->source_names_storage)
        ctx->source_names_c.push_back(s.c_str());

    return ctx;
}

void mel_band_roformer_free(mel_band_roformer_context* ctx) {
    if (!ctx)
        return;
    if (ctx->weights.buf)
        ggml_backend_buffer_free(ctx->weights.buf);
    if (ctx->weights.ctx)
        ggml_free(ctx->weights.ctx);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

int mel_band_roformer_sample_rate(const mel_band_roformer_context* ctx) {
    return ctx ? ctx->hp.sample_rate : 0;
}
int mel_band_roformer_n_sources(const mel_band_roformer_context* ctx) {
    return ctx ? (int)ctx->source_names_storage.size() : 0;
}
const char* mel_band_roformer_source_name(const mel_band_roformer_context* ctx, int idx) {
    if (!ctx || idx < 0 || idx >= (int)ctx->source_names_storage.size())
        return nullptr;
    return ctx->source_names_storage[idx].c_str();
}

mel_band_roformer_result* mel_band_roformer_separate(mel_band_roformer_context* ctx, const float* pcm, int n_samples,
                                                     int in_channels) {
    (void)pcm;
    (void)n_samples;
    (void)in_channels;
    // Phase 2: transformer graph + mask + iSTFT. The front-end (STFT + gather)
    // is implemented and diff-validated; the decode graph is built next,
    // stage-by-stage against ref_mbr.gguf. Return null until then rather than
    // emit unvalidated audio.
    fprintf(stderr, "mel_band_roformer_separate: forward graph not yet implemented (Phase 2)\n");
    (void)ctx;
    return nullptr;
}

void mel_band_roformer_result_free(mel_band_roformer_result* r) {
    if (!r)
        return;
    if (r->sources) {
        for (int s = 0; s < r->n_sources; s++)
            free(r->sources[s]);
        free(r->sources);
    }
    free(r->source_names);
    free(r);
}

// ---------------------------------------------------------------------------
// Diff harness — Phase 1: front-end stages (input_audio, stft_packed,
// band_gathered) vs the reference fixture. Later stages are reported PENDING.
// ---------------------------------------------------------------------------

namespace {

double cosine(const float* a, const float* b, int64_t n) {
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

double max_abs_diff(const float* a, const float* b, int64_t n) {
    double m = 0;
    for (int64_t i = 0; i < n; i++)
        m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

double l2_norm(const float* a, int64_t n) {
    double s = 0;
    for (int64_t i = 0; i < n; i++)
        s += (double)a[i] * a[i];
    return std::sqrt(s);
}

// Fetch a reference tensor's data as a flat float vector.
bool ref_get(core_gguf::WeightLoad& rw, const char* name, std::vector<float>& out, int64_t& nelem) {
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

int mel_band_roformer_diff(const char* model_gguf, const char* ref_gguf, const char* audio_wav, int verbosity) {
    (void)audio_wav; // Phase 1 reads input_audio FROM the ref (input-aligned).

    mel_band_roformer_context* ctx = mel_band_roformer_init_from_file(model_gguf, mel_band_roformer_default_params());
    if (!ctx) {
        fprintf(stderr, "mbr_diff: failed to load model %s\n", model_gguf);
        return 2;
    }
    core_gguf::WeightLoad rw;
    if (!core_gguf::load_weights(ref_gguf, ctx->backend, "mbr_ref", rw)) {
        fprintf(stderr, "mbr_diff: failed to load reference %s\n", ref_gguf);
        mel_band_roformer_free(ctx);
        return 2;
    }

    const auto& hp = ctx->hp;
    const int channels = hp.audio_channels;
    const int n_freqs = ctx->n_freqs();

    int n_fail = 0;
    const double COS_MIN = 0.9995;

    auto report = [&](const char* stage, const std::vector<float>& mine, const std::vector<float>& ref) {
        const int64_t n = (int64_t)std::min(mine.size(), ref.size());
        const double cos = cosine(mine.data(), ref.data(), n);
        const double mad = max_abs_diff(mine.data(), ref.data(), n);
        const bool ok = cos >= COS_MIN && mine.size() == ref.size();
        if (!ok)
            n_fail++;
        if (verbosity >= 1 || !ok) {
            fprintf(stderr, "  %-16s %s cos=%.6f max_abs=%.3e  (mine=%zu ref=%zu)%s\n", stage, ok ? "PASS" : "FAIL",
                    cos, mad, mine.size(), ref.size(),
                    verbosity >= 2 ? ("  |mine|=" + std::to_string(l2_norm(mine.data(), n)) +
                                      " |ref|=" + std::to_string(l2_norm(ref.data(), n)))
                                         .c_str()
                                   : "");
        }
    };

    // --- input_audio (2, T_samp), row-major s-major ---
    std::vector<float> in_audio;
    int64_t in_n = 0;
    if (!ref_get(rw, "input_audio", in_audio, in_n)) {
        fprintf(stderr, "mbr_diff: reference has no input_audio stage — re-dump with the updated dumper\n");
        mel_band_roformer_free(ctx);
        return 2;
    }
    const int T_samp = (int)(in_n / channels);
    const int T = stft_n_frames(T_samp, hp.hop);

    // --- run our STFT front-end on the reference input ---
    std::vector<float> window;
    hann_periodic(hp.win, window);
    std::vector<std::vector<float>> chan_spec(channels);
    for (int s = 0; s < channels; s++)
        stft_one_channel(in_audio.data() + (size_t)s * T_samp, T_samp, hp.n_fft, hp.hop, window, T, n_freqs,
                         chan_spec[s]);

    std::vector<float> packed;
    pack_stft(chan_spec, n_freqs, T, channels, packed);
    std::vector<float> gathered;
    band_gather(packed, ctx->freq_indices, T, gathered);

    fprintf(stderr, "mel_band_roformer diff (T_samp=%d, T=%d, n_freqs=%d, channels=%d, N_gather=%zu):\n", T_samp, T,
            n_freqs, channels, ctx->freq_indices.size());

    // --- freq_indices (integer membership) as float compare ---
    {
        std::vector<float> mine_fi(ctx->freq_indices.begin(), ctx->freq_indices.end());
        std::vector<float> ref_fi;
        int64_t nn = 0;
        if (ref_get(rw, "freq_indices", ref_fi, nn))
            report("freq_indices", mine_fi, ref_fi);
    }
    // --- stft_packed (f*s, T, 2) ---
    {
        std::vector<float> ref_sp;
        int64_t nn = 0;
        if (ref_get(rw, "stft_packed", ref_sp, nn))
            report("stft_packed", packed, ref_sp);
    }
    // --- band_gathered (T, N*2) ---
    {
        std::vector<float> ref_bg;
        int64_t nn = 0;
        if (ref_get(rw, "band_gathered", ref_bg, nn))
            report("band_gathered", gathered, ref_bg);
    }

    fprintf(stderr, "  [PENDING] band_split_out, layer*_time/freq, mask_raw, output_vocals — Phase 2 graph.\n");

    if (rw.buf)
        ggml_backend_buffer_free(rw.buf);
    if (rw.ctx)
        ggml_free(rw.ctx);
    mel_band_roformer_free(ctx);
    fprintf(stderr, "mel_band_roformer diff: %d front-end stage(s) FAILED.\n", n_fail);
    return n_fail == 0 ? 0 : 1;
}
