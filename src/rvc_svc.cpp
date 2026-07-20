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
