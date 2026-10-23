// Native, uncached voice-reference A/B. Includes the actual private runtime.
#define GGML_MAX_NAME 128
#ifdef OV_ENCODER_BASELINE
#include "baseline.cpp"
#else
#include "../../src/omnivoice.cpp"
#endif
#include <ctime>
#include <fstream>
static void knob(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: encoder-bench TOKENIZER REF.wav OUTPUT_PREFIX "
                        "cache|fastconv|graph|fused|hybrid|all|cpu|defaults|full-bake|precise|precise-all|promoted|"
                        "combined|first-cpu|first-default|baseline\n");
        return 2;
    }
    const std::string mode(argv[4]);
    const char* names[] = {"CACHE", "FASTCONV", "GRAPH", "FUSED", "HYBRID", "SEMANTIC_F32"};
    if (mode != "cache" && mode != "fastconv" && mode != "graph" && mode != "fused" && mode != "hybrid" &&
        mode != "all" && mode != "cpu" && mode != "defaults" && mode != "full-bake" && mode != "precise" &&
        mode != "precise-all" && mode != "promoted" && mode != "combined" && mode != "first-cpu" &&
        mode != "first-default" && mode != "baseline")
        return 2;
    knob("CRISPASR_TTS_REF_CACHE", "0");
    knob("CRISPASR_OMNIVOICE_VOICE_CACHE", "0");
    knob("CRISPASR_OMNIVOICE_CODEC_GPU", "0");
    knob("CRISPASR_OMNIVOICE_CODEC_SPLIT", "0");
    for (auto* name : names)
        knob((std::string("CRISPASR_OMNIVOICE_ENCODER_") + name).c_str(), "0");
    ggml_backend_load_all();
    auto* ctx = new omnivoice_context;
    ctx->use_gpu = true;
    ctx->n_threads = 4;
    ctx->verbosity = 0;
    ctx->backend = crispasr_init_gpu_backend();
    if (!ctx->backend)
        ctx->backend = core_cpu_backend::init();
    if (!load_tokenizer(ctx, argv[1]))
        return 3;
    int order[] = {0, 1, 0, 1, 1, 0, 0, 1};
    if (env_bool("CRISPASR_OMNIVOICE_ENCODER_BENCH_REVERSE"))
        for (auto& arm : order)
            arm = 1 - arm;
    const bool first = mode == "first-cpu" || mode == "first-default";
    if (first)
        order[0] = mode == "first-default";
    for (int i = 0; i < (first ? 1 : mode == "baseline" ? 2 : 8); i++) {
        for (auto* name : names) {
            std::string lower(name);
            std::transform(lower.begin(), lower.end(), lower.begin(),
                           [](unsigned char c) { return (char)std::tolower(c); });
            bool on = order[i] && ((mode == "all" && lower != "semantic_f32") || mode == lower ||
                                   (mode == "full-bake" && lower == "fastconv") ||
                                   (mode == "cpu" && (lower == "cache" || lower == "fastconv")));
            if (mode == "precise" || mode == "precise-all")
                on = lower == "cache" || lower == "fastconv" ||
                     (order[i] && (mode == "precise-all" || lower == "hybrid" || lower == "semantic_f32"));
            if (mode == "combined")
                on = lower == "cache" || lower == "fastconv" || lower == "hybrid" || lower == "semantic_f32" ||
                     (order[i] && (lower == "graph" || lower == "fused"));
            if ((mode == "promoted" && !order[i]) || mode == "first-cpu")
                on = lower == "cache" || lower == "fastconv";
            const auto key = std::string("CRISPASR_OMNIVOICE_ENCODER_") + name;
            if ((mode == "defaults" || mode == "promoted" || mode == "first-default") && order[i]) {
#ifdef _WIN32
                _putenv_s(key.c_str(), "");
#else
                unsetenv(key.c_str());
#endif
            } else
                knob(key.c_str(), on ? (mode == "full-bake" ? "2" : "1") : "0");
        }
        std::printf("ARM i=%d variant=%s on=%d warmup=%d\n", i, mode.c_str(), order[i], i < 2);
        std::fflush(stdout);
        auto t = std::chrono::steady_clock::now();
        auto cpu = std::clock();
        if (omnivoice_set_voice_prompt(ctx, argv[2], "Synthetic reference.") != 0 || ctx->ref_T < 1 ||
            ctx->ref_audio_codes.size() != (size_t)ctx->ref_T * 8)
            return 4;
        double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count(),
               cpus = (double)(std::clock() - cpu) / CLOCKS_PER_SEC;
#ifndef OV_ENCODER_BASELINE
        if (order[i] && (mode == "hybrid" || mode == "all" || mode == "precise" || mode == "precise-all" ||
                         mode == "promoted" || mode == "combined" || mode == "first-default")) {
            if (!ctx->semantic_encoder.loaded || core_cpu_backend::is_cpu(ctx->semantic_encoder.backend)) {
                std::fprintf(stderr, "FAIL requested hybrid GPU stage did not run on GPU\n");
                return 6;
            }
            std::printf("SEMANTIC_BACKEND %s\n", ggml_backend_name(ctx->semantic_encoder.backend));
        }
#endif
        std::printf("ENCODE i=%d wall=%.9f cpu=%.9f frames=%d codes=%zu\n", i, wall, cpus, ctx->ref_T,
                    ctx->ref_audio_codes.size());
        std::fflush(stdout);
#ifndef OV_ENCODER_BASELINE
        size_t host_bytes = ctx->tokenizer.enc_posconv.size() * sizeof(float);
        for (auto& v : ctx->tokenizer.enc_values)
            host_bytes += v.second.size() * sizeof(float);
        for (auto& v : ctx->tokenizer.enc_norms)
            host_bytes += v.second.size() * sizeof(float);
        auto bytes = [](ggml_backend_buffer_t b) -> size_t { return b ? ggml_backend_buffer_get_size(b) : 0; };
        std::printf("ENCODER_BUFFERS host_cache=%zu fastconv=%zu projections=%zu semantic_gpu=%zu\n", host_bytes,
                    bytes(ctx->tokenizer.enc_fc.buf), bytes(ctx->tokenizer.enc_proj.buf),
                    bytes(ctx->semantic_encoder.buf_w));
        std::printf("SEMANTIC_CACHES fastconv=%zu projections=%zu positional_host=%zu\n",
                    bytes(ctx->semantic_encoder.enc_fc.buf), bytes(ctx->semantic_encoder.enc_proj.buf),
                    ctx->semantic_encoder.enc_posconv.size() * sizeof(float));
#endif
        std::ofstream f(std::string(argv[3]) + "-" + std::to_string(i) + ".codes", std::ios::binary);
        f.write((const char*)ctx->ref_audio_codes.data(), ctx->ref_audio_codes.size() * sizeof(int32_t));
        if (!f)
            return 5;
    }
    omnivoice_free(ctx);
}
