// Public native API benchmark. Compile once per engine against its own libraries:
// define CRISPASR_BENCH_OVCPP for omnivoice.cpp, otherwise use CrispASR.
// Both expose an omnivoice.h; choose the appropriate include path at compile time.
#include "omnivoice.h"
#ifndef CRISPASR_BENCH_OVCPP
#include "ggml-backend.h"
#endif
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    if (argc != 7 && argc != 8) {
        std::fprintf(stderr, "usage: engine-bench MODEL CODEC TEXT DURATION REPEATS OUTPUT_PREFIX "
                             "[--paired|--paired-cfg|--paired-proj|--paired-attn|--load-only]\n");
        return 2;
    }
    const bool paired = argc == 8 && std::string(argv[7]) == "--paired";
    const bool paired_cfg = argc == 8 && std::string(argv[7]) == "--paired-cfg";
    const bool paired_proj = argc == 8 && std::string(argv[7]) == "--paired-proj";
    const bool paired_attn = argc == 8 && std::string(argv[7]) == "--paired-attn";
    const bool load_only = argc == 8 && std::string(argv[7]) == "--load-only";
    if (argc == 8 && !paired && !paired_cfg && !paired_proj && !paired_attn && !load_only)
        return 2;
#ifdef CRISPASR_BENCH_OVCPP
    if (paired || paired_cfg || paired_proj || paired_attn)
        return 2;
#else
    if (paired) {
        setenv("CRISPASR_OMNIVOICE_CODEC_SPLIT", "0", 1);
        setenv("CRISPASR_OMNIVOICE_CODEC_DECOMP", "0", 1);
    }
#endif
    const double duration = std::atof(argv[4]);
    const int repeats = std::atoi(argv[5]);
    if (!(duration > 0) || repeats < 2 || ((paired || paired_cfg || paired_proj || paired_attn) && repeats != 8))
        return 2;
    const auto load_cpu = std::clock();
    double start = now();
#ifdef CRISPASR_BENCH_OVCPP
    ov_init_params p;
    ov_init_default_params(&p);
    p.model_path = argv[1];
    p.codec_path = argv[2];
    p.device = "MTL0";
    auto* ctx = ov_init(&p);
    if (!ctx) {
        std::fprintf(stderr, "%s\n", ov_last_error());
        return 3;
    }
#else
    ggml_backend_load_all();
    auto p = omnivoice_context_default_params();
    p.n_threads = 4;
    p.use_gpu = true;
    p.flash_attn = true;
    p.seed = 42;
    p.num_steps = 32;
    auto* ctx = omnivoice_init_from_file(argv[1], p);
    if (!ctx || omnivoice_set_tokenizer_path(ctx, argv[2])) {
        omnivoice_free(ctx);
        return 3;
    }
    omnivoice_set_target_duration(ctx, (float)duration);
    if (omnivoice_set_language(ctx, "en"))
        return 3;
#endif
    std::printf("LOAD %.6f\n", now() - start);
    std::printf("LOAD_CPU %.6f\n", (double)(std::clock() - load_cpu) / CLOCKS_PER_SEC);
    std::fflush(stdout);
    if (load_only) {
#ifdef CRISPASR_BENCH_OVCPP
        ov_free(ctx);
#else
        omnivoice_free(ctx);
#endif
        return 0;
    }
    for (int i = 0; i < repeats; i++) {
#ifndef CRISPASR_BENCH_OVCPP
        if (paired) {
            const int order[] = {0, 1, 0, 1, 1, 0, 0, 1};
            const bool resident = order[i] != 0;
            setenv("CRISPASR_OMNIVOICE_CODEC_SPLIT", resident ? "1" : "0", 1);
            setenv("CRISPASR_OMNIVOICE_CODEC_RESIDENT", resident ? "1" : "0", 1);
            setenv("CRISPASR_OMNIVOICE_CODEC_DECOMP", resident ? "1" : "0", 1);
            std::printf("ARM %d resident=%d warmup=%d\n", i, resident, i < 2);
        }
        if (paired_cfg) {
            const int order[] = {0, 1, 0, 1, 1, 0, 0, 1};
            setenv("CRISPASR_OMNIVOICE_UNIFIED_CFG", order[i] ? "1" : "0", 1);
            std::printf("ARM %d unified_cfg=%d warmup=%d\n", i, order[i], i < 2);
        }
        if (paired_proj) {
            const int order[] = {0, 1, 0, 1, 1, 0, 0, 1};
            setenv("CRISPASR_OMNIVOICE_PACKED_PROJ", order[i] ? "1" : "0", 1);
            std::printf("ARM %d packed_proj=%d warmup=%d\n", i, order[i], i < 2);
        }
        if (paired_attn) {
            const int order[] = {0, 1, 0, 1, 1, 0, 0, 1};
            setenv("CRISPASR_OMNIVOICE_ATTN_VIEWS", order[i] ? "1" : "0", 1);
            std::printf("ARM %d attn_views=%d warmup=%d\n", i, order[i], i < 2);
        }
#endif
        start = now();
        const auto cpu = std::clock();
        int ns = 0;
        float* pcm = nullptr;
#ifdef CRISPASR_BENCH_OVCPP
        ov_tts_params q;
        ov_tts_default_params(&q);
        q.text = argv[3];
        q.lang = "en";
        q.T_override = ov_duration_sec_to_tokens(ctx, (float)duration);
        q.mg_seed = 42;
        q.mg_num_step = 32;
        ov_audio audio = {};
        if (ov_synthesize(ctx, &q, &audio) != OV_STATUS_OK) {
            std::fprintf(stderr, "%s\n", ov_last_error());
            ov_free(ctx);
            return 4;
        }
        pcm = audio.samples;
        ns = audio.n_samples;
#else
        omnivoice_set_seed(ctx, 42);
        int nc = 0;
        auto* codes = omnivoice_synthesize_codes(ctx, argv[3], &nc);
        if (!codes || nc <= 0) {
            omnivoice_codes_free(codes);
            omnivoice_free(ctx);
            return 4;
        }
        const double gen = now() - start;
        const double decode_start = now();
        pcm = omnivoice_decode_codes(ctx, codes, nc, &ns);
        const double decode = now() - decode_start;
        std::printf("STAGES %d gen %.6f decode %.6f\n", i, gen, decode);
#endif
        const double total = now() - start;
        const double cpu_s = (double)(std::clock() - cpu) / CLOCKS_PER_SEC;
        if (!pcm || ns < 24000)
            return 5;
        double ss = 0;
        for (int j = 0; j < ns; j++) {
            if (!std::isfinite(pcm[j]))
                return 6;
            ss += (double)pcm[j] * pcm[j];
        }
        if (ss < 1e-6)
            return 7;
        std::printf("RESULT %d total %.6f samples %d rms %.8f cpu %.6f\n", i, total, ns, std::sqrt(ss / ns), cpu_s);
        std::fflush(stdout);
        std::ofstream f(std::string(argv[6]) + "-" + std::to_string(i) + ".f32", std::ios::binary);
        f.write((const char*)pcm, ns * sizeof(float));
        if (!f)
            return 8;
#ifdef CRISPASR_BENCH_OVCPP
        ov_audio_free(&audio);
#else
        std::ofstream c(std::string(argv[6]) + "-" + std::to_string(i) + ".codes", std::ios::binary);
        c.write((const char*)codes, nc * sizeof(int32_t));
        omnivoice_codes_free(codes);
        omnivoice_pcm_free(pcm);
        if (!c)
            return 8;
#endif
    }
#ifdef CRISPASR_BENCH_OVCPP
    ov_free(ctx);
#else
    omnivoice_free(ctx);
#endif
}
