// POSIX live probe: generate each target once, then decode the same codes with both upsamplers.
#include "omnivoice.h"
#include "ggml-backend.h"
#undef NDEBUG
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
static double ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}
int main(int argc, char** argv) {
    assert(argc == 6); // model, tokenizer, reference, reference text, output prefix
    ggml_backend_load_all();
    auto p = omnivoice_context_default_params();
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    p.num_steps = 32;
    p.seed = 42;
    auto* ctx = omnivoice_init_from_file(argv[1], p);
    assert(ctx);
    setenv("CRISPASR_OMNIVOICE_CODEC_DECOMP", "0", 1);
    assert(omnivoice_set_tokenizer_path(ctx, argv[2]) == 0);
    if (std::string(argv[3]) != "none")
        assert(omnivoice_set_voice_prompt(ctx, argv[3], argv[4]) == 0);
    const char* texts[] = {"The quick brown fox jumps over the lazy dog.",
                           "We should bring enough water and food for the long walk home."};
    for (int target = 0; target < 2; target++) {
        omnivoice_set_seed(ctx, 42);
        int nc = 0;
        auto* codes = omnivoice_synthesize_codes(ctx, texts[target], &nc);
        assert(codes && nc > 0);
        std::string prefix = std::string(argv[5]) + "-" + std::to_string(target);
        {
            std::ofstream f(prefix + ".codes", std::ios::binary);
            f.write((const char*)codes, nc * sizeof(int32_t));
        }
        for (int arm = 0; arm < 2; arm++) {
            setenv("CRISPASR_OMNIVOICE_CODEC_DECOMP", arm ? "1" : "0", 1);
            auto t = std::chrono::steady_clock::now();
            assert(omnivoice_set_tokenizer_path(ctx, argv[2]) == 0);
            double reload = ms(t);
            t = std::chrono::steady_clock::now();
            int ns = 0;
            float* pcm = omnivoice_decode_codes(ctx, codes, nc, &ns);
            assert(pcm && ns > 24000);
            std::cout << prefix << " arm=" << arm << " reload_ms=" << reload << " decode_ms=" << ms(t)
                      << " samples=" << ns << "\n"
                      << std::flush;
            {
                std::ofstream f(prefix + "-" + std::to_string(arm) + ".f32", std::ios::binary);
                f.write((const char*)pcm, ns * sizeof(float));
            }
            omnivoice_pcm_free(pcm);
        }
        omnivoice_codes_free(codes);
    }
    omnivoice_free(ctx);
}
