// POSIX decoder-only probe; two independently prepared contexts, warmup then ABBAAB.
// Match the repository library ABI when this probe is compiled outside CMake.
#ifndef GGML_MAX_NAME
#define GGML_MAX_NAME 128
#endif
#include "../../src/omnivoice.cpp"
#undef NDEBUG
#include <cassert>
#include <ctime>
#include <fstream>
#include <iostream>
int main(int argc, char** argv) {
    assert(argc == 3); // tokenizer, proof directory
    setenv("CRISPASR_OMNIVOICE_CODEC_DECOMP", "1", 1);
    setenv("CRISPASR_OMNIVOICE_CODEC_FASTCONV", "1", 1);
    setenv("CRISPASR_OMNIVOICE_CODEC_SPLIT", "0", 1);
    omnivoice_context contexts[2];
    for (int arm = 0; arm < 2; arm++) {
        auto& ctx = contexts[arm];
        ctx.use_gpu = false;
        ctx.n_threads = 4;
        ctx.verbosity = 0;
        setenv("CRISPASR_OMNIVOICE_CODEC_DECOMP", arm ? "1" : "0", 1);
        assert(load_tokenizer(&ctx, argv[1]));
        std::cout << "arm " << arm << " cache_bytes " << ggml_backend_buffer_get_size(ctx.tokenizer.fc.buf)
                  << " perm_bytes "
                  << (ctx.tokenizer.buf_perm ? ggml_backend_buffer_get_size(ctx.tokenizer.buf_perm) : 0) << "\n";
    }
    for (const char* ref : {"window", "station"}) {
        std::vector<float> wav;
        int sr = 0;
        std::string base = std::string(argv[2]) + "/" + ref;
        assert(crispasr::core::read_wav_mono_pcm16((base + ".wav").c_str(), wav, sr));
        if (sr != 24000)
            wav = core_audio::resample_polyphase(wav.data(), (int)wav.size(), sr, 24000);
        double ss = 0;
        for (float v : wav)
            ss += (double)v * v;
        for (auto& ctx : contexts) {
            ctx.ref_rms = (float)std::sqrt(ss / wav.size());
            ctx.ref_T = 1;
        }
        for (int target = 0; target < 2; target++) {
            std::string prefix = base + "-" + std::to_string(target);
            std::ifstream f(prefix + ".codes", std::ios::binary | std::ios::ate);
            auto bytes = f.tellg();
            assert(bytes > 0);
            f.seekg(0);
            std::vector<int32_t> codes((size_t)bytes / 4);
            f.read((char*)codes.data(), bytes);
            assert(f.good());
            const int order[] = {0, 1, 0, 1, 1, 0, 0, 1};
            for (int rep = 0; rep < 8; rep++) {
                int arm = order[rep];
                auto& ctx = contexts[arm];
                const auto t = std::chrono::steady_clock::now();
                const auto cpu = std::clock();
                int ns = 0;
                float* pcm = omnivoice_decode_codes(&ctx, codes.data(), (int)codes.size(), &ns);
                assert(pcm && ns > 0);
                double cpu_ms = 1000.0 * (std::clock() - cpu) / CLOCKS_PER_SEC;
                double wall_ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
                std::ifstream expected(prefix + "-" + std::to_string(arm) + ".f32", std::ios::binary | std::ios::ate);
                assert(expected.tellg() == ns * 4);
                expected.seekg(0);
                std::vector<float> values(ns);
                expected.read((char*)values.data(), ns * 4);
                assert(expected.good());
                assert(std::memcmp(values.data(), pcm, ns * 4) == 0);
                omnivoice_pcm_free(pcm);
                std::cout << ref << "-" << target << " rep=" << rep << " arm=" << arm << " cpu_ms=" << cpu_ms
                          << " wall_ms=" << wall_ms << " exact=1\n"
                          << std::flush;
            }
        }
    }
    for (auto& ctx : contexts)
        tok_release_backend(ctx.tokenizer);
}
