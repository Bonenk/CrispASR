// POSIX M1 probe: fixed-code CPU / split-reload / resident-split A/B.
// Build against the same ggml/core libraries as the runtime, without libomnivoice.
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
    assert(argc == 3 || (argc == 4 && std::string(argv[3]) == "--voice-check"));
    // tokenizer, prior codec_ab proof directory, optional fresh-encoder proof
    ggml_backend_load_all();
    omnivoice_context contexts[3];
    for (int arm = 0; arm < 3; arm++) {
        auto& ctx = contexts[arm];
        ctx.use_gpu = true;
        ctx.n_threads = 4;
        ctx.verbosity = 0;
        setenv("CRISPASR_OMNIVOICE_CODEC_GPU", "0", 1);
        setenv("CRISPASR_OMNIVOICE_CODEC_SPLIT", "1", 1);
        setenv("CRISPASR_OMNIVOICE_CODEC_DECOMP", arm ? "1" : "0", 1);
        assert(load_tokenizer(&ctx, argv[1]));
    }
    if (argc == 3) {
        for (const char* ref : {"window", "station"}) {
            std::vector<float> wav;
            int sr = 0;
            const std::string base = std::string(argv[2]) + "/" + ref;
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
                const std::string prefix = base + "-" + std::to_string(target);
                std::ifstream f(prefix + ".codes", std::ios::binary | std::ios::ate);
                const auto bytes = f.tellg();
                assert(bytes > 0 && bytes % 4 == 0);
                f.seekg(0);
                std::vector<int32_t> codes((size_t)bytes / 4);
                f.read((char*)codes.data(), bytes);
                assert(f.good());
                std::vector<float> baseline, split;
                // One warmup per arm, then three counterbalanced samples per arm.
                const int order[] = {0, 1, 2, 0, 1, 2, 2, 1, 0, 0, 1, 2};
                for (int rep = 0; rep < 12; rep++) {
                    const int arm = order[rep];
                    auto& ctx = contexts[arm];
                    setenv("CRISPASR_OMNIVOICE_CODEC_SPLIT", arm ? "1" : "0", 1);
                    setenv("CRISPASR_OMNIVOICE_CODEC_RESIDENT", arm == 2 ? "1" : "0", 1);
                    setenv("CRISPASR_OMNIVOICE_CODEC_DECOMP", arm ? "1" : "0", 1);
                    const auto t = std::chrono::steady_clock::now();
                    const auto cpu = std::clock();
                    // Model the placement transition needed before a new voice is
                    // encoded. Encoder compute is deliberately excluded from timing.
                    assert(tok_rebind(&ctx, false));
                    auto* encoder_buffer = ctx.tokenizer.buf_w;
                    int ns = 0;
                    float* pcm = omnivoice_decode_codes(&ctx, codes.data(), (int)codes.size(), &ns);
                    assert(pcm && ns > 0);
                    const double cpu_ms = 1000.0 * (std::clock() - cpu) / CLOCKS_PER_SEC;
                    const double wall_ms =
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
                    if (arm == 0)
                        baseline.assign(pcm, pcm + ns);
                    else {
                        assert((int)baseline.size() == ns);
                        double err = 0;
                        for (int i = 0; i < ns; i++)
                            err = std::max(err, std::abs((double)pcm[i] - baseline[i]));
                        assert(err <= 0.002); // numerical screen; ASR is a separate gate
                        if (arm == 1)
                            split.assign(pcm, pcm + ns);
                        else {
                            assert(split.size() == (size_t)ns);
                            assert(std::memcmp(split.data(), pcm, ns * sizeof(float)) == 0);
                            assert(ctx.tokenizer.buf_w == encoder_buffer && !ctx.tokenizer.gpu_requested);
                            assert(ctx.resident_decoder.loaded && !ctx.resident_decoder.enc_conv1_w);
                        }
                    }
                    if (rep < 3) {
                        std::ofstream out(prefix + "-resident-arm" + std::to_string(arm) + ".f32", std::ios::binary);
                        out.write((const char*)pcm, ns * sizeof(float));
                    }
                    std::cout << ref << "-" << target << " rep=" << rep << " arm=" << arm << " cpu_ms=" << cpu_ms
                              << " wall_ms=" << wall_ms << " samples=" << ns << "\n"
                              << std::flush;
                    omnivoice_pcm_free(pcm);
                }
            }
        }
    }
    {
        setenv("CRISPASR_OMNIVOICE_VOICE_CACHE", "0", 1);
        setenv("CRISPASR_OMNIVOICE_CODEC_SPLIT", "1", 1);
        setenv("CRISPASR_OMNIVOICE_CODEC_RESIDENT", "1", 1);
        for (const char* ref : {"window", "station"}) {
            const std::string base = std::string(argv[2]) + "/" + ref;
            const char* text = std::string(ref) == "window" ? "Please close the window before you leave the room."
                                                            : "We will meet at the train station tomorrow morning.";
            for (int arm : {0, 2})
                assert(omnivoice_set_voice_prompt(&contexts[arm], (base + ".wav").c_str(), text) == 0);
            assert(contexts[0].ref_audio_codes == contexts[2].ref_audio_codes);
            assert(contexts[0].ref_T == contexts[2].ref_T);
            auto& ctx = contexts[2];
            auto* encoder = ctx.tokenizer.buf_w;
            int ns = 0;
            float* pcm = omnivoice_decode_codes(&ctx, ctx.ref_audio_codes.data(), (int)ctx.ref_audio_codes.size(), &ns);
            assert(pcm && ns > 24000 && ctx.tokenizer.buf_w == encoder);
            omnivoice_pcm_free(pcm);
            std::cout << "voice " << ref << " fresh_codes_exact=1 frames=" << ctx.ref_T << " encoder_retained=1\n"
                      << std::flush;
            std::ofstream out(base + "-resident-ref.codes", std::ios::binary);
            out.write((const char*)ctx.ref_audio_codes.data(), ctx.ref_audio_codes.size() * sizeof(int32_t));
        }
    }
    auto& decoder = contexts[2].resident_decoder;
    std::cout << "resident serialized_bytes=" << ggml_backend_buffer_get_size(decoder.buf_w)
              << " cache_bytes=" << ggml_backend_buffer_get_size(decoder.fc.buf)
              << " perm_bytes=" << ggml_backend_buffer_get_size(decoder.buf_perm) << "\n";
    for (auto& ctx : contexts) {
        tok_release_backend(ctx.resident_decoder);
        tok_release_backend(ctx.tokenizer);
    }
}
