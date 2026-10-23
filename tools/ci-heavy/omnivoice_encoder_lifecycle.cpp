#define GGML_MAX_NAME 128
#include "../../src/omnivoice.cpp"
#include "ggml-backend-impl.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <filesystem>
#include <fstream>
static decltype(ggml_backend_i::graph_compute) real_compute;
static int fail_after;
static bool injected;
int main(int argc, char** argv) {
    assert(argc == 4);
    for (auto* suffix : {"CACHE", "FASTCONV", "GRAPH", "FUSED", "HYBRID", "SEMANTIC_F32"})
        unsetenv((std::string("CRISPASR_OMNIVOICE_ENCODER_") + suffix).c_str());
    setenv("CRISPASR_OMNIVOICE_CODEC_GPU", "0", 1);
    setenv("CRISPASR_OMNIVOICE_CODEC_SPLIT", "0", 1);
    const auto cache = std::filesystem::path(argv[3]) / ("run-" + std::to_string(getpid()));
    std::filesystem::create_directories(cache);
    setenv("CRISPASR_TTS_REF_CACHE_DIR", cache.c_str(), 1);
    setenv("CRISPASR_TTS_REF_CACHE", "1", 1);
    unsetenv("CRISPASR_OMNIVOICE_VOICE_CACHE");
    ggml_backend_load_all();
    auto make = [&]() {
        auto* c = new omnivoice_context;
        c->use_gpu = true;
        c->n_threads = 4;
        c->verbosity = 1;
        c->backend = crispasr_init_gpu_backend();
        assert(c->backend && load_tokenizer(c, argv[1]));
        return c;
    };
    auto* c = make();
    assert(encoder_gpu_defaults(c));
    assert(omnivoice_set_voice_prompt(c, argv[2], "Synthetic reference.") == 0);
    auto expected = c->ref_audio_codes;
    assert(c->semantic_encoder.loaded);
    omnivoice_free(c);
    c = make();
    assert(omnivoice_set_voice_prompt(c, argv[2], "Synthetic reference.") == 0);
    assert(c->ref_audio_codes == expected && !c->semantic_encoder.loaded);
    omnivoice_free(c);
    puts("PASS fresh-context cache hit skips GPU allocation");
    setenv("CRISPASR_TTS_REF_CACHE", "0", 1);
    for (int stage = 1; stage <= 3; stage++) {
        c = make();
        auto* g = encoder_semantic_tokenizer(c);
        assert(g != &c->tokenizer);
        real_compute = g->backend->iface.graph_compute;
        fail_after = stage;
        injected = false;
        g->backend->iface.graph_compute = [](ggml_backend_t b, ggml_cgraph* graph) {
            if (--fail_after == 0) {
                injected = true;
                return GGML_STATUS_FAILED;
            }
            return real_compute(b, graph);
        };
        assert(omnivoice_set_voice_prompt(c, argv[2], "Synthetic reference.") == 0);
        assert(injected && c->encoder_gpu_failed && !c->semantic_encoder.loaded && c->ref_audio_codes == expected);
        assert(omnivoice_set_voice_prompt(c, argv[2], "Synthetic reference.") == 0 && c->ref_audio_codes == expected);
        omnivoice_free(c);
        std::printf("PASS injected GPU compute failure stage%d recovers exact CPU codes; retry stays CPU\n", stage);
    }
    c = make();
    c->use_gpu = false;
    assert(!encoder_hybrid_enabled(c));
    assert(omnivoice_set_voice_prompt(c, argv[2], "Synthetic reference.") == 0);
    assert(!c->semantic_encoder.loaded && c->ref_audio_codes == expected);
    omnivoice_free(c);
    puts("PASS no-GPU policy preserves CPU encoding");
}
