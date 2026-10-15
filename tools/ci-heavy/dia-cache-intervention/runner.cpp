// Diagnostic-only forced-history runner. Null PCM is expected without DAC;
// the Python owner must validate complete logits, inputs and captured stages.
#include "dia_tts.h"

#include <cstdio>

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    auto params = dia_tts_context_default_params();
    params.n_threads = 4;
    params.verbosity = 0;
    params.use_gpu = false;
    params.seed = 42;
    params.max_tokens = 979;
    auto* ctx = dia_tts_init_from_file(argv[1], params);
    if (!ctx)
        return 3;
    int samples = 0;
    float* pcm = dia_tts_synthesize(ctx, "[S1] We are committed to making it secure.", &samples);
    const bool unexpected_pcm = pcm || samples != 0;
    dia_tts_pcm_free(pcm);
    dia_tts_free(ctx);
    if (unexpected_pcm)
        return 4;
    std::puts("DIA_DIAGNOSTIC_CALL_RETURNED; complete numerical files still require validation");
    return 0;
}
