// Live public-C-API lifecycle checks. Synthetic packs test loading only;
// recognizable speech is verified separately with an actual reference WAV.
#include "qwen3_tts.h"

#include <cstdio>
#include <filesystem>
#include <string>

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s model.gguf fixtures cpu|cuda\n", argv[0]);
        return 2;
    }
    auto params = qwen3_tts_context_default_params();
    params.use_gpu = std::string(argv[3]) == "cuda";
    params.verbosity = 1;
    params.n_threads = 4;
    auto* ctx = qwen3_tts_init_from_file(argv[1], params);
    if (!ctx)
        return 3;
    int checks = 0;
    auto load = [&](const char* name, bool expected) {
        const std::string path = (std::filesystem::path(argv[2]) / name).string();
        const bool ok = qwen3_tts_load_voice_pack(ctx, path.c_str()) == 0;
        ++checks;
        if (ok != expected) {
            fprintf(stderr, "unexpected load result: %s, expected %d\n", name, (int)expected);
            return false;
        }
        return true;
    };
    bool ok = load("good-two.gguf", true) && qwen3_tts_select_voice(ctx, "second") == 0;
    for (const char* name :
         {"wrong-dimension.gguf", "half-speaker.gguf", "float-codes.gguf", "missing-speaker.gguf", "missing-text.gguf",
          "duplicate-names.gguf", "wrong-groups.gguf", "extra-axis.gguf", "missing-file.gguf"}) {
        ok = ok && load(name, false) && qwen3_tts_select_voice(ctx, "second") == 0;
    }
    // Old active index 1 cannot survive replacement with a single-voice pack.
    ok = ok && load("good-one.gguf", true) && qwen3_tts_select_voice(ctx, "first") == 0 &&
         qwen3_tts_select_voice(ctx, "second") == -2;
    qwen3_tts_free(ctx);
    fprintf(stderr, "QWEN3_VOICE_PACK_%s checks=%d\n", ok ? "PASS" : "FAIL", checks);
    return ok && checks == 11 ? 0 : 1;
}
