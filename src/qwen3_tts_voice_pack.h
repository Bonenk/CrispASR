#pragma once

#include "ggml.h"

#include <string>
#include <vector>

// Voice embeddings are model-sized: 0.6B uses 1024, 1.7B uses 2048.
// Validate before backend reads; a pack is not interchangeable across sizes.
namespace qwen3_tts_voice_pack {

inline const char* default_filename(int dimension) {
    switch (dimension) {
    case 1024:
        return "qwen3-tts-voice-0.6b-designed.gguf";
    case 2048:
        return "qwen3-tts-voice-1.7b-designed.gguf";
    default:
        return nullptr;
    }
}

inline bool valid_tensors(const ggml_tensor* speaker, const ggml_tensor* codes, int d, int groups) {
    return speaker && codes && speaker->type == GGML_TYPE_F32 && speaker->ne[0] == d && speaker->ne[1] == 1 &&
           speaker->ne[2] == 1 && speaker->ne[3] == 1 && ggml_is_contiguous(speaker) && codes->type == GGML_TYPE_I32 &&
           codes->ne[0] == groups && codes->ne[1] > 0 && codes->ne[2] == 1 && codes->ne[3] == 1 &&
           ggml_is_contiguous(codes);
}

inline bool valid_names(const std::vector<std::string>& names, const std::vector<std::string>& texts) {
    if (names.empty() || names.size() != texts.size())
        return false;
    for (size_t i = 0; i < names.size(); ++i) {
        if (names[i].empty() || texts[i].empty())
            return false;
        for (size_t j = 0; j < i; ++j) {
            if (names[i] == names[j])
                return false;
        }
    }
    return true;
}

} // namespace qwen3_tts_voice_pack
