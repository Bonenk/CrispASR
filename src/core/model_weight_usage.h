// Audited model usage contracts. Never infer "all 2-D weights are linear".
#pragma once
#include "gguf_loader.h"
#include <cstring>

namespace core_gguf {
inline bool weight_suffix(const char* name, const char* suffix) {
    const size_t n = std::strlen(name), s = std::strlen(suffix);
    return n >= s && std::strcmp(name + n - s, suffix) == 0 && (n == s || name[n - s - 1] == '.');
}
inline bool weight_layer_prefix(const char* name, const char* prefix) {
    const size_t n = std::strlen(prefix);
    if (std::strncmp(name, prefix, n) != 0)
        return false;
    const char* p = name + n;
    if (*p < '0' || *p > '9')
        return false;
    while (*p >= '0' && *p <= '9')
        ++p;
    return *p == '.';
}
inline WeightUsage mimo_weight_usage(const char* name, void*) {
    // Separate LM head; embedding tables are NOT tied to it in this backend.
    if (std::strcmp(name, "llm.lm_head.weight") == 0 || std::strcmp(name, "audio.group_proj.weight") == 0)
        return WeightUsage::MatmulOnly;
    if (weight_layer_prefix(name, "model.layers.") || weight_layer_prefix(name, "audio.blk.")) {
        const char* suffixes[] = {"attn.q.weight",   "attn.k.weight",   "attn.v.weight", "attn.o.weight",
                                  "attn.qkv.weight", "ffn.gate.weight", "ffn.up.weight", "ffn.down.weight"};
        for (auto suffix : suffixes)
            if (weight_suffix(name, suffix))
                return WeightUsage::MatmulOnly;
    }
    return WeightUsage::General;
}
inline WeightUsage moonshine_weight_usage(const char* name, void*) {
    // decoder.embed_tokens is also used for output when no separate head exists.
    if (std::strcmp(name, "decoder.output.weight") == 0)
        return WeightUsage::MatmulOnly;
    if (weight_layer_prefix(name, "encoder.layers.") || weight_layer_prefix(name, "decoder.layers.")) {
        const char* suffixes[] = {"attn.q.weight",       "attn.k.weight",       "attn.v.weight",
                                  "attn.o.weight",       "cross_attn.q.weight", "cross_attn.k.weight",
                                  "cross_attn.v.weight", "cross_attn.o.weight", "ffn.fc1.weight",
                                  "ffn.fc2.weight"};
        for (auto suffix : suffixes)
            if (weight_suffix(name, suffix))
                return WeightUsage::MatmulOnly;
    }
    return WeightUsage::General;
}
} // namespace core_gguf
