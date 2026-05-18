// crispasr_speaker_embedder.cpp — concrete embedder adapters and the
// factory that picks one for a given CLI model spec.
//
// Currently only the TitaNet-Large adapter exists (wraps the C
// titanet_init / titanet_embed / titanet_free trio). Adding a new
// model is intentionally cheap: subclass CrispasrSpeakerEmbedder,
// implement embed(), and add a dispatch branch in
// crispasr_make_speaker_embedder().

#include "crispasr_speaker_embedder.h"

#include "crispasr_model_registry.h"
#include "titanet.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

namespace {

class TitaNetEmbedder : public CrispasrSpeakerEmbedder {
public:
    explicit TitaNetEmbedder(titanet_context* ctx) : ctx_(ctx) {}
    ~TitaNetEmbedder() override {
        if (ctx_)
            titanet_free(ctx_);
    }

    int dim() const override { return 192; }
    const char* name() const override { return "titanet-large"; }

    bool embed(const float* pcm_16k, int n_samples, float* out) override {
        if (!ctx_ || !pcm_16k || n_samples <= 0 || !out)
            return false;
        const int n = titanet_embed(ctx_, pcm_16k, n_samples, out);
        return n == 192;
    }

private:
    titanet_context* ctx_;
};

} // namespace

std::unique_ptr<CrispasrSpeakerEmbedder> crispasr_make_speaker_embedder(const std::string& model_spec, int n_threads,
                                                                        const std::string& cache_dir) {
    if (model_spec.empty())
        return nullptr;

    // Resolve "auto" + bare filenames through the model registry, which
    // handles HF auto-download into the cache. Currently the only known
    // family is TitaNet; future embedders can be dispatched here on
    // model_id / file extension / GGUF general.architecture.
    std::string resolved = model_spec;
    if (resolved == "auto" || resolved.find('/') == std::string::npos) {
        // No-prints stays true here so callers control verbosity.
        resolved = crispasr_resolve_model(resolved == "auto" ? std::string("auto") : resolved, "titanet",
                                          /*no_prints=*/false, cache_dir, /*auto_download=*/true, "");
    }
    if (resolved.empty()) {
        fprintf(stderr, "crispasr[diarize]: failed to resolve speaker embedder model '%s'\n", model_spec.c_str());
        return nullptr;
    }

    titanet_context* ctx = titanet_init(resolved.c_str(), n_threads);
    if (!ctx) {
        fprintf(stderr, "crispasr[diarize]: failed to load speaker embedder from '%s'\n", resolved.c_str());
        return nullptr;
    }
    return std::make_unique<TitaNetEmbedder>(ctx);
}
