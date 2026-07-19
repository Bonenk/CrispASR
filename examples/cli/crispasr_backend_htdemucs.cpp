// crispasr_backend_htdemucs.cpp — HTDemucs source separation adapter.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "htdemucs.h"
#include "whisper_params.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

class HtdemucsBackend : public CrispasrBackend {
public:
    HtdemucsBackend() = default;

    const char* name() const override { return "htdemucs"; }

    uint32_t capabilities() const override { return CAP_SEPARATE | CAP_AUTO_DOWNLOAD | CAP_INTERNAL_CHUNKING; }

    bool init(const whisper_params& p) override {
        htdemucs_params hp = htdemucs_default_params();
        hp.n_threads = p.n_threads;
        hp.use_gpu = p.use_gpu;

        ctx_ = htdemucs_init_from_file(p.model.c_str(), hp);
        return ctx_ != nullptr;
    }

    // Source separation produces multiple output streams, not text segments.
    // The CLI adapter stores the separation result for later retrieval.
    std::vector<crispasr_segment> transcribe(const float* samples, int n, int64_t t_offset_cs,
                                             const whisper_params& p) override {
        // HTDemucs expects stereo interleaved at 44100 Hz.
        // The CLI pipeline provides mono 16 kHz. We need to handle resampling
        // and stereo conversion in the CLI dispatcher, not here.
        // For now, assume the caller provides the right format.
        if (!ctx_)
            return {};

        htdemucs_result* r = htdemucs_separate(ctx_, samples, n);
        if (!r)
            return {};

        // Store result for the CLI to write as separate WAV files
        last_result_ = r;

        // Return a single segment with source info
        crispasr_segment seg;
        seg.t0 = 0;
        seg.t1 = (int64_t)((double)n / htdemucs_sample_rate(ctx_) * 100.0);
        seg.text = "[separated: ";
        for (int i = 0; i < r->n_sources; i++) {
            if (i > 0)
                seg.text += ", ";
            seg.text += r->source_names[i];
        }
        seg.text += "]";
        return {seg};
    }

    void shutdown() override {
        if (last_result_) {
            htdemucs_result_free(last_result_);
            last_result_ = nullptr;
        }
        if (ctx_) {
            htdemucs_free(ctx_);
            ctx_ = nullptr;
        }
    }

    // Access the last separation result (for CLI WAV writing)
    htdemucs_result* get_last_result() { return last_result_; }

private:
    htdemucs_context* ctx_ = nullptr;
    htdemucs_result* last_result_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_make_htdemucs_backend() {
    return std::make_unique<HtdemucsBackend>();
}
