// Bounded encoder inference with global-frame ownership for CTC alignment.
#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace core_ctc_windows {
// Center-padded mel followed by stride-eight FastConformer emits floor(N/stride)+1
// frames. Chunk origins/context are aligned to that stride. No transcript split
// or blank-frame fabrication: every global frame is copied from one real window.
// Compute returns row-major logits plus their actual frame/vocabulary dimensions.
template <typename Compute>
int compute(const float* samples, int n_samples, int frame_samples, int core_frames, int context_frames, Compute run,
            std::vector<float>& logits, int& frames, int& vocab) {
    logits.clear();
    frames = vocab = 0;
    if (!samples || n_samples <= 0 || frame_samples <= 0 || core_frames <= 0 || context_frames < 0)
        return -1;
    const int64_t total_wide = int64_t(n_samples) / frame_samples + 1;
    if (total_wide > std::numeric_limits<int>::max())
        return -1;
    const int total = int(total_wide);
    std::vector<float> chunk;
    int output_vocab = 0;
    for (int64_t first = 0; first < total; first += core_frames) {
        const int64_t last = std::min<int64_t>(total, first + core_frames);
        const int64_t origin = std::max<int64_t>(0, first - context_frames);
        const int64_t sample_start = origin * frame_samples;
        const int64_t sample_end = std::min<int64_t>(n_samples, (last + context_frames) * frame_samples);
        // A final frame exactly at N belongs to the preceding audio window.
        // Include a frame of left context even when context_frames is zero.
        const int64_t actual_origin = sample_start < n_samples ? origin : std::max<int64_t>(0, origin - 1);
        const int64_t actual_start = actual_origin * frame_samples;
        if (sample_end <= actual_start)
            return -2;
        int t = 0, v = 0;
        chunk.clear();
        const int rc = run(samples + actual_start, int(sample_end - actual_start), chunk, t, v);
        if (rc != 0) {
            logits.clear();
            return rc;
        }
        const int64_t begin = first - actual_origin;
        const int64_t count = last - first;
        if (t <= 0 || v <= 0 || begin < 0 || begin + count > t ||
            size_t(t) > std::numeric_limits<size_t>::max() / size_t(v) || chunk.size() != size_t(t) * size_t(v) ||
            (output_vocab != 0 && v != output_vocab)) {
            logits.clear();
            return -3;
        }
        if (output_vocab == 0) {
            if (size_t(total) > logits.max_size() / size_t(v))
                return -4;
            output_vocab = v;
            logits.resize(size_t(total) * size_t(v));
        }
        std::copy_n(chunk.data() + size_t(begin) * size_t(v), size_t(count) * size_t(v),
                    logits.data() + size_t(first) * size_t(v));
    }
    frames = total;
    vocab = output_vocab;
    return 0;
}
} // namespace core_ctc_windows
