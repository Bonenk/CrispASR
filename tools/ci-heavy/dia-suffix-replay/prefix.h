#pragma once
#include <algorithm>
#include <cstddef>
#include <vector>

// Captured [batch][time][channel] -> runtime [time][batch][channel].
// Exclude the captured current position; it must be computed by the replay.
inline bool dia_suffix_prefix(const std::vector<float>& full, size_t batch, size_t time, size_t channels,
                              std::vector<float>& prefix) {
    if (!batch || batch > 2 || time < 2 || time > 979 || !channels || channels > 512 ||
        full.size() != batch * time * channels)
        return false;
    prefix.resize(batch * (time - 1) * channels);
    for (size_t t = 0; t < time - 1; ++t)
        for (size_t b = 0; b < batch; ++b)
            std::copy_n(full.data() + (b * time + t) * channels, channels, prefix.data() + (t * batch + b) * channels);
    return true;
}
