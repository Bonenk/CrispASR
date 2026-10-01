#pragma once
#include <algorithm>
#include <utility>
#include <vector>

namespace core_index_echo {
struct Window { double start, end; bool hard_cut; };

// Released infer.py vad_windows: split speech, greedily merge with lead/trail,
// then fold a short last window into its predecessor when it fits.
inline std::vector<Window> windows(const std::vector<std::pair<double, double>>& speech,
                                   double total, double maximum = 60) {
    if (maximum <= 0 || total <= 0) return {};
    std::vector<Window> flat, result;
    for (auto segment : speech) {
        double start = segment.first, end = segment.second;
        while (end - start > maximum) {
            flat.push_back({start, start + maximum, true}); start += maximum;
        }
        flat.push_back({start, end, false});
    }
    for (size_t i = 0; i < flat.size();) {
        double start = std::max(flat[i].start - 0.3, result.empty() ? 0.0 : result.back().end);
        size_t j = i;
        bool hard = flat[i].hard_cut;
        while (j + 1 < flat.size() && flat[j + 1].end + 0.5 - start <= maximum && !flat[j].hard_cut) ++j;
        double end = std::min(flat[j].end + 0.5, j + 1 < flat.size() ? flat[j + 1].start : total);
        result.push_back({start, end, hard}); i = j + 1;
    }
    if (result.size() >= 2 && result.back().end - result.back().start < 5 &&
        result.back().end - result[result.size() - 2].start <= maximum) {
        result[result.size() - 2].end = result.back().end; result.pop_back();
    }
    return result;
}
}
