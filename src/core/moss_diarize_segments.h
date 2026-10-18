// Shared native MOSS diarization parser. No inference or timestamp interpolation.
#pragma once

#include "../moss_transcribe_diarize.h"
#include <algorithm>
#include <cstring>
#include <regex>
#include <string>
#include <vector>

inline std::vector<moss_diarize_segment> moss_diarize_parse_output(const std::string& text) {
    std::vector<moss_diarize_segment> segs;
    // Pattern: [start_time][Sxx]text[end_time]
    std::regex seg_re(R"(\[(\d+\.?\d*)\]\[S(\d+)\](.*?)\[(\d+\.?\d*)\])");
    auto it = std::sregex_iterator(text.begin(), text.end(), seg_re);
    auto end = std::sregex_iterator();

    for (; it != end; ++it) {
        const std::smatch& m = *it;
        moss_diarize_segment seg;
        seg.t0_cs = (int64_t)(std::stof(m[1].str()) * 100.0f + 0.5f);
        seg.t1_cs = (int64_t)(std::stof(m[4].str()) * 100.0f + 0.5f);
        seg.speaker_id = std::stoi(m[2].str());
        std::string seg_text = m[3].str();
        // Trim whitespace
        size_t s = seg_text.find_first_not_of(" \t\n");
        size_t e = seg_text.find_last_not_of(" \t\n");
        if (s != std::string::npos && e != std::string::npos)
            seg_text = seg_text.substr(s, e - s + 1);
        size_t copy_len = std::min(seg_text.size(), sizeof(seg.text) - 1);
        memcpy(seg.text, seg_text.c_str(), copy_len);
        seg.text[copy_len] = '\0';
        segs.push_back(seg);
    }
    return segs;
}
