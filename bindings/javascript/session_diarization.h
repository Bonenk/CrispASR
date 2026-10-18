#pragma once
#include "session_options.h"
#include "session_speaker.h"

struct crispasr_diarize_turn_abi {
    int64_t t0_cs;
    int64_t t1_cs;
    int32_t speaker;
    int32_t _pad;
};
struct crispasr_diarize_opts_abi {
    int32_t method;
    int32_t n_threads;
    int64_t slice_t0_cs;
    const char* pyannote_model_path;
    const char* foxnose_embedder_path;
    int32_t min_speakers;
    int32_t max_speakers;
    int32_t num_speakers;
    int32_t _pad2;
};
static_assert(sizeof(crispasr_diarize_turn_abi) == 24, "Diarization turn ABI changed");
static_assert(sizeof(crispasr_diarize_opts_abi) == (sizeof(void*) == 4 ? 40 : 48), "Diarization options ABI changed");
static_assert(offsetof(crispasr_diarize_opts_abi, min_speakers) == 16 + 2 * sizeof(void*),
              "Speaker limit offset changed");
static int64_t binding_diarization_time(const emscripten::val& value) {
    if (value.typeOf().as<std::string>() != "number")
        throw std::invalid_argument("Centisecond time must be numeric");
    double n = value.as<double>();
    if (!std::isfinite(n) || n != std::trunc(n) || n < 0 || n > 9007199254740991.0)
        throw std::invalid_argument("Centisecond time must be a nonnegative safe integer");
    return int64_t(n);
}
static std::string binding_diarization_path(const emscripten::val& options, const char* key) {
    auto value = options[key];
    if (value.isUndefined() || value.isNull())
        return {};
    if (value.typeOf().as<std::string>() != "string")
        throw std::invalid_argument("Model path must be a string");
    auto path = value.as<std::string>();
    if (path.find('\0') != std::string::npos)
        throw std::invalid_argument("Embedded NUL in model path");
    return path;
}
template <typename Segment> static emscripten::val binding_diarization_encode(const std::vector<Segment>& segments) {
    auto out = emscripten::val::array();
    for (const auto& segment : segments) {
        auto row = emscripten::val::object();
        row.set("t0", double(segment.t0_cs));
        row.set("t1", double(segment.t1_cs));
        row.set("speaker", segment.speaker);
        out.call<void>("push", row);
    }
    return out;
}
static emscripten::val binding_diarize(const emscripten::val& audio, const emscripten::val& right,
                                       const emscripten::val& grid, const emscripten::val& options, bool wantTurns) {
    using emscripten::val;
    binding_check_reentry();
    auto left = binding_pcm(audio), r = binding_pcm(right, true);
    if (!r.empty() && r.size() != left.size())
        throw std::invalid_argument("Stereo channel lengths must match");
    auto input = options;
    if (input.isNull() || input.isUndefined())
        input = val::object();
    if (input.typeOf().as<std::string>() != "object" || val::global("Array").call<bool>("isArray", input))
        throw std::invalid_argument("Options must be an object");
    auto model = binding_diarization_path(input, "model_path");
    auto embedder = binding_diarization_path(input, "embedder_path");
    crispasr_diarize_opts_abi p{};
    p.method = binding_option_integer(input, "method", 0);
#ifdef __EMSCRIPTEN_PTHREADS__
    const int defaults = 4;
#else
    const int defaults = 1;
#endif
    p.n_threads = binding_option_integer(input, "n_threads", defaults);
    p.slice_t0_cs = input["slice_t0_cs"].isUndefined() ? 0 : binding_diarization_time(input["slice_t0_cs"]);
    p.min_speakers = binding_option_integer(input, "min_speakers", 1);
    p.max_speakers = binding_option_integer(input, "max_speakers", 8);
    p.num_speakers = binding_option_integer(input, "num_speakers", 0);
    if (p.method < 0 || p.method > 5 || p.n_threads < 1 || p.min_speakers < 1 || p.max_speakers < p.min_speakers ||
        p.num_speakers < 0 || (p.num_speakers && (p.num_speakers < p.min_speakers || p.num_speakers > p.max_speakers)))
        throw std::invalid_argument("Invalid diarization options");
    if (((p.method == 3 || p.method == 5) && model.empty()) || (p.method == 4 && embedder.empty()))
        throw std::invalid_argument("Diarization method requires a model path");
    p.pyannote_model_path = model.empty() ? nullptr : model.c_str();
    p.foxnose_embedder_path = embedder.empty() ? nullptr : embedder.c_str();
    int n = grid["length"].as<int>();
    if (n <= 0)
        throw std::invalid_argument("Nonempty segment array required");
    std::vector<crispasr_diarize_seg_abi> segments(n);
    for (int i = 0; i < n; ++i) {
        segments[i].t0_cs = binding_diarization_time(grid[i]["t0"]);
        segments[i].t1_cs = binding_diarization_time(grid[i]["t1"]);
        segments[i].speaker = -1;
        if (segments[i].t1_cs < segments[i].t0_cs)
            throw std::invalid_argument("Segment end precedes start");
    }
    if (!wantTurns) {
        int rc = crispasr_diarize_segments_abi(left.data(), r.empty() ? nullptr : r.data(), left.size(), !r.empty(),
                                               segments.data(), n, &p);
        if (rc != 0)
            throw std::runtime_error("Diarization failed: " + std::to_string(rc));
        return binding_diarization_encode(segments);
    }
    int capacity = 256;
    for (int attempt = 0; attempt < 16; ++attempt) {
        std::vector<crispasr_diarize_turn_abi> turns(capacity);
        int32_t count = 0;
        int rc =
            crispasr_diarize_segments_turns_abi(left.data(), r.empty() ? nullptr : r.data(), left.size(), !r.empty(),
                                                segments.data(), n, &p, turns.data(), capacity, &count);
        if (rc == 2) {
            if (count <= capacity || count > 1024 * 1024)
                throw std::runtime_error("Invalid required turn count");
            capacity = count;
            continue;
        }
        if (rc != 0 || count < 0 || count > capacity)
            throw std::runtime_error("Invalid diarization result");
        turns.resize(count);
        auto out = val::object();
        out.set("segments", binding_diarization_encode(segments));
        out.set("turns", binding_diarization_encode(turns));
        return out;
    }
    throw std::runtime_error("Diarization retry limit exceeded");
}
static void binding_register_diarization() {
    using namespace emscripten;
    function("diarizeSegments",
             optional_override([](const val& audio, const val& right, const val& segments, const val& options) {
                 return binding_diarize(audio, right, segments, options, false);
             }));
    function("diarizeSegmentsWithTurns",
             optional_override([](const val& audio, const val& right, const val& segments, const val& options) {
                 return binding_diarize(audio, right, segments, options, true);
             }));
}
