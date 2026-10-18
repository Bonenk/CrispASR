#pragma once
#include "crispasr_session.h"
#include <cmath>
#include <cstddef>

// Exact native POD from crispasr_c_api.cpp. The C fixture compiles the original.
struct crispasr_diarize_seg_abi {
    int64_t t0_cs;
    int64_t t1_cs;
    int32_t speaker;
    int32_t _pad;
};
static_assert(sizeof(crispasr_diarize_seg_abi) == 24, "Diarization segment ABI changed");
static_assert(offsetof(crispasr_diarize_seg_abi, speaker) == 16, "Diarization speaker offset changed");
class BindingSpeakerEmbedder {
    void* handle = nullptr;
    void check() const {
        binding_check_reentry();
        if (!handle)
            throw std::runtime_error("Speaker embedder is closed");
    }

public:
    BindingSpeakerEmbedder(const std::string& model, int threads, const std::string& cache) {
        binding_check_reentry();
        if (threads <= 0)
            throw std::invalid_argument("Positive threads required");
        handle = crispasr_speaker_embedder_make_abi(model.c_str(), threads, cache.empty() ? nullptr : cache.c_str());
        if (!handle)
            throw std::runtime_error("speaker_embedder_make failed");
        if (crispasr_speaker_embedder_dim_abi(handle) <= 0) {
            crispasr_speaker_embedder_free_abi(handle);
            handle = nullptr;
            throw std::runtime_error("Invalid embedding dimension");
        }
    }
    BindingSpeakerEmbedder(const BindingSpeakerEmbedder&) = delete;
    ~BindingSpeakerEmbedder() {
        if (handle)
            crispasr_speaker_embedder_free_abi(handle);
    }
    void close() {
        binding_check_reentry();
        if (handle) {
            crispasr_speaker_embedder_free_abi(handle);
            handle = nullptr;
        }
    }
    int dimension() const {
        check();
        return crispasr_speaker_embedder_dim_abi(handle);
    }
    std::string name() const {
        check();
        auto p = crispasr_speaker_embedder_name_abi(handle);
        return p ? p : "";
    }
    emscripten::val embed(const emscripten::val& audio) const {
        check();
        auto pcm = binding_pcm(audio);
        int dim = dimension();
        if (dim <= 0)
            throw std::runtime_error("Invalid embedding dimension");
        std::vector<float> output(dim);
        if (crispasr_speaker_embedder_embed_abi(handle, pcm.data(), pcm.size(), output.data()) != 1)
            throw std::runtime_error("speaker_embedder_embed failed");
        return binding_floats(output.data(), output.size());
    }
};
static int64_t binding_speaker_time(double time) {
    if (!std::isfinite(time) || time < double(INT64_MIN) || time >= double(INT64_MAX))
        throw std::invalid_argument("Invalid centisecond time");
    return int64_t(time);
}
class BindingPyannoteCache {
    void* handle = nullptr;

public:
    BindingPyannoteCache(const emscripten::val& audio, const std::string& model, int threads) {
        binding_check_reentry();
        if (threads <= 0)
            throw std::invalid_argument("Positive threads required");
        auto pcm = binding_pcm(audio);
        handle = crispasr_pyannote_cache_compute_abi(pcm.data(), pcm.size(), model.c_str(), threads);
        if (!handle)
            throw std::runtime_error("pyannote_cache_compute failed");
    }
    BindingPyannoteCache(const BindingPyannoteCache&) = delete;
    ~BindingPyannoteCache() {
        if (handle)
            crispasr_pyannote_cache_free_abi(handle);
    }
    void close() {
        binding_check_reentry();
        if (handle) {
            crispasr_pyannote_cache_free_abi(handle);
            handle = nullptr;
        }
    }
    emscripten::val apply(const emscripten::val& input, double offsetCs) const {
        binding_check_reentry();
        if (!handle)
            throw std::runtime_error("Pyannote cache is closed");
        int64_t offset = binding_speaker_time(offsetCs);
        int n = input["length"].as<int>();
        if (n <= 0)
            throw std::invalid_argument("Nonempty segment array required");
        std::vector<crispasr_diarize_seg_abi> segments(n);
        for (int i = 0; i < n; ++i) {
            auto item = input[i];
            auto& segment = segments[i];
            segment.t0_cs = binding_speaker_time(item["t0"].as<double>());
            segment.t1_cs = binding_speaker_time(item["t1"].as<double>());
            if (segment.t0_cs < 0 || segment.t1_cs < segment.t0_cs)
                throw std::invalid_argument("Invalid segment times");
            segment.speaker = -1;
        }
        if (crispasr_pyannote_cache_apply_abi(handle, offset, segments.data(), n) != 0)
            throw std::runtime_error("pyannote_cache_apply failed");
        auto out = emscripten::val::array();
        for (const auto& segment : segments) {
            auto item = emscripten::val::object();
            item.set("t0", double(segment.t0_cs));
            item.set("t1", double(segment.t1_cs));
            item.set("speaker", segment.speaker);
            out.call<void>("push", item);
        }
        return out;
    }
};
static void binding_register_speaker() {
    using namespace emscripten;
    class_<BindingSpeakerEmbedder>("SpeakerEmbedder")
        .constructor<const std::string&, int, const std::string&>()
        .function("dimension", &BindingSpeakerEmbedder::dimension)
        .function("name", &BindingSpeakerEmbedder::name)
        .function("embed", &BindingSpeakerEmbedder::embed)
        .function("close", &BindingSpeakerEmbedder::close);
    class_<BindingPyannoteCache>("PyannoteCache")
        .constructor<const val&, const std::string&, int>()
        .function("apply", &BindingPyannoteCache::apply)
        .function("close", &BindingPyannoteCache::close);
    function("speakerCluster", optional_override([](const val& input, float threshold, int maximum) {
                 binding_check_reentry();
                 int n = input["length"].as<int>();
                 if (n <= 0 || maximum < 0 || !std::isfinite(threshold))
                     throw std::invalid_argument("Invalid cluster input");
                 int dim = input[0]["length"].as<int>();
                 if (dim <= 0 || n > INT_MAX / dim)
                     throw std::invalid_argument("Invalid embedding dimensions");
                 std::vector<float> matrix;
                 matrix.reserve(size_t(n) * dim);
                 for (int i = 0; i < n; ++i) {
                     auto row = binding_pcm(input[i]);
                     if (row.size() != size_t(dim))
                         throw std::invalid_argument("Embedding dimensions must match");
                     matrix.insert(matrix.end(), row.begin(), row.end());
                 }
                 std::vector<int32_t> labels(n);
                 int clusters = crispasr_speaker_cluster_abi(matrix.data(), n, dim, threshold, maximum, labels.data());
                 if (clusters < 0 || clusters > n)
                     throw std::runtime_error("speaker_cluster failed");
                 auto out = val::object(), values = val::array();
                 for (auto label : labels)
                     values.call<void>("push", label);
                 out.set("labels", values);
                 out.set("nClusters", clusters);
                 return out;
             }));
}
