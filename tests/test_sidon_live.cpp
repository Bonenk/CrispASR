// Sidon live integration test — model load + 16 kHz speech restoration.
//
// Requires CRISPASR_MODEL_SIDON to point to a Sidon GGUF. Skips cleanly
// when the model is not available.

#include <catch2/catch_test_macros.hpp>

#include "sidon.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void set_test_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}

struct ScopedTestEnv {
    std::string name;
    std::string previous;
    bool had_previous = false;

    ScopedTestEnv(const char* key, const char* value) : name(key) {
        if (const char* current = std::getenv(key)) {
            previous = current;
            had_previous = true;
        }
        set_test_env(key, value);
    }

    ~ScopedTestEnv() { set_test_env(name.c_str(), had_previous ? previous.c_str() : nullptr); }
};

static std::vector<float> load_wav_16k(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f)
        return {};

    fseek(f, 0, SEEK_END);
    const long size = ftell(f) - 44;
    fseek(f, 44, SEEK_SET);
    if (size <= 0 || size % (long)sizeof(int16_t) != 0) {
        fclose(f);
        return {};
    }

    std::vector<int16_t> raw((size_t)size / sizeof(int16_t));
    const size_t read = fread(raw.data(), sizeof(int16_t), raw.size(), f);
    fclose(f);
    if (read != raw.size())
        return {};

    std::vector<float> pcm(raw.size());
    for (size_t i = 0; i < raw.size(); ++i)
        pcm[i] = raw[i] / 32768.0f;
    return pcm;
}

TEST_CASE("sidon speech restoration", "[integration][sidon]") {
    const char* model_path = std::getenv("CRISPASR_MODEL_SIDON");
    if (!model_path || !*model_path)
        SKIP("CRISPASR_MODEL_SIDON not set");

    auto params = sidon_context_default_params();
    params.verbosity = 0;
    auto* ctx = sidon_init_from_file(model_path, params);
    REQUIRE(ctx != nullptr);

    const auto input = load_wav_16k("samples/jfk.wav");
    REQUIRE(!input.empty());

    const auto output = sidon_restore(ctx, input.data(), (int)input.size());
    REQUIRE(output.size() == input.size() * 3);
    REQUIRE(std::all_of(output.begin(), output.end(), [](float sample) { return std::isfinite(sample); }));

    float peak = 0.0f;
    for (float sample : output)
        peak = std::max(peak, std::fabs(sample));
    CHECK(peak > 0.01f);

    // samples/jfk.wav ends in silence. Decoding without Sidon's prescribed
    // right-side lookahead used to expose the model boundary response here,
    // producing a near-full-scale transient in the final ~12 ms.
    float tail_peak = 0.0f;
    const size_t tail_samples = std::min<size_t>(output.size(), 48 * 12);
    for (size_t i = output.size() - tail_samples; i < output.size(); ++i)
        tail_peak = std::max(tail_peak, std::fabs(output[i]));
    CHECK(tail_peak < 0.05f);

    // The bounded DAC path must preserve the full-graph decoder output. This
    // also exercises scheduler teardown/recreation on a persistent context.
    {
        ScopedTestEnv full_decode("CRISPASR_SIDON_DECODER_CHUNK_FRAMES", "0");
        const auto full_output = sidon_restore(ctx, input.data(), (int)input.size());
        REQUIRE(full_output.size() == output.size());
        double dot = 0.0, chunked_norm = 0.0, full_norm = 0.0;
        for (size_t i = 0; i < output.size(); ++i) {
            dot += (double)output[i] * full_output[i];
            chunked_norm += (double)output[i] * output[i];
            full_norm += (double)full_output[i] * full_output[i];
        }
        const double cosine = dot / std::sqrt(chunked_norm * full_norm);
        CHECK(cosine > 0.999);
    }

    // O(T^2) length cap: an over-long input (well past the ~58.5 s / 3000-frame
    // default) must fail cleanly with an empty result — never OOM/crash. The
    // guard trips right after the cheap STFT front-end, so this stays fast even
    // with the model loaded. ~90 s of silence plus lookahead ⇒ T ≈ 4575 > 3000.
    {
        std::vector<float> too_long(16000 * 90, 0.0f);
        // A little energy so peak-normalization doesn't divide near-zero.
        for (size_t i = 0; i < too_long.size(); i += 160)
            too_long[i] = 0.1f;
        const auto capped = sidon_restore(ctx, too_long.data(), (int)too_long.size());
        CHECK(capped.empty());
    }

    sidon_free(ctx);
}
