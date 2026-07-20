// test-core-cqt.cpp — core/cqt.h constant-Q transform.
//
// Deterministic properties verified here (no model, no network, CI-safe):
//   * geometric bin spacing and Q-derived kernel lengths
//   * a pure tone peaks in the bin its frequency maps to
//   * octave-related tones land exactly bins_per_octave apart
//   * silence -> zeros; L1 normalisation makes bins comparable across lengths
//
// Cross-checking against librosa is a SEPARATE step and is not done here (CI has
// no numpy/librosa). Run `test-core-cqt --dump <file>` to write the magnitude
// matrix for the shared test signal, then score it with
// `python tools/cqt_librosa_parity.py <file>`.

#include "core/cqt.h"

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

constexpr int SR = 22050;
constexpr float FMIN = 32.703195662574829f; // C1
constexpr int HOP = 2048;

core_cqt::Params btc_params() {
    core_cqt::Params p;
    p.sample_rate = SR;
    p.fmin = FMIN;
    p.n_bins = 144;
    p.bins_per_octave = 24;
    p.hop_length = HOP;
    return p;
}

std::vector<float> tone(int n, double f, double amp = 0.5) {
    std::vector<float> x((size_t)n);
    for (int i = 0; i < n; i++)
        x[(size_t)i] = (float)(amp * std::sin(2.0 * M_PI * f * (double)i / (double)SR));
    return x;
}

// Must match tools/cqt_librosa_parity.py test_signal() exactly.
std::vector<float> shared_test_signal(int n) {
    std::vector<float> x((size_t)n, 0.0f);
    const double fs[3] = {130.8127826502993, 261.6255653005986, 523.2511306011972};
    const int third = n / 3;
    for (int i = 0; i < 3; i++) {
        const int lo = i * third;
        const int hi = (i < 2) ? (i + 1) * third : n;
        for (int s = lo; s < hi; s++)
            x[(size_t)s] = (float)(0.5 * std::sin(2.0 * M_PI * fs[i] * (double)s / (double)SR));
    }
    return x;
}

int expected_bin(const core_cqt::Params& p, double f) {
    return (int)std::lround((double)p.bins_per_octave * std::log2(f / (double)p.fmin));
}

int argmax_frame(const std::vector<float>& m, int t, int n_bins) {
    int best = 0;
    for (int k = 1; k < n_bins; k++)
        if (m[(size_t)t * (size_t)n_bins + (size_t)k] > m[(size_t)t * (size_t)n_bins + (size_t)best])
            best = k;
    return best;
}

} // namespace

TEST_CASE("cqt: bin frequencies are geometric", "[core][cqt]") {
    const auto p = btc_params();
    REQUIRE_THAT(core_cqt::bin_frequency(p, 0), Catch::Matchers::WithinRel(FMIN, 1e-6f));
    // One octave up = bins_per_octave bins.
    REQUIRE_THAT(core_cqt::bin_frequency(p, p.bins_per_octave), Catch::Matchers::WithinRel(2.0f * FMIN, 1e-5f));
    // Every adjacent pair has the same ratio.
    const float r = core_cqt::bin_frequency(p, 1) / core_cqt::bin_frequency(p, 0);
    for (int k = 1; k < 40; k++) {
        const float rk = core_cqt::bin_frequency(p, k + 1) / core_cqt::bin_frequency(p, k);
        REQUIRE_THAT(rk, Catch::Matchers::WithinRel(r, 1e-5f));
    }
}

TEST_CASE("cqt: kernel length shrinks with frequency, Q is constant", "[core][cqt]") {
    const auto p = btc_params();
    const int n0 = core_cqt::kernel_length(p, 0);
    const int n_oct = core_cqt::kernel_length(p, p.bins_per_octave);
    REQUIRE(n0 > 0);
    // An octave up halves the required window.
    REQUIRE(std::abs((double)n_oct - (double)n0 / 2.0) <= 2.0);
    // Lengths are monotonically non-increasing.
    for (int k = 1; k < p.n_bins; k++)
        REQUIRE(core_cqt::kernel_length(p, k) <= core_cqt::kernel_length(p, k - 1));
}

TEST_CASE("cqt: a pure tone peaks in its own bin", "[core][cqt]") {
    auto p = btc_params();
    // 5 octaves: the top test tone (C5, 523.25 Hz) maps to bin 96, so n_bins
    // must EXCEED 96 or argmax saturates at the last bin and the octave-
    // spacing check silently reads 23 instead of 24.
    p.n_bins = 120;
    const auto kernels = core_cqt::build_kernels(p);

    for (double f : {130.8127826502993, 261.6255653005986, 523.2511306011972}) {
        const auto x = tone(SR, f);
        std::vector<float> m;
        const int T = core_cqt::magnitude(p, kernels, x.data(), (int)x.size(), m);
        REQUIRE(T > 2);
        // Use a middle frame: the first/last are edge-padded.
        const int k = argmax_frame(m, T / 2, p.n_bins);
        INFO("f=" << f << " expected bin " << expected_bin(p, f) << " got " << k);
        REQUIRE(std::abs(k - expected_bin(p, f)) <= 1);
    }
}

TEST_CASE("cqt: octave-related tones are exactly bins_per_octave apart", "[core][cqt]") {
    auto p = btc_params();
    p.n_bins = 120; // must exceed bin 96 (C5) -- see the tone test above
    const auto kernels = core_cqt::build_kernels(p);

    std::vector<float> m1, m2;
    const auto x1 = tone(SR, 261.6255653005986);
    const auto x2 = tone(SR, 523.2511306011972);
    const int T1 = core_cqt::magnitude(p, kernels, x1.data(), (int)x1.size(), m1);
    const int T2 = core_cqt::magnitude(p, kernels, x2.data(), (int)x2.size(), m2);
    const int k1 = argmax_frame(m1, T1 / 2, p.n_bins);
    const int k2 = argmax_frame(m2, T2 / 2, p.n_bins);
    REQUIRE(k2 - k1 == p.bins_per_octave);
}

TEST_CASE("cqt: silence is zero, shape is right", "[core][cqt]") {
    auto p = btc_params();
    p.n_bins = 48;
    std::vector<float> x((size_t)SR, 0.0f);
    std::vector<float> m;
    const int T = core_cqt::magnitude(p, x.data(), (int)x.size(), m);
    REQUIRE(T == core_cqt::n_frames(p, (int)x.size()));
    REQUIRE(m.size() == (size_t)T * (size_t)p.n_bins);
    for (float v : m)
        REQUIRE(v == 0.0f);
}

TEST_CASE("cqt: L1 normalisation makes bins comparable across kernel lengths", "[core][cqt]") {
    auto p = btc_params();
    p.n_bins = 120;
    // Equal-amplitude tones two octaves apart should give comparable magnitude
    // once normalised, despite a 4x difference in kernel length.
    const auto kernels = core_cqt::build_kernels(p);
    std::vector<float> lo_m, hi_m;
    const auto lo = tone(SR, 130.8127826502993);
    const auto hi = tone(SR, 523.2511306011972);
    const int Tl = core_cqt::magnitude(p, kernels, lo.data(), (int)lo.size(), lo_m);
    const int Th = core_cqt::magnitude(p, kernels, hi.data(), (int)hi.size(), hi_m);
    const float pl = lo_m[(size_t)(Tl / 2) * (size_t)p.n_bins + (size_t)argmax_frame(lo_m, Tl / 2, p.n_bins)];
    const float ph = hi_m[(size_t)(Th / 2) * (size_t)p.n_bins + (size_t)argmax_frame(hi_m, Th / 2, p.n_bins)];
    REQUIRE(pl > 0.0f);
    REQUIRE(ph > 0.0f);
    // Within 3x is the bar: exact equality is not expected (window mainlobe
    // interacts with bin spacing), but an unnormalised CQT differs by ~4x here.
    const float ratio = pl > ph ? pl / ph : ph / pl;
    INFO("low peak " << pl << " high peak " << ph << " ratio " << ratio);
    REQUIRE(ratio < 3.0f);
}

// `test-core-cqt --dump <file>` writes the shared signal's magnitude matrix for
// tools/cqt_librosa_parity.py. Not part of the Catch2 run.
int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            const auto p = btc_params();
            const int T_target = 40;
            const auto x = shared_test_signal(T_target * HOP);
            std::vector<float> m;
            const int T = core_cqt::magnitude(p, x.data(), (int)x.size(), m);
            FILE* f = std::fopen(argv[i + 1], "wb");
            if (!f)
                return 1;
            const int32_t hdr[2] = {(int32_t)T, (int32_t)p.n_bins};
            std::fwrite(hdr, sizeof(int32_t), 2, f);
            std::fwrite(m.data(), sizeof(float), m.size(), f);
            std::fclose(f);
            std::printf("wrote %s (%d frames x %d bins)\n", argv[i + 1], T, p.n_bins);
            return 0;
        }
    }
    return Catch::Session().run(argc, argv);
}
