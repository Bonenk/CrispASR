// encode() validation: codes -> decode -> encode -> compare.
//
// decode() is already trusted (it ships and produces correct audio), so if
// encode() is the right analysis inverse, re-encoding decode's output must
// recover approximately the codes we started from. A wrong nearest-entry rule
// (say L2 where the model uses cosine) puts quantizer-0 agreement at chance,
// 1/1024 ~ 0.1%, which this separates unambiguously from a correct inverse.
#include "moss_tts_local_codec.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
using namespace moss_tts_local_codec;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <codec.gguf> [t_audio]\n", argv[0]); return 2; }
    const int T = argc > 2 ? std::atoi(argv[2]) : 24;

    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_t bes[1] = {be};
    ggml_backend_sched_t sched = ggml_backend_sched_new(bes, nullptr, 1, 8192, false, false);

    Codec* c = load(argv[1], be, sched, /*verbosity=*/1);
    if (!c) { std::fprintf(stderr, "load failed\n"); return 1; }
    std::printf("encoder_ready = %s\n", encoder_ready(c) ? "true" : "false");
    if (!encoder_ready(c)) return 1;

    // Deterministic pseudo-codes in range [0,1024).
    const int NVQ = 12;
    std::mt19937 rng(1234);
    std::vector<int32_t> codes((size_t)NVQ * T);
    for (auto& v : codes) v = (int32_t)(rng() % 1024);

    std::vector<float> wav = decode(c, codes.data(), NVQ, T);
    std::printf("decode -> %zu samples\n", wav.size());
    if (wav.empty()) return 1;

    int nvq_out = 0, t_out = 0;
    std::vector<int32_t> rec = encode(c, wav.data(), (int64_t)wav.size(), nvq_out, t_out);
    std::printf("encode -> n_vq=%d t_audio=%d (%zu codes)\n", nvq_out, t_out, rec.size());
    if (rec.empty()) return 1;

    const int t_cmp = t_out < T ? t_out : T;
    for (int q = 0; q < nvq_out && q < NVQ; q++) {
        int agree = 0;
        for (int t = 0; t < t_cmp; t++)
            if (rec[(size_t)q * t_out + t] == codes[(size_t)q * T + t]) agree++;
        std::printf("  q%-2d agreement %3d/%-3d = %5.1f%%\n", q, agree, t_cmp, 100.0 * agree / t_cmp);
    }
    std::printf("(chance would be %.2f%%)\n", 100.0 / 1024.0);
    free(c);
    return 0;
}
