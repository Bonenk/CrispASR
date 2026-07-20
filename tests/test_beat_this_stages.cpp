// test_beat_this_stages.cpp — per-stage parity for the beat-this graph.
//
//   ./test-beat-this-stages <gguf> <logmel.bin> <stem_out.bin>
//
// Computes one stage and dumps it for tools/cmp_beat_this_stages.py to score
// against the torch reference. Stage-by-stage rather than end-to-end: first
// divergence is the bug.
#include "beat_this.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static std::vector<float> read_f32(const char* p) {
    std::vector<float> v;
    FILE* f = fopen(p, "rb");
    if (!f)
        return v;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    v.resize((size_t)(n / 4));
    if (fread(v.data(), 4, v.size(), f) != v.size())
        v.clear();
    fclose(f);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <gguf> <logmel.bin> <stem_out.bin>\n", argv[0]);
        return 2;
    }
    auto lm = read_f32(argv[2]);
    if (lm.empty()) {
        fprintf(stderr, "cannot read logmel\n");
        return 1;
    }
    const int T = (int)(lm.size() / BEAT_THIS_MEL_BINS);

    beat_this_context* ctx = beat_this_init(argv[1], 4);
    if (!ctx) {
        fprintf(stderr, "init failed\n");
        return 1;
    }

    std::vector<float> stem((size_t)T * 32 * 32);
    const int n = beat_this_debug_stem(ctx, lm.data(), T, stem.data());
    printf("stem: %d elements (expect %d) for T=%d\n", n, T * 32 * 32, T);
    if (n <= 0) {
        beat_this_free(ctx);
        return 1;
    }

    FILE* f = fopen(argv[3], "wb");
    if (!f) {
        beat_this_free(ctx);
        return 1;
    }
    const int32_t hdr[4] = {T, 32, 32, 0}; // t, f, c
    fwrite(hdr, sizeof(int32_t), 4, f);
    fwrite(stem.data(), sizeof(float), (size_t)n, f);
    fclose(f);
    printf("wrote %s\n", argv[3]);
    beat_this_free(ctx);
    return 0;
}
