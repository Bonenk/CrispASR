#include "core/ctc_windows.h"
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

static void require(bool ok, const char* message) {
    if (!ok) {
        fprintf(stderr, "%s\n", message);
        exit(1);
    }
}
struct canary_ctc_context {};
static const float* input;
static int calls = 0, max_samples = 0, fail_call = -1;
static int canary_ctc_frame_dur_cs(canary_ctc_context*) {
    return 8;
}
static int canary_ctc_compute_logits(canary_ctc_context*, const float* samples, int n_samples, float** raw, int* t,
                                     int* v) {
    ++calls;
    max_samples = std::max(max_samples, n_samples);
    require((samples - input) % 1280 == 0, "encoder input is off frame grid");
    *t = n_samples / 1280 + 1;
    *v = 2;
    *raw = (float*)malloc(size_t(*t) * *v * sizeof(float));
    for (int i = 0; i < *t; ++i) {
        (*raw)[i * 2] = float((samples - input) / 1280 + i);
        (*raw)[i * 2 + 1] = -(*raw)[i * 2];
    }
    return calls == fail_call ? -7 : 0;
}
static std::vector<float> actual_dispatch(const float* samples, int n_samples, int chunk_seconds) {
    canary_ctc_context context;
    auto* actx = &context;
    std::vector<float> out;
    // PRODUCTION_DISPATCH
    return {ctc_logits, ctc_logits + size_t(T_ctc) * V_ctc};
}
static void check_rows(const std::vector<float>& rows, int count) {
    require(rows.size() == size_t(count) * 2, "wrong global row count");
    for (int i = 0; i < count; ++i)
        require(rows[i * 2] == i && rows[i * 2 + 1] == -i, "dropped, repeated or shifted global frame");
}
int main(int argc, char** argv) {
    require(argc == 2, "case required");
    const std::string test = argv[1];
    if (test.find("dispatch-") == 0) {
        const int n = test == "dispatch-short" ? 16000 : 600123;
        std::vector<float> pcm(n);
        input = pcm.data();
        if (test == "dispatch-failure")
            fail_call = 2;
        const auto rows = actual_dispatch(pcm.data(), n, test == "dispatch-disabled" ? 0 : 30);
        if (test == "dispatch-failure") {
            require(rows.empty() && calls == 2, "dispatch accepted partial output");
        } else {
            check_rows(rows, n / 1280 + 1);
            if (test == "dispatch-long")
                require(calls == 2 && max_samples <= 34 * 16000, "encoder was not bounded");
            else
                require(calls == 1 && max_samples == n, "short/disabled path changed");
        }
        return 0;
    }
    std::vector<float> pcm(87), rows{999};
    int frames = -1, vocab = -1;
    const int context = test == "endpoint" ? 0 : 2;
    const int n = test == "endpoint" ? 80 : 87;
    auto run = [&](const float* samples, int count, std::vector<float>& chunk, int& t, int& v) {
        ++calls;
        if (test == "failure" && calls == 2)
            return -7;
        require((samples - pcm.data()) % 4 == 0, "off-grid window start");
        require(count <= (5 + context * 2) * 4, "window call unbounded");
        t = count / 4 + 1;
        v = test == "shape" && calls == 2 ? 3 : 2;
        chunk.resize(size_t(t) * v);
        for (int i = 0; i < t; ++i) {
            chunk[i * v] = float((samples - pcm.data()) / 4 + i);
            chunk[i * v + 1] = -chunk[i * v];
        }
        return 0;
    };
    if (test == "invalid") {
        for (const int stride : {0, -1})
            require(core_ctc_windows::compute(pcm.data(), n, stride, 5, 2, run, rows, frames, vocab) < 0,
                    "invalid stride accepted");
        require(core_ctc_windows::compute(pcm.data(), n, 4, 0, 2, run, rows, frames, vocab) < 0, "zero core accepted");
        require(calls == 0 && rows.empty() && frames == 0 && vocab == 0, "invalid input reached encoder");
        return 0;
    }
    const int rc = core_ctc_windows::compute(pcm.data(), n, 4, 5, context, run, rows, frames, vocab);
    if (test == "failure" || test == "shape") {
        require(rc != 0 && rows.empty() && frames == 0 && vocab == 0, "failed run published partial logits");
    } else {
        require(rc == 0 && frames == n / 4 + 1 && vocab == 2, "wrong output geometry");
        check_rows(rows, frames);
    }
    return 0;
}
