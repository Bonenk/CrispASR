// Metal-only hot matmul probe using the actual OmniVoice layer-0 FFN-down weight.
// Usage: omnivoice-matmul-bench OMNIVOICE_Q8_0_GGUF
// Arm 0 uses serialized Q8_0; arm 1 widens to F32 on CPU then rounds to F16.
// Tests two token counts with eight total warmups, then ten measured samples per arm.
// This measures one op, not generation. It does not publish/replace model files.
#define GGML_MAX_NAME 128
#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-metal.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <fstream>
#include <vector>
static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: omnivoice-matmul-bench OMNIVOICE_Q8_0_GGUF\n");
        return 2;
    }
    ggml_context* metadata = nullptr;
    gguf_init_params gp = {true, &metadata};
    auto* file = gguf_init_from_file(argv[1], gp);
    if (!file)
        return 3;
    const char* name = "llm.blk.0.ffn_down.weight";
    auto* src = ggml_get_tensor(metadata, name);
    int ti = gguf_find_tensor(file, name);
    if (!src || ti < 0 || src->type != GGML_TYPE_Q8_0)
        return 4;
    const auto K = src->ne[0], M = src->ne[1];
    std::vector<uint8_t> raw(ggml_nbytes(src));
    std::ifstream in(argv[1], std::ios::binary);
    in.seekg(gguf_get_data_offset(file) + gguf_get_tensor_offset(file, ti));
    in.read((char*)raw.data(), raw.size());
    if (!in)
        return 5;
    std::vector<float> values(K * M);
    ggml_get_type_traits(src->type)->to_float(raw.data(), values.data(), values.size());
    std::vector<ggml_fp16_t> half(values.size());
    ggml_fp32_to_fp16_row(values.data(), half.data(), values.size());
    auto* gpu = ggml_backend_metal_init();
    if (!gpu)
        return 6;
    ggml_init_params wp = {4 * ggml_tensor_overhead(), nullptr, true};
    auto* wc = ggml_init(wp);
    auto* q = ggml_new_tensor_2d(wc, GGML_TYPE_Q8_0, K, M);
    auto* h = ggml_new_tensor_2d(wc, GGML_TYPE_F16, K, M);
    auto* wb = ggml_backend_alloc_ctx_tensors(wc, gpu);
    if (!wb)
        return 7;
    ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(q, raw.data(), 0, raw.size());
    ggml_backend_tensor_set(h, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
    for (int T : {150, 450}) {
        struct G {
            ggml_context* c;
            ggml_tensor *x, *y;
            ggml_cgraph* g;
            ggml_gallocr_t a;
        };
        G graphs[2];
        std::vector<float> x(K * T);
        for (size_t i = 0; i < x.size(); i++)
            x[i] = std::sin((float)i * .0017f);
        for (int arm = 0; arm < 2; arm++) {
            auto& g = graphs[arm];
            ggml_init_params ip = {1024 * 1024, nullptr, true};
            g.c = ggml_init(ip);
            g.x = ggml_new_tensor_2d(g.c, GGML_TYPE_F32, K, T);
            ggml_set_input(g.x);
            g.y = ggml_mul_mat(g.c, arm ? h : q, g.x);
            ggml_set_output(g.y);
            g.g = ggml_new_graph(g.c);
            ggml_build_forward_expand(g.g, g.y);
            g.a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu));
            if (!ggml_gallocr_alloc_graph(g.a, g.g))
                return 8;
        }
        std::vector<float> reference, actual(M * T);
        std::vector<double> times[2], cpu_times[2];
        for (int i = 0; i < 28; i++) {
            int arm = (i % 4 == 0 || i % 4 == 3) ? 0 : 1;
            auto& g = graphs[arm];
            ggml_backend_tensor_set(g.x, x.data(), 0, x.size() * sizeof(float));
            const auto cpu_start = std::clock();
            double t = now();
            auto rc = ggml_backend_graph_compute(gpu, g.g);
            double dt = now() - t;
            const double cpu_s = (double)(std::clock() - cpu_start) / CLOCKS_PER_SEC;
            if (rc != GGML_STATUS_SUCCESS)
                return 9;
            ggml_backend_tensor_get(g.y, actual.data(), 0, actual.size() * sizeof(float));
            if (i == 0)
                reference = actual;
            double dot = 0, aa = 0, bb = 0, maxabs = 0;
            bool exact = std::memcmp(reference.data(), actual.data(), actual.size() * sizeof(float)) == 0;
            for (size_t j = 0; j < actual.size(); j++) {
                if (!std::isfinite(actual[j]))
                    return 10;
                dot += (double)reference[j] * actual[j];
                aa += (double)reference[j] * reference[j];
                bb += (double)actual[j] * actual[j];
                maxabs = std::max(maxabs, std::abs((double)actual[j] - reference[j]));
            }
            if (aa < 1e-12 || bb < 1e-12)
                return 11;
            if (i >= 8) {
                times[arm].push_back(dt);
                cpu_times[arm].push_back(cpu_s);
            }
            std::printf("SAMPLE T=%d i=%d arm=%d warm=%d ms=%.6f exact=%d maxabs=%.9g cosine=%.12f refnorm=%.9g "
                        "norm=%.9g cpu_ms=%.6f\n",
                        T, i, arm, i < 8, dt * 1000, exact, maxabs, dot / std::sqrt(aa * bb), std::sqrt(aa),
                        std::sqrt(bb), cpu_s * 1000);
            std::fflush(stdout);
        }
        for (int arm = 0; arm < 2; arm++) {
            auto& v = times[arm];
            std::sort(v.begin(), v.end());
            std::sort(cpu_times[arm].begin(), cpu_times[arm].end());
            std::printf("SUMMARY T=%d arm=%d median_ms=%.6f samples=%zu cpu_ms=%.6f\n", T, arm, (v[4] + v[5]) * 500,
                        v.size(), (cpu_times[arm][4] + cpu_times[arm][5]) * 500);
            auto& g = graphs[arm];
            ggml_gallocr_free(g.a);
            ggml_free(g.c);
        }
    }
    ggml_backend_buffer_free(wb);
    ggml_free(wc);
    ggml_backend_free(gpu);
    gguf_free(file);
    ggml_free(metadata);
}
