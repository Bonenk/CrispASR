// Review probe for shared ggml PR2. Requires its build option ON.
// Checks actual lowering, both numerical arms, batches, stride and dilation.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool run(ggml_backend_t backend, int k, int ic, int oc, int batch, int stride, int pad, int dilation,
                ggml_type type, bool large, bool direct, bool strided = false) {
    setenv("GGML_CONV1D_VIA_CONV2D", direct ? "1" : "0", 1);
    const int width = 23;
    ggml_context* ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    ggml_context* input_ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    auto* w = ggml_new_tensor_3d(input_ctx, type, k, ic, oc);
    const int storage_width = width + (strided ? 2 : 0);
    auto* storage = ggml_new_tensor_3d(input_ctx, GGML_TYPE_F32, storage_width, ic, batch);
    auto* x = strided ? ggml_view_3d(ctx, storage, width, ic, batch, storage->nb[1], storage->nb[2], 0) : storage;
    ggml_set_input(storage);
    ggml_set_input(w);
    auto* y = ggml_conv_1d(ctx, w, x, stride, pad, dilation);
    ggml_set_output(y);
    auto* graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(graph, y);
    bool lowered = false;
    for (int n = 0; n < ggml_graph_n_nodes(graph); ++n)
        lowered |= ggml_graph_node(graph, n)->op == GGML_OP_CONV_2D;
    if (lowered != (direct && !strided && (type == GGML_TYPE_F16 || type == GGML_TYPE_F32))) {
        std::fprintf(stderr, "lowering gate did not reach graph\n");
        std::exit(2);
    }
    auto buffer = ggml_backend_alloc_ctx_tensors(input_ctx, backend);
    if (!buffer)
        std::exit(3);
    std::vector<float> weights(k * ic * oc), inputs(storage_width * ic * batch);
    std::vector<ggml_fp16_t> half(weights.size());
    std::vector<ggml_bf16_t> bfloat(weights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = large ? 1.0f : std::sin(float(i + 1) * .37f) * .2f;
        half[i] = ggml_fp32_to_fp16(weights[i]);
        bfloat[i] = ggml_fp32_to_bf16(weights[i]);
        if (type == GGML_TYPE_F16)
            weights[i] = ggml_fp16_to_fp32(half[i]);
        else if (type == GGML_TYPE_BF16)
            weights[i] = ggml_bf16_to_fp32(bfloat[i]);
    }
    for (size_t i = 0; i < inputs.size(); ++i)
        inputs[i] = large ? 40000.0f : std::cos(float(i + 1) * .23f);
    const void* weight_data = type == GGML_TYPE_F16    ? static_cast<const void*>(half.data())
                              : type == GGML_TYPE_BF16 ? static_cast<const void*>(bfloat.data())
                                                       : static_cast<const void*>(weights.data());
    ggml_backend_tensor_set(w, weight_data, 0, ggml_nbytes(w));
    ggml_backend_tensor_set(storage, inputs.data(), 0, ggml_nbytes(storage));
    // Preallocate leaves only, so the normal scheduler can place unsupported
    // GPU operations on CPU and allocate their output buffers appropriately.
    ggml_backend_t cpu = ggml_backend_is_cpu(backend) ? nullptr : ggml_backend_cpu_init();
    if (cpu)
        ggml_backend_cpu_set_n_threads(cpu, 2);
    ggml_backend_t backends[] = {backend, cpu};
    auto sched = ggml_backend_sched_new(backends, nullptr, cpu ? 2 : 1, 256, false, true);
    if (!sched || !ggml_backend_sched_alloc_graph(sched, graph))
        std::exit(4);
    int cpu_fallbacks = 0;
    for (int n = 0; n < ggml_graph_n_nodes(graph); ++n) {
        auto* node = ggml_graph_node(graph, n);
        if (node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE ||
            node->op == GGML_OP_TRANSPOSE)
            continue;
        auto selected = ggml_backend_sched_get_tensor_backend(sched, node);
        const bool supported = ggml_backend_supports_op(backend, node);
        if (!selected || !ggml_backend_supports_op(selected, node) || (!supported && selected == backend))
            std::exit(6);
        cpu_fallbacks += selected == cpu;
        std::printf("placement op=%s requested=%s selected=%s requested_support=%d\n", ggml_op_name(node->op),
                    ggml_backend_name(backend), ggml_backend_name(selected), supported);
    }
    if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS)
        std::exit(4);
    std::vector<float> output(ggml_nelements(y));
    ggml_backend_tensor_get(y, output.data(), 0, ggml_nbytes(y));
    double squared_error = 0, squared_ref = 0, max_abs = 0;
    bool finite = true;
    for (int n = 0; n < batch; ++n)
        for (int c = 0; c < oc; ++c)
            for (int t = 0; t < y->ne[0]; ++t) {
                double expected = 0;
                for (int ch = 0; ch < ic; ++ch)
                    for (int j = 0; j < k; ++j) {
                        int pos = t * stride + j * dilation - pad;
                        if (pos >= 0 && pos < width)
                            expected +=
                                double(inputs[(n * ic + ch) * storage_width + pos]) * weights[(c * ic + ch) * k + j];
                    }
                float actual = output[(n * oc + c) * y->ne[0] + t];
                finite &= std::isfinite(actual);
                double error = actual - expected;
                squared_error += error * error;
                squared_ref += expected * expected;
                max_abs = std::max(max_abs, std::abs(error));
            }
    double rel = std::sqrt(squared_error / std::max(squared_ref, 1e-30));
    bool pass = finite && rel < .002 && (large || max_abs < .003);
    std::printf(
        "case k=%d ic=%d oc=%d batch=%d s=%d p=%d d=%d type=%s large=%d direct=%d strided=%d finite=%d rel_l2=%.9g "
        "max_abs=%.9g cpu_fallbacks=%d splits=%d pass=%d\n",
        k, ic, oc, batch, stride, pad, dilation, ggml_type_name(type), large, direct, strided, finite, rel, max_abs,
        cpu_fallbacks, ggml_backend_sched_get_n_splits(sched), pass);
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_free(input_ctx);
    if (cpu)
        ggml_backend_free(cpu);
    return pass;
}

int main(int argc, char** argv) {
    // Keep complete case records intact when merged with backend stderr.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc != 2 && argc != 3)
        return 2;
    const bool fallback_only = argc == 3 && std::strcmp(argv[2], "--fallback-only") == 0;
    const bool bf16_only = argc == 3 && std::strcmp(argv[2], "--bf16-fallback-only") == 0;
    if (argc == 3 && !fallback_only && !bf16_only)
        return 2;
    ggml_backend_load_all();
    auto backend = ggml_backend_init_by_name(argv[1], nullptr);
    if (!backend)
        return 3;
    if (ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU)
        ggml_backend_cpu_set_n_threads(backend, 2);
    int count = 0, failures = 0;
    if (!fallback_only && !bf16_only) {
        for (auto type : {GGML_TYPE_F16, GGML_TYPE_F32})
            for (int batch : {1, 3})
                for (int k : {1, 3, 5})
                    for (int stride : {1, 2})
                        for (int dilation : {1, 2})
                            for (bool direct : {false, true}) {
                                failures += !run(backend, k, 3, 2, batch, stride, k / 2, dilation, type, false, direct);
                                ++count;
                            }
        for (auto type : {GGML_TYPE_F16, GGML_TYPE_F32})
            for (bool direct : {false, true}) {
                failures += !run(backend, 3, 3, 2, 3, 1, 1, 1, type, true, direct);
                ++count;
            }
    }
    if (fallback_only) {
        for (auto type : {GGML_TYPE_F16, GGML_TYPE_F32})
            for (bool direct : {false, true}) {
                failures += !run(backend, 3, 3, 2, 3, 2, 1, 2, type, false, direct, true);
                ++count;
            }
    }
    if (bf16_only) {
        for (bool strided : {false, true})
            for (bool direct : {false, true}) {
                failures += !run(backend, 3, 3, 2, 3, 2, 1, 2, GGML_TYPE_BF16, false, direct, strided);
                ++count;
            }
    }
    std::printf("CONV1D_PR2_REVIEW backend=%s cases=%d failed=%d\n", argv[1], count, failures);
    ggml_backend_free(backend);
    return failures ? 1 : 0;
}
