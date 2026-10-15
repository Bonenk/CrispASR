#include "prefix.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include <cstdio>

int main() {
    std::vector<float> full(24), packed;
    for (int b = 0; b < 2; ++b)
        for (int t = 0; t < 4; ++t)
            for (int d = 0; d < 3; ++d)
                full[(b * 4 + t) * 3 + d] = float(100 * b + 10 * t + d);
    if (!dia_suffix_prefix(full, 2, 4, 3, packed) || packed.size() != 18)
        return 2;
    for (int t = 0; t < 3; ++t)
        for (int b = 0; b < 2; ++b)
            for (int d = 0; d < 3; ++d)
                if (packed[(t * 2 + b) * 3 + d] != float(100 * b + 10 * t + d))
                    return 3;
    if (dia_suffix_prefix(full, 2, 1, 3, packed) || dia_suffix_prefix(full, 0, 4, 3, packed) ||
        dia_suffix_prefix(full, 2, 4, 2, packed))
        return 4;
    auto* cpu = ggml_backend_cpu_init();
    auto* ctx = ggml_init({1024 * 1024, nullptr, true});
    if (!cpu || !ctx)
        return 5;
    auto* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_input(x);
    auto* computed = ggml_scale(ctx, x, 2.0f);
    ggml_set_output(computed);
    auto* replacement = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_input(replacement);
    ggml_set_output(replacement);
    ggml_set_name(replacement, "dia_suffix_activation");
    auto* held = replacement;
    ggml_set_name(replacement, "dia_decoder_layer_0");
    auto* result = ggml_scale(ctx, replacement, 3.0f);
    ggml_set_output(result);
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    ggml_build_forward_expand(graph, computed);
    auto* alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    if (!ggml_gallocr_alloc_graph(alloc, graph) || ggml_graph_get_tensor(graph, "dia_decoder_layer_0") != held ||
        ggml_graph_get_tensor(graph, "dia_suffix_activation"))
        return 6;
    const float input[4] = {1, 2, 3, 4}, owned[4] = {2, 4, 6, 8};
    ggml_backend_tensor_set(x, input, 0, sizeof(input));
    ggml_backend_tensor_set(held, owned, 0, sizeof(owned));
    if (ggml_backend_graph_compute(cpu, graph) != GGML_STATUS_SUCCESS)
        return 7;
    float a[4], b[4], c[4];
    ggml_backend_tensor_get(result, a, 0, sizeof(a));
    ggml_backend_tensor_get(computed, b, 0, sizeof(b));
    ggml_backend_tensor_get(held, c, 0, sizeof(c));
    for (int i = 0; i < 4; ++i)
        if (a[i] != 3 * owned[i] || b[i] != owned[i] || c[i] != owned[i])
            return 8;
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_free(cpu);
    std::puts("DIA_SUFFIX_GUARD_PASS: prefix excludes current, batch/time exact, renamed owned input and computed root "
              "survive");
}
