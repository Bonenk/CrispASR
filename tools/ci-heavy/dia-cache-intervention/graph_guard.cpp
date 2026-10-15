// Reproduce the actual pruned-leaf/name-mutation conditions without model weights.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cstdio>

int main() {
    auto* backend = ggml_backend_cpu_init();
    if (!backend)
        return 2;
    auto* ctx = ggml_init({1024 * 1024, nullptr, true});
    if (!ctx)
        return 3;
    auto* past = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    ggml_set_name(past, "past_k_0");
    ggml_set_input(past);
    auto* current = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
    ggml_set_name(current, "new_k_0");
    ggml_set_input(current);
    ggml_set_output(current);
    auto* keys = ggml_concat(ctx, past, current, 1);
    keys = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    auto* retained = keys;
    ggml_set_input(keys);
    ggml_set_name(keys, "dia_intervention_full_key");
    // The existing attention capture renames this same tensor.
    ggml_set_name(keys, "dia_decoder_sa0_k");
    ggml_set_output(keys);
    auto* values = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    ggml_set_name(values, "past_v_0");
    ggml_set_input(values);
    auto* result = ggml_add(ctx, keys, values);
    ggml_set_output(result);
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    ggml_build_forward_expand(graph, current);
    auto* allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(allocator, graph))
        return 4;
    // Both obsolete lookups are absent: the previous diagnostic would pass a
    // null tensor to tensor_set before it could reach a valid intervention.
    if (ggml_graph_get_tensor(graph, "past_k_0") || ggml_graph_get_tensor(graph, "dia_intervention_full_key") ||
        ggml_graph_get_tensor(graph, "dia_decoder_sa0_k") != retained)
        return 5;
    const float expected[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const float zero[12] = {};
    ggml_backend_tensor_set(current, expected + 8, 0, 4 * sizeof(float));
    ggml_backend_tensor_set(values, zero, 0, sizeof(zero));
    // Skip only the unused past key, and use the retained new input pointer.
    ggml_backend_tensor_set(retained, expected, 0, sizeof(expected));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
        return 6;
    float actual[12];
    ggml_backend_tensor_get(result, actual, 0, sizeof(actual));
    for (int i = 0; i < 12; ++i)
        if (actual[i] != expected[i])
            return 7;
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    ggml_backend_free(backend);
    std::puts("DIA_INTERVENTION_GRAPH_GUARD_PASS: pruned past key absent, old input name absent, retained pointer "
              "output exact");
    return 0;
}
