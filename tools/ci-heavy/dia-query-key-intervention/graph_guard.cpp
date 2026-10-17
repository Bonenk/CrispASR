// Paired replacement leaves survive capture renaming; discarded producers stay pruned.
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
    auto* old_q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
    ggml_set_name(old_q, "discarded_query_producer");
    auto* q = ggml_scale(ctx, old_q, 2.0f);
    q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
    auto* held_q = q;
    ggml_set_input(q);
    ggml_set_name(q, "dia_intervention_current_query");
    ggml_set_name(q, "dia_decoder_sa0_q");
    auto* k = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    auto* held_k = k;
    ggml_set_input(k);
    ggml_set_name(k, "dia_intervention_full_key");
    ggml_set_name(k, "dia_decoder_sa0_k");
    auto* output = ggml_mul_mat(ctx, k, q);
    ggml_set_output(q);
    ggml_set_output(k);
    ggml_set_output(output);
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    auto* allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(allocator, graph))
        return 4;
    if (ggml_graph_get_tensor(graph, "discarded_query_producer") ||
        ggml_graph_get_tensor(graph, "dia_intervention_current_query") ||
        ggml_graph_get_tensor(graph, "dia_intervention_full_key") ||
        ggml_graph_get_tensor(graph, "dia_decoder_sa0_q") != held_q ||
        ggml_graph_get_tensor(graph, "dia_decoder_sa0_k") != held_k)
        return 5;
    const float queries[4] = {1, 2, 3, 4};
    const float keys[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const float expected[3] = {30, 70, 110};
    ggml_backend_tensor_set(held_q, queries, 0, sizeof(queries));
    ggml_backend_tensor_set(held_k, keys, 0, sizeof(keys));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
        return 6;
    float actual[3];
    ggml_backend_tensor_get(output, actual, 0, sizeof(actual));
    for (int i = 0; i < 3; ++i)
        if (actual[i] != expected[i])
            return 7;
    float actual_q[4], actual_k[12];
    ggml_backend_tensor_get(held_q, actual_q, 0, sizeof(actual_q));
    ggml_backend_tensor_get(held_k, actual_k, 0, sizeof(actual_k));
    for (int i = 0; i < 4; ++i)
        if (actual_q[i] != queries[i])
            return 8;
    for (int i = 0; i < 12; ++i)
        if (actual_k[i] != keys[i])
            return 9;
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    ggml_backend_free(backend);
    std::puts(
        "DIA_QUERY_KEY_GRAPH_GUARD_PASS: renamed Q/K leaves, pruned producer, consumed inputs and dot products exact");
    return 0;
}
