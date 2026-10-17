// Standalone exact-input attention probe; no neural model is loaded.
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

static void require(bool ok, const char* message) {
    if (!ok) {
        std::fprintf(stderr, "REJECT: %s\n", message);
        std::exit(2);
    }
}
static void read(const std::filesystem::path& file, ggml_tensor* t) {
    FILE* f = std::fopen(file.c_str(), "rb");
    require(f, "open input");
    require(std::fread(t->data, 1, ggml_nbytes(t), f) == ggml_nbytes(t), "input bytes");
    require(std::fgetc(f) == EOF, "trailing bytes");
    std::fclose(f);
    const auto* x = static_cast<const float*>(t->data);
    for (int64_t i = 0; i < ggml_nelements(t); ++i)
        require(std::isfinite(x[i]), "nonfinite input");
}
static void write(const std::filesystem::path& file, ggml_tensor* t) {
    FILE* f = std::fopen(file.c_str(), "wb");
    require(f, "open output");
    require(std::fwrite(t->data, 1, ggml_nbytes(t), f) == ggml_nbytes(t), "output bytes");
    require(std::fclose(f) == 0, "close output");
}
int main(int argc, char** argv) {
    require(argc == 10, "input fresh_output dim qheads kvheads time batch mode threads");
    const std::filesystem::path input = argv[1], output = argv[2];
    const int head_dim = std::atoi(argv[3]), n_heads = std::atoi(argv[4]), n_kv_heads = std::atoi(argv[5]);
    const int T_full = std::atoi(argv[6]), B = std::atoi(argv[7]);
    const int mode = std::atoi(argv[8]), threads = std::atoi(argv[9]);
    require(head_dim > 0 && head_dim <= 128 && n_heads > 0 && n_heads <= 16 && n_kv_heads > 0 &&
                n_heads % n_kv_heads == 0 && T_full > 0 && T_full <= 979 && B > 0 && B <= 2,
            "dimensions");
    require(mode >= 0 && mode <= 2 && (threads == 1 || threads == 4), "mode/threads");
    require(!std::filesystem::exists(output), "preserve evidence");
    std::filesystem::create_directories(output);
    auto* ctx0 = ggml_init({128 * 1024 * 1024, nullptr, false});
    require(ctx0, "context");
    auto* Q = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, head_dim, n_heads, 1, B);
    auto* K_full = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, head_dim, n_kv_heads, T_full, B);
    auto* V_full = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, head_dim, n_kv_heads, T_full, B);
    read(input / "q.f32", Q);
    read(input / "k.f32", K_full);
    read(input / "v.f32", V_full);
    auto* owned = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, T_full, 1, n_heads, B);
    if (mode)
        read(input / (mode == 1 ? "shadow-scores.f32" : "shadow-probabilities.f32"), owned);
    for (auto* t : {Q, K_full, V_full, owned})
        ggml_set_input(t);

    // The following operations match the pinned runtime's explicit self-attention.
    const int n_rep = n_heads / n_kv_heads;
    if (n_rep > 1) {
        K_full = ggml_reshape_4d(ctx0, K_full, head_dim, 1, n_kv_heads, T_full * B);
        K_full = ggml_repeat_4d(ctx0, K_full, head_dim, n_rep, n_kv_heads, T_full * B);
        K_full = ggml_cont(ctx0, ggml_reshape_4d(ctx0, K_full, head_dim, n_heads, T_full, B));
        V_full = ggml_reshape_4d(ctx0, V_full, head_dim, 1, n_kv_heads, T_full * B);
        V_full = ggml_repeat_4d(ctx0, V_full, head_dim, n_rep, n_kv_heads, T_full * B);
        V_full = ggml_cont(ctx0, ggml_reshape_4d(ctx0, V_full, head_dim, n_heads, T_full, B));
    }
    auto* q_perm = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
    auto* k_perm = ggml_cont(ctx0, ggml_permute(ctx0, K_full, 0, 2, 1, 3));
    auto* scores = ggml_mul_mat(ctx0, k_perm, q_perm);
    auto* used_scores = mode == 1 ? owned : scores;
    auto* probabilities = ggml_soft_max(ctx0, used_scores);
    auto* used_probabilities = mode == 2 ? owned : probabilities;
    auto* v_perm = ggml_cont(ctx0, ggml_permute(ctx0, V_full, 0, 2, 1, 3));
    auto* v_t = ggml_cont(ctx0, ggml_transpose(ctx0, v_perm));
    auto* weighted = ggml_mul_mat(ctx0, v_t, used_probabilities);
    weighted = ggml_cont(ctx0, ggml_permute(ctx0, weighted, 0, 2, 1, 3));
    auto* graph = ggml_new_graph(ctx0);
    for (auto* t : {scores, used_scores, probabilities, used_probabilities, weighted}) {
        ggml_set_output(t);
        ggml_build_forward_expand(graph, t);
    }
    auto plan = ggml_graph_plan(graph, threads, nullptr);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.data();
    require(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS, "actual ggml compute");
    write(output / "scores.f32", scores);
    write(output / "used-scores.f32", used_scores);
    write(output / "probabilities.f32", probabilities);
    write(output / "used-probabilities.f32", used_probabilities);
    write(output / "weighted.f32", weighted);
    ggml_free(ctx0);
    std::printf("DIA_ATTENTION_ARITHMETIC_PASS mode=%d threads=%d\n", mode, threads);
}
