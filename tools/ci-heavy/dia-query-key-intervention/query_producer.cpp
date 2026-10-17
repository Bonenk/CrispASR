// One-step CPU query producer using the literal pinned Dia projection/RoPE.
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>
struct dia_model {
    ggml_tensor* rope_timescale = nullptr;
};
static ggml_tensor* dia_rotary(ggml_context* ctx, ggml_tensor* value, ggml_tensor* positions, int dims, int mode,
                               const dia_model& model, bool source_rope) {
    if (!source_rope)
        return ggml_rope(ctx, value, positions, dims, mode);
    // The original divides positions by its generated timescale buffer. Base
    // one removes ggml's repeated geometric multiplication without a custom op.
    return ggml_rope_ext(ctx, value, positions, model.rope_timescale, dims, mode, 0, 1.0f, 1.0f, 0.0f, 1.0f, 32.0f,
                         1.0f);
}

static void require(bool ok, const char* message) {
    if (!ok) {
        std::fprintf(stderr, "REJECT: %s\n", message);
        std::exit(2);
    }
}
static void read(const char* path, ggml_tensor* tensor) {
    FILE* f = std::fopen(path, "rb");
    require(f, "open input");
    require(std::fread(tensor->data, 1, ggml_nbytes(tensor), f) == ggml_nbytes(tensor), "input size");
    require(std::fgetc(f) == EOF, "input trailing bytes");
    std::fclose(f);
    const auto* data = static_cast<const float*>(tensor->data);
    for (int64_t i = 0; i < ggml_nelements(tensor); ++i)
        require(std::isfinite(data[i]), "nonfinite input");
}
static void write(const std::filesystem::path& path, ggml_tensor* tensor) {
    FILE* f = std::fopen(path.c_str(), "wb");
    require(f, "open output");
    require(std::fwrite(tensor->data, 1, ggml_nbytes(tensor), f) == ggml_nbytes(tensor), "output bytes");
    require(std::fclose(f) == 0, "close output");
}
int main(int argc, char** argv) {
    require(argc == 7, "weight norm owned_pre_query fresh_output mode threads");
    const int mode = std::atoi(argv[5]), threads = std::atoi(argv[6]);
    require((mode == 0 || mode == 1) && (threads == 1 || threads == 4), "mode/threads");
    const std::filesystem::path out = argv[4];
    require(!std::filesystem::exists(out), "preserve evidence");
    std::filesystem::create_directories(out);
    auto* ctx = ggml_init({32 * 1024 * 1024, nullptr, false});
    require(ctx, "ggml context");
    auto* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 2048);
    read(argv[1], weight);
    auto* normalized = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2048, 1, 2);
    read(argv[2], normalized);
    auto* owned = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 16, 1, 2);
    read(argv[3], owned);
    auto* position = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    *static_cast<int32_t*>(position->data) = 978;
    ggml_set_input(normalized);
    ggml_set_input(owned);
    ggml_set_input(position);
    auto* pre = ggml_mul_mat(ctx, weight, normalized);
    pre = ggml_reshape_4d(ctx, pre, 128, 16, 1, 2);
    auto* used = mode == 0 ? pre : owned;
    dia_model model;
    auto* post = dia_rotary(ctx, used, position, 128, 2, model, false);
    auto* graph = ggml_new_graph(ctx);
    for (auto* t : {pre, used, post}) {
        ggml_set_output(t);
        ggml_build_forward_expand(graph, t);
    }
    auto plan = ggml_graph_plan(graph, threads, nullptr);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.data();
    require(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS, "actual ggml compute");
    write(out / "pre_rope.f32", pre);
    write(out / "used_pre_rope.f32", used);
    write(out / "post_rope.f32", post);
    ggml_free(ctx);
    std::printf("DIA_QUERY_PRODUCER_PASS mode=%d threads=%d step=978 actual_ggml=1\n", mode, threads);
}
