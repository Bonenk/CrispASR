// CPU-only literal Dia producer diagnostic; no production/default changes.
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>
struct dia_model {
    uint32_t n_output_heads = 9;
    struct {
        std::vector<ggml_tensor*> embeddings;
    } decoder;
    ggml_tensor* rope_timescale = nullptr;
    float rms_norm_eps = 1e-5f;
};
static ggml_tensor* dia_rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, float eps) {
    x = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, x, weight);
}
static ggml_tensor* dia_rotary(ggml_context* ctx, ggml_tensor* value, ggml_tensor* positions, int dims, int mode,
                               const dia_model& model, bool source_rope) {
    if (!source_rope)
        return ggml_rope(ctx, value, positions, dims, mode);
    // The original divides positions by its generated timescale buffer. Base
    // one removes ggml's repeated geometric multiplication without a custom op.
    return ggml_rope_ext(ctx, value, positions, model.rope_timescale, dims, mode, 0, 1.0f, 1.0f, 0.0f, 1.0f, 32.0f,
                         1.0f);
}
static ggml_tensor* build_dia_decoder_embedding(ggml_context* ctx, dia_model& m,
                                                ggml_tensor* audio_tokens // (n_output_heads * 2,) I32
) {
    const int B = 2;
    ggml_tensor* emb = nullptr;

    for (int i = 0; i < (int)m.n_output_heads; i++) {
        // Stride view: pick tokens for this codebook (interleaved cond+uncond)
        ggml_tensor* view = ggml_view_1d(ctx, audio_tokens, B, i * ggml_element_size(audio_tokens));
        view->nb[0] = m.n_output_heads * ggml_element_size(audio_tokens);

        // §232 GPU: ggml_get_rows requires a CONTIGUOUS index tensor on CUDA
        // (GGML_ASSERT(src1->nb[0] == ggml_type_size) — the strided view above
        // aborts the CUDA decode; CPU/Metal tolerate the stride). ggml_cont
        // materialises the 2 indices contiguously — negligible cost, correct on
        // every backend.
        ggml_tensor* idx = ggml_cont(ctx, view);
        ggml_tensor* e = ggml_get_rows(ctx, m.decoder.embeddings[i], idx);
        emb = (i == 0) ? e : ggml_add(ctx, emb, e);
    }
    return emb;
}
static void require(bool ok, const char* msg) {
    if (!ok) {
        fprintf(stderr, "REJECT: %s\n", msg);
        exit(2);
    }
}
static void read(const std::filesystem::path& path, ggml_tensor* tensor) {
    FILE* f = fopen(path.c_str(), "rb");
    require(f, "input open");
    require(fread(tensor->data, 1, ggml_nbytes(tensor), f) == ggml_nbytes(tensor), "input size");
    require(fgetc(f) == EOF, "input trailing bytes");
    fclose(f);
}
int main(int argc, char** argv) {
    require(argc == 6, "inputs native_inputs fresh_output mode threads");
    std::filesystem::path inputs = argv[1], native = argv[2], out = argv[3];
    int mode = atoi(argv[4]), threads = atoi(argv[5]);
    require(mode >= 0 && mode <= 2 && (threads == 1 || threads == 4), "mode/thread request");
    require(!std::filesystem::exists(out), "preserve evidence");
    std::filesystem::create_directories(out);
    auto* ctx0 = ggml_init({128 * 1024 * 1024, nullptr, false});
    require(ctx0, "ggml context");
    dia_model m;
    for (int i = 0; i < 9; ++i) {
        auto* w = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2048, 1028);
        read(inputs / "coefficients" / ("decoder.embeddings." + std::to_string(i) + ".weight.f32"), w);
        m.decoder.embeddings.push_back(w);
    }
    auto* weight = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2048, 512);
    read(native / "key-weight.f32", weight);
    auto* gamma = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 2048);
    read(inputs / "coefficients/pre_sa_norm.weight.original.f32", gamma);
    auto* audio_tokens_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 18);
    ggml_set_input(audio_tokens_t);
    auto* dec_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_input(dec_pos);
    auto* owned_norm = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 2048, 1, 2);
    ggml_set_input(owned_norm);
    auto* owned_key = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 128, 4, 1, 2);
    ggml_set_input(owned_key);
    auto* hidden = ggml_reshape_3d(ctx0, build_dia_decoder_embedding(ctx0, m, audio_tokens_t), 2048, 1, 2);
    auto* norm = dia_rms_norm(ctx0, hidden, gamma, m.rms_norm_eps);
    auto* pre = ggml_mul_mat(ctx0, weight, mode == 1 ? owned_norm : norm);
    auto* key = mode == 2 ? owned_key : ggml_reshape_4d(ctx0, pre, 128, 4, 1, 2);
    auto* post = dia_rotary(ctx0, key, dec_pos, 128, 2, m, false);
    std::vector<std::pair<std::string, ggml_tensor*>> outputs = {
        {"hidden", hidden}, {"normalized", norm}, {"pre_rope", pre}, {"post_rope", post}};
    auto* graph = ggml_new_graph(ctx0);
    for (auto& v : outputs) {
        ggml_set_output(v.second);
        ggml_build_forward_expand(graph, v.second);
    }
    auto plan = ggml_graph_plan(graph, threads, nullptr);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.data();
    std::vector<FILE*> streams;
    for (auto& v : outputs) {
        FILE* f = fopen((out / (v.first + ".f32")).c_str(), "wb");
        require(f, "output open");
        streams.push_back(f);
    }
    FILE* f = fopen((native / "tokens.i32").c_str(), "rb");
    require(f, "tokens");
    FILE* fn = fopen((native / "normalized.f32").c_str(), "rb");
    require(fn, "norms");
    FILE* fk = fopen((native / "pre_rope.f32").c_str(), "rb");
    require(fk, "keys");
    for (int step = 0; step < 979; ++step) {
        auto* ids = (int32_t*)audio_tokens_t->data;
        require(fread(ids, 4, 9, f) == 9, "token row");
        for (int c = 0; c < 9; ++c)
            ids[c + 9] = ids[c];
        require(fread(owned_norm->data, 4, 4096, fn) == 4096, "norm row");
        require(fread(owned_key->data, 4, 1024, fk) == 1024, "key row");
        *(int32_t*)dec_pos->data = step;
        require(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS, "graph compute");
        for (size_t i = 0; i < outputs.size(); ++i)
            require(fwrite(outputs[i].second->data, 1, ggml_nbytes(outputs[i].second), streams[i]) ==
                        ggml_nbytes(outputs[i].second),
                    "output bytes");
    }
    require(fgetc(f) == EOF && fgetc(fn) == EOF && fgetc(fk) == EOF, "complete retained history");
    fclose(f);
    fclose(fn);
    fclose(fk);
    for (auto* stream : streams)
        fclose(stream);
    ggml_free(ctx0);
    printf("PRODUCER_PASS mode=%d threads=%d steps=979 actual_ggml=1\n", mode, threads);
}
