#include <catch2/catch_test_macros.hpp>
#include "core/model_weight_usage.h"
#include "ggml-cann/weight_layout.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>

TEST_CASE("CANN names are bounded suffixes and cover fused projections", "[unit][weight-usage]") {
    ggml_context* ctx = ggml_init({16384, nullptr, true});
    REQUIRE(ctx);
    auto* t = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 32, 48);
    for (const char* name :
         {"output.weight", "blk.0.attn_q.weight", "audio.blk.0.attn.q.weight", "model.layers.0.attn.qkv.weight",
          "blk.1.attn_qkv.weight", "llm.lm_head.weight", "model.layers.0.ffn.down.weight"}) {
        ggml_set_name(t, name);
        INFO(name);
        CHECK(ggml_cann_nz_candidate(t));
    }
    for (const char* name : {"token_embd.weight", "output.weight.bias", "notoutput.weight", "attn.q.weight.extra",
                             "myattn.q.weight", "blk.0.attn.q.bias", "norm.weight", "conv.weight"}) {
        ggml_set_name(t, name);
        INFO(name);
        CHECK_FALSE(ggml_cann_nz_candidate(t));
    }
    ggml_set_name(t, "arbitrary.name");
    t->flags = GGML_TENSOR_FLAG_WEIGHT_MATMUL_ONLY;
    CHECK(ggml_cann_nz_candidate(t));
    t->flags |= GGML_TENSOR_FLAG_WEIGHT_GENERAL;
    CHECK_FALSE(ggml_cann_nz_candidate(t));
    t->flags = GGML_TENSOR_FLAG_WEIGHT_MATMUL_ONLY;
    t->type = GGML_TYPE_BF16;
    CHECK_FALSE(ggml_cann_nz_candidate(t));
    t->type = GGML_TYPE_Q8_0;
    CHECK_FALSE(ggml_cann_nz_candidate(t));
    t->type = GGML_TYPE_F16;
    t->ne[2] = 2;
    CHECK_FALSE(ggml_cann_nz_candidate(t));
    t->ne[2] = 1;
    t->view_src = t;
    CHECK_FALSE(ggml_cann_nz_candidate(t));
    t->view_src = nullptr;
    t->op = GGML_OP_MUL_MAT;
    CHECK_FALSE(ggml_cann_nz_candidate(t));
    ggml_free(ctx);
}

TEST_CASE("Model policies exclude embeddings and support current and legacy MiMo", "[unit][weight-usage]") {
    using core_gguf::WeightUsage;
    for (const char* name : {"model.layers.0.attn.qkv.weight", "model.layers.3.attn.q.weight",
                             "audio.blk.2.ffn.gate.weight", "llm.lm_head.weight"})
        CHECK(core_gguf::mimo_weight_usage(name, nullptr) == WeightUsage::MatmulOnly);
    for (const char* name : {"llm.embed.weight", "audio.emb.0.weight", "model.layers.0.attn.q.bias",
                             "llm.codebook_head.0.weight", "audio.hidden_proj.weight", "model.layers.x.attn.q.weight"})
        CHECK(core_gguf::mimo_weight_usage(name, nullptr) == WeightUsage::General);
    for (const char* name : {"encoder.layers.0.attn.q.weight", "decoder.layers.2.cross_attn.k.weight",
                             "decoder.layers.0.ffn.fc1.weight", "decoder.output.weight"})
        CHECK(core_gguf::moonshine_weight_usage(name, nullptr) == WeightUsage::MatmulOnly);
    for (const char* name : {"decoder.embed_tokens.weight", "encoder.conv1.weight", "decoder.output_norm.weight"})
        CHECK(core_gguf::moonshine_weight_usage(name, nullptr) == WeightUsage::General);
}

TEST_CASE("Usage is applied across GGUF loader paths without renaming or changing bytes", "[unit][weight-usage]") {
    const char* path = "weight-usage-test.gguf";
    ggml_context* ctx = ggml_init({16384, nullptr, false});
    REQUIRE(ctx);
    auto* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    auto* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    ggml_set_name(a, "arbitrary.linear");
    ggml_set_name(b, "decoder.embed_tokens.weight");
    for (int i = 0; i < 16; ++i) {
        ((float*)a->data)[i] = float(i);
        ((float*)b->data)[i] = float(100 + i);
    }
    gguf_context* g = gguf_init_empty();
    gguf_add_tensor(g, a);
    gguf_add_tensor(g, b);
    REQUIRE(gguf_write_to_file(g, path, false));
    gguf_free(g);
    ggml_backend_t backend = ggml_backend_cpu_init();
    REQUIRE(backend);
    core_gguf::WeightUsagePolicy policy{[](const char* name, void*) {
                                            return std::strcmp(name, "arbitrary.linear") == 0
                                                       ? core_gguf::WeightUsage::MatmulOnly
                                                       : core_gguf::WeightUsage::General;
                                        },
                                        nullptr};
    for (int mode = 0; mode < 5; ++mode) {
        core_gguf::WeightLoad wl;
        bool ok = false;
        if (mode == 0)
            ok = core_gguf::load_weights(path, backend, "usage-test", wl, policy);
        if (mode == 1)
            ok = core_gguf::load_weights_filtered(
                path, backend, [](const char*, void*) { return true; }, nullptr, "usage-test", wl, policy);
        if (mode == 2)
            ok = core_gguf::load_weights_split(
                path, backend, backend,
                [](const char* name, void*) { return std::strcmp(name, "arbitrary.linear") == 0; }, nullptr,
                "usage-test", wl, policy);
        if (mode == 3)
            ok = core_gguf::load_weights_fit(path, backend, backend, "usage-test", wl, nullptr, policy);
        if (mode == 4)
            ok = core_gguf::load_weights_repack(
                path, backend, [](const char* name, void*) { return std::strcmp(name, "arbitrary.linear") == 0; },
                nullptr, "usage-test", wl);
        INFO(mode);
        REQUIRE(ok);
        REQUIRE(wl.tensors.size() == 2);
        CHECK(wl.tensors.at("arbitrary.linear")->flags & GGML_TENSOR_FLAG_WEIGHT_MATMUL_ONLY);
        CHECK(wl.tensors.at("decoder.embed_tokens.weight")->flags & GGML_TENSOR_FLAG_WEIGHT_GENERAL);
        float data[16];
        ggml_backend_tensor_get(wl.tensors.at("arbitrary.linear"), data, 0, sizeof(data));
        CHECK(std::memcmp(data, a->data, sizeof(data)) == 0);
        ggml_backend_tensor_get(wl.tensors.at("decoder.embed_tokens.weight"), data, 0, sizeof(data));
        CHECK(std::memcmp(data, b->data, sizeof(data)) == 0);
        core_gguf::free_weights(wl);
    }
    ggml_backend_free(backend);
    ggml_free(ctx);
    std::remove(path);
}

TEST_CASE("CANN chunk coverage rejects holes, overlap counts and invalid ranges", "[unit][weight-usage]") {
    ggml_cann_upload_coverage coverage;
    CHECK_FALSE(coverage.add(50, 50, 100));
    CHECK_FALSE(coverage.add(50, 50, 100));
    CHECK_FALSE(coverage.add(0, 25, 100));
    CHECK_FALSE(coverage.add(20, 10, 100));
    CHECK_FALSE(coverage.add(101, 1, 100));
    CHECK_FALSE(coverage.add(99, 2, 100));
    CHECK_FALSE(coverage.add(0, 0, 100));
    CHECK(coverage.add(30, 20, 100));
    coverage.clear();
    CHECK_FALSE(coverage.add(0, 99, 100));
    CHECK(coverage.add(99, 1, 100));
}
