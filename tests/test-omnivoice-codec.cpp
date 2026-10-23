// Hermetic tokenizer lifetime, placement-policy and allocation-failure regression.
// Include the runtime so these checks exercise its actual private load helpers.
#include "core/ggml_cpu_backend.h"
#include "../src/omnivoice.cpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "ggml-backend-impl.h"
#include <cassert>
#include <fstream>
#include <iostream>

static void test_env(const char* key, const char* value) {
#ifdef _WIN32
    _putenv_s(key, value ? value : "");
#else
    if (value)
        setenv(key, value, 1);
    else
        unsetenv(key);
#endif
}

static void fixture(const char* path) {
    ggml_init_params ip = {1024 * 1024, nullptr, false};
    auto* weights = ggml_init(ip);
    auto* file = gguf_init_empty();
    auto add = [&](const std::string& name) {
        auto* t = ggml_new_tensor_3d(
            weights, name.find("conv_t1.weight") != std::string::npos ? GGML_TYPE_F16 : GGML_TYPE_F32, 2, 2, 2);
        ggml_set_name(t, name.c_str());
        ggml_set_f32(t, 0.25f);
        gguf_add_tensor(file, t);
    };
    add("fc2.weight");
    add("acoustic_encoder.conv1.weight");
    add("quantizer.quantizers.0.project_in.weight");
    add("quantizer.quantizers.0.codebook.embed_avg");
    add("acoustic_decoder.conv1.weight");
    add("acoustic_decoder.block.0.conv_t1.weight");
    for (int i = 0; i < 8; i++)
        add("quantizer.quantizers." + std::to_string(i) + ".codebook.embed");
    assert(gguf_write_to_file(file, path, false));
    gguf_free(file);
    ggml_free(weights);
}
// Exercise the real transformer builder, including asymmetric Q/KV head counts,
// strided projection slices and independent CFG attention blocks. No model files.
static void test_projection_graphs() {
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K}) {
        auto* ctx = new omnivoice_context;
        ctx->backend = core_cpu_backend::init();
        core_cpu_backend::set_n_threads(ctx->backend, 1);
        auto& hp = ctx->hp;
        hp.n_layers = 1;
        hp.d_model = 256;
        hp.head_dim = 128;
        hp.n_heads = 2;
        hp.n_kv_heads = 1;
        hp.ff_dim = 256;
        ggml_init_params ip = {32 * ggml_tensor_overhead(), nullptr, true};
        ctx->ctx_w = ggml_init(ip);
        ctx->model.blocks.resize(1);
        auto& b = ctx->model.blocks[0];
        auto matrix = [&](int input, int output) { return ggml_new_tensor_2d(ctx->ctx_w, type, input, output); };
        auto norm = [&](int n) { return ggml_new_tensor_1d(ctx->ctx_w, GGML_TYPE_F32, n); };
        b.attn_norm_w = norm(256);
        b.ffn_norm_w = norm(256);
        b.attn_q_norm_w = norm(128);
        b.attn_k_norm_w = norm(128);
        b.attn_q_w = matrix(256, 256);
        b.attn_k_w = matrix(256, 128);
        b.attn_v_w = matrix(256, 128);
        b.attn_output_w = matrix(256, 256);
        b.ffn_gate_w = matrix(256, 256);
        b.ffn_up_w = matrix(256, 256);
        b.ffn_down_w = matrix(256, 256);
        ctx->model.output_norm_w = norm(256);
        ctx->model.audio_output_w = matrix(256, 16);
        ctx->buf_w = ggml_backend_alloc_ctx_tensors(ctx->ctx_w, ctx->backend);
        int index = 0;
        for (auto* t = ggml_get_first_tensor(ctx->ctx_w); t; t = ggml_get_next_tensor(ctx->ctx_w, t)) {
            std::vector<float> values(ggml_nelements(t));
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = ggml_is_vector(t) ? 1.f : 0.03f * std::sin((float)(i + ++index) * 0.17f);
            std::vector<uint8_t> bytes(ggml_nbytes(t));
            if (t->type == GGML_TYPE_F32)
                std::memcpy(bytes.data(), values.data(), bytes.size());
            else
                assert(ggml_quantize_chunk(t->type, values.data(), bytes.data(), 0, t->ne[1], t->ne[0], nullptr) ==
                       bytes.size());
            ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
        }
        assert(prepare_packed_projections(ctx));
        for (int split : {0, 3}) {
            std::vector<float> reference;
            for (int mode = 0; mode < 4; ++mode) {
                test_env("CRISPASR_OMNIVOICE_PACKED_PROJ", mode & 1 ? "1" : "0");
                test_env("CRISPASR_OMNIVOICE_ATTN_VIEWS", mode & 2 ? "1" : "0");
                ggml_init_params gp = {4 * 1024 * 1024, nullptr, true};
                auto* graph_ctx = ggml_init(gp);
                auto* input = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_F32, 256, 5);
                ggml_set_input(input);
                auto* graph = build_llm_graph(ctx, graph_ctx, input, 5, split);
                auto* allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
                assert(ggml_gallocr_alloc_graph(allocator, graph));
                std::vector<float> x(256 * 5);
                for (size_t i = 0; i < x.size(); ++i)
                    x[i] = std::cos((float)i * 0.13f);
                ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
                int32_t positions[] = {0, 1, 2, 3, 4};
                if (split) {
                    positions[3] = 0;
                    positions[4] = 1;
                }
                ggml_backend_tensor_set(ggml_get_tensor(graph_ctx, "pos_ids"), positions, 0, sizeof(positions));
                assert(ggml_backend_graph_compute(ctx->backend, graph) == GGML_STATUS_SUCCESS);
                std::vector<float> out(16 * 5);
                ggml_backend_tensor_get(ggml_get_tensor(graph_ctx, "audio_logits"), out.data(), 0,
                                        out.size() * sizeof(float));
                if (!mode) {
                    reference = out;
                    double norm_squared = 0;
                    for (float value : out)
                        norm_squared += (double)value * value;
                    assert(norm_squared > 1e-12);
                }
                for (size_t i = 0; i < out.size(); ++i) {
                    assert(std::isfinite(out[i]));
                    assert(std::abs(out[i] - reference[i]) < 1e-5f);
                }
                ggml_gallocr_free(allocator);
                ggml_free(graph_ctx);
            }
        }
        omnivoice_free(ctx);
    }
    test_env("CRISPASR_OMNIVOICE_PACKED_PROJ", nullptr);
    test_env("CRISPASR_OMNIVOICE_ATTN_VIEWS", nullptr);
    std::cout << "PASS actual F32/F16/Q8_0/Q4_K transformer logits, packed/views, separate/unified CFG\n";
}

static void test_packed_projections() {
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K}) {
        auto* ctx = new omnivoice_context;
        ctx->backend = core_cpu_backend::init();
        ggml_init_params ip = {16 * ggml_tensor_overhead(), nullptr, true};
        ctx->ctx_w = ggml_init(ip);
        ctx->model.blocks.resize(1);
        auto& b = ctx->model.blocks[0];
        auto matrix = [&](int rows) { return ggml_new_tensor_2d(ctx->ctx_w, type, 256, rows); };
        b.attn_q_w = matrix(3);
        b.attn_k_w = matrix(2);
        b.attn_v_w = matrix(2);
        b.ffn_gate_w = matrix(5);
        b.ffn_up_w = matrix(5);
        ctx->buf_w = ggml_backend_alloc_ctx_tensors(ctx->ctx_w, ctx->backend);
        int tag = 0;
        for (auto* t : {b.attn_q_w, b.attn_k_w, b.attn_v_w, b.ffn_gate_w, b.ffn_up_w}) {
            std::vector<uint8_t> bytes(ggml_nbytes(t));
            for (size_t i = 0; i < bytes.size(); ++i)
                bytes[i] = (uint8_t)(i * 13 + ++tag);
            ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
        }
        // The actual backend allocator fails: no slots may be published.
        auto* cpu = ctx->backend;
        ggml_backend_buffer_type failing = *core_cpu_backend::buffer_type();
        failing.iface.alloc_buffer = [](ggml_backend_buffer_type_t, size_t) -> ggml_backend_buffer_t {
            return nullptr;
        };
        ggml_backend_device device = *ggml_backend_get_device(cpu);
        device.context = &failing;
        device.iface.get_buffer_type = [](ggml_backend_dev_t d) {
            return static_cast<ggml_backend_buffer_type_t>(d->context);
        };
        ggml_backend backend = *cpu;
        backend.device = &device;
        ctx->backend = &backend;
        assert(!prepare_packed_projections(ctx));
        assert(!ctx->ctx_packed && !ctx->buf_packed && !b.attn_qkv_w && !b.ffn_gate_up_w);
        ctx->backend = cpu;
        assert(prepare_packed_projections(ctx));
        auto check = [](ggml_tensor* packed, std::vector<ggml_tensor*> parts) {
            assert(packed);
            size_t offset = 0;
            for (auto* t : parts) {
                std::vector<uint8_t> source(ggml_nbytes(t)), actual(source.size());
                ggml_backend_tensor_get(t, source.data(), 0, source.size());
                ggml_backend_tensor_get(packed, actual.data(), offset, actual.size());
                assert(source == actual);
                offset += source.size();
            }
            assert(offset == ggml_nbytes(packed));
        };
        check(b.attn_qkv_w, {b.attn_q_w, b.attn_k_w, b.attn_v_w});
        check(b.ffn_gate_up_w, {b.ffn_gate_w, b.ffn_up_w});
        auto* retained = ctx->buf_packed;
        assert(prepare_packed_projections(ctx) && ctx->buf_packed == retained);
        omnivoice_free(ctx);
    }
    // Mixed precisions and incompatible widths must use the original projections.
    omnivoice_context ctx;
    ctx.backend = core_cpu_backend::init();
    ggml_init_params ip = {8 * ggml_tensor_overhead(), nullptr, true};
    ctx.ctx_w = ggml_init(ip);
    ctx.model.blocks.resize(1);
    auto& b = ctx.model.blocks[0];
    b.attn_q_w = ggml_new_tensor_2d(ctx.ctx_w, GGML_TYPE_F32, 256, 3);
    b.attn_k_w = b.attn_q_w;
    b.attn_v_w = ggml_new_tensor_2d(ctx.ctx_w, GGML_TYPE_F16, 256, 2);
    b.ffn_gate_w = b.attn_q_w;
    b.ffn_up_w = ggml_new_tensor_2d(ctx.ctx_w, GGML_TYPE_F32, 128, 3);
    assert(!prepare_packed_projections(&ctx));
    assert(!ctx.ctx_packed && !ctx.buf_packed && !b.attn_qkv_w && !b.ffn_gate_up_w);
    ggml_free(ctx.ctx_w);
    ggml_backend_free(ctx.backend);
    std::cout << "PASS packed F32/F16/Q8_0/Q4_K bytes, allocation failure, reuse and type/shape fallback\n";
}

int main(int argc, char** argv) {
    test_packed_projections();
    test_projection_graphs();
    (void)argc;
    (void)argv;
    const auto scratch =
        std::filesystem::temp_directory_path() /
        ("crispasr-codec-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(scratch);
    const std::string path = (scratch / "tiny.gguf").string();
    fixture(path.c_str());
    test_env("CRISPASR_OMNIVOICE_CODEC_GPU", "0");
    test_env("CRISPASR_OMNIVOICE_CODEC_DECOMP", "1");
    omnivoice_context ctx;
    ctx.use_gpu = false;
    ctx.verbosity = 0;
    ctx.n_threads = 1;
    assert(load_tokenizer(&ctx, path.c_str()));
    assert(ctx.tokenizer.loaded && ctx.tokenizer.ctx_perm);
    auto* upsampler = ctx.tokenizer.dec_blocks[0].conv_t1_w;
    assert(ctx.tokenizer.fc.get(upsampler) == upsampler);
    test_env("CRISPASR_OMNIVOICE_CODEC_DECOMP", "0");
    assert(load_tokenizer(&ctx, path.c_str()));
    upsampler = ctx.tokenizer.dec_blocks[0].conv_t1_w;
    assert(!ctx.tokenizer.dec_blocks[0].conv_t1_w_perm);
    assert(ctx.tokenizer.fc.get(upsampler) != upsampler);
    test_env("CRISPASR_OMNIVOICE_CODEC_DECOMP", "1");
    assert(load_tokenizer(&ctx, path.c_str()));
    assert(ctx.tokenizer.loaded && ctx.tokenizer.dec_blocks[0].conv_t1_w_perm);
    tok_release_backend(ctx.tokenizer);
    assert(!ctx.tokenizer.loaded && !ctx.tokenizer.ctx_w && ctx.tokenizer.enc_blocks.empty());
    assert(!load_tokenizer(&ctx, (scratch / "missing.gguf").string().c_str()));
    assert(!ctx.tokenizer.loaded && !ctx.tokenizer.backend && !ctx.tokenizer.ctx_w);
    assert(load_tokenizer(&ctx, path.c_str()));
    // Force a read failure after the metadata was parsed and buffers allocated.
    const std::string truncated = (scratch / "truncated.gguf").string();
    {
        std::ifstream in(path, std::ios::binary);
        std::ofstream out(truncated, std::ios::binary);
        std::string data((std::istreambuf_iterator<char>(in)), {});
        data.resize(data.size() - 16);
        out.write(data.data(), data.size());
    }
    assert(!load_tokenizer(&ctx, truncated.c_str()));
    assert(!ctx.tokenizer.loaded && !ctx.tokenizer.backend && !ctx.tokenizer.ctx_w && ctx.tokenizer.dec_blocks.empty());
    assert(load_tokenizer(&ctx, path.c_str()));
    // Decoder-only loading must exclude encoder weights while preserving all
    // selected serialized bytes. Reuse succeeds even without the source file;
    // replacing the tokenizer invalidates the resident copy.
    test_env("CRISPASR_OMNIVOICE_CODEC_RESIDENT", nullptr);
    assert(!codec_resident_enabled());
    test_env("CRISPASR_OMNIVOICE_CODEC_RESIDENT", "1");
    assert(codec_resident_enabled());
    assert(tok_prepare_resident_decoder(&ctx));
    auto& decoder = ctx.resident_decoder;
    assert(decoder.loaded && !decoder.enc_conv1_w && !decoder.quantizers[0].proj_in_w);
    assert(!ggml_get_tensor(decoder.ctx_w, "quantizer.quantizers.0.codebook.embed_avg"));
    assert(ctx.tokenizer.enc_conv1_w && ctx.tokenizer.quantizers[0].proj_in_w);
    for (auto* t = ggml_get_first_tensor(decoder.ctx_w); t; t = ggml_get_next_tensor(decoder.ctx_w, t)) {
        assert(tok_is_decode_weight(t->name));
        auto* original = ggml_get_tensor(ctx.tokenizer.ctx_w, t->name);
        assert(original && ggml_nbytes(original) == ggml_nbytes(t));
        std::vector<uint8_t> a(ggml_nbytes(t)), b(a.size());
        ggml_backend_tensor_get(original, a.data(), 0, a.size());
        ggml_backend_tensor_get(t, b.data(), 0, b.size());
        assert(a == b);
    }
    auto* retained = decoder.buf_w;
    std::filesystem::rename(path, path + ".moved");
    assert(tok_prepare_resident_decoder(&ctx) && decoder.buf_w == retained);
    std::filesystem::rename(path + ".moved", path);
    assert(load_tokenizer(&ctx, path.c_str()));
    assert(!decoder.loaded && !decoder.ctx_w && !decoder.backend);
    ctx.tokenizer.path = truncated;
    assert(!tok_prepare_resident_decoder(&ctx));
    assert(!decoder.loaded && !decoder.ctx_w && !decoder.backend);
    assert(load_tokenizer(&ctx, path.c_str()));
    test_env("CRISPASR_OMNIVOICE_CODEC_RESIDENT", "0");
    assert(!codec_resident_enabled());
    std::cout << "PASS filtered decoder bytes, file-independent reuse, invalidation and failed-load cleanup\n";
    // A cache miss with SPLIT disabled must not force a different placement.
    ctx.use_gpu = true;
    crispasr_set_gpu_backend_pref("cpu");
    assert(tok_rebind(&ctx, true));
    assert(ctx.tokenizer.gpu_requested && core_cpu_backend::is_cpu(ctx.tokenizer.backend));
    auto* fallback_backend = ctx.tokenizer.backend;
    assert(tok_rebind(&ctx, true));
    assert(ctx.tokenizer.backend == fallback_backend);
    ctx.preprocess_prompt = false;
    test_env("CRISPASR_OMNIVOICE_CODEC_SPLIT", nullptr);
    assert(omnivoice_set_voice_prompt(&ctx, TEST_SOURCE_DIR "/samples/jfk.wav", "Reference.") == -1);
    assert(ctx.tokenizer.gpu_requested);
    tok_release_backend(ctx.tokenizer);
    std::cout << "PASS repeated, missing, truncated and recovered tokenizer loads\n";
    auto* cpu = core_cpu_backend::init();
    test_env("CRISPASR_OMNIVOICE_CODEC_DECOMP", nullptr);
    assert(!codec_decomp_enabled());
    test_env("CRISPASR_OMNIVOICE_CODEC_DECOMP", "0");
    assert(!codec_decomp_enabled());
    test_env("CRISPASR_OMNIVOICE_CODEC_DECOMP", "1");
    assert(codec_decomp_enabled());
    test_env("CRISPASR_OMNIVOICE_CODEC_DECOMP", nullptr);
    // A CPU device with a failing allocator exercises actual ggml allocation.
    ggml_backend_buffer_type failing = *core_cpu_backend::buffer_type();
    failing.iface.alloc_buffer = [](ggml_backend_buffer_type_t, size_t) -> ggml_backend_buffer_t { return nullptr; };
    ggml_backend_device device = *ggml_backend_get_device(cpu);
    device.context = &failing;
    device.iface.get_buffer_type = [](ggml_backend_dev_t dev) {
        return static_cast<ggml_backend_buffer_type_t>(dev->context);
    };
    ggml_backend backend = *cpu;
    backend.device = &device;
    ggml_init_params ip = {1024 * 1024, nullptr, true};
    auto* wc = ggml_init(ip);
    auto* w = ggml_new_tensor_3d(wc, GGML_TYPE_F32, 2, 2, 2);
    auto* wb = ggml_backend_alloc_ctx_tensors(wc, cpu);
    ggml_set_f32(w, 0.25f);
    ggml_tensor* dst = nullptr;
    ggml_tensor** ds[] = {&dst};
    ggml_tensor* ss[] = {w};
    ggml_context* pc = nullptr;
    ggml_backend_buffer_t pb = nullptr;
    assert(!core_convt::permute_convt1d_weights_batch(ss, ds, 1, &backend, &pc, &pb));
    assert(!dst && !pc && !pb);
    assert(core_convt::permute_convt1d_weights_batch(ss, ds, 1, cpu, &pc, &pb));
    assert(dst && pc && pb);
    ggml_backend_buffer_free(pb);
    ggml_free(pc);
    ggml_backend_buffer_free(wb);
    ggml_free(wc);
    ggml_backend_free(cpu);
    std::cout << "PASS CPU decomposition gates and failed/successful permutation allocation\n";
    std::filesystem::remove_all(scratch);
}
