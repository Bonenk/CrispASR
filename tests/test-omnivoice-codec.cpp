// Hermetic tokenizer lifetime, placement-policy and allocation-failure regression.
// Include the runtime so these checks exercise its actual private load helpers.
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
int main(int argc, char** argv) {
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
