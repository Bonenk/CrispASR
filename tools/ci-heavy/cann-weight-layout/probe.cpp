// Real backend execution with an independent scalar oracle. CPU is a harness control.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

static void check(bool ok, const char* message) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}
static void save(const std::filesystem::path& path, const void* data, size_t size) {
    FILE* file = std::fopen(path.string().c_str(), "wb");
    check(file, "open payload");
    check(std::fwrite(data, 1, size, file) == size, "write payload");
    check(std::fclose(file) == 0, "close payload");
}
int main(int argc, char** argv) {
    std::string mode;
    std::filesystem::path out;
    bool require_nz = false, require_nd = false;
    int device = 0;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--backend" && i + 1 < argc)
            mode = argv[++i];
        else if (arg == "--out" && i + 1 < argc)
            out = argv[++i];
        else if (arg == "--require-nz")
            require_nz = true;
        else if (arg == "--require-nd")
            require_nd = true;
        else if (arg == "--device" && i + 1 < argc) {
            char* end = nullptr;
            const char* value = argv[++i];
            const long n = std::strtol(value, &end, 10);
            check(end != value && *end == '\0' && n >= 0 && n <= 128, "device index");
            device = int(n);
        } else
            check(false, "usage: --backend cpu|cann --device N --out FRESH_DIR --require-nz|--require-nd");
    }
    check(mode == "cpu" || mode == "cann", "explicit backend required");
    check(require_nz != require_nd, "choose exactly one expected layout");
    check(!require_nz || mode == "cann", "NZ proof requires actual CANN");
    check(!out.empty() && !std::filesystem::exists(out), "fresh output directory required");
    ggml_backend_load_all();
    auto reg = ggml_backend_reg_by_name(mode == "cann" ? "CANN" : "CPU");
    check(reg && size_t(device) < ggml_backend_reg_dev_count(reg), "requested device exists");
    auto dev = ggml_backend_reg_dev_get(reg, device);
    auto backend = ggml_backend_dev_init(dev, nullptr);
    check(backend && ggml_backend_get_device(backend) == dev, "exact requested backend initializes");
    using CacheSize = size_t (*)(const ggml_tensor*);
    auto cache_size =
        reinterpret_cast<CacheSize>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cann_nz_cache_size"));
    check(mode != "cann" || cache_size, "CANN cache inspection interface required");
    std::filesystem::create_directories(out);
    std::printf("backend=%s device=%s NZ=%s CACHE=%s\n", ggml_backend_name(backend), ggml_backend_dev_name(dev),
                std::getenv("GGML_CANN_WEIGHT_NZ") ? std::getenv("GGML_CANN_WEIGHT_NZ") : "default",
                std::getenv("GGML_CANN_WEIGHT_NZ_CACHE") ? std::getenv("GGML_CANN_WEIGHT_NZ_CACHE") : "default");
    int computes = 0;
    struct Case {
        const char* name;
        int flags;
        bool eligible;
    };
    const Case cases[] = {
        {"blk.0.attn_q.weight", 0, true},
        {"audio.blk.0.attn.q.weight", 0, true},
        {"model.layers.0.attn.qkv.weight", 0, true},
        {"llm.lm_head.weight", 0, true},
        {"arbitrary.linear", GGML_TENSOR_FLAG_WEIGHT_MATMUL_ONLY, true},
        {"decoder.output.weight", GGML_TENSOR_FLAG_WEIGHT_GENERAL, false},
        {"token_embd.weight", 0, false},
        {"notoutput.weight", 0, false},
    };
    int case_no = 0;
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16})
        for (int k : {32, 37, 32})
            for (const auto& item : cases) {
                constexpr int rows = 48;
                auto weights = ggml_init({ggml_tensor_overhead() * 4, nullptr, true});
                check(weights, "weight context");
                auto w = ggml_new_tensor_2d(weights, type, k, rows);
                auto copied = ggml_new_tensor_2d(weights, type, k, rows);
                ggml_set_name(w, item.name);
                w->flags |= item.flags;
                ggml_set_name(copied, "plain.copy");
                copied->flags |= GGML_TENSOR_FLAG_WEIGHT_GENERAL;
                auto wb = ggml_backend_alloc_ctx_tensors(weights, backend);
                check(wb, "weight allocation");
                std::vector<float> wf(k * rows);
                for (size_t i = 0; i < wf.size(); ++i)
                    wf[i] = float(int(i % 5) - 2);
                std::vector<ggml_fp16_t> wh(wf.size());
                ggml_fp32_to_fp16_row(wf.data(), wh.data(), wh.size());
                const void* raw = type == GGML_TYPE_F32 ? (void*)wf.data() : (void*)wh.data();
                const size_t bytes = ggml_nbytes(w), half = bytes / 2;
                // Out-of-order and overlapping chunks. A hole must prevent readiness.
                ggml_backend_tensor_set(w, (const char*)raw + half, half, bytes - half);
                ggml_backend_tensor_set(w, (const char*)raw + half, half, bytes - half);
                if (cache_size)
                    check(cache_size(w) == 0, "overlap cannot complete upload with a hole");
                ggml_backend_tensor_set(w, raw, 0, half);
                const bool expect_cache = require_nz && item.eligible;
                if (cache_size)
                    check((cache_size(w) > 0) == expect_cache, "actual uploaded cache layout");
                std::vector<char> back(bytes);
                ggml_backend_tensor_get(w, back.data(), 0, bytes);
                check(std::memcmp(back.data(), raw, bytes) == 0, "ordinary readback preserves exact bytes");
                ggml_backend_tensor_copy(w, copied);
                ggml_backend_tensor_get(copied, back.data(), 0, bytes);
                check(std::memcmp(back.data(), raw, bytes) == 0, "ordinary copy preserves exact bytes");
                // Actual generic transfer into a CPU buffer, including when the
                // source has a ready CANN NZ cache. No packed bytes may escape.
                auto cpu = ggml_backend_cpu_init();
                check(cpu, "CPU copy destination backend");
                auto host_ctx = ggml_init({ggml_tensor_overhead() * 2, nullptr, true});
                check(host_ctx, "CPU copy context");
                auto host_weight = ggml_new_tensor_2d(host_ctx, type, k, rows);
                auto host_buffer = ggml_backend_alloc_ctx_tensors(host_ctx, cpu);
                check(host_buffer, "CPU copy buffer");
                ggml_backend_tensor_copy(w, host_weight);
                ggml_backend_tensor_get(host_weight, back.data(), 0, bytes);
                check(std::memcmp(back.data(), raw, bytes) == 0, "CPU transfer preserves exact ordinary bytes");
                ggml_backend_buffer_free(host_buffer);
                ggml_free(host_ctx);
                ggml_backend_free(cpu);
                save(out / (std::to_string(case_no) + "-weights.bin"), raw, bytes);
                for (int columns : {1, 5, 1}) {
                    const size_t mem = ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(32, false);
                    auto ctx = ggml_init({mem, nullptr, true});
                    check(ctx, "graph context");
                    auto x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, columns);
                    ggml_set_input(x);
                    auto y = ggml_mul_mat(ctx, w, x);
                    ggml_set_output(y);
                    auto graph = ggml_new_graph_custom(ctx, 32, false);
                    ggml_build_forward_expand(graph, y);
                    check(ggml_backend_dev_supports_op(dev, y), "requested backend supports matrix product");
                    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
                    check(buffer, "graph buffer");
                    for (int repeat = 0; repeat < 3; ++repeat) {
                        if (repeat > 0) {
                            wf[0] += 1.0f;
                            wh[0] = ggml_fp32_to_fp16(wf[0]);
                            // Same graph: invalidate NZ with an asynchronous write, then
                            // reactivate it with a synchronous partial write. Captured
                            // graph replay must follow the changed ND/NZ pointer choice.
                            if (repeat == 1) {
                                ggml_backend_tensor_set_async(backend, w, raw, 0, ggml_type_size(type));
                                ggml_backend_synchronize(backend);
                                if (cache_size)
                                    check(cache_size(w) == 0, "async write invalidates cache");
                            } else {
                                ggml_backend_tensor_set(w, raw, 0, ggml_type_size(type));
                                if (cache_size)
                                    check((cache_size(w) > 0) == expect_cache, "partial write rebuilds cache");
                            }
                        }
                        std::vector<float> input(k * columns), expected(rows * columns), actual(rows * columns);
                        for (size_t i = 0; i < input.size(); ++i)
                            input[i] = float((int(i) + repeat) % 7 - 3);
                        for (int n = 0; n < columns; ++n)
                            for (int r = 0; r < rows; ++r)
                                for (int j = 0; j < k; ++j)
                                    expected[r + rows * n] += wf[j + k * r] * input[j + k * n];
                        ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
                        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
                              "actual backend compute");
                        ggml_backend_synchronize(backend);
                        ggml_backend_tensor_get(y, actual.data(), 0, actual.size() * sizeof(float));
                        check(actual == expected, "exact independent scalar matrix oracle");
                        auto stem = out / (std::to_string(case_no) + "-" + std::to_string(computes++));
                        save(stem.string() + "-weights.bin", raw, bytes);
                        save(stem.string() + "-input.f32", input.data(), input.size() * sizeof(float));
                        save(stem.string() + "-actual.f32", actual.data(), actual.size() * sizeof(float));
                        save(stem.string() + "-expected.f32", expected.data(), expected.size() * sizeof(float));
                    }
                    ggml_backend_buffer_free(buffer);
                    ggml_free(ctx);
                }
                // Mutation of the ordinary representation invalidates/rebuilds cache.
                ggml_backend_tensor_set(w, raw, 0, bytes);
                if (cache_size)
                    check((cache_size(w) > 0) == expect_cache, "full rewrite cache readiness");
                ggml_backend_buffer_clear(wb, 0);
                if (cache_size)
                    check(cache_size(w) == 0, "clear invalidates NZ cache");
                std::printf("case=%d type=%s k=%d name=%s eligible=%d completed\n", case_no++, ggml_type_name(type), k,
                            item.name, item.eligible);
                ggml_backend_buffer_free(wb);
                ggml_free(weights);
            }
    ggml_backend_free(backend);
    std::printf("PASS computes=%d cases=%d\n", computes, case_no);
    return 0;
}
