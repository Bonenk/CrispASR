// Actual composed scheduler graph. CPU controls do not simulate CANN execution.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

static void require(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

static void save(const std::filesystem::path& path, const std::vector<float>& values) {
    FILE* file = std::fopen(path.string().c_str(), "wb");
    require(file != nullptr, "open output payload");
    require(std::fwrite(values.data(), sizeof(float), values.size(), file) == values.size(), "write output payload");
    require(std::fclose(file) == 0, "close output payload");
}

int main(int argc, char** argv) {
    std::string mode;
    std::filesystem::path output;
    long device_index = 0;
    bool require_fallback = false;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--require-cpu-fallback") {
            require_fallback = true;
        } else if (key == "--backend" && i + 1 < argc) {
            mode = argv[++i];
        } else if (key == "--out" && i + 1 < argc) {
            output = argv[++i];
        } else if (key == "--device" && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            const char* value = argv[++i];
            device_index = std::strtol(value, &end, 10);
            require(errno == 0 && end != value && *end == '\0' && device_index >= 0, "invalid device index");
        } else {
            require(false, "usage: --backend cpu|cann --device N --out FRESH_PATH [--require-cpu-fallback]");
        }
    }
    require(mode == "cpu" || mode == "cann", "backend must be cpu or cann");
    require(!output.empty(), "fresh output directory required");
    require(!require_fallback || mode == "cann", "CANN fallback proof requires the CANN backend");
    require(mode != "cpu" || device_index == 0, "CPU control accepts only device zero");
    require(!std::filesystem::exists(output), "output directory already exists; preserve previous evidence");

    ggml_backend_load_all();
    auto cpu = ggml_backend_cpu_init();
    require(cpu != nullptr, "CPU backend initializes");
    ggml_backend_cpu_set_n_threads(cpu, 2);
    ggml_backend_t selected = cpu;
    if (mode == "cann") {
        auto reg = ggml_backend_reg_by_name("CANN");
        require(reg != nullptr, "CANN registry unavailable; CPU cannot satisfy this request");
        require(static_cast<size_t>(device_index) < ggml_backend_reg_dev_count(reg),
                "requested CANN device unavailable");
        auto dev = ggml_backend_reg_dev_get(reg, static_cast<size_t>(device_index));
        require(dev && ggml_backend_dev_backend_reg(dev) == reg, "exact CANN device identity");
        selected = ggml_backend_dev_init(dev, nullptr);
        require(selected != nullptr && ggml_backend_get_device(selected) == dev, "requested CANN device initializes");
        std::printf("requested_registry=CANN device=%ld device_name=%s backend=%s\n", device_index,
                    ggml_backend_dev_name(dev), ggml_backend_name(selected));
    } else {
        std::printf("requested_registry=CPU device=0 backend=%s\n", ggml_backend_name(cpu));
    }
    std::filesystem::create_directories(output);
    ggml_backend_t backends[] = {selected, cpu};
    auto sched = ggml_backend_sched_new(backends, nullptr, mode == "cann" ? 2 : 1, 64, false, true);
    require(sched != nullptr, "scheduler initializes");
    const std::array<int, 3> shapes = {16, 32, 16};
    const std::array<int, 6> stimuli = {0, 0, 1, 2, 2, 3};
    int calls = 0;
    for (size_t shape_case = 0; shape_case < shapes.size(); ++shape_case) {
        const int n = shapes[shape_case];
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 24 + ggml_graph_overhead_custom(64, false);
        ip.no_alloc = true;
        auto inputs = ggml_init(ip);
        auto arena = ggml_init(ip);
        require(inputs && arena, "tensor contexts initialize");
        auto a = ggml_new_tensor_2d(inputs, GGML_TYPE_F32, 32, 32);
        auto b = ggml_new_tensor_2d(inputs, GGML_TYPE_F32, 32, n);
        ggml_set_input(a);
        ggml_set_input(b);
        auto buffer = ggml_backend_alloc_ctx_tensors(inputs, cpu);
        require(buffer != nullptr, "CPU input buffer allocates");
        auto mm = ggml_mul_mat(arena, a, b);
        auto rounded = ggml_cast(arena, mm, GGML_TYPE_BF16);
        auto sum = ggml_add(arena, rounded, rounded);
        auto as_float = ggml_cast(arena, sum, GGML_TYPE_F32);
        auto final = ggml_scale(arena, as_float, 0.25f);
        ggml_set_name(mm, "supported_f32_matmul");
        ggml_set_name(sum, "bf16_add_boundary");
        ggml_set_name(final, "supported_f32_after_boundary");
        ggml_set_output(mm);
        ggml_set_output(sum);
        ggml_set_output(final);
        auto graph = ggml_new_graph_custom(arena, 64, false);
        ggml_build_forward_expand(graph, final);
        require(ggml_backend_dev_supports_op(ggml_backend_get_device(selected), mm),
                "selected backend supports F32 matmul");
        require(ggml_backend_dev_supports_op(ggml_backend_get_device(selected), final),
                "selected backend supports F32 tail");
        // Force supported work onto the requested device; let the scheduler
        // choose the BF16 boundary from actual device capability declarations.
        ggml_backend_sched_set_tensor_backend(sched, mm, selected);
        ggml_backend_sched_set_tensor_backend(sched, final, selected);
        require(ggml_backend_sched_alloc_graph(sched, graph), "composed graph allocates");
        require(ggml_backend_sched_get_tensor_backend(sched, mm) == selected, "matrix placed on requested device");
        require(ggml_backend_sched_get_tensor_backend(sched, final) == selected, "tail placed on requested device");
        auto boundary_backend = ggml_backend_sched_get_tensor_backend(sched, sum);
        require(boundary_backend != nullptr, "BF16 boundary placement available");
        if (require_fallback) {
            require(!ggml_backend_dev_supports_op(ggml_backend_get_device(selected), sum),
                    "CANN really rejects BF16 ADD");
            require(boundary_backend == cpu, "rejected BF16 ADD placed on CPU");
        }
        std::printf("shape_case=%zu n=%d matmul=%s bf16_add=%s tail=%s required_cpu_fallback=%d\n", shape_case, n,
                    ggml_backend_name(selected), ggml_backend_name(boundary_backend), ggml_backend_name(selected),
                    require_fallback);
        std::vector<float> av(32 * 32), bv(32 * n), expected(32 * n), actual(32 * n), matrix(32 * n);
        for (int m = 0; m < 32; ++m)
            for (int k = 0; k < 32; ++k)
                av[k + 32 * m] = (k == m ? 2.0f : 0.0f) + (k == (m + 3) % 32 ? 1.0f : 0.0f);
        for (size_t iteration = 0; iteration < stimuli.size(); ++iteration) {
            for (int column = 0; column < n; ++column)
                for (int k = 0; k < 32; ++k)
                    bv[k + 32 * column] = float((k * 3 + column * 5 + stimuli[iteration]) % 7 - 3);
            // Independent scalar matrix product. Every result is an integer
            // in [-9,9], so BF16 conversion/addition and the 1/4 scale are exact.
            for (int column = 0; column < n; ++column)
                for (int m = 0; m < 32; ++m) {
                    float value = 0;
                    for (int k = 0; k < 32; ++k)
                        value += av[k + 32 * m] * bv[k + 32 * column];
                    matrix[m + 32 * column] = value;
                    expected[m + 32 * column] = value * 0.5f;
                }
            auto stem = output / ("shape-" + std::to_string(shape_case) + "-iter-" + std::to_string(iteration));
            save(stem.string() + "-a.f32", av);
            save(stem.string() + "-b.f32", bv);
            save(stem.string() + "-expected.f32", expected);
            ggml_backend_tensor_set(a, av.data(), 0, av.size() * sizeof(float));
            ggml_backend_tensor_set(b, bv.data(), 0, bv.size() * sizeof(float));
            require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "composed compute succeeds");
            ggml_backend_sched_synchronize(sched);
            ggml_backend_tensor_get(final, actual.data(), 0, actual.size() * sizeof(float));
            save(stem.string() + "-actual.f32", actual);
            require(actual == expected, "actual output equals exact independent matrix oracle");
            ggml_backend_tensor_get(mm, actual.data(), 0, actual.size() * sizeof(float));
            save(stem.string() + "-matmul.f32", actual);
            require(actual == matrix, "actual intermediate matrix equals independent oracle");
            std::printf("COMPUTE_PASS shape_case=%zu iteration=%zu stimulus=%d values=%zu\n", shape_case, iteration,
                        stimuli[iteration], expected.size());
            ++calls;
        }
        ggml_backend_sched_reset(sched);
        ggml_backend_buffer_free(buffer);
        ggml_free(arena);
        ggml_free(inputs);
    }
    ggml_backend_sched_free(sched);
    if (selected != cpu)
        ggml_backend_free(selected);
    ggml_backend_free(cpu);
    std::printf("COMPOSED_PROBE_PASS backend=%s calls=%d shapes=3 required_cpu_fallback=%d\n", mode.c_str(), calls,
                require_fallback);
}
