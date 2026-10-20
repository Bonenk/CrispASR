#pragma once

#include "ggml-backend.h"
#include "core/ggml_cpu_backend.h"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>
#include <set>
#include <cstdint>

// Opt-in validation only. Select an exact registry and device ordinal without
// the normal preference helper's auto fallback. No graph/eval callback changes.
namespace core_backend_validation {

inline std::string lower(const char* value) {
    std::string result = value ? value : "";
    for (char& c : result)
        if (c >= 'A' && c <= 'Z')
            c += 'a' - 'A';
    return result == "metal" ? "mtl" : result;
}

struct Request {
    std::string name;
    int device = 0;
    bool valid = true;

    Request(const char* backend, const char* ordinal) : name(lower(backend)) {
        if (name.empty() || !ordinal)
            return;
        if (!*ordinal) {
            valid = false;
            return;
        }
        for (const char* p = ordinal; *p; ++p) {
            if (*p < '0' || *p > '9' || device > (INT_MAX - (*p - '0')) / 10) {
                valid = false;
                return;
            }
            device = device * 10 + (*p - '0');
        }
    }

    bool active() const { return !name.empty(); }
    bool cpu() const { return name == "cpu"; }
};

inline ggml_backend_t init(const Request& request, bool use_gpu) {
    if (!request.active() || !request.valid || (!request.cpu() && !use_gpu)) {
        std::fprintf(stderr, "backend_validation: invalid request or GPU disabled; refusing fallback\n");
        return nullptr;
    }
    if (request.cpu()) {
        if (request.device == 0)
            return core_cpu_backend::init();
    } else {
        int ordinal = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t device = ggml_backend_dev_get(i);
            auto type = ggml_backend_dev_type(device);
            if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU)
                continue;
            auto registry = ggml_backend_dev_backend_reg(device);
            if (!registry || lower(ggml_backend_reg_name(registry)) != request.name)
                continue;
            if (ordinal++ == request.device) {
                auto backend = ggml_backend_dev_init(device, nullptr);
                std::fprintf(stderr, "backend_validation: registry=%s device=%s ordinal=%d initialized=%d\n",
                             ggml_backend_reg_name(registry), ggml_backend_dev_name(device), request.device,
                             backend != nullptr);
                return backend;
            }
        }
    }
    std::fprintf(stderr, "backend_validation: required registry=%s ordinal=%d unavailable; refusing fallback\n",
                 request.name.c_str(), request.device);
    return nullptr;
}

struct Placement {
    std::map<std::string, int> operations;
    int required_matmuls = 0;
    int unassigned = 0;
    using CacheSize = size_t (*)(const ggml_tensor*);
    using CacheSelections = uint64_t (*)(const ggml_tensor*);
    CacheSize cache_size = nullptr;
    CacheSelections cache_selections = nullptr;
    std::vector<ggml_tensor*> matmul_weights;

    struct NzUsage {
        int matmuls = 0, weights = 0, selected_weights = 0;
        size_t bytes = 0;
        uint64_t selections = 0;
    };
    NzUsage nz_usage() const {
        NzUsage usage;
        std::set<const void*> seen;
        if (!cache_size || !cache_selections)
            return usage;
        for (auto* weight : matmul_weights) {
            const size_t bytes = cache_size(weight);
            if (!bytes)
                continue;
            ++usage.matmuls;
            if (!seen.insert(weight->data).second)
                continue;
            ++usage.weights;
            usage.bytes += bytes;
            const uint64_t selections = cache_selections(weight);
            usage.selections += selections;
            if (selections)
                ++usage.selected_weights;
        }
        return usage;
    }

    Placement() = default;

    Placement(ggml_backend_sched_t sched, ggml_cgraph* graph, ggml_backend_t required) {
        const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(required));
        if (reg) {
            cache_size =
                reinterpret_cast<CacheSize>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cann_nz_cache_size"));
            cache_selections = reinterpret_cast<CacheSelections>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_cann_nz_cache_selections"));
        }
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto* node = ggml_graph_node(graph, i);
            auto backend = ggml_backend_sched_get_tensor_backend(sched, node);
            if (!backend) {
                unassigned++;
                continue;
            }
            operations[std::string(ggml_backend_name(backend)) + "/" + ggml_op_name(node->op)]++;
            if (backend == required && node->op == GGML_OP_MUL_MAT) {
                required_matmuls++;
                matmul_weights.push_back(node->src[0]);
            }
        }
    }

    bool valid() const { return required_matmuls > 0 && unassigned == 0; }

    void dump(const char* label, bool computed) const {
        for (const auto& row : operations)
            std::fprintf(stderr, "backend_validation[%s]: assigned=%s nodes=%d\n", label, row.first.c_str(),
                         row.second);
        std::fprintf(stderr, "backend_validation[%s]: required_matmuls=%d unassigned=%d compute_success=%d\n", label,
                     required_matmuls, unassigned, computed);
        const auto nz = nz_usage();
        std::fprintf(stderr,
                     "backend_validation[%s]: nz_cached_matmuls=%d nz_unique_weights=%d nz_bytes=%zu "
                     "nz_selections=%llu nz_selected_weights=%d\n",
                     label, nz.matmuls, nz.weights, nz.bytes, (unsigned long long)nz.selections, nz.selected_weights);
    }
};

} // namespace core_backend_validation
