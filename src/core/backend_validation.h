#pragma once

#include "ggml-backend.h"
#include "core/ggml_cpu_backend.h"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

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

    Placement() = default;

    Placement(ggml_backend_sched_t sched, ggml_cgraph* graph, ggml_backend_t required) {
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto* node = ggml_graph_node(graph, i);
            auto backend = ggml_backend_sched_get_tensor_backend(sched, node);
            if (!backend) {
                unassigned++;
                continue;
            }
            operations[std::string(ggml_backend_name(backend)) + "/" + ggml_op_name(node->op)]++;
            if (backend == required && node->op == GGML_OP_MUL_MAT)
                required_matmuls++;
        }
    }

    bool valid() const { return required_matmuls > 0 && unassigned == 0; }

    void dump(const char* label, bool computed) const {
        for (const auto& row : operations)
            std::fprintf(stderr, "backend_validation[%s]: assigned=%s nodes=%d\n", label, row.first.c_str(),
                         row.second);
        std::fprintf(stderr, "backend_validation[%s]: required_matmuls=%d unassigned=%d compute_success=%d\n", label,
                     required_matmuls, unassigned, computed);
    }
};

} // namespace core_backend_validation
