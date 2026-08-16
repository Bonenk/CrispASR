// src/core/ggml_cpu_backend.h — CPU-backend access that survives GGML_BACKEND_DL.
//
// Issue #355: the shipped `-cuda` tarball carries libcuda.so.1 as a hard
// DT_NEEDED, so on a host without the NVIDIA driver the loader kills the
// process with exit 127 before main() runs and the advertised CPU fallback
// never gets a chance. The fix ggml provides is GGML_BACKEND_DL: every backend
// becomes a dlopen'd module, so a CUDA backend that cannot load simply is not
// registered and ggml_backend_init_best() picks CPU.
//
// Two things block that in this tree. The first is CMake — every per-model
// library links `ggml-cuda` / `ggml-metal` explicitly (a MODULE target cannot
// be linked), handled by the crispasr_link_ggml_* interface targets in the
// top-level CMakeLists.
//
// The second is this file's reason to exist. Under DL the CPU backend is a
// module too, so `ggml_backend_cpu_init`, `ggml_backend_is_cpu` and friends are
// not linkable — and this tree calls them at ~424 sites across 104 files. All
// six have exact registry equivalents; these wrappers are those, selected at
// compile time.
//
// **Without GGML_BACKEND_DL every wrapper is the direct call it replaces**, so
// the default build is unchanged — same symbols, same codegen, no runtime
// lookup. That is deliberate: the DL path is new and unproven on CUDA/HIP/
// Vulkan hardware, so it must not be able to alter the build everyone ships.
#pragma once

#include "ggml-backend.h"

#include <cstring>

#ifndef GGML_BACKEND_DL
#include "ggml-cpu.h"
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#endif

namespace core_cpu_backend {

#ifndef GGML_BACKEND_DL

// Statically linked CPU backend: call straight through. Identical to what the
// call sites did before, so the shipped build is byte-for-byte the same.
inline ggml_backend_t init() {
    return ggml_backend_cpu_init();
}
inline bool is_cpu(ggml_backend_t b) {
    return ggml_backend_is_cpu(b);
}
inline void set_n_threads(ggml_backend_t b, int n) {
    ggml_backend_cpu_set_n_threads(b, n);
}
inline ggml_backend_buffer_type_t buffer_type() {
    return ggml_backend_cpu_buffer_type();
}
inline ggml_backend_reg_t reg() {
    return ggml_backend_cpu_reg();
}
inline void set_threadpool(ggml_backend_t b, ggml_threadpool_t tp) {
    ggml_backend_cpu_set_threadpool(b, tp);
}
inline bool is_metal(ggml_backend_t b) {
#ifdef GGML_USE_METAL
    return ggml_backend_is_metal(b);
#else
    (void)b;
    return false;
#endif
}

#else

// Dynamic backends: the CPU backend is a module like any other, reached
// through the registry. ggml_backend_load_all() must have run first — it does,
// from crispasr_c_api.cpp and the CLI entry points.
inline ggml_backend_dev_t cpu_device() {
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
}

inline ggml_backend_t init() {
    ggml_backend_dev_t dev = cpu_device();
    return dev ? ggml_backend_dev_init(dev, nullptr) : nullptr;
}

inline bool is_cpu(ggml_backend_t b) {
    if (!b) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    return dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

inline void set_n_threads(ggml_backend_t b, int n) {
    if (!b) {
        return;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    ggml_backend_reg_t r = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (!r) {
        return;
    }
    // The registry exposes it as a named proc rather than a linked symbol.
    typedef void (*set_n_threads_fn)(ggml_backend_t, int);
    auto fn = (set_n_threads_fn)ggml_backend_reg_get_proc_address(r, "ggml_backend_set_n_threads");
    if (fn) {
        fn(b, n);
    }
}

inline ggml_backend_buffer_type_t buffer_type() {
    ggml_backend_dev_t dev = cpu_device();
    return dev ? ggml_backend_dev_buffer_type(dev) : nullptr;
}

inline ggml_backend_reg_t reg() {
    ggml_backend_dev_t dev = cpu_device();
    return dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
}

// ⚠ No-op under DL, and this is a real functional difference rather than an
// oversight: ggml's CPU registry exposes `ggml_backend_set_n_threads` through
// get_proc_address but NOT the threadpool setter (see
// ggml_backend_cpu_get_proc_address in ggml/src/ggml-cpu/ggml-cpu.cpp — the
// list is n_threads, extra_bufts, features, abort_callback, numa_init,
// is_numa). So a DL build cannot install the shared worker pool and falls back
// to ggml's own per-call threading. Exposing it needs a fork patch to that
// switch; until then this is why the DL path stays opt-in.
inline void set_threadpool(ggml_backend_t, ggml_threadpool_t) {}

inline bool is_metal(ggml_backend_t b) {
    if (!b) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    ggml_backend_reg_t r = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    const char* n = r ? ggml_backend_reg_name(r) : nullptr;
    return n && std::strcmp(n, "Metal") == 0;
}

#endif

} // namespace core_cpu_backend
