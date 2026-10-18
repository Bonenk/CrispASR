#pragma once
#include <emscripten.h>
#include <atomic>
#include <stdexcept>
// Shared native flag also guards callers on another runtime thread. Only
// supported atomic progress queries bypass the active-compute restriction.
static std::atomic<bool> binding_browser_busy{false};
EM_JS(void, binding_check_callback, (), {
    if (Module['crispasrInCallback'])
        throw new Error('Session re-entry from a native callback is not allowed');
});
static void binding_check_reentry() {
    binding_check_callback();
    if (binding_browser_busy.load(std::memory_order_acquire))
        throw std::runtime_error("Browser compute is busy");
}
