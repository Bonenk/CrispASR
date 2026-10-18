// Test-only live C++ allocation measurement, shared by standalone probes.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <new>

namespace {
struct alignas(std::max_align_t) Header {
    size_t size;
    bool tracked;
};
bool measuring = false;
size_t current_bytes = 0, peak_bytes = 0, allocations = 0;
} // namespace
void* operator new(size_t n) {
    auto* h = static_cast<Header*>(std::malloc(sizeof(Header) + n));
    if (!h)
        throw std::bad_alloc();
    h->size = n;
    h->tracked = measuring;
    if (measuring) {
        current_bytes += n;
        peak_bytes = std::max(peak_bytes, current_bytes);
        ++allocations;
    }
    return h + 1;
}
void operator delete(void* p) noexcept {
    if (!p)
        return;
    auto* h = static_cast<Header*>(p) - 1;
    if (h->tracked)
        current_bytes -= h->size;
    std::free(h);
}
void operator delete(void* p, size_t) noexcept {
    ::operator delete(p);
}
void* operator new[](size_t n) {
    return ::operator new(n);
}
void operator delete[](void* p) noexcept {
    ::operator delete(p);
}
void operator delete[](void* p, size_t) noexcept {
    ::operator delete(p);
}
