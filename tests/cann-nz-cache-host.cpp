// Host lifecycle fixture: production CANN cache functions, mocked memory/SDK only.
// This deliberately proves no Ascend conversion or numerical inference behavior.
#include "ggml.h"
#include "ggml-backend.h"
#include "weight_layout.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <unordered_map>
#include <vector>

static int checks = 0;
static void expect(bool ok, const char* name) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL host lifecycle: %s\n", name);
        std::exit(1);
    }
}
static bool fail_allocation = false;
static std::set<void*> allocations;
static int aclrtFree(void* p) {
    allocations.erase(p);
    std::free(p);
    return 0;
}
static int aclrtMalloc(void** p, size_t size, int) {
    if (fail_allocation) {
        *p = nullptr;
        return 1;
    }
    *p = std::malloc(size);
    allocations.insert(*p);
    return *p ? 0 : 1;
}
static int aclrtMemcpy(void* dst, size_t capacity, const void* src, size_t size, int) {
    expect(capacity >= size, "copy fits allocation");
    std::memcpy(dst, src, size);
    return 0;
}
#define ACL_CHECK(call) expect((call) == 0, #call)
static constexpr int ACL_SUCCESS = 0, ACL_MEM_MALLOC_HUGE_FIRST = 0;
static constexpr int ACL_MEMCPY_HOST_TO_DEVICE = 0, ACL_MEMCPY_DEVICE_TO_DEVICE = 1;
struct aclIntArray {
    int64_t ne0, ne1;
};
static const aclIntArray* aclCreateIntArray(const int64_t* shape, int) {
    return new aclIntArray{shape[0], shape[1]};
}
static int aclDestroyIntArray(const aclIntArray* shape) {
    delete shape;
    return 0;
}
static int ggml_cann_type_mapping(ggml_type type) {
    return type == GGML_TYPE_F32 ? 4 : 2;
}
static int aclnnCalculateMatmulWeightSizeV2(const aclIntArray* shape, int type, size_t* size) {
    *size = size_t(shape->ne0 * shape->ne1) * type;
    return 0;
}
static void ggml_cann_set_device(int) {}
static bool ggml_cann_nz_cache_enabled() {
    return true;
}
static bool parse_bool(const std::string& value) {
    return value == "on";
}
static std::optional<std::string> get_env_as_lowercase(const char*) {
    return "on";
}
extern "C" const char* ggml_get_name(const ggml_tensor* t) {
    return t->name;
}
extern "C" size_t ggml_row_size(ggml_type type, int64_t n) {
    return size_t(n) * (type == GGML_TYPE_F32 ? 4 : 2);
}
extern "C" size_t ggml_nbytes(const ggml_tensor* t) {
    return ggml_row_size(t->type, t->ne[0]) * t->ne[1];
}
extern "C" bool ggml_is_contiguous(const ggml_tensor* t) {
    return t->nb[0] == ggml_row_size(t->type, 1) && t->nb[1] == ggml_row_size(t->type, t->ne[0]);
}
struct ggml_backend_buffer_type {
    bool cann;
};
struct ggml_backend_buffer {
    ggml_backend_buffer_type_t buft;
    void* context;
};
struct ggml_backend_cann_context {};
static bool ggml_backend_buft_is_cann(ggml_backend_buffer_type_t buft) {
    return buft && buft->cann;
}
static void weight_format_to_nz(ggml_tensor* t, int) {
    // Observable stand-in to catch accidental replacement/copy of ordinary bytes.
    auto* p = static_cast<unsigned char*>(t->data);
    for (size_t i = 0; i < ggml_nbytes(t); ++i)
        p[i] ^= 0x5a;
}
static std::atomic<uint64_t> g_nz_cache_epoch[1]{};
#include "cann-cache-production.inc"

static ggml_tensor tensor(void* data, ggml_backend_buffer_t buffer) {
    ggml_tensor t{};
    t.type = GGML_TYPE_F32;
    t.ne[0] = 8;
    t.ne[1] = 4;
    t.ne[2] = t.ne[3] = 1;
    t.nb[0] = 4;
    t.nb[1] = 32;
    t.data = data;
    t.buffer = buffer;
    t.flags = GGML_TENSOR_FLAG_WEIGHT_MATMUL_ONLY;
    std::strcpy(t.name, "arbitrary.linear");
    return t;
}
int main() {
    {
        auto* ordinary = static_cast<unsigned char*>(std::calloc(1, 1024));
        ggml_backend_cann_buffer_context ctx(0, ordinary);
        ggml_backend_buffer_type type{true};
        ggml_backend_buffer buffer{&type, &ctx};
        auto a = tensor(ordinary, &buffer), b = tensor(ordinary + 256, &buffer), c = tensor(ordinary + 512, &buffer);
        ggml_backend_cann_context backend;
        std::vector<unsigned char> raw(128), other(128, 0x33);
        for (size_t i = 0; i < raw.size(); ++i)
            raw[i] = static_cast<unsigned char>(i);
        expect(ggml_cann_set_cached_weight(&ctx, &a, raw.data() + 64, 64, 64), "first chunk accepted");
        expect(ggml_cann_set_cached_weight(&ctx, &a, raw.data() + 64, 64, 64), "duplicate chunk accepted");
        expect(ggml_cann_nz_cache_size(&a) == 0, "duplicate does not fill hole");
        ggml_cann_set_cached_weight(&ctx, &a, raw.data(), 0, 64);
        expect(ggml_cann_nz_cache_size(&a) == 128, "complete cache ready");
        auto* nz = ggml_cann_cached_nz_weight(backend, &a);
        expect(nz && std::memcmp(nz, raw.data(), 128) != 0, "mock packed representation is distinct");
        expect(std::memcmp(a.data, raw.data(), 128) == 0, "ND bytes preserved");
        expect(ggml_cann_nz_cache_selections(&a) == 1, "actual selection count");
        ggml_cann_set_cached_weight(&ctx, &b, other.data(), 0, 128);
        ggml_cann_invalidate_nz(&buffer, ordinary + 128, 1);
        expect(ggml_cann_nz_cache_size(&a) == 128 && ggml_cann_nz_cache_size(&b) == 128,
               "adjacent gap preserves caches");
        const auto epoch = g_nz_cache_epoch[0].load();
        ggml_cann_invalidate_nz(&buffer, ordinary + 4, 0);
        expect(ggml_cann_nz_cache_size(&a) == 128, "zero-length alias write preserves cache");
        ggml_cann_set_cached_weight(&ctx, &a, raw.data(), 0, 0);
        expect(ggml_cann_nz_cache_size(&a) == 128 && g_nz_cache_epoch[0] == epoch,
               "empty upload does not change layout epoch");
        ggml_cann_invalidate_nz(&buffer, ordinary + 4, 1);
        ordinary[4] = 0x77;
        raw[4] = 0x77;
        expect(ggml_cann_nz_cache_size(&a) == 0 && ggml_cann_nz_cache_size(&b) == 128,
               "alias invalidates only overlapping weight");
        expect(ggml_cann_cached_nz_weight(backend, &a) == nullptr, "invalidated cache cannot be selected");
        ggml_cann_set_cached_weight(&ctx, &a, raw.data() + 8, 8, 4);
        expect(ggml_cann_nz_cache_size(&a) == 128 && std::memcmp(a.data, raw.data(), 128) == 0,
               "partial rebuild retains alias mutation");
        ggml_cann_invalidate_nz(&buffer, a.data, 128);
        std::memcpy(a.data, b.data, 128);
        expect(!ggml_cann_nz_cache_size(&a) && !std::memcmp(a.data, other.data(), 128),
               "copy into cached destination invalidates packed data");
        fail_allocation = true;
        ggml_cann_set_cached_weight(&ctx, &c, raw.data(), 0, 128);
        expect(!ggml_cann_nz_cache_size(&c) && !std::memcmp(c.data, raw.data(), 128),
               "allocator failure retains usable ordinary bytes");
        expect(ggml_cann_cached_nz_weight(backend, &c) == nullptr, "allocator failure cannot select a cache");
        fail_allocation = false;
        ggml_cann_set_cached_weight(&ctx, &c, raw.data() + 4, 4, 4);
        expect(ggml_cann_nz_cache_size(&c) == 128, "later upload recovers from allocation failure");
        ggml_cann_invalidate_nz(&buffer);
        expect(!ggml_cann_nz_cache_size(&a) && !ggml_cann_nz_cache_size(&b) && !ggml_cann_nz_cache_size(&c),
               "buffer clear invalidates every cache");
    }
    expect(allocations.empty(), "buffer destruction releases all cache allocations");
    std::printf("PASS host lifecycle checks=%d; mocked SDK, no NPU claim\n", checks);
}
