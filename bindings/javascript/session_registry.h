#pragma once
#include "crispasr_session.h"
#include <algorithm>

static void binding_registry_input(const std::string& value) {
    if (value.find('\0') != std::string::npos)
        throw std::invalid_argument("Embedded NUL in registry/cache argument");
}
static std::string binding_registry_string(const std::vector<char>& buffer) {
    auto end = std::find(buffer.begin(), buffer.end(), '\0');
    if (end == buffer.end())
        throw std::runtime_error("Unterminated native registry/cache output");
    return std::string(buffer.begin(), end);
}
// Native APIs have different buffer-small status values. Retry only that
// value, with a fixed ceiling; never treat failures as truncated success.
static constexpr int binding_registry_limit = 1024 * 1024;
template <typename Lookup> static emscripten::val binding_registry_entry(Lookup lookup, bool artifact = false) {
    using emscripten::val;
    for (int capacity = 256; capacity <= binding_registry_limit; capacity *= 2) {
        std::vector<char> filename(capacity), url(capacity), size(capacity);
        int32_t kind = -1;
        int rc = lookup(filename, url, size, kind);
        if (rc == 2)
            continue;
        if (rc == 1)
            return val::null();
        if (rc != 0)
            throw std::runtime_error("Registry lookup failed: " + std::to_string(rc));
        auto out = val::object();
        out.set("filename", binding_registry_string(filename));
        out.set("url", binding_registry_string(url));
        out.set("approxSize", binding_registry_string(size));
        if (artifact) {
            if (kind < CRISPASR_REGISTRY_ARTIFACT_PRIMARY || kind > CRISPASR_REGISTRY_ARTIFACT_EXTRA)
                throw std::runtime_error("Invalid registry artifact kind");
            out.set("kind", kind);
        }
        return out;
    }
    throw std::runtime_error("Registry output exceeds buffer limit");
}
static emscripten::val binding_registry_lookup(const std::string& key, bool filename) {
    binding_check_reentry();
    binding_registry_input(key);
    return binding_registry_entry([&](auto& file, auto& url, auto& size, int32_t&) {
        auto lookup = filename ? crispasr_registry_lookup_by_filename_abi : crispasr_registry_lookup_abi;
        return lookup(key.c_str(), file.data(), file.size(), url.data(), url.size(), size.data(), size.size());
    });
}
static emscripten::val binding_registry_bundle(const std::string& backend) {
    using emscripten::val;
    binding_check_reentry();
    binding_registry_input(backend);
    for (int capacity = 256; capacity <= binding_registry_limit; capacity *= 2) {
        std::vector<char> canonical(capacity), license(capacity);
        int32_t acceptance = 0;
        int n = crispasr_registry_default_bundle_info_abi(backend.c_str(), canonical.data(), capacity, license.data(),
                                                          capacity, &acceptance);
        if (n == -2)
            continue;
        if (n < 0 || n > 10000)
            throw std::runtime_error("Invalid registry bundle count");
        auto name = binding_registry_string(canonical);
        if (name.empty() && n == 0)
            return val::null();
        if (name.empty())
            throw std::runtime_error("Missing canonical backend");
        auto out = val::object(), artifacts = val::array();
        out.set("backend", name);
        out.set("license", binding_registry_string(license));
        out.set("requiresAcceptance", acceptance != 0);
        for (int i = 0; i < n; ++i) {
            auto item = binding_registry_entry(
                [&](auto& file, auto& url, auto& size, int32_t& kind) {
                    return crispasr_registry_default_bundle_artifact_abi(backend.c_str(), i, &kind, file.data(),
                                                                         file.size(), url.data(), url.size(),
                                                                         size.data(), size.size());
                },
                true);
            if (item.isNull())
                throw std::runtime_error("Registry bundle artifact disappeared");
            artifacts.call<void>("push", item);
        }
        out.set("artifacts", artifacts);
        return out;
    }
    throw std::runtime_error("Registry bundle exceeds buffer limit");
}
template <typename Operation> static std::string binding_cache_path(Operation operation) {
    for (int capacity = 256; capacity <= binding_registry_limit; capacity *= 2) {
        std::vector<char> path(capacity);
        int rc = operation(path);
        if (rc == 2)
            continue;
        if (rc != 0)
            throw std::runtime_error("Cache operation failed: " + std::to_string(rc));
        return binding_registry_string(path);
    }
    throw std::runtime_error("Cache path exceeds buffer limit");
}
static void binding_register_registry() {
    using namespace emscripten;
    function("registryLookup",
             optional_override([](const std::string& backend) { return binding_registry_lookup(backend, false); }));
    function("registryLookupByFilename",
             optional_override([](const std::string& filename) { return binding_registry_lookup(filename, true); }));
    function("registryDefaultBundle", &binding_registry_bundle);
    function("registryBackends", optional_override([]() {
                 binding_check_reentry();
                 for (int capacity = 256; capacity <= binding_registry_limit; capacity *= 2) {
                     std::vector<char> csv(capacity);
                     int rc = crispasr_registry_list_backends_abi(csv.data(), capacity);
                     if (rc == -2)
                         continue;
                     if (rc < 0 || rc >= capacity)
                         throw std::runtime_error("Registry backend list failed");
                     auto text = binding_registry_string(csv);
                     if (text.size() != size_t(rc))
                         throw std::runtime_error("Registry backend list length mismatch");
                     auto out = val::array();
                     size_t start = 0;
                     while (start < text.size()) {
                         auto end = text.find(',', start);
                         out.call<void>("push", text.substr(start, end - start));
                         if (end == std::string::npos)
                             break;
                         start = end + 1;
                     }
                     return out;
                 }
                 throw std::runtime_error("Registry backend list exceeds buffer limit");
             }));
    function("cacheDirectory", optional_override([](const std::string& overrideDir) {
                 binding_check_reentry();
                 binding_registry_input(overrideDir);
                 return binding_cache_path([&](auto& path) {
                     return crispasr_cache_dir_abi(overrideDir.empty() ? nullptr : overrideDir.c_str(), path.data(),
                                                   path.size());
                 });
             }));
    function("cacheEnsureFile", optional_override([](const std::string& filename, const std::string& url, bool quiet,
                                                     const std::string& overrideDir) {
                 binding_check_reentry();
                 binding_registry_input(filename);
                 binding_registry_input(url);
                 binding_registry_input(overrideDir);
                 return binding_cache_path([&](auto& path) {
                     return crispasr_cache_ensure_file_abi(filename.c_str(), url.c_str(), quiet,
                                                           overrideDir.empty() ? nullptr : overrideDir.c_str(),
                                                           path.data(), path.size());
                 });
             }));
}
