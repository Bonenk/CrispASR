#pragma once
#include "crispasr_session.h"
#include <array>
#include <atomic>
#include <climits>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

// The WASM ASR surface owns one singleton. Userdata contains a nonreused
// generation ticket, never a pointer to a JS value or a destructible owner.
struct binding_asr_event {
    int kind = 0, version = 0, index = 0, processed = 0, total = 0;
    int64_t t0 = 0, t1 = 0;
    std::string text;
};
static std::mutex binding_asr_events_mutex;
static crispasr_session* binding_asr_events_parent = nullptr;
static std::array<std::atomic<int>, 3> binding_asr_event_versions{};
static int binding_asr_event_next_version = 1;
static std::deque<binding_asr_event> binding_asr_events;
static size_t binding_asr_events_bytes = 0;
static std::atomic<bool> binding_asr_events_failed{false};
static std::atomic<int> binding_asr_event_live{0};
static constexpr size_t binding_asr_event_byte_limit = 8 * 1024 * 1024;
static void binding_asr_event_push(int kind, void* userdata, const char* text, int index, int64_t t0, int64_t t1,
                                   int processed, int total) noexcept {
    try {
        std::lock_guard<std::mutex> lock(binding_asr_events_mutex);
        int version = int(reinterpret_cast<uintptr_t>(userdata));
        if (!version || version != binding_asr_event_versions[kind])
            return;
        if (binding_asr_events_failed.load())
            return;
        size_t n = text ? strnlen(text, binding_asr_event_byte_limit + 1) : 0;
        if (binding_asr_events.size() >= 4096 || n > binding_asr_event_byte_limit - binding_asr_events_bytes ||
            index < 0 || processed < 0 || total < processed || t0 < -9007199254740991LL || t0 > 9007199254740991LL ||
            t1 < -9007199254740991LL || t1 > 9007199254740991LL) {
            binding_asr_events_failed = true;
            return;
        }
        binding_asr_event event{kind, version, index, processed, total, t0, t1, text ? std::string(text, n) : ""};
        binding_asr_events.push_back(std::move(event));
        binding_asr_events_bytes += n;
    } catch (...) {
        // Never propagate allocation errors through a native decoder callback.
        binding_asr_events_failed = true;
    }
}
static void binding_asr_segment_event(const char* text, int64_t t0, int64_t t1, int index, void* data) {
    binding_asr_event_push(0, data, text, index, t0, t1, 0, 0);
}
static void binding_asr_token_event(const char* text, int index, void* data) {
    binding_asr_event_push(1, data, text, index, 0, 0, 0, 0);
}
static void binding_asr_progress_event(int processed, int total, void* data) {
    binding_asr_event_push(2, data, nullptr, 0, 0, 0, processed, total);
}
static void binding_asr_native_set_callback(crispasr_session* parent, int kind, int version) {
    void* userdata = reinterpret_cast<void*>(uintptr_t(version));
    if (kind == 0)
        crispasr_session_set_segment_callback(parent, version ? binding_asr_segment_event : nullptr, userdata);
    else if (kind == 1)
        crispasr_session_set_token_callback(parent, version ? binding_asr_token_event : nullptr, userdata);
    else
        crispasr_session_set_progress_callback(parent, version ? binding_asr_progress_event : nullptr, userdata);
}
static int binding_asr_install_event(crispasr_session* parent, int kind, bool enabled) {
    int version;
    {
        std::lock_guard<std::mutex> lock(binding_asr_events_mutex);
        if (enabled && binding_asr_event_next_version == INT_MAX)
            throw std::runtime_error("ASR callback generation space exhausted");
        version = enabled ? binding_asr_event_next_version++ : 0;
        binding_asr_events_parent = parent;
        binding_asr_event_versions[kind] = version;
        for (auto it = binding_asr_events.begin(); it != binding_asr_events.end();) {
            if (it->kind == kind) {
                binding_asr_events_bytes -= it->text.size();
                it = binding_asr_events.erase(it);
            } else {
                ++it;
            }
        }
    }
    binding_asr_native_set_callback(parent, kind, version);
    return version;
}
static void binding_asr_clear_events(crispasr_session* parent) {
    {
        std::lock_guard<std::mutex> lock(binding_asr_events_mutex);
        if (binding_asr_events_parent != parent)
            return;
        binding_asr_events_parent = nullptr;
        for (auto& version : binding_asr_event_versions)
            version.store(0);
        binding_asr_events.clear();
        binding_asr_events_bytes = 0;
        binding_asr_events_failed = false;
    }
    // Clearing segment/token callbacks restores the native default pollers.
    for (int kind = 0; kind < 3; ++kind)
        binding_asr_native_set_callback(parent, kind, 0);
}
static int binding_asr_event_version(int kind) {
    return binding_asr_event_versions[kind].load();
}
static int binding_asr_event_take_error() {
    return binding_asr_events_failed.exchange(false);
}
static uintptr_t binding_asr_event_pop() noexcept {
    try {
        // The servicer must never wait for a producer which may need browser
        // services. Retry on the next pump when the short queue lock is busy.
        std::unique_lock<std::mutex> lock(binding_asr_events_mutex, std::try_to_lock);
        if (!lock.owns_lock() || binding_asr_events.empty())
            return 0;
        auto event = std::make_unique<binding_asr_event>(std::move(binding_asr_events.front()));
        binding_asr_events_bytes -= event->text.size();
        binding_asr_events.pop_front();
        ++binding_asr_event_live;
        return reinterpret_cast<uintptr_t>(event.release());
    } catch (...) {
        binding_asr_events_failed = true;
        return 0;
    }
}
static void binding_asr_event_destroy(uintptr_t raw) {
    delete reinterpret_cast<binding_asr_event*>(raw);
    --binding_asr_event_live;
}
