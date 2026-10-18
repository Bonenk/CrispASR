// Model-free measurement bridge for our CTC kernel and our immutable baseline.
#include "align.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

std::vector<ctc_word_stamp> ctc_forced_align_reference(const float*, int, int, const std::vector<std::string>&,
                                                       const std::vector<std::string>&, int, float);

#include "ctc_alloc_tracker.h"

namespace {
template <class T> void put(std::string& s, T v) {
    s.append(reinterpret_cast<const char*>(&v), sizeof(v));
}
void put_text(std::string& s, const std::string& text) {
    put<uint32_t>(s, static_cast<uint32_t>(text.size()));
    s += text;
}
} // namespace
// Binary wire format is test-only: host little-endian counts and F32 bit patterns.
extern "C" void* probe(const float* logits, int T, int V, const char** words, int n_words, const char** vocab,
                       int n_vocab, int blank, float frame_dur, int reference, size_t* out_size, size_t* out_peak,
                       size_t* out_allocs, uint64_t* out_ns) {
    std::vector<std::string> ws, vs;
    for (int i = 0; i < n_words; ++i)
        ws.emplace_back(words[i]);
    for (int i = 0; i < n_vocab; ++i)
        vs.emplace_back(vocab[i]);
    current_bytes = peak_bytes = allocations = 0;
    measuring = true;
    const auto start = std::chrono::steady_clock::now();
    auto result = reference ? ctc_forced_align_reference(logits, T, V, ws, vs, blank, frame_dur)
                            : ctc_forced_align(logits, T, V, ws, vs, blank, frame_dur);
    const auto end = std::chrono::steady_clock::now();
    measuring = false;
    *out_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    *out_peak = peak_bytes;
    *out_allocs = allocations;
    std::string bytes;
    put<uint32_t>(bytes, static_cast<uint32_t>(result.size()));
    for (const auto& word : result) {
        put_text(bytes, word.word);
        put(bytes, word.t0);
        put(bytes, word.t1);
        put<uint32_t>(bytes, static_cast<uint32_t>(word.characters.size()));
        for (const auto& ch : word.characters) {
            put_text(bytes, ch.character);
            put(bytes, ch.t0);
            put(bytes, ch.t1);
        }
    }
    *out_size = bytes.size();
    void* out = std::malloc(bytes.size());
    if (!out)
        return nullptr;
    std::memcpy(out, bytes.data(), bytes.size());
    return out;
}
extern "C" void dispose(void* p) {
    std::free(p);
}

namespace {
bool read_exact(void* p, size_t n) {
    return std::fread(p, 1, n, stdin) == n;
}
std::string read_text() {
    uint32_t n = 0;
    if (!read_exact(&n, sizeof(n)) || n > 1048576)
        std::exit(3);
    std::string s(n, '\0');
    if (!read_exact(s.data(), n))
        std::exit(3);
    return s;
}
} // namespace
int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    int32_t h[8];
    while (read_exact(h, sizeof(h))) {
        float duration;
        if (!read_exact(&duration, sizeof(duration)))
            return 3;
        if (h[2] < 0 || h[3] < 0 || h[7] < 0)
            return 3;
        std::vector<std::string> words, vocab;
        for (int i = 0; i < h[2]; ++i)
            words.push_back(read_text());
        for (int i = 0; i < h[3]; ++i)
            vocab.push_back(read_text());
        std::vector<const char*> wp, vp;
        for (const auto& w : words)
            wp.push_back(w.c_str());
        for (const auto& v : vocab)
            vp.push_back(v.c_str());
        std::vector<float> logits(h[7]);
        if (!read_exact(logits.data(), logits.size() * sizeof(float)))
            return 3;
        if (h[6] < 0)
            unsetenv(argv[1]);
        else
            setenv(argv[1], h[6] == 0 ? "0" : "1", 1);
        size_t size = 0, peak = 0, allocs = 0;
        uint64_t ns = 0;
        void* bytes = probe(logits.empty() ? nullptr : logits.data(), h[0], h[1], wp.data(), h[2], vp.data(), h[3],
                            h[4], duration, h[5], &size, &peak, &allocs, &ns);
        if (!bytes)
            return 4;
        uint64_t metrics[] = {static_cast<uint64_t>(size), static_cast<uint64_t>(peak), static_cast<uint64_t>(allocs),
                              ns};
        if (std::fwrite(metrics, 1, sizeof(metrics), stdout) != sizeof(metrics) ||
            std::fwrite(bytes, 1, size, stdout) != size || std::fflush(stdout) != 0)
            return 5;
        dispose(bytes);
    }
    return std::feof(stdin) ? 0 : 3;
}
