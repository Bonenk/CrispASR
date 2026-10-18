// Test-only executable body, appended to the literal production postprocessor.
#include "ctc_alloc_tracker.h"
#include <chrono>
#include <cstdio>

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
template <class T> void put(const T& value) {
    std::fwrite(&value, sizeof(value), 1, stdout);
}
} // namespace
int main() {
    int32_t h[8]; // T,V,word count,vocab count,blank,duration,reference,gate
    while (read_exact(h, sizeof(h))) {
        if (h[0] <= 0 || h[1] <= 0 || h[2] < 0 || h[3] < 0)
            return 3;
        canary_ctc_context ctx;
        ctx.model.hparams.blank_id = h[4];
        ctx.model.hparams.frame_dur_cs = h[5];
        for (int i = 0; i < h[3]; ++i) {
            auto text = read_text();
            ctx.vocab.id_to_token.push_back(text);
            ctx.vocab.token_to_id[text] = i;
        }
        std::vector<std::string> words;
        for (int i = 0; i < h[2]; ++i)
            words.push_back(read_text());
        std::vector<const char*> pointers;
        for (const auto& word : words)
            pointers.push_back(word.c_str());
        std::vector<float> logits((size_t)h[0] * h[1]);
        if (!read_exact(logits.data(), logits.size() * sizeof(float)))
            return 3;
        std::vector<canary_ctc_word> output(h[2]);
        if (h[7] < 0)
            unsetenv("CRISPASR_SUBWORD_CTC_ALIGN_OPT");
        else
            setenv("CRISPASR_SUBWORD_CTC_ALIGN_OPT", h[7] == 1 ? "1" : h[7] == 2 ? "true" : "0", 1);
        current_bytes = peak_bytes = allocations = 0;
        measuring = true;
        auto start = std::chrono::steady_clock::now();
        int rc = h[6] ? canary_ctc_align_words_reference(&ctx, logits.data(), h[0], h[1], pointers.data(), h[2],
                                                         output.data())
                      : canary_ctc_align_words(&ctx, logits.data(), h[0], h[1], pointers.data(), h[2], output.data());
        auto end = std::chrono::steady_clock::now();
        measuring = false;
        put((uint64_t)peak_bytes);
        put((uint64_t)allocations);
        put((uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        put((int32_t)rc);
        for (const auto& word : output) {
            std::fwrite(word.text, sizeof(word.text), 1, stdout);
            put(word.t0);
            put(word.t1);
        }
        std::fflush(stdout);
    }
    return 0;
}
