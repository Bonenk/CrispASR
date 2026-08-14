// test-chat-ggml.cpp — end-to-end smoke for the crispasr_chat_* C ABI.
//
// Gated on CRISPASR_CHAT_TEST_MODEL — a path to a small GGUF chat model
// (e.g. harrier-270m-q4_k.gguf, qwen2.5-0.5b-instruct, smollm2-360m).
// When unset the test is reported as SKIPPED so unrelated builds stay
// green without a model on disk.
//
// Verifies in one pass:
//   • crispasr_chat_open with default params returns a session
//   • crispasr_chat_n_ctx / _template_name return non-trivial values
//   • crispasr_chat_generate returns non-empty UTF-8 (one-shot path)
//   • crispasr_chat_generate_stream fires on_token at least once and
//     the concatenated chunks equal the one-shot output for the same
//     seed (regression guard against streaming drift)
//   • crispasr_chat_reset clears history without crashing
//   • a prompt longer than n_batch still prefills, and the result does
//     not depend on how many prompt batches it was split across

#include <catch2/catch_test_macros.hpp>

#include "crispasr_chat.h"

#include <cstdlib>
#include <cstring>
#include <string>

namespace {

const char* test_model_path() {
    return std::getenv("CRISPASR_CHAT_TEST_MODEL");
}

void on_token_appender(const char* chunk, void* user) {
    auto* out = static_cast<std::string*>(user);
    out->append(chunk);
}

// Whitespace-separated words are never fewer tokens than words, so a
// 1080-word body is at least 1080 prompt tokens — comfortably above the
// 512-token default n_batch and comfortably below the 2048-token n_ctx
// the long-prompt cases open with.
constexpr int kLongPromptSentences = 120;

std::string long_user_message() {
    std::string body;
    body.reserve(kLongPromptSentences * 45);
    for (int i = 0; i < kLongPromptSentences; ++i) {
        body += "The quick brown fox jumps over the lazy dog. ";
    }
    body += "\nReply with the single word: fox.";
    return body;
}

// Greedy generation over the long prompt, opened with the given prompt
// batch size. Returns the generated text; asserts the whole path
// succeeded.
std::string generate_over_long_prompt(const char* model, int32_t n_batch) {
    crispasr_chat_open_params op;
    crispasr_chat_open_params_default(&op);
    op.n_gpu_layers = -1;
    op.n_ctx = 2048;
    op.n_batch = n_batch;

    crispasr_chat_error err{};
    crispasr_chat_session_t s = crispasr_chat_open(model, &op, &err);
    REQUIRE(s != nullptr);
    REQUIRE(err.code == 0);

    crispasr_chat_generate_params gp;
    crispasr_chat_generate_params_default(&gp);
    gp.max_tokens = 16;
    gp.temperature = 0.0f; // greedy → reproducible across batch splits
    gp.seed = 1;

    const std::string user = long_user_message();
    crispasr_chat_message messages[] = {
        {"system", "You are a terse assistant. Answer in one word."},
        {"user", user.c_str()},
    };

    char* out = crispasr_chat_generate(s, messages, 2, &gp, &err);
    REQUIRE(out != nullptr);
    REQUIRE(err.code == 0);
    const std::string text = out;
    crispasr_chat_string_free(out);
    crispasr_chat_close(s);
    return text;
}

} // namespace

TEST_CASE("crispasr_chat one-shot generate", "[chat][gguf]") {
    const char* model = test_model_path();
    if (!model) {
        SKIP("CRISPASR_CHAT_TEST_MODEL not set; skipping chat smoke");
    }

    crispasr_chat_open_params op;
    crispasr_chat_open_params_default(&op);
    op.n_gpu_layers = -1;
    op.n_ctx = 1024;

    crispasr_chat_error err{};
    crispasr_chat_session_t s = crispasr_chat_open(model, &op, &err);
    REQUIRE(s != nullptr);
    REQUIRE(err.code == 0);

    REQUIRE(crispasr_chat_n_ctx(s) > 0);
    const char* tmpl = crispasr_chat_template_name(s);
    REQUIRE(tmpl != nullptr);
    REQUIRE(std::strlen(tmpl) > 0);

    crispasr_chat_generate_params gp;
    crispasr_chat_generate_params_default(&gp);
    gp.max_tokens = 16;
    gp.temperature = 0.0f; // greedy → reproducible across one-shot + stream
    gp.seed = 1;

    crispasr_chat_message messages[] = {
        {"system", "You are a terse assistant. Answer in one word."},
        {"user", "Say hello."},
    };

    char* out = crispasr_chat_generate(s, messages, 2, &gp, &err);
    REQUIRE(out != nullptr);
    REQUIRE(err.code == 0);
    REQUIRE(std::strlen(out) > 0);
    const std::string one_shot = out;
    crispasr_chat_string_free(out);

    // Streaming path with the same seed + greedy must reproduce one-shot.
    REQUIRE(crispasr_chat_reset(s, &err) == 0);
    std::string streamed;
    int32_t rc = crispasr_chat_generate_stream(s, messages, 2, &gp, on_token_appender, &streamed, &err);
    REQUIRE(rc == 0);
    REQUIRE(err.code == 0);
    REQUIRE_FALSE(streamed.empty());
    REQUIRE(streamed == one_shot);

    crispasr_chat_close(s);
}

TEST_CASE("crispasr_chat prompt longer than the prompt batch", "[chat][gguf]") {
    const char* model = test_model_path();
    if (!model) {
        SKIP("CRISPASR_CHAT_TEST_MODEL not set; skipping long-prompt prefill");
    }

    // The prompt exceeds the default 512-token n_batch and fits the 2048
    // context, so it must be prefilled in several prompt batches rather
    // than one oversized decode.
    const std::string text = generate_over_long_prompt(model, /*n_batch=*/512);
    REQUIRE_FALSE(text.empty());
}

TEST_CASE("crispasr_chat long-prompt output is independent of the prompt batch size", "[chat][gguf]") {
    const char* model = test_model_path();
    if (!model) {
        SKIP("CRISPASR_CHAT_TEST_MODEL not set; skipping prompt-batch equivalence");
    }

    // n_batch == n_ctx prefills in a single batch; the 512 default needs
    // several. Greedy sampling must not notice the difference.
    const std::string one_batch = generate_over_long_prompt(model, /*n_batch=*/2048);
    const std::string many_batches = generate_over_long_prompt(model, /*n_batch=*/512);
    REQUIRE_FALSE(one_batch.empty());
    REQUIRE(one_batch == many_batches);
}
