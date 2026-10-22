// #551: schema compatibility is independent of the shared architecture name.
#include <catch2/catch_test_macros.hpp>
#include "core/parakeet_gguf.h"
#include "crispasr_session.h"
#include "ggml.h"

#include <chrono>
#include <filesystem>
#include <string>

namespace {
struct Fixture {
    gguf_context* meta = gguf_init_empty();
    ggml_context* tensors = ggml_init({4096, nullptr, false});
    std::filesystem::path path;

    Fixture() {
        gguf_set_val_str(meta, "general.architecture", "parakeet");
        gguf_set_val_str(meta, "general.name", "parakeet-unified-en-0.6b");
    }
    void tensor(const char* name) {
        auto* t = ggml_new_tensor_1d(tensors, GGML_TYPE_F32, 1);
        ggml_set_name(t, name);
        *static_cast<float*>(t->data) = 0.0f;
        gguf_add_tensor(meta, t);
    }
    void write() {
        path = std::filesystem::temp_directory_path() /
               ("crispasr-parakeet-schema-" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
        REQUIRE(gguf_write_to_file(meta, path.string().c_str(), false));
    }
    ~Fixture() {
        gguf_free(meta);
        ggml_free(tensors);
        if (!path.empty()) {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
        }
    }
};
} // namespace

TEST_CASE("missing familiar predictor names does not identify CTC", "[unit][parakeet-schema]") {
    Fixture f;
    CHECK_FALSE(core_parakeet_gguf::compatible_ctc(f.meta));
    f.tensor("encoder.pre.conv.0.weight");
    CHECK_FALSE(core_parakeet_gguf::compatible_ctc(f.meta));
    f.tensor("ctc.weight");
    CHECK_FALSE(core_parakeet_gguf::compatible_ctc(f.meta));
    f.tensor("ctc.bias");
    CHECK(core_parakeet_gguf::compatible_ctc(f.meta));
}

TEST_CASE("native transducer and hybrid heads never reroute to CTC", "[unit][parakeet-schema]") {
    Fixture f;
    SECTION("embedding predictor") {
        f.tensor("decoder.embed.weight");
    }
    SECTION("LSTM predictor") {
        f.tensor("decoder.lstm.0.w_ih");
    }
    f.tensor("encoder.pre.conv.0.weight");
    f.tensor("ctc.weight");
    f.tensor("ctc.bias");
    CHECK(core_parakeet_gguf::has_predictor(f.meta));
    CHECK_FALSE(core_parakeet_gguf::compatible_ctc(f.meta));
}

TEST_CASE("external Parakeet schema is unsupported regardless of head kind", "[unit][parakeet-schema]") {
    Fixture f;
    SECTION("RNN-T metadata") {
        gguf_set_val_str(f.meta, "stt.parakeet.head_kind", "rnnt");
    }
    SECTION("CTC metadata") {
        gguf_set_val_str(f.meta, "stt.parakeet.head_kind", "ctc");
    }
    SECTION("other schema key") {
        gguf_set_val_u32(f.meta, "stt.parakeet.encoder.d_model", 1024);
    }
    SECTION("external embedding without metadata") {
        f.tensor("pred.embed.weight");
    }
    SECTION("external LSTM without metadata") {
        f.tensor("pred.lstm.0.Wx");
    }
    // Even native-looking head markers cannot override an explicit foreign schema.
    f.tensor("encoder.pre.conv.0.weight");
    f.tensor("ctc.weight");
    f.tensor("ctc.bias");
    CHECK(core_parakeet_gguf::external_schema(f.meta));
    CHECK_FALSE(core_parakeet_gguf::compatible_ctc(f.meta));
    f.write();
    char detected[64] = {};
    CHECK(crispasr_detect_backend_from_gguf(f.path.string().c_str(), detected, sizeof(detected)) == 0);
    CHECK(std::string(detected).empty());
    CHECK(crispasr_detect_backends_from_gguf(f.path.string().c_str(), detected, sizeof(detected)) == 0);
    CHECK(std::string(detected).empty());
    CHECK(crispasr_session_open(f.path.string().c_str(), 1) == nullptr);
    for (const char* backend : {"parakeet", "phonon2", "reazonspeech", "fastconformer-ctc", "canary-ctc",
                                "fastconformer_ctc", "canary_ctc", "quds", "quds-fa", "whisper"}) {
        INFO(backend);
        CHECK(crispasr_session_open_explicit(f.path.string().c_str(), backend, 1) == nullptr);
    }
}

TEST_CASE("C ABI still detects native transducer and compatible CTC schemas", "[unit][parakeet-schema]") {
    Fixture f;
    std::string expected;
    SECTION("native transducer") {
        f.tensor("decoder.embed.weight");
        expected = "parakeet";
    }
    SECTION("compatible CTC under parakeet architecture") {
        f.tensor("encoder.pre.conv.0.weight");
        f.tensor("ctc.weight");
        f.tensor("ctc.bias");
        expected = "fastconformer-ctc";
    }
    SECTION("native CTC architecture") {
        gguf_set_val_str(f.meta, "general.architecture", "canary-ctc");
        expected = "fastconformer-ctc";
    }
    f.write();
    char detected[64] = {};
    CHECK(crispasr_detect_backend_from_gguf(f.path.string().c_str(), detected, sizeof(detected)) > 0);
    CHECK(std::string(detected) == expected);
    CHECK(crispasr_detect_backends_from_gguf(f.path.string().c_str(), detected, sizeof(detected)) > 0);
    CHECK(std::string(detected) == expected);
}
