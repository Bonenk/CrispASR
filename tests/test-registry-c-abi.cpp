// test-registry-c-abi.cpp — pin canonical default-bundle C ABI semantics.

#include <catch2/catch_test_macros.hpp>

#include "crispasr_session.h"

#include <cstdint>
#include <string>

TEST_CASE("registry C ABI: Granite5 default includes pinned Apache notices", "[unit][registry][c-abi][granite5]") {
    char backend[64] = {};
    char license[256] = {};
    int32_t requires_acceptance = -1;
    REQUIRE(crispasr_registry_default_bundle_info_abi("granite", backend, sizeof(backend), license, sizeof(license),
                                                      &requires_acceptance) == 6);
    REQUIRE(std::string(backend) == "granite");
    REQUIRE(std::string(license) == "Apache-2.0");
    REQUIRE(requires_acceptance == 0);
    const char* legal_names[] = {"LICENSE", "LICENSE-RECIPE.txt", "NOTICE", "original-publisher-README.md",
                                 "validation-evidence.json"};
    std::string release_prefix;
    for (int index = 0; index < 6; ++index) {
        int32_t kind = -1;
        char filename[128] = {}, url[512] = {}, size[64] = {};
        REQUIRE(crispasr_registry_default_bundle_artifact_abi("granite", index, &kind, filename, sizeof(filename), url,
                                                              sizeof(url), size, sizeof(size)) == 0);
        REQUIRE(kind == (index == 0 ? CRISPASR_REGISTRY_ARTIFACT_PRIMARY : CRISPASR_REGISTRY_ARTIFACT_EXTRA));
        REQUIRE(std::string(url).find("/granite-speech-5.0-2b-GGUF/resolve/") != std::string::npos);
        REQUIRE(std::string(url).find("/resolve/main/") == std::string::npos);
        if (index == 0) {
            REQUIRE(std::string(filename) == "granite-speech-5.0-2b-q4_k.gguf");
            REQUIRE_FALSE(std::string(size).empty());
            const std::string primary_url = url;
            release_prefix = primary_url.substr(0, primary_url.rfind('/') + 1);
        } else {
            REQUIRE(std::string(filename) == std::string("granite-speech-5.0-2b/") + legal_names[index - 1]);
            REQUIRE(std::string(url) == release_prefix + legal_names[index - 1]);
        }
    }
}

TEST_CASE("registry C ABI: Granite5 Q4 default and explicit F16 preserve profile identity",
          "[unit][registry][c-abi][granite5]") {
    char filename[128] = {}, url[512] = {}, size[64] = {};
    REQUIRE(crispasr_registry_lookup_abi("granite", filename, sizeof(filename), url, sizeof(url), size, sizeof(size)) ==
            0);
    REQUIRE(std::string(filename) == "granite-speech-5.0-2b-q4_k.gguf");
    const std::string default_url = url;
    for (const char* quant : {"q4_k", "f16"}) {
        const std::string requested = std::string("granite-speech-5.0-2b-") + quant + ".gguf";
        REQUIRE(crispasr_registry_lookup_by_filename_abi(requested.c_str(), filename, sizeof(filename), url,
                                                         sizeof(url), size, sizeof(size)) == 0);
        REQUIRE(std::string(filename) == requested);
        REQUIRE_FALSE(std::string(size).empty());
        if (std::string(quant) == "f16")
            REQUIRE(std::string(url).find("/resolve/701f5529d8459412cc1b6053b5880c630125b138/") != std::string::npos);
        else
            REQUIRE(std::string(url) == default_url);
    }
    for (const char* quant : {"q8_0", "q6_k"}) {
        const std::string rejected = std::string("granite-speech-5.0-2b-") + quant + ".gguf";
        REQUIRE(crispasr_registry_lookup_by_filename_abi(rejected.c_str(), filename, sizeof(filename), url, sizeof(url),
                                                         size, sizeof(size)) == 1);
    }
}

TEST_CASE("registry C ABI: default bundle metadata is exact", "[unit][registry][c-abi]") {
    REQUIRE(std::string(crispasr_c_api_version()) == "0.7.1");

    char backend[64] = {};
    char license[256] = {};
    int32_t requires_acceptance = -1;

    const int count = crispasr_registry_default_bundle_info_abi("cosyvoice3", backend, sizeof(backend), license,
                                                                sizeof(license), &requires_acceptance);
    REQUIRE(count == 6);
    REQUIRE(std::string(backend) == "cosyvoice3-tts");
    REQUIRE(std::string(license).empty());
    REQUIRE(requires_acceptance == 0);

    int32_t kind = -1;
    char filename[128] = {};
    char url[512] = {};
    char size[64] = {};
    REQUIRE(crispasr_registry_default_bundle_artifact_abi("cosyvoice3", 0, &kind, filename, sizeof(filename), url,
                                                          sizeof(url), size, sizeof(size)) == 0);
    REQUIRE(kind == CRISPASR_REGISTRY_ARTIFACT_PRIMARY);
    REQUIRE(std::string(filename) == "cosyvoice3-llm-q4_k.gguf");
    REQUIRE(std::string(url).find("huggingface.co/") != std::string::npos);
    REQUIRE_FALSE(std::string(size).empty());

    REQUIRE(crispasr_registry_default_bundle_artifact_abi("cosyvoice3", 2, &kind, filename, sizeof(filename), url,
                                                          sizeof(url), size, sizeof(size)) == 0);
    REQUIRE(kind == CRISPASR_REGISTRY_ARTIFACT_EXTRA);
    REQUIRE(std::string(size).empty());
}

TEST_CASE("registry C ABI: default bundle reports license policy", "[unit][registry][c-abi]") {
    char backend[64] = {};
    char license[256] = {};
    int32_t requires_acceptance = 0;

    REQUIRE(crispasr_registry_default_bundle_info_abi("voxtral-tts", backend, sizeof(backend), license, sizeof(license),
                                                      &requires_acceptance) == 1);
    REQUIRE(std::string(backend) == "voxtral-tts");
    REQUIRE(std::string(license).find("CC-BY-NC-4.0") == 0);
    REQUIRE(requires_acceptance == 1);
}

TEST_CASE("registry C ABI: default bundle validates misses and buffers", "[unit][registry][c-abi]") {
    char backend[64] = {};
    char license[256] = {};
    int32_t requires_acceptance = 0;

    REQUIRE(crispasr_registry_default_bundle_info_abi("nonexistent-backend-xyz", backend, sizeof(backend), license,
                                                      sizeof(license), &requires_acceptance) == 0);
    REQUIRE(crispasr_registry_default_bundle_info_abi("omnivoice", backend, 1, license, sizeof(license),
                                                      &requires_acceptance) == -2);

    int32_t kind = -1;
    char filename[128] = {};
    char url[512] = {};
    char size[64] = {};
    REQUIRE(crispasr_registry_default_bundle_artifact_abi("omnivoice", 99, &kind, filename, sizeof(filename), url,
                                                          sizeof(url), size, sizeof(size)) == 1);
    REQUIRE(crispasr_registry_default_bundle_artifact_abi("omnivoice", 0, &kind, filename, 1, url, sizeof(url), size,
                                                          sizeof(size)) == 2);
}
