// Deliberately nontrivial by-value structs for testing the compiled adapter.
// This is an ABI transport fixture, not a substitute for the real model API.
#pragma once
#include <stdint.h>
struct whisper_context;
struct whisper_context_params {
    int enabled, preset, top, padding[13];
};
struct whisper_full_params {
    int marker;
    const char* language;
    const char* prompt;
    int timestamps;
    double padding[8];
};
int whisper_full(struct whisper_context*, struct whisper_full_params, const float*, int);
struct whisper_context* whisper_init_from_file_with_params(const char*, struct whisper_context_params);
