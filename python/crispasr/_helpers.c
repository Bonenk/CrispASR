// Thin C helpers for Python ctypes binding.
// These wrap whisper.h functions that take structs by value,
// providing pointer-based alternatives that ctypes can call.

#include "crispasr.h"
#include "crispasr_session.h"

#ifdef _WIN32
#define CRISPASR_HELPER_API __declspec(dllexport)
#else
#define CRISPASR_HELPER_API
#endif

// whisper_full takes params by value — this wrapper takes a pointer.
CRISPASR_HELPER_API int whisper_full_ptr(struct whisper_context* ctx, const struct whisper_full_params* params,
                                         const float* samples, int n_samples) {
    return whisper_full(ctx, *params, samples, n_samples);
}

// whisper_init_from_file_with_params takes params by value — pointer wrapper.
CRISPASR_HELPER_API struct whisper_context* whisper_init_from_file_ptr(const char* path,
                                                                       const struct whisper_context_params* params) {
    return whisper_init_from_file_with_params(path, *params);
}

// Preserve the native progress-tracking parallel entry point without exposing
// a compiler-dependent whisper_full_params layout to ctypes.
CRISPASR_HELPER_API int crispasr_transcribe_parallel_ptr(struct whisper_context* ctx,
                                                         const struct whisper_full_params* params, const float* samples,
                                                         int n_samples, int n_processors) {
    return crispasr_transcribe_parallel(ctx, *params, samples, n_samples, n_processors);
}
