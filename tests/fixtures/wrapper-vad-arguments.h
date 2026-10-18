// Compiled only by model-free probes. Layout is extracted from the native implementation.
#include "runtime-vad-layout.h"
#include <string.h>
static int vad_expected, vad_calls;
void probe_vad_expect(int expected) {
    vad_expected = expected;
}
int probe_vad_calls(void) {
    return vad_calls;
}
static int probe_vad_arguments(const float* pcm, int n, int rate, const char* model, const crispasr_vad_abi_opts* opts,
                               const char* language) {
    ++vad_calls;
    if (!pcm || n != 3 || rate != 16000 || !model || strcmp(model, "vad"))
        return 0;
    if (vad_expected == 3)
        return 0;
    if (!vad_expected)
        return opts == NULL && language == NULL;
    if (!language || strcmp(language, "ja"))
        return 0;
    if (vad_expected == 2)
        return opts == NULL;
    return opts && sizeof(*opts) == 24 && opts->threshold == .375f && opts->min_speech_duration_ms == 123 &&
           opts->min_silence_duration_ms == 456 && opts->speech_pad_ms == 78 && opts->chunk_seconds == 9 &&
           opts->n_threads == 2;
}
