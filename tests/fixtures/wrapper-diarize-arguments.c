#include "crispasr_session.h"
#include "runtime-abi-layout.h"
#include <string.h>
static int dia_mode, dia_calls, dia_retried;
void probe_diarize_mode(int mode) {
    dia_mode = mode;
    dia_retried = 0;
}
int probe_diarize_calls(void) {
    return dia_calls;
}
static int dia_arguments(const float* left, int n, crispasr_diarize_seg_abi* segs, int count,
                         const crispasr_diarize_opts_abi* opts) {
    ++dia_calls;
    return left && n == 3 && left[0] == 1 && left[2] == 5 && count == 2 && segs && segs[0].t0_cs == 125 &&
           segs[0].t1_cs == 175 && segs[1].t0_cs == 200 && segs[1].t1_cs == 250 && opts && opts->method == 4 &&
           opts->n_threads == 2 && opts->slice_t0_cs == 125 && opts->pyannote_model_path == NULL &&
           opts->foxnose_embedder_path && !strcmp(opts->foxnose_embedder_path, "embed") && opts->min_speakers == 2 &&
           opts->max_speakers == 4 && opts->num_speakers == 3;
}
int32_t crispasr_diarize_segments_abi(const float* left, const float* right, int32_t n, int32_t stereo,
                                      crispasr_diarize_seg_abi* segs, int32_t count,
                                      const crispasr_diarize_opts_abi* opts) {
#ifdef CRISPASR_PROBE_WASM_DIA
    if ((stereo && (!right || right[0] != 5 || right[2] != 1)) || (!stereo && right) || !opts || opts->_pad2)
        return 1;
#endif
    if (!dia_arguments(left, n, segs, count, opts))
        return 1;
    if (dia_mode == 4)
        return 1;
    segs[0].speaker = 1;
    segs[1].speaker = 0;
    return 0;
}
int32_t crispasr_diarize_segments_turns_abi(const float* left, const float* right, int32_t n, int32_t stereo,
                                            crispasr_diarize_seg_abi* segs, int32_t count,
                                            const crispasr_diarize_opts_abi* opts, crispasr_diarize_turn_abi* turns,
                                            int32_t cap, int32_t* out_count) {
#ifdef CRISPASR_PROBE_WASM_DIA
    if ((stereo && (!right || right[0] != 5 || right[2] != 1)) || (!stereo && right) || !opts || opts->_pad2)
        return 1;
#endif
    if (!dia_arguments(left, n, segs, count, opts))
        return 1;
    if (dia_mode == 4)
        return 1;
    if (dia_mode == 1 && !dia_retried) {
        dia_retried = 1;
        *out_count = cap + 1;
        return 2;
    }
    if (dia_mode == 7) {
        segs[0].speaker = 1;
        segs[1].speaker = 0;
        *out_count = 0;
        return 0;
    }
    if (dia_mode == 8) {
        *out_count = cap + 1;
        return 2;
    }
    if (dia_mode == 5) {
        *out_count = cap;
        return 2;
    }
    if (dia_mode == 6) {
        *out_count = 1024 * 1024 + 1;
        return 2;
    }
    if (dia_mode == 2) {
        *out_count = -1;
        return 0;
    }
    if (dia_mode == 3) {
        *out_count = cap + 1;
        return 0;
    }
    if (cap < 2) {
        *out_count = 2;
        return 2;
    }
    segs[0].speaker = 1;
    segs[1].speaker = 0;
    turns[0] = (crispasr_diarize_turn_abi){125, 150, 1, 0};
    turns[1] = (crispasr_diarize_turn_abi){150, 250, 0, 0};
    *out_count = 2;
    return 0;
}
