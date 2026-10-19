#include "crispasr.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// PRODUCTION_VAD_TYPES
// PRODUCTION_SEGMENT
struct whisper_state {
    std::vector<whisper_segment> result_all;
    int id = 0;
    int64_t t_mel_us = 0, t_sample_us = 0, t_encode_us = 0, t_decode_us = 0, t_batchd_us = 0, t_prompt_us = 0;
    int n_sample = 0, n_encode = 0, n_decode = 0, n_batchd = 0, n_prompt = 0;
    bool has_vad_segments = false;
    whisper_vad_context* vad_context = nullptr;
    // PRODUCTION_VAD_SEGMENT_INFO
    std::vector<vad_segment_info> vad_segments;
    std::vector<vad_time_mapping> vad_mapping_table;
};
struct whisper_context {
    whisper_state* state;
};
// PRODUCTION_RESET_MAPPING
static int allocated = 0, freed = 0, fail_allocation = -1, failed_worker = -1, throws_worker = -1;
static int vad_calls = 0, sequential_calls = 0, fail_sequential = 0;
static bool vad_silence = false, vad_failure = false, allocation_throws = false;
static std::atomic<int> completed{0}, callbacks{0};
static std::vector<int64_t> logged_boundaries;
struct decode_call {
    const float* samples = nullptr;
    int count = 0, offset_ms = 0, duration_ms = 0;
};
static const float* tracked_pcm = nullptr;
static bool allow_precomputed_mel = false;
static std::array<decode_call, 4> decode_calls;

static void log_message(const char*, ...) {}
#define CRISPASR_LOG_INFO log_message
#define CRISPASR_LOG_ERROR log_message
#define CRISPASR_LOG_WARN log_message
static std::string to_timestamp(int64_t t) {
    logged_boundaries.push_back(t);
    return std::to_string(t);
}
static void require(bool pass, const char* message) {
    if (!pass) {
        fprintf(stderr, "%s\n", message);
        exit(1);
    }
}
struct whisper_state* whisper_init_state(struct whisper_context*) {
    if (allocated == fail_allocation) {
        if (allocation_throws) {
            throw std::bad_alloc();
        }
        return nullptr;
    }
    auto* state = new whisper_state;
    state->id = ++allocated;
    return state;
}
void whisper_free_state(struct whisper_state* state) {
    require(state != nullptr, "freed null state");
    ++freed;
    delete state;
}
static bool whisper_vad(whisper_context*, whisper_state* state, whisper_full_params, const float* samples, int count,
                        std::vector<float>& out) {
    ++vad_calls;
    if (vad_failure) {
        return false;
    }
    state->has_vad_segments = true;
    state->vad_mapping_table = {{0, 100}, {200, 400}};
    if (!vad_silence) {
        out.assign(samples, samples + count);
    }
    return true;
}
int whisper_full_with_state(whisper_context* ctx, whisper_state* state, whisper_full_params params,
                            const float* samples, int n_samples) {
    // PRODUCTION_DECODE_ENTRY
    require(state != nullptr, "decode received null worker state");
    require(n_samples > 0 || allow_precomputed_mel, "decode received empty chunk");
    if (tracked_pcm) {
        require(state->id >= 0 && state->id < int(decode_calls.size()), "unexpected worker ID");
        decode_calls[state->id] = {samples, n_samples, params.offset_ms, params.duration_ms};
    }
    state->result_all.clear();
    const int64_t offset = params.offset_ms / 10;
    whisper_segment seg{};
    seg.t0 = offset + 1;
    seg.t1 = offset + 5;
    whisper_token_data token{};
    token.id = 7;
    token.t0 = offset + 1;
    token.t1 = offset + 2;
    token.t_dtw = offset + 3;
    seg.tokens.push_back(token);
    token.t0 = token.t1 = token.t_dtw = -1;
    seg.tokens.push_back(token);
    seg.alts.resize(2);
    whisper_alt_token alt{};
    alt.id = 42;
    alt.p = .75f;
    seg.alts[0].push_back(alt);
    state->result_all.push_back(std::move(seg));
    state->t_encode_us = 30;
    state->n_encode = 1;
    ++completed;
    if (state->id == throws_worker) {
        throw std::runtime_error("injected worker failure");
    }
    if (state->id == failed_worker) {
        return -17;
    }
    if (state == ctx->state && fail_sequential) {
        return fail_sequential;
    }
    if (params.new_segment_callback) {
        params.new_segment_callback(ctx, state, 1, params.new_segment_callback_user_data);
    }
    return 0;
}
int whisper_full(whisper_context* ctx, whisper_full_params params, const float* samples, int n) {
    ++sequential_calls;
    return whisper_full_with_state(ctx, ctx->state, params, samples, n);
}
// PRODUCTION_PARALLEL

int main(int argc, char** argv) {
    require(argc == 2, "select a case");
    std::string test = argv[1];
    whisper_state state;
    state.result_all.push_back({}); // A previous decode must not survive failure/silence.
    whisper_context ctx{&state};
    whisper_full_params params{};
    params.new_segment_callback = [](whisper_context*, whisper_state*, int, void*) { ++callbacks; };
    int n = 64000, processors = 3;
    params.offset_ms = 1000;
    if (test.rfind("duration", 0) == 0) {
        n = 100007;
        params.duration_ms = 1700;
        if (test == "duration-zero-offset") {
            params.offset_ms = 0;
        } else if (test == "duration-clipped") {
            params.duration_ms = 6000;
        } else if (test == "duration-large") {
            params.duration_ms = 2147483647;
        } else if (test == "duration-fractional") {
            params.offset_ms = 1009;
            params.duration_ms = 1709;
        } else if (test == "duration-tiny") {
            params.duration_ms = 1;
            processors = 32;
        } else if (test == "duration-negative") {
            params.duration_ms = -1;
        }
    }
    if (test == "fractional-offset") {
        params.offset_ms = 1009;
    }
    if (test == "offset") {
        params.offset_ms = 150000;
        n = 2464000;
    }
    if (test == "long") {
        params.offset_ms = 0;
        n = 45000000;
        processors = 2;
    }
    if (test == "worker-error") {
        failed_worker = 1;
    }
    if (test == "main-error") {
        fail_sequential = -19;
    }
    if (test == "worker-throw") {
        throws_worker = 1;
    }
    if (test == "main-throw") {
        throws_worker = 0;
    }
    if (test == "allocation" || test == "allocation-throw") {
        fail_allocation = 1;
        allocation_throws = test == "allocation-throw";
    }
    if (test == "silence" || test == "vad" || test == "vad-failure") {
        params.vad = true;
        vad_silence = test == "silence";
        vad_failure = test == "vad-failure";
    }
    // Uninitialized allocation keeps pointer arithmetic valid even in the long
    // geometry case. Fake decoders never read PCM; only the small VAD case does.
    std::unique_ptr<float[]> pcm(new float[n]);
    if (params.vad) {
        std::fill_n(pcm.get(), n, 0.f);
    }
    if (test.rfind("duration", 0) == 0) {
        tracked_pcm = pcm.get();
        if (test == "duration-negative") {
            require(whisper_full_parallel(&ctx, params, pcm.get(), n, 3) < 0, "negative duration accepted");
            require(whisper_full_parallel(&ctx, params, pcm.get(), n, 1) < 0,
                    "single-worker negative duration accepted");
            require(completed == 0 && allocated == 0, "negative duration reached decoding");
            return 0;
        }
        require(whisper_full_parallel(&ctx, params, pcm.get(), n, processors) == 0, "bounded decode failed");
        require(allocated == freed, "bounded decode leaked worker states");
        // Independent interval oracle: these are exact 16 kHz PCM endpoints,
        // not a second implementation of the production split/merge loop.
        const int64_t begin = int64_t(params.offset_ms) * 16;
        const int64_t end = std::min<int64_t>(n, begin + int64_t(params.duration_ms) * 16);
        const int expected_decoders = test == "duration-tiny" ? 1 : processors;
        require(completed == expected_decoders, "bounded interval launched empty/excess workers");
        int64_t cursor = begin;
        for (int i = 0; i < expected_decoders; ++i) {
            const auto& call = decode_calls[i];
            require(call.samples != nullptr && call.count > 0, "bounded decoder call missing");
            const int64_t effective_begin = call.samples - tracked_pcm + int64_t(call.offset_ms) * 16;
            const int64_t effective_end = call.samples - tracked_pcm + call.count;
            require(effective_begin == cursor, "bounded split skipped or repeated input samples");
            require(effective_end > effective_begin && effective_end <= end,
                    "worker saw audio outside requested interval");
            require(call.duration_ms == 0, "global duration was reapplied separately per worker");
            cursor = effective_end;
        }
        require(cursor == end, "bounded split omitted interval tail/remainder");
        require(decode_calls[0].samples == pcm.get() && decode_calls[0].offset_ms == params.offset_ms,
                "first-chunk offset routing changed");
        return 0;
    }
    if (test == "inputs") {
        require(whisper_full_parallel(&ctx, params, pcm.get(), n, 0) < 0, "accepted zero processors");
        require(whisper_full_parallel(&ctx, params, pcm.get(), n, -1) < 0, "accepted negative processors");
        require(whisper_full_parallel(nullptr, params, pcm.get(), n, 2) < 0, "accepted null context");
        require(whisper_full_parallel(&ctx, params, nullptr, n, 2) < 0, "accepted null PCM");
        require(whisper_full_parallel(&ctx, params, pcm.get(), -1, 2) < 0, "accepted negative samples");
        params.offset_ms = -1;
        require(whisper_full_parallel(&ctx, params, pcm.get(), n, 2) < 0, "accepted negative offset");
        params.offset_ms = 2147483647;
        require(whisper_full_parallel(&ctx, params, pcm.get(), n, 2) == 0, "past-end offset failed");
        require(state.result_all.empty(), "past-end offset retained old results");
        params.offset_ms = 0;
        require(whisper_full_parallel(&ctx, params, pcm.get(), 1, 3) == 0, "small PCM fallback failed");
        require(completed == 1 && allocated == 0, "empty workers were created");
        return 0;
    }
    if (test == "single") {
        processors = 1;
    }
    int rc = whisper_full_parallel(&ctx, params, pcm.get(), n, processors);
    require(allocated == freed, "worker state leaked");
    if (test == "worker-error" || test == "main-error" || test == "worker-throw" || test == "main-throw" ||
        test == "allocation" || test == "allocation-throw" || test == "vad-failure") {
        const int expected = test == "worker-error" ? -17 : test == "main-error" ? -19 : -1;
        require(rc == expected, "failed worker/allocation reported success or wrong status");
        require(state.result_all.empty(), "failure retained partial transcript");
        if (test != "allocation" && test != "allocation-throw" && test != "vad-failure") {
            require(completed == 3, "not all workers joined before return");
        }
        return 0;
    }
    require(rc == 0, "unexpected decode failure");
    if (test == "silence") {
        require(state.result_all.empty(), "VAD silence retained previous transcript");
        require(vad_calls == 1 && completed == 0 && allocated == 0, "silence launched decoders");
        return 0;
    }
    if (test == "single") {
        require(sequential_calls == 1 && allocated == 0, "single processor bypassed sequential route");
        return 0;
    }
    if (test == "vad") {
        require(vad_calls == 1, "VAD applied more than once");
        require(state.has_vad_segments && state.vad_mapping_table.size() == 2 &&
                    state.vad_mapping_table[0].original_time == 100 && state.vad_mapping_table[1].original_time == 400,
                "parent VAD mapping lost");
    }
    require(state.result_all.size() == size_t(processors), "missing merged segments");
    require(callbacks == processors, "segment callback count changed");
    const int64_t offset_samples = int64_t(params.offset_ms) * CRISPASR_SAMPLE_RATE / 1000;
    const int64_t chunk = (n - offset_samples) / processors;
    for (int i = 1; i < processors; ++i) {
        const int64_t shift = 100 * (offset_samples + i * chunk) / CRISPASR_SAMPLE_RATE;
        const auto& seg = state.result_all[i];
        require(seg.t0 == shift + 1 && seg.t1 == shift + 5, "merged segment time is not absolute");
        require(seg.tokens[0].t0 == shift + 1 && seg.tokens[0].t1 == shift + 2 && seg.tokens[0].t_dtw == shift + 3,
                "merged token/DTW time is not absolute");
        require(seg.tokens[1].t0 == -1 && seg.tokens[1].t1 == -1 && seg.tokens[1].t_dtw == -1,
                "unavailable timestamp sentinel changed");
        require(seg.alts.size() == 2 && seg.alts[0][0].id == 42 && seg.alts[0][0].p == .75f,
                "token alternative association changed");
        require(logged_boundaries[i - 1] == shift, "logged split time differs from merge");
    }
    require(state.t_encode_us == 30 && state.n_encode == processors, "timing aggregation changed");
    return 0;
}
