#include "crispasr.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// PRODUCTION_SEGMENT
struct fake_sequence {
    int result_len = 0;
    float avg_logprobs = 0;
    std::vector<whisper_token_data> tokens;
    std::vector<std::vector<whisper_alt_token>> alts;
};
struct fake_decoder {
    int seek_delta = 688;
    fake_sequence sequence;
};
struct whisper_state {
    std::vector<whisper_segment> result_all;
    std::vector<fake_decoder> decoders{1};
    float no_speech_prob = 0;
};
struct fake_vocab {
    bool has_speakers = false;
    bool is_speaker(int) { return false; }
    bool is_timestamp(int id) { return id >= 100 && id < 1000; }
};
struct whisper_context {
    fake_vocab vocab;
    struct {
        int n_loaded = 1;
    } model;
    whisper_context_params params{};
};
static bool unavailable = false;
static int callback_calls = 0, expected_new = 1;
static int64_t lower = 0, upper = 550;
static void require(bool ok, const char* message) {
    if (!ok) {
        fprintf(stderr, "%s\n", message);
        exit(1);
    }
}
whisper_token whisper_token_beg(whisper_context*) {
    return 100;
}
whisper_token whisper_token_eot(whisper_context*) {
    return 90;
}
whisper_token whisper_token_solm(whisper_context*) {
    return 95;
}
whisper_token whisper_token_prev(whisper_context*) {
    return 96;
}
const char* whisper_token_to_str(whisper_context*, whisper_token id) {
    return id == 7 ? " speech" : "";
}
static std::string to_timestamp(int64_t t) {
    return std::to_string(t);
}
static void whisper_exp_compute_token_level_timestamps(whisper_context&, whisper_state& state, int i, float, float) {
    // Simulate energy-based expansion beyond both ends, and preserve unavailable fields.
    for (auto& tok : state.result_all[i].tokens) {
        tok.t0 = unavailable ? -1 : lower - 20;
        tok.t1 = unavailable ? -1 : upper + 138;
        tok.t_dtw = -1;
    }
}
static int whisper_wrap_segment(whisper_context&, whisper_state& state, int, bool) {
    auto copy = state.result_all.back();
    state.result_all.push_back(copy);
    return 2;
}
static void whisper_exp_compute_token_level_timestamps_dtw(whisper_context*, whisper_state* state, whisper_full_params,
                                                           int first, int count, int, int, int, int) {
    for (int i = first; i < first + count; ++i) {
        for (auto& tok : state->result_all[i].tokens) {
            tok.t_dtw = upper + 77;
        }
    }
}
static void callback(whisper_context*, whisper_state* state, int count, void*) {
    ++callback_calls;
    require(count == expected_new, "callback must report the number of newly available segments");
    for (size_t i = state->result_all.size() - count; i < state->result_all.size(); ++i) {
        const auto& seg = state->result_all[i];
        require(seg.t0 >= lower && seg.t1 <= upper && seg.t0 <= seg.t1, "callback sees out-of-window segment");
        for (const auto& tok : seg.tokens) {
            require(tok.t0 < 0 || (tok.t0 >= seg.t0 && tok.t0 <= seg.t1), "callback sees unbounded token start");
            require(tok.t1 < 0 || (tok.t1 >= seg.t0 && tok.t1 <= seg.t1), "callback sees unbounded token end");
            require(tok.t_dtw < 0 || (tok.t_dtw >= seg.t0 && tok.t_dtw <= seg.t1), "callback sees unbounded DTW time");
            if (unavailable) {
                require(tok.t0 == -1 && tok.t1 == -1, "unavailable times changed");
            }
        }
        require(seg.alts.size() == seg.tokens.size() && seg.alts[0][0].id == 42, "candidate ownership changed");
    }
}
static void emit(whisper_context* ctx, whisper_state* state, whisper_full_params params, int seek, int seek_end) {
    auto& result_all = state->result_all;
    int best_decoder_id = 0;
    std::vector<whisper_token> prompt_past1, prompt, prompt_init;
    // PRODUCTION_EMISSION
}
int main(int argc, char** argv) {
    require(argc == 2, "case required");
    const std::string test = argv[1];
    whisper_context ctx;
    whisper_state state;
    whisper_full_params params{};
    params.no_speech_thold = .6f;
    params.logprob_thold = -1;
    params.token_timestamps = true;
    params.new_segment_callback = callback;
    params.single_segment = test == "tail";
    unavailable = test == "unavailable";
    if (test == "wrapped") {
        params.max_len = 1;
        expected_new = 2;
    }
    if (test == "offset") {
        lower = 125;
        upper = 675;
    }
    if (test == "dtw") {
        ctx.params.dtw_token_timestamps = true;
        // A prior window must not suppress the next window's callback.
        state.result_all.resize(2);
    }
    auto& seq = state.decoders[0].sequence;
    whisper_token_data start{}, word{}, end{};
    start.id = start.tid = 100;
    word.id = 7;
    word.tid = 100;
    end.id = end.tid = 444; // 688 cs predicted from 550 cs of actual audio.
    start.t0 = start.t1 = start.t_dtw = -1;
    word.t0 = word.t1 = word.t_dtw = -1;
    end.t0 = end.t1 = end.t_dtw = -1;
    seq.tokens = {start, word, end};
    seq.result_len = seq.tokens.size();
    seq.alts.resize(3);
    for (auto& alts : seq.alts) {
        whisper_alt_token alt{};
        alt.id = 42;
        alt.p = .75f;
        alts.push_back(alt);
    }
    emit(&ctx, &state, params, lower, upper);
    require(callback_calls == 1, "missing or duplicate callback");
    callback_calls = 0;
    callback(&ctx, &state, expected_new, nullptr); // returned values match callback-time bounds.
    return 0;
}
