// Hosted-only real-model ABI smoke; no transcript or numerical-parity claim.
#include "crispasr.h"
#include "crispasr_session.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

static void check(bool pass, const char* message) {
    if (!pass) {
        throw std::runtime_error(message);
    }
}
static std::string hex(const char* text) {
    static const char digits[] = "0123456789abcdef";
    std::string out;
    for (const unsigned char* p = reinterpret_cast<const unsigned char*>(text); p && *p; ++p) {
        out += digits[*p >> 4];
        out += digits[*p & 15];
    }
    return out;
}
int main(int argc, char** argv) {
    check(argc == 4, "expected model, WAV, output TSV");
    float* raw = nullptr;
    int count = 0, rate = 0;
    check(crispasr_audio_load(argv[2], &raw, &count, &rate) == 0 && raw && count > 0 && rate == 16000,
          "audio fixture load failed");
    std::unique_ptr<float, decltype(&crispasr_audio_free)> pcm(raw, crispasr_audio_free);
    std::ofstream out(argv[3]);
    check(bool(out), "cannot open receipt");
    out << std::unitbuf; // Preserve the last diagnostic row if a later check fails.
    for (bool dtw : {false, true}) {
        std::unique_ptr<whisper_context_params, decltype(&whisper_free_context_params)> cp(
            whisper_context_default_params_by_ref(), whisper_free_context_params);
        check(bool(cp), "context params allocation failed");
        cp->use_gpu = false;
        crispasr_ctx_params_set_dtw(cp.get(), dtw, CRISPASR_AHEADS_TINY_EN, 0);
        check(!dtw || !cp->flash_attn, "DTW/flash policy violated");
        std::unique_ptr<whisper_context, decltype(&whisper_free)> ctx(
            whisper_init_from_file_with_params_by_ref(argv[1], cp.get()), whisper_free);
        check(bool(ctx), "model initialization failed");
        for (int processors : {1, 2, 3}) {
            for (bool bounded : {false, true}) {
                std::unique_ptr<whisper_full_params, decltype(&whisper_free_params)> p(
                    whisper_full_default_params_by_ref(CRISPASR_SAMPLING_GREEDY), whisper_free_params);
                check(bool(p), "decode params allocation failed");
                p->language = "en";
                p->n_threads = 2;
                p->no_context = true;
                p->token_timestamps = true;
                p->temperature = 0;
                p->greedy.best_of = 1;
                p->print_realtime = p->print_progress = p->print_timestamps = p->print_special = false;
                p->offset_ms = bounded ? 1000 : 0;
                p->duration_ms = bounded ? 6000 : 0;
                crispasr_params_set_alt_n(p.get(), 2);
                const int rc = crispasr_transcribe_parallel_by_ref(ctx.get(), p.get(), pcm.get(), count, processors);
                check(rc == 0 && crispasr_get_progress() == -1, "pointer decode/progress failed");
                const int segments = whisper_full_n_segments(ctx.get());
                check(segments > 0 && segments < 10000, "invalid/empty segments");
                const int64_t end = bounded ? std::min<int64_t>(count / 160, 700) : count / 160;
                int aligned = 0, dtw_aligned = 0, candidates = 0;
                out << "run\t" << dtw << '\t' << processors << '\t' << bounded << '\t' << segments << '\n';
                for (int i = 0; i < segments; ++i) {
                    const auto start = whisper_full_get_segment_t0(ctx.get(), i);
                    const auto finish = whisper_full_get_segment_t1(ctx.get(), i);
                    // Exact interval geometry is gated separately by the tiny
                    // runtime suite. Model timestamps have a one-second margin.
                    out << "segment\t" << i << '\t' << start << '\t' << finish << '\t'
                        << hex(whisper_full_get_segment_text(ctx.get(), i)) << '\n';
                    check(start >= 0 && finish >= start && finish <= end + 100, "segment outside model time margin");
                    const int tokens = whisper_full_n_tokens(ctx.get(), i);
                    check(tokens >= 0 && tokens < 100000, "invalid token count");
                    for (int j = 0; j < tokens; ++j) {
                        const auto t = whisper_full_get_token_data(ctx.get(), i, j);
                        check(crispasr_token_t0(ctx.get(), i, j) == t.t0 &&
                                  crispasr_token_t1(ctx.get(), i, j) == t.t1 &&
                                  crispasr_token_dtw_t(ctx.get(), i, j) == t.t_dtw,
                              "scalar/struct token timing differs");
                        check(std::isfinite(t.p), "nonfinite token probability");
                        out << "token\t" << i << '\t' << j << '\t' << t.id << '\t' << t.t0 << '\t' << t.t1 << '\t'
                            << t.t_dtw << '\t' << t.p << '\t' << hex(whisper_full_get_token_text(ctx.get(), i, j))
                            << '\n';
                        if (t.id < whisper_token_eot(ctx.get()) && t.t0 >= 0 && t.t1 >= t.t0) {
                            check(t.t0 >= start - 100 && t.t1 <= finish + 100 && t.t1 <= end + 100,
                                  "content token/chunk timestamp outside model time margin");
                            ++aligned;
                        }
                        if (t.id < whisper_token_eot(ctx.get()) && t.t_dtw >= 0) {
                            check(t.t_dtw >= start - 100 && t.t_dtw <= finish + 100 && t.t_dtw <= end + 100,
                                  "DTW/chunk timestamp outside model time margin");
                            ++dtw_aligned;
                        }
                        const int n_alts = crispasr_token_n_alts(ctx.get(), i, j);
                        check(n_alts >= 0 && n_alts <= 2, "unexpected alternative count");
                        for (int k = 0; k < n_alts; ++k) {
                            char text[4096];
                            check(crispasr_token_alt_text(ctx.get(), i, j, k, text, sizeof(text)) >= 0,
                                  "alternative text read failed");
                            const float probability = crispasr_token_alt_p(ctx.get(), i, j, k);
                            check(std::isfinite(probability), "nonfinite alternative probability");
                            out << "alt\t" << i << '\t' << j << '\t' << k << '\t'
                                << crispasr_token_alt_id(ctx.get(), i, j, k) << '\t' << probability << '\t' << hex(text)
                                << '\n';
                            ++candidates;
                        }
                    }
                }
                check(aligned > 0 && candidates > 0 && (!dtw || dtw_aligned > 0), "requested metadata missing");
                out.flush();
                check(bool(out), "receipt write failed");
            }
        }
    }
}
