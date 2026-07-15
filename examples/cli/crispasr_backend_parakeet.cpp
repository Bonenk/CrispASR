// crispasr_backend_parakeet.cpp — adapter for nvidia/parakeet-tdt-0.6b-v3.
//
// Wraps parakeet_init_from_file + parakeet_transcribe_ex and converts the
// native parakeet_result into a std::vector<crispasr_segment>. One segment
// per transcribe() call, with word-level data attached (parakeet emits word
// timestamps for free via its TDT duration head).

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"
#include "core/asr_context_bias.h"
#include "core/asr_segment_group.h" // issue #257: output-segment grouping

#include "parakeet.h"

#include <cstdio>
#include <cstring>

namespace {

class ParakeetBackend : public CrispasrBackend {
public:
    ParakeetBackend() = default;
    ~ParakeetBackend() override { ParakeetBackend::shutdown(); }

    const char* name() const override { return "parakeet"; }

    uint32_t capabilities() const override {
        // CAP_LANGUAGE_DETECT intentionally NOT declared: the parakeet
        // backend has no native LID code path. Declaring the cap would
        // disable the framework's pre-step LID gate
        // (crispasr_run.cpp:`!has_native_lid`), so users wanting LID
        // get nothing. With the cap absent, `-dl` correctly routes
        // through the whisper-tiny pre-step.
        //
        uint32_t caps = CAP_TIMESTAMPS_NATIVE | CAP_WORD_TIMESTAMPS | CAP_TOKEN_CONFIDENCE | CAP_FLASH_ATTN |
                        CAP_PUNCTUATION_TOGGLE | CAP_TEMPERATURE | CAP_BEAM_SEARCH | CAP_DIARIZE |
                        CAP_PARALLEL_PROCESSORS | CAP_AUTO_DOWNLOAD | CAP_UNBOUNDED_INPUT;
        // CAP_INTERNAL_CHUNKING for non-JA models (2026-06-21): a single
        // full-attention pass is byte-for-byte NeMo-exact and far better
        // than the dispatcher's chunk-30 + overlap-save + LCS-merge, which
        // leaves a duplicated phrase at every 30 s boundary because the LCS
        // dedup is token-id-exact and adjacent chunks transcribe the overlap
        // differently. Verified 100% word match vs nvidia/parakeet-tdt-0.6b-v3
        // (30 s→5 min). With this flag the dispatcher hands us the whole clip;
        // transcribe() runs single-pass up to a memory-safe cap and a
        // silence-split single-pass longform beyond it. The JA model collapses
        // on single-pass (#89), so it keeps the dispatcher's VAD/chunk path.
        // Default: non-JA models advertise internal chunking so the dispatcher
        // hands us the whole clip (transcribe() runs single-pass / longform).
        // Override with CRISPASR_PARAKEET_INTERNAL_CHUNKING=0 to fall back to
        // the dispatcher's chunk-30 + overlap-save + LCS-merge path (e.g. for
        // A/B comparison), or =1 to force it on even for JA.
        bool internal_chunking = !is_ja_model_;
        if (const char* e = getenv("CRISPASR_PARAKEET_INTERNAL_CHUNKING"))
            internal_chunking = atoi(e) != 0;
        if (internal_chunking)
            caps |= CAP_INTERNAL_CHUNKING;
        return caps;
    }

    bool init(const whisper_params& p) override {
        parakeet_context_params cp = parakeet_context_default_params();
        cp.n_threads = p.n_threads;
        cp.use_flash = p.flash_attn;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);

        ctx_ = parakeet_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[parakeet]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        // Issue #89: JA-only models (vocab=3072) collapse past ~12 s on
        // real audio. Auto-chunk at 10 s instead of the global 30 s default.
        // Issue #257: detect JA by vocab content, not size — small-vocab ENGLISH
        // models (parakeet-tdt-1.1b, vocab ~1024) were misclassified as Japanese
        // and forced onto the JA short-chunk path, corrupting long/chunked output.
        is_ja_model_ = parakeet_vocab_is_japanese(ctx_) != 0;
        // CTC decode mode (hybrid TDT+CTC models).
        if (p.parakeet_decoder == "ctc") {
            if (parakeet_has_ctc(ctx_)) {
                parakeet_set_ctc_mode(ctx_, true);
                if (!p.no_prints)
                    fprintf(stderr, "crispasr[parakeet]: using CTC decoder\n");
            } else {
                fprintf(stderr, "crispasr[parakeet]: --parakeet-decoder ctc requested but model has no CTC head\n");
            }
        }
        return true;
    }

    void warmup() override {
        if (!ctx_)
            return;
        // 0.5 s of silence at 16 kHz — touches mel, encoder, and decoder
        // graphs once so subsequent calls hit pre-allocated buffers.
        std::vector<float> silence(8000, 0.0f);
        parakeet_result* r = parakeet_transcribe_ex(ctx_, silence.data(), (int)silence.size(), 0);
        if (r)
            parakeet_result_free(r);
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        // Sticky per-call sampling state. The setter just stores the
        // value on the parakeet_context, so subsequent transcribe calls
        // re-pick it up. We zero it on the first temp==0 call so a user
        // who toggles --temperature back off doesn't keep the previous
        // sampling state from a prior file.
        parakeet_set_temperature(ctx_, params.temperature, params.seed);
        parakeet_set_beam_size(ctx_, params.beam_size > 0 ? params.beam_size : 1);

        // Issue #257: local-attention window (--att-context "L,R") — NeMo
        // rel_pos_local_attn, bounds long-audio encoder VRAM. INT_MIN = unset
        // (keep the model default loaded from the GGUF / env).
        if (params.att_context_left != INT_MIN && params.att_context_right != INT_MIN) {
            parakeet_set_att_context(ctx_, params.att_context_left, params.att_context_right);
        }

        // MAES beam search (env: CRISPASR_PARAKEET_MAES=1, or --decode maes).
        // Requires beam_size > 1. Configurable via env vars.
        {
            const char* maes_env = std::getenv("CRISPASR_PARAKEET_MAES");
            bool use_maes = (maes_env && atoi(maes_env) > 0) || params.parakeet_decoder == "maes";
            if (use_maes && params.beam_size > 1) {
                int num_steps = 2;
                float gamma = 2.3f;
                int beta = 2;
                if (const char* v = std::getenv("CRISPASR_MAES_NUM_STEPS"))
                    num_steps = atoi(v);
                if (const char* v = std::getenv("CRISPASR_MAES_GAMMA"))
                    gamma = (float)atof(v);
                if (const char* v = std::getenv("CRISPASR_MAES_BETA"))
                    beta = atoi(v);
                parakeet_set_maes(ctx_, true, num_steps, gamma, beta);
            }
        }

        // PLAN #98: CTC-WS hotword phrase boost
        if (!params.hotwords.empty()) {
            auto hw = core_context_bias::parse_hotwords(params.hotwords);
            std::vector<const char*> ptrs;
            for (auto& s : hw)
                ptrs.push_back(s.c_str());
            parakeet_set_hotwords(ctx_, ptrs.data(), (int)ptrs.size(), params.hotwords_boost);
        }

        // Issue #89: encoding-path selection.
        //
        // parakeet_transcribe_ex (single-pass) routes the full audio through
        // one bidirectional FastConformer encoder pass. The attention is
        // numerically unstable past ~10-20 s in a way that depends on
        // per-feature z-norm statistics: two codec-quantized copies of the
        // same speech (≈0.3% RMS diff, 0.998 waveform corr) can flip the
        // encoder output std by 10-15 % and drive the TDT decoder into
        // emit-blank-forever past a few seconds. Repro: lenhone's clip in
        // issue #89, comment 4529025103 (60 s file → 20 s of output).
        //
        // parakeet_transcribe_streamed encodes the (globally-z-normed) mel
        // in overlapping 8 s windows and concatenates encoder outputs
        // before a single TDT decode. The attention is local-bounded so
        // codec-level perturbations don't amplify, and the decoder still
        // sees one contiguous encoder sequence (no LSTM cold-start).
        // On audio where single-pass works, streamed produces byte-
        // identical or near-identical text. On audio where single-pass
        // collapses, streamed still covers ~99 % of the clip.
        //
        // We therefore always go through the streamed path. Single-pass
        // is preserved as an opt-in escape hatch for callers who really
        // want the bidirectional-over-everything behaviour (e.g. for
        // bit-exact reproduction of upstream NeMo on test data) — set
        // `CRISPASR_PARAKEET_STREAM_THRESHOLD=999` to bypass streaming
        // for audio shorter than that.
        //
        // Env knobs:
        //   CRISPASR_PARAKEET_STREAM_THRESHOLD : single-pass for audio ≤
        //       this duration (seconds). Default 0 = always streamed.
        //   CRISPASR_PARAKEET_STREAM_CHUNK     : encoder chunk size (s).
        //       Default 0 = let the C library pick per-model: 8 s for the
        //       JA-only model (vocab=3072), 30 s for the multilingual / v3
        //       family (vocab=8192). Manual override here for the rare
        //       case where neither heuristic fits the audio at hand.
        //   CRISPASR_PARAKEET_STREAM_OVERLAP   : encoder overlap (s).
        //       Default 2. Covers FastConformer's receptive field at the
        //       chunk boundary; overlap frames from later chunks are
        //       discarded before decode.
        // ---- Long-audio path selection (model-dependent; all overridable) ----
        //
        // Default per model family, then env gates. JA (vocab<=4096) and DE/EN
        // (v3 / multilingual) behave very differently, so the defaults differ:
        //
        //   non-JA: single full-attention pass is byte-for-byte NeMo-exact
        //     (verified 100% word match vs nvidia/parakeet-tdt-0.6b-v3, 30 s→
        //     5 min). Use it up to a memory-safe cap (full attention is O(T^2):
        //     ~5 min safe on 16 GB, 28 min OOMs); past the cap, silence-split
        //     into <=cap single-pass pieces and concatenate (no boundary dups).
        //   JA: collapses on single-pass (#89), so single-pass is disabled
        //     (threshold 0 → streamed) and CAP_INTERNAL_CHUNKING is NOT set, so
        //     the dispatcher keeps driving it via VAD / 30 s chunking. Behaviour
        //     is unchanged from before this routing change.
        //
        // Env gates (seconds / 0|1), so JA vs DE and per-machine memory can be
        // tuned without a rebuild:
        //   CRISPASR_PARAKEET_STREAM_THRESHOLD : single-pass cap. 0 disables
        //       single-pass entirely (always streamed). Default JA=0, else 300.
        //   CRISPASR_PARAKEET_LONGFORM         : 1 = silence-split single-pass
        //       above the cap; 0 = fall back to the streamed path. Default
        //       follows the model (non-JA=1, JA=0).
        //   CRISPASR_PARAKEET_STREAM_CHUNK / _OVERLAP : streamed encoder window.
        // CLI escape hatches (no env needed): --chunk-seconds N forces the
        // dispatcher's N-second chunk+merge; --vad forces the VAD path.
        // JA: single-pass is NeMo-exact and safe up to ~12 s (issue #89 —
        // past that the encoder collapses on real speech); with the VAD
        // slice cap at 12 s every slice decodes single-pass. Longer inputs
        // (explicit --chunk-seconds 0) still stream.
        // Issue #257: honor CLI --chunk-seconds / --chunk-overlap. The dispatcher
        // stops slicing us when --chunk-seconds is explicit (CAP_INTERNAL_CHUNKING,
        // non-JA), so we receive the whole audio here and drive the library's
        // internal encoder-frame streaming (one coherent decode over concatenated
        // encoder output) instead of the dispatcher's per-slice transcribe+merge,
        // which corrupts this full-attention encoder.
        //
        // Two things are DECOUPLED here (they were conflated before, which both
        // truncated text and produced one giant segment — issue #257 reports):
        //   1. ENCODER window — kept at the model's quality default (30 s for
        //      non-JA), NOT the requested chunk length. Small encoder windows
        //      shift this full-attention FastConformer's per-feature statistics
        //      and make the TDT decoder emit a sparse/truncated token path (see
        //      parakeet_transcribe_streamed): --chunk-seconds 7 was dropping the
        //      tail. The 30 s window keeps text complete AND bounds encoder VRAM
        //      (no O(T^2) single-pass blow-up). Power users can still force the
        //      encoder window via CRISPASR_PARAKEET_STREAM_CHUNK.
        //   2. OUTPUT segmentation — the coherent decode's words/tokens are then
        //      grouped into ~chunk_seconds segments (core_segment::group_by_window)
        //      so -ojf/-osrt emit per-segment offsets like whisper/cohere/granite,
        //      instead of one blob. No overlap/merge → no boundary duplicates.
        // JA keeps the dispatcher's VAD/12 s slicing (no CAP_INTERNAL_CHUNKING).
        if (!is_ja_model_ && params.chunk_seconds_explicit && params.chunk_seconds > 0) {
            const int seg_seconds = std::max(2, params.chunk_seconds); // output segment length
            int enc_window = 0;                                        // 0 → library quality default (30 s non-JA)
            if (const char* e = getenv("CRISPASR_PARAKEET_STREAM_CHUNK"))
                enc_window = std::max(2, atoi(e));
            const int ov = std::max(0, (int)(params.chunk_overlap_seconds + 0.5f));
            parakeet_result* rc = parakeet_transcribe_streamed(ctx_, samples, n_samples, t_offset_cs, enc_window, ov);
            if (!rc)
                return out;
            out = split_result_into_segments(rc, t_offset_cs, seg_seconds);
            parakeet_result_free(rc);
            return out;
        }

        int stream_threshold_s = is_ja_model_ ? 12 : 300;
        bool longform_enabled = !is_ja_model_;
        int stream_chunk_s = 0; // 0 = let the C library pick per-model
        int stream_overlap_s = 2;
        if (const char* e = getenv("CRISPASR_PARAKEET_STREAM_THRESHOLD"))
            stream_threshold_s = std::max(0, atoi(e));
        if (const char* e = getenv("CRISPASR_PARAKEET_LONGFORM"))
            longform_enabled = atoi(e) != 0;
        if (const char* e = getenv("CRISPASR_PARAKEET_STREAM_CHUNK"))
            stream_chunk_s = std::max(2, atoi(e));
        if (const char* e = getenv("CRISPASR_PARAKEET_STREAM_OVERLAP"))
            stream_overlap_s = std::max(0, atoi(e));

        const int SR = 16000;
        if (longform_enabled && stream_threshold_s > 0 && n_samples > stream_threshold_s * SR) {
            return transcribe_longform(samples, n_samples, t_offset_cs, stream_threshold_s * SR);
        }

        parakeet_result* r;
        const bool use_single_pass = stream_threshold_s > 0 && n_samples <= stream_threshold_s * SR;
        if (use_single_pass) {
            r = parakeet_transcribe_ex(ctx_, samples, n_samples, t_offset_cs);
        } else {
            r = parakeet_transcribe_streamed(ctx_, samples, n_samples, t_offset_cs, stream_chunk_s, stream_overlap_s);
        }
        if (!r)
            return out;

        out.push_back(result_to_segment(r, t_offset_cs));
        parakeet_result_free(r);
        return out;
    }

    // Convert a parakeet_result into one crispasr_segment. Does NOT free r.
    static crispasr_segment result_to_segment(const parakeet_result* r, int64_t fallback_t0_cs) {
        crispasr_segment seg;
        seg.t0 = fallback_t0_cs;
        seg.t1 = fallback_t0_cs;
        seg.text = r->text ? r->text : "";

        seg.words.reserve(r->n_words);
        for (int i = 0; i < r->n_words; i++) {
            const auto& w = r->words[i];
            crispasr_word cw;
            cw.text = w.text;
            cw.t0 = w.t0;
            cw.t1 = w.t1;
            seg.words.push_back(std::move(cw));
        }

        seg.tokens.reserve(r->n_tokens);
        for (int i = 0; i < r->n_tokens; i++) {
            const auto& t = r->tokens[i];
            crispasr_token ct;
            ct.text = t.text;
            ct.id = t.id;
            ct.t0 = t.t0;
            ct.t1 = t.t1;
            ct.confidence = t.p;
            seg.tokens.push_back(std::move(ct));
        }

        if (!seg.words.empty()) {
            seg.t0 = seg.words.front().t0;
            seg.t1 = seg.words.back().t1;
        } else if (!seg.tokens.empty()) {
            seg.t0 = seg.tokens.front().t0;
            seg.t1 = seg.tokens.back().t1;
        }
        return seg;
    }

    // Issue #257: split ONE coherent parakeet decode into ~seg_seconds output
    // segments, snapped to word boundaries. The decode is unchanged (full
    // context → complete, uncorrupted text); we only group the resulting words
    // and tokens into multiple crispasr_segments so -ojf/-osrt emit per-segment
    // offsets like the whisper/cohere/granite backends. Contiguous (no overlap)
    // → the whole transcript is covered exactly once, no boundary duplicates.
    static std::vector<crispasr_segment> split_result_into_segments(const parakeet_result* r, int64_t fallback_t0_cs,
                                                                    int seg_seconds) {
        std::vector<crispasr_segment> out;
        if (!r)
            return out;
        // No word timings (e.g. empty / all-blank decode) → one segment.
        if (r->n_words <= 0) {
            crispasr_segment seg = result_to_segment(r, fallback_t0_cs);
            if (!seg.text.empty() || !seg.tokens.empty())
                out.push_back(std::move(seg));
            return out;
        }

        std::vector<int64_t> word_t0;
        word_t0.reserve(r->n_words);
        for (int i = 0; i < r->n_words; i++)
            word_t0.push_back(r->words[i].t0);
        const std::vector<int> starts = core_segment::group_by_window(word_t0, (int64_t)seg_seconds * 100);

        int ti = 0; // token cursor (r->tokens is time-ordered)
        for (size_t si = 0; si < starts.size(); si++) {
            const int w_begin = starts[si];
            const int w_end = (si + 1 < starts.size()) ? starts[si + 1] : r->n_words;

            crispasr_segment seg;
            std::string text;
            for (int wi = w_begin; wi < w_end; wi++) {
                const auto& w = r->words[wi];
                // parakeet word.text has the leading space dropped and
                // punctuation attached; re-join with a single space.
                if (!text.empty())
                    text += ' ';
                text += w.text;
                crispasr_word cw;
                cw.text = w.text;
                cw.t0 = w.t0;
                cw.t1 = w.t1;
                seg.words.push_back(std::move(cw));
            }
            seg.text = std::move(text);
            seg.t0 = seg.words.front().t0;
            seg.t1 = seg.words.back().t1;

            // Assign tokens up to the next segment's first word start (the last
            // segment takes the remainder) so the token stream stays complete.
            const int64_t next_start = (w_end < r->n_words) ? r->words[w_end].t0 : INT64_MAX;
            for (; ti < r->n_tokens && r->tokens[ti].t0 < next_start; ti++) {
                const auto& t = r->tokens[ti];
                crispasr_token ct;
                ct.text = t.text;
                ct.id = t.id;
                ct.t0 = t.t0;
                ct.t1 = t.t1;
                ct.confidence = t.p;
                seg.tokens.push_back(std::move(ct));
            }
            out.push_back(std::move(seg));
        }
        // Safety: any trailing tokens (t0 beyond the last word) → last segment.
        for (; ti < r->n_tokens && !out.empty(); ti++) {
            const auto& t = r->tokens[ti];
            crispasr_token ct;
            ct.text = t.text;
            ct.id = t.id;
            ct.t0 = t.t0;
            ct.t1 = t.t1;
            ct.confidence = t.p;
            out.back().tokens.push_back(std::move(ct));
        }
        return out;
    }

    // Find a silence cut near `target` within [target - window, target] by
    // locating the lowest-RMS 100 ms frame. Returns the cut sample index.
    // Keeps chunk boundaries off mid-word so single-pass pieces concatenate
    // cleanly with no overlap/merge needed (PLAN #80b energy-minimum trick).
    static int find_silence_cut(const float* s, int n, int target, int window, int sr) {
        const int lo = std::max(1, target - window);
        const int hi = std::min(n - 1, target);
        if (hi <= lo)
            return std::min(std::max(target, 1), n);
        const int win = std::max(1, sr / 10); // 100 ms
        double best = 1e30;
        int best_pos = target;
        for (int c = lo; c <= hi; c += win / 2) {
            const int a = std::max(0, c - win / 2);
            const int b = std::min(n, c + win / 2);
            double e = 0.0;
            for (int i = a; i < b; i++)
                e += (double)s[i] * (double)s[i];
            e /= std::max(1, b - a);
            if (e < best) {
                best = e;
                best_pos = c;
            }
        }
        return best_pos;
    }

    // Long-audio path for non-JA models: split the clip at silence into
    // pieces no longer than `cap_samples`, transcribe each with the
    // NeMo-exact single full-attention pass, and concatenate. Each piece is
    // independent (cuts in silence), so there is no overlap-merge step and
    // hence no boundary duplicates — and memory stays bounded by the cap.
    std::vector<crispasr_segment> transcribe_longform(const float* samples, int n_samples, int64_t t_offset_cs,
                                                      int cap_samples) {
        std::vector<crispasr_segment> out;
        const int SR = 16000;
        const int search = 5 * SR; // search last 5 s of each window for silence
        const int ctx = 2 * SR;    // ±2 s acoustic context around each piece
        int pos = 0;
        while (pos < n_samples) {
            int end;
            if (n_samples - pos <= cap_samples) {
                end = n_samples;
            } else {
                end = find_silence_cut(samples, n_samples, pos + cap_samples, search, SR);
                if (end <= pos)
                    end = std::min(n_samples, pos + cap_samples); // safety: never stall
            }
            // Transcribe with context on both sides so boundary words aren't
            // dropped for lack of encoder context, then commit only the
            // [pos,end) core by word timestamp. Adjacent pieces share the
            // single cut `end` (A keeps t0<end, B keeps t0>=end) → no gap,
            // no overlap, no duplicates.
            const int ext_s = std::max(0, pos - ctx);
            const int ext_e = std::min(n_samples, end + ctx);
            const int64_t ext_t0 = t_offset_cs + (int64_t)((double)ext_s / SR * 100.0);
            const int64_t left_cs = (pos == 0) ? INT64_MIN : t_offset_cs + (int64_t)((double)pos / SR * 100.0);
            const int64_t right_cs = (end == n_samples) ? INT64_MAX : t_offset_cs + (int64_t)((double)end / SR * 100.0);

            parakeet_result* r = parakeet_transcribe_ex(ctx_, samples + ext_s, ext_e - ext_s, ext_t0);
            if (r) {
                crispasr_segment full = result_to_segment(r, ext_t0);
                parakeet_result_free(r);

                crispasr_segment seg;
                seg.t0 = left_cs == INT64_MIN ? full.t0 : left_cs;
                seg.t1 = seg.t0;
                // parakeet word.text has the leading space dropped and
                // punctuation attached, so re-join committed words with a
                // single space (non-JA = space-delimited languages only).
                std::string text;
                for (auto& w : full.words) {
                    if (w.t0 >= left_cs && w.t0 < right_cs) {
                        if (!text.empty())
                            text += ' ';
                        text += w.text;
                        seg.words.push_back(std::move(w));
                    }
                }
                for (auto& tk : full.tokens) {
                    if (tk.t0 >= left_cs && tk.t0 < right_cs)
                        seg.tokens.push_back(std::move(tk));
                }
                seg.text = std::move(text);
                if (!seg.words.empty()) {
                    seg.t0 = seg.words.front().t0;
                    seg.t1 = seg.words.back().t1;
                }
                if (!seg.text.empty() || !seg.words.empty())
                    out.push_back(std::move(seg));
            }
            pos = end;
        }
        return out;
    }

    bool prefers_vad() const override {
        // Issue #89: parakeet-ja's encoder degenerates on arbitrary chunks
        // (repetition loops). VAD gives silence-bounded segments matching
        // the ~10-15 s utterances the model was trained on.
        return is_ja_model_;
    }

    int vad_slice_cap_seconds() const override {
        // Issue #89: on continuous speech (podcasts) VAD merges slices far
        // past the JA encoder's ~12 s safe single-pass window and the
        // decode goes sparse (56 % content recall on the reporter's clip).
        // Capping slices at 12 s + single-pass per slice measured best
        // (73-81 % vs whisper-large-v3 reference, NeMo's own long-form
        // paths score 15-46 % on the same audio). Non-JA models are exact
        // in single-pass and don't need a cap.
        int cap = is_ja_model_ ? 12 : 0;
        if (const char* e = getenv("CRISPASR_PARAKEET_VAD_SLICE_CAP"))
            cap = std::max(0, atoi(e));
        return cap;
    }

    void shutdown() override {
        if (ctx_) {
            parakeet_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    parakeet_context* ctx_ = nullptr;
    bool is_ja_model_ = false;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_parakeet_backend() {
    return std::unique_ptr<CrispasrBackend>(new ParakeetBackend());
}
