// test-issue-356-gap-fill-overlap.cpp — issue #356 regression guard.
//
// The issue #89 gap-fill second pass recovers speech the parakeet-ja encoder
// blanks inside a slice, but it appended the recovered segments and sorted by
// t0 only. The first pass emits ONE segment whose sparse words straddle the
// hole, so its t0..t1 encloses every recovery — sorting cannot fix that, and
// the SRT/VTT output carried cues jumping backwards by up to the hole length
// (10.5 s on the downstream reporter's file; the 32.66→43.30 s excerpt below
// mirrors it). crispasr_gap_fill_resolve_overlaps now splits the covering
// segment at each recovery so the list comes out monotone with no text lost.
//
// Pure CPU, no model load: the backend is a scripted fake that "recovers"
// preset words for whatever window it is asked to transcribe.

#include "crispasr_gap_fill.h"
#include "whisper_params.h"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

namespace {

constexpr int kSampleRate = 16000;

crispasr_word make_word(const char* text, int64_t t0_cs, int64_t t1_cs) {
    crispasr_word w;
    w.text = text;
    w.t0 = t0_cs;
    w.t1 = t1_cs;
    return w;
}

crispasr_segment make_segment(std::vector<crispasr_word> words) {
    crispasr_segment seg;
    seg.words = std::move(words);
    seg.text = crispasr_rebuild_text_from_words(seg.words);
    seg.t0 = seg.words.front().t0;
    seg.t1 = seg.words.back().t1;
    return seg;
}

// Backend whose "audio" contains a fixed script of absolute-timed words: a
// transcribe() call returns every scripted word that falls inside the
// requested window. This is exactly the contract gap-fill relies on when it
// re-transcribes a hole in isolation.
class ScriptedBackend : public CrispasrBackend {
  public:
    explicit ScriptedBackend(std::vector<crispasr_word> script) : script_(std::move(script)) {}

    const char* name() const override { return "scripted"; }
    uint32_t capabilities() const override { return 0; }
    bool init(const whisper_params&) override { return true; }
    void shutdown() override {}

    std::vector<crispasr_segment> transcribe(const float*, int n_samples, int64_t t_offset_cs,
                                             const whisper_params&) override {
        const int64_t win0 = t_offset_cs;
        const int64_t win1 = t_offset_cs + (int64_t)n_samples * 100 / kSampleRate;
        crispasr_segment seg;
        for (const auto& w : script_) {
            if (w.t0 >= win0 && w.t1 <= win1) {
                seg.words.push_back(w);
                crispasr_token tk;
                tk.text = w.text;
                tk.t0 = w.t0;
                tk.t1 = w.t1;
                seg.tokens.push_back(tk);
            }
        }
        if (seg.words.empty())
            return {};
        seg.t0 = seg.words.front().t0;
        seg.t1 = seg.words.back().t1;
        seg.text = crispasr_rebuild_text_from_words(seg.words);
        return {seg};
    }

  private:
    std::vector<crispasr_word> script_;
};

void require_monotone(const std::vector<crispasr_segment>& segs) {
    for (size_t i = 1; i < segs.size(); i++) {
        INFO("segment " << i << " [" << segs[i].t0 << ".." << segs[i].t1 << "] vs previous [" << segs[i - 1].t0 << ".."
                        << segs[i - 1].t1 << "]");
        REQUIRE(segs[i].t0 >= segs[i - 1].t1);
    }
}

std::string joined_text(const std::vector<crispasr_segment>& segs) {
    std::string all;
    for (const auto& s : segs)
        all += s.text;
    return all;
}

} // namespace

TEST_CASE("issue #356: recovery inside a straddling segment splits it, output monotone",
          "[unit][gap-fill][issue-356]") {
    // Mirrors the reporter's 32.66→43.30 s slice: the first pass emitted one
    // segment whose words cover the edges but skip 33.1→39.4 s; gap-fill
    // recovers the middle from the scripted backend.
    const crispasr_audio_slice sl = {3250 * kSampleRate / 100, 4330 * kSampleRate / 100, 3250, 4330};
    std::vector<float> samples((size_t)sl.end, 0.0f);

    std::vector<crispasr_segment> segs;
    segs.push_back(make_segment({
        make_word("まずね", 3266, 3310),
        make_word("暮らすのは好きなんです。", 3946, 4330),
    }));

    ScriptedBackend be({
        make_word("日本にもそんないない", 3318, 3454),
        make_word("です。", 3460, 3518),
        make_word("えっと。", 3686, 3782),
    });
    whisper_params params;

    crispasr_gap_fill_slice(be, params, samples.data(), sl.end, kSampleRate, sl, segs);

    // All three parts present, in reading order, with no overlap: the
    // covering segment must have been split around the recovery instead of
    // being left spanning it.
    require_monotone(segs);
    REQUIRE(segs.size() == 3);
    REQUIRE(segs[0].text == "まずね");
    REQUIRE(segs[1].text == "日本にもそんないないです。えっと。");
    REQUIRE(segs[2].text == "暮らすのは好きなんです。");
    REQUIRE(segs[0].t1 <= segs[1].t0);
    REQUIRE(segs[1].t1 <= segs[2].t0);

    // The fill's tokens ride along with the kept words: token-level outputs
    // read seg.tokens, and a recovery with an empty list vanishes from them.
    REQUIRE(segs[1].tokens.size() == 3);
    REQUIRE(segs[1].tokens.front().text == "日本にもそんないない");
    REQUIRE(segs[1].tokens.back().text == "えっと。");
}

TEST_CASE("issue #356: segment enclosing two recoveries is split at each", "[unit][gap-fill][issue-356]") {
    std::vector<crispasr_segment> segs;
    segs.push_back(make_segment({
        make_word("head", 100, 150),
        make_word("mid", 700, 780),
        make_word("tail", 1400, 1500),
    }));
    segs.push_back(make_segment({make_word("first-recovery", 300, 500)}));
    segs.push_back(make_segment({make_word("second-recovery", 900, 1200)}));

    crispasr_gap_fill_resolve_overlaps(segs);

    require_monotone(segs);
    REQUIRE(segs.size() == 5);
    REQUIRE(segs[0].text == "head");
    REQUIRE(segs[1].text == "first-recovery");
    REQUIRE(segs[2].text == "mid");
    REQUIRE(segs[3].text == "second-recovery");
    REQUIRE(segs[4].text == "tail");
}

TEST_CASE("issue #356: recovery before the first word moves the covering segment after it",
          "[unit][gap-fill][issue-356]") {
    // The covering segment's span starts at/before the recovery but all its
    // words sit past the recovery's end — nothing to keep on the left, so the
    // segment itself becomes the tail.
    std::vector<crispasr_segment> segs;
    auto covering = make_segment({make_word("late-words", 800, 1000)});
    covering.t0 = 200; // span inflated to the left (e.g. by an earlier aligner pass)
    segs.push_back(covering);
    segs.push_back(make_segment({make_word("recovery", 250, 600)}));

    crispasr_gap_fill_resolve_overlaps(segs);

    require_monotone(segs);
    REQUIRE(segs.size() == 2);
    REQUIRE(segs[0].text == "recovery");
    REQUIRE(segs[1].text == "late-words");
    REQUIRE(segs[1].t0 == 800);
}

TEST_CASE("issue #356: word-less covering segment is clamped at the recovery", "[unit][gap-fill][issue-356]") {
    std::vector<crispasr_segment> segs;
    crispasr_segment plain;
    plain.text = "segment-level only";
    plain.t0 = 100;
    plain.t1 = 900;
    segs.push_back(plain);
    segs.push_back(make_segment({make_word("recovery", 400, 700)}));

    crispasr_gap_fill_resolve_overlaps(segs);

    require_monotone(segs);
    REQUIRE(segs.size() == 2);
    REQUIRE(segs[0].text == "segment-level only");
    REQUIRE(segs[0].t1 == 400);
}

TEST_CASE("issue #356: already-monotone segments come through unchanged", "[unit][gap-fill][issue-356]") {
    std::vector<crispasr_segment> segs;
    segs.push_back(make_segment({make_word("one", 100, 200)}));
    segs.push_back(make_segment({make_word("two", 200, 300)}));
    segs.push_back(make_segment({make_word("three", 500, 600)}));
    const auto before = joined_text(segs);

    crispasr_gap_fill_resolve_overlaps(segs);

    require_monotone(segs);
    REQUIRE(segs.size() == 3);
    REQUIRE(joined_text(segs) == before);
    REQUIRE(segs[0].t0 == 100);
    REQUIRE(segs[2].t1 == 600);
}

TEST_CASE("issue #356: no text is lost when splitting", "[unit][gap-fill][issue-356]") {
    std::vector<crispasr_segment> segs;
    segs.push_back(make_segment({
        make_word("alpha", 0, 100),
        make_word("omega", 2000, 2100),
    }));
    segs.push_back(make_segment({make_word("beta", 500, 900)}));
    const auto before = std::string("alpha") + "beta" + "omega"; // reading order

    crispasr_gap_fill_resolve_overlaps(segs);

    require_monotone(segs);
    REQUIRE(joined_text(segs) == before);
}
