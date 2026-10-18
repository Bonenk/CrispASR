// Immutable CrispASR tokenizer/aligner from deaf9c010acce1410f470d6df421d0f77ff16be2.
// Our own source; extracted without encoder/model dependencies.
static bool tokenise_word_reference(const std::string& word, const canary_ctc_vocab& vocab, std::vector<int>& out) {
    // The vocab tokens look like ["▁the", "▁of", "ing", "ed", "<unk>", ...]
    // For a word "Americans" we want something like ["▁Americ", "ans"] or
    // similar — start with the ▁-prefixed form and then continue with
    // continuations.
    const std::string boundary = "\xE2\x96\x81"; // U+2581 ▁
    std::string remaining = boundary + word;

    // Lowercase doesn't really matter for SentencePiece (it preserves case),
    // but we keep the original.
    while (!remaining.empty()) {
        // Find the longest prefix of `remaining` that's in the vocab
        int best_len = 0;
        int best_id = -1;
        // Limit to ~32 byte prefixes (SentencePiece pieces are typically short)
        const int max_prefix = std::min<int>(remaining.size(), 32);
        for (int len = max_prefix; len >= 1; len--) {
            std::string prefix = remaining.substr(0, len);
            auto it = vocab.token_to_id.find(prefix);
            if (it != vocab.token_to_id.end()) {
                best_len = len;
                best_id = it->second;
                break;
            }
        }
        if (best_id < 0) {
            // No match — fall back to a single byte. If even single bytes
            // aren't in the vocab, give up on this word.
            // (This is rare for a SentencePiece BPE on Latin/Cyrillic text.)
            std::string one(1, remaining[0]);
            auto it = vocab.token_to_id.find(one);
            if (it == vocab.token_to_id.end()) {
                // Try <unk>
                it = vocab.token_to_id.find("<unk>");
                if (it == vocab.token_to_id.end())
                    return false;
            }
            out.push_back(it->second);
            remaining = remaining.substr(1);
        } else {
            out.push_back(best_id);
            remaining = remaining.substr(best_len);
        }
    }
    return !out.empty();
}

static int canary_ctc_align_words_reference(struct canary_ctc_context* ctx, const float* logits, int T_enc, int V,
                                      const char** words, int n_words, struct canary_ctc_word* out_words) {
    if (!ctx || !logits || !words || !out_words || T_enc <= 0 || V <= 0 || n_words <= 0)
        return -1;
    const int blank_id = (int)ctx->model.hparams.blank_id;
    const int frame_dur_cs = (int)ctx->model.hparams.frame_dur_cs;
    if (blank_id < 0 || blank_id >= V)
        return -1;

    // ----- 1. Tokenise each word and build per-word token ranges in `chars` -----
    struct word_range {
        int cs, ce;
    };
    std::vector<int> chars;
    std::vector<word_range> wranges(n_words);

    for (int wi = 0; wi < n_words; wi++) {
        int cs = (int)chars.size();
        std::vector<int> tok_ids;
        bool ok = words[wi] ? tokenise_word_reference(words[wi], ctx->vocab, tok_ids) : false;
        if (!ok || tok_ids.empty()) {
            wranges[wi] = {-1, -1};
            // Still copy the word text into the output for the caller
        } else {
            for (int t : tok_ids)
                chars.push_back(t);
            wranges[wi] = {cs, (int)chars.size() - 1};
        }
        // Pre-fill output text
        size_t L = std::min(strlen(words[wi] ? words[wi] : ""), sizeof(out_words[wi].text) - 1);
        memcpy(out_words[wi].text, words[wi] ? words[wi] : "", L);
        out_words[wi].text[L] = '\0';
        out_words[wi].t0 = 0;
        out_words[wi].t1 = 0;
    }

    int N = (int)chars.size();
    if (N == 0)
        return -2;
    for (int token : chars)
        if (token < 0 || token >= V)
            return -1;

    // ----- 2. CTC-expanded label sequence with blanks between every label -----
    int S = 2 * N + 1;
    std::vector<int> seq(S);
    for (int j = 0; j < S; j++)
        seq[j] = (j % 2 == 0) ? blank_id : chars[j / 2];

    // ----- 3. Log-softmax over the CTC logits per frame -----
    std::vector<float> lp((size_t)T_enc * V);
    for (int t = 0; t < T_enc; t++) {
        const float* src = logits + (size_t)t * V;
        float* dst = lp.data() + (size_t)t * V;
        float mx = *std::max_element(src, src + V);
        float s = 0.f;
        for (int i = 0; i < V; i++) {
            dst[i] = src[i] - mx;
            s += expf(dst[i]);
        }
        float ls = logf(std::max(s, 1e-30f));
        for (int i = 0; i < V; i++)
            dst[i] -= ls;
    }

    // ----- 4. Viterbi DP -----
    const float NEG_INF = -1e30f;
    std::vector<float> alpha(S, NEG_INF), alpha_next(S);
    std::vector<std::vector<int8_t>> back(T_enc, std::vector<int8_t>(S, 0));

    // t = 0
    {
        const float* lp0 = lp.data();
        alpha[0] = lp0[seq[0]];
        if (S > 1)
            alpha[1] = lp0[seq[1]];
    }

    for (int t = 1; t < T_enc; t++) {
        const float* lpt = lp.data() + (size_t)t * V;
        std::fill(alpha_next.begin(), alpha_next.end(), NEG_INF);

        for (int j = 0; j < S; j++) {
            int tok = seq[j];
            float best = alpha[j];
            int8_t bsrc = 0;
            if (j >= 1 && alpha[j - 1] > best) {
                best = alpha[j - 1];
                bsrc = 1;
            }
            if (j >= 2 && tok != blank_id && seq[j] != seq[j - 2] && alpha[j - 2] > best) {
                best = alpha[j - 2];
                bsrc = 2;
            }
            if (best > NEG_INF) {
                alpha_next[j] = best + lpt[tok];
                back[t][j] = bsrc;
            }
        }
        alpha.swap(alpha_next);
    }

    // ----- 5. Traceback -----
    std::vector<int> path(T_enc);
    int j_cur = (alpha[S - 1] >= alpha[S - 2]) ? S - 1 : S - 2;
    // A transcript that cannot reach either terminal state has no complete
    // CTC path. Do not turn default backpointers into successful zero times.
    if (!std::isfinite(alpha[j_cur]) || alpha[j_cur] <= NEG_INF)
        return -1;
    for (int t = T_enc - 1; t >= 0; t--) {
        path[t] = j_cur;
        if (t > 0) {
            switch (back[t][j_cur]) {
            case 0:
                break;
            case 1:
                j_cur--;
                break;
            case 2:
                j_cur -= 2;
                break;
            }
        }
    }

    // ----- 6. Map path → per-word t0/t1 -----
    for (int wi = 0; wi < n_words; wi++) {
        if (wranges[wi].cs < 0)
            continue;
        int es0 = 2 * wranges[wi].cs + 1;
        int es1 = 2 * wranges[wi].ce + 1;
        int t0_frame = -1, t1_frame = -1;
        for (int t = 0; t < T_enc; t++) {
            int j = path[t];
            if (j >= es0 && j <= es1) {
                if (t0_frame < 0)
                    t0_frame = t;
                t1_frame = t;
            }
        }
        if (t0_frame >= 0) {
            out_words[wi].t0 = (int64_t)t0_frame * frame_dur_cs;
            out_words[wi].t1 = (int64_t)(t1_frame + 1) * frame_dur_cs;
        }
    }
    return 0;
}
