// Immutable own native word grouping from 08aea3537ad075ffa17e66a6f601c42519213aea.
static bool parakeet_group_words_reference(parakeet_result* r, int frame_dur_cs) {
    // ----- Group sub-word tokens into words -----
    //
    // Latin SentencePiece convention: a token starting with U+2581 (▁ → ' ')
    // begins a new word. Punctuation tokens attach to the previous word.
    //
    // Japanese parakeet (and other no-space tokenizers) emit no leading-space
    // markers because written Japanese has no inter-word spaces. In that
    // mode every non-punctuation token is its own "word" — sufficient
    // granularity for word-level SRT (issue #37).
    std::vector<parakeet_word_data> words;
    words.reserve(r->n_tokens);

    // Detect tokenizer style: count tokens that look like real
    // word-starts (leading space + at least one more char). A
    // standalone " " token (BOS-ish) doesn't count.
    int n_space_word_starts = 0;
    for (int i = 0; i < r->n_tokens; i++) {
        const char* t = r->tokens[i].text;
        if (t[0] == ' ' && t[1] != '\0')
            n_space_word_starts++;
    }
    const bool space_prefix_style = (n_space_word_starts >= 2);

    auto is_punct_only = [](const char* s) {
        if (!s || !*s)
            return false;
        const unsigned char* p = (const unsigned char*)s;
        while (*p) {
            unsigned char c = *p;
            if (c < 0x80) {
                if (!(c == '.' || c == ',' || c == '?' || c == '!' || c == ';' || c == ':' || c == '\'' || c == '"' ||
                      c == '(' || c == ')' || c == '-'))
                    return false;
                p++;
            } else if (c == 0xE3 && p[1] == 0x80 && p[2] >= 0x80 && p[2] <= 0xBF) {
                p += 3; // U+3000–U+303F CJK Symbols and Punctuation
            } else if (c == 0xE3 && p[1] == 0x83 && p[2] == 0xBB) {
                p += 3; // U+30FB ・
            } else if (c == 0xEF && p[1] == 0xBC && ((p[2] >= 0x81 && p[2] <= 0x8F) || p[2] == 0x9F)) {
                p += 3; // U+FF01–U+FF0F, U+FF1F
            } else {
                return false;
            }
        }
        return true;
    };

    auto ends_with_sentence_punct = [](const char* s) {
        size_t len = s ? strlen(s) : 0;
        if (len == 0)
            return false;
        unsigned char last = (unsigned char)s[len - 1];
        if (last == '.' || last == '!' || last == '?')
            return true;
        if (len >= 3) {
            const unsigned char* tail = (const unsigned char*)(s + len - 3);
            if (tail[0] == 0xE3 && tail[1] == 0x80 && tail[2] == 0x82)
                return true;
            if (tail[0] == 0xEF && tail[1] == 0xBC && (tail[2] == 0x81 || tail[2] == 0x9F))
                return true;
        }
        return false;
    };

    parakeet_word_data cur = {};
    bool have_cur = false;
    float cur_p_sum = 0.0f;
    int cur_p_cnt = 0;

    auto flush_cur = [&]() {
        if (cur_p_cnt > 0)
            cur.p = cur_p_sum / (float)cur_p_cnt;
        words.push_back(cur);
    };

    for (int i = 0; i < r->n_tokens; i++) {
        const auto& td = r->tokens[i];
        if (!td.text[0])
            continue;
        if (td.text[0] == ' ' && td.text[1] == '\0')
            continue; // standalone space / BOS marker

        const bool has_leading_space = (td.text[0] == ' ');
        const bool is_punct = is_punct_only(td.text);
        const bool is_new_word = !is_punct && (has_leading_space || !space_prefix_style);

        if (is_new_word && have_cur) {
            flush_cur();
            cur = {};
            cur_p_sum = 0.0f;
            cur_p_cnt = 0;
            have_cur = false;
        }

        if (!have_cur) {
            cur.t0 = td.t0;
            have_cur = true;
        }
        cur.t1 = td.t1;
        cur_p_sum += td.p;
        cur_p_cnt += 1;

        const char* src = td.text + (has_leading_space ? 1 : 0);
        size_t cur_len = strlen(cur.text);
        size_t cap = sizeof(cur.text) - cur_len - 1;
        size_t add = strlen(src);
        if (add > cap)
            add = cap;
        memcpy(cur.text + cur_len, src, add);
        cur.text[cur_len + add] = '\0';
    }
    if (have_cur)
        flush_cur();

    // Insert minimum gaps after sentence-ending punctuation (subtitle spacing).
    for (size_t wi = 0; wi + 1 < words.size(); wi++) {
        if (!ends_with_sentence_punct(words[wi].text))
            continue;
        if (words[wi].t1 >= words[wi + 1].t0 && words[wi].t1 > words[wi].t0) {
            int64_t shrunk = words[wi].t1 - frame_dur_cs;
            if (shrunk > words[wi].t0)
                words[wi].t1 = shrunk;
        }
    }

    r->n_words = (int)words.size();
    r->words = (parakeet_word_data*)calloc(r->n_words > 0 ? r->n_words : 1, sizeof(parakeet_word_data));
    if (!r->words) {
        fprintf(stderr, "parakeet: failed to allocate word results\n");
        r->n_words = 0;
        return false;
    }
    for (int i = 0; i < r->n_words; i++)
        r->words[i] = words[i];
    return true;
}
