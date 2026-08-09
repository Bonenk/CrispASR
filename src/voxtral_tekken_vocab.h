// src/voxtral_tekken_vocab.h — decoding the packed Tekken vocabulary blob,
// and the one bound that decode has to respect (#338).
//
// The blob is a flat `[u16 len][len bytes]` stream of BPE pieces. Token ids
// are assigned `n_specials + rank`, because ids `0 .. n_specials-1` belong to
// the special tokens. That much is mechanical.
//
// What is NOT mechanical: **the blob can be longer than the model's embedding
// table.** Mistral's Tekken vocabularies serialize more entries than a given
// checkpoint activates — Voxtral-4B-TTS-2603 has `llm_vocab_size = 131072`
// with 1000 specials, so only the first 130072 BPE pieces are live, and the
// tail is inert padding. Feeding a tail piece into the encoder yields a token
// id >= llm_vocab_size, which then reaches
// `ggml_get_rows(model.token_embd, …)` out of bounds: a row-index assertion on
// CPU, and on CUDA a non-finite first frame followed by runaway generation
// until the KV cache fills. It is input-dependent — it fires only for texts
// whose BPE merge path happens to land on a tail entry — which is why it
// survived every smoke test (reported in #338).
//
// The vendored reference tokenizer used by the diff harness
// (`tools/kaggle/voxtral-diff-harness/refsrc/voxtral_tts_tokenizer.c`) has
// always enforced this: `#define MAX_VOCAB 130072`, with "IDs 1000..131071"
// spelled out at the top of the file. The production runtime did not. This
// header exists so the two cannot drift again, and so the rule is reachable
// from a unit test without loading a 4 B model.

#pragma once

#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace voxtral_tekken {

// Number of BPE pieces a checkpoint can actually address, i.e. the ids left
// over once the specials have taken `0 .. n_specials-1`. Returns 0 when the
// header values are nonsensical rather than a negative count.
inline int active_bpe_count(int llm_vocab_size, int n_specials) {
    if (llm_vocab_size <= 0 || n_specials < 0 || n_specials >= llm_vocab_size)
        return 0;
    return llm_vocab_size - n_specials;
}

// True when `token_id` can be used as a row index into an embedding table of
// `llm_vocab_size` rows. Callers must gate on this before `ggml_get_rows`.
inline bool token_id_in_range(int32_t token_id, int llm_vocab_size) {
    return token_id >= 0 && token_id < llm_vocab_size;
}

struct DecodeStats {
    int n_active = 0;   // pieces admitted to the encoder
    int n_inactive = 0; // pieces parsed but past the embedding table
};

// Decode the packed blob into `id_to_piece` (every parsed piece, so a debug
// dump of an inert tail is still possible) and `piece_to_id` (**only** the
// pieces the model can address — this map is what the BPE encoder searches,
// so an entry here is a token the runtime may emit).
//
// `active_limit` is `active_bpe_count(llm_vocab_size, n_specials)`; pass 0 to
// admit everything, which is only correct when the caller genuinely has no
// embedding table to overrun.
inline DecodeStats decode_blob(const std::vector<uint8_t>& blob, int n_specials,
                               const std::vector<std::string>& specials, int active_limit,
                               std::vector<std::string>& id_to_piece, std::map<std::string, int>& piece_to_id) {
    DecodeStats st;
    id_to_piece.clear();
    piece_to_id.clear();

    // Specials occupy the low ids. They are always addressable — they are
    // inside llm_vocab_size by construction.
    for (int i = 0; i < n_specials && i < (int)specials.size(); i++) {
        id_to_piece.push_back(specials[i]);
        piece_to_id[specials[i]] = i;
    }
    while ((int)id_to_piece.size() < n_specials)
        id_to_piece.push_back("");

    const uint8_t* p = blob.data();
    const uint8_t* end = p + blob.size();
    int bpe_id = n_specials;
    while (p + 2 <= end) {
        uint16_t len = 0;
        std::memcpy(&len, p, 2);
        p += 2;
        if (p + len > end)
            break;
        std::string piece((const char*)p, len);
        p += len;
        id_to_piece.push_back(piece);
        // Keep parsing past the limit so the blob is fully consumed and the
        // inactive count is honest, but do NOT let those pieces into the map
        // the encoder searches.
        if (active_limit <= 0 || st.n_active < active_limit) {
            piece_to_id[piece] = bpe_id;
            st.n_active++;
        } else {
            st.n_inactive++;
        }
        bpe_id++;
    }
    return st;
}

} // namespace voxtral_tekken
