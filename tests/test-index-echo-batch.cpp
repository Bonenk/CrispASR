#include <catch2/catch_test_macros.hpp>
#include "core/index_echo_batch.h"
#include "llama-batch.h"
#include "llama-vocab.h"

TEST_CASE("Index-Echo audio positions survive llama M-RoPE microbatch splitting", "[unit][index-echo]") {
    llama_batch batch = llama_batch_init(5, 8, 1);
    batch.n_tokens = 5;
    REQUIRE(core_index_echo::positions(batch, 137, 4));
    for (int i = 0; i < batch.n_tokens; ++i) {
        for (int d = 0; d < 8; ++d)
            batch.embd[i * 8 + d] = 0;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = i == 4;
    }
    llama_vocab vocab;
    llama_batch_allocr allocator(4);
    REQUIRE(allocator.init(batch, vocab, nullptr, 8, 1, false));
    allocator.split_reset();
    int consumed = 0;
    while (consumed < batch.n_tokens) {
        auto microbatch = allocator.split_simple(2);
        REQUIRE(microbatch.n_tokens > 0);
        REQUIRE(microbatch.n_pos == 4);
        for (unsigned axis = 0; axis < microbatch.n_pos; ++axis)
            for (unsigned i = 0; i < microbatch.n_tokens; ++i)
                REQUIRE(microbatch.pos[axis * microbatch.n_tokens + i] == 137 + consumed + (int)i);
        consumed += microbatch.n_tokens;
    }
    llama_batch_free(batch);
}
