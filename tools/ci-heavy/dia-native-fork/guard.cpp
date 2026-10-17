#include "dia_sampling.h"
#include "fork.h"
#include <cassert>
#include <cstdio>
#include <limits>

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    std::mt19937 initial(42);
    std::vector<float> noise(9 * 1028);
    dia_rng::exponential_batch(noise.data(), noise.size(), initial);
    std::vector<float> logits(1028, -std::numeric_limits<float>::infinity());
    logits[904] = 1;
    std::vector<uint32_t> tokens;
    for (size_t head = 0; head < 9; ++head)
        tokens.push_back(dia_sample_token(logits.data(), 1028, 1.2f, .95f, 45, initial, noise.data() + head * 1028));
    assert(tokens == std::vector<uint32_t>(9, 904));
    int rejected = 0;
    for (const auto* mode : {"off", "identity", "restore"}) {
        dia_native_fork fork;
        assert(fork.configure(mode, argv[1]));
        auto used = tokens;
        auto rng = initial;
        assert(fork.apply(977, used, rng));
        assert(used == tokens && rng == initial && !fork.seen);
        assert(fork.apply(978, used, rng));
        auto wanted = tokens;
        if (std::strcmp(mode, "restore") == 0)
            wanted[2] = 877;
        assert(used == wanted && rng == initial && fork.seen);
        assert(!fork.apply(978, used, rng));
        ++rejected;
        assert(fork.finish(used, used, rng));
    }
    for (const auto* mode : {"", "877", "restore-all"}) {
        dia_native_fork fork;
        assert(!fork.configure(mode, argv[1]));
        ++rejected;
    }
    for (const auto& bad : {std::vector<uint32_t>(8, 904), std::vector<uint32_t>(9, 877)}) {
        dia_native_fork fork;
        assert(fork.configure("restore", argv[1]));
        auto used = bad;
        auto rng = initial;
        assert(!fork.apply(978, used, rng) && used == bad && rng == initial);
        ++rejected;
    }
    dia_native_fork absent;
    assert(absent.configure("off", argv[1]));
    assert(!absent.finish(tokens, tokens, initial));
    ++rejected;
    assert(rejected == 9);
    std::puts("DIA_NATIVE_FORK_GUARD_PASS: actual pinned sampler; off/identity/restore, exact RNG and nine rejected "
              "controls");
}
