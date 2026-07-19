// tools/mbr_diff_probe.cpp — standalone §248 Mel-Band RoFormer diff probe.
//
// Runs mel_band_roformer_diff (front-end stages in Phase 1) without linking the
// full crispasr-lib, so the front-end can be validated against the reference
// fixture in isolation while the backend is mid-port.
//
//   mbr-diff-probe <model.gguf> <ref.gguf> [verbosity]

#include "mel_band_roformer.h"

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <ref.gguf> [verbosity]\n", argv[0]);
        return 2;
    }
    const int verbosity = (argc > 3) ? atoi(argv[3]) : 1;
    return mel_band_roformer_diff(argv[1], argv[2], /*audio_wav=*/nullptr, verbosity);
}
