#include "crispasr_openvoice_diff.h"
#include <cstdlib>
int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: openvoice-bake model.gguf reference.gguf\n");
        return 2;
    }
    crispasr_diff::Ref reference;
    if (!reference.load(argv[2]))
        return 4;
    return crispasr_openvoice_diff(argv[1], reference, std::getenv("CRISPASR_DIFF_USE_GPU") != nullptr);
}
