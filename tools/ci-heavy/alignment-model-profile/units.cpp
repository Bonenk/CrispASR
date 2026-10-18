// Query the actual built library's label/display splitters, no local imitation.
#include "crispasr_aligner.h"
#include "json.hpp"
#include <iostream>
int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    nlohmann::json out = {{"labels", crispasr_tokenise_align_words(argv[1])},
                          {"display", crispasr_tokenise_align_display_words(argv[1])}};
    std::cout << out.dump() << '\n';
}
