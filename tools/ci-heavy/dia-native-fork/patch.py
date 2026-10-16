"""Patch only the pinned scratch runtime; never change neural/sampling arithmetic."""
from pathlib import Path

SOURCE = '8d0ad8862d2cad5d305a48a57f2a99b5024bc39a'


def patch(file):
    file = Path(file); text = file.read_text()
    assert 'dia_native_fork fork;' not in text and '"dia_native_fork.h"' not in text
    changes = [('#include "dia_dump_step.h"', '#include "dia_dump_step.h"\n#include "dia_native_fork.h"'),
               ('    // Reset generation state\n', '''    dia_native_fork fork;
    if (!fork.configure(std::getenv("DIA_NATIVE_FORK"), std::getenv("DIA_NATIVE_FORK_DIR"))) {
        fprintf(stderr, "DIA_NATIVE_FORK_INVALID_RECIPE\\n");
        return nullptr;
    }

    // Reset generation state
'''),
               ('            if (dia_dump_tokens) {', '''            if (!fork.apply(step, ctx->current_audio_tokens, ctx->rng)) {
                fprintf(stderr, "DIA_NATIVE_FORK_INVALID_SAMPLE\\n");
                ggml_free(ctx0);
                return nullptr;
            }

            if (dia_dump_tokens) {'''),
               ('    size_t n_frames = filtered_codes.size() / m.n_output_heads;', '''    if (!fork.finish(ctx->output_tokens, filtered_codes, ctx->rng)) {
        fprintf(stderr, "DIA_NATIVE_FORK_INVALID_OUTPUT\\n");
        return nullptr;
    }
    fprintf(stderr, "DIA_NATIVE_FORK_CAPTURE_COMPLETE\\n");
    size_t n_frames = filtered_codes.size() / m.n_output_heads;''')]
    for old, new in changes:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    file.write_text(text)
