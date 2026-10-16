"""Patch only the pinned scratch runtime; reject changed or repeated anchors."""
from pathlib import Path


def patch(file):
    source = Path(file).read_text()
    assert 'dia_serial_head' not in source
    anchor = '#include "dia_sampling.h"'
    assert source.count(anchor) == 1
    source = source.replace(anchor,anchor+'\n#include "dia_serial_head.h"')
    start = source.index('float* dia_tts_synthesize(')
    pos = source.index('    auto& p = ctx->params;',start)+len('    auto& p = ctx->params;')
    source = source[:pos]+'''
    const char* serial_head_env = crispasr_env::get("CRISPASR_DIA_SERIAL_HEAD");
    const int serial_head_mode = dia_serial_head::mode(serial_head_env);
    if (serial_head_mode < 0) {
        fprintf(stderr, "DIA_SERIAL_HEAD: invalid mode\\n");
        return nullptr;
    }
    if (serial_head_env)
        fprintf(stderr, "DIA_SERIAL_HEAD: mode=%d CPU_graph_diagnostic=1\\n", serial_head_mode);
'''+source[pos:]
    anchor = '                head_logits[h] = ggml_mul_mat(ctx0, m.decoder.heads[h], cur); // (output_vocab_size, 1, B)'
    assert source.count(anchor) == 1
    source = source.replace(anchor,anchor+'''
                head_logits[h] = dia_serial_head::apply(ctx0, head_logits[h], m.decoder.heads[h], cur, serial_head_mode);''')
    anchor = '    size_t n_frames = filtered_codes.size() / m.n_output_heads;'
    assert source.count(anchor) == 1
    source = source.replace(anchor,'''    if (!dia_serial_head::finish(crispasr_env::get("CRISPASR_DIA_SERIAL_HEAD_DIR"), ctx->output_tokens,
                                  filtered_codes, ctx->rng)) {
        fprintf(stderr, "DIA_SERIAL_HEAD: failed raw evidence capture\\n");
        return nullptr;
    }
'''+anchor)
    Path(file).write_text(source)
