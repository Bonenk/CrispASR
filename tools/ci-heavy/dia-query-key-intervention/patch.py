"""Scratch-only paired Q/K inputs at the fixed original diagnostic boundary."""
import importlib.util
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent/'dia-cache-intervention'
spec = importlib.util.spec_from_file_location('original_key_patch', BASE/'patch.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
SOURCE = base.SOURCE


def patch(path):
    base.patch(path)
    path = Path(path)
    text = path.read_text()
    needle = '    const int head_dim = (int)m.head_dim;'
    addition = '''
    std::vector<float> dia_intervention_query;
    if (const char* replacement = std::getenv("DIA_QUERY_INTERVENTION_FILE")) {
        if (B != 2 || n_heads != 16 || head_dim != 128 || max_gen != 979 ||
            !std::getenv("DIA_KEY_INTERVENTION_FILE") ||
            !dia_read_intervention_keys(replacement, (size_t)2 * 16 * 128, dia_intervention_query)) {
            fprintf(stderr, "DIA_QUERY_INTERVENTION_INVALID_FILE\\n");
            return nullptr;
        }
    }
'''
    assert text.count(needle) == 1
    text = text.replace(needle, needle+addition)
    needle = '            ggml_tensor* dia_intervention_key_input = nullptr;'
    assert text.count(needle) == 1
    text = text.replace(needle, needle+'\n            ggml_tensor* dia_intervention_query_input = nullptr;')
    needle = '                    mark_sa(Q, l, "q");'
    addition = '''                    if (l == 0 && step == 978 && !dia_intervention_query.empty()) {
                        Q = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, head_dim, n_heads, T_cur, B);
                        dia_intervention_query_input = Q;
                        ggml_set_input(Q);
                        ggml_set_name(Q, "dia_intervention_current_query");
                    }

'''
    assert text.count(needle) == 1
    text = text.replace(needle, addition+needle)
    needle = '            // Compute\n            ggml_status st = ggml_backend_sched_graph_compute(ctx->sched, gf);'
    addition = '''            if (step == 978 && !dia_intervention_query.empty()) {
                if (!dia_intervention_query_input) {
                    ggml_free(ctx0);
                    return nullptr;
                }
                ggml_backend_tensor_set(dia_intervention_query_input, dia_intervention_query.data(), 0,
                                        dia_intervention_query.size() * sizeof(float));
                fprintf(stderr, "DIA_QUERY_INTERVENTION_APPLIED step=978 layer=0 values=%zu\\n",
                        dia_intervention_query.size());
            }

'''
    assert text.count(needle) == 1
    text = text.replace(needle, addition+needle)
    path.write_text(text)
