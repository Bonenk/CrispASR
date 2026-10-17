"""Apply a bounded experiment to the immutable step978 diagnostic runtime.

This patches a scratch checkout only. It is not a production implementation.
"""
from pathlib import Path

SOURCE = '8d0ad8862d2cad5d305a48a57f2a99b5024bc39a'


def patch(path):
    path = Path(path)
    text = path.read_text()
    needle = '                    mark_sa(Q, l, "q");'
    addition = '''                    // Scratch-only full-key replacement at the fixed diagnostic boundary.
                    // The native producer/cache remains intact; only this attention input changes.
                    if (l == 0 && step == 978 && std::getenv("DIA_KEY_INTERVENTION_FILE")) {
                        K_full = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, head_dim, n_kv_heads, T_past + T_cur, B);
                        dia_intervention_key_input = K_full;
                        ggml_set_name(K_full, "dia_intervention_full_key");
                        ggml_set_input(K_full);
                    }

'''
    assert text.count(needle) == 1
    text = text.replace(needle, addition + needle)
    needle = '            // Compute\n            ggml_status st = ggml_backend_sched_graph_compute(ctx->sched, gf);'
    addition = '''            if (step == 978 && !dia_intervention_keys.empty()) {
                ggml_tensor* key = dia_intervention_key_input;
                if (!key) {
                    ggml_free(ctx0);
                    return nullptr;
                }
                ggml_backend_tensor_set(key, dia_intervention_keys.data(), 0,
                                        dia_intervention_keys.size() * sizeof(float));
                fprintf(stderr, "DIA_KEY_INTERVENTION_APPLIED step=978 layer=0 values=%zu\\n",
                        dia_intervention_keys.size());
            }

'''
    assert text.count(needle) == 1
    text = text.replace(needle, addition + needle)
    needle = '#include "dia_dump_step.h"'
    assert text.count(needle) == 1
    text = text.replace(needle, needle + '\n#include "dia_intervention_key_input.h"')
    needle = '    const int head_dim = (int)m.head_dim;'
    addition = '''
    std::vector<float> dia_intervention_keys;
    if (const char* replacement = std::getenv("DIA_KEY_INTERVENTION_FILE")) {
        if (B != 2 || kv_dim != 512 || max_gen != 979 ||
            !dia_read_intervention_keys(replacement, (size_t)2 * 979 * 512, dia_intervention_keys)) {
            fprintf(stderr, "DIA_KEY_INTERVENTION_INVALID_FILE\\n");
            return nullptr;
        }
    }
'''
    assert text.count(needle) == 1
    text = text.replace(needle, needle + addition)
    needle = '            // --- New K/V outputs for self-attention (to append to cache) ---'
    assert text.count(needle) == 1
    text = text.replace(needle, '            ggml_tensor* dia_intervention_key_input = nullptr;\n' + needle)
    needle = '''                    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, kn.c_str()), pk.data(), 0,
                                            pk.size() * sizeof(float));'''
    assert text.count(needle) == 1
    text = text.replace(needle, '''                    // A replacement full-key leaf prunes layer0's unused past-key input.
                    // Every other cache upload remains mandatory.
                    if (!(l == 0 && step == 978 && !dia_intervention_keys.empty())) {
                        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, kn.c_str()), pk.data(), 0,
                                                pk.size() * sizeof(float));
                    }''')
    path.write_text(text)
