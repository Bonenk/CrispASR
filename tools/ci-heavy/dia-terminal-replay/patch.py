"""Extend the accepted scratch replay with an owned post-final-norm leaf."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('accepted_suffix_patch', Path(__file__).parent.parent/'dia-suffix-replay/patch.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
SOURCE = base.SOURCE


def patch(path):
    base.patch(path)
    path = Path(path); text = path.read_text()
    def replace(needle, value):
        nonlocal text
        assert text.count(needle) == 1, needle
        text = text.replace(needle, value)
    needle = '    std::vector<float> suffix_values;'
    replace(needle, needle+'''
    const char* terminal_file = std::getenv("DIA_TERMINAL_HIDDEN");
    std::vector<float> terminal_values;
    if (terminal_file && (!suffix_dir || suffix_file || suffix_layer_text ||
                         !dia_read_intervention_keys(terminal_file, (size_t)2048 * 2, terminal_values))) {
        fprintf(stderr, "DIA_TERMINAL_INVALID_HIDDEN\\n");
        return nullptr;
    }
''')
    needle = '            ggml_tensor* suffix_input = nullptr;'
    replace(needle, needle+'''
            ggml_tensor* terminal_computed = nullptr;
            ggml_tensor* terminal_input = nullptr;
''')
    needle = '            // cur: (dec_hidden, 1, B)'
    replace(needle, needle+'''
            if (terminal_file) {
                terminal_computed = cur;
                ggml_set_name(terminal_computed, "dia_terminal_computed");
                ggml_set_output(terminal_computed);
                terminal_input = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, dec_hidden, T_cur, B);
                ggml_set_input(terminal_input);
                ggml_set_output(terminal_input);
                cur = terminal_input;
            }
''')
    needle = '            // Preserve the replaced producer as a graph root for honest captures.'
    replace(needle, '''            if (terminal_computed)
                ggml_build_forward_expand(gf, terminal_computed);
'''+needle)
    needle = '            if (suffix_input) {\n                ggml_backend_tensor_set(suffix_input'
    replace(needle, '''            if (terminal_input) {
                ggml_backend_tensor_set(terminal_input, terminal_values.data(), 0, terminal_values.size() * sizeof(float));
                fprintf(stderr, "DIA_TERMINAL_HIDDEN_APPLIED\\n");
            }
'''+needle)
    needle = '            if (step == dia_dump_step && dia_dump_dir) {\n                if (FILE* marker'
    replace(needle, '''            if (terminal_input && dia_dump_dir) {
                for (auto entry : {std::make_pair("terminal-computed.f32", terminal_computed),
                                   std::make_pair("terminal-used.f32", terminal_input)}) {
                    std::vector<float> values((size_t)dec_hidden * B);
                    ggml_backend_tensor_get(entry.second, values.data(), 0, values.size() * sizeof(float));
                    FILE* file = fopen(dia_dpath(entry.first).c_str(), "wb");
                    if (!file || fwrite(values.data(), sizeof(float), values.size(), file) != values.size()) {
                        if (file) fclose(file);
                        ggml_free(ctx0);
                        return nullptr;
                    }
                    fclose(file);
                }
            }
'''+needle)
    path.write_text(text)
