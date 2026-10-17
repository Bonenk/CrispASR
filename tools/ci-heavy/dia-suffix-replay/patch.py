"""Scratch-only one-step native replay and owned decoder activation injection."""
from pathlib import Path
SOURCE = '8d0ad8862d2cad5d305a48a57f2a99b5024bc39a'


def patch(path):
    path = Path(path); text = path.read_text()
    def replace(needle, value):
        nonlocal text
        assert text.count(needle) == 1, needle
        text = text.replace(needle, value)
    replace('#include "dia_dump_step.h"', '#include "dia_dump_step.h"\n#include "dia_intervention_key_input.h"\n#include "dia_suffix_prefix.h"')
    needle = '    std::vector<std::vector<float>> self_v(m.n_decoder_layers);'
    replace(needle, needle+'''
    const char* suffix_dir = std::getenv("DIA_SUFFIX_INPUT_DIR");
    const char* suffix_file = std::getenv("DIA_SUFFIX_ACTIVATION");
    const char* suffix_layer_text = std::getenv("DIA_SUFFIX_LAYER");
    int suffix_layer = -1;
    std::vector<float> suffix_values;
    if (suffix_file || suffix_layer_text) {
        char* end = nullptr;
        const long layer = suffix_layer_text ? std::strtol(suffix_layer_text, &end, 10) : -1;
        if (!suffix_dir || !suffix_file || !suffix_layer_text || !end || end == suffix_layer_text || *end ||
            layer < 0 || layer >= (long)m.n_decoder_layers ||
            !dia_read_intervention_keys(suffix_file, (size_t)2048 * 2, suffix_values)) {
            fprintf(stderr, "DIA_SUFFIX_INVALID_ACTIVATION\\n");
            return nullptr;
        }
        suffix_layer = (int)layer;
    }
    if (suffix_dir) {
        if (!dia_force || max_gen != 979 || B != 2 || kv_dim != 512 || dec_hidden != 2048 ||
            m.n_decoder_layers != 18 || !cross_f32) {
            fprintf(stderr, "DIA_SUFFIX_INVALID_RECIPE\\n");
            return nullptr;
        }
        FILE* cross_file = dia_dump_dir ? fopen(dia_dpath("suffix-cross-cache.f32").c_str(), "wb") : nullptr;
        if (!cross_file) {
            fprintf(stderr, "DIA_SUFFIX_CROSS_CAPTURE_FAILED\\n");
            return nullptr;
        }
        for (int l = 0; l < 18; ++l) {
            for (const auto* values : {&cross_k_f32[l], &cross_v_f32[l]}) {
                if (fwrite(values->data(), sizeof(float), values->size(), cross_file) != values->size()) {
                    fclose(cross_file);
                    return nullptr;
                }
            }
        }
        if (fclose(cross_file))
            return nullptr;
        for (int l = 0; l < 18; ++l) {
            for (const char* part : {"k", "v"}) {
                const std::string name = std::string(suffix_dir) + "/cpp_decoder_sa" + std::to_string(l) + "_" + part + ".f32";
                std::vector<float> full;
                auto& target = part[0] == 'k' ? self_k[l] : self_v[l];
                if (!dia_read_intervention_keys(name.c_str(), (size_t)2 * 979 * 512, full) ||
                    !dia_suffix_prefix(full, 2, 979, 512, target)) {
                    fprintf(stderr, "DIA_SUFFIX_INVALID_PREFIX layer=%d part=%s\\n", l, part);
                    return nullptr;
                }
            }
        }
        fprintf(stderr, "DIA_SUFFIX_PREFIX_READY positions=978 layers=18\\n");
    }
''')
    replace('for (uint32_t step = 0; step < max_gen; step++) {', 'for (uint32_t step = suffix_dir ? 978 : 0; step < max_gen; step++) {')
    needle = '            // --- New K/V outputs for self-attention (to append to cache) ---'
    replace(needle, '''            ggml_tensor* suffix_computed = nullptr;
            ggml_tensor* suffix_input = nullptr;
'''+needle)
    needle = '                mark_boundary(cur, l, "output");'
    replace(needle, '''                if (l == suffix_layer) {
                    suffix_computed = cur;
                    ggml_set_name(suffix_computed, "dia_suffix_computed");
                    ggml_set_output(suffix_computed);
                    suffix_input = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, dec_hidden, T_cur, B);
                    ggml_set_input(suffix_input);
                    ggml_set_output(suffix_input);
                    cur = suffix_input;
                }
'''+needle)
    needle = '            ggml_backend_sched_reset(ctx->sched);'
    # Only the decoder reset has this indentation; encoder/cross reset differs.
    replace(needle, '''            // Preserve the replaced producer as a graph root for honest captures.
            if (suffix_computed)
                ggml_build_forward_expand(gf, suffix_computed);
'''+needle)
    needle = '            // Compute\n            ggml_status st = ggml_backend_sched_graph_compute(ctx->sched, gf);'
    replace(needle, '''            if (suffix_input) {
                ggml_backend_tensor_set(suffix_input, suffix_values.data(), 0, suffix_values.size() * sizeof(float));
                fprintf(stderr, "DIA_SUFFIX_ACTIVATION_APPLIED layer=%d\\n", suffix_layer);
            }
'''+needle)
    needle = '            if (step == dia_dump_step && dia_dump_dir) {\n                if (FILE* marker'
    replace(needle, '''            if (suffix_input && dia_dump_dir) {
                for (auto entry : {std::make_pair("suffix-computed.f32", suffix_computed),
                                   std::make_pair("suffix-used.f32", suffix_input)}) {
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
