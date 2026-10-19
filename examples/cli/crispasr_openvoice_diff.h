// Strict OpenVoice blueprint replay. Inputs are owned by the pinned archive.
#pragma once
#include "openvoice2.h"
#include "crispasr_diff.h"
#include <map>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <limits>

inline int crispasr_openvoice_diff(const std::string& model, crispasr_diff::Ref& ref, bool gpu) {
    const auto source = ref.get_f32("raw_audio");
    const auto reference = ref.get_f32("reference_audio");
    const std::string rate_text = ref.meta("ov2_sample_rate");
    const std::string tau_text = ref.meta("ov2_tau");
    const bool noisy = tau_text == "0.3";
    if (!source.first || !reference.first || !source.second || !reference.second || source.second > INT32_MAX ||
        reference.second > INT32_MAX || rate_text != "22050" || (tau_text != "0" && tau_text != "0.0" && !noisy)) {
        fprintf(stderr, "openvoice2: missing or invalid native-rate replay inputs/recipe\n");
        return 4;
    }
    auto params = openvoice2_context_default_params();
    params.use_gpu = gpu;
    params.tau = noisy ? .3f : 0;
    params.verbosity = 1;
    std::unique_ptr<openvoice2_context, decltype(&openvoice2_free)> ctx(
        openvoice2_init_from_file(model.c_str(), params), openvoice2_free);
    if (!ctx || !openvoice2_set_source_speaker(ctx.get(), ref.meta("ov2_source_speaker").c_str())) {
        fprintf(stderr, "openvoice2: missing model or exact source-speaker identity\n");
        return 4;
    }
    if (noisy) {
        const auto noise = ref.get_f32("enc_q_noise");
        if (!noise.first || !noise.second || noise.second > INT32_MAX ||
            !openvoice2_set_posterior_noise(ctx.get(), noise.first, (int)noise.second)) {
            fprintf(stderr, "openvoice2: missing or invalid original posterior noise\n");
            return 4;
        }
    }
    using Captures = std::map<std::string, std::vector<float>>;
    Captures captures;
    openvoice2_set_stage_callback(
        ctx.get(),
        [](const char* name, const float* data, size_t n, void* user) {
            (*static_cast<Captures*>(user))[name] = std::vector<float>(data, data + n);
        },
        &captures);
    float* output = nullptr;
    int n = 0;
    const bool ok = openvoice2_convert(ctx.get(), source.first, (int)source.second, 22050, reference.first,
                                       (int)reference.second, 22050, &output, &n);
    free(output);
    if (!ok || n <= 0)
        return 4;
    std::vector<const char*> stages = {"src_pcm",           "ref_pcm",       "src_spec",         "ref_spec",
                                       "ref_enc_layernorm", "ref_enc_conv0", "ref_enc_conv1",    "ref_enc_conv2",
                                       "ref_enc_conv3",     "ref_enc_conv4", "ref_enc_conv5",    "ref_enc_gru_out",
                                       "target_se",         "src_se",        "enc_q_pre",        "enc_q_wn_out",
                                       "enc_q_stats",       "enc_q_z",       "z_after_flow_fwd", "z_after_flow_rev",
                                       "dec_conv_pre",      "dec_stage0",    "dec_stage1",       "dec_stage2",
                                       "dec_stage3",        "dec_output",    "output_pcm"};
    if (noisy)
        stages.push_back("enc_q_noise");
    int failed = 0;
    for (const char* name : stages) {
        auto found = captures.find(name);
        auto truth = ref.get_f32(name);
        if (found == captures.end() || !truth.first || found->second.size() != truth.second || !truth.second) {
            printf("[FAIL] %-24s missing/empty stage or wrong element count\n", name);
            ++failed;
            continue;
        }
        double aa = 0, bb = 0, ab = 0, error = 0, max_error = 0;
        bool finite = true;
        for (size_t i = 0; i < truth.second; ++i) {
            const double a = found->second[i], b = truth.first[i], d = a - b;
            finite = finite && std::isfinite(a) && std::isfinite(b);
            aa += a * a;
            bb += b * b;
            ab += a * b;
            error += d * d;
            max_error = std::max(max_error, std::abs(d));
        }
        const double cosine = aa && bb ? ab / std::sqrt(aa * bb) : (aa == bb ? 1 : 0);
        const double relative = bb ? std::sqrt(error / bb) : (aa ? INFINITY : 0);
        const double ratio = bb ? std::sqrt(aa / bb) : (aa ? INFINITY : 1);
        const bool input = std::string(name) == "src_pcm" || std::string(name) == "ref_pcm";
        const bool pass =
            finite && cosine >= .99 && std::abs(ratio - 1) <= .02 && relative <= .15 && (!input || max_error == 0);
        printf("[%s] %-24s n=%zu cos=%.9f |mine|=%.9g |ref|=%.9g ratio=%.9g rel_l2=%.9g max_abs=%.9g\n",
               pass ? "PASS" : "FAIL", name, truth.second, cosine, std::sqrt(aa), std::sqrt(bb), ratio, relative,
               max_error);
        failed += !pass;
    }
    printf("openvoice2: %zu required stages, %d failures (no optional skips)\n", stages.size(), failed);
    return failed ? 1 : 0;
}
