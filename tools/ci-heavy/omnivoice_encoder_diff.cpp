// Actual-runtime CPU/Metal semantic boundary isolation. Diagnostic, no defaults changed.
#define GGML_MAX_NAME 128
#include "../../src/omnivoice.cpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
static std::filesystem::path output_dir;
template <typename T> static void save(const std::string& name, const std::vector<T>& values) {
    if (values.empty())
        throw std::runtime_error("empty stage: " + name);
    std::ofstream f(output_dir / name, std::ios::binary);
    f.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
    if (!f)
        throw std::runtime_error("write failed: " + name);
}
static std::vector<float> project(ov_higgs_tokenizer& tok, const std::vector<float>& acoustic,
                                  const std::vector<float>& semantic, int T) {
    int ca = tok.dac_hidden, cs = tok.sem_d, h = tok.hidden_size;
    std::vector<float> cat((size_t)T * h), result;
    for (int t = 0; t < T; t++) {
        std::copy_n(acoustic.data() + (size_t)t * ca, ca, cat.data() + (size_t)t * h);
        std::copy_n(semantic.data() + (size_t)t * cs, cs, cat.data() + (size_t)t * h + ca);
    }
    higgs_linear(read_tensor_f32(tok.fc_w), read_tensor_f32(tok.fc_b), cat.data(), T, h, h, result);
    return result;
}
static void margins(ov_higgs_tokenizer& tok, std::vector<float> residual, int T, const std::string& name) {
    std::ofstream f(output_dir / (name + ".margins"));
    f << std::setprecision(10);
    const int H = tok.hidden_size, D = tok.codebook_dim, S = tok.codebook_size;
    std::vector<int32_t> expected;
    if (!higgs_rvq_encode(tok, residual, T, expected))
        throw std::runtime_error("RVQ failed");
    for (int stage = 0; stage < tok.n_quantizers; stage++) {
        auto& q = tok.quantizers[stage];
        auto pin = read_tensor_f32(q.proj_in_w), bias = read_tensor_f32(q.proj_in_b);
        auto cb = read_tensor_f32(q.codebook), out = read_tensor_f32(q.proj_out_w), ob = read_tensor_f32(q.proj_out_b);
        std::vector<float> norm(S, 0), p;
        for (int s = 0; s < S; s++)
            for (int d = 0; d < D; d++)
                norm[s] += cb[(size_t)s * D + d] * cb[(size_t)s * D + d];
        higgs_linear(pin, bias, residual.data(), T, H, D, p);
        for (int t = 0; t < T; t++) {
            int best = 0, second = 0;
            float first_score = -1e30f, second_score = -1e30f;
            for (int s = 0; s < S; s++) {
                float dot = 0;
                for (int d = 0; d < D; d++)
                    dot += p[(size_t)t * D + d] * cb[(size_t)s * D + d];
                float score = 2 * dot - norm[s];
                if (score > first_score) {
                    second = best;
                    second_score = first_score;
                    best = s;
                    first_score = score;
                } else if (score > second_score) {
                    second = s;
                    second_score = score;
                }
            }
            if (best != expected[(size_t)stage * T + t])
                throw std::runtime_error("margin probe disagrees with runtime");
            f << stage << ' ' << t << ' ' << best << ' ' << second << ' ' << first_score << ' ' << second_score << '\n';
            for (int o = 0; o < H; o++) {
                float acc = ob[o];
                for (int d = 0; d < D; d++)
                    acc += out[(size_t)o * D + d] * cb[(size_t)best * D + d];
                residual[(size_t)t * H + o] -= acc;
            }
        }
    }
}
// Return each genuine truncated graph output, rather than snapshots of scratch.
static void frontend_boundaries(ov_higgs_tokenizer& tok, const std::vector<float>& wav, const std::string& prefix) {
    for (int boundary = 0; boundary < 10; boundary++) {
        ggml_init_params ip = {8192 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8192, false), nullptr, true};
        auto* c = ggml_init(ip);
        auto* full = ggml_new_graph_custom(c, 4096, false);
        auto* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, wav.size(), 1);
        ggml_set_input(x);
        auto* out = build_hubert_frontend(c, tok, x);
        ggml_build_forward_expand(full, out);
        std::vector<ggml_tensor*> targets;
        for (int i = 0; i < ggml_graph_n_nodes(full); i++) {
            auto* t = ggml_graph_node(full, i);
            if (t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_NORM || t->op == GGML_OP_UNARY)
                targets.push_back(t);
        }
        if (boundary >= (int)targets.size()) {
            ggml_free(c);
            break;
        }
        auto* selected = targets[boundary];
        ggml_set_output(selected);
        auto* graph = ggml_new_graph_custom(c, 4096, false);
        ggml_build_forward_expand(graph, selected);
        encoder_semantic_precision(graph, tok);
        auto* a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(tok.backend));
        if (!ggml_gallocr_alloc_graph(a, graph))
            throw std::runtime_error("boundary allocation");
        ggml_backend_tensor_set(x, wav.data(), 0, wav.size() * sizeof(float));
        if (ggml_backend_graph_compute(tok.backend, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("boundary compute");
        std::vector<float> v(ggml_nelements(selected));
        ggml_backend_tensor_get(selected, v.data(), 0, v.size() * sizeof(float));
        save(prefix + "-node" + std::to_string(boundary) + ".f32", v);
        std::printf("BOUNDARY %s %d %s %lld %lld %lld\n", prefix.c_str(), boundary, ggml_op_desc(selected),
                    (long long)selected->ne[0], (long long)selected->ne[1], (long long)selected->ne[2]);
        ggml_gallocr_free(a);
        ggml_free(c);
    }
}
int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: encoder-diff TOKENIZER REF.wav OUTPUT_DIR\n");
        return 2;
    }
    output_dir = argv[3];
    std::filesystem::create_directories(output_dir);
    setenv("CRISPASR_TTS_REF_CACHE", "0", 1);
    setenv("CRISPASR_OMNIVOICE_CODEC_GPU", "0", 1);
    setenv("CRISPASR_OMNIVOICE_CODEC_SPLIT", "0", 1);
    for (const char* name : {"FASTCONV", "GRAPH", "FUSED", "HYBRID"})
        setenv((std::string("CRISPASR_OMNIVOICE_ENCODER_") + name).c_str(), "0", 1);
    setenv("CRISPASR_OMNIVOICE_ENCODER_CACHE", "1", 1);
    ggml_backend_load_all();
    auto* ctx = new omnivoice_context;
    ctx->use_gpu = true;
    ctx->n_threads = 4;
    ctx->verbosity = 0;
    ctx->backend = crispasr_init_gpu_backend();
    if (!ctx->backend || core_cpu_backend::is_cpu(ctx->backend) || !load_tokenizer(ctx, argv[1]))
        return 3;
    if (omnivoice_set_voice_prompt(ctx, argv[2], "Synthetic reference.") != 0)
        return 4;
    auto original = ctx->ref_audio_codes;
    setenv("CRISPASR_OMNIVOICE_ENCODER_HYBRID", "1", 1);
    auto* gpu = encoder_semantic_tokenizer(ctx);
    if (gpu == &ctx->tokenizer || core_cpu_backend::is_cpu(gpu->backend))
        return 5;
    std::vector<float> pcm;
    int sr = 0;
    if (!crispasr::core::read_wav_mono_pcm16(argv[2], pcm, sr) || pcm.empty())
        return 6;
    if (sr != 24000)
        pcm = core_audio::resample_polyphase(pcm.data(), (int)pcm.size(), sr, 24000);
    double ss = 0;
    for (float v : pcm)
        ss += (double)v * v;
    float rms = (float)std::sqrt(ss / pcm.size());
    if (rms > 0 && rms < .1f)
        for (auto& v : pcm)
            v *= .1f / rms;
    pcm = core_omnivoice_audio::remove_silence(pcm, 200, 100, 200);
    pcm.resize(pcm.size() - pcm.size() % 960);
    save("input24.f32", pcm);
    int Ta = 0;
    auto acoustic = higgs_acoustic_encode(ctx, pcm.data(), (int)pcm.size(), &Ta);
    save("acoustic.f32", acoustic);
    auto wav = core_audio::resample_polyphase(pcm.data(), (int)pcm.size(), 24000, 16000);
    std::vector<float> padded(wav.size() + 320, 0);
    std::copy(wav.begin(), wav.end(), padded.begin() + 160);
    save("input16.f32", padded);
    if (env_bool("CRISPASR_OMNIVOICE_ENCODER_DIFF_NODES")) {
        frontend_boundaries(ctx->tokenizer, padded, "cpu");
        frontend_boundaries(*gpu, padded, "gpu");
    }
    for (int fe = 0; fe < 2; fe++) {
        int Tf = 0;
        auto feature =
            higgs_hubert_frontend(ctx, padded.data(), (int)padded.size(), &Tf, nullptr, fe ? gpu : &ctx->tokenizer);
        auto fn = std::string("fe") + char('0' + fe);
        save(fn + ".f32", feature);
        for (int hb = 0; hb < 2; hb++) {
            auto mean = higgs_hubert_encoder(ctx, feature.data(), Tf, nullptr, hb ? gpu : &ctx->tokenizer);
            auto hn = fn + "-hb" + char('0' + hb);
            save(hn + ".f32", mean);
            int Ts = (Tf + 1) / 2;
            std::vector<float> ds((size_t)Ts * ctx->tokenizer.sem_d);
            for (int t = 0; t < Ts; t++)
                std::copy_n(mean.data() + (size_t)(2 * t) * ctx->tokenizer.sem_d, ctx->tokenizer.sem_d,
                            ds.data() + (size_t)t * ctx->tokenizer.sem_d);
            for (int sem = 0; sem < 2; sem++) {
                auto semantic = higgs_encoder_semantic(ctx, ds.data(), Ts, sem ? gpu : &ctx->tokenizer);
                auto name = hn + "-sem" + char('0' + sem);
                save(name + ".f32", semantic);
                int T = std::min(Ta, Ts);
                auto embedding = project(ctx->tokenizer, acoustic, semantic, T);
                save(name + "-fc.f32", embedding);
                std::vector<int32_t> codes;
                if (!higgs_rvq_encode(ctx->tokenizer, embedding, T, codes))
                    return 7;
                if (fe == 0 && hb == 0 && sem == 0 && codes != original)
                    return 8;
                save(name + ".codes", codes);
                margins(ctx->tokenizer, embedding, T, name);
                int diff = 0;
                for (size_t i = 0; i < codes.size(); i++)
                    diff += codes[i] != original[i];
                std::printf("PIPELINE %s frames=%d differences=%d\n", name.c_str(), T, diff);
                std::fflush(stdout);
            }
        }
    }
    omnivoice_free(ctx);
}
