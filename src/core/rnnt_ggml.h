// src/core/rnnt_ggml.h — RNNT/TDT transducer decode as ggml graphs (§232).
//
// Shared by every NeMo-style transducer backend (parakeet, nemotron, …) whose
// predictor is a 2-layer LSTM and whose joint is pred/enc projections + ReLU +
// output projection. The default runtimes step these on the CPU via
// cblas_sgemv, leaving the GPU idle (the §232 P100 decode bottleneck). These
// helpers run the SAME math as ggml graphs on the sched's backend so the whole
// per-step decode executes on the GPU (opt-in per backend).
//
// Correctness-first: per-step sched dispatch, LSTM state carried via CPU
// readback. Transcript-validated identical to the cblas path on M1 (parakeet,
// jfk + multispeaker, CPU & Metal). The perf win is P100-only — Apple Accelerate
// cblas is already fast, so M1 is neutral (see LEARNINGS 29-30). A persistent-
// graph / in-graph-argmax variant is the follow-up if per-step is launch-bound.
//
// LSTM recurrence (PyTorch convention, concatenated gate order [i,f,g,o]):
//   gates = w_ih·x + b_ih + w_hh·h + b_hh
//   c' = sigmoid(f)·c + sigmoid(i)·tanh(g);   h' = sigmoid(o)·tanh(c')
// Joint (ReLU — NOT tanh):
//   mid = pred_w·pred_u + pred_b;   logits = out_w·relu(proj_e + mid) + out_b
#pragma once

#include <vector>

#include "ggml-backend.h"
#include "ggml.h"

namespace core_rnnt_ggml {

// One LSTM layer as ggml ops. x/h_in/c_in are [H]; returns h' ([H]), writes c'.
static inline ggml_tensor* lstm_layer(ggml_context* c0, ggml_tensor* x, ggml_tensor* w_ih, ggml_tensor* b_ih,
                                      ggml_tensor* w_hh, ggml_tensor* b_hh, ggml_tensor* h_in, ggml_tensor* c_in, int H,
                                      ggml_tensor** c_out) {
    ggml_tensor* g = ggml_add(c0, ggml_mul_mat(c0, w_ih, x), b_ih);
    g = ggml_add(c0, g, ggml_add(c0, ggml_mul_mat(c0, w_hh, h_in), b_hh));
    const size_t fs = sizeof(float);
    ggml_tensor* i_ = ggml_sigmoid(c0, ggml_view_1d(c0, g, H, 0 * (size_t)H * fs));
    ggml_tensor* f_ = ggml_sigmoid(c0, ggml_view_1d(c0, g, H, 1 * (size_t)H * fs));
    ggml_tensor* g_ = ggml_tanh(c0, ggml_view_1d(c0, g, H, 2 * (size_t)H * fs));
    ggml_tensor* o_ = ggml_sigmoid(c0, ggml_view_1d(c0, g, H, 3 * (size_t)H * fs));
    ggml_tensor* c_new = ggml_add(c0, ggml_mul(c0, f_, c_in), ggml_mul(c0, i_, g_)); // f·c + i·g
    ggml_tensor* h_new = ggml_mul(c0, o_, ggml_tanh(c0, c_new));                     // o·tanh(c')
    *c_out = c_new;
    return h_new;
}

// One predictor step on the sched's backend. Reads/writes the CPU LSTM state
// (h0/c0/h1/c1, each [H]); pred_out (= top-layer hidden) is filled on return.
// All weight tensors are backend-resident (any dtype: F16/F32/quantized).
static inline void predictor_step(ggml_backend_sched_t sched, ggml_tensor* embed_w, ggml_tensor* l0_wih,
                                  ggml_tensor* l0_bih, ggml_tensor* l0_whh, ggml_tensor* l0_bhh, ggml_tensor* l1_wih,
                                  ggml_tensor* l1_bih, ggml_tensor* l1_whh, ggml_tensor* l1_bhh, int token_id, int H,
                                  std::vector<float>& h0, std::vector<float>& c0, std::vector<float>& h1,
                                  std::vector<float>& c1, std::vector<float>& pred_out) {
    const size_t mem = ggml_tensor_overhead() * 64 + ggml_graph_overhead();
    ggml_init_params ip = {mem, nullptr, true};
    ggml_context* cx = ggml_init(ip);

    ggml_tensor* tok = ggml_new_tensor_1d(cx, GGML_TYPE_I32, 1);
    ggml_set_input(tok);
    ggml_tensor* h0i = ggml_new_tensor_1d(cx, GGML_TYPE_F32, H);
    ggml_tensor* c0i = ggml_new_tensor_1d(cx, GGML_TYPE_F32, H);
    ggml_tensor* h1i = ggml_new_tensor_1d(cx, GGML_TYPE_F32, H);
    ggml_tensor* c1i = ggml_new_tensor_1d(cx, GGML_TYPE_F32, H);
    for (ggml_tensor* t : {h0i, c0i, h1i, c1i})
        ggml_set_input(t);

    ggml_tensor* emb = ggml_reshape_1d(cx, ggml_get_rows(cx, embed_w, tok), H);
    if (emb->type != GGML_TYPE_F32)
        emb = ggml_cast(cx, emb, GGML_TYPE_F32);

    ggml_tensor *c0o, *c1o;
    ggml_tensor* h0o = lstm_layer(cx, emb, l0_wih, l0_bih, l0_whh, l0_bhh, h0i, c0i, H, &c0o);
    ggml_tensor* h1o = lstm_layer(cx, h0o, l1_wih, l1_bih, l1_whh, l1_bhh, h1i, c1i, H, &c1o);
    for (ggml_tensor* t : {h0o, c0o, h1o, c1o})
        ggml_set_output(t);

    ggml_cgraph* gf = ggml_new_graph(cx);
    for (ggml_tensor* t : {h0o, c0o, h1o, c1o})
        ggml_build_forward_expand(gf, t);

    ggml_backend_sched_reset(sched);
    ggml_backend_sched_alloc_graph(sched, gf);
    int32_t tid = token_id;
    ggml_backend_tensor_set(tok, &tid, 0, sizeof(int32_t));
    ggml_backend_tensor_set(h0i, h0.data(), 0, H * sizeof(float));
    ggml_backend_tensor_set(c0i, c0.data(), 0, H * sizeof(float));
    ggml_backend_tensor_set(h1i, h1.data(), 0, H * sizeof(float));
    ggml_backend_tensor_set(c1i, c1.data(), 0, H * sizeof(float));
    ggml_backend_sched_graph_compute(sched, gf);

    h0.resize(H);
    c0.resize(H);
    h1.resize(H);
    c1.resize(H);
    ggml_backend_tensor_get(h0o, h0.data(), 0, H * sizeof(float));
    ggml_backend_tensor_get(c0o, c0.data(), 0, H * sizeof(float));
    ggml_backend_tensor_get(h1o, h1.data(), 0, H * sizeof(float));
    ggml_backend_tensor_get(c1o, c1.data(), 0, H * sizeof(float));
    pred_out = h1;
    ggml_free(cx);
}

// One joint step: logits = out_w·relu(proj_e + pred_w·pred_u + pred_b) + out_b.
// proj_e is [Jh] (precomputed encoder projection), pred_u is [H]. Vocab size is
// read from out_w->ne[1]. Weight tensors are backend-resident.
static inline void joint_step(ggml_backend_sched_t sched, ggml_tensor* pred_w, ggml_tensor* pred_b, ggml_tensor* out_w,
                              ggml_tensor* out_b, const float* proj_e, const float* pred_u, int Jh, int H,
                              std::vector<float>& logits) {
    const int Vt = (int)out_w->ne[1];
    const size_t mem = ggml_tensor_overhead() * 32 + ggml_graph_overhead();
    ggml_init_params ip = {mem, nullptr, true};
    ggml_context* cx = ggml_init(ip);

    ggml_tensor* pe = ggml_new_tensor_1d(cx, GGML_TYPE_F32, Jh);
    ggml_set_input(pe);
    ggml_tensor* pu = ggml_new_tensor_1d(cx, GGML_TYPE_F32, H);
    ggml_set_input(pu);

    ggml_tensor* mid = ggml_add(cx, ggml_mul_mat(cx, pred_w, pu), pred_b); // pred_w·pred_u + pred_b
    mid = ggml_relu(cx, ggml_add(cx, mid, pe));                            // relu(proj_e + mid)
    ggml_tensor* lg = ggml_add(cx, ggml_mul_mat(cx, out_w, mid), out_b);   // out_w·mid + out_b
    ggml_set_output(lg);

    ggml_cgraph* gf = ggml_new_graph(cx);
    ggml_build_forward_expand(gf, lg);

    ggml_backend_sched_reset(sched);
    ggml_backend_sched_alloc_graph(sched, gf);
    ggml_backend_tensor_set(pe, proj_e, 0, Jh * sizeof(float));
    ggml_backend_tensor_set(pu, pred_u, 0, H * sizeof(float));
    ggml_backend_sched_graph_compute(sched, gf);
    logits.resize(Vt);
    ggml_backend_tensor_get(lg, logits.data(), 0, Vt * sizeof(float));
    ggml_free(cx);
}

} // namespace core_rnnt_ggml
