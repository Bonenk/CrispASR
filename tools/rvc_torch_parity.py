#!/usr/bin/env python3
"""Executable spec + reference dumper for the RVC voice-conversion port (§CB1).

    python tools/rvc_torch_parity.py <rvc.gguf> <f0G*.pth> <configs/vN/<sr>.json> \
        <RVC-repo-dir> [ref-dump.gguf]

WHY THIS EXISTS BEFORE ANY C++. RVC inference is STOCHASTIC, so the usual
"generate audio and compare waveforms" acceptance test is invalid — two runs of
the reference disagree with each other. Everything downstream of the latent
sample is unverifiable unless the noise is REPLAYED. This tool proves the
replay design works, and only then dumps a reference the C++ diff can use.

Two live RNG sites (docs/music-transcription/RVC_BLUEPRINT.md §2):

  A. z_p = (m_p + exp(logs_p) * randn_like(m_p) * 0.66666) * x_mask   models.py:684
  B. SineGen additive noise, voicing-dependent                        models.py:358

Site B's *phase* term is NOT a third site: GeneratorNSF hardcodes
harmonic_num=0, so SineGen.dim == 1 and `rand_ini[:, 0] = 0` zeroes the only
element. Verified here rather than asserted — see check_phase_is_deterministic.

CometBeat's offline rvc.dart exposes the same z_p noise as an injectable
`rnd [1,192,T]` input (cos 0.99994 vs the ONNX graph), so once this tool and
the ggml graph both consume an injected buffer, a three-way deterministic
harness is possible: this reference -> their Dart -> our ggml, identical noise.
"""

import json
import sys
from pathlib import Path

import numpy as np


def make_injector(buffers):
    """Replace torch's RNG with deterministic draws from `buffers`.

    Returns a context manager. Each call to the patched randn_like/rand pops the
    next buffer whose shape matches, so the SAME sequence is reproducible across
    implementations — which seeding alone cannot give us, since different RNGs
    produce different numbers from the same seed. This mirrors how the RVC ONNX
    export exposes its `rnd` input.
    """
    import torch

    class _Injector:
        def __init__(self):
            self.log = []          # (site, shape) in call order
            self._orig_randn_like = torch.randn_like
            self._orig_rand = torch.rand
            self._queue = list(buffers)

        def _next(self, shape, site):
            self.log.append((site, tuple(shape)))
            for i, b in enumerate(self._queue):
                if tuple(b.shape) == tuple(shape):
                    return torch.from_numpy(self._queue.pop(i).astype(np.float32))
            # Nothing supplied for this draw: use zeros and RECORD it, rather
            # than silently falling back to real randomness (which would make
            # the run non-reproducible again without telling anyone).
            self.log[-1] = (site + ":ZEROED", tuple(shape))
            return torch.zeros(*shape, dtype=torch.float32)

        def __enter__(self):
            inj = self

            def randn_like(t, *a, **k):
                return inj._next(t.shape, "randn_like")

            def rand(*size, **k):
                if len(size) == 1 and isinstance(size[0], (tuple, list)):
                    size = tuple(size[0])
                return inj._next(size, "rand")

            torch.randn_like = randn_like
            torch.rand = rand
            return self

        def __exit__(self, *exc):
            torch.randn_like = self._orig_randn_like
            torch.rand = self._orig_rand
            return False

    return _Injector()


def build_model(ckpt_path, cfg_path, repo_dir):
    import torch
    sys.path.insert(0, str(repo_dir))
    from infer.module.models import SynthesizerTrnMs768NSFsid, SynthesizerTrnMs256NSFsid

    cfg = json.load(open(cfg_path))
    m, d = cfg["model"], cfg["data"]
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck

    content_dim = sd["enc_p.emb_phone.weight"].shape[1]
    n_spk = sd["emb_g.weight"].shape[0]
    Klass = SynthesizerTrnMs768NSFsid if content_dim == 768 else SynthesizerTrnMs256NSFsid

    net = Klass(
        d["filter_length"] // 2 + 1,
        m.get("segment_size", 12800) // int(np.prod(m["upsample_rates"])),
        m["inter_channels"], m["hidden_channels"], m["filter_channels"],
        m["n_heads"], m["n_layers"], m["kernel_size"], m["p_dropout"],
        m["resblock"], m["resblock_kernel_sizes"], m["resblock_dilation_sizes"],
        m["upsample_rates"], m["upsample_initial_channel"], m["upsample_kernel_sizes"],
        n_spk, m["gin_channels"], d["sampling_rate"],
        is_half=False,   # required kwarg; fp32 throughout for a clean reference
    )
    # enc_q is training-only; the generator checkpoint has it but infer() never
    # touches it, so a strict load would fail for the wrong reason.
    missing, unexpected = net.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if not k.startswith("enc_q.")]
    if real_missing:
        raise RuntimeError(
            f"{len(real_missing)} weight(s) missing -> the model would be partly random, "
            f"and a reference dumped from it would silently disagree with everything. "
            f"First few: {real_missing[:5]}"
        )
    net.eval()
    return net, cfg, content_dim


def check_phase_is_deterministic(net):
    """Confirm SineGen's random initial phase is structurally zero.

    The blueprint claims harmonic_num == 0 makes rand_ini a 1-element tensor
    that the next line zeroes. Assert it against the built model instead of
    trusting the reading.
    """
    sg = net.dec.m_source.l_sin_gen
    assert sg.harmonic_num == 0, f"harmonic_num={sg.harmonic_num}, expected 0"
    assert sg.dim == 1, f"SineGen.dim={sg.dim}, expected 1"
    print(f"  SineGen: harmonic_num={sg.harmonic_num} dim={sg.dim} -> rand_ini is 1 element, zeroed => phase deterministic")


def main():
    if len(sys.argv) not in (5, 6):
        sys.exit(__doc__)
    gguf_path, ckpt, cfg_path, repo = sys.argv[1:5]
    import torch

    net, cfg, content_dim = build_model(ckpt, cfg_path, repo)
    m, d = cfg["model"], cfg["data"]
    upp = int(np.prod(m["upsample_rates"]))
    sr = int(d["sampling_rate"])
    inter = m["inter_channels"]
    print(f"rvc spec: content_dim={content_dim} inter={inter} sr={sr} upp={upp} ({sr/upp:.0f} fps)")
    check_phase_is_deterministic(net)

    # Deterministic inputs. T frames at 100 Hz.
    rng = np.random.default_rng(0)
    T = 64
    phone = rng.standard_normal((1, T, content_dim)).astype(np.float32) * 0.1
    f0_hz = np.abs(rng.standard_normal((1, T)).astype(np.float32)) * 120.0 + 100.0
    f0_hz[0, T // 3 : T // 3 + 6] = 0.0  # an unvoiced stretch: exercises the uv branch
    pitch_coarse = coarse_pitch(f0_hz).astype(np.int64)
    sid = np.array([0], dtype=np.int64)

    # The injected noise. Site A is (1, inter, T); site B is sized by SineGen.
    noise_zp = rng.standard_normal((1, inter, T)).astype(np.float32)
    noise_sine = rng.standard_normal((1, T * upp, 1)).astype(np.float32)

    def run():
        with make_injector([noise_zp.copy(), noise_sine.copy()]) as inj, torch.no_grad():
            o, x_mask, (z, z_p, m_p, logs_p) = net.infer(
                torch.from_numpy(phone), torch.tensor([T]),
                torch.from_numpy(pitch_coarse), torch.from_numpy(f0_hz),
                torch.from_numpy(sid),
            )
        return o.numpy(), dict(z=z.numpy(), z_p=z_p.numpy(), m_p=m_p.numpy(), logs_p=logs_p.numpy()), inj.log

    a, sa, log1 = run()
    b, sb, log2 = run()

    print("  RNG draws intercepted (site, shape), in call order:")
    for site, shape in log1:
        print(f"    {site:22} {shape}")

    same = np.array_equal(a, b)
    print(f"\nDETERMINISM CHECK: two runs with identical injected noise -> "
          f"{'BIT-IDENTICAL' if same else 'DIFFER'}  max_abs={np.abs(a-b).max():.3e}")
    if not same:
        sys.exit(
            "FAIL: injection did not make inference deterministic. Some RNG site is "
            "still unpatched — find it before building anything on this."
        )
    zeroed = [s for s, _ in log1 if s.endswith(":ZEROED")]
    if zeroed:
        print(f"NOTE: {len(zeroed)} draw(s) fell back to zeros (no buffer of that shape supplied): {set(zeroed)}")

    # ---- numpy spec vs torch, for enc_p ----
    from gguf import GGUFReader
    G = {t.name: np.array(t.data, dtype=np.float32).reshape([int(d) for d in reversed(t.shape)])
         for t in GGUFReader(gguf_path).tensors}
    mp_np, logs_np = enc_p_numpy(
        G, phone[0], pitch_coarse[0], m["hidden_channels"], m["n_heads"], m["n_layers"],
        window=(G["enc_p.encoder.attn_layers.0.emb_rel_k"].shape[1] - 1) // 2,
        out_channels=inter,
    )
    print("\nNUMPY SPEC vs TORCH (enc_p):")
    ok = True
    for name, mine, ref in (("m_p", mp_np, sa["m_p"][0]), ("logs_p", logs_np, sa["logs_p"][0])):
        a, b = mine.ravel(), ref.ravel().astype(np.float64)
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        mad = float(np.abs(a - b).max())
        good = cos > 0.99999
        ok &= good
        print(f"  {name:8} {'PASS' if good else 'FAIL'} cos={cos:.8f} max_abs={mad:.3e} "
              f"|mine|={np.linalg.norm(a):.4f} |ref|={np.linalg.norm(b):.4f}")
    # ---- flow (reverse) ----
    g_emb = G["emb_g.weight"][0][:, None].astype(np.float64)  # sid 0
    # (5, 1, 3) = kernel_size, dilation_rate, n_layers -- HARDCODED in
    # models.py:624 (ResidualCouplingBlock(inter, hidden, 5, 1, 3, ...)), not
    # config-derived, so these hold for every checkpoint.
    z_np = flow_numpy(G, sa["z_p"][0], g_emb, n_flows=4, hidden=m["hidden_channels"],
                      n_layers=3, kernel_size=5, dilation_rate=1)
    a, b = z_np.ravel(), sa["z"][0].ravel().astype(np.float64)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    good = cos > 0.99999
    ok &= good
    print(f"NUMPY SPEC vs TORCH (flow, reverse):\n  {'z':8} {'PASS' if good else 'FAIL'} "
          f"cos={cos:.8f} max_abs={np.abs(a-b).max():.3e} |mine|={np.linalg.norm(a):.4f} |ref|={np.linalg.norm(b):.4f}")

    if not ok:
        sys.exit("FAIL: a numpy spec does not match torch — fix the spec before any ggml.")

    if len(sys.argv) > 5:
        import gguf as _g
        stages = {"input_phone": phone[0], "input_f0": f0_hz, "input_pitch": pitch_coarse.astype(np.float32),
                  "noise_zp": noise_zp, "noise_sine": noise_sine,
                  "m_p": sa["m_p"][0], "logs_p": sa["logs_p"][0], "z_p": sa["z_p"][0], "z": sa["z"][0],
                  "output_audio": a[0]}
        w = _g.GGUFWriter(sys.argv[5], "rvc-ref")
        for k, v in stages.items():
            w.add_tensor(k, np.ascontiguousarray(v, dtype=np.float32))
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
        print(f"wrote reference dump {sys.argv[5]}: {len(stages)} stages "
              f"(inputs + BOTH noise buffers + latents + audio)")
    print("PASS")


# ---------------------------------------------------------------------------
# numpy reimplementation — enc_p (TextEncoder)
#
# Traps this encodes, each a silent accuracy bug if assumed:
#   * LeakyReLU slope is 0.1, NOT torch's 0.01 default (models.py:38).
#   * x is scaled by sqrt(hidden_channels) BEFORE the lrelu (models.py:62-63).
#   * Residuals are POST-norm: x = norm(x + sublayer(x)), not pre-norm.
#   * Attention is RELATIVE-position (window 10), not absolute PE.
#   * FFN convs use SAME padding and a plain ReLU (activation is not "gelu"),
#     and the output is re-masked.
#   * LayerNorm is over the CHANNEL dim (modules.py:29-32 transposes first).
# ---------------------------------------------------------------------------

def _layer_norm(x, gamma, beta, eps=1e-5):
    """x: (C, T) -> normalise over C per time step (their LayerNorm transposes)."""
    xt = x.T                                    # (T, C)
    mu = xt.mean(-1, keepdims=True)
    var = xt.var(-1, keepdims=True)
    y = (xt - mu) / np.sqrt(var + eps) * gamma + beta
    return y.T


def _conv1d(x, w, b, pad):
    """x: (Cin, T), w: (Cout, Cin, K) -> (Cout, T) with SAME zero padding."""
    Cin, T = x.shape
    Cout, _, K = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad)))
    out = np.zeros((Cout, T), dtype=np.float64)
    for k in range(K):
        out += w[:, :, k] @ xp[:, k : k + T]
    return out + b[:, None]


def _get_relative_embeddings(emb, T, window):
    """emb: (1, 2w+1, d) -> (2T-1, d), padded/sliced exactly as upstream."""
    pad_len = max(T - (window + 1), 0)
    start = max((window + 1) - T, 0)
    e = emb[0]
    if pad_len > 0:
        e = np.pad(e, ((pad_len, pad_len), (0, 0)))
    return e[start : start + 2 * T - 1]


def _rel_to_abs(x):
    """x: (H, T, 2T-1) -> (H, T, T). The skew from relative to absolute indexing."""
    H, T, _ = x.shape
    x = np.pad(x, ((0, 0), (0, 0), (0, 1)))          # (H, T, 2T)
    flat = x.reshape(H, T * 2 * T)
    flat = np.pad(flat, ((0, 0), (0, T - 1)))        # (H, T*2T + T-1)
    return flat.reshape(H, T + 1, 2 * T - 1)[:, :T, T - 1 :]


def enc_p_numpy(G, phone, pitch, hidden, n_heads, n_layers, window, out_channels):
    """phone: (T, content_dim), pitch: (T,) int -> (m_p, logs_p), each (out, T)."""
    W = lambda k: G[k].astype(np.float64)
    x = phone.astype(np.float64) @ W("enc_p.emb_phone.weight").T + W("enc_p.emb_phone.bias")
    x = x + W("enc_p.emb_pitch.weight")[pitch]          # embedding lookup
    x = x * np.sqrt(hidden)                             # BEFORE the lrelu
    x = np.where(x < 0, 0.1 * x, x)                     # LeakyReLU(0.1), not 0.01
    x = x.T                                             # (C, T)

    hd = hidden // n_heads
    for i in range(n_layers):
        p = f"enc_p.encoder.attn_layers.{i}."
        q = _conv1d(x, W(p + "conv_q.weight"), W(p + "conv_q.bias"), 0)
        k = _conv1d(x, W(p + "conv_k.weight"), W(p + "conv_k.bias"), 0)
        v = _conv1d(x, W(p + "conv_v.weight"), W(p + "conv_v.bias"), 0)
        T = x.shape[1]
        qh = q.reshape(n_heads, hd, T).transpose(0, 2, 1)   # (H, T, hd)
        kh = k.reshape(n_heads, hd, T).transpose(0, 2, 1)
        vh = v.reshape(n_heads, hd, T).transpose(0, 2, 1)
        scores = (qh / np.sqrt(hd)) @ kh.transpose(0, 2, 1)
        rel_k = _get_relative_embeddings(W(p + "emb_rel_k"), T, window)
        scores = scores + _rel_to_abs((qh / np.sqrt(hd)) @ rel_k.T)
        e = np.exp(scores - scores.max(-1, keepdims=True))
        attn = e / e.sum(-1, keepdims=True)
        out = attn @ vh                                     # (H, T, hd)
        # relative VALUES: upstream adds them via the inverse skew
        rel_v = _get_relative_embeddings(W(p + "emb_rel_v"), T, window)
        out = out + _abs_to_rel(attn) @ rel_v
        out = out.transpose(0, 2, 1).reshape(hidden, T)
        y = _conv1d(out, W(p + "conv_o.weight"), W(p + "conv_o.bias"), 0)
        n1 = f"enc_p.encoder.norm_layers_1.{i}."
        x = _layer_norm(x + y, W(n1 + "gamma"), W(n1 + "beta"))     # POST-norm

        f = f"enc_p.encoder.ffn_layers.{i}."
        w1 = W(f + "conv_1.weight")
        h = _conv1d(x, w1, W(f + "conv_1.bias"), (w1.shape[2] - 1) // 2)
        h = np.maximum(h, 0.0)                                       # plain ReLU
        w2 = W(f + "conv_2.weight")
        y = _conv1d(h, w2, W(f + "conv_2.bias"), (w2.shape[2] - 1) // 2)
        n2 = f"enc_p.encoder.norm_layers_2.{i}."
        x = _layer_norm(x + y, W(n2 + "gamma"), W(n2 + "beta"))

    stats = _conv1d(x, W("enc_p.proj.weight"), W("enc_p.proj.bias"), 0)
    return stats[:out_channels], stats[out_channels:]


# ---------------------------------------------------------------------------
# numpy reimplementation — flow (ResidualCouplingBlock, REVERSE pass)
#
# Traps:
#   * mean_only=True (models.py), so `logs` is ZERO and the coupling is purely
#     ADDITIVE. The reverse is x1 = (x1 - m), not (x1 - m) * exp(-logs).
#   * `flows` interleaves [Coupling, Flip] x 4; the reverse pass walks the whole
#     list backwards, so Flip comes FIRST.
#   * Flip reverses the CHANNEL axis (torch.flip(x, [1])).
#   * The WaveNet is gated: tanh(first half) * sigmoid(second half) of
#     (x_in + g_l), with the speaker conditioning sliced per layer.
#   * Dilated convs with padding = (k*d - d)/2, i.e. SAME for odd k.
# ---------------------------------------------------------------------------

def _wn_numpy(G, prefix, x, g, hidden, n_layers, kernel_size, dilation_rate):
    """WaveNet residual stack. x: (hidden, T), g: (gin, 1) speaker embedding."""
    W = lambda k: G[k].astype(np.float64)
    T = x.shape[1]
    output = np.zeros_like(x)
    # cond_layer projects g once to 2*hidden*n_layers, then each layer slices it.
    gc = W(prefix + "cond_layer.weight")[:, :, 0] @ g[:, 0] + W(prefix + "cond_layer.bias")
    for i in range(n_layers):
        d = dilation_rate ** i
        pad = int((kernel_size * d - d) / 2)
        w = W(prefix + f"in_layers.{i}.weight")
        b = W(prefix + f"in_layers.{i}.bias")
        xp = np.pad(x, ((0, 0), (pad, pad)))
        x_in = np.zeros((w.shape[0], T), dtype=np.float64)
        for k in range(w.shape[2]):
            x_in += w[:, :, k] @ xp[:, k * d : k * d + T]
        x_in += b[:, None]
        g_l = gc[i * 2 * hidden : (i + 1) * 2 * hidden][:, None]
        in_act = x_in + g_l
        acts = np.tanh(in_act[:hidden]) * (1.0 / (1.0 + np.exp(-in_act[hidden:])))
        rw = W(prefix + f"res_skip_layers.{i}.weight")[:, :, 0]
        rb = W(prefix + f"res_skip_layers.{i}.bias")
        rs = rw @ acts + rb[:, None]
        if i < n_layers - 1:
            x = x + rs[:hidden]
            output = output + rs[hidden:]
        else:
            output = output + rs
    return output


def flow_numpy(G, z_p, g, n_flows, hidden, n_layers, kernel_size, dilation_rate):
    """Reverse pass. z_p: (C, T) -> z: (C, T)."""
    W = lambda k: G[k].astype(np.float64)
    x = z_p.astype(np.float64)
    half = x.shape[0] // 2
    # flows = [Coupling, Flip] * n_flows; reverse traversal hits Flip first.
    for idx in range(n_flows - 1, -1, -1):
        x = x[::-1]                                     # Flip: reverse channels
        p = f"flow.flows.{idx * 2}."
        x0, x1 = x[:half], x[half:]
        h = W(p + "pre.weight")[:, :, 0] @ x0 + W(p + "pre.bias")[:, None]
        h = _wn_numpy(G, p + "enc.", h, g, hidden, n_layers, kernel_size, dilation_rate)
        m = W(p + "post.weight")[:, :, 0] @ h + W(p + "post.bias")[:, None]
        x1 = x1 - m                                     # mean_only => no exp(-logs)
        x = np.concatenate([x0, x1], axis=0)
    return x


def _abs_to_rel(x):
    """(H, T, T) -> (H, T, 2T-1); inverse of _rel_to_abs, for relative values."""
    H, T, _ = x.shape
    x = np.pad(x, ((0, 0), (0, 0), (0, T - 1)))
    flat = x.reshape(H, T * (2 * T - 1))
    flat = np.pad(flat, ((0, 0), (T, 0)))
    return flat.reshape(H, T, 2 * T)[:, :, 1:]


def coarse_pitch(f0):
    """f0 (Hz) -> coarse 1..255, exactly as pipeline.py:73-137."""
    f0_min, f0_max = 50.0, 1100.0
    mel_min = 1127 * np.log(1 + f0_min / 700)
    mel_max = 1127 * np.log(1 + f0_max / 700)
    mel = 1127 * np.log(1 + f0 / 700)
    mel = np.where(mel > 0, (mel - mel_min) * 254 / (mel_max - mel_min) + 1, mel)
    mel = np.where(mel <= 1, 1, mel)
    mel = np.where(mel > 255, 255, mel)
    return np.rint(mel)


if __name__ == "__main__":
    main()
