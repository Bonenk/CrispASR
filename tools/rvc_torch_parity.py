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
