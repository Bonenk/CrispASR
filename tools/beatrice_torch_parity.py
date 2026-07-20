#!/usr/bin/env python3
"""Dump Beatrice per-stage reference intermediates to a -ref.gguf.

    python tools/beatrice_torch_parity.py \
        --component pitch_estimator --model 104_3_checkpoint_00300000.pt \
        --audio samples/jfk.wav --output beatrice-pitch-ref.gguf \
        --trainer-path <dir containing beatrice_trainer>

This is the HARD RULE #2 artifact: the ggml port is validated stage by stage
against these, earliest layer first, NOT by comparing final output. See
docs/music-transcription/BEATRICE_BLUEPRINT.md.

The forward pass is RE-IMPLEMENTED here step by step rather than captured with
hooks, for two reasons:

  1. It is the executable spec. If my reading of the architecture is wrong, the
     final-output assertion below fails and I find out immediately -- before any
     C++ is written.
  2. register_forward_hook is a known trap in this codebase: it does not fire
     when a module's .forward() is called directly, and during the RVC port it
     silently produced a harness that compared NOTHING while reporting
     "0 FAILED". Beatrice compounds that -- its VectorQuantizer is itself
     installed as a forward hook, so hook-based capture would interact with the
     model's own hooks.

The reimplementation is checked against the module's real forward() at the end;
if they disagree the tool refuses to write a reference file, because a wrong
reference is worse than none -- it makes a broken port look correct.

LAYOUT. Torch stores [batch, channels, time] with TIME fastest. Tensors are
dumped in exactly that memory order, which lands in ggml as ne = [time,
channels] -- the layout ggml_conv_1d expects. Do NOT transpose on the way in or
out. Three separate bugs during the RVC port came from transposing here: the
per-stage cosines went to ~0 on graphs that were in fact correct.
"""

import argparse
import sys

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    import torchaudio
except ImportError:
    sys.exit("error: pip install torch torchaudio")
try:
    from gguf import GGUFWriter
except ImportError:
    sys.exit("error: pip install gguf")


class Dump:
    def __init__(self):
        self.stages = {}

    def __call__(self, name, t):
        a = t.detach().cpu().float().numpy()
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]                      # drop batch; keep [channels, time]
        self.stages[name] = np.ascontiguousarray(a)
        return t


def cos(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def spec_pitch_estimator(m, wav, d):
    """Step-by-step PitchEstimator.forward, dumping every boundary."""
    d("input_wav", wav)

    # --- DSP front end (extract_pitch_features)
    instfreq, corr_diff, energy = m_extract(m, wav, d)

    # --- two embedding branches. GELU is the TANH approximation, not erf.
    x_if = F.gelu(m.instfreq_embed_0(instfreq), approximate="tanh")
    d("instfreq_embed_0_gelu", x_if)
    x_if = m.instfreq_embed_1(x_if)
    d("instfreq_embed_1", x_if)

    x_c = F.gelu(m.corr_embed_0(corr_diff), approximate="tanh")
    d("corr_embed_0_gelu", x_c)
    x_c = m.corr_embed_1(x_c)
    d("corr_embed_1", x_c)

    x = F.gelu(x_if + x_c, approximate="tanh")
    d("branch_sum_gelu", x)

    # --- ConvNeXtStack
    bb = m.backbone
    x = bb.embed(x)
    d("backbone_embed", x)
    # stack-level LayerNorm KEEPS its affine (merge_weights does not fold it)
    x = bb.norm(x.transpose(1, 2)).transpose(1, 2)
    d("backbone_norm", x)

    for i, blk in enumerate(bb.convnext):
        identity = x
        h = blk.dwconv(x)                       # depthwise, strictly causal
        d(f"block{i}_dwconv", h)
        h = h.transpose(1, 2)
        h = blk.norm(h)                         # affine folded away -> normalise only
        h = blk.pwconv1(h)
        h = F.gelu(h, approximate="tanh")
        h = blk.pwconv2(h)
        # gamma / pre_scale / post_scale / post_scale_weight are all identically
        # 1.0 post-merge and are deliberately NOT applied here.
        h = h.transpose(1, 2)
        x = h + identity
        d(f"block{i}_out", x)

    x = bb.final_layer_norm(x.transpose(1, 2)).transpose(1, 2)
    d("backbone_final_norm", x)

    logits = m.head(x)
    d("logits", logits)
    d("energy", energy)

    quantized = m.sample_pitch(logits.clone())
    d("quantized_pitch", quantized.float())
    return logits, energy


def m_extract(m, wav, d):
    """extract_pitch_features, dumped. Constants are the function's defaults."""
    hop, win = 160, 560
    max_corr_period, corr_win_length, cutoff = 256, 304, 64
    assert max_corr_period + corr_win_length == win

    y = wav.squeeze(1)
    pad = (win - hop) // 2
    y = F.pad(y, (pad, pad))
    frames = y.unfold(-1, win, hop).transpose(-2, -1)
    d("dsp_frames", frames)

    spec = torch.fft.rfft(frames, n=win, dim=-2)[..., :cutoff, :]
    log_power = spec.abs().add(1e-5).log10()
    d("dsp_log_power_spec", log_power)

    delta = spec[..., :, 1:] * spec[..., :, :-1].conj()
    delta = delta / delta.abs().add(1e-5)
    delta = torch.cat([torch.zeros_like(delta[..., :, :1]), delta], dim=-1)
    instfreq = torch.cat([log_power, delta.real, delta.imag], dim=-2)
    d("dsp_instfreq_features", instfreq)

    # autocorrelation via FFT of the FLIPPED frames (that is what makes it a
    # correlation rather than a convolution -- there is no conj() here)
    flipped = frames.flip((-2,))
    a = torch.fft.rfft(flipped, n=win, dim=-2)
    b = torch.fft.rfft(frames[..., -corr_win_length:, :], n=win, dim=-2)
    corr = torch.fft.irfft(a * b, n=win, dim=-2)[..., corr_win_length:, :]
    d("dsp_corr", corr)

    energy_c = flipped.square().cumsum(-2)
    energy0 = energy_c[..., corr_win_length - 1 : corr_win_length, :]
    energy_w = energy_c[..., corr_win_length:, :] - energy_c[..., :-corr_win_length, :]
    corr_diff = (energy0 + energy_w) - corr * 2.0
    assert float(corr_diff.min()) >= -1e-3, float(corr_diff.min())
    corr_diff = corr_diff.clamp(min=0.0) * (2.0 / corr_win_length)
    corr_diff = corr_diff.sqrt()
    d("dsp_corr_diff", corr_diff)

    win_cos = torch.signal.windows.cosine(win, device=y.device)[..., None]
    energy = (frames * win_cos).square().sum(-2, keepdim=True)
    energy = energy.clamp(min=1e-3).log10() * 0.5
    d("dsp_energy", energy)
    return instfreq, corr_diff, energy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--component", default="pitch_estimator", choices=["pitch_estimator"])
    ap.add_argument("--audio", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--trainer-path")
    ap.add_argument("--max-seconds", type=float, default=4.0)
    args = ap.parse_args()

    if args.trainer_path:
        sys.path.insert(0, args.trainer_path)
    import beatrice_trainer.__main__ as bt

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    sd = ck[args.component]
    m = bt.PitchEstimator(
        pitch_bins=sd["head.weight"].shape[0],
        channels=sd["head.weight"].shape[1],
        n_blocks=1 + max(int(k.split(".")[2]) for k in sd if k.startswith("backbone.convnext.")),
    )
    m.load_state_dict(sd)
    m.eval()
    # Fuse exactly as the converter does, so the reference describes the SHIPPED
    # weights. Dumping pre-merge intermediates would compare the port against a
    # model it is not running.
    m.merge_weights()

    wav, sr = torchaudio.load(args.audio)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    n = int(args.max_seconds * 16000)
    wav = wav[:, :n][None]  # [1, 1, T]

    d = Dump()
    with torch.inference_mode():
        logits, energy = spec_pitch_estimator(m, wav, d)
        ref_logits, ref_energy = m(wav)

    # THE GATE. If the step-by-step spec disagrees with the module's own
    # forward, this reference is wrong and must not be written.
    #
    # The bound is on RELATIVE MAX-ABS, not cosine. Both arms are torch running
    # the same weights, so the control is exactly bit-identical (max_abs 0.0) and
    # anything above f32 rounding is a real discrepancy. Cosine is far too blunt
    # here -- measured: swapping the tanh-approximate GELU for the exact erf one
    # (a genuine bug, blueprint detail 2) still scores cos=0.9999996, which sails
    # through any "cos > 0.999999" check while carrying max_abs 2.4e-02. That is
    # HARD RULE #2b in the concrete.
    # FINITENESS FIRST. Every NaN/Inf comparison is False, so a NaN spec sails
    # through any `rel > TOL` check and writes a reference file full of NaN while
    # exiting 0. Measured: dropping the 1e-5 in the delta_spec normalisation (a
    # divide-by-zero on silent frames) did exactly that.
    for name, a in d.stages.items():
        if not np.isfinite(a).all():
            n_bad = int((~np.isfinite(a)).sum())
            sys.exit(
                f"error: stage '{name}' contains {n_bad} non-finite values.\n"
                "       Refusing to write the reference. NOTE this cannot be caught "
                "by a tolerance check -- NaN compares False against everything."
            )

    c_log, c_en = cos(logits, ref_logits), cos(energy, ref_energy)
    peak = float(ref_logits.abs().max())
    mx = float((logits - ref_logits).abs().max())
    rel = mx / peak
    print(
        f"spec vs module forward(): logits cos={c_log:.10f} max_abs={mx:.3e} "
        f"rel={rel:.2e}  energy cos={c_en:.10f}"
    )
    TOL = 1e-6
    if rel > TOL or c_en < 0.999999:
        sys.exit(
            f"error: the step-by-step spec does NOT reproduce the module's forward() "
            f"(relative max-abs {rel:.2e} > {TOL:.0e}).\n"
            "       Refusing to write a reference file -- a wrong reference makes a "
            "broken port look correct.\n"
            "       NOTE cosine can look perfect here; check max_abs, not cos."
        )

    w = GGUFWriter(str(args.output), "beatrice-ref")
    w.add_string("beatrice.ref.component", args.component)
    w.add_string("beatrice.ref.audio", args.audio)
    w.add_uint32("beatrice.ref.n_samples", int(wav.shape[-1]))
    w.add_uint32("beatrice.ref.n_stages", len(d.stages))
    for name, a in d.stages.items():
        w.add_tensor(f"ref.{name}", a)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    print(f"wrote {args.output}: {len(d.stages)} stages")
    for name, a in d.stages.items():
        print(f"    ref.{name:<28} {tuple(a.shape)}")


if __name__ == "__main__":
    main()
