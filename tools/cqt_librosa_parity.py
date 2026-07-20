#!/usr/bin/env python3
"""Score src/core/cqt.h against librosa's CQT.

Usage:
    python tools/cqt_librosa_parity.py <path-to-test-core-cqt-dump-binary>

The C++ side (tests/test-core-cqt.cpp --dump) writes a raw float32 magnitude
matrix for a deterministic test signal; this script computes librosa's CQT on
the identical signal and reports per-bin correlation.

Why correlation and not cosine-to-1.0: librosa's CQT uses recursive downsampling
with a different normalisation lineage, so an exact match is not the target. The
question BTC actually cares about is whether the two front ends rank bins the
same way and put energy in the same places. Report the numbers, do not paper
over them.
"""
import sys
import struct
import numpy as np

SR = 22050
FMIN = 32.703195662574829  # C1
N_BINS = 144
BPO = 24
HOP = 2048


def test_signal(n):
    """Deterministic: three sustained tones an octave apart + a chirp tail.
    Must match tests/test-core-cqt.cpp exactly."""
    t = np.arange(n) / SR
    x = np.zeros(n, dtype=np.float64)
    third = n // 3
    for i, f in enumerate((130.8127826502993, 261.6255653005986, 523.2511306011972)):
        lo, hi = i * third, (i + 1) * third if i < 2 else n
        x[lo:hi] = 0.5 * np.sin(2 * np.pi * f * t[lo:hi])
    return x.astype(np.float32)


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: cqt_librosa_parity.py <dump.bin>")
    import librosa

    with open(sys.argv[1], "rb") as fh:
        T, K = struct.unpack("<ii", fh.read(8))
        mine = np.frombuffer(fh.read(T * K * 4), dtype=np.float32).reshape(T, K)
    print(f"c++  : {T} frames x {K} bins")

    x = test_signal(T * HOP)
    ref = np.abs(librosa.cqt(y=x.astype(np.float64), sr=SR, fmin=FMIN, n_bins=N_BINS,
                             bins_per_octave=BPO, hop_length=HOP)).T  # (frames, bins)
    print(f"librosa: {ref.shape[0]} frames x {ref.shape[1]} bins")

    n = min(T, ref.shape[0])
    a, b = mine[:n], ref[:n]

    # Per-frame correlation over the bin axis: does the spectral SHAPE agree?
    cors = []
    for i in range(n):
        va, vb = a[i], b[i]
        if va.std() < 1e-12 or vb.std() < 1e-12:
            continue
        cors.append(float(np.corrcoef(va, vb)[0, 1]))
    cors = np.array(cors)

    # Does the peak bin agree? That is what a chord/pitch model keys on.
    peak_match = float(np.mean(a.argmax(1) == b.argmax(1)))
    # ...and within one bin (half a semitone at bpo=24)?
    peak_close = float(np.mean(np.abs(a.argmax(1).astype(int) - b.argmax(1).astype(int)) <= 1))

    print(f"\nper-frame shape correlation: mean {cors.mean():.4f}  min {cors.min():.4f}  "
          f"median {np.median(cors):.4f}  (n={len(cors)})")
    print(f"peak-bin exact match : {peak_match:.1%}")
    print(f"peak-bin within +/-1 : {peak_close:.1%}")

    # Where the expected tones land, as an absolute sanity check.
    for f in (130.8127826502993, 261.6255653005986, 523.2511306011972):
        k = int(round(BPO * np.log2(f / FMIN)))
        print(f"  {f:8.2f} Hz -> bin {k:3d}")

    ok = cors.mean() > 0.9 and peak_close > 0.9
    print("\n" + ("PASS" if ok else "FAIL") + " (shape corr > 0.9 and peak within 1 bin > 90%)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
