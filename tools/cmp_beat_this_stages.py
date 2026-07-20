#!/usr/bin/env python3
"""Score a beat-this C++ stage dump against the torch reference (§251b-1).

    python tools/cmp_beat_this_stages.py <stage_cpp.bin> <bt_stages.npz> <stage>

LAYOUT: the reference is torch (b, c, f, t); ggml is reversed, so the C++ dump
is (t, f, c). This script transposes before comparing — a layout mismatch here
would masquerade as a numerical bug, which is exactly the trap worth avoiding.

Prints |mine| and |ref| per stage as well as cosine: a 10-30x magnitude outlier
on either side says "same name, wrong data" immediately, whereas cosine alone
can look like plausible drift.
"""
import struct
import sys

import numpy as np


def main():
    cpp_path, npz_path, stage = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(cpp_path, "rb") as f:
        t, fq, c, _ = struct.unpack("<iiii", f.read(16))
        mine = np.frombuffer(f.read(t * fq * c * 4), dtype=np.float32)
    mine = mine.reshape(c, fq, t)          # ggml ne (t, fq, c) -> numpy reversed
    ref = np.load(npz_path)[stage]         # torch (b, c, f, t)
    ref = ref[0]                            # -> (c, f, t)
    print(f"  cpp (c,f,t)={mine.shape}   ref (c,f,t)={ref.shape}")
    if mine.shape != ref.shape:
        print("  SHAPE MISMATCH")
        return 1
    a, b = mine.ravel(), ref.ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    print(f"  cos={cos:.8f}  max_abs={np.abs(a-b).max():.4e}  "
          f"|mine|={np.linalg.norm(a):.4f} |ref|={np.linalg.norm(b):.4f}")
    ok = cos > 0.9999
    print("  " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
