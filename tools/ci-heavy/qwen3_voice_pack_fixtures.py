#!/usr/bin/env python3
"""Generate small invalid/valid GGUFs for the #555 public loader checks.

These contain synthetic data and must never be used to claim speech quality.
"""
from pathlib import Path
import sys

import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType as Type


def write_fixtures(folder, dimension):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cases = ["good-two", "good-one", "wrong-dimension", "half-speaker", "float-codes",
             "missing-speaker", "missing-text", "duplicate-names", "wrong-groups", "extra-axis"]
    for case in cases:
        writer = GGUFWriter(str(folder / (case + ".gguf")), arch="qwen3tts.voicepack")
        names = ["first", "second"] if case == "good-two" else ["first"]
        if case == "duplicate-names":
            names = ["first", "first"]
        writer.add_array("voicepack.names", names)
        if case != "missing-text":
            writer.add_array("voicepack.ref_texts", ["Synthetic fixture."] * len(names))
        for name in dict.fromkeys(names):
            size = (1024 if dimension == 2048 else 2048) if case == "wrong-dimension" else dimension
            half = case == "half-speaker"
            if case != "missing-speaker":
                writer.add_tensor("voicepack.spk." + name + ".embd",
                                  np.zeros(size, dtype=np.float16 if half else np.float32),
                                  raw_dtype=Type.F16 if half else Type.F32)
            shape = (2, 16) if case != "wrong-groups" else (2, 8)
            if case == "extra-axis":
                shape = (2, 2, 16)
            float_codes = case == "float-codes"
            writer.add_tensor("voicepack.code." + name + ".codes",
                              np.zeros(shape, dtype=np.float32 if float_codes else np.int32),
                              raw_dtype=Type.F32 if float_codes else Type.I32)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()


if __name__ == "__main__":
    write_fixtures(sys.argv[1], int(sys.argv[2]))
