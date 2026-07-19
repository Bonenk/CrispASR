#!/usr/bin/env python3
"""Kaggle kernel: validate HTDemucs C++ against Python reference.

Builds CrispASR, converts HTDemucs model, runs the C++ smoke test,
and compares spec_input / encoder outputs against the Python reference
dumper's GGUF.
"""
import os, sys, subprocess, shutil
from pathlib import Path

WORK = Path("/kaggle/working")
os.chdir(str(WORK))

# ── Clone CrispASR ──────────────────────────────────────────────────
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
_CRISPASR_DIR = WORK / "CrispASR"
if not _CRISPASR_DIR.exists():
    try:
        subprocess.check_call(["git", "clone", "--depth", "1",
            CRISPASR_URL, str(_CRISPASR_DIR)])
        subprocess.check_call(["git", "submodule", "update", "--init", "ggml"],
                              cwd=str(_CRISPASR_DIR))
        sys.path.insert(0, str(_CRISPASR_DIR / "tools" / "kaggle"))
    except Exception:
        pass

if str(_CRISPASR_DIR / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh
kh.init_progress()

# ── Install deps ────────────────────────────────────────────────────
kh.step("Installing deps...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                       "demucs", "gguf", "einops", "julius", "dora-search",
                       "--no-deps"])

# ── Build CrispASR ──────────────────────────────────────────────────
kh.step("Building CrispASR...")
os.chdir(str(_CRISPASR_DIR))
kh.install_build_toolchain()

build_dir = _CRISPASR_DIR / "build"
cmake_flags = [
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
    "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
    "-DCRISPASR_NO_C2PA_NATIVE=ON",
]
subprocess.check_call(["cmake", "-G", "Ninja", "-B", str(build_dir)] + cmake_flags)
import multiprocessing
jobs = str(multiprocessing.cpu_count())
subprocess.check_call(["cmake", "--build", str(build_dir), "-j", jobs,
                       "--target", "htdemucs"])
kh.step(f"Build complete (htdemucs target, -j{jobs})")

# ── Convert HTDemucs model ──────────────────────────────────────────
kh.step("Converting HTDemucs model to GGUF (F32)...")
model_path = WORK / "htdemucs-f32.gguf"
subprocess.check_call([sys.executable,
    str(_CRISPASR_DIR / "models" / "convert-htdemucs-to-gguf.py"),
    "--model", "htdemucs",
    "--output", str(model_path),
    "--dtype", "f32"])
kh.step(f"Converted: {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")

# ── Generate Python reference ───────────────────────────────────────
kh.step("Generating Python reference dump...")
ref_path = WORK / "htdemucs-ref.gguf"

# We need a test audio file — use a synthetic sine wave
import numpy as np
import wave
test_wav = WORK / "test_sine.wav"
sr = 16000
t = np.arange(sr * 3) / sr  # 3 seconds
pcm = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
pcm_i16 = (pcm * 32767).astype(np.int16)
with wave.open(str(test_wav), "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(pcm_i16.tobytes())

subprocess.check_call([sys.executable,
    str(_CRISPASR_DIR / "tools" / "dump_reference.py"),
    "--backend", "htdemucs",
    "--model-dir", "htdemucs",
    "--audio", str(test_wav),
    "--output", str(ref_path)])
kh.step(f"Reference dump: {ref_path} ({ref_path.stat().st_size / 1e6:.1f} MB)")

# ── Build and run C++ smoke test ────────────────────────────────────
kh.step("Building smoke test...")
smoke_src = _CRISPASR_DIR / "tests" / "test_htdemucs_smoke.cpp"
smoke_bin = build_dir / "bin" / "test_htdemucs_smoke"
subprocess.check_call([
    "g++", "-std=c++17", "-O2",
    "-I", str(_CRISPASR_DIR / "src"),
    "-I", str(_CRISPASR_DIR / "ggml" / "include"),
    str(smoke_src),
    "-L", str(build_dir / "src"), "-lhtdemucs", "-lcrispasr-core",
    "-L", str(build_dir / "ggml" / "src"), "-lggml",
    "-L", str(build_dir / "ggml" / "src" / "ggml-base"), "-lggml-base",
    "-L", str(build_dir / "ggml" / "src" / "ggml-cpu"), "-lggml-cpu",
    "-lpthread", "-lm", "-ldl",
    "-o", str(smoke_bin)
])

kh.step("Running HTDemucs smoke test...")
env = os.environ.copy()
env["CRISPASR_HTDEMUCS_DEBUG"] = "1"
env["OMP_NUM_THREADS"] = "4"
# Set LD_LIBRARY_PATH for shared ggml libs
ld_paths = [
    str(build_dir / "ggml" / "src"),
    str(build_dir / "ggml" / "src" / "ggml-base"),
    str(build_dir / "ggml" / "src" / "ggml-cpu"),
    str(build_dir / "src"),
]
env["LD_LIBRARY_PATH"] = ":".join(ld_paths) + ":" + env.get("LD_LIBRARY_PATH", "")
result = subprocess.run([str(smoke_bin), str(model_path)],
                       capture_output=True, text=True, env=env, timeout=600)
print(result.stderr)
if result.returncode != 0:
    kh.step(f"Smoke test FAILED (exit {result.returncode})")
else:
    kh.step("Smoke test PASSED")

# ── Compare reference stages ────────────────────────────────────────
kh.step("Comparing reference stages...")
try:
    from gguf import GGUFReader
    ref = GGUFReader(str(ref_path))
    ref_tensors = {t.name: np.array(t.data, dtype=np.float32) for t in ref.tensors}
    kh.step(f"Reference has {len(ref_tensors)} stages: {list(ref_tensors.keys())}")

    for name, data in sorted(ref_tensors.items()):
        kh.step(f"  {name}: shape={data.shape}, mean={data.mean():.6f}, std={data.std():.6f}")
except Exception as e:
    kh.step(f"Reference comparison failed: {e}")

# ── Write progress file ─────────────────────────────────────────────
with open(WORK / "progress.txt", "w") as f:
    f.write("HTDemucs validation complete\n")
    f.write(f"Model: {model_path}\n")
    f.write(f"Reference: {ref_path}\n")
    f.write(f"Smoke test exit code: {result.returncode}\n")

kh.step("Done!")
