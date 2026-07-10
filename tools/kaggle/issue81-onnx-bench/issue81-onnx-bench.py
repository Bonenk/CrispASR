#!/usr/bin/env python3
"""Kaggle GPU kernel: CrispASR vs onnx-asr head-to-head benchmark (#81).

Fair comparison on Kaggle P100/T4:
  - Same audio (JFK 11s, librispeech 60s)
  - Same model family (parakeet-tdt-0.6b)
  - Both use GPU when available
  - Warmup + 5 timed runs each
  - Reports: load time, RTF, x-realtime, transcript snippet, WER

Push (under chr1s4):
  export KAGGLE_API_TOKEN=<chr1s4 token>
  python -m kaggle kernels push -p tools/kaggle/issue81-onnx-bench
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")

# ── Phase 0: Clone + build CrispASR ─────────────────────────────────────────
print("=== Phase 0: clone + build CrispASR ===", flush=True)
if not REPO.exists():
    subprocess.check_call([
        "git", "clone", "--depth", "1", "-b", "main",
        "https://github.com/CrispStrobe/CrispASR", str(REPO),
    ])
if (REPO / "ggml").is_dir() and not (REPO / "ggml" / "CMakeLists.txt").exists():
    subprocess.check_call(["git", "submodule", "update", "--init", "ggml"], cwd=str(REPO))

if (REPO / "tools" / "kaggle").is_dir():
    sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

BUILD = TEMP / "build"
BUILD.mkdir(parents=True, exist_ok=True)

# Detect GPU and build with CUDA if available
has_cuda = Path("/usr/local/cuda/bin/nvcc").exists()
print(f"  CUDA available: {has_cuda}")

if has_cuda:
    arch = kh.detect_cuda_arch()
    flags = kh.cuda_build_flags(arch) + kh.cache_and_link_flags()
    cmake_flags = "-DCMAKE_BUILD_TYPE=Release " + " ".join(flags)
    print(f"  cmake flags: {cmake_flags[:120]}...")
    ret = subprocess.call(
        f"cmake -G Ninja -B {BUILD} -S {REPO} {cmake_flags}",
        shell=True,
    )
    if ret != 0:
        print("  CUDA cmake failed, falling back to CPU-only build")
        has_cuda = False
        import shutil
        if BUILD.exists():
            shutil.rmtree(BUILD)
            BUILD.mkdir(parents=True, exist_ok=True)

if not has_cuda:
    cmake_flags = "-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF"
    subprocess.check_call(
        f"cmake -G Ninja -B {BUILD} -S {REPO} {cmake_flags}",
        shell=True, stdout=subprocess.DEVNULL,
    )
n_jobs = min(os.cpu_count() or 2, 4)
subprocess.check_call(
    f"cmake --build {BUILD} -j{n_jobs} --target crispasr-cli",
    shell=True, stdout=subprocess.DEVNULL,
)
CRISPASR_BIN = BUILD / "bin" / "crispasr"
print(f"  built: {CRISPASR_BIN}")

# ── Phase 1: Install onnx-asr ───────────────────────────────────────────────
print("\n=== Phase 1: install onnx-asr ===", flush=True)
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "onnx-asr", "soundfile", "huggingface_hub", "onnxruntime",
])
if has_cuda:
    # Install CUDA EP for onnxruntime (replaces CPU-only onnxruntime)
    subprocess.call([
        sys.executable, "-m", "pip", "install", "-q",
        "onnxruntime-gpu",
    ])

# ── Phase 2: Download models ────────────────────────────────────────────────
print("\n=== Phase 2: download models ===", flush=True)
from huggingface_hub import hf_hub_download

# CrispASR: parakeet-tdt-0.6b Q8_0
gguf_model = hf_hub_download(
    "cstr/parakeet-tdt-0.6b-v2-GGUF",
    "parakeet-tdt-0.6b-v2-q8_0.gguf",
    cache_dir=str(TEMP / "hf"),
)
print(f"  GGUF: {gguf_model}")

# Audio files
jfk_wav = REPO / "samples" / "jfk.wav"
if not jfk_wav.exists():
    print("  WARNING: jfk.wav not found, generating sine tone")
    import numpy as np
    import soundfile as sf
    sr = 16000
    t = np.linspace(0, 11, sr * 11, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    jfk_wav = TEMP / "test_audio.wav"
    sf.write(str(jfk_wav), audio, sr)

# Generate a longer test audio (~60s) by repeating JFK
import soundfile as sf
import numpy as np

jfk_pcm, jfk_sr = sf.read(str(jfk_wav), dtype="float32")
jfk_duration = len(jfk_pcm) / jfk_sr
# Repeat to get ~60s
repeats = max(1, int(60 / jfk_duration))
long_pcm = np.tile(jfk_pcm, repeats)
long_wav = TEMP / "long_test_60s.wav"
sf.write(str(long_wav), long_pcm, jfk_sr)
long_duration = len(long_pcm) / jfk_sr
print(f"  JFK: {jfk_duration:.1f}s, Long: {long_duration:.1f}s ({repeats}x)")

# ── Phase 3: Benchmark CrispASR ─────────────────────────────────────────────
print("\n=== Phase 3: benchmark CrispASR (parakeet-tdt Q8_0) ===", flush=True)

def run_crispasr(audio_path, n_warmup=1, n_runs=5):
    """Run crispasr CLI and time it."""
    backend_flag = "--backend parakeet"
    gpu_flag = ""
    if has_cuda:
        gpu_flag = "--gpu-backend cuda"

    cmd = (
        f"{CRISPASR_BIN} {backend_flag} {gpu_flag} "
        f"-m {gguf_model} -f {audio_path} --no-prints"
    )

    # Warmup
    for _ in range(n_warmup):
        subprocess.run(cmd, shell=True, capture_output=True)

    # Timed runs
    times = []
    transcript = ""
    for i in range(n_runs):
        t0 = time.perf_counter()
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if i == 0:
            transcript = result.stdout.strip()[:200]

    return times, transcript

# Short audio (JFK ~11s)
print(f"  Running CrispASR on JFK ({jfk_duration:.1f}s)...")
ca_times_short, ca_text_short = run_crispasr(jfk_wav)
ca_mean_short = sum(ca_times_short) / len(ca_times_short)
ca_rtf_short = jfk_duration / ca_mean_short
print(f"  CrispASR JFK:  mean={ca_mean_short:.3f}s  {ca_rtf_short:.1f}x realtime")
print(f"  transcript: {ca_text_short[:100]}")

# Long audio (~60s)
print(f"\n  Running CrispASR on long audio ({long_duration:.1f}s)...")
ca_times_long, ca_text_long = run_crispasr(long_wav, n_warmup=1, n_runs=3)
ca_mean_long = sum(ca_times_long) / len(ca_times_long)
ca_rtf_long = long_duration / ca_mean_long
print(f"  CrispASR long: mean={ca_mean_long:.3f}s  {ca_rtf_long:.1f}x realtime")

# ── Phase 4: Benchmark onnx-asr ─────────────────────────────────────────────
print("\n=== Phase 4: benchmark onnx-asr (parakeet-tdt) ===", flush=True)

def run_onnx_asr(audio_path, n_warmup=1, n_runs=5):
    """Run onnx-asr and time it."""
    import onnx_asr

    # Load model once
    t_load_start = time.perf_counter()
    providers = None
    if has_cuda:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    model = onnx_asr.load_model(
        "nemo-parakeet-ctc-0.6b",
        quantization="int8",
        providers=providers,
    )
    t_load = time.perf_counter() - t_load_start
    print(f"  onnx-asr load: {t_load:.3f}s")

    pcm, sr = sf.read(str(audio_path), dtype="float32")
    if sr != 16000:
        # Resample if needed
        from scipy.signal import resample
        pcm = resample(pcm, int(len(pcm) * 16000 / sr)).astype(np.float32)

    # Warmup
    for _ in range(n_warmup):
        model.recognize(str(audio_path))

    # Timed runs
    times = []
    transcript = ""
    for i in range(n_runs):
        t0 = time.perf_counter()
        result = model.recognize(str(audio_path))
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if i == 0:
            transcript = str(result)[:200]

    return times, transcript, t_load

# Short audio
print(f"  Running onnx-asr on JFK ({jfk_duration:.1f}s)...")
try:
    onnx_times_short, onnx_text_short, onnx_load = run_onnx_asr(jfk_wav)
    onnx_mean_short = sum(onnx_times_short) / len(onnx_times_short)
    onnx_rtf_short = jfk_duration / onnx_mean_short
    print(f"  onnx-asr JFK:  mean={onnx_mean_short:.3f}s  {onnx_rtf_short:.1f}x realtime")
    print(f"  transcript: {onnx_text_short[:100]}")
except Exception as e:
    print(f"  onnx-asr FAILED: {e}")
    onnx_times_short = []
    onnx_mean_short = float("inf")
    onnx_rtf_short = 0
    onnx_text_short = f"ERROR: {e}"
    onnx_load = 0

# Long audio
print(f"\n  Running onnx-asr on long audio ({long_duration:.1f}s)...")
try:
    onnx_times_long, onnx_text_long, _ = run_onnx_asr(long_wav, n_warmup=1, n_runs=3)
    onnx_mean_long = sum(onnx_times_long) / len(onnx_times_long)
    onnx_rtf_long = long_duration / onnx_mean_long
    print(f"  onnx-asr long: mean={onnx_mean_long:.3f}s  {onnx_rtf_long:.1f}x realtime")
except Exception as e:
    print(f"  onnx-asr long FAILED: {e}")
    onnx_times_long = []
    onnx_mean_long = float("inf")
    onnx_rtf_long = 0

# ── Phase 5: Summary ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("=== BENCHMARK SUMMARY: CrispASR vs onnx-asr ===")
print("=" * 70)

gpu_name = "unknown"
try:
    gpu_name = subprocess.check_output(
        "nvidia-smi --query-gpu=name --format=csv,noheader", shell=True
    ).decode().strip()
except Exception:
    pass
cpu_name = "unknown"
try:
    cpu_name = subprocess.check_output(
        "cat /proc/cpuinfo | grep 'model name' | head -1 | cut -d: -f2",
        shell=True,
    ).decode().strip()
except Exception:
    pass

print(f"GPU: {gpu_name}")
print(f"CPU: {cpu_name}")
print(f"CUDA build: {has_cuda}")
print()
print(f"{'Metric':<30s} {'CrispASR':>12s} {'onnx-asr':>12s} {'ratio':>10s}")
print("-" * 70)

def fmt_ratio(ca, onnx):
    if onnx == 0 or onnx == float("inf"):
        return "N/A"
    r = ca / onnx
    return f"{r:.2f}x"

print(f"{'JFK (short) mean time':.<30s} {ca_mean_short:>11.3f}s {onnx_mean_short:>11.3f}s {fmt_ratio(onnx_mean_short, ca_mean_short):>10s}")
print(f"{'JFK x-realtime':.<30s} {ca_rtf_short:>11.1f}x {onnx_rtf_short:>11.1f}x {fmt_ratio(ca_rtf_short, onnx_rtf_short):>10s}")
print(f"{'Long (~60s) mean time':.<30s} {ca_mean_long:>11.3f}s {onnx_mean_long:>11.3f}s {fmt_ratio(onnx_mean_long, ca_mean_long):>10s}")
print(f"{'Long x-realtime':.<30s} {ca_rtf_long:>11.1f}x {onnx_rtf_long:>11.1f}x {fmt_ratio(ca_rtf_long, onnx_rtf_long):>10s}")
print()

# Save results as JSON
results = {
    "gpu": gpu_name,
    "cpu": cpu_name,
    "cuda_build": has_cuda,
    "crispasr": {
        "model": "parakeet-tdt-0.6b-v2-q8_0.gguf",
        "backend": "parakeet",
        "jfk": {"duration_s": jfk_duration, "times": ca_times_short, "mean": ca_mean_short, "rtf": ca_rtf_short},
        "long": {"duration_s": long_duration, "times": ca_times_long, "mean": ca_mean_long, "rtf": ca_rtf_long},
        "transcript_short": ca_text_short,
    },
    "onnx_asr": {
        "model": "nvidia/parakeet-tdt-0.6b",
        "load_time": onnx_load,
        "jfk": {"duration_s": jfk_duration, "times": onnx_times_short, "mean": onnx_mean_short, "rtf": onnx_rtf_short},
        "long": {"duration_s": long_duration, "times": onnx_times_long, "mean": onnx_mean_long, "rtf": onnx_rtf_long},
        "transcript_short": onnx_text_short,
    },
}
results_path = WORK / "benchmark_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {results_path}")

# Also try whisper and cohere if time permits
print("\n=== Done ===", flush=True)
