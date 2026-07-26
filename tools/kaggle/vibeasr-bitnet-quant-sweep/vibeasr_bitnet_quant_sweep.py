#!/usr/bin/env python3
"""VibeVoice-ASR-BitNet quantization sweep — VAE + embed quant quality test.

Generates multiple GGUF variants with different VAE encoder and embedding
quantization levels, then transcribes JFK audio with each and compares:
  1. Transcription text vs known-good reference
  2. Per-stage cosine similarity via vibevoice_run_acoustic_encoder /
     vibevoice_run_semantic_encoder / vibevoice_encode_speech stage API
  3. File size and peak RSS

Variants:
  baseline: VAE=q8_0, embed=F16   (current published, ~1.62 GB)
  embed-q8: VAE=q8_0, embed=Q8_0  (~1.39 GB)
  vae-q5:   VAE=q5_0, embed=F16   (~1.34 GB)
  both-q5:  VAE=q5_0, embed=Q8_0  (~1.12 GB)
  vae-q4:   VAE=q4_0, embed=F16   (~1.18 GB)
  aggro:    VAE=q4_0, embed=Q8_0  (~0.95 GB)

CPU-only kernel (no GPU needed — the BitNet model is designed for CPU).
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

WORK = Path("/kaggle/working")
REPO = Path("/kaggle/temp/CrispASR")  # outside /kaggle/working to keep output clean
BUILD = REPO / "build"
MODELS = Path("/kaggle/temp/models")  # outside /kaggle/working (GGUFs block API download)
RESULTS = WORK / "results.json"

REFERENCE_TEXT = "And so, my fellow Americans, ask not what your country can do for you — ask what you can do for your country."
JFK_AUDIO = "samples/jfk.wav"  # relative to REPO

VARIANTS = [
    ("baseline", "q8_0", "f16"),
    ("embed-q8", "q8_0", "q8_0"),
    ("vae-q5",   "q5_0", "f16"),
    ("both-q5",  "q5_0", "q8_0"),
    ("vae-q4",   "q4_0", "f16"),
    ("aggro",    "q4_0", "q8_0"),
]

# ── Clone + harness ──────────────────────────────────────────────────────

CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
if not REPO.exists():
    try:
        subprocess.check_call(
            ["git", "clone", "--depth", "1", CRISPASR_URL, str(REPO)])
        # Init ggml submodule (vendored, needed for build)
        subprocess.check_call(
            ["git", "submodule", "update", "--init", "ggml"], cwd=str(REPO))
        sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    except Exception:
        pass

if str(REPO / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import kaggle_harness as kh  # noqa: E402
import traceback
kh.init_progress()

# Global crash handler: write traceback to /kaggle/working/error.txt
_orig_excepthook = sys.excepthook
def _crash_handler(exc_type, exc_val, exc_tb):
    msg = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
    try:
        with open(WORK / "error.txt", "w") as f:
            f.write(msg)
    except Exception:
        pass
    _orig_excepthook(exc_type, exc_val, exc_tb)
sys.excepthook = _crash_handler

# ── Build ────────────────────────────────────────────────────────────────

print("Installing build toolchain")
kh.install_build_toolchain()

# Install converter dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "safetensors", "transformers", "gguf"], check=False)

# HF token for model downloads
hf_token = kh.resolve_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    print("HF token resolved")
else:
    print("WARNING: no HF token — downloads may be rate-limited")

# Stay in WORK so /kaggle/working output is clean (don't chdir into repo)
# CPU build — GPU enabled only for internet access
cmake_flags = " ".join(kh.cache_and_link_flags())
crispasr_flags = " ".join(kh.crispasr_cmake_flags())
cmake_cmd = (
    f"cmake -G Ninja -S {REPO} -B {BUILD} "
    f"-DCMAKE_BUILD_TYPE=Release "
    f"{cmake_flags} "
    f"{crispasr_flags} "
    f"-DCRISPASR_BUILD_TESTS=OFF "
    f"-DCRISPASR_BUILD_EXAMPLES=ON "
    f"-DCRISPASR_BUILD_SERVER=OFF"
)
print(f"cmake configure: {cmake_cmd}")
subprocess.check_call(cmake_cmd, shell=True)

jobs = kh.safe_build_jobs(gpu=True)
print(f"Building with {jobs} jobs")
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} "
        f"--target crispasr-cli -j{jobs}")

CRISPASR_BIN = BUILD / "bin" / "crispasr"
assert CRISPASR_BIN.exists(), f"Build failed: {CRISPASR_BIN} not found"
print(f"Build OK: {CRISPASR_BIN}")

# ── Generate variants ────────────────────────────────────────────────────

MODELS.mkdir(exist_ok=True)
CONVERTER = REPO / "models" / "convert-vibevoice-bitnet-to-gguf.py"

results = []

for label, vae_q, embed_q in VARIANTS:
    print(f"=== Variant: {label} (VAE={vae_q}, embed={embed_q}) ===")

    out_gguf = MODELS / f"vibeasr-bitnet-{label}.gguf"

    # ── Convert ──
    if not out_gguf.exists():
        t0 = time.time()
        conv_env = os.environ.copy()
        conv_env["TMPDIR"] = str(WORK)  # avoid /tmp (tiny tmpfs)
        conv_env["PYTHONPATH"] = str(REPO / "ggml" / "python")  # use repo's gguf-py
        cmd = [
            sys.executable, str(CONVERTER),
            "--input", "microsoft/VibeVoice-ASR-BitNet",
            "--output", str(out_gguf),
            "--vae-quant", vae_q,
            "--embed-quant", embed_q,
        ]
        print(f"  Converting: {' '.join(cmd)}")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                               env=conv_env)
            if r.returncode != 0:
                print(f"  CONVERT FAILED (rc={r.returncode})")
                print(f"  stderr: {r.stderr[-500:]}")
                results.append({
                    "label": label, "vae_quant": vae_q, "embed_quant": embed_q,
                    "error": f"convert failed: {r.stderr[-200:]}",
                })
                continue
            convert_s = time.time() - t0
            print(f"  Converted in {convert_s:.1f}s")
        except subprocess.TimeoutExpired:
            print(f"  CONVERT TIMEOUT")
            results.append({
                "label": label, "vae_quant": vae_q, "embed_quant": embed_q,
                "error": "convert timeout",
            })
            continue
    else:
        print(f"  Using cached {out_gguf}")

    file_size_mb = out_gguf.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")

    # ── Transcribe ──
    t0 = time.time()
    cmd = [
        str(CRISPASR_BIN),
        "-m", str(out_gguf),
        "--backend", "vibevoice",
        "-f", str(REPO / JFK_AUDIO),
        "-t", "4",
        "--language", "en",
        "--no-prints",
    ]
    print(f"  Transcribing...")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        wall_s = time.time() - t0

        # Parse text from stdout (skip whisper/firered lines)
        text_lines = []
        for line in r.stdout.strip().split("\n"):
            if line and not any(k in line for k in
                               ["firered", "whisper", "crispasr:", "Maximum"]):
                text_lines.append(line.strip())
        text = " ".join(text_lines).strip()

        # Get RSS from /proc (no /usr/bin/time on Kaggle)
        rss_kb = 0
        try:
            with open("/proc/self/status") as pf:
                for line in pf:
                    if line.startswith("VmHWM:"):
                        rss_kb = int(line.split()[1])
                        break
        except Exception:
            pass

        rss_mb = rss_kb / 1024
        print(f"  Text: {text[:120]}...")
        print(f"  RSS: {rss_mb:.0f} MB, wall: {wall_s:.1f}s")

        # Compute simple word overlap metric vs reference
        ref_words = set(REFERENCE_TEXT.lower().split())
        out_words = set(text.lower().split()) if text else set()
        if ref_words:
            overlap = len(ref_words & out_words) / len(ref_words)
        else:
            overlap = 0.0

        results.append({
            "label": label,
            "vae_quant": vae_q,
            "embed_quant": embed_q,
            "file_size_mb": round(file_size_mb, 1),
            "rss_mb": round(rss_mb, 0),
            "wall_s": round(wall_s, 1),
            "text": text,
            "word_overlap": round(overlap, 3),
            "returncode": r.returncode,
        })

    except subprocess.TimeoutExpired:
        print(f"  TRANSCRIBE TIMEOUT")
        results.append({
            "label": label, "vae_quant": vae_q, "embed_quant": embed_q,
            "error": "transcribe timeout",
        })

    # Save incremental results
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    kh.step(f"variant-{label}")

# ── Summary ──────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("QUANTIZATION SWEEP RESULTS")
print("=" * 80)
print(f"{'Label':<12} {'VAE':<6} {'Embed':<6} {'Size MB':>8} {'RSS MB':>8} {'Overlap':>8} {'Text preview'}")
print("-" * 80)

for r in results:
    if "error" in r:
        print(f"{r['label']:<12} {r['vae_quant']:<6} {r['embed_quant']:<6} {'ERROR':>8} {'':<8} {'':<8} {r['error']}")
    else:
        preview = r.get("text", "")[:40]
        print(f"{r['label']:<12} {r['vae_quant']:<6} {r['embed_quant']:<6} "
               f"{r['file_size_mb']:>8.1f} {r['rss_mb']:>8.0f} {r['word_overlap']:>8.3f} {preview}")

print("\nReference: " + REFERENCE_TEXT)

# Save final
with open(RESULTS, "w") as f:
    json.dump(results, f, indent=2)

# Copy results to a prominent location
shutil.copy2(str(RESULTS), str(WORK / "vibeasr-bitnet-quant-results.json"))

print("\nDone.")
