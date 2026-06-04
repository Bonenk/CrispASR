#!/usr/bin/env python3
"""Kaggle kernel: convert KugelAudio-0-Open → F16 GGUF → quantize → upload to HF.

CPU-only (30 GB RAM is enough for the 18.7 GB model).
Produces kugelaudio-0-open-f16.gguf and kugelaudio-0-open-q4_k.gguf,
uploads both to cstr/kugelaudio-0-open-GGUF on HuggingFace.

Push: python -m kaggle kernels push -p tools/kaggle/kugelaudio-convert
"""

import os, sys, subprocess, shutil
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"

# ── Phase 0: HF token ──────────────────────────────────────────────────────
hf_token = None
# Try Kaggle Secrets first (works in UI-triggered runs)
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    print("[phase 0] HF_TOKEN from Kaggle Secrets OK")
except Exception:
    pass
# Fallback: dataset file
if not hf_token:
    token_path = "/kaggle/input/crispasr-hf-token/hf_token.txt"
    if os.path.exists(token_path):
        hf_token = open(token_path).read().strip()
        print("[phase 0] HF_TOKEN from dataset file OK")
# Fallback: env
if not hf_token:
    hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
else:
    print("[phase 0] WARNING: no HF_TOKEN — will stage for local pickup")

# ── Phase 1: Clone repo + install deps ──────────────────────────────────────
print("=== Phase 1: clone + deps ===", flush=True)
if not REPO.exists():
    subprocess.check_call([
        "git", "clone", "--depth", "1", "-b", "feature/kugelaudio-tts",
        "https://github.com/CrispStrobe/CrispASR", str(REPO),
    ])
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "torch", "transformers", "safetensors", "gguf", "huggingface_hub",
    "hf_transfer",
])
print("[phase 1] deps installed")

# ── Phase 2: Download source model ─────────────────────────────────────────
print("=== Phase 2: download model ===", flush=True)
from huggingface_hub import snapshot_download

# Use /kaggle/temp for the source cache (not /kaggle/working — capped at ~20 GB)
for candidate in ("/kaggle/temp", "/tmp"):
    if os.path.isdir(candidate):
        scratch = Path(candidate) / "kugelaudio-src"
        break
scratch.mkdir(parents=True, exist_ok=True)
print(f"  source cache: {scratch} (free: {shutil.disk_usage(scratch).free / (1024**3):.1f} GiB)")

src = snapshot_download(
    repo_id="kugelaudio/kugelaudio-0-open",
    cache_dir=str(scratch),
)
print(f"  source dir: {src}")

# ── Phase 3: Convert to F16 GGUF ───────────────────────────────────────────
print("=== Phase 3: convert to F16 GGUF ===", flush=True)
f16_path = WORK / "kugelaudio-0-open-f16.gguf"

subprocess.check_call([
    sys.executable, "models/convert-kugelaudio-to-gguf.py",
    "--input", src,
    "--output", str(f16_path),
    "--no-encoders",
    "--type", "f16",
], cwd=str(REPO))
print(f"  F16 GGUF: {f16_path} ({f16_path.stat().st_size / (1024**3):.1f} GiB)")

# ── Phase 4: Build crispasr-quantize ────────────────────────────────────────
print("=== Phase 4: build quantizer ===", flush=True)
# Install ninja for faster builds
subprocess.run(["pip", "install", "--quiet", "ninja"], check=False)

BUILD.mkdir(parents=True, exist_ok=True)
subprocess.check_call([
    "cmake", "-G", "Ninja",
    "-B", str(BUILD), "-S", str(REPO),
    "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=OFF",
])

# Only build the quantize target
subprocess.check_call([
    "cmake", "--build", str(BUILD),
    "-j4", "--target", "crispasr-quantize",
])
quantize_bin = BUILD / "bin" / "crispasr-quantize"
print(f"  quantizer: {quantize_bin} ({quantize_bin.stat().st_size / (1024**2):.1f} MiB)")

# ── Phase 5: Quantize to Q4_K ──────────────────────────────────────────────
print("=== Phase 5: quantize Q4_K ===", flush=True)
q4k_path = WORK / "kugelaudio-0-open-q4_k.gguf"

subprocess.check_call([
    str(quantize_bin), str(f16_path), str(q4k_path), "q4_k",
])
print(f"  Q4_K GGUF: {q4k_path} ({q4k_path.stat().st_size / (1024**3):.1f} GiB)")

# ── Phase 6: Upload to HF ──────────────────────────────────────────────────
print("=== Phase 6: upload to HF ===", flush=True)
HF_REPO = "cstr/kugelaudio-0-open-GGUF"

if hf_token:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)

    # Create repo if needed
    try:
        api.create_repo(repo_id=HF_REPO, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"  repo create: {e}")

    for local_path, remote_name in [
        (f16_path, "kugelaudio-0-open-f16.gguf"),
        (q4k_path, "kugelaudio-0-open-q4_k.gguf"),
    ]:
        print(f"  uploading {remote_name} ({local_path.stat().st_size / (1024**3):.1f} GiB)...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_name,
            repo_id=HF_REPO,
            repo_type="model",
            commit_message=f"Add {remote_name} (KugelAudio-0-Open TTS, --no-encoders)",
        )
        print(f"  uploaded {remote_name}")

    print(f"\n  All uploaded to https://huggingface.co/{HF_REPO}")
else:
    print(f"  No HF_TOKEN — staged for local pickup:")
    print(f"    kaggle kernels output ${KAGGLE_ACCOUNT}/crispasr-kugelaudio-convert -p .")
    for p in [f16_path, q4k_path]:
        print(f"    {p.name}: {p.stat().st_size / (1024**3):.1f} GiB")

print("\n=== Done ===")
