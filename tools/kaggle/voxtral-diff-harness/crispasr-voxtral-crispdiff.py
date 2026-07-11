# ─────────────────────────── cell 0 (markdown) ───────────────────────────
# # CrispASR — voxtral-tts PROPER per-stage crispasr-diff on F16
#
# The real HARD-RULE-#2 harness (not the codes-level mudler comparison):
#   python tools/dump_reference.py --backend voxtral-tts --model-dir <m> --output ref.gguf
#   build/bin/crispasr-diff voxtral-tts voxtral-4b-tts-f16.gguf ref.gguf <wav>
#
# The Python dumper is a MANUAL PyTorch LLM forward (no vllm). At F16 the runtime
# should match the BF16 reference per layer (cos>=0.99) — a clean green — vs the
# local Q4_K run where the special AUDIO-token embed row quantized to cos 0.44.
#
# Datasets: chr1str/crispasr-hf-token, chr1str/crispasr-ccache.
# GPU + Internet + ~18 GB disk (ref model 8 GB + F16 GGUF 8 GB → /tmp).

# ─────────────────────────── cell 1 (code) ───────────────────────────
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
TMP = Path("/tmp/vtts-cd")
REFMODEL = TMP / "refmodel"
MODELS = TMP / "models"
for d in (REFMODEL, MODELS):
    d.mkdir(parents=True, exist_ok=True)

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
TEXT = os.environ.get("VOXTRAL_TTS_TEXT", "Hello world.")
VOICE = os.environ.get("VOXTRAL_TTS_VOICE", "neutral_female")


def sh(cmd, check=True, env=None, cwd=None, timeout=None):
    e = {**os.environ, **(env or {})}
    r = subprocess.run(cmd, env=e, cwd=cwd, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(r.stdout[-6000:], flush=True)
    if check and r.returncode != 0:
        raise SystemExit(f"cmd failed ({r.returncode}): {cmd}")
    return r


# ── clone + build crispasr-diff ──
if REPO.exists():
    shutil.rmtree(REPO)
sh(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, "--recursive",
    "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
print("sha", sha, flush=True)
gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
BUILD.mkdir(exist_ok=True)
sh(["cmake", "-S", str(REPO), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON"]
   + kh.cuda_build_flags(arch) + kh.cache_and_link_flags())
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-diff -j{kh.safe_build_jobs(gpu=True)}")
DIFF = next(c for c in BUILD.rglob("crispasr-diff") if c.is_file() and os.access(c, os.X_OK))
os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
print("crispasr-diff:", DIFF, flush=True)

# ── deps for the dumper ──
sh([sys.executable, "-m", "pip", "install", "-q", "mistral_common", "safetensors", "gguf"], check=False)

# ── downloads (→ /tmp) ──
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402
TOKEN = kh.resolve_hf_token()
print("downloading reference model…", flush=True)
snapshot_download("mistralai/Voxtral-4B-TTS-2603", local_dir=str(REFMODEL), token=TOKEN or None,
                  allow_patterns=["consolidated.safetensors", "params.json", "tekken.json", "voice_embedding/*"])
F16 = hf_hub_download("cstr/voxtral-4b-tts-GGUF", "voxtral-4b-tts-f16.gguf", local_dir=str(MODELS), token=TOKEN or None)
print("F16:", round(os.path.getsize(F16) / 1e9, 2), "GB", flush=True)

# ── 1) dump the reference (manual PyTorch LLM forward, no vllm) ──
REFGGUF = TMP / "voxtral-tts-ref.gguf"
AUDIO = str(REPO / "samples" / "jfk.wav")  # 16 kHz; unused by voxtral-tts, required by the harness
print("\n===== dump_reference.py (manual PyTorch, no vllm) =====", flush=True)
sh([sys.executable, str(REPO / "tools" / "dump_reference.py"), "--backend", "voxtral-tts",
    "--model-dir", str(REFMODEL), "--audio", AUDIO, "--output", str(REFGGUF)],
   env={"VOXTRAL_TTS_TEXT": TEXT, "VOXTRAL_TTS_VOICE": VOICE, "OMP_NUM_THREADS": "4"}, timeout=2400)

# ── 2) crispasr-diff voxtral-tts <F16> <ref.gguf> — the clean F16 per-layer green ──
print("\n===== crispasr-diff voxtral-tts (F16 vs BF16 reference) =====", flush=True)
r = sh([str(DIFF), "voxtral-tts", F16, str(REFGGUF), AUDIO],
       env={"VOXTRAL_TTS_TEXT": TEXT, "VOXTRAL_TTS_VOICE": VOICE}, check=False, timeout=1200)
print(f"\n=== DONE  gpu={gpu} sha={sha[:8]}  exit={r.returncode} "
      f"({'ALL PASS' if r.returncode == 0 else 'divergence'}) ===", flush=True)
