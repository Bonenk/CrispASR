# %% [markdown]
# # TADA language voice reference GGUFs
#
# Generates tada-ref-{ar,ch,de,es,fr,it,ja,pl,pt}.gguf from FLEURS CC-BY-4.0
# clips using the HumeAI/tada-codec language-specific aligners.
#
# NO HF upload from here — fetch locally + upload:
#
#   kaggle kernels output ${KAGGLE_ACCOUNT}/tada-language-voice-reference-ggufs \
#       -p /Volumes/backups/code/tada-lang-refs-stash/kaggle-out/
#   python tools/upload_tada_lang_refs.py \
#       --dir /Volumes/backups/code/tada-lang-refs-stash/kaggle-out/lang-refs/

# %% [code]
import os, sys, subprocess
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
OUT  = WORK / "lang-refs"
OUT.mkdir(exist_ok=True)

# %% [code]  clone CrispASR + import harness (bundled copy as fallback)
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
if not REPO.exists():
    try:
        subprocess.check_call(["git", "clone", "--depth", "1",
                               CRISPASR_URL, str(REPO)])
        sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    except Exception:
        pass  # fall through to bundled copy
else:
    subprocess.check_call(["git", "-C", str(REPO), "pull", "--ff-only"])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))

if str(REPO / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import kaggle_harness as kh
kh.init_progress()

# ── HF auth — MUST be first after harness init ──────────────────────────
# Resolves: env HF_TOKEN → Kaggle Secret → mounted dataset file
# Also sets HF_HUB_ENABLE_HF_TRANSFER=1 and exports to subprocess env.
token = kh.resolve_hf_token()

# %% [code]  remove tensorflow to prevent protobuf clash
kh.step("pip-remove-tf")
subprocess.check_call(
    [sys.executable, "-m", "pip", "uninstall", "-y", "tensorflow", "tf-keras"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)

# %% [code]  install deps
# NOTE: do NOT install torch/torchaudio — Kaggle pre-installs them (gotcha #11)
kh.step("pip-install")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "gguf", "datasets", "soundfile", "scipy",
    "hume-tada",   # PyPI package; not git URL
])

# %% [code]  generate (subprocess inherits HF_TOKEN env set by resolve_hf_token)
kh.step("gen.begin")
result = subprocess.run(
    [sys.executable, str(REPO / "tools" / "gen_tada_lang_refs.py"),
     "--output-dir", str(OUT),
     "--skip-existing"],
    check=False,
)
kh.step("gen.done", returncode=result.returncode)

# %% [code]  report
files = sorted(OUT.glob("*.gguf"))
print(f"\n=== generated {len(files)} lang ref(s) ===")
for p in files:
    print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")

print("\n=== fetch + upload locally (NOT on Kaggle) ===")
print("  kaggle kernels output ${KAGGLE_ACCOUNT}/tada-language-voice-reference-ggufs \\")
print("      -p /Volumes/backups/code/tada-lang-refs-stash/kaggle-out/")
print("  python tools/upload_tada_lang_refs.py \\")
print("      --dir /Volumes/backups/code/tada-lang-refs-stash/kaggle-out/lang-refs/")
