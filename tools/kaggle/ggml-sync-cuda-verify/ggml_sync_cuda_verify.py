# CrispASR — CUDA validation of a ggml bump (Kaggle, GPU).
#
# Why this needs a GPU: a ggml upstream merge is resolved by reading for the
# backends the author cannot build. CI compiles CUDA; nothing in CI RUNS it.
# This kernel builds the ref below with -DGGML_CUDA=ON and runs the regression
# suite on the GPU for a set of backends chosen to cover the code the merge
# touched: conformer encoders (parakeet, canary, cohere, nemotron), flash
# attention with masks (the fork's per-head-mask patch sits next to upstream's
# new sparse path in fattn.cu), quantised LLM decoders (qwen3-asr, voxtral,
# index-echo's Qwen3.5), and small conv/CTC models (moonshine, wav2vec2,
# sensevoice).
#
# It is a thin bootstrap: it pins the knobs, clones the repo, and execs the
# canonical tools/kaggle/crispasr-regression.py from the ref under test — the
# same script the weekly validate kernel runs, so a failure here means the
# same thing it means there.
#
# One push, then read the log. Do not re-push in a loop (tools/kaggle/README.md).
import os
import subprocess
import sys
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

# The ref under test. Edit and push once per bump.
os.environ.setdefault("CRISPASR_REF", "live-translate")
os.environ.setdefault("CRISPASR_REGRESSION_MODE", "validate")
os.environ.setdefault("CRISPASR_REGRESSION_BUILD", "cuda")
os.environ.setdefault(
    "CRISPASR_REGRESSION_BACKENDS",
    ",".join(
        [
            "parakeet-tdt-0.6b-en",
            "canary-1b-v2",
            "cohere-transcribe",
            "nemotron-3.5-asr-streaming-0.6b",
            "qwen3-asr-0.6b",
            "voxtral-mini-3b-2507",
            "index-echo-2b",
            "moonshine-tiny",
            "wav2vec2-xlsr-en",
            "sensevoice-small",
        ]
    ),
)

subprocess.run(["nvidia-smi"], check=False)

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR-bootstrap"
if not REPO.exists():
    subprocess.check_call(
        ["git", "clone", "--depth", "50", "--no-single-branch", "https://github.com/CrispStrobe/CrispASR.git", str(REPO)]
    )
subprocess.check_call(["git", "checkout", os.environ["CRISPASR_REF"]], cwd=str(REPO))

script = REPO / "tools" / "kaggle" / "crispasr-regression.py"
print(f"exec {script} at ref {os.environ['CRISPASR_REF']}", flush=True)
sys.argv[0] = str(script)
exec(compile(script.read_text(), str(script), "exec"))
