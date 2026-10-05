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
os.environ.setdefault("CRISPASR_REF", "ci/kaggle-log")
os.environ.setdefault("CRISPASR_REGRESSION_MODE", "validate")
os.environ.setdefault("CRISPASR_REGRESSION_BUILD", "cuda")
# Second pass: only the backends the first pass (version 1, ten backends:
# parakeet, canary, cohere, sensevoice, qwen3-asr, nemotron passed) did not
# settle. moonshine-tiny and index-echo-2b died for want of their companion
# files (harness gap, fixed in the ref above); wav2vec2 flipped a word.
os.environ.setdefault(
    "CRISPASR_REGRESSION_BACKENDS",
    ",".join(["moonshine-tiny", "index-echo-2b", "wav2vec2-xlsr-en"]),
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
suite_exit = 0
try:
    exec(compile(script.read_text(), str(script), "exec"))
except SystemExit as e:  # keep going: the diagnostic below is the point
    suite_exit = e.code or 0

# wav2vec2 on the SAME binary, GPU vs CPU. If the CPU run prints the expected
# sentence and the GPU run does not, the difference is GPU arithmetic on a
# borderline CTC frame, not a wrong graph.
try:
    from huggingface_hub import hf_hub_download

    w2v = hf_hub_download(
        repo_id="cstr/wav2vec2-large-xlsr-53-english-GGUF",
        filename="wav2vec2-xlsr-en-q4_k.gguf",
        revision="3de5f69700e163286eaae85952fba9bf4ab761f6",
    )
    exe = "/kaggle/working/build/bin/crispasr"
    wav = "/kaggle/working/CrispASR/samples/jfk.wav"
    for label, extra in (("GPU", []), ("CPU (-ng)", ["-ng"])):
        r = subprocess.run([exe, "-m", w2v, "-f", wav, "-np"] + extra, capture_output=True, text=True, timeout=600)
        print(f"wav2vec2 {label}: rc={r.returncode} text={r.stdout.strip()!r}", flush=True)
        if r.returncode != 0:
            print(r.stderr[-1500:], flush=True)
except Exception as e:  # diagnostic only
    print(f"wav2vec2 diagnostic failed: {e!r}", flush=True)

sys.exit(suite_exit)
