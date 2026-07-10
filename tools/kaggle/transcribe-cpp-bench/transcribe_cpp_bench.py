# # CrispASR vs transcribe.cpp — head-to-head GPU benchmark
#
# Systematic evaluation of CrispASR against transcribe.cpp
# (https://github.com/handy-computer/transcribe.cpp) on shared ASR models.
#
# For each overlapping model family this kernel:
#   1. Downloads the CrispASR GGUF from cstr/ HF repos
#   2. Downloads the transcribe.cpp GGUF from handy-computer/ HF repos
#   3. Runs both on jfk.wav and measures:
#        - Transcript text (WER vs reference)
#        - Wall-clock inference time → RTF
#        - Model load time
#        - Peak RSS (via /proc/self/status)
#   4. Cleans up both GGUFs before the next model to stay within ~20 GB scratch
#   5. Streams per-model results to cstr/crispasr-kaggle-progress on HF
#
# "transcribe.cpp only" models (GigaAM family) are tested with transcribe.cpp
# only — these represent a CrispASR coverage gap.
#
# References: tools/kaggle-benchmark-all-backends.py (CrispASR full sweep),
# transcribe.cpp docs/models/*, scripts/wer/run.py
#
# Account:  chr1s4 (GPU quota, separate from chr1str's 30 h/week)
# Datasets: chr1s4/crispasr-hf-token  (HF auth)
#           chr1s4/crispasr-ccache     (warm CrispASR build, ~3 min vs ~20 min)

import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

WORK = Path("/kaggle/working")
_PROGRESS = WORK / "progress.jsonl"
_T0 = time.time()
_HF_LAST_PUSH = 0.0
_HF_REPO = "cstr/crispasr-kaggle-progress"
_HF_PATH = (
    f"runs/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    f"-transcribe-cpp-bench.jsonl"
)
MAX_WALL_S = 8 * 3600  # leave 1 h margin vs Kaggle's 9 h session limit

# ── Reference JFK transcript ──────────────────────────────────────────────────
JFK_REF = "and so my fellow americans ask not what your country can do for you ask what you can do for your country"
JFK_DURATION_S = 11.0  # approximate duration of jfk.wav

# ── Model pairs to benchmark ─────────────────────────────────────────────────
# Format:
#   family:              human-readable name
#   ca_backend:          CrispASR -b flag (or None = whisper file-based)
#   ca_url:              CrispASR GGUF download URL
#   ca_file:             local filename for CrispASR GGUF
#   tc_url:              transcribe.cpp GGUF download URL
#   tc_file:             local filename for transcribe.cpp GGUF
#   ca_wer_libri:        CrispASR LibriSpeech test-clean WER (from docs, None if unknown)
#   tc_wer_libri:        transcribe.cpp LibriSpeech test-clean WER (from their docs)
#   timeout_s:           per-model timeout for both inference calls
#   notes:               benchmark notes

SHARED_MODELS = [
    {
        "family":        "Whisper base",
        "ca_backend":    "whisper",
        "ca_url":        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
        "ca_file":       "ggml-base.bin",
        "tc_url":        "https://huggingface.co/handy-computer/whisper-base-gguf/resolve/main/whisper-base-Q8_0.gguf",
        "tc_file":       "whisper-base-Q8_0.gguf",
        "ca_wer_libri":  None,
        "tc_wer_libri":  "5.12%",
        "timeout_s":     60,
        "notes":         "CrispASR ggml-base.bin vs t.cpp Q8_0 (same weights, different GGUF schemas)",
    },
    {
        "family":        "Moonshine Tiny",
        "ca_backend":    "moonshine",
        "ca_url":        "https://huggingface.co/cstr/moonshine-tiny-GGUF/resolve/main/moonshine-tiny-q4_k.gguf",
        "ca_file":       "moonshine-tiny-q4_k.gguf",
        "tc_url":        "https://huggingface.co/handy-computer/moonshine-tiny-gguf/resolve/main/moonshine-tiny-Q8_0.gguf",
        "tc_file":       "moonshine-tiny-Q8_0.gguf",
        "ca_wer_libri":  None,
        "tc_wer_libri":  "4.60%",
        "timeout_s":     60,
        "notes":         "CrispASR Q4_K vs t.cpp Q8_0",
    },
    {
        "family":        "SenseVoice Small",
        "ca_backend":    "sensevoice",
        "ca_url":        "https://huggingface.co/cstr/sensevoice-small-GGUF/resolve/main/sensevoice-small-q4_k.gguf",
        "ca_file":       "sensevoice-small-q4_k.gguf",
        "tc_url":        "https://huggingface.co/handy-computer/SenseVoiceSmall-gguf/resolve/main/SenseVoiceSmall-Q8_0.gguf",
        "tc_file":       "SenseVoiceSmall-Q8_0.gguf",
        "ca_wer_libri":  None,
        "tc_wer_libri":  "3.13%",
        "timeout_s":     60,
        "notes":         "CrispASR Q4_K vs t.cpp Q8_0; SenseVoice emits emotion/LID tags in CrispASR",
    },
    {
        "family":        "Moonshine Streaming Tiny",
        "ca_backend":    "moonshine-streaming",
        "ca_url":        "https://huggingface.co/cstr/moonshine-streaming-tiny-GGUF/resolve/main/moonshine-streaming-tiny-q4_k.gguf",
        "ca_file":       "moonshine-streaming-tiny-q4_k.gguf",
        "tc_url":        "https://huggingface.co/handy-computer/moonshine-streaming-tiny-gguf/resolve/main/moonshine-streaming-tiny-Q8_0.gguf",
        "tc_file":       "moonshine-streaming-tiny-Q8_0.gguf",
        "ca_wer_libri":  None,
        "tc_wer_libri":  "4.52%",
        "timeout_s":     60,
        "notes":         "Both run in offline mode on jfk.wav",
    },
    {
        "family":        "Parakeet TDT 0.6B",
        "ca_backend":    "parakeet",
        "ca_url":        "https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF/resolve/main/parakeet-tdt-0.6b-v3-q4_k.gguf",
        "ca_file":       "parakeet-tdt-0.6b-v3-q4_k.gguf",
        "tc_url":        "https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v2-gguf/resolve/main/parakeet-tdt-0.6b-v2-Q4_K_M.gguf",
        "tc_file":       "parakeet-tdt-0.6b-v2-Q4_K_M.gguf",
        "ca_wer_libri":  None,
        "tc_wer_libri":  "1.72%",
        "timeout_s":     90,
        "notes":         "CrispASR v3 (25 EU langs) vs t.cpp v2 (EN only) — different checkpoints",
    },
    {
        "family":        "Qwen3-ASR 0.6B",
        "ca_backend":    "qwen3",
        "ca_url":        "https://huggingface.co/cstr/qwen3-asr-0.6b-GGUF/resolve/main/qwen3-asr-0.6b-q4_k.gguf",
        "ca_file":       "qwen3-asr-0.6b-q4_k.gguf",
        "tc_url":        "https://huggingface.co/handy-computer/Qwen3-ASR-0.6B-gguf/resolve/main/Qwen3-ASR-0.6B-Q4_K_M.gguf",
        "tc_file":       "Qwen3-ASR-0.6B-Q4_K_M.gguf",
        "ca_wer_libri":  None,
        "tc_wer_libri":  "2.26%",
        "timeout_s":     90,
        "notes":         "CrispASR Q4_K vs t.cpp Q4_K_M; same upstream checkpoint; Q8_0 t.cpp=2.11%",
    },
    {
        "family":        "Canary 1B v2",
        "ca_backend":    "canary",
        "ca_url":        "https://huggingface.co/cstr/canary-1b-v2-GGUF/resolve/main/canary-1b-v2-q4_k.gguf",
        "ca_file":       "canary-1b-v2-q4_k.gguf",
        "tc_url":        "https://huggingface.co/handy-computer/canary-1b-v2-gguf/resolve/main/canary-1b-v2-Q4_K_M.gguf",
        "tc_file":       "canary-1b-v2-Q4_K_M.gguf",
        "ca_wer_libri":  None,
        "tc_wer_libri":  None,  # canary-1b-v2 specific docs may not have WER yet
        "timeout_s":     120,
        "notes":         "Both Q4_K on same NVIDIA canary-1b-v2 checkpoint",
    },
    {
        "family":        "FunASR Nano 2512",
        "ca_backend":    "funasr",
        "ca_url":        "https://huggingface.co/cstr/funasr-nano-GGUF/resolve/main/funasr-nano-2512-q8_0.gguf",
        "ca_file":       "funasr-nano-2512-q8_0.gguf",
        "tc_url":        "https://huggingface.co/handy-computer/Fun-ASR-Nano-2512-gguf/resolve/main/Fun-ASR-Nano-2512-Q8_0.gguf",
        "tc_file":       "Fun-ASR-Nano-2512-Q8_0.gguf",
        "ca_wer_libri":  None,
        "tc_wer_libri":  "1.79%",
        "timeout_s":     120,
        "notes":         "Both Q8_0 (CrispASR uses Q8_0 to avoid CUDA F16×F32 saturation bug)",
    },
    {
        "family":        "Nemotron 3.5 ASR Streaming 0.6B",
        "ca_backend":    "nemotron",
        "ca_url":        "https://huggingface.co/cstr/nemotron-3.5-asr-streaming-0.6b-GGUF/resolve/main/nemotron-3.5-asr-streaming-0.6b-q4_k.gguf",
        "ca_file":       "nemotron-3.5-asr-streaming-0.6b-q4_k.gguf",
        "tc_url":        "https://huggingface.co/handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf/resolve/main/nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf",
        "tc_file":       "nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf",
        "ca_wer_libri":  None,
        "tc_wer_libri":  None,  # check docs
        "timeout_s":     120,
        "notes":         "CrispASR Q4_K vs t.cpp Q8_0",
    },
]

# Models that exist ONLY in transcribe.cpp (CrispASR coverage gaps)
TC_ONLY_MODELS = [
    {
        "family":        "GigaAM v3 E2E-CTC",
        "tc_url":        "https://huggingface.co/handy-computer/gigaam-v3-e2e-ctc-gguf/resolve/main/gigaam-v3-e2e-ctc-Q8_0.gguf",
        "tc_file":       "gigaam-v3-e2e-ctc-Q8_0.gguf",
        "tc_wer_libri":  "5.50%",
        "timeout_s":     90,
        "notes":         "Russian-focused + EN ASR; no CrispASR equivalent yet",
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _push_hf_progress():
    global _HF_LAST_PUSH
    if time.time() - _HF_LAST_PUSH < 30:
        return
    if not os.environ.get("HF_TOKEN") or not _PROGRESS.exists():
        return
    try:
        from huggingface_hub import HfApi
        HfApi(token=os.environ["HF_TOKEN"]).upload_file(
            path_or_fileobj=str(_PROGRESS),
            path_in_repo=_HF_PATH,
            repo_id=_HF_REPO,
            repo_type="dataset",
            commit_message=f"bench progress @ {int(time.time() - _T0)}s",
        )
        _HF_LAST_PUSH = time.time()
    except Exception:
        pass


def step(name, **extra):
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - _T0, 2),
        "step": name,
        **extra,
    }
    _PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with _PROGRESS.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(
        f"[step {rec['elapsed_s']:>7.1f}s] {name}"
        + (f"  {extra}" if extra else ""),
        flush=True,
    )
    _push_hf_progress()


def run_cmd(cmd: str, timeout: int = 600, stream: bool = False):
    """Run shell command; return (ok, stdout, stderr, elapsed_s)."""
    t0 = time.time()
    if stream:
        try:
            proc = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            stderr_buf = []
            import select
            while True:
                if time.time() - t0 > timeout:
                    proc.kill()
                    return False, "", "TIMEOUT", time.time() - t0
                ready, _, _ = select.select([proc.stderr], [], [], 1.0)
                if ready:
                    line = proc.stderr.readline()
                    if line:
                        stderr_buf.append(line)
                        print(f"  [build] {line.rstrip()}", flush=True)
                if proc.poll() is not None:
                    for line in proc.stderr:
                        stderr_buf.append(line)
                    break
            stdout = proc.stdout.read()
            return proc.returncode == 0, stdout, "".join(stderr_buf), time.time() - t0
        except Exception as e:
            return False, "", str(e), time.time() - t0
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return r.returncode == 0, r.stdout, r.stderr, time.time() - t0
    except subprocess.TimeoutExpired:
        return False, "", "TIMEOUT", time.time() - t0


def peak_rss_mb() -> float:
    """Read current process peak RSS in MB from /proc/self/status."""
    try:
        for line in open("/proc/self/status").readlines():
            if line.startswith("VmPeak:"):
                return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0


def wer_simple(ref: str, hyp: str) -> float:
    """Simple word error rate (substitutions + deletions + insertions) / len(ref)."""
    ref_words = ref.lower().split()
    hyp_words = hyp.lower().split()
    if not ref_words:
        return 0.0
    # DP levenshtein on words
    m, n = len(ref_words), len(hyp_words)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        new_dp = [i] + [0] * n
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                new_dp[j] = dp[j - 1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j - 1], dp[j - 1])
        dp = new_dp
    return dp[n] / m


def strip_tags(text: str) -> str:
    """Remove SenseVoice-style <|TAG|> tokens and normalise whitespace."""
    text = re.sub(r"<\|[^|]*\|>", "", text)
    return " ".join(text.split())


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, normalise whitespace for WER."""
    text = strip_tags(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ''-]", " ", text)
    return " ".join(text.split())


def download_gguf(url: str, dest: Path, timeout: int = 600) -> bool:
    """Download a GGUF file with wget (HF_HUB_ENABLE_HF_TRANSFER=1 for hf_transfer)."""
    if dest.exists():
        step(f"download.skip", file=dest.name, reason="already exists")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN", "")
    auth = f'--header "Authorization: Bearer {token}"' if token else ""
    ok, _, err, elapsed = run_cmd(
        f'wget -q --show-progress {auth} -O "{dest}" "{url}"',
        timeout=timeout,
    )
    if not ok or not dest.exists() or dest.stat().st_size < 1024:
        step("download.failed", url=url, error=err[-200:])
        dest.unlink(missing_ok=True)
        return False
    step("download.done", file=dest.name, size_mb=round(dest.stat().st_size / 1e6, 1),
         elapsed_s=round(elapsed, 1))
    return True


def run_crispasr(binary: Path, model: Path, audio: Path, backend: str | None,
                 timeout: int = 120) -> dict:
    """Run crispasr; return {"transcript": str, "infer_s": float, "rtf": float, "ok": bool}.

    CrispASR outputs the transcript to stdout and timing info to stderr:
      stderr: "crispasr: transcribed X.Xs audio in Y.YYs (Z.Zx realtime)"
    """
    b_flag = f"--backend {backend}" if backend else ""
    t0 = time.time()
    # Run with stdout and stderr separate
    # Use -ng (no GPU) to force CPU inference — transcribe.cpp also runs CPU
    # so this gives a fair comparison, and avoids P100 sm_60 CUDA issues.
    try:
        proc = subprocess.run(
            f'"{binary}" -m "{model}" {b_flag} -ng --auto-download "{audio}"',
            shell=True, capture_output=True, text=True, timeout=timeout
        )
        ok = proc.returncode == 0
        out = proc.stdout
        err = proc.stderr
    except subprocess.TimeoutExpired:
        return {"transcript": "", "infer_s": timeout, "rtf": None, "ok": False,
                "stderr_tail": "TIMEOUT"}
    infer_s = round(time.time() - t0, 3)

    # Extract transcript from stdout — CrispASR outputs plain text, one segment per line
    transcript = out.strip()

    # Extract RTF from stderr: "crispasr: transcribed X.Xs audio in Y.YYs (Z.Zx realtime)"
    rtf = None
    for line in err.splitlines():
        m = re.search(r"transcribed.*?(\d+\.?\d*)x realtime", line)
        if m:
            try:
                rtf_factor = float(m.group(1))
                rtf = round(1.0 / rtf_factor, 4)  # convert Nx to RTF (< 1 means faster than real-time)
            except Exception:
                pass

    return {
        "transcript": transcript,
        "infer_s": infer_s,
        "rtf": rtf,
        "ok": ok,
        "stderr_tail": err[-500:],  # always include for diagnostics
    }


def run_transcribe_cpp(binary: Path, model: Path, audio: Path,
                       timeout: int = 120) -> dict:
    """Run transcribe-cli in --batch-jsonl mode; return structured result.

    transcribe-cli single-file mode outputs human-readable text; batch-jsonl
    mode outputs clean JSON with per-utterance timings:
      line 0:  {"type":"batch_header","load_ms":...}
      line 1+: {"file":"...","text":"...","mel_ms":...,"encode_ms":...,"decode_ms":...}
    """
    import tempfile
    # Write a one-line batch file pointing at the audio file
    batch_file = Path(tempfile.mktemp(suffix=".txt", dir=str(audio.parent)))
    batch_file.write_text(str(audio) + "\n")
    t0 = time.time()
    try:
        ok, out, err, elapsed = run_cmd(
            f'"{binary}" -m "{model}" --batch "{batch_file}" --batch-jsonl 2>/dev/null',
            timeout=timeout,
        )
    finally:
        batch_file.unlink(missing_ok=True)
    infer_s = time.time() - t0

    transcript = ""
    mel_ms = encode_ms = decode_ms = load_ms = 0.0

    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") == "batch_header":
            load_ms = d.get("load_ms", 0)
        elif "text" in d:
            # Per-utterance result line
            transcript = d.get("text", "")
            mel_ms = d.get("mel_ms", 0)
            encode_ms = d.get("encode_ms", 0)
            decode_ms = d.get("decode_ms", 0)
            if d.get("error"):
                ok = False

    return {
        "transcript": transcript,
        "infer_s": round(infer_s, 3),
        "mel_ms": mel_ms,
        "encode_ms": encode_ms,
        "decode_ms": decode_ms,
        "load_ms": load_ms,
        "ok": ok,
        "stderr_tail": err[-300:] if not ok else "",
    }


# ── Stream results to HF dataset ─────────────────────────────────────────────
SWEEP_REPO = "cstr/crispasr-kaggle-progress"
RUN_TAG = f"transcribe-cpp-bench-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
SWEEP_PREFIX = f"transcribe-cpp-bench/{RUN_TAG}"
_hf_api = None
_sweep_ok = False


def _init_hf_streaming():
    global _hf_api, _sweep_ok
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[sweep] No HF_TOKEN — results will be LOCAL ONLY", flush=True)
        return
    try:
        from huggingface_hub import HfApi
        _hf_api = HfApi(token=token)
        _hf_api.upload_file(
            path_or_fileobj=io.BytesIO(b"alive\n"),
            path_in_repo=f"{SWEEP_PREFIX}/_heartbeat.txt",
            repo_type="dataset",
            repo_id=SWEEP_REPO,
            commit_message="transcribe-cpp bench heartbeat",
        )
        _sweep_ok = True
        print(f"[sweep] HF streaming OK → {SWEEP_REPO}/{SWEEP_PREFIX}", flush=True)
    except Exception as e:
        print(f"[sweep] HF streaming disabled: {e!r}", flush=True)


def sweep_publish(key: str, payload: dict):
    if not _sweep_ok or _hf_api is None:
        return
    data = json.dumps({**payload, "_run": RUN_TAG}, indent=2, default=str).encode()
    try:
        _hf_api.upload_file(
            path_or_fileobj=io.BytesIO(data),
            path_in_repo=f"{SWEEP_PREFIX}/results/{key}.json",
            repo_type="dataset",
            repo_id=SWEEP_REPO,
            commit_message=f"bench {RUN_TAG}: {key}",
        )
        print(f"  [sweep] ↑ {key}.json", flush=True)
    except Exception as e:
        print(f"  [sweep] ! upload failed: {e!r}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0: Install Python deps
# ─────────────────────────────────────────────────────────────────────────────
step("install_deps.begin")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "huggingface_hub", "hf_transfer",
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
step("install_deps.done")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Clone CrispASR (feature branch) + kaggle_harness
# ─────────────────────────────────────────────────────────────────────────────
CRISPASR_DIR = WORK / "CrispASR"
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
CRISPASR_BRANCH = "feat/transcribe-cpp-eval"

step("clone_crispasr.begin", branch=CRISPASR_BRANCH)
if CRISPASR_DIR.exists():
    run_cmd(
        f"cd {CRISPASR_DIR} && git fetch --depth 1 origin {CRISPASR_BRANCH} "
        f"&& git reset --hard FETCH_HEAD",
        timeout=120,
    )
    step("clone_crispasr.updated")
else:
    ok, _, err, _ = run_cmd(
        f"git clone --depth 1 --branch {CRISPASR_BRANCH} {CRISPASR_URL} {CRISPASR_DIR}",
        timeout=300,
    )
    if not ok:
        # Branch may not be pushed yet — fall back to main
        step("clone_crispasr.branch_missing", fallback="main")
        run_cmd(
            f"git clone --depth 1 {CRISPASR_URL} {CRISPASR_DIR}",
            timeout=300,
        )
    step("clone_crispasr.done")

# git clone --depth 1 does not pull submodules; ggml is one since 2026-07-07
# and cmake fails at add_subdirectory(ggml) without it (#238 lesson).
if not (CRISPASR_DIR / "ggml" / "CMakeLists.txt").exists():
    step("clone_crispasr.submodule_init.begin")
    subprocess.run(
        ["git", "-C", str(CRISPASR_DIR), "submodule", "update",
         "--init", "--recursive", "--depth", "1"],
        check=True,
    )
    step("clone_crispasr.submodule_init.done")

sys.path.insert(0, str(CRISPASR_DIR / "tools" / "kaggle"))
# Also accept bundled harness in same dir as this script
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress(progress_path=str(WORK / "progress.jsonl"))
kh.resolve_hf_token()

_init_hf_streaming()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Build CrispASR with CUDA (using kaggle_harness)
# ─────────────────────────────────────────────────────────────────────────────
kh.step("build_crispasr.begin")
_build_log = (WORK / "build_crispasr.log").open("a")

_tc = kh.install_build_toolchain()
has_gpu = os.path.exists("/usr/local/cuda/bin/nvcc")
CRISPASR_BUILD = CRISPASR_DIR / "build"
CRISPASR_BUILD.mkdir(exist_ok=True)

_cuda_arch = kh.detect_cuda_arch() if has_gpu else None
_cuda_flags = kh.cuda_build_flags(_cuda_arch) if has_gpu else []
_cache_flags = kh.cache_and_link_flags()
_jobs = kh.safe_build_jobs(gpu=has_gpu)
_gen = ["-G", "Ninja"] if _tc.get("ninja") else []

_cmake_ret = subprocess.run(
    ["cmake"] + _gen + [
        "-B", str(CRISPASR_BUILD),
        "-DCMAKE_BUILD_TYPE=Release",
    ] + _cuda_flags + _cache_flags + [str(CRISPASR_DIR)],
    stdout=_build_log, stderr=_build_log,
    env={**os.environ, "CCACHE_DIR": "/kaggle/working/.ccache"},
)
if _cmake_ret.returncode != 0:
    _build_log.flush()
    print("=== build_crispasr.log (tail) ===", flush=True)
    print(open(str(WORK / "build_crispasr.log")).read()[-4000:], flush=True)
    raise subprocess.CalledProcessError(_cmake_ret.returncode, "cmake configure (CrispASR)")

with kh.build_heartbeat("cmake.build.crispasr"):
    subprocess.run(
        ["cmake", "--build", str(CRISPASR_BUILD), f"-j{_jobs}", "--target", "crispasr"],
        check=True, stdout=_build_log, stderr=_build_log, timeout=30 * 60,
    )

CRISPASR_BIN = CRISPASR_BUILD / "bin" / "crispasr"
assert CRISPASR_BIN.is_file(), f"crispasr binary missing: {CRISPASR_BIN}"
kh.step("build_crispasr.done", binary=str(CRISPASR_BIN))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Clone and build transcribe.cpp
# ─────────────────────────────────────────────────────────────────────────────
TC_DIR = WORK / "transcribe.cpp"
TC_URL = "https://github.com/handy-computer/transcribe.cpp.git"

step("clone_transcribecpp.begin")
if TC_DIR.exists():
    run_cmd(
        f"cd {TC_DIR} && git fetch --depth 1 origin main && git reset --hard FETCH_HEAD",
        timeout=120,
    )
    step("clone_transcribecpp.updated")
else:
    ok, _, err, _ = run_cmd(
        f"git clone --depth 1 {TC_URL} {TC_DIR}", timeout=300
    )
    assert ok, f"transcribe.cpp clone failed: {err[-500:]}"
    step("clone_transcribecpp.done")

step("build_transcribecpp.begin")
_tc_build_log = (WORK / "build_transcribecpp.log").open("a")
TC_BUILD = TC_DIR / "build"

_tc_cuda_flags = ["-DTRANSCRIBE_CUDA=ON"] if has_gpu else []
# Pin the same arch for transcribe.cpp to avoid fat-binary OOM.
# Also set CMAKE_CUDA_COMPILER explicitly — without it cmake may not create
# the CUDA::cuda_driver imported target, causing "target not found" at
# Generate step.  Point at the driver stub so the linker target resolves.
if has_gpu and _cuda_arch:
    _tc_cuda_flags += [
        f"-DCMAKE_CUDA_ARCHITECTURES={_cuda_arch}",
        "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
        "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache",
        # Kaggle P100: libcuda.so lives only in the stubs dir
        "-DCUDA_DRIVER_LIBRARY=/usr/local/cuda/lib64/stubs/libcuda.so",
    ]

def _cmake_configure_tc(cuda_flags):
    return subprocess.run(
        ["cmake"] + _gen + [
            "-B", str(TC_BUILD),
            "-DCMAKE_BUILD_TYPE=Release",
        ] + cuda_flags + _cache_flags + [str(TC_DIR)],
        stdout=_tc_build_log, stderr=_tc_build_log,
        env={**os.environ, "CCACHE_DIR": "/kaggle/working/.ccache"},
    )

_tc_cmake_ret = _cmake_configure_tc(_tc_cuda_flags)
if _tc_cmake_ret.returncode != 0 and _tc_cuda_flags:
    # CUDA configure failed — fall back to CPU-only build
    _tc_build_log.flush()
    print("=== build_transcribecpp.log (tail, CUDA attempt) ===", flush=True)
    print(open(str(WORK / "build_transcribecpp.log")).read()[-2000:], flush=True)
    step("build_transcribecpp.cuda_fallback_cpu")
    import shutil as _shutil
    _shutil.rmtree(TC_BUILD, ignore_errors=True)
    TC_BUILD.mkdir(exist_ok=True)
    _tc_cmake_ret = _cmake_configure_tc([])  # CPU-only retry

if _tc_cmake_ret.returncode != 0:
    _tc_build_log.flush()
    print("=== build_transcribecpp.log (tail) ===", flush=True)
    print(open(str(WORK / "build_transcribecpp.log")).read()[-4000:], flush=True)
    raise subprocess.CalledProcessError(_tc_cmake_ret.returncode, "cmake configure (transcribe.cpp)")

with kh.build_heartbeat("cmake.build.transcribecpp"):
    subprocess.run(
        ["cmake", "--build", str(TC_BUILD), f"-j{_jobs}", "--target", "transcribe-cli"],
        check=True, stdout=_tc_build_log, stderr=_tc_build_log, timeout=30 * 60,
    )

TC_BIN = TC_BUILD / "bin" / "transcribe-cli"
assert TC_BIN.is_file(), f"transcribe-cli binary missing: {TC_BIN}"
step("build_transcribecpp.done", binary=str(TC_BIN))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Prepare test audio
# ─────────────────────────────────────────────────────────────────────────────
JFK_WAV = WORK / "jfk.wav"
# Use jfk.wav from CrispASR samples (16 kHz mono WAV)
if not JFK_WAV.exists():
    shutil.copy(CRISPASR_DIR / "samples" / "jfk.wav", JFK_WAV)
step("audio.ready", file=str(JFK_WAV))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Run head-to-head benchmark on shared models
# ─────────────────────────────────────────────────────────────────────────────

MODELS_DIR = WORK / "models"
MODELS_DIR.mkdir(exist_ok=True)
results = []

for m in SHARED_MODELS:
    fam = m["family"]
    if time.time() - _T0 > MAX_WALL_S:
        step("bench.wall_limit_reached", remaining_models=fam)
        break

    step(f"bench.start", family=fam)
    ca_gguf = MODELS_DIR / m["ca_file"]
    tc_gguf = MODELS_DIR / m["tc_file"]

    # Download both GGUFs
    ca_dl = download_gguf(m["ca_url"], ca_gguf, timeout=600)
    tc_dl = download_gguf(m["tc_url"], tc_gguf, timeout=600)

    row = {
        "family": fam,
        "notes": m["notes"],
        "ca_wer_libri_docs": m["ca_wer_libri"],
        "tc_wer_libri_docs": m["tc_wer_libri"],
        "ca_model": m["ca_file"],
        "tc_model": m["tc_file"],
    }

    # ── CrispASR inference ──────────────────────────────────────────────
    if ca_dl:
        ca_res = run_crispasr(
            CRISPASR_BIN, ca_gguf, JFK_WAV,
            backend=m["ca_backend"],
            timeout=m["timeout_s"],
        )
        ca_norm = normalise(ca_res["transcript"])
        ca_wer = wer_simple(JFK_REF, ca_norm)
        # Prefer RTF from CrispASR stderr (inference-only) over wall-clock
        ca_rtf = ca_res.get("rtf") or round(ca_res["infer_s"] / JFK_DURATION_S, 3)
        row.update({
            "ca_ok": ca_res["ok"],
            "ca_transcript": ca_res["transcript"],
            "ca_transcript_norm": ca_norm,
            "ca_jfk_wer": round(ca_wer, 4),
            "ca_infer_s": ca_res["infer_s"],
            "ca_rtf": ca_rtf,
            "ca_stderr_tail": ca_res.get("stderr_tail", ""),
        })
        step(f"bench.ca_done", family=fam, ok=ca_res["ok"],
             rtf=ca_rtf, wer=row["ca_jfk_wer"],
             transcript=ca_norm[:80])
    else:
        row.update({"ca_ok": False, "ca_transcript": "", "ca_jfk_wer": None,
                    "ca_infer_s": None, "ca_rtf": None})
        step(f"bench.ca_download_failed", family=fam)

    # ── transcribe.cpp inference ────────────────────────────────────────
    if tc_dl:
        tc_res = run_transcribe_cpp(
            TC_BIN, tc_gguf, JFK_WAV, timeout=m["timeout_s"]
        )
        tc_norm = normalise(tc_res["transcript"])
        tc_wer = wer_simple(JFK_REF, tc_norm)
        # RTF from t.cpp detailed timing (mel+encode+decode) is more accurate than wall-clock
        tc_infer_ms = tc_res["mel_ms"] + tc_res["encode_ms"] + tc_res["decode_ms"]
        tc_rtf = (round(tc_infer_ms / (JFK_DURATION_S * 1000), 3)
                  if tc_infer_ms > 0 else round(tc_res["infer_s"] / JFK_DURATION_S, 3))
        row.update({
            "tc_ok": tc_res["ok"],
            "tc_transcript": tc_res["transcript"],
            "tc_transcript_norm": tc_norm,
            "tc_jfk_wer": round(tc_wer, 4),
            "tc_infer_s": tc_res["infer_s"],
            "tc_rtf": tc_rtf,
            "tc_mel_ms": tc_res["mel_ms"],
            "tc_encode_ms": tc_res["encode_ms"],
            "tc_decode_ms": tc_res["decode_ms"],
            "tc_load_ms": tc_res["load_ms"],
            "tc_stderr_tail": tc_res.get("stderr_tail", ""),
        })
        step(f"bench.tc_done", family=fam, ok=tc_res["ok"],
             rtf=tc_rtf, wer=row["tc_jfk_wer"],
             transcript=tc_norm[:80])
    else:
        row.update({"tc_ok": False, "tc_transcript": "", "tc_jfk_wer": None,
                    "tc_infer_s": None, "tc_rtf": None})
        step(f"bench.tc_download_failed", family=fam)

    # ── Print side-by-side comparison ──────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  {fam}")
    print(f"  Reference: {JFK_REF}")
    print(f"  CrispASR : {row.get('ca_transcript_norm', '(failed)')}  [RTF={row.get('ca_rtf')}, WER={row.get('ca_jfk_wer')}]")
    print(f"  t.cpp    : {row.get('tc_transcript_norm', '(failed)')}  [RTF={row.get('tc_rtf')}, WER={row.get('tc_jfk_wer')}]")
    print(f"{'='*72}\n", flush=True)

    results.append(row)
    sweep_publish(f"shared__{fam.replace(' ', '_')}", row)

    # Clean up both GGUFs to reclaim disk space
    ca_gguf.unlink(missing_ok=True)
    tc_gguf.unlink(missing_ok=True)
    step(f"bench.cleanup", family=fam)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: transcribe.cpp-only models (CrispASR coverage gaps)
# ─────────────────────────────────────────────────────────────────────────────
tc_only_results = []
for m in TC_ONLY_MODELS:
    fam = m["family"]
    if time.time() - _T0 > MAX_WALL_S:
        step("bench.wall_limit_reached", remaining_models=fam)
        break
    step(f"bench.tc_only.start", family=fam)
    tc_gguf = MODELS_DIR / m["tc_file"]
    dl = download_gguf(m["tc_url"], tc_gguf, timeout=600)
    row = {
        "family": fam,
        "notes": m["notes"],
        "tc_wer_libri_docs": m["tc_wer_libri"],
        "tc_model": m["tc_file"],
        "ca_equivalent": None,
    }
    if dl:
        tc_res = run_transcribe_cpp(TC_BIN, tc_gguf, JFK_WAV, timeout=m["timeout_s"])
        tc_norm = normalise(tc_res["transcript"])
        tc_wer = wer_simple(JFK_REF, tc_norm)
        row.update({
            "tc_ok": tc_res["ok"],
            "tc_transcript": tc_res["transcript"],
            "tc_transcript_norm": tc_norm,
            "tc_jfk_wer": round(tc_wer, 4),
            "tc_infer_s": tc_res["infer_s"],
            "tc_rtf": round(tc_res["infer_s"] / JFK_DURATION_S, 3),
        })
        print(f"\n  {fam} (t.cpp only)")
        print(f"  t.cpp: {tc_norm}  [RTF={row['tc_rtf']}, WER={row['tc_jfk_wer']}]")
        step(f"bench.tc_only.done", family=fam, rtf=row["tc_rtf"], wer=row["tc_jfk_wer"])
    else:
        row.update({"tc_ok": False, "tc_jfk_wer": None, "tc_rtf": None})
        step(f"bench.tc_only.download_failed", family=fam)

    tc_only_results.append(row)
    sweep_publish(f"tc_only__{fam.replace(' ', '_')}", row)
    tc_gguf.unlink(missing_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Print final summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("BENCHMARK SUMMARY — CrispASR vs transcribe.cpp")
print(f"GPU: {_cuda_arch or 'CPU only'}  | Branch: {CRISPASR_BRANCH}")
print("=" * 80)
print(f"{'Family':<32} {'CA RTF':>8} {'TC RTF':>8} {'CA WER':>8} {'TC WER':>8}  CA  TC")
print("-" * 80)
for row in results:
    ca_rtf = f"{row.get('ca_rtf', ''):.3f}" if row.get("ca_rtf") is not None else "FAIL"
    tc_rtf = f"{row.get('tc_rtf', ''):.3f}" if row.get("tc_rtf") is not None else "FAIL"
    ca_wer = f"{row.get('ca_jfk_wer', 0)*100:.1f}%" if row.get("ca_jfk_wer") is not None else "FAIL"
    tc_wer = f"{row.get('tc_jfk_wer', 0)*100:.1f}%" if row.get("tc_jfk_wer") is not None else "FAIL"
    ca_ok = "✓" if row.get("ca_ok") else "✗"
    tc_ok = "✓" if row.get("tc_ok") else "✗"
    print(f"{row['family']:<32} {ca_rtf:>8} {tc_rtf:>8} {ca_wer:>8} {tc_wer:>8}  {ca_ok}   {tc_ok}")
print("=" * 80)

if tc_only_results:
    print("\ntranscribe.cpp-only models (CrispASR coverage gaps):")
    for row in tc_only_results:
        tc_rtf = f"{row.get('tc_rtf', ''):.3f}" if row.get("tc_rtf") is not None else "FAIL"
        tc_wer = f"{row.get('tc_jfk_wer', 0)*100:.1f}%" if row.get("tc_jfk_wer") is not None else "FAIL"
        print(f"  {row['family']:<40} RTF={tc_rtf}  WER={tc_wer}")

print(f"\nTotal elapsed: {round(time.time() - _T0, 0):.0f}s", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: Save and upload full summary JSON
# ─────────────────────────────────────────────────────────────────────────────
summary = {
    "run_tag": RUN_TAG,
    "branch": CRISPASR_BRANCH,
    "gpu": _cuda_arch,
    "elapsed_s": round(time.time() - _T0, 1),
    "shared_models": results,
    "tc_only_models": tc_only_results,
}
summary_path = WORK / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2, default=str))
step("summary.written", path=str(summary_path))
sweep_publish("summary", summary)
step("bench.complete", total_models=len(results) + len(tc_only_results))
