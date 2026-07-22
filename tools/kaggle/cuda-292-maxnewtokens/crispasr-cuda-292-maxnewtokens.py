"""CUDA live validation: #292 --max-new-tokens forwarding + #290 canary-qwen long audio.

#292 shipped as source + CI-compile + unit tests, but the ACCEPTANCE TEST — does
the decode cap actually track --max-new-tokens on real audio — never ran (the dev
box was out of memory). This kernel is that test, on CUDA, under the full harness.

moss-diarize IS the reporter's exact backend and it IS published
(cstr/MOSS-Transcribe-Diarize-GGUF, q4_k per the "doable with q4k" ask), so it is
tested directly. canary-qwen (also carries the #290 fix) and mimo-asr confirm the
same hardcoded-cap fix generalises.

DECIDABLE ACCEPTANCE GATES (never "sounds right"):

  #292  Single-pass (--chunk-seconds 0), same ~55 s clip, CPU and CUDA. Run with
        --max-new-tokens 64 (must truncate) and --max-new-tokens 4096 (must not).
        Gate: the 4096 run emits substantially more words (>2x, +20). Equal counts
        mean the flag is ignored — the bug. This proves the flag is honored in
        BOTH directions and does not depend on the backend default or clip length.

  #290  canary-qwen on ~176 s with DEFAULT handling (chunking, no --chunk-seconds
        0) must complete non-empty and without the O(T^2) blowup (was 384 MiB ->
        10.2 GiB). Peak RSS recorded; crash/OOM is FAIL.

  chunk_id  A chunked moss-diarize run's .json must carry chunk_id with >1 distinct
        value — the reporter's part 2: speaker labels are chunk-local, so a
        consumer needs chunk_id to tell continuity from an ID swap.

  tabcnn  New backend: --tab decided frets must AGREE between CPU and CUDA (port
        correctness, decidable), independent of musical accuracy.

PROOF OF WORK (kaggle_usage.md gotcha #24): nothing is reported unless it also
produced output. Non-zero exit or empty transcript is FAIL, never a speedup.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
REPO = TEMP / "CrispASR"
BUILD = TEMP / "build"
MODELS = TEMP / "models"
FIX = TEMP / "fixtures"
OUT = WORK / "outputs"
CRISPASR_REPO = "https://github.com/CrispStrobe/CrispASR.git"
CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
for d in (MODELS, FIX, OUT):
    d.mkdir(parents=True, exist_ok=True)

RESULTS = {"tests": [], "bench": [], "env": {}}
_T0 = time.time()


def run(cmd, check=True, env=None, cwd=None, timeout=None, capture=True):
    e = {**os.environ, **(env or {})}
    if capture:
        r = subprocess.run(cmd, env=e, cwd=cwd, timeout=timeout, shell=isinstance(cmd, str),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    else:
        r = subprocess.run(cmd, env=e, cwd=cwd, timeout=timeout, shell=isinstance(cmd, str))
        r.stdout = ""
    if check and r.returncode != 0:
        print((r.stdout or "")[-8000:], flush=True)
        raise SystemExit(f"command failed ({r.returncode}): {cmd}")
    return r


def record(name, passed, **kw):
    entry = {"test": name, "pass": bool(passed), **kw}
    RESULTS["tests"].append(entry)
    print(("PASS  " if passed else "FAIL  ") + name + "  " + json.dumps(kw), flush=True)
    (WORK / "results.json").write_text(json.dumps(RESULTS, indent=2))
    return passed


# ── cell 1: clone + harness (FULL regime) ──────────────────────────────────
print(json.dumps({"step": "start", "ref": CRISPASR_REF}), flush=True)
if REPO.exists():
    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, "--recursive",
     CRISPASR_REPO, str(REPO)], capture=False)
run(["git", "-C", str(REPO), "submodule", "update", "--init", "--recursive"],
    check=False, capture=False, timeout=1800)

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
kh.step("cloned", sha=sha)
RESULTS["env"]["sha"] = sha
RESULTS["env"]["ref"] = CRISPASR_REF
run(["nvidia-smi", "-L"], check=False)
gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
kh.step("gpu", gpu=gpu)
RESULTS["env"]["gpu"] = gpu

# ── cell 2: build (CUDA) ───────────────────────────────────────────────────
kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
kh.step("cuda_arch", arch=arch)
RESULTS["env"]["cuda_arch"] = arch
BUILD.mkdir(parents=True, exist_ok=True)
cmake_args = [
    "cmake", "-S", str(REPO), "-B", str(BUILD),
    "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON",
    "-DCRISPASR_NO_C2PA_NATIVE=ON",
] + kh.cuda_build_flags(arch) + kh.cache_and_link_flags()
run(cmake_args, capture=False)
kh.step("cmake_done")
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli "
        f"-j{kh.safe_build_jobs(gpu=True)}")
kh.step("build_done")

CLI = BUILD / "examples" / "cli" / "crispasr"
if not CLI.exists():
    cands = [c for c in BUILD.rglob("crispasr") if c.is_file() and os.access(c, os.X_OK)]
    if not cands:
        raise SystemExit("crispasr binary not found after build")
    CLI = cands[0]
os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
kh.step("cli", path=str(CLI))

# ── cell 3: downloads ──────────────────────────────────────────────────────
token = kh.resolve_hf_token()
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
kh.step("hf_token", have=bool(token))


def hf_get(repo, filename, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    hdr = ["-H", f"Authorization: Bearer {token}"] if token else []
    run(["curl", "-fL", "-C", "-", "--retry", "5", "--retry-delay", "5",
         "-o", str(dest), *hdr, url], capture=False, timeout=3600)
    kh.step("download.done", file=dest.name, mb=round(dest.stat().st_size / 1e6, 1))
    return dest


M = {}
# The reporter's EXACT backend (#292 was filed against moss-diarize). q4_k per the
# "doable with q4k" ask. Tests both parts: --max-new-tokens AND chunk_id/speaker.
M["moss_diarize"] = hf_get("cstr/MOSS-Transcribe-Diarize-GGUF",
                           "moss-transcribe-diarize-0.9b-q4_k.gguf",
                           MODELS / "moss-diarize-q4_k.gguf")
# Pattern-confirmation backends (same hardcoded-cap fix): canary-qwen also carries
# the #290 fix, mimo-asr is a second independent confirmation.
M["canary_qwen"] = hf_get("cstr/canary-qwen-2.5b-GGUF", "canary-qwen-2.5b-q8_0.gguf",
                          MODELS / "canary-qwen-q8_0.gguf")
M["mimo"] = hf_get("cstr/mimo-asr-GGUF", "mimo-asr-q4_k.gguf", MODELS / "mimo-asr-q4_k.gguf")
M["tabcnn"] = hf_get("cstr/tabcnn-GGUF", "tabcnn-f16.gguf", MODELS / "tabcnn-f16.gguf")
kh.step("downloads_done")

# ── cell 4: fixtures ───────────────────────────────────────────────────────
JFK = REPO / "samples" / "jfk.wav"
# MED (~55 s): the #292 single-pass cap test. Long enough that cap-4096 >> cap-64,
# short enough that a single FastConformer pass fits GPU memory on every backend.
# LONG (~176 s): the #290 default-chunking test — must stay bounded via chunking.
MED = FIX / "jfk_med.wav"
LONG = FIX / "jfk_long.wav"
run(["ffmpeg", "-y", "-stream_loop", "4", "-i", str(JFK), "-ar", "16000", "-ac", "1", str(MED)], capture=False)
run(["ffmpeg", "-y", "-stream_loop", "15", "-i", str(JFK), "-ar", "16000", "-ac", "1", str(LONG)], capture=False)


def wav_seconds(p):
    with wave.open(str(p), "rb") as w:
        return w.getnframes() / w.getframerate()


kh.step("fixtures", med_s=round(wav_seconds(MED), 1), long_s=round(wav_seconds(LONG), 1))


def transcribe(model, backend, wav, device, extra=None, timeout=1200):
    """Run the CLI; return (ok, text, wall_s, peak_kb).

    Device is selected by the flag, not an env var: CPU = --no-gpu, CUDA = default
    (the only GPU backend built on Kaggle). There is no CRISPASR_DEVICE knob.
    """
    args = [str(CLI), "-m", str(model), "--backend", backend, "-f", str(wav), "--no-prints"]
    if device == "cpu":
        args += ["--no-gpu"]
    if extra:
        args += extra
    t0 = time.time()
    # /usr/bin/time -v gives peak RSS; guard its absence.
    timed = ["/usr/bin/time", "-v"] + args if Path("/usr/bin/time").exists() else args
    r = run(timed, check=False, timeout=timeout)
    wall = time.time() - t0
    out = r.stdout or ""
    peak_kb = 0
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", out)
    if m:
        peak_kb = int(m.group(1))
    # strip the /usr/bin/time trailer to isolate the transcript
    text = re.split(r"\n\tCommand being timed:", out)[0]
    text = re.sub(r"\n\s*(User time|System time|Percent of CPU|Elapsed|Maximum resident|.*resident set|.*page|.*Context|.*Swaps|.*I/O|.*Signals|.*socket|.*Exit status).*", "", text)
    ok = (r.returncode == 0) and bool(text.strip())
    return ok, text.strip(), wall, peak_kb


def word_count(t):
    return len(re.findall(r"\S+", t))


# ── cell 5: #292 — --max-new-tokens is HONORED (both directions) ────────────
# The decidable gate does not depend on audio length or the backend's default:
# a SMALL explicit cap (64) must truncate hard, a LARGE one (4096) must not. If
# the flag were ignored (the bug) the two runs would emit the SAME word count.
# Both single-pass (--chunk-seconds 0) so the one decode pass is bounded by the
# flag directly. moss-diarize first (the reporter's exact backend), then
# canary-qwen (also carries the #290 fix) and mimo-asr confirm the pattern.
for key, backend in [("moss_diarize", "moss-diarize"), ("canary_qwen", "canary-qwen"),
                     ("mimo", "mimo-asr")]:
    for device in ["cpu", "cuda"]:
        with kh.build_heartbeat(f"292.{backend}.{device}"):
            ok_lo, lo_txt, _, _ = transcribe(M[key], backend, MED, device,
                                             extra=["--chunk-seconds", "0", "--max-new-tokens", "64"])
            ok_hi, hi_txt, _, _ = transcribe(M[key], backend, MED, device,
                                             extra=["--chunk-seconds", "0", "--max-new-tokens", "4096"])
        n_lo, n_hi = word_count(lo_txt), word_count(hi_txt)
        # Gate: both produce output AND the large-cap run emits substantially more
        # (>2x and at least 20 extra words) — equal counts mean the flag is ignored.
        passed = ok_lo and ok_hi and n_hi > max(2 * n_lo, n_lo + 20)
        record(f"292:{backend}:{device}:max_new_tokens_honored", passed,
               words_cap64=n_lo, words_cap4096=n_hi, ratio=round(n_hi / max(n_lo, 1), 1),
               note="cap 64 truncates, cap 4096 does not; equal = flag ignored (the bug)")
        RESULTS["bench"].append({"test": f"292:{backend}:{device}", "words_cap64": n_lo, "words_cap4096": n_hi})

# ── cell 6: #290 — canary-qwen long audio stays bounded, non-empty ─────────
for device in ["cpu", "cuda"]:
    with kh.build_heartbeat(f"290.canary_qwen.{device}"):
        ok, txt, wall, peak_kb = transcribe(M["canary_qwen"], "canary-qwen", LONG, device, timeout=1800)
    peak_gb = round(peak_kb / 1e6, 2)
    # #290 regression was 10.2 GiB on ~55 s; a 176 s clip that chunks properly
    # must stay well under that. Gate: completes, non-empty, peak < 8 GiB.
    passed = ok and (peak_kb == 0 or peak_gb < 8.0)
    record(f"290:canary_qwen:{device}:long_audio_bounded", passed,
           words=word_count(txt), peak_gb=peak_gb, wall_s=round(wall, 1),
           note="default long-audio handling: non-empty + no O(T^2) blowup")

# ── cell 7: chunk_id + speaker in the JSON of a chunked moss-diarize run ────
# The reporter's part 2: with chunking, "(speaker N)" restarts per chunk and the
# merged JSON must carry chunk_id so a consumer can tell continuity from an ID
# swap. moss-diarize is the diarize backend, so this is the real scenario. The
# reporter reached 298 s with --chunk-seconds 120 --chunk-overlap 3.
with kh.build_heartbeat("chunk_id"):
    jbase = OUT / "chunked_diarize"
    r = run([str(CLI), "-m", str(M["moss_diarize"]), "--backend", "moss-diarize", "-f", str(LONG),
             "--chunk-seconds", "60", "--chunk-overlap", "3", "--diarize",
             "--output-json", "-of", str(jbase), "--no-prints"],
            check=False, timeout=1800)
    jf = Path(str(jbase) + ".json")
    body = jf.read_text() if jf.exists() else ""
    has_chunk_id = "chunk_id" in body
    has_multiple_chunks = len(set(re.findall(r'"chunk_id":\s*(\d+)', body))) > 1
record("chunk_id:present_in_multichunk_diarize_json", has_chunk_id and has_multiple_chunks and r.returncode == 0,
       json=str(jf), distinct_chunk_ids=len(set(re.findall(r'"chunk_id":\s*(\d+)', body))),
       note="chunk_id must appear with >1 distinct value on a chunked diarize run")

# ── cell 8: tabcnn --tab CPU vs CUDA parity (new backend port) ─────────────
# --tab prints one line per frame: "time<TAB>fret fret fret fret fret fret" (the
# argmax-decided frets, low string first, "-" = silent). The decidable parity gate
# is that the per-frame FRET columns agree between CPU and CUDA — sub-argmax
# emission noise is invisible, which is correct for a tablature backend.
with kh.build_heartbeat("tabcnn.parity"):
    def tab_frets(device):
        args = [str(CLI), "-m", str(M["tabcnn"]), "--tab", "-f", str(JFK)]
        if device == "cpu":
            args += ["--no-gpu"]
        r = run(args, check=False, timeout=600)
        # keep only the frame lines (start with a float time then tab-separated frets)
        frets = [ln.split("\t", 1)[1] for ln in (r.stdout or "").splitlines()
                 if "\t" in ln and re.match(r"^\d+\.\d+\t", ln)]
        return r.returncode == 0, frets
    ok_cpu, f_cpu = tab_frets("cpu")
    ok_cuda, f_cuda = tab_frets("cuda")
    agree = ok_cpu and ok_cuda and len(f_cpu) > 0 and f_cpu == f_cuda
    mismatches = sum(1 for a, b in zip(f_cpu, f_cuda) if a != b) if (f_cpu and f_cuda) else -1
record("tabcnn:cpu_cuda_parity", agree, n_frames_cpu=len(f_cpu), n_frames_cuda=len(f_cuda),
       mismatched_frames=mismatches, note="decided frets must match across devices (port correctness)")

# ── summary ────────────────────────────────────────────────────────────────
npass = sum(1 for t in RESULTS["tests"] if t["pass"])
ntot = len(RESULTS["tests"])
RESULTS["summary"] = {"passed": npass, "total": ntot, "wall_s": round(time.time() - kh._t0 if hasattr(kh, "_t0") else 0, 1)}
(WORK / "results.json").write_text(json.dumps(RESULTS, indent=2))
print(json.dumps({"step": "done", "passed": npass, "total": ntot}), flush=True)
kh.step("done", passed=npass, total=ntot)
