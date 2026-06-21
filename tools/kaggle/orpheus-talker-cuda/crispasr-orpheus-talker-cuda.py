"""
CrispASR -- orpheus talker CUDA pipeline: build + quantize + diff + upload.

Orpheus passes on M1 Metal/CPU but the sweep reported a 0-byte on CUDA
(PLAN §201). This kernel, in one CUDA run:

  1. Build CrispASR (CUDA): crispasr-diff + crispasr-quantize, warm ccache.
  2. Download unsloth/orpheus-3b-0.1-ft (tara) + convert to F16 GGUF.
  3. Quantize -> Q8_0 and Q4_K.
  4. Generate the talker reference (greedy codec stream, PyTorch F32 on GPU).
  5. Diff the talker AR decode: CPU vs GPU vs ground truth (localizes any
     CUDA-specific talker divergence), and the SNAC vocoder on GPU.
  6. Upload f16/q8_0/q4_k GGUFs + the talker ref to cstr/orpheus-3b-0.1-ft-GGUF.
  7. Save the warmed ccache back to /kaggle/working/ccache.tar (download it
     and `datasets version` both crispasr-ccache copies).
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
RESULTS = WORK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
HF_DIR = WORK / "orpheus-3b-hf"
F16 = WORK / "orpheus-3b-0.1-ft-f16.gguf"
Q8 = WORK / "orpheus-3b-0.1-ft-q8_0.gguf"
Q4 = WORK / "orpheus-3b-0.1-ft-q4_k.gguf"
REF = WORK / "orpheus-talker-ref.gguf"

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
CRISPASR_REPO = os.environ.get("CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git")
HF_MODEL = os.environ.get("ORPHEUS_HF_MODEL", "unsloth/orpheus-3b-0.1-ft")
HF_REPO = os.environ.get("ORPHEUS_HF_REPO", "cstr/orpheus-3b-0.1-ft-GGUF")
TEXT = os.environ.get("ORPHEUS_TEXT", "Hey there, my name is Tara.")
SPEAKER = os.environ.get("ORPHEUS_SPEAKER", "tara")
MAXGEN = os.environ.get("ORPHEUS_DIFF_MAXGEN", "48")
DO_UPLOAD = os.environ.get("ORPHEUS_UPLOAD", "1") != "0"

PROGRESS = RESULTS / "progress.jsonl"
_T0 = time.time()


def step(name, **kv):
    line = json.dumps({"t": round(time.time() - _T0, 2), "step": name, **kv})
    print(f"[step] {line}", flush=True)
    with open(PROGRESS, "a") as f:
        f.write(line + "\n")


def run(cmd, check=True, env=None, cwd=None, timeout=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    e = os.environ.copy()
    if env:
        e.update(env)
    r = subprocess.run(cmd, env=e, cwd=cwd, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
    return r


# ── Clone + build ──────────────────────────────────────────────────────────
step("start", ref=CRISPASR_REF)
if REPO.exists():
    import shutil
    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, "--recursive",
     CRISPASR_REPO, str(REPO)])
sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
step("cloned", sha=sha)

run(["nvidia-smi", "-L"])
gpu_name = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
step("gpu", gpu_name=gpu_name)

kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
step("cuda_arch", arch=arch)

BUILD.mkdir(parents=True, exist_ok=True)
run(["cmake", "-S", str(REPO), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release",
     "-DBUILD_SHARED_LIBS=ON"] + kh.cuda_build_flags(arch) + kh.cache_and_link_flags())
step("cmake_done")
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-diff crispasr-quantize "
        f"-j{kh.safe_build_jobs(gpu=True)}")
step("build_done")


def _find(name):
    p = BUILD / "bin" / name
    if p.exists():
        return p
    c = [x for x in BUILD.rglob(name) if x.is_file() and os.access(x, os.X_OK)]
    if not c:
        raise SystemExit(f"{name} not found after build")
    return c[0]


DIFF = _find("crispasr-diff")
QUANT = _find("crispasr-quantize")
step("binaries", diff=str(DIFF), quant=str(QUANT))
os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"

# ── Save warmed ccache back (download as output, then `datasets version`) ────
def save_ccache():
    cc = WORK / ".ccache"
    if cc.exists():
        run(["tar", "cf", str(WORK / "ccache.tar"), "-C", str(WORK), ".ccache"], check=False)
        step("ccache_saved", mb=round((WORK / "ccache.tar").stat().st_size / 1e6, 1)
             if (WORK / "ccache.tar").exists() else 0)


# ── Download HF model + convert + quantize ─────────────────────────────────
hf_token = kh.resolve_hf_token()
step("hf_token", have=bool(hf_token))
from huggingface_hub import snapshot_download  # noqa: E402
snapshot_download(HF_MODEL, local_dir=str(HF_DIR), token=hf_token,
                  allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*", "*.txt"])
step("hf_downloaded")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gguf"], check=False)
run([sys.executable, str(REPO / "models" / "convert-orpheus-to-gguf.py"),
     "--input", str(HF_DIR), "--output", str(F16), "--outtype", "f16", "--variant", "fixed_speaker"],
    env={"OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
step("converted_f16", mb=round(F16.stat().st_size / 1e6, 1))

import gguf as _g  # noqa: E402
_r = _g.GGUFReader(str(F16))
offset = int(_r.fields["orpheus.custom_token_offset"].parts[-1][0])
count = int(_r.fields["orpheus.custom_token_count"].parts[-1][0])
step("codec_range", offset=offset, count=count)

for out, ftype in [(Q8, "q8_0"), (Q4, "q4_k")]:
    run([str(QUANT), str(F16), str(out), ftype])
    step(f"quantized_{ftype}", mb=round(out.stat().st_size / 1e6, 1))

save_ccache()  # cache is warm after the build; snapshot it now

# ── Generate talker reference (PyTorch F32 on GPU) ─────────────────────────
with kh.build_heartbeat("ref.gen"):
    run([sys.executable, str(REPO / "tools" / "dump_reference.py"),
         "--backend", "orpheus-talker", "--model-dir", str(HF_DIR),
         "--audio", str(REPO / "samples" / "jfk.wav"), "--output", str(REF),
         "--max-new-tokens", MAXGEN],
        env={"ORPHEUS_TEXT": TEXT, "ORPHEUS_SPEAKER": SPEAKER,
             "ORPHEUS_CUSTOM_OFFSET": str(offset), "ORPHEUS_CUSTOM_COUNT": str(count),
             "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1"})
step("ref_generated", ref_kb=round(REF.stat().st_size / 1e3, 1))


# ── Talker diff: CPU vs GPU vs ground truth ────────────────────────────────
def run_diff(label, backend, model, ref, extra_env, timeout=900):
    step(f"{label}_start")
    cmd = [str(DIFF), backend, str(model), str(ref), str(REPO / "samples" / "jfk.wav")]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, timeout=timeout, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, env={**os.environ, **extra_env})
        rc, out = r.returncode, r.stdout
    except subprocess.TimeoutExpired as ex:
        rc = -1
        out = ex.stdout.decode(errors="replace") if isinstance(ex.stdout, bytes) else (ex.stdout or "")
    (RESULTS / f"{label}.txt").write_text(out or "")
    print((out or "")[-3500:], flush=True)
    verdict = "PASS" if "→ PASS" in (out or "") or "0 fail" in (out or "") else \
        ("FAIL" if "FAIL" in (out or "") or "fail" in (out or "") else "?")
    step(f"{label}_done", rc=rc, elapsed=round(time.time() - t0, 2), verdict=verdict)
    return {"label": label, "rc": rc, "verdict": verdict}


results = []
results.append(run_diff("talker_cpu", "orpheus-talker", F16, REF, {"ORPHEUS_DIFF_MAXGEN": MAXGEN}))
results.append(run_diff("talker_gpu", "orpheus-talker", F16, REF,
                        {"ORPHEUS_DIFF_MAXGEN": MAXGEN, "ORPHEUS_DIFF_GPU": "1"}))

# ── SNAC vocoder diff on GPU (the next 0-byte suspect) ─────────────────────
try:
    snac = WORK / "snac-24khz.gguf"
    snac_ref = WORK / "orpheus-snac-ref.gguf"
    from huggingface_hub import hf_hub_download
    hf_hub_download("cstr/snac-24khz-GGUF", "snac-24khz.gguf", local_dir=str(WORK), token=hf_token)
    hf_hub_download("cstr/snac-24khz-GGUF", "diff-harness-ref/orpheus-snac-ref.gguf",
                    local_dir=str(WORK), token=hf_token)
    snac_ref = WORK / "diff-harness-ref" / "orpheus-snac-ref.gguf"
    results.append(run_diff("snac_cpu", "orpheus", snac, snac_ref, {}))
    results.append(run_diff("snac_gpu", "orpheus", snac, snac_ref, {"ORPHEUS_SNAC_GPU": "1"}))
except Exception as e:
    step("snac_skip", err=str(e))

# ── Upload GGUFs + ref to HF ───────────────────────────────────────────────
if DO_UPLOAD and hf_token:
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        api.create_repo(HF_REPO, repo_type="model", exist_ok=True)
        for f, dst in [(F16, F16.name), (Q8, Q8.name), (Q4, Q4.name),
                       (REF, "diff-harness-ref/orpheus-talker-ref.gguf")]:
            api.upload_file(path_or_fileobj=str(f), path_in_repo=dst, repo_id=HF_REPO, repo_type="model")
            step("uploaded", file=dst)
    except Exception as e:
        step("upload_err", err=str(e))

(RESULTS / "summary.json").write_text(json.dumps(
    {"sha": sha, "gpu": gpu_name, "model": HF_MODEL, "hf_repo": HF_REPO,
     "codec_offset": offset, "codec_count": count, "results": results}, indent=2))
step("done", results=results)
print("\n=== SUMMARY ===", flush=True)
for r in results:
    print(f"  {r['label']:12s} rc={r['rc']} verdict={r['verdict']}", flush=True)
