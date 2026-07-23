"""#296 mel-band-roformer: PROFILE where the forward spends time (with OpenBLAS).

The prior A/B showed cos=1.0 but only 24->10 min: the build had linked the SLOW
reference libcblas.so, not OpenBLAS. This build prefers OpenBLAS and adds
CRISPASR_MBR_PROFILE per-stage timing. Runs a short clip with profiling ON (prints
the stft/band_split/run_time/run_freq/mask/synthesize breakdown) and times 11s jfk.
"""
import json, os, re, shutil, subprocess, sys, time
from pathlib import Path

_T0 = time.time()
TEMP = Path("/kaggle/temp"); OUT = Path("/kaggle/working")
REPO = TEMP / "CrispASR"; MODELS = TEMP / "models"; BUILD = TEMP / "build"
for d in (TEMP, OUT, MODELS): d.mkdir(parents=True, exist_ok=True)
FIX = "17e3ce277"

import traceback as _tb
def _eh(et, ev, tb):
    try: (OUT / "error.txt").write_text("".join(_tb.format_exception(et, ev, tb)))
    except Exception: pass
    sys.__excepthook__(et, ev, tb)
sys.excepthook = _eh

def run(cmd, **kw):
    kw.setdefault("capture_output", True); kw.setdefault("text", True)
    return subprocess.run(cmd, **kw)

print(json.dumps({"step": "start"}), flush=True)
if REPO.exists(): shutil.rmtree(REPO)
run(["git", "clone", "--depth", "3", "https://github.com/CrispStrobe/CrispASR.git", str(REPO)], capture_output=False)
run(["git", "-C", str(REPO), "checkout", "-q", FIX], capture_output=False)
run(["git", "-C", str(REPO), "submodule", "update", "--init", "--recursive", "--depth", "1"], capture_output=False, timeout=1800)
sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh
kh.init_progress(); kh.step("cloned", fix=FIX)
kh.install_build_toolchain()
run(["apt-get", "install", "-y", "-q", "libopenblas-dev"], capture_output=False)
kh.step("blas.installed")

# configure — WATCH the log for 'mel-band-roformer: linking OpenBLAS'
cfg = ["cmake", "-G", "Ninja", "-B", str(BUILD), "-S", str(REPO),
       "-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_NO_C2PA_NATIVE=ON"] + kh.cache_and_link_flags()
r = run(cfg, capture_output=False)
if r.returncode: kh.step("configure.FAIL"); raise SystemExit(1)
JOBS = str(min(4, os.cpu_count() or 2))
with kh.build_heartbeat("build"):
    r = run(["cmake", "--build", str(BUILD), "--target", "crispasr-cli", "-j", JOBS], capture_output=False)
if r.returncode: kh.step("build.FAIL"); raise SystemExit(1)
CLI = BUILD / "bin" / "crispasr"
if not CLI.exists():
    c = [p for p in BUILD.rglob("crispasr") if p.is_file() and os.access(p, os.X_OK)]; CLI = c[0] if c else None
if CLI is None: kh.step("build.MISSING"); raise SystemExit(1)
kh.step("build.done", cli=str(CLI))

from huggingface_hub import hf_hub_download
with kh.build_heartbeat("download"):
    MODEL = Path(hf_hub_download(repo_id="cstr/mel-band-roformer-vocals-GGUF",
                                 filename="mel-band-roformer-vocals-f16.gguf", local_dir=str(MODELS)))
JFK = REPO / "samples" / "jfk.wav"
CLIP4 = TEMP / "jfk4.wav"
run(["ffmpeg", "-y", "-i", str(JFK), "-t", "4", str(CLIP4)], capture_output=False)

def separate(clip, tag, profile=False, timeout=2400):
    env = {**os.environ, "LD_LIBRARY_PATH": f"{BUILD}/src:{os.environ.get('LD_LIBRARY_PATH','')}"}
    if profile: env["CRISPASR_MBR_PROFILE"] = "1"
    odir = TEMP / f"stems_{tag}"; odir.mkdir(exist_ok=True)
    t0 = time.time()
    r = run([str(CLI), "--separate", "-m", str(MODEL), "-f", str(clip), "--sep-output-dir", str(odir)],
            env=env, timeout=timeout)
    dt = time.time() - t0
    return dt, r

# 4s WITH profiling — print the per-stage breakdown
d4, r4 = separate(CLIP4, "prof", profile=True)
prof_lines = [l for l in (r4.stderr or "").splitlines() if "mbr-prof" in l or "layer" in l]
print(f"=== 4s clip: {d4:.1f}s ===", flush=True)
for l in prof_lines: print("  ", l, flush=True)
kh.step("prof.4s", secs=round(d4, 1), lines=prof_lines[:20])

# 11s timing (no profiling noise, default threads)
d11, r11 = separate(JFK, "jfk11")
kh.step("jfk11", secs=round(d11, 1), rc=r11.returncode)

R = {"clip4_s": round(d4, 1), "jfk11_s": round(d11, 1), "profile": prof_lines,
     "openblas": ("linking OpenBLAS" in (r4.stderr or "") or None)}
(OUT / "results.json").write_text(json.dumps(R, indent=2))
print(json.dumps({"step": "done", "clip4_s": R["clip4_s"], "jfk11_s": R["jfk11_s"]}), flush=True)
kh.step("done", clip4_s=R["clip4_s"], jfk11_s=R["jfk11_s"])
