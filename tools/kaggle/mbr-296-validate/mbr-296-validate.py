"""#296 mel-band-roformer: validate the parallelize+float+progress fix.

A/B on Kaggle (Linux, OpenMP): build the PRE-FIX commit (single-threaded, double
attention) and the FIX commit; separate the SAME short clip on both to get
  - correctness: cos(vocals_base, vocals_fix) must be ~1.0 (float change is benign)
  - speedup: base_time / fix_time
then time the full 11s jfk on the fix build to prove it completes (not a hang) with
per-layer progress. Model: mel-band-roformer f16 (compute is quant-independent).
"""
import json, os, re, shutil, subprocess, sys, time, wave
from pathlib import Path
import numpy as np

_T0 = time.time()
TEMP = Path("/kaggle/temp"); OUT = Path("/kaggle/working")
REPO = TEMP / "CrispASR"; MODELS = TEMP / "models"
for d in (TEMP, OUT, MODELS): d.mkdir(parents=True, exist_ok=True)
PARENT = "6fca326a0"; FIX = "ebe082d25"

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
run(["git", "clone", "--depth", "5", "https://github.com/CrispStrobe/CrispASR.git", str(REPO)], capture_output=False)
run(["git", "-C", str(REPO), "submodule", "update", "--init", "--recursive", "--depth", "1"], capture_output=False, timeout=1800)
sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh
kh.init_progress(); kh.step("cloned")
kh.install_build_toolchain()

JOBS = str(min(4, os.cpu_count() or 2))
def build_at(commit, bdir):
    run(["git", "-C", str(REPO), "checkout", "-q", commit], capture_output=False)
    cfg = ["cmake", "-G", "Ninja", "-B", str(bdir), "-S", str(REPO),
           "-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_NO_C2PA_NATIVE=ON"] + kh.cache_and_link_flags()
    r = run(cfg, capture_output=False)
    if r.returncode: kh.step(f"configure.FAIL.{commit}"); raise SystemExit(1)
    with kh.build_heartbeat(f"build.{commit}"):
        r = run(["cmake", "--build", str(bdir), "--target", "crispasr-cli", "-j", JOBS], capture_output=False)
    if r.returncode: kh.step(f"build.FAIL.{commit}"); raise SystemExit(1)
    cli = bdir / "bin" / "crispasr"
    if not cli.exists():
        c = [p for p in bdir.rglob("crispasr") if p.is_file() and os.access(p, os.X_OK)]
        cli = c[0] if c else None
    if cli is None: kh.step(f"build.MISSING.{commit}"); raise SystemExit(1)
    return cli

# model (f16) via HF
from huggingface_hub import hf_hub_download
kh.step("download.begin")
with kh.build_heartbeat("download"):
    MODEL = Path(hf_hub_download(repo_id="cstr/mel-band-roformer-vocals-GGUF",
                                 filename="mel-band-roformer-vocals-f16.gguf", local_dir=str(MODELS)))
kh.step("download.done", mb=round(MODEL.stat().st_size / 1e6))

JFK = REPO / "samples" / "jfk.wav"
CLIP4 = TEMP / "jfk4.wav"
run(["ffmpeg", "-y", "-i", str(JFK), "-t", "4", str(CLIP4)], capture_output=False)

def separate(cli, bdir, clip, tag, threads=None):
    env = {**os.environ, "LD_LIBRARY_PATH": f"{bdir}/src:{os.environ.get('LD_LIBRARY_PATH','')}"}
    if threads is not None: env["OMP_NUM_THREADS"] = str(threads)
    odir = TEMP / f"stems_{tag}"; odir.mkdir(exist_ok=True)
    t0 = time.time()
    r = run([str(cli), "--separate", "-m", str(MODEL), "-f", str(clip), "--sep-output-dir", str(odir)],
            env=env, timeout=2400)
    dt = time.time() - t0
    voc = odir / (Path(clip).stem + "_vocals.wav")
    return dt, (voc if voc.exists() else None), r

def read_wav(p):
    with wave.open(str(p), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)

def cos(a, b):
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0

R = {}
# baseline: pre-fix (serial, double)
base_cli = build_at(PARENT, TEMP / "build-base")
bt, bvoc, br = separate(base_cli, TEMP / "build-base", CLIP4, "base")
kh.step("base.sep", secs=round(bt, 1), voc=bool(bvoc), rc=br.returncode)
# fix: parallel + float
fix_cli = build_at(FIX, TEMP / "build-fix")
ft, fvoc, fr = separate(fix_cli, TEMP / "build-fix", CLIP4, "fix")
kh.step("fix.sep", secs=round(ft, 1), voc=bool(fvoc), rc=fr.returncode)
# correctness + speedup on the 4s clip
c = cos(read_wav(bvoc), read_wav(fvoc)) if (bvoc and fvoc) else None
R["clip4"] = {"base_s": round(bt, 1), "fix_s": round(ft, 1),
              "speedup": round(bt / ft, 2) if ft else None, "cos_base_vs_fix": c}
kh.step("ab.done", **R["clip4"])
# fix serial vs parallel to isolate threading scaling
st, svoc, _ = separate(fix_cli, TEMP / "build-fix", CLIP4, "fix1", threads=1)
R["clip4"]["fix_1thread_s"] = round(st, 1)
R["clip4"]["thread_scaling"] = round(st / ft, 2) if ft else None
# full 11s on the fix build (non-hang proof, default threads)
f11, v11, r11 = separate(fix_cli, TEMP / "build-fix", JFK, "fix11")
R["jfk11_fix_s"] = round(f11, 1); R["jfk11_completed"] = bool(v11)
kh.step("jfk11.done", secs=round(f11, 1), completed=bool(v11))

R["verdict"] = {
    "correct": (c is not None and c > 0.999),
    "not_hung": bool(v11) and f11 < 300,
    "cos": c, "clip4_speedup": R["clip4"]["speedup"], "jfk11_s": R.get("jfk11_fix_s"),
}
(OUT / "results.json").write_text(json.dumps(R, indent=2))
print(json.dumps({"step": "done", **R["verdict"]}), flush=True)
kh.step("done", **R["verdict"])
