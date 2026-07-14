#!/usr/bin/env python3
"""Kaggle CUDA A/B: OmniVoice CFG forward — 2-forward vs unified-graph (#254).

On M1 Metal the two CFG forwards are compute-bound and fusing them (unified
seq + per-block attention split, OMNIVOICE_UNIFIED_CFG) is ~neutral-to-worse.
This kernel re-runs the A/B on a real CUDA box (P100/T4), where batching /
dispatch behave differently, and — best-effort — benchmarks
ServeurpersoCom/omnivoice.cpp (B'=2 batched) head-to-head.

Configs (crispasr CLI, --backend omnivoice, GPU):
  2forward — default (two separate forwards)
  unified  — OMNIVOICE_UNIFIED_CFG=1 (one graph, per-block flash-attn)

Per step we read the OMNIVOICE_BENCH stderr timers (fwd_cond+fwd_uncond vs
fwd_unified) and the OMNIVOICE_DEBUG_CODES line. Proof-of-work (#24): every run
must exit 0, decode non-empty audio, and the unified codes must match 2forward
(cb0 prefix) — a fake "win" from a crash/no-op is rejected.

Push (chr1s4): tools/kaggle/omnivoice-cfg-cuda-ab/push.sh
"""

import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
BRANCH = "main"
TXT = "The quick brown fox jumps over the lazy dog."
STEPS = 24  # enough steady-state steps for a median

print("=== Phase 0: clone + build ours ===", flush=True)
if not REPO.exists():
    subprocess.check_call(["git", "clone", "--depth", "1", "-b", BRANCH,
                           "https://github.com/CrispStrobe/CrispASR", str(REPO)])
if (REPO / "ggml").is_dir() and not (REPO / "ggml" / "CMakeLists.txt").exists():
    subprocess.check_call(["git", "submodule", "update", "--init", "ggml"], cwd=str(REPO))
sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress(hf_progress_repo="cstr/crispasr-kaggle-progress")
step = kh.step
step("script.start", branch=BRANCH)
TOKEN = kh.resolve_hf_token("HF_TOKEN")

subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                       "huggingface_hub", "hf_transfer", "soundfile"])
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download  # noqa: E402

BUILD = TEMP / "build"
BUILD.mkdir(parents=True, exist_ok=True)
kh.install_build_toolchain()
import shutil  # noqa: E402

_ccache_run = TEMP / ".ccache"
_warmed = Path("/kaggle/working/.ccache")
if _warmed.exists():
    if _ccache_run.exists():
        shutil.rmtree(_ccache_run, ignore_errors=True)
    shutil.move(str(_warmed), str(_ccache_run))
else:
    _ccache_run.mkdir(parents=True, exist_ok=True)
os.environ["CCACHE_DIR"] = str(_ccache_run)

has_cuda = Path("/usr/local/cuda/bin/nvcc").exists()
step("build.begin", cuda=has_cuda)
flags = (kh.cuda_build_flags(kh.detect_cuda_arch()) if has_cuda else []) + kh.cache_and_link_flags()
subprocess.check_call(f"cmake -G Ninja -B {BUILD} -S {REPO} -DCMAKE_BUILD_TYPE=Release "
                      + " ".join(flags), shell=True)
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} -j {kh.safe_build_jobs(has_cuda)} --target crispasr")
CLI = BUILD / "bin" / "crispasr"
assert CLI.is_file(), f"CLI not built at {CLI}"
step("build.done")

MODELS = TEMP / "models"
MODELS.mkdir(parents=True, exist_ok=True)
MODEL = hf_hub_download("cstr/omnivoice-GGUF", "omnivoice-q8_0.gguf", local_dir=str(MODELS), token=TOKEN)
TOK = hf_hub_download("cstr/omnivoice-GGUF", "omnivoice-tokenizer-f16.gguf", local_dir=str(MODELS), token=TOKEN)
step("model.downloaded")

_FWD = re.compile(r"omnivoice_bench:\s+(fwd_cond|fwd_uncond|fwd_unified)\s+([0-9.]+)\s+ms")
_CODES = re.compile(r"omnivoice-codes:.*cb0\[0:24\]=([0-9 ]+)")
_BACKEND = re.compile(r"compute backend = (\S+)")


def run_cfg(name, env_extra):
    """Run the CLI once; return per-step fwd time series + codes prefix."""
    env = dict(os.environ, OMNIVOICE_BENCH="1", OMNIVOICE_DEBUG_CODES="1", **env_extra)
    out = MODELS / f"{name}.wav"
    r = subprocess.run([str(CLI), "--backend", "omnivoice", "--model", MODEL, "--codec-model", TOK,
                        "--tts", TXT, "--tts-output", str(out), "--tts-steps", str(STEPS)],
                       capture_output=True, text=True, timeout=3600, env=env)
    # Sum fwd_cond+fwd_uncond per step (2forward) or take fwd_unified (unified).
    per_step = []
    pending = {}
    for m in _FWD.finditer(r.stderr):
        k, v = m.group(1), float(m.group(2))
        if k == "fwd_unified":
            per_step.append(v)
        else:
            pending[k] = v
            if "fwd_cond" in pending and "fwd_uncond" in pending:
                per_step.append(pending["fwd_cond"] + pending["fwd_uncond"])
                pending = {}
    codes = None
    cm = _CODES.search(r.stderr)
    if cm:
        codes = cm.group(1).strip()
    bm = _BACKEND.search(r.stderr)
    backend = bm.group(1) if bm else "?"
    # proof-of-work: audio present + non-trivial
    dur = 0.0
    if out.is_file():
        import soundfile as sf
        info = sf.info(str(out))
        dur = info.frames / info.samplerate
    ok = r.returncode == 0 and dur > 0.5 and len(per_step) >= 4
    return {"rc": r.returncode, "per_step": per_step, "warm_median": (statistics.median(per_step[3:]) if len(per_step) > 4 else None),
            "codes": codes, "backend": backend, "dur": round(dur, 2), "ok": ok, "err": r.stderr[-500:] if not ok else ""}


results = {}
for name, env_extra in [("2forward", {}), ("unified", {"OMNIVOICE_UNIFIED_CFG": "1"})]:
    res = run_cfg(name, env_extra)
    results[name] = res
    step(f"cfg.{name}", ok=res["ok"], warm_median_ms=res["warm_median"], dur=res["dur"], rc=res["rc"],
         backend=res["backend"])
    if not res["ok"]:
        step(f"cfg.{name}.FAIL", err=res["err"][-300:].replace("\n", " / "))

# Proof-of-work: codes must match between configs (output-equivalent).
codes_match = (results["2forward"].get("codes") and
               results["2forward"]["codes"] == results["unified"].get("codes"))
step("ours.verdict", codes_match=codes_match,
     twoforward_ms=results["2forward"]["warm_median"], unified_ms=results["unified"]["warm_median"])

# ---- Best-effort: build + bench omnivoice.cpp (B'=2 batched) on CUDA ----
theirs = {"attempted": True}
try:
    OVCPP = TEMP / "omnivoice.cpp"
    if not OVCPP.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", "--recursive",
                               "https://github.com/ServeurpersoCom/omnivoice.cpp", str(OVCPP)])
    # Inject a per-step timer around their batched forward (static_graph_compute).
    pt = OVCPP / "src" / "pipeline-tts.cpp"
    txt = pt.read_text()
    if "OVCPP_BENCH" not in txt:
        if "#include <chrono>" not in txt:
            txt = "#include <chrono>\n" + txt
        needle = "enum ggml_status st = static_graph_compute(&ctx->lm_graph, pt->backend, pt->sched, ctx->lm_gf);"
        inj = (needle +
               '\n    if (getenv("OVCPP_BENCH")) { static auto _t0=std::chrono::steady_clock::now();'
               ' auto _t1=std::chrono::steady_clock::now();'
               ' fprintf(stderr,"OVCPP_STEP %.3f\\n", std::chrono::duration<double,std::milli>(_t1-_t0).count()); _t0=_t1; }')
        # simpler robust timer: measure around the call
        inj = ('auto _ovt0=std::chrono::steady_clock::now();\n    ' + needle +
               '\n    if (getenv("OVCPP_BENCH")) fprintf(stderr,"OVCPP_STEP %.3f\\n",'
               ' std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-_ovt0).count());')
        txt = txt.replace(needle, inj, 1)
        pt.write_text(txt)
    obuild = OVCPP / "build"
    cuda_flag = "-DGGML_CUDA=ON" if has_cuda else ""
    subprocess.check_call(f"cmake -B {obuild} -S {OVCPP} -DCMAKE_BUILD_TYPE=Release {cuda_flag} -DGGML_NATIVE=OFF",
                          shell=True)
    with kh.build_heartbeat("ovcpp.build"):
        kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {obuild} -j {kh.safe_build_jobs(has_cuda)} --target omnivoice-tts")
    otts = obuild / "omnivoice-tts"
    if not otts.is_file():
        otts = obuild / "bin" / "omnivoice-tts"
    om = hf_hub_download("Serveurperso/OmniVoice-GGUF", "omnivoice-base-Q8_0.gguf", local_dir=str(MODELS / "ovcpp"), token=TOKEN)
    ot = hf_hub_download("Serveurperso/OmniVoice-GGUF", "omnivoice-tokenizer-F32.gguf", local_dir=str(MODELS / "ovcpp"), token=TOKEN)
    step("ovcpp.built")
    env = dict(os.environ, OVCPP_BENCH="1")
    r = subprocess.run([str(otts), "--model", om, "--codec", ot, "--lang", "English",
                        "--duration", "4.0", "--steps", str(STEPS), "-o", str(MODELS / "theirs.wav")],
                       input=TXT, capture_output=True, text=True, timeout=3600, env=env)
    steps = [float(x) for x in re.findall(r"OVCPP_STEP\s+([0-9.]+)", r.stderr)]
    theirs.update({"rc": r.returncode, "per_step": steps,
                   "warm_median": (statistics.median(steps[3:]) if len(steps) > 4 else None),
                   "n_steps": len(steps), "err": r.stderr[-400:] if r.returncode else ""})
    step("ovcpp.bench", rc=r.returncode, warm_median_ms=theirs["warm_median"], n=len(steps))
except Exception as e:
    theirs["error"] = str(e)[:300]
    step("ovcpp.error", err=str(e)[:200])

verdict = {
    "gpu": has_cuda,
    "arch": kh.detect_cuda_arch() if has_cuda else "cpu",
    "compute_backend": results["2forward"].get("backend"),
    "ours_2forward_ms": results["2forward"]["warm_median"],
    "ours_unified_ms": results["unified"]["warm_median"],
    "codes_match": codes_match,
    "theirs_ovcpp_ms": theirs.get("warm_median"),
}
step("script.done", **{k: v for k, v in verdict.items() if not isinstance(v, (dict, list))})
(WORK / "results.json").write_text(json.dumps({"verdict": verdict, "ours": results, "theirs": theirs}, indent=1))
print("DONE", json.dumps(verdict), flush=True)
