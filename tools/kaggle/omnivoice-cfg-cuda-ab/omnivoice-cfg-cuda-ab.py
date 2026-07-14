#!/usr/bin/env python3
"""Kaggle CUDA A/B: OmniVoice CFG forward — 2-forward vs unified-graph (#254).

On M1 Metal the two CFG forwards are compute-bound and fusing them
(OMNIVOICE_UNIFIED_CFG, one graph + per-block attention split) is ~neutral-to-worse.
This kernel re-runs the A/B on a real CUDA box (P100/T4), where batching / dispatch
behave differently, and — best-effort — benchmarks ServeurpersoCom/omnivoice.cpp
(B'=2 batched) head-to-head.

REGIME (hard-won on this box): step() before AND after EVERY operation so a hang is
visible; HF network ops in a daemon thread with a join timeout (CLAUDE.md: Kaggle HF
calls strand); short subprocess timeouts (fail fast, don't burn hours); confirm the
run is really on CUDA0; proof-of-work (#24) — every run must exit 0, decode non-empty
audio, and unified codes must match 2forward. A top-level guard always writes
results.json so a failure is diagnosable.

Push (chr1s4): tools/kaggle/omnivoice-cfg-cuda-ab/push.sh
"""

import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
BRANCH = "main"
TXT = "The quick brown fox jumps over the lazy dog."
STEPS = 24
RESULTS = {"stage": "start", "ours": {}, "theirs": {}, "verdict": {}}


def dump_results():
    try:
        (WORK / "results.json").write_text(json.dumps(RESULTS, indent=1))
    except Exception:
        pass


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

subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "huggingface_hub", "soundfile"])
from huggingface_hub import hf_hub_download  # noqa: E402


def hf_get(repo, fname, dest, timeout=300):
    """Download in a daemon thread with a join timeout (Kaggle HF ops strand)."""
    step("dl.begin", file=fname)
    holder = {}

    def _work():
        try:
            holder["path"] = hf_hub_download(repo, fname, local_dir=str(dest), token=TOKEN)
        except Exception as e:  # noqa: BLE001
            holder["err"] = str(e)[:300]

    th = threading.Thread(target=_work, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        raise TimeoutError(f"hf_hub_download({fname}) hung > {timeout}s")
    if "err" in holder:
        raise RuntimeError(f"hf_hub_download({fname}) failed: {holder['err']}")
    p = holder["path"]
    sz = os.path.getsize(p)
    step("dl.done", file=fname, mb=round(sz / 1e6, 1))
    return p


try:
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
    RESULTS["stage"] = "build"
    step("build.begin", cuda=has_cuda)
    flags = (kh.cuda_build_flags(kh.detect_cuda_arch()) if has_cuda else []) + kh.cache_and_link_flags()
    subprocess.check_call(f"cmake -G Ninja -B {BUILD} -S {REPO} -DCMAKE_BUILD_TYPE=Release " + " ".join(flags),
                          shell=True)
    with kh.build_heartbeat("cmake.build"):
        kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} -j {kh.safe_build_jobs(has_cuda)} --target crispasr")
    CLI = BUILD / "bin" / "crispasr"
    assert CLI.is_file(), f"CLI not built at {CLI}"
    step("build.done")

    RESULTS["stage"] = "download"
    MODELS = TEMP / "models"
    MODELS.mkdir(parents=True, exist_ok=True)
    MODEL = hf_get("cstr/omnivoice-GGUF", "omnivoice-q8_0.gguf", MODELS)
    TOK = hf_get("cstr/omnivoice-GGUF", "omnivoice-tokenizer-f16.gguf", MODELS)
    step("model.downloaded")

    _FWD = re.compile(r"omnivoice_bench:\s+(fwd_cond|fwd_uncond|fwd_unified)\s+([0-9.]+)\s+ms")
    _CODES = re.compile(r"omnivoice-codes:.*cb0\[0:24\]=([0-9 ]+)")
    _BACKEND = re.compile(r"compute backend = (\S+)")

    def run_cfg(name, env_extra, timeout=420):
        step("cfg.begin", cfg=name)
        env = dict(os.environ, OMNIVOICE_BENCH="1", OMNIVOICE_DEBUG_CODES="1", **env_extra)
        out = MODELS / f"{name}.wav"
        errf = MODELS / f"{name}.stderr.txt"
        # Redirect to a file (no capture-pipe deadlock risk) + hard timeout.
        with open(errf, "w") as ef:
            try:
                rc = subprocess.call([str(CLI), "--backend", "omnivoice", "--model", MODEL, "--codec-model", TOK,
                                      "--tts", TXT, "--tts-output", str(out), "--tts-steps", str(STEPS)],
                                     stdout=ef, stderr=subprocess.STDOUT, env=env, timeout=timeout)
                timed_out = False
            except subprocess.TimeoutExpired:
                rc, timed_out = -9, True
        err = errf.read_text(errors="replace")
        per_step, pending = [], {}
        for m in _FWD.finditer(err):
            k, v = m.group(1), float(m.group(2))
            if k == "fwd_unified":
                per_step.append(v)
            else:
                pending[k] = v
                if "fwd_cond" in pending and "fwd_uncond" in pending:
                    per_step.append(pending["fwd_cond"] + pending["fwd_uncond"])
                    pending = {}
        cm = _CODES.search(err)
        bm = _BACKEND.search(err)
        dur = 0.0
        if out.is_file():
            import soundfile as sf
            info = sf.info(str(out))
            dur = info.frames / info.samplerate
        ok = rc == 0 and dur > 0.5 and len(per_step) >= 4
        res = {"rc": rc, "timed_out": timed_out, "per_step": per_step,
               "warm_median": (statistics.median(per_step[3:]) if len(per_step) > 4 else None),
               "codes": (cm.group(1).strip() if cm else None), "backend": (bm.group(1) if bm else "?"),
               "dur": round(dur, 2), "ok": ok, "err_tail": ("" if ok else err[-600:])}
        step("cfg.done", cfg=name, ok=ok, warm_median_ms=res["warm_median"], backend=res["backend"],
             dur=res["dur"], rc=rc, timed_out=timed_out)
        return res

    RESULTS["stage"] = "bench_ours"
    for nm, ev in [("2forward", {}), ("unified", {"OMNIVOICE_UNIFIED_CFG": "1"})]:
        RESULTS["ours"][nm] = run_cfg(nm, ev)
        dump_results()

    two, uni = RESULTS["ours"]["2forward"], RESULTS["ours"]["unified"]
    codes_match = bool(two.get("codes") and two["codes"] == uni.get("codes"))
    on_cuda = "CUDA" in (two.get("backend") or "")
    RESULTS["verdict"] = {"gpu": has_cuda, "compute_backend": two.get("backend"), "on_cuda": on_cuda,
                          "codes_match": codes_match, "ours_2forward_ms": two.get("warm_median"),
                          "ours_unified_ms": uni.get("warm_median")}
    step("ours.verdict", **RESULTS["verdict"])
    dump_results()

    # ---- Best-effort: omnivoice.cpp (B'=2 batched) on CUDA ----
    RESULTS["stage"] = "bench_theirs"
    theirs = RESULTS["theirs"]
    try:
        OVCPP = TEMP / "omnivoice.cpp"
        step("ovcpp.clone")
        if not OVCPP.exists():
            subprocess.check_call(["git", "clone", "--depth", "1", "--recursive",
                                   "https://github.com/ServeurpersoCom/omnivoice.cpp", str(OVCPP)])
        pt = OVCPP / "src" / "pipeline-tts.cpp"
        txt = pt.read_text()
        needle = "enum ggml_status st = static_graph_compute(&ctx->lm_graph, pt->backend, pt->sched, ctx->lm_gf);"
        if "OVCPP_STEP" not in txt and needle in txt:
            if "#include <chrono>" not in txt:
                txt = "#include <chrono>\n" + txt
            inj = ('auto _ovt0=std::chrono::steady_clock::now();\n    ' + needle +
                   '\n    if (getenv("OVCPP_BENCH")) fprintf(stderr,"OVCPP_STEP %.3f\\n",'
                   ' std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-_ovt0).count());')
            pt.write_text(txt.replace(needle, inj, 1))
        obuild = OVCPP / "build"
        cuda_flag = "-DGGML_CUDA=ON" if has_cuda else ""
        step("ovcpp.build.begin")
        subprocess.check_call(f"cmake -B {obuild} -S {OVCPP} -DCMAKE_BUILD_TYPE=Release {cuda_flag} -DGGML_NATIVE=OFF",
                              shell=True)
        with kh.build_heartbeat("ovcpp.build"):
            kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {obuild} -j {kh.safe_build_jobs(has_cuda)} "
                                f"--target omnivoice-tts")
        otts = obuild / "omnivoice-tts"
        if not otts.is_file():
            otts = obuild / "bin" / "omnivoice-tts"
        step("ovcpp.built", exists=otts.is_file())
        om = hf_get("Serveurperso/OmniVoice-GGUF", "omnivoice-base-Q8_0.gguf", MODELS / "ovcpp")
        ot = hf_get("Serveurperso/OmniVoice-GGUF", "omnivoice-tokenizer-F32.gguf", MODELS / "ovcpp")
        errf = MODELS / "theirs.stderr.txt"
        with open(errf, "w") as ef:
            try:
                rc = subprocess.run([str(otts), "--model", om, "--codec", ot, "--lang", "English",
                                     "--duration", "4.0", "--steps", str(STEPS), "-o", str(MODELS / "theirs.wav")],
                                    input=TXT, text=True, stdout=ef, stderr=subprocess.STDOUT,
                                    timeout=420).returncode
            except subprocess.TimeoutExpired:
                rc = -9
        err = errf.read_text(errors="replace")
        steps = [float(x) for x in re.findall(r"OVCPP_STEP\s+([0-9.]+)", err)]
        theirs.update({"rc": rc, "n_steps": len(steps),
                       "warm_median": (statistics.median(steps[3:]) if len(steps) > 4 else None),
                       "err_tail": ("" if rc == 0 and steps else err[-600:])})
        step("ovcpp.bench", rc=rc, warm_median_ms=theirs["warm_median"], n=len(steps))
    except Exception as e:  # noqa: BLE001
        theirs["error"] = str(e)[:300]
        step("ovcpp.error", err=str(e)[:200])

    RESULTS["verdict"]["theirs_ovcpp_ms"] = theirs.get("warm_median")
    RESULTS["stage"] = "done"
    step("script.done", **RESULTS["verdict"])
    dump_results()
    print("DONE", json.dumps(RESULTS["verdict"]), flush=True)

except Exception as e:  # noqa: BLE001
    import traceback
    RESULTS["fatal"] = {"stage": RESULTS["stage"], "error": str(e), "tb": traceback.format_exc()[-1500:]}
    dump_results()
    try:
        step("script.FATAL", stage=RESULTS["stage"], err=str(e)[:200])
    except Exception:
        pass
    print("FATAL at", RESULTS["stage"], ":", e, flush=True)
    raise
