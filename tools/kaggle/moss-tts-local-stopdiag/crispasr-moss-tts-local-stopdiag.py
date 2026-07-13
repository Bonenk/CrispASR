# CrispASR — MOSS-TTS-Local 4B STOP-HEAD diagnostic (#249, P5 debug)
#
# run1/run2 proved the codec decode is correct (ASR overlap 1.0) but generation
# RUNS AWAY (binary stop head never fires) for short text + Q4_K; the stop logit
# sits at continue~8 / stop~-3 every frame. Static inspection shows the C++ port
# matches the HF reference structurally. This kernel is the decisive empirical
# test — NO codec, NO ASR, so it's fast (~15-20 min):
#
#   HF side: load MossTTSLocal + processor (trust_remote_code), hook
#     local_text_lm_head, run model.generate(do_sample=True, text_temperature=1.0)
#     on SHORT ("Hello world.") and LONG text -> does the reference STOP? at what
#     frame? what is the [continue, stop] trajectory (does the gap ever narrow)?
#   C++ side: moss-tts-local-smoke with CRISPASR_MOSS_TTS_LOCAL_DEBUG on the same
#     texts (f16 + q4k), max 512 frames -> frames + stopped + stop logits.
#
# Verdict: if the reference STOPS for "Hello world" and C++ doesn't -> C++ bug
# (compare the trajectories). If the reference ALSO runs away -> model/param
# behavior (the fix is generation handling, e.g. text-conditioned max frames).

import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

WORK = Path("/kaggle/working")
TMP = Path("/tmp")
REPO = TMP / "CrispASR"
BUILD = REPO / "build"
MODELS = TMP / "moss-local-models"
RESULTS = WORK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

CRISPASR_REF = os.environ.get("CRISPASR_REF", "feat/moss-tts-local-4b")
CRISPASR_REPO = os.environ.get("CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git")
HF_MODEL = os.environ.get("MOSS_MODEL", "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5")
GGUF_REPO = os.environ.get("MOSS_GGUF_REPO", "cstr/moss-tts-local-v1.5-GGUF")
MAXF = int(os.environ.get("MOSS_MAXF", "512"))
SEED = int(os.environ.get("MOSS_SEED", "1234"))

SHORT_TEXT = "Hello world."
LONG_TEXT = ("The quick brown fox jumps over the lazy dog. "
             "Speech synthesis should stay intelligible over a longer passage.")
_T0 = time.time()


def log(m):
    print(f"[{round(time.time() - _T0, 1)}s] {m}", flush=True)


def build_cpp(kh):
    kh.install_build_toolchain()
    arch = kh.detect_cuda_arch()
    os.environ["CCACHE_DIR"] = "/kaggle/temp/.ccache"
    env = os.environ.copy()
    subprocess.run(["cmake", "-G", "Ninja", "-B", str(BUILD), "-S", str(REPO), "-DCMAKE_BUILD_TYPE=Release"]
                   + list(kh.cache_and_link_flags()) + list(kh.cuda_build_flags(arch)),
                   env=env, check=True, timeout=300)
    with kh.build_heartbeat("stopdiag build"):
        kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} "
                            f"--target crispasr-quantize moss-tts-local-smoke -j{kh.safe_build_jobs(gpu=True)}")
    smoke = next((c for c in BUILD.rglob("moss-tts-local-smoke") if c.is_file() and os.access(c, os.X_OK)), None)
    quant = next((c for c in BUILD.rglob("crispasr-quantize") if c.is_file() and os.access(c, os.X_OK)), None)
    if not smoke or not quant:
        raise SystemExit(f"binaries missing: smoke={smoke} quant={quant}")
    os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    return smoke, quant


def run_cpp(smoke, gguf, tag, text):
    env = os.environ.copy()
    env["CRISPASR_MOSS_TTS_LOCAL_DEBUG"] = "1"
    r = subprocess.run([str(smoke), str(gguf), text, str(MAXF)], capture_output=True, text=True, timeout=1800, env=env)
    out = r.stdout + "\n--STDERR--\n" + r.stderr
    (RESULTS / f"cpp_{tag}.log").write_text(out)
    m = re.search(r"generated (\d+) frames .*?(runaway|stopped naturally)", out)
    frames = int(m.group(1)) if m else None
    stopped = (m.group(2) == "stopped naturally") if m else None
    # capture a few stop-logit lines
    logits = re.findall(r"frame (\d+): stop_head continue=(-?[\d.]+) stop=(-?[\d.]+)", out)
    traj = [(int(a), float(b), float(c)) for a, b, c in logits]
    return {"rc": r.returncode, "frames": frames, "stopped": stopped,
            "logit_first": traj[:6], "logit_last": traj[-4:]}


def run_reference(hf_token):
    """Load MossTTSLocal + processor, hook the stop head, generate short+long."""
    import torch
    from transformers import AutoModel, AutoProcessor
    log("load reference model (bf16, cuda)")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(HF_MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                      token=hf_token).to(dev).eval()
    proc = AutoProcessor.from_pretrained(HF_MODEL, trust_remote_code=True, token=hf_token)

    out = {}
    for tag, text in (("short", SHORT_TEXT), ("long", LONG_TEXT)):
        stop_logits = []

        def hook(_m, _i, o):
            t = o.detach().float().cpu().reshape(-1, o.shape[-1])
            stop_logits.append(t[-1].tolist())  # [continue, stop]

        h = model.local_text_lm_head.register_forward_hook(hook)
        try:
            msg = proc.build_user_message(text=text)
            feat = proc([msg], mode="generation")
            input_ids = feat["input_ids"] if isinstance(feat, dict) else feat.input_ids
            input_ids = input_ids.to(dev)
            torch.manual_seed(SEED)
            gen = model.generate(input_ids=input_ids, max_new_frames=MAXF, do_sample=True,
                                 text_temperature=1.0, temperature=1.0, top_p=0.95, top_k=50)
            n_frames = len(stop_logits)
            stopped = n_frames < MAXF
            traj = [(i, round(cl, 3), round(sl, 3)) for i, (cl, sl) in enumerate(stop_logits)]
            # where does stop first exceed continue (gap crosses 0)?
            cross = next((i for i, (cl, sl) in enumerate(stop_logits) if sl >= cl), None)
            out[tag] = {"frames": n_frames, "stopped": stopped, "stop_crosses_at": cross,
                        "logit_first": traj[:6], "logit_last": traj[-4:],
                        "min_gap": min((cl - sl for cl, sl in stop_logits), default=None)}
            log(f"REF {tag}: frames={n_frames} stopped={stopped} cross_at={cross} "
                f"min_gap={out[tag]['min_gap']}")
        except Exception as e:  # noqa: BLE001
            out[tag] = {"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]}
            log(f"REF {tag} ERROR: {e}")
        finally:
            h.remove()
    return out


def main():
    summary = {"ref_branch": CRISPASR_REF, "seed": SEED, "maxf": MAXF}
    log(f"clone {CRISPASR_REF}")
    if not REPO.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, "--recursive",
                               CRISPASR_REPO, str(REPO)])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.init_progress()
    hf_token = kh.resolve_hf_token()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    summary["sha"] = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()

    # ── C++ side: build + download F16 + quantize + run smoke ──────────────
    smoke, quant = build_cpp(kh)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import hf_hub_download
    f16 = Path(hf_hub_download(GGUF_REPO, "moss-tts-local-v1.5-f16.gguf", local_dir=str(MODELS), token=hf_token))
    q4k = MODELS / "moss-tts-local-v1.5-q4_k.gguf"
    subprocess.run([str(quant), str(f16), str(q4k), "q4_k"], check=True, timeout=1800)
    cpp = {}
    for tag, text in (("short", SHORT_TEXT), ("long", LONG_TEXT)):
        cpp[f"f16_{tag}"] = run_cpp(smoke, f16, f"f16_{tag}", text)
        cpp[f"q4k_{tag}"] = run_cpp(smoke, q4k, f"q4k_{tag}", text)
        log(f"CPP f16_{tag}: {cpp[f'f16_{tag}']['frames']} frames stopped={cpp[f'f16_{tag}']['stopped']}; "
            f"q4k_{tag}: {cpp[f'q4k_{tag}']['frames']} frames stopped={cpp[f'q4k_{tag}']['stopped']}")
    summary["cpp"] = cpp

    # ── HF reference side ──────────────────────────────────────────────────
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers", "accelerate"])
    try:
        summary["reference"] = run_reference(hf_token)
    except Exception as e:  # noqa: BLE001
        summary["reference"] = {"fatal": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-2000:]}
        log(f"reference fatal: {e}")

    # ── verdict ────────────────────────────────────────────────────────────
    ref = summary.get("reference", {})
    ref_short_stopped = isinstance(ref.get("short"), dict) and ref["short"].get("stopped")
    cpp_short_stopped = cpp.get("f16_short", {}).get("stopped")
    summary["verdict"] = (
        "C++_BUG (ref stops short, C++ runs away)" if (ref_short_stopped and not cpp_short_stopped) else
        "MODEL_BEHAVIOR (ref also runs away on short)" if (ref_short_stopped is False) else
        "INCONCLUSIVE (see reference errors)")
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60 + "\n" + json.dumps(summary, indent=2) + "\n" + "=" * 60)
    log(f"VERDICT: {summary['verdict']}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        log(f"FATAL: {e}\n{traceback.format_exc()}")
        sys.exit(1)
