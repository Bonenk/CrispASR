# CrispASR — MOSS-TTS greedy code-parity (#249, Phase-3 gate)
#
# The ship kernel's parity step OOM'd (torch 8B + the crispasr process on one
# 16 GB P100). Fix: (1) download the already-shipped GGUFs instead of
# re-converting; (2) run the HF reference STANDALONE first, then free the GPU
# BEFORE the ggml side loads; (3) compare the greedy code grids.
#
# Greedy (temps=0) makes both sides deterministic. The C++ side runs Q4_K (F16's
# 17 GB backbone won't fit a 16 GB P100), so — per the voxtral-tts lesson — expect
# a byte-identical PREFIX then divergence as Q4_K-vs-BF16 rounding flips an argmax.
# The gate is "prefix matches" (backbone + 33 heads + delay are structurally
# correct), reported alongside the first-divergence frame.
#
# ccache under /kaggle/temp (not /kaggle/working) so the log/artifacts stay
# reachable (kaggle_usage #22).

import ctypes
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
TMP = Path("/tmp")
REPO = TMP / "CrispASR"
BUILD = REPO / "build"
MODELS = TMP / "moss-models"
WORK = Path("/kaggle/working")
MODELS.mkdir(parents=True, exist_ok=True)

REF = os.environ.get("CRISPASR_REF", "feat/moss-tts-parity-diff")
HF_MODEL = os.environ.get("MOSS_TTS_MODEL", "OpenMOSS-Team/MOSS-TTS-v1.5")
HF_GGUF = os.environ.get("MOSS_TTS_GGUF_REPO", "cstr/moss-tts-v1.5-GGUF")
TEXT = os.environ.get("MOSS_TTS_TEXT", "The quick brown fox jumps over the lazy dog.")
MAXNEW = int(os.environ.get("MOSS_TTS_MAXNEW", "160"))

_T0 = time.time()
PROGRESS = WORK / "progress.txt"


def log(m):
    line = f"[{round(time.time() - _T0, 1)}s] {m}"
    print(line, flush=True)
    with open(PROGRESS, "a") as f:
        f.write(line + "\n")


# ctypes mirror of struct moss_tts_synth_params (field order MUST match moss_tts.h)
class SynthParams(ctypes.Structure):
    _fields_ = [
        ("max_new_tokens", ctypes.c_int),
        ("text_temperature", ctypes.c_float),
        ("text_top_p", ctypes.c_float),
        ("text_top_k", ctypes.c_int),
        ("audio_temperature", ctypes.c_float),
        ("audio_top_p", ctypes.c_float),
        ("audio_top_k", ctypes.c_int),
        ("audio_repetition_penalty", ctypes.c_float),
        ("min_audio_frames", ctypes.c_int),
        ("max_audio_frames", ctypes.c_int),
        ("seed", ctypes.c_uint64),
        ("language", ctypes.c_char_p),
        ("instruction", ctypes.c_char_p),
    ]


def cpp_greedy_codes(lib_path, backbone, codec, text):
    """Call the moss_tts C ABI greedily via ctypes → (n_vq, T) int32 numpy."""
    import numpy as np
    lib = ctypes.CDLL(lib_path)
    lib.moss_tts_context_default_params.restype = None
    # context params struct: {int n_threads; int verbosity; bool use_gpu; bool flash_attn;}
    class CtxParams(ctypes.Structure):
        _fields_ = [("n_threads", ctypes.c_int), ("verbosity", ctypes.c_int),
                    ("use_gpu", ctypes.c_bool), ("flash_attn", ctypes.c_bool)]
    lib.moss_tts_context_default_params.restype = CtxParams
    lib.moss_tts_init_from_file.restype = ctypes.c_void_p
    lib.moss_tts_init_from_file.argtypes = [ctypes.c_char_p, CtxParams]
    lib.moss_tts_set_codec_path.restype = ctypes.c_bool
    lib.moss_tts_set_codec_path.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.moss_tts_generate_codes.restype = ctypes.POINTER(ctypes.c_int32)
    lib.moss_tts_generate_codes.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                            ctypes.POINTER(SynthParams),
                                            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    lib.moss_tts_free.argtypes = [ctypes.c_void_p]

    cp = lib.moss_tts_context_default_params()
    cp.use_gpu = True
    ctx = lib.moss_tts_init_from_file(backbone.encode(), cp)
    if not ctx:
        raise RuntimeError("moss_tts_init_from_file returned null")
    lib.moss_tts_set_codec_path(ctx, codec.encode())  # not needed for codes, but harmless
    sp = SynthParams()
    sp.max_new_tokens = MAXNEW
    sp.text_temperature = 0.0   # greedy
    sp.audio_temperature = 0.0  # greedy
    sp.text_top_p = 1.0
    sp.audio_top_p = 1.0
    sp.text_top_k = 0
    sp.audio_top_k = 0
    sp.audio_repetition_penalty = 1.0
    sp.seed = 0
    nvq = ctypes.c_int(0)
    t_audio = ctypes.c_int(0)
    ptr = lib.moss_tts_generate_codes(ctx, text.encode(), ctypes.byref(sp),
                                      ctypes.byref(nvq), ctypes.byref(t_audio))
    if not ptr or nvq.value <= 0 or t_audio.value <= 0:
        lib.moss_tts_free(ctx)
        raise RuntimeError(f"generate_codes empty (nvq={nvq.value} T={t_audio.value})")
    n = nvq.value * t_audio.value
    arr = np.ctypeslib.as_array(ptr, shape=(n,)).reshape(nvq.value, t_audio.value).copy()
    lib.moss_tts_free(ctx)
    return arr


def main():
    import numpy as np
    summary = {"text": TEXT, "max_new": MAXNEW}
    log(f"clone {REF}")
    if not REPO.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", "--branch", REF, "--recursive",
                               "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.init_progress()

    subprocess.run(["nvidia-smi", "-L"], check=False)

    # ── build crispasr-lib (ctypes target; faster than the full CLI) ──
    kh.install_build_toolchain()
    arch = kh.detect_cuda_arch()
    env = os.environ.copy()
    env["CCACHE_DIR"] = "/kaggle/temp/.ccache"  # keep /kaggle/working small (#22)
    subprocess.run(["cmake", "-G", "Ninja", "-B", str(BUILD), "-S", str(REPO), "-DCMAKE_BUILD_TYPE=Release",
                    "-DBUILD_SHARED_LIBS=ON"] + list(kh.cache_and_link_flags()) + list(kh.cuda_build_flags(arch)),
                   env=env, check=True, timeout=300)
    with kh.build_heartbeat("moss-tts parity build"):
        kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-lib "
                            f"-j{kh.safe_build_jobs(gpu=True)}")
    import glob
    libs = glob.glob(str(BUILD / "src" / "libcrispasr.so*"))
    if not libs:
        raise SystemExit("libcrispasr.so not built")
    lib_path = libs[0]
    os.environ["LD_LIBRARY_PATH"] = f"{BUILD/'src'}:{os.environ.get('LD_LIBRARY_PATH','')}"
    log(f"built {lib_path}")

    # ── download shipped GGUFs (no re-convert) ──
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import hf_hub_download
    tok = kh.resolve_hf_token()
    q4k = hf_hub_download(HF_GGUF, "moss-tts-v1.5-q4_k.gguf", local_dir=str(MODELS), token=tok)
    codec = hf_hub_download(HF_GGUF, "moss-tts-v1.5-codec.gguf", local_dir=str(MODELS), token=tok)
    log("downloaded shipped GGUFs")

    # ── (1) HF reference greedy codes, STANDALONE, then free the GPU ──
    ref_codes = None
    try:
        log("HF reference greedy generate")
        renv = os.environ.copy()
        # 8B bf16 (~16 GB) won't fit the 16 GB P100 — run the reference on CPU
        # (fits ~29 GB host RAM); the C++ side runs Q4_K on the GPU.
        renv.update(MOSS_TTS_MODEL=HF_MODEL, MOSS_TTS_TEXT=TEXT, MOSS_TTS_SEED="0",
                    MOSS_TTS_MAXNEW=str(MAXNEW), MOSS_TTS_REF_DEVICE="cpu",
                    CUDA_VISIBLE_DEVICES="")
        r = subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0, r'{REPO/'tools'/'reference_backends'}');"
             f"import moss_tts as m; m.run(out_dir=r'{MODELS/'ref'}')"],
            env=renv, capture_output=True, text=True, timeout=2400)
        (WORK / "ref.log").write_text(r.stdout + "\n--STDERR--\n" + r.stderr)
        rp = MODELS / "ref" / "codes.npy"
        if rp.exists():
            ref_codes = np.load(rp)
            log(f"ref codes {ref_codes.shape}")
        else:
            log("ref codes NOT produced — see ref.log (HF API extraction may need a fix)")
            summary["ref_error"] = r.stderr[-800:]
    except Exception as e:  # noqa: BLE001
        log(f"ref failed: {e}")
        summary["ref_error"] = str(e)

    # ── (2) C++ greedy codes via ctypes (Q4_K) ──
    try:
        log("C++ greedy generate_codes (Q4_K)")
        cpp = cpp_greedy_codes(lib_path, q4k, codec, TEXT)
        np.save(MODELS / "codes_cpp.npy", cpp)
        log(f"cpp codes {cpp.shape}")
        summary["cpp_shape"] = list(cpp.shape)
    except Exception as e:  # noqa: BLE001
        log(f"cpp failed: {e}\n{traceback.format_exc()}")
        summary["cpp_error"] = str(e)
        cpp = None

    # ── (3) compare the greedy prefix ──
    if ref_codes is not None and cpp is not None:
        T = min(ref_codes.shape[1], cpp.shape[1])
        nvq = min(ref_codes.shape[0], cpp.shape[0])
        a, b = ref_codes[:nvq, :T], cpp[:nvq, :T]
        eq = (a == b)
        # first frame (column) with any mismatch
        first_div = None
        for t in range(T):
            if not eq[:, t].all():
                first_div = t
                break
        exact_prefix = first_div if first_div is not None else T
        summary.update({"compare_T": T, "n_vq": nvq,
                        "exact_prefix_frames": exact_prefix,
                        "first_divergence_frame": first_div,
                        "overall_match_frac": float(eq.mean())})
        log(f"PARITY: exact greedy prefix = {exact_prefix}/{T} frames; "
            f"first divergence at frame {first_div}; overall match {eq.mean():.3f}")
        # Gate: a non-trivial exact prefix proves structural correctness.
        summary["parity_gate"] = "PASS" if exact_prefix >= min(8, T) else "FAIL"
    else:
        summary["parity_gate"] = "INCOMPLETE"

    (WORK / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60 + "\n" + json.dumps(summary, indent=2) + "\n" + "=" * 60)
    log(f"parity gate: {summary.get('parity_gate')}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        log(f"FATAL: {e}\n{traceback.format_exc()}")
        sys.exit(1)
