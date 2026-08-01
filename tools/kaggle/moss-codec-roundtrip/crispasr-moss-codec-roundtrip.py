#!/usr/bin/env python3
"""
Validate moss_tts_local_codec::encode() by round-trip.

decode() is already trusted — it ships and produces correct audio. So if encode()
is the right analysis inverse, re-encoding decode's output must recover
approximately the codes we started from. A wrong nearest-entry rule (L2 where the
model uses cosine, say) leaves quantizer-0 agreement at chance, 1/1024 ~= 0.1%,
which separates cleanly from a correct inverse.

This exists because encode() was written from INFERENCE, not measurement: the
cosine rule and the reversed attention contexts were reasoned from the v1 codec
runtime. #249 is the standing reminder that plausible-and-wrong survives a long
time unless something measures it.

Follows the harness regime (clone in-kernel, import from the clone, heartbeat).
"""
import os, subprocess, sys, json
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp"); TMP.mkdir(parents=True, exist_ok=True)
REPO = TMP / "CrispASR"
BUILD = TMP / "build"
CODEC_REPO = "cstr/moss-tts-local-v1.5-GGUF"
CODEC_FILE = os.environ.get("MOSS_CODEC_FILE", "moss-tts-local-v1.5-codec-enc.gguf")

if not REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--recurse-submodules",
                    "--shallow-submodules", "https://github.com/CrispStrobe/CrispASR.git", str(REPO)],
                   check=True, timeout=2400)

_h = REPO / "tools" / "kaggle"
sys.path.insert(0, str(_h if (_h / "kaggle_harness.py").exists() else Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
res = {}


def sh(cmd, cwd=None, timeout=3600):
    print(f"$ {cmd}", flush=True)
    p = subprocess.run(cmd, shell=True, cwd=cwd, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


kh.step("toolchain")
kh.install_build_toolchain()
token = kh.resolve_hf_token()
if token:
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token

# CPU-only: the round-trip is tiny (a couple of dozen frames) and this needs no
# GPU, so it stays off the GPU quota entirely.
kh.step("cmake")
flags = ["-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_BUILD_TESTS=OFF",
         "-DCRISPASR_BUILD_EXAMPLES=OFF", "-DCRISPASR_BUILD_SERVER=OFF"] + kh.cache_and_link_flags()
with kh.build_heartbeat("cmake.configure"):
    rc, out = sh(f"cmake -S {REPO} -B {BUILD} -G Ninja " + " ".join(flags))
if rc != 0:
    print(out[-6000:], flush=True); raise SystemExit("configure failed")
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-lib -j{kh.safe_build_jobs(gpu=False)}")

kh.step("compile probe")
probe = REPO / "tools" / "moss-codec" / "moss_codec_roundtrip.cpp"
rc, out = sh(f"c++ -std=gnu++17 -O2 -I {REPO}/src -I {REPO}/ggml/include {probe} "
             f"-o {TMP}/roundtrip -L {BUILD}/src -lcrispasr "
             f"-L {BUILD}/ggml/src -lggml-base -lggml-cpu -lggml "
             f"-Wl,-rpath,{BUILD}/src -Wl,-rpath,{BUILD}/ggml/src")
if rc != 0:
    print(out[-4000:], flush=True); raise SystemExit("probe compile failed")

kh.step("download codec")
from huggingface_hub import hf_hub_download  # noqa: E402
codec = hf_hub_download(repo_id=CODEC_REPO, filename=CODEC_FILE, token=token,
                        local_dir=str(TMP / "codec"))
print(f"codec: {codec} ({Path(codec).stat().st_size/1e9:.2f} GB)", flush=True)

kh.step("roundtrip")
with kh.build_heartbeat("roundtrip", 30.0):
    rc, out = sh(f"{TMP}/roundtrip {codec} 24", timeout=3600)
print(out, flush=True)
res["rc"] = rc
res["output"] = out[-4000:]
(WORK / "moss_codec_roundtrip.json").write_text(json.dumps(res, indent=2))
kh.step("done", rc=rc)
if rc != 0:
    raise SystemExit(rc)
