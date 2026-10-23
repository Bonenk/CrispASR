#!/usr/bin/env python3
"""Run the CI-built #555 runtime on actual CUDA; no build or Torch on Kaggle."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

VERSION = "issue555-cuda-v1"
SOURCE = "682b54af2a883371541b861b434600bbf9b3a8e5"
# Fill only after auditing the completed CI archive and its anonymous readback.
BUNDLE_REVISION = '65b4a0b8a390e3fcab5211d96adcccde893c7be3'
BUNDLE_SHA256 = '3487f38ff77e10fedb946b317761d1a911f025eedf2dccd9fba1cc57eb224bfd'
BUNDLE_PATH = 'issue555-voice-pack-20261023/cuda-682b54af2/qwen3-voice-pack-cuda-speech-validation.tar.gz'
assert len(BUNDLE_REVISION) == 40 and len(BUNDLE_SHA256) == 64, "CI archive not yet qualified; do not launch"
OUT = Path("/kaggle/working")
SCRATCH = Path("/kaggle/temp/issue555")
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ.update(TMPDIR=str(SCRATCH), HF_HOME=str(SCRATCH / "hf"),
                  HF_XET_CACHE=str(SCRATCH / "xet"), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4")
hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version",
                                   "--format=csv,noheader"], text=True).strip()
assert hardware, "Actual CUDA device required"
print("version", VERSION, "source", SOURCE, "hardware", hardware, flush=True)
repo = SCRATCH / "CrispASR"
for command in (["git", "init", repo], ["git", "-C", repo, "remote", "add", "origin", "https://github.com/CrispStrobe/CrispASR.git"],
                ["git", "-C", repo, "fetch", "--depth=1", "origin", SOURCE], ["git", "-C", repo, "checkout", "FETCH_HEAD"]):
    subprocess.run(list(map(str, command)), check=True)
assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() == SOURCE
sys.path.insert(0, str(repo / "tools/kaggle"))
import kaggle_harness as kh
kh.init_progress()
kh.provenance(VERSION, repo)
os.environ["HF_TOKEN"] = kh.resolve_hf_token(require=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy<3", "huggingface_hub", "gguf"], check=True)
from huggingface_hub import hf_hub_download
archive = Path(hf_hub_download("cstr/crispasr-regression-fixtures", BUNDLE_PATH,
                               repo_type="dataset", revision=BUNDLE_REVISION))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

assert sha(archive) == BUNDLE_SHA256
with tarfile.open(archive) as stream:
    stream.extractall(SCRATCH, filter="data")
bundle = SCRATCH / "bundle"
provenance = json.loads((bundle / "provenance.json").read_text())
assert provenance["source"] == SOURCE and provenance["build_complete"] and not provenance["gpu_executed"]
for name, row in provenance["files"].items():
    assert sha(bundle / name) == row["sha256"] and (bundle / name).stat().st_size == row["bytes"], name
assert not list(bundle.glob("libcuda.so*")), "Driver or stub must never be bundled"
# ELF dependency search must be set before starting each native process.
env = dict(os.environ, LD_LIBRARY_PATH=str(bundle) + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
(OUT / "provenance.json").write_text(json.dumps(dict(version=VERSION, source=SOURCE, hardware=hardware,
    entry_sha256=sha(__file__), archive_revision=BUNDLE_REVISION, archive_sha256=BUNDLE_SHA256,
    native_files=provenance["files"]), indent=2) + "\n")
kh.step("bundle.ready", source=SOURCE)
command = [sys.executable, repo / "tools/ci-heavy/qwen3_voice_pack_validate.py", "--device", "cuda",
           "--bundle", bundle, "--scratch", SCRATCH / "models-and-fixtures", "--out", OUT / "cuda"]
with kh.build_heartbeat("cuda.validation"):
    result = subprocess.run(list(map(str, command)), env=env)
summary = json.loads((OUT / "cuda/validation.json").read_text())
kh.step("cuda.result", returncode=result.returncode, complete=summary["complete"], speech_pass=summary["speech_pass"])
assert result.returncode == 0 and summary["complete"] and summary["speech_pass"], summary
