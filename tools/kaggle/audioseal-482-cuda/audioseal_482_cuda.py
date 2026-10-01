#!/usr/bin/env python3
"""Run the CI-built #482 CUDA proof on real GPU hardware; never build on Kaggle."""
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import requests

SCRIPT_VERSION = "audioseal-482-cuda-v1"
BUILD_RUN = 36832617360
BUILD_SHA = "5f88f631299aa74acbf433034b832cda94865f96"
WORK = Path("/kaggle/working")
SCRATCH = Path("/kaggle/temp/audioseal-482")
SCRATCH.mkdir(parents=True, exist_ok=True)
REPO = SCRATCH / "CrispASR"
subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"], check=True)
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/CrispStrobe/CrispASR.git", str(REPO)], check=True)
sys.path.insert(0, str(REPO / "tools/kaggle"))
import kaggle_harness as kh

kh.init_progress()
kh.resolve_hf_token()
kh.provenance(SCRIPT_VERSION, REPO)
gh_token = kh.kaggle_token_from_dataset("gh_token.txt")
if not gh_token:
    raise RuntimeError("GitHub artifact token unavailable")
headers = {"Authorization": f"Bearer {gh_token}", "Accept": "application/vnd.github+json"}
api_root = "https://api.github.com/repos/CrispStrobe/CrispASR/actions"
run = requests.get(f"{api_root}/runs/{BUILD_RUN}", headers=headers, timeout=60)
run.raise_for_status()
run = run.json()
if run["head_sha"] != BUILD_SHA or run["conclusion"] != "success":
    raise RuntimeError("proof build is not the expected successful commit")
response = requests.get(f"{api_root}/runs/{BUILD_RUN}/artifacts", headers=headers, timeout=60)
response.raise_for_status()
artifacts = response.json()["artifacts"]
artifact = next(a for a in artifacts if a["name"] == f"audioseal-cuda-proof-{BUILD_SHA}" and not a["expired"])
# Keep the API credential on api.github.com; download the signed storage URL without it.
response = requests.get(artifact["archive_download_url"], headers=headers, timeout=60, allow_redirects=False)
response.raise_for_status()
archive_zip = SCRATCH / "artifact.zip"
with requests.get(response.headers["Location"], timeout=300, stream=True) as download:
    if download.status_code != 200:
        raise RuntimeError(f"artifact storage download failed ({download.status_code})")
    with archive_zip.open("wb") as dst:
        for chunk in download.iter_content(1024 * 1024):
            dst.write(chunk)
with zipfile.ZipFile(archive_zip) as archive:
    archive.extractall(SCRATCH / "artifact")
artifact_dir = SCRATCH / "artifact"
tar_path = artifact_dir / "audioseal-cuda-proof.tar.gz"
if hashlib.sha256(tar_path.read_bytes()).hexdigest() != (artifact_dir / "sha256.txt").read_text().strip():
    raise RuntimeError("proof archive checksum mismatch")
with tarfile.open(tar_path) as archive:
    for member in archive.getmembers():
        if not (member.isfile() or member.isdir()) or member.name.startswith("/") or ".." in Path(member.name).parts:
            raise RuntimeError("unexpected proof archive member")
    archive.extractall(SCRATCH)
bundle = SCRATCH / "bundle"
provenance = json.loads((bundle / "provenance.json").read_text())
if provenance["sha"] != BUILD_SHA:
    raise RuntimeError("proof bundle provenance mismatch")
kh.step("bundle.ready", build_sha=BUILD_SHA, build_run=BUILD_RUN)
model = SCRATCH / "audioseal.gguf"
urllib.request.urlretrieve("https://huggingface.co/cstr/audioseal-GGUF/resolve/main/audioseal.gguf", model)
subprocess.run(["uptime"], check=True)
subprocess.run(["free", "-h"], check=True)
env = dict(os.environ, LD_LIBRARY_PATH=str(bundle))
with kh.build_heartbeat("cuda.inference"):
    result = subprocess.run([str(bundle / "audioseal-cuda-proof"), str(model)],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=1800)
(WORK / "cuda-proof.log").write_text(result.stdout)
print(result.stdout, flush=True)
passed = result.returncode == 0 and "AUDIOSEAL_CUDA_PASS" in result.stdout
# The model loader must select CUDA, in addition to the explicit CUDA API check.
passed = passed and "using preferred GPU backend: CUDA" in result.stdout
summary = {"script_version": SCRIPT_VERSION, "build_sha": BUILD_SHA, "build_run": BUILD_RUN,
           "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version",
                                             "--format=csv,noheader"], text=True).strip(),
           "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
           "rc": result.returncode, "passed": passed}
(WORK / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
kh.step("cuda.result", **summary)
if not passed:
    raise SystemExit("AudioSeal CUDA proof failed")
